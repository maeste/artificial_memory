"""Indicizzazione atomica, scansione e watcher (piano Ingestion G3/G4).

Sequenza normativa di index_file (G3):
  canonicalizza path -> verifica formato -> SHA-256 -> confronto storage.file_hash
  (invariato: False, zero inferenza) -> staging univoco -> estrazione -> embed
  per chunk (retry singolo su TIMEOUT con stesso payload/request_id) ->
  SHA-256 finale (diverso: SourceChangedError, staging abbandonato, vecchio DB
  intatto) -> storage.replace_file -> True. Nel finally si rimuove soltanto lo
  staging proprio. Un file in errore non impedisce i successivi.

scan_directory e il watcher condividono index_file; la riconciliazione del
riavvio rimuove dal DB le fonti sparite dalla radice e segnala senza toccarle
le fonti fuori dalla radice configurata.

Il WatcherService (G4) serializza l'indicizzazione: deduplica coda per path,
debounce con stabilità size+mtime fra due rilevazioni distanti (il doppio hash
di G3 resta la protezione reale contro le copie concorrenti), riaccodamento se
un evento arriva durante il lavoro. Un move rimuove il vecchio path e indicizza
il nuovo: gli archi della vecchia identità non migrano automaticamente.

CLI (contratto piano): uv run python -m src.ingestion.watcher --config PATH [--once]
"""

import argparse
import hashlib
import json
import os
import shutil
import signal
import threading
import time
import uuid
from collections import deque
from pathlib import Path

from src.config import DEFAULT_EMBED_TIMEOUT_MS
from src.contracts import EmbedRequest, MemoryRecord, RuntimePaths, SpaceSpec, validate_vector
from src.ingestion.extractors import (
    SUPPORTED_SUFFIXES,
    IngestionError,
    InvalidMediaError,
    UnsupportedFormatError,
    extract_file_chunks,
)

EXPORT_DIR_NAMES = frozenset({"exports"})  # directory runtime/export escluse dalla scansione


class SourceChangedError(IngestionError):
    """Il sorgente è cambiato durante l'estrazione: la versione viene riaccodata."""

    error_code = "SOURCE_CHANGED"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_timeout_error(exc: Exception) -> bool:
    """Riconosce un TIMEOUT dell'inferenza secondo il contratto d'errore.

    Il client reale (Agent-2) segnala EmbeddingError con attributi .code e
    .message ('CODE: message'); i fake dei test usano .error_code. Il nome
    classe resta un canale di compatibilità."""
    if getattr(exc, "code", None) == "TIMEOUT":
        return True
    if getattr(exc, "error_code", None) == "TIMEOUT":
        return True
    return "TIMEOUT" in type(exc).__name__.upper()


def _embed_with_retry(client, request: EmbedRequest) -> list[float]:
    """Un solo tentativo aggiuntivo su TIMEOUT, stesso payload e stesso request_id."""
    try:
        return client.embed(request)
    except Exception as exc:  # noqa: BLE001 - la classificazione decide il retry
        if not _is_timeout_error(exc):
            raise
    return client.embed(request)


def index_file(file_path: str, storage, client, tokenizer: object, temp_root: str) -> bool:
    """Indicizza un file con sostituzione atomica. False se hash invariato.

    Un errore (inferenza, media, sorgente cambiata) lascia intatta la versione
    precedente già indicizzata: replace_file è chiamato solo dopo il doppio
    hash coincidente e il embedding riuscito di tutti i chunk.
    """
    canonical = str(Path(file_path).resolve())
    if Path(canonical).suffix.lower() not in SUPPORTED_SUFFIXES:
        raise UnsupportedFormatError(f"Formato non supportato: {canonical}")
    if not Path(canonical).is_file():
        raise InvalidMediaError(f"File sorgente inesistente: {canonical}")
    source_sha = _sha256_file(Path(canonical))
    if storage.file_hash(canonical) == source_sha:
        return False
    staging = Path(temp_root) / f"index-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        pairs = extract_file_chunks(canonical, tokenizer, str(staging))
        records: list[MemoryRecord] = []
        for memory, request in pairs:
            vector = _embed_with_retry(client, request)
            validate_vector(vector)
            records.append(MemoryRecord(memory=memory, embedding=vector))
        if _sha256_file(Path(canonical)) != source_sha:
            # hash iniziale e finale devono coincidere per potere committare
            raise SourceChangedError(f"Sorgente cambiata durante l'indicizzazione: {canonical}")
        storage.replace_file(canonical, source_sha, records)
        return True
    finally:
        shutil.rmtree(staging, ignore_errors=True)  # soltanto lo staging proprio


def _iter_supported_files(root: Path):
    """File supportati sotto la radice, ricorsivi, senza directory nascoste,
    export e symlink: i symlink di directory non vengono seguiti (niente cicli),
    i file symlinkati portano fuori radice sono ignorati."""
    for current, directories, files in os.walk(root, followlinks=False):
        directories[:] = [name for name in directories if not name.startswith(".") and name not in EXPORT_DIR_NAMES]
        base = Path(current)
        for name in files:
            if name.startswith("."):
                continue
            path = base / name
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            if path.is_symlink() and not path.resolve().is_relative_to(root):
                continue  # symlink che porta fuori dalla radice
            yield path.resolve()


def scan_directory(watch_dir: str, storage, client, tokenizer: object, temp_root: str) -> dict[str, str]:
    """Scansione ricorsiva + riconciliazione: path -> stato.

    Stati: 'indexed', 'unchanged', 'deleted' oppure 'error:CODE'. Le fonti nel
    DB fuori dalla radice configurata sono segnalate con OUTSIDE_WATCH_ROOT e
    non vengono toccate.
    """
    root = Path(watch_dir).resolve()
    results: dict[str, str] = {}
    present: set[str] = set()
    for path in _iter_supported_files(root):
        canonical = str(path)
        present.add(canonical)
        try:
            indexed = index_file(canonical, storage, client, tokenizer, temp_root)
        except IngestionError as exc:
            results[canonical] = f"error:{exc.error_code}"
        except Exception:  # noqa: BLE001 - un file in errore non blocca gli altri
            results[canonical] = "error:INGESTION_ERROR"
        else:
            results[canonical] = "indexed" if indexed else "unchanged"
    for stored in storage.list_files():
        if stored in present:
            continue
        if not Path(stored).is_relative_to(root):
            results[stored] = "error:OUTSIDE_WATCH_ROOT"
            continue
        storage.delete_by_filepath(stored)
        results[stored] = "deleted"
    return results


# --------------------------------------------------------------------------- watcher service (G4)


def wait_stable(path: str, interval_sec: float = 1.0, attempts: int = 5) -> bool:
    """True se size e mtime risultano identici fra due rilevazioni distanti
    `interval_sec`. La stabilità è un filtro di debounce, non una garanzia di
    copia finita: la protezione reale è il doppio hash di index_file."""
    target = Path(path)
    previous = None
    for _ in range(max(1, attempts)):
        try:
            stat = target.stat()
        except OSError:
            return False
        signature = (stat.st_size, stat.st_mtime_ns)
        if previous is not None and signature == previous:
            return True
        previous = signature
        time.sleep(interval_sec)
    return False


class CharTokenizer:
    """Tokenizer deterministica a carattere per il backend mock: conteggio
    conservativo dei limiti (1 carattere = 1 token), round-trip esatto."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(ch) for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(int(i)) for i in ids)


def load_tokenizer(space: SpaceSpec):
    """Tokenizer della revisione contrattualizzata: quella del modello reale
    via transformers (senza pesi), a caratteri per il mock."""
    if space.backend == "mock":
        return CharTokenizer()
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(space.model_id, revision=space.revision)


def _real_storage():
    try:
        from src.storage import BrainStorage  # [INTERFACE_READY] di Agent-1-Storage
    except ImportError as exc:
        raise SystemExit(f"API reale BrainStorage non disponibile ([INTERFACE_READY] pendente): {exc}") from exc
    return BrainStorage


def _real_client():
    try:
        from src.inference.client import EmbedClient  # [INTERFACE_READY] di Agent-2-Inference
    except ImportError as exc:
        raise SystemExit(f"API reale EmbedClient non disponibile ([INTERFACE_READY] pendente): {exc}") from exc
    return EmbedClient


def _load_config(config_path: str) -> tuple[SpaceSpec, RuntimePaths, dict]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    space = SpaceSpec(**config["space"])
    paths = RuntimePaths(**config["paths"])
    return space, paths, config


def run_once(
    config_path: str,
    storage=None,
    client=None,
    tokenizer: object | None = None,
    temp_root: str | None = None,
) -> dict[str, str]:
    """Modalità --once: riconciliazione singola dal config.json del launcher.

    Storage/client/tokenizer possono essere iniettati (test con fake); senza
    iniezione usa le API reali di Storage e Inferenza.
    """
    space, paths, config = _load_config(config_path)
    if storage is None:
        storage = _real_storage()(paths.db_path, space)
    if client is None:
        client = _real_client()(
            paths.embed_endpoint, space.space_id, config.get("embed_timeout_ms", DEFAULT_EMBED_TIMEOUT_MS)
        )
    if tokenizer is None:
        tokenizer = load_tokenizer(space)
    root = temp_root or str(Path(paths.run_dir) / "staging")
    return scan_directory(paths.watch_dir, storage, client, tokenizer, root)


class WatcherService:
    """Coda seriale di indicizzazione dei file osservati.

    Deduplica per path, debounce con stabilità size+mtime, riaccodamento quando
    un evento arriva durante il lavoro. drain() lavora la coda nel thread
    chiamante (test e flush); start()/stop() aggiungono il thread worker e
    l'observer watchdog per l'uso da servizio.
    """

    def __init__(
        self,
        watch_dir: str,
        storage,
        client,
        tokenizer: object,
        temp_root: str,
        stability_interval_sec: float = 1.0,
        stability_attempts: int = 5,
    ):
        self.root = Path(watch_dir).resolve()
        self.storage = storage
        self.client = client
        self.tokenizer = tokenizer
        self.temp_root = temp_root
        self.stability_interval_sec = stability_interval_sec
        self.stability_attempts = stability_attempts
        self.events: list[str] = []  # log ispezionabile dei risultati
        self._queue: deque = deque()
        self._queued: set = set()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._observer = None

    # ---------------------------------------------------------------- sottomissione eventi

    def submit_index(self, path: str) -> None:
        self._submit(("index", str(path)))

    def submit_delete(self, path: str) -> None:
        self._submit(("delete", str(path)))

    def submit_move(self, old_path: str, new_path: str) -> None:
        self._submit(("move", str(old_path), str(new_path)))

    def _submit(self, item: tuple) -> None:
        with self._lock:
            if item in self._queued:
                return  # deduplica per path
            self._queued.add(item)
            self._queue.append(item)
        self._wake.set()

    # ---------------------------------------------------------------- lavorazione

    def drain(self) -> list[str]:
        """Lavora la coda fino a svuotarla nel thread chiamante."""
        results: list[str] = []
        while True:
            with self._lock:
                item = self._queue.popleft() if self._queue else None
                if item is not None:
                    self._queued.discard(item)
            if item is None:
                return results
            results.append(self._process(item))

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="ingestion-watcher", daemon=True)
        self._thread.start()
        self._start_observer()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=2.0)
            self._observer = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def request_stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def wait(self) -> None:
        while not self._stop.is_set():
            time.sleep(0.2)

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                item = self._queue.popleft() if self._queue else None
                if item is not None:
                    self._queued.discard(item)
            if item is None:
                self._wake.wait(timeout=0.5)
                self._wake.clear()
                continue
            self._process(item)

    def _process(self, item: tuple) -> str:
        try:
            if item[0] == "index":
                result = self._process_index(item[1])
            elif item[0] == "delete":
                result = self._process_delete(item[1])
            elif item[0] == "move":
                result = self._process_move(item[1], item[2])
            else:  # pragma: no cover - tipi di item chiusi
                result = "noop"
        except Exception as exc:  # noqa: BLE001 - un errore non ferma la coda
            code = getattr(exc, "error_code", "INGESTION_ERROR")
            result = f"error:{code}"
            self.events.append(f"{item[0]} {item[1]} -> {result}: {exc}")
        else:
            self.events.append(f"{item[0]} {item[1]} -> {result}")
        return result

    def _inside_root(self, canonical: str) -> bool:
        return Path(canonical).is_relative_to(self.root)

    def _process_index(self, path: str) -> str:
        canonical = str(Path(path).resolve())
        if not self._inside_root(canonical):
            self.events.append(f"index {canonical} -> fuori radice: ignorato")
            return "skipped"
        if not Path(canonical).is_file():
            self.events.append(f"index {canonical} -> assente: nessuna azione")
            return "skipped"
        if not wait_stable(canonical, self.stability_interval_sec, self.stability_attempts):
            # non stabile dopo i tentativi: si procede, il doppio hash di G3 decide
            self.events.append(f"index {canonical} -> instabile, doppio hash decide")
        indexed = index_file(canonical, self.storage, self.client, self.tokenizer, self.temp_root)
        return "indexed" if indexed else "unchanged"

    def _process_delete(self, path: str) -> str:
        canonical = str(Path(path).resolve())
        if not self._inside_root(canonical):
            return "skipped"
        removed = self.storage.delete_by_filepath(canonical)
        return "deleted" if removed else "noop"

    def _process_move(self, old_path: str, new_path: str) -> str:
        old_canonical = Path(old_path).resolve()
        new_canonical = Path(new_path).resolve()
        if new_canonical == old_canonical:
            return "noop"
        if not self._inside_root(str(new_canonical)):
            # spostato fuori radice: vale come cancellazione del vecchio path
            self.storage.delete_by_filepath(str(old_canonical))
            return "deleted"
        removed = 0
        if old_canonical.is_dir() or new_canonical.is_dir():
            # directory spostata: rimuovi ogni fonte sotto il vecchio prefisso
            for stored in self.storage.list_files():
                if Path(stored).is_relative_to(old_canonical):
                    removed += self.storage.delete_by_filepath(stored)
            for path in _iter_supported_files(new_canonical):
                index_file(str(path), self.storage, self.client, self.tokenizer, self.temp_root)
            return "moved-dir"
        self.storage.delete_by_filepath(str(old_canonical))
        wait_stable(str(new_canonical), self.stability_interval_sec, self.stability_attempts)
        indexed = index_file(str(new_canonical), self.storage, self.client, self.tokenizer, self.temp_root)
        return "moved" if indexed else "moved-unchanged"

    # ---------------------------------------------------------------- watchdog

    def _start_observer(self) -> None:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer

        service = self

        class _Handler(FileSystemEventHandler):
            def on_created(self, event):
                if not event.is_directory:
                    service.submit_index(event.src_path)

            def on_modified(self, event):
                if not event.is_directory:
                    service.submit_index(event.src_path)

            def on_deleted(self, event):
                if event.is_directory:
                    for stored in service.storage.list_files():
                        if Path(stored).is_relative_to(Path(event.src_path).resolve()):
                            service.submit_delete(stored)
                else:
                    service.submit_delete(event.src_path)

            def on_moved(self, event):
                service.submit_move(event.src_path, event.dest_path)

        self._observer = Observer(timeout=0.5)
        self._observer.schedule(_Handler(), str(self.root), recursive=True)
        self._observer.start()


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    """CLI del watcher: --config PATH (config.json del launcher), --once."""
    parser = argparse.ArgumentParser(
        prog="python -m src.ingestion.watcher",
        description="Indicizzazione della directory osservata (Ambient Second Brain)",
    )
    parser.add_argument("--config", required=True, help="config.json prodotto da run_poc.py")
    parser.add_argument("--once", action="store_true", help="riconciliazione singola ed exit")
    args = parser.parse_args(argv)
    if not Path(args.config).is_file():
        parser.error(f"config non trovato: {args.config}")

    space, paths, config = _load_config(args.config)
    temp_root = str(Path(paths.run_dir) / "staging")
    Path(temp_root).mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(space)

    if args.once:
        results = run_once(args.config, tokenizer=tokenizer, temp_root=temp_root)
        print(json.dumps(results, ensure_ascii=False, sort_keys=True))
        return 0

    storage = _real_storage()(paths.db_path, space)
    client = _real_client()(
        paths.embed_endpoint, space.space_id, config.get("embed_timeout_ms", DEFAULT_EMBED_TIMEOUT_MS)
    )
    results = scan_directory(paths.watch_dir, storage, client, tokenizer, temp_root)
    print(json.dumps(results, ensure_ascii=False, sort_keys=True), flush=True)

    service = WatcherService(paths.watch_dir, storage, client, tokenizer, temp_root)

    def _handle_signal(signum, _frame):
        service.request_stop()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    service.start()
    try:
        service.wait()
    finally:
        service.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
