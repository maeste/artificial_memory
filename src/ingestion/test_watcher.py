"""Test indicizzazione atomica, registro hash e watcher (G3/G4).

I fake sono aderenti alle firme dei contratti §4 (BrainStorage, EmbedClient)
finché le API reali di Agent-1/Agent-2 non sono dichiarate disponibili con
[INTERFACE_READY]: l'integrazione reale sostituisce i fake senza toccare la
logica di index_file/scan_directory.
"""

import hashlib
import json
import math
from pathlib import Path

import pytest

from src.contracts import EMBED_DIM, EmbedRequest, MemoryRecord
from src.ingestion.extractors import InvalidMediaError, UnsupportedFormatError
from src.ingestion.watcher import (
    SourceChangedError,
    index_file,
    scan_directory,
)


# --------------------------------------------------------------------------- fake contratti §4


class FakeTimeout(Exception):
    """Timeout dell'inferenza: stesso contratto d'errore del client reale."""

    error_code = "TIMEOUT"


class FakeEmbedClient:
    """EmbedClient finto: vettori deterministici per request_id, guasti iniettabili."""

    def __init__(self, failures: dict[str, Exception] | None = None):
        self.calls: list[EmbedRequest] = []
        self.failures = dict(failures or {})

    def embed(self, request: EmbedRequest) -> list[float]:
        self.calls.append(request)
        failure = self.failures.pop(request.request_id, None)
        if failure is not None:
            raise failure
        seed = hashlib.sha256(request.request_id.encode("utf-8")).digest()
        vector = [((seed[i % len(seed)] / 255.0) - 0.5) for i in range(EMBED_DIM)]
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]


class FakeBrainStorage:
    """BrainStorage finto: file_hash persistente, replace_file atomico, archi
    conservati sugli ID sopravvissuti (contratto §4)."""

    def __init__(self):
        self.files: dict[str, str] = {}
        self.memories: dict[str, MemoryRecord] = {}
        self.edges: dict[frozenset[str], float] = {}
        self.generation = 0
        self.replace_calls: list[tuple[str, str, list[MemoryRecord]]] = []
        self.commit_failure: Exception | None = None

    def file_hash(self, file_path: str) -> str | None:
        return self.files.get(str(Path(file_path).resolve()))

    def list_files(self) -> list[str]:
        return sorted(self.files)

    def index_generation(self) -> int:
        return self.generation

    def all_memories(self) -> list[MemoryRecord]:
        return list(self.memories.values())

    def get_memory(self, memory_id: str):
        record = self.memories.get(memory_id)
        return record.memory if record else None

    def replace_file(self, file_path: str, file_sha256: str, records: list[MemoryRecord]) -> None:
        if self.commit_failure is not None:
            raise self.commit_failure
        path = str(Path(file_path).resolve())
        self.replace_calls.append((path, file_sha256, list(records)))
        old_ids = {mid for mid, rec in self.memories.items() if rec.memory.file_path == path}
        new_ids = {rec.memory.id for rec in records}
        surviving = (set(self.memories) - old_ids) | new_ids
        self.edges = {edge: weight for edge, weight in self.edges.items() if set(edge) <= surviving}
        for mid in old_ids:
            self.memories.pop(mid, None)
        for record in records:
            self.memories[record.memory.id] = record
        self.files[path] = file_sha256
        self.generation += 1

    def delete_by_filepath(self, file_path: str) -> int:
        path = str(Path(file_path).resolve())
        removed = [mid for mid, rec in self.memories.items() if rec.memory.file_path == path]
        for mid in removed:
            self.memories.pop(mid, None)
        self.edges = {edge: weight for edge, weight in self.edges.items() if not (set(edge) & set(removed))}
        self.files.pop(path, None)
        self.generation += 1
        return len(removed)

    def reinforce_synapses(self, clicked_id: str, other_ids: list[str], delta: float = 0.25) -> None:
        for other in other_ids:
            if other == clicked_id or clicked_id not in self.memories or other not in self.memories:
                continue
            edge = frozenset((clicked_id, other))
            self.edges[edge] = min(1.0, self.edges.get(edge, 0.0) + delta)


class CharTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(ch) for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(int(i)) for i in ids)


@pytest.fixture
def storage():
    return FakeBrainStorage()


@pytest.fixture
def client():
    return FakeEmbedClient()


@pytest.fixture
def tokenizer():
    return CharTokenizer()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- index_file (G3)


def test_unchanged_file_returns_false_with_zero_embeddings(tmp_path, storage, client, tokenizer):
    source = tmp_path / "a.md"
    source.write_text("# Titolo\nCorpo di prova.\n")
    assert index_file(str(source), storage, client, tokenizer, str(tmp_path / "tmp")) is True
    calls_after_first = len(client.calls)
    assert index_file(str(source), storage, client, tokenizer, str(tmp_path / "tmp")) is False
    assert len(client.calls) == calls_after_first  # zero richieste inferenza


def test_atomic_rollback_when_client_fails_on_second_chunk(tmp_path, storage, client, tokenizer):
    source = tmp_path / "a.md"
    # quattro paragrafi da ~750 token: la sezione viene frazionata in piu parti
    paragraphs = "\n\n".join(f"Paragrafo numero {i} " + "contenuto " * 115 for i in range(4))
    source.write_text(f"# Sezione\n\n{paragraphs}\n")
    # prima versione indicizzata: riferimento per hash, record e generazione
    assert index_file(str(source), storage, client, tokenizer, str(tmp_path / "tmp")) is True
    old_hash = storage.file_hash(str(source.resolve()))
    old_ids = set(storage.memories)

    # nuova versione da indicizzare, con guasto sul secondo chunk: la versione
    # precedente (DB, hash, archi) deve restare intatta
    paragraphs = "\n\n".join(f"Paragrafo numero {i} " + "contenuto2 " * 115 for i in range(4))
    source.write_text(f"# Sezione\n\n{paragraphs}\n")
    pairs = _pairs_of(source, tokenizer, tmp_path)
    victim = pairs[1][1].request_id
    client.failures = {victim: RuntimeError("MODEL_UNAVAILABLE")}

    with pytest.raises(RuntimeError):
        index_file(str(source), storage, client, tokenizer, str(tmp_path / "tmp"))

    assert storage.file_hash(str(source.resolve())) == old_hash
    assert set(storage.memories) == old_ids
    assert storage.index_generation() == 1  # nessun commit aggiuntivo
    # lo staging è stato rimosso dal finally
    assert list((tmp_path / "tmp").iterdir()) == []


def test_timeout_retries_once_with_same_request_id(tmp_path, storage, client, tokenizer):
    source = tmp_path / "a.md"
    source.write_text("# Titolo\nCorpo di prova.\n")
    first_request = _pairs_of(source, tokenizer, tmp_path)[0][1]
    client.failures = {first_request.request_id: FakeTimeout("TIMEOUT")}
    assert index_file(str(source), storage, client, tokenizer, str(tmp_path / "tmp")) is True
    assert len(client.calls) == 2  # tentativo + retry
    assert client.calls[0] is client.calls[1]  # stesso payload/request_id
    assert client.calls[0].request_id == first_request.request_id


def test_timeout_twice_is_definitive_and_keeps_old_version(tmp_path, storage, client, tokenizer):
    source = tmp_path / "a.md"
    source.write_text("# Titolo\nCorpo di prova.\n")
    assert index_file(str(source), storage, client, tokenizer, str(tmp_path / "tmp")) is True
    old_hash = storage.file_hash(str(source.resolve()))

    # modifica il file: la reindicizzazione incontrera due TIMEOUT consecutive
    source.write_text("# Titolo\nCorpo modificato.\n")

    def persistent_timeout(request):
        client.calls.append(request)
        raise FakeTimeout("TIMEOUT")

    client.embed = persistent_timeout  # type: ignore[method-assign]
    with pytest.raises(FakeTimeout):
        index_file(str(source), storage, client, tokenizer, str(tmp_path / "tmp"))
    assert storage.file_hash(str(source.resolve())) == old_hash
    assert storage.index_generation() == 1


def test_source_changed_during_extraction_rolls_back(tmp_path, storage, client, tokenizer, monkeypatch):
    import src.ingestion.watcher as watcher

    source = tmp_path / "a.md"
    source.write_text("# Titolo\nVersione iniziale.\n")
    assert index_file(str(source), storage, client, tokenizer, str(tmp_path / "tmp")) is True
    old_hash = storage.file_hash(str(source.resolve()))
    old_ids = set(storage.memories)

    # il contenuto cambia prima del secondo giro (hash diverso dal DB) e la
    # hook di estrazione lo riscrive ancora: il doppio hash deve rilevarlo
    source.write_text("# Titolo\nVersione riscritta prima del secondo giro.\n")

    real_extract = watcher.extract_file_chunks

    def extracting_then_touching(file_path, tok, temp_dir):
        pairs = real_extract(file_path, tok, temp_dir)
        source.write_text("# Titolo\nVersione cambiata durante il lavoro.\n")
        return pairs

    monkeypatch.setattr(watcher, "extract_file_chunks", extracting_then_touching)
    with pytest.raises(SourceChangedError):
        index_file(str(source), storage, client, tokenizer, str(tmp_path / "tmp"))
    assert storage.file_hash(str(source.resolve())) == old_hash
    assert set(storage.memories) == old_ids


def test_empty_file_version_indexed_successfully(tmp_path, storage, client, tokenizer):
    source = tmp_path / "a.md"
    source.write_text("")
    assert index_file(str(source), storage, client, tokenizer, str(tmp_path / "tmp")) is True
    assert storage.file_hash(str(source.resolve())) == sha256_file(source)
    assert storage.memories == {}
    assert len(client.calls) == 0


def test_section_update_preserves_invariant_section_id_and_edge(tmp_path, storage, client, tokenizer):
    source = tmp_path / "a.md"

    def write(sections: dict[str, str]) -> None:
        source.write_text("".join(f"# {t}\n\n{b}\n" for t, b in sections.items()))

    write({"Cache": "Contenuto cache stabile.", "Backup": "Contenuto backup v1."})
    assert index_file(str(source), storage, client, tokenizer, str(tmp_path / "tmp")) is True
    cache_id = next(mid for mid, rec in storage.memories.items() if rec.memory.chunk_key == "section:Cache:0:part:0")
    backup_id = next(mid for mid, rec in storage.memories.items() if rec.memory.chunk_key == "section:Backup:0:part:0")
    storage.reinforce_synapses(cache_id, [backup_id])
    assert frozenset((cache_id, backup_id)) in storage.edges

    # modifica solo la sezione Backup: Cache deve conservare ID e arco
    write({"Cache": "Contenuto cache stabile.", "Backup": "Contenuto backup v2."})
    assert index_file(str(source), storage, client, tokenizer, str(tmp_path / "tmp")) is True

    assert cache_id in storage.memories  # sezione invariata: stesso ID
    assert backup_id not in storage.memories  # vecchia sezione rimossa
    assert frozenset((cache_id, backup_id)) not in storage.edges  # arco del ID rimosso: cade
    new_backup = next(rec for rec in storage.memories.values() if rec.memory.chunk_key == "section:Backup:0:part:0")
    assert "backup v2" in new_backup.memory.content_text
    assert storage.reinforce_synapses(cache_id, [new_backup.memory.id]) is None
    assert frozenset((cache_id, new_backup.memory.id)) in storage.edges


def test_index_unsupported_format_raises(tmp_path, storage, client, tokenizer):
    source = tmp_path / "a.txt"
    source.write_text("testo")
    with pytest.raises(UnsupportedFormatError):
        index_file(str(source), storage, client, tokenizer, str(tmp_path / "tmp"))


def test_index_missing_file_raises_invalid_media(tmp_path, storage, client, tokenizer):
    with pytest.raises(InvalidMediaError):
        index_file(str(tmp_path / "assente.md"), storage, client, tokenizer, str(tmp_path / "tmp"))


# --------------------------------------------------------------------------- scan_directory (G3)


def test_scan_directory_reports_indexed_unchanged_deleted_and_error(tmp_path, storage, client, tokenizer):
    watch = tmp_path / "watch"
    sub = watch / "sub"
    sub.mkdir(parents=True)
    (watch / "a.md").write_text("# A\nContenuto a.\n")
    (sub / "b.md").write_text("# B\nContenuto b.\n")
    (watch / "bad.png").write_bytes(b"corrotto")
    (watch / "ignora.txt").write_text("non supportato")

    results = scan_directory(str(watch), storage, client, tokenizer, str(tmp_path / "tmp"))
    assert results[str((watch / "a.md").resolve())] == "indexed"
    assert results[str((sub / "b.md").resolve())] == "indexed"
    assert results[str((watch / "bad.png").resolve())] == "error:INVALID_MEDIA"
    assert not any("ignora.txt" in key for key in results)
    assert storage.file_hash(str((watch / "a.md").resolve())) == sha256_file(watch / "a.md")

    # secondo scan senza modifiche: i .md sono unchanged, nessun embedding nuovo
    calls_before = len(client.calls)
    results = scan_directory(str(watch), storage, client, tokenizer, str(tmp_path / "tmp"))
    assert results[str((watch / "a.md").resolve())] == "unchanged"
    assert results[str((sub / "b.md").resolve())] == "unchanged"
    assert results[str((watch / "bad.png").resolve())] == "error:INVALID_MEDIA"
    assert len(client.calls) == calls_before  # il corrotto fallisce prima dell'embed

    # cancellazione + aggiunta: riconciliazione dal filesystem
    (sub / "b.md").unlink()
    (watch / "c.md").write_text("# C\nContenuto c.\n")
    results = scan_directory(str(watch), storage, client, tokenizer, str(tmp_path / "tmp"))
    assert results[str((sub / "b.md").resolve())] == "deleted"
    assert results[str((watch / "c.md").resolve())] == "indexed"
    assert str((sub / "b.md").resolve()) not in storage.list_files()


def test_scan_reconciles_files_deleted_while_off(tmp_path, storage, client, tokenizer):
    watch = tmp_path / "watch"
    watch.mkdir()
    source = watch / "a.md"
    source.write_text("# A\nContenuto.\n")
    assert index_file(str(source), storage, client, tokenizer, str(tmp_path / "tmp")) is True
    # il file sparisce a riposo: il prossimo scan lo rimuove dal DB
    source.unlink()
    results = scan_directory(str(watch), storage, client, tokenizer, str(tmp_path / "tmp"))
    assert results[str(source.resolve())] == "deleted"
    assert storage.list_files() == []


def test_scan_does_not_touch_sources_outside_root(tmp_path, storage, client, tokenizer):
    outside = tmp_path / "outside.md"
    outside.write_text("# Fuori\nFonte esterna alla radice.\n")
    assert index_file(str(outside), storage, client, tokenizer, str(tmp_path / "tmp")) is True

    watch = tmp_path / "watch"
    watch.mkdir()
    (watch / "b.md").write_text("# B\nContenuto interno.\n")
    results = scan_directory(str(watch), storage, client, tokenizer, str(tmp_path / "tmp"))
    assert results[str(outside.resolve())] == "error:OUTSIDE_WATCH_ROOT"
    assert str(outside.resolve()) in storage.list_files()  # non toccata


def test_scan_skips_hidden_and_exports_directories(tmp_path, storage, client, tokenizer):
    watch = tmp_path / "watch"
    hidden = watch / ".nascondo"
    exports = watch / "exports"
    hidden.mkdir(parents=True)
    exports.mkdir()
    (hidden / "a.md").write_text("# A\nnascosto\n")
    (exports / "b.md").write_text("# B\nexport\n")
    (watch / ".tmp-file.md").write_text("# T\ntemporaneo\n")
    (watch / "vero.md").write_text("# V\ncontenuto vero\n")
    results = scan_directory(str(watch), storage, client, tokenizer, str(tmp_path / "tmp"))
    assert list(results) == [str((watch / "vero.md").resolve())]


def test_scan_ignores_symlinks_outside_root(tmp_path, storage, client, tokenizer):
    watch = tmp_path / "watch"
    watch.mkdir()
    outside_dir = tmp_path / "fuori"
    outside_dir.mkdir()
    (outside_dir / "x.md").write_text("# X\nfuori radice\n")
    (watch / "link.md").symlink_to(outside_dir / "x.md")
    results = scan_directory(str(watch), storage, client, tokenizer, str(tmp_path / "tmp"))
    assert results == {}  # symlink fuori radice: ignorato


# --------------------------------------------------------------------------- helper


def _pairs_of(source: Path, tokenizer, tmp_path):
    from src.ingestion.extractors import extract_file_chunks

    return extract_file_chunks(str(source), tokenizer, str(tmp_path / "probe-stage"))


# --------------------------------------------------------------------------- watcher service (G4)


def drain_results(service):
    return service.drain()


def test_service_lifecycle_scan_modify_delete_restart_move(tmp_path, storage, client, tokenizer):
    from src.ingestion.watcher import WatcherService

    watch = tmp_path / "watch"
    watch.mkdir()
    a = watch / "a.md"
    b = watch / "b.md"
    c = watch / "c.md"
    a.write_text("# A\nVersione 1.\n")
    b.write_text("# B\nContenuto b.\n")
    c.write_text("# C\nContenuto c.\n")

    results = scan_directory(str(watch), storage, client, tokenizer, str(tmp_path / "tmp"))
    assert set(results.values()) == {"indexed"}
    gen_after_scan = storage.index_generation()
    calls_after_scan = len(client.calls)

    # modifica: hash aggiornato, nuovi embedding
    a.write_text("# A\nVersione 2.\n")
    service = WatcherService(
        str(watch), storage, client, tokenizer, str(tmp_path / "tmp"), stability_interval_sec=0.01
    )
    service.submit_index(str(a))
    service.drain()
    assert storage.file_hash(str(a.resolve())) == sha256_file(a)
    assert len(client.calls) > calls_after_scan

    # cancellazione: nessun record residuo per b
    b.unlink()
    service.submit_delete(str(b))
    service.drain()
    assert str(b.resolve()) not in storage.list_files()
    b_ids = [mid for mid, rec in storage.memories.items() if rec.memory.file_path == str(b.resolve())]
    assert b_ids == []

    # riavvio senza modifiche: nessun embedding nuovo (riconciliazione)
    client.calls.clear()
    results = scan_directory(str(watch), storage, client, tokenizer, str(tmp_path / "tmp"))
    assert set(results.values()) == {"unchanged"}
    assert client.calls == []

    # move: il vecchio path esce dal DB, il nuovo entra con ID nuovi (il path entra nell'ID)
    d = watch / "d.md"
    c.rename(d)
    service.submit_move(str(c), str(d))
    service.drain()
    assert str(c.resolve()) not in storage.list_files()
    assert storage.file_hash(str(d.resolve())) == sha256_file(d)
    c_ids = [rec for rec in storage.memories.values() if rec.memory.file_path == str(c.resolve())]
    assert c_ids == []
    d_records = [rec for rec in storage.memories.values() if rec.memory.file_path == str(d.resolve())]
    assert d_records and all(rec.memory.id != "" for rec in d_records)
    assert storage.index_generation() > gen_after_scan


def test_service_deduplicates_queue_per_path(tmp_path, storage, client, tokenizer):
    from src.ingestion.watcher import WatcherService

    watch = tmp_path / "watch"
    watch.mkdir()
    a = watch / "a.md"
    a.write_text("# A\nContenuto unico.\n")
    service = WatcherService(
        str(watch), storage, client, tokenizer, str(tmp_path / "tmp"), stability_interval_sec=0.01
    )
    for _ in range(5):
        service.submit_index(str(a))
    service.drain()
    # un solo giro di indicizzazione: il numero di chiamate = numero di chunk
    assert len(client.calls) == 1


def test_event_during_processing_requeues_path_once(tmp_path, storage, client, tokenizer, monkeypatch):
    import src.ingestion.watcher as watcher
    from src.ingestion.watcher import WatcherService

    watch = tmp_path / "watch"
    watch.mkdir()
    a = watch / "a.md"
    a.write_text("# A\nContenuto.\n")
    service = WatcherService(
        str(watch), storage, client, tokenizer, str(tmp_path / "tmp"), stability_interval_sec=0.01
    )

    real_index = watcher.index_file
    resubmitted = []

    def indexing_and_resubmitting(file_path, stor, cli, tok, root):
        result = real_index(file_path, stor, cli, tok, root)
        if file_path not in resubmitted:
            resubmitted.append(file_path)
            service.submit_index(file_path)  # evento durante il lavoro
        return result

    monkeypatch.setattr(watcher, "index_file", indexing_and_resubmitting)
    service.submit_index(str(a))
    service.drain()  # deve terminare dopo il secondo giro, senza loop infinito
    assert len(resubmitted) == 1
    assert len(client.calls) == 1  # primo giro: 1 embed; secondo giro: unchanged, zero embed


def test_service_skips_missing_file_and_out_of_root(tmp_path, storage, client, tokenizer):
    from src.ingestion.watcher import WatcherService

    watch = tmp_path / "watch"
    watch.mkdir()
    outside = tmp_path / "fuori.md"
    outside.write_text("# Fuori\nesterno\n")
    service = WatcherService(
        str(watch), storage, client, tokenizer, str(tmp_path / "tmp"), stability_interval_sec=0.01
    )
    service.submit_index(str(watch / "assente.md"))  # sparisce prima del lavoro
    service.submit_index(str(outside))  # fuori radice: non indicizzata
    service.drain()
    assert storage.list_files() == []


def test_wait_stable_detects_quiet_and_missing_files(tmp_path):
    from src.ingestion.watcher import wait_stable

    target = tmp_path / "quiet.md"
    target.write_text("stabile")
    assert wait_stable(str(target), interval_sec=0.01, attempts=3) is True
    assert wait_stable(str(tmp_path / "assente.md"), interval_sec=0.01, attempts=3) is False


def test_directory_move_removes_old_paths_and_indexes_new(tmp_path, storage, client, tokenizer):
    from src.ingestion.watcher import WatcherService

    watch = tmp_path / "watch"
    old_dir = watch / "vecchia"
    old_dir.mkdir(parents=True)
    (old_dir / "x.md").write_text("# X\ncontenuto x\n")
    scan_directory(str(watch), storage, client, tokenizer, str(tmp_path / "tmp"))
    assert len(storage.list_files()) == 1

    new_dir = watch / "nuova"
    old_dir.rename(new_dir)
    service = WatcherService(
        str(watch), storage, client, tokenizer, str(tmp_path / "tmp"), stability_interval_sec=0.01
    )
    service.submit_move(str(old_dir), str(new_dir))
    service.drain()
    assert storage.list_files() == [str((new_dir / "x.md").resolve())]


def test_once_mode_uses_config_and_reports_results(tmp_path, storage, client, tokenizer):
    """La logica --once accetta il config.json del launcher e stampa i risultati."""
    from src.ingestion.watcher import run_once

    watch = tmp_path / "watch"
    watch.mkdir()
    (watch / "a.md").write_text("# A\ncontenuto\n")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "space": {
                    "backend": "mock",
                    "model_id": "semantic-hash-mock",
                    "revision": "1",
                    "dimension": EMBED_DIM,
                    "preprocessing": "eg2-poc-v1",
                },
                "paths": {
                    "run_dir": str(tmp_path / "run"),
                    "embed_endpoint": "ipc://x",
                    "events_endpoint": "ipc://y",
                    "control_endpoint": "ipc://z",
                    "db_path": str(tmp_path / "b.db"),
                    "watch_dir": str(watch),
                    "export_dir": str(tmp_path / "exports"),
                },
            }
        )
    )
    temp_root = str(tmp_path / "stage")
    results = run_once(str(config_path), storage=storage, client=client, tokenizer=tokenizer, temp_root=temp_root)
    assert list(results.values()) == ["indexed"]


class EmbeddingError(Exception):
    """Forma d'errore del client reale (Agent-2): .code + .message."""

    def __init__(self, code: str, message: str = ""):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def test_is_timeout_error_recognizes_real_client_shape():
    import src.ingestion.watcher as watcher

    assert watcher._is_timeout_error(EmbeddingError("TIMEOUT", "scaduto")) is True
    assert watcher._is_timeout_error(EmbeddingError("INVALID_MEDIA", "corrotto")) is False
    assert watcher._is_timeout_error(FakeTimeout("TIMEOUT")) is True  # forma fake (.error_code)
    assert watcher._is_timeout_error(TimeoutError("socket")) is True  # nome classe
    assert watcher._is_timeout_error(RuntimeError("boom")) is False


def test_timeout_retry_uses_real_client_error_shape(tmp_path, storage, client, tokenizer):
    source = tmp_path / "a.md"
    source.write_text("# Titolo\nCorpo di prova.\n")

    class RealShapeClient(FakeEmbedClient):
        def embed(self, request):
            if not self.calls:
                self.calls.append(request)
                raise EmbeddingError("TIMEOUT", "primo tentativo scaduto")
            return super().embed(request)

    real_shape = RealShapeClient()
    assert index_file(str(source), storage, real_shape, tokenizer, str(tmp_path / "tmp")) is True
    assert len(real_shape.calls) == 2  # tentativo + retry
    assert real_shape.calls[0].request_id == real_shape.calls[1].request_id  # stesso request_id
