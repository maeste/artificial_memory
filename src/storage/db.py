"""Archivio persistente: schema, identità dello spazio, CRUD atomico.

Piano: untracked_doc/02_AGENT_1_STORAGE.md (ST1-ST2); API pubblica nei contratti
§4 (untracked_doc/07_TECHNICAL_CONTRACTS.md). Un DB ospita un solo space_id:
la riapertura con uno spazio diverso fallisce con SPACE_MISMATCH senza
distruggere nulla. I vettori sono BLOB little-endian float32 di EMBED_DIM
elementi, finiti e normalizzati; la riga in `memories` resta la fonte di verità
(export, test, fallback), mentre `vec_memories` (vec0) è il mirror mantenuto
nella stessa transazione quando il backend sqlite-vec è attivo.
"""

import json
import math
import os
import sqlite3
import struct
import time
from collections.abc import Sequence
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from src.contracts import (
    EMBED_DIM,
    MODALITY_VALUES,
    MemoryRecord,
    MemoryView,
    ScoredMemory,
    SearchSettings,
    SpaceSpec,
    make_memory_id,
    validate_vector,
)
from src.storage import ranking

SCHEMA_VERSION = 1
_SHA256_HEX = frozenset("0123456789abcdef")
_MAX_SNIPPET_CHARS = 350
_VALID_BACKENDS = ("auto", "numpy", "sqlite_vec")

_SCHEMA_SQL = (
    "CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    """CREATE TABLE IF NOT EXISTS files(
         file_path TEXT PRIMARY KEY, sha256 TEXT NOT NULL, indexed_at REAL NOT NULL
       )""",
    """CREATE TABLE IF NOT EXISTS memories(
         id TEXT PRIMARY KEY,
         file_path TEXT NOT NULL REFERENCES files(file_path) ON DELETE CASCADE,
         modality TEXT NOT NULL,
         payload_json TEXT NOT NULL,
         embedding BLOB NOT NULL
       )""",
    "CREATE INDEX IF NOT EXISTS memories_filepath ON memories(file_path)",
    """CREATE TABLE IF NOT EXISTS synaptic_edges(
         source_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
         target_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
         weight REAL NOT NULL CHECK(weight > 0 AND weight <= 1),
         updated_at REAL NOT NULL,
         PRIMARY KEY(source_id,target_id),
         CHECK(source_id <> target_id)
       )""",
)

# F3: registry dei ricordi di sessione. Additiva e idempotente: i DB esistenti
# la ricevono alla prima riapertura senza cancellazioni e senza bump di schema.
_SESSION_TABLE_SQL = """CREATE TABLE IF NOT EXISTS session_memories(
     session_id TEXT NOT NULL,
     memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
     file_path TEXT NOT NULL,
     promoted INTEGER NOT NULL DEFAULT 0,
     created_at REAL NOT NULL
   )"""


def _canonical_path(path: str) -> str:
    """Canonica dei contratti 07 §3: Path.resolve(), la stessa di make_memory_id."""
    return str(Path(path).resolve())


def _load_sqlite_vec(conn: sqlite3.Connection) -> None:
    import sqlite_vec

    conn.enable_load_extension(True)
    try:
        sqlite_vec.load(conn)
    finally:
        conn.enable_load_extension(False)


def _is_sha256_hex(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _SHA256_HEX


def _encode_vector(vector: list[float]) -> bytes:
    return struct.pack(f"<{EMBED_DIM}f", *(float(component) for component in vector))


def _decode_vector(blob: bytes) -> list[float]:
    if len(blob) != 4 * EMBED_DIM:
        raise ValueError(f"CORRUPT_ROW: BLOB embedding di {len(blob)} byte, attesi {4 * EMBED_DIM}")
    return list(struct.unpack(f"<{EMBED_DIM}f", blob))


def _payload_json(memory: MemoryView) -> str:
    # allow_nan=False: un created_at non finito non deve entrare nel DB come null.
    return json.dumps(asdict(memory), ensure_ascii=False, allow_nan=False)


class BrainStorage:
    """Indice persistente con sostituzione atomica dei file e grafo sinaptico.

    Attributi diagnostici (non API contrattuale): `vector_backend_effective`
    è 'numpy' o 'sqlite_vec'; `backend_fallback` è True quando 'auto' voleva
    sqlite-vec ma è passato a NumPy; `backend_note` riporta il dettaglio.
    """

    def __init__(self, db_path: str, space: SpaceSpec, vector_backend: str = "auto"):
        if not isinstance(space, SpaceSpec):
            raise ValueError("INVALID_REQUEST: space non è uno SpaceSpec")
        if space.dimension != EMBED_DIM:
            raise ValueError(f"INVALID_REQUEST: dimensione spazio {space.dimension}, supportata {EMBED_DIM}")
        if vector_backend not in _VALID_BACKENDS:
            raise ValueError(f"INVALID_REQUEST: vector_backend non ammesso: {vector_backend!r}")

        self.db_path = str(db_path)
        self.space = space
        self.vector_backend_effective: Literal["numpy", "sqlite_vec"] = "numpy"
        self.backend_fallback = False
        self.backend_note = ""
        self.conn: sqlite3.Connection | None = None

        conn = sqlite3.connect(self.db_path, isolation_level=None, check_same_thread=False)
        self.conn = conn
        # Le PRAGMA non vivono in transazione: journal_mode è persistente nel file,
        # foreign_keys e busy_timeout valgono per questa connessione.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")

        if vector_backend == "numpy":
            self._set_backend("numpy")
        else:
            try:
                _load_sqlite_vec(conn)
                self._set_backend("sqlite_vec")
            except Exception as exc:  # modulo assente, .so non caricabile, versione incompatibile
                if vector_backend == "sqlite_vec":
                    raise ValueError(f"SQLITE_VEC_UNAVAILABLE: estensione richiesta ma non caricabile: {exc}") from exc
                self._set_backend("numpy", fallback_note=f"sqlite-vec non caricabile ({exc})")

        self._init_or_verify_schema()
        if self.vector_backend_effective == "sqlite_vec":
            self._ensure_vec_mirror()

    # ------------------------------------------------------------------ apertura

    def _set_backend(self, backend: Literal["numpy", "sqlite_vec"], fallback_note: str = "") -> None:
        self.vector_backend_effective = backend
        if fallback_note:
            self.backend_fallback = True
            self.backend_note = f"fallback a NumPy: {fallback_note}"

    def _init_or_verify_schema(self) -> None:
        conn = self.conn
        has_metadata = (
            conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='metadata'").fetchone() is not None
        )
        if not has_metadata:
            alien = conn.execute(
                "SELECT count(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchone()[0]
            if alien:
                raise ValueError("SCHEMA_INCOMPATIBLE: il file non è un archivio; serve un nuovo DB per il PoC")
            conn.execute("BEGIN IMMEDIATE")
            try:
                for statement in _SCHEMA_SQL:
                    conn.execute(statement)
                if self.vector_backend_effective == "sqlite_vec":
                    self._create_vec_table()
                conn.executemany(
                    "INSERT INTO metadata(key, value) VALUES (?, ?)",
                    (
                        ("schema_version", str(SCHEMA_VERSION)),
                        ("space", json.dumps(asdict(self.space), sort_keys=True, separators=(",", ":"))),
                        ("index_generation", "0"),
                    ),
                )
                conn.execute("COMMIT")
            except BaseException:
                self._rollback()
                raise
            conn.execute(_SESSION_TABLE_SQL)  # DB nuovo: registry creato subito (idempotente)
            return

        meta = dict(conn.execute("SELECT key, value FROM metadata"))
        if meta.get("schema_version") != str(SCHEMA_VERSION):
            raise ValueError(
                f"SCHEMA_VERSION_MISMATCH: DB con schema {meta.get('schema_version')!r}, "
                f"attesa {SCHEMA_VERSION}; serve un nuovo DB per il PoC"
            )
        stored_space = meta.get("space", "")
        try:
            stored = json.loads(stored_space)
        except json.JSONDecodeError as exc:
            raise ValueError(f"SPACE_MISMATCH: spazio registrato illeggibile: {exc}") from exc
        if stored != asdict(self.space):
            raise ValueError(
                f"SPACE_MISMATCH: il DB appartiene allo spazio {stored.get('space_id', stored_space[:16])}, "
                f"richiesto {self.space.space_id}"
            )
        missing = {"files", "memories", "synaptic_edges"} - {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if missing:
            raise ValueError(f"SCHEMA_INCOMPATIBLE: tabelle mancanti {sorted(missing)}; serve un nuovo DB")

        # Migrazione additiva F3: eseguita a ogni apertura, idempotente, senza
        # cancellazioni e senza toccare schema_version né index_generation.
        conn.execute(_SESSION_TABLE_SQL)

    def _create_vec_table(self) -> None:
        self.conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0("
            f"id TEXT PRIMARY KEY, embedding float[{EMBED_DIM}])"
        )

    def _ensure_vec_mirror(self) -> None:
        """All'apertura col backend vec: crea/allinea il mirror ai BLOB (fonte di verità)."""
        self._create_vec_table()
        memory_ids = {row[0] for row in self.conn.execute("SELECT id FROM memories")}
        vec_ids = {row[0] for row in self.conn.execute("SELECT id FROM vec_memories")}
        to_insert = memory_ids - vec_ids
        to_delete = vec_ids - memory_ids
        if not to_insert and not to_delete:
            return
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            for memory_id in to_insert:
                blob = self.conn.execute("SELECT embedding FROM memories WHERE id = ?", (memory_id,)).fetchone()[0]
                self.conn.execute("INSERT INTO vec_memories(id, embedding) VALUES (?, ?)", (memory_id, blob))
            for memory_id in to_delete:
                self.conn.execute("DELETE FROM vec_memories WHERE id = ?", (memory_id,))
            self.conn.execute("COMMIT")
        except BaseException:
            self._rollback()
            raise

    def _rollback(self) -> None:
        try:
            self.conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass  # nessuna transazione attiva: non mascherare l'errore originale

    def _check_open(self) -> sqlite3.Connection:
        if self.conn is None:
            raise ValueError("STORAGE_CLOSED: connessione già chiusa")
        return self.conn

    # ------------------------------------------------------------------ validazione

    def _validate_record(self, record: MemoryRecord, file_sha256: str, expected_path: str) -> None:
        if not isinstance(record, MemoryRecord):
            raise ValueError("INVALID_REQUEST: record non è un MemoryRecord")
        memory = record.memory
        if not isinstance(memory, MemoryView):
            raise ValueError("INVALID_REQUEST: memory non è una MemoryView")
        if not os.path.isabs(memory.file_path):
            raise ValueError(f"INVALID_REQUEST: file_path non assoluto: {memory.file_path!r}")
        if memory.file_path != expected_path:
            raise ValueError(
                f"INVALID_REQUEST: file_path del record {memory.file_path!r} != file sostituito {expected_path!r}"
            )
        if not memory.chunk_key:
            raise ValueError("INVALID_REQUEST: chunk_key vuota")
        for name, value in (
            ("id", memory.id),
            ("content_sha256", memory.content_sha256),
            ("source_sha256", memory.source_sha256),
        ):
            if not _is_sha256_hex(value):
                raise ValueError(f"INVALID_REQUEST: {name} non è un digest sha256 minuscolo: {value!r}")
        if memory.modality not in MODALITY_VALUES:
            raise ValueError(f"INVALID_REQUEST: modality non ammessa: {memory.modality!r}")
        if not math.isfinite(memory.created_at) or memory.created_at < 0:
            raise ValueError(f"INVALID_REQUEST: created_at non finito o negativo: {memory.created_at!r}")
        for name in ("context_header", "snippet_text", "content_text"):
            if not isinstance(getattr(memory, name), str):
                raise ValueError(f"INVALID_REQUEST: {name} non è una stringa")
        if len(memory.snippet_text) > _MAX_SNIPPET_CHARS:
            raise ValueError(f"INVALID_REQUEST: snippet_text oltre {_MAX_SNIPPET_CHARS} caratteri")
        if not _is_sha256_hex(file_sha256):
            raise ValueError(f"INVALID_REQUEST: file_sha256 non è un digest sha256 minuscolo: {file_sha256!r}")
        if memory.source_sha256 != file_sha256:
            raise ValueError(f"FILE_HASH_MISMATCH: source_sha256 del record {memory.id} differisce dall'hash del file")
        expected_id = make_memory_id(memory.file_path, memory.chunk_key, memory.content_sha256)
        if memory.id != expected_id:
            raise ValueError(
                f"MEMORY_ID_MISMATCH: id {memory.id} non deriva da (path, chunk_key, content_sha256): {expected_id}"
            )
        validate_vector(record.embedding)

    # ------------------------------------------------------------------ transazioni

    def _vec_write(self, rows: list[tuple[str, bytes]], removed_ids: list[str]) -> None:
        for memory_id, blob in rows:
            self.conn.execute("DELETE FROM vec_memories WHERE id = ?", (memory_id,))
            self.conn.execute("INSERT INTO vec_memories(id, embedding) VALUES (?, ?)", (memory_id, blob))
        for memory_id in removed_ids:
            self.conn.execute("DELETE FROM vec_memories WHERE id = ?", (memory_id,))

    def _bump_generation(self, previous: int | None = None) -> None:
        if previous is None:
            row = self.conn.execute("SELECT value FROM metadata WHERE key='index_generation'").fetchone()
            previous = int(row[0])
        self.conn.execute("UPDATE metadata SET value = ? WHERE key='index_generation'", (str(previous + 1),))

    # ------------------------------------------------------------------ API (contratti §4)

    def replace_file(self, file_path: str, file_sha256: str, records: list[MemoryRecord]) -> None:
        """Sostituzione atomica di tutte le tracce di un file con i chunk dati.

        Valida tutto prima del commit; gli ID sopravvissuti conservano
        created_at e gli archi (cascade tocca solo le righe rimosse).
        """
        conn = self._check_open()
        canonical = _canonical_path(file_path)
        if not isinstance(records, list):
            raise ValueError("INVALID_REQUEST: records non è una lista")
        for record in records:
            self._validate_record(record, file_sha256, canonical)
        new_ids = {record.memory.id for record in records}
        if len(new_ids) != len(records):
            raise ValueError("INVALID_REQUEST: ID duplicati nella stessa sostituzione")

        conn.execute("BEGIN IMMEDIATE")
        try:
            old_rows = dict(conn.execute("SELECT id, payload_json FROM memories WHERE file_path = ?", (canonical,)))
            old_ids = set(old_rows)
            # Stesso ID con contenuto diverso: scrittura corrotta o identità violata, si rifiuta.
            for memory_id in new_ids & self._existing_ids(new_ids):
                stored = json.loads(
                    conn.execute("SELECT payload_json FROM memories WHERE id = ?", (memory_id,)).fetchone()[0]
                )
                incoming = next(record.memory.content_sha256 for record in records if record.memory.id == memory_id)
                if stored["content_sha256"] != incoming:
                    raise ValueError(
                        f"MEMORY_ID_MISMATCH: ID {memory_id} esiste già con contenuto {stored['content_sha256']}, "
                        f"in arrivo {incoming}"
                    )

            surviving = new_ids & old_ids
            obsolete = old_ids - new_ids
            old_created = {}
            for memory_id in surviving:
                old_created[memory_id] = json.loads(old_rows[memory_id])["created_at"]

            conn.execute(
                "INSERT INTO files(file_path, sha256, indexed_at) VALUES (?, ?, ?) "
                "ON CONFLICT(file_path) DO UPDATE SET sha256 = excluded.sha256, indexed_at = excluded.indexed_at",
                (canonical, file_sha256, time.time()),
            )
            vec_rows = []
            for record in records:
                memory = record.memory
                if memory.id in surviving:
                    payload = asdict(memory)
                    payload["created_at"] = old_created[memory.id]
                    blob = _payload_json_from_dict(payload)
                    conn.execute(
                        "UPDATE memories SET payload_json = ?, modality = ?, embedding = ? WHERE id = ?",
                        (blob, memory.modality, _encode_vector(record.embedding), memory.id),
                    )
                else:
                    blob = _payload_json(memory)
                    conn.execute(
                        "INSERT INTO memories(id, file_path, modality, payload_json, embedding) VALUES (?, ?, ?, ?, ?)",
                        (memory.id, canonical, memory.modality, blob, _encode_vector(record.embedding)),
                    )
                vec_rows.append((memory.id, _encode_vector(record.embedding)))
            for memory_id in obsolete:
                conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            if self.vector_backend_effective == "sqlite_vec":
                self._vec_write(vec_rows, sorted(obsolete))
            self._bump_generation(previous=int(self._read_meta("index_generation")))
            conn.execute("COMMIT")
        except BaseException:
            self._rollback()
            raise

    def file_hash(self, file_path: str) -> str | None:
        """Digest dell'ultima versione indicizzata con successo, o None."""
        conn = self._check_open()
        row = conn.execute("SELECT sha256 FROM files WHERE file_path = ?", (_canonical_path(file_path),)).fetchone()
        return None if row is None else row[0]

    def list_files(self) -> list[str]:
        conn = self._check_open()
        return [row[0] for row in conn.execute("SELECT file_path FROM files ORDER BY file_path")]

    def index_generation(self) -> int:
        return int(self._read_meta("index_generation"))

    def delete_by_filepath(self, file_path: str) -> int:
        """Rimuove il file e i suoi record; gli archi cadono a cascata solo per gli ID rimossi."""
        conn = self._check_open()
        canonical = _canonical_path(file_path)
        conn.execute("BEGIN IMMEDIATE")
        try:
            ids = [row[0] for row in conn.execute("SELECT id FROM memories WHERE file_path = ?", (canonical,))]
            if not ids:
                self._rollback()
                return 0
            conn.execute("DELETE FROM memories WHERE file_path = ?", (canonical,))
            conn.execute("DELETE FROM files WHERE file_path = ?", (canonical,))
            if self.vector_backend_effective == "sqlite_vec":
                self._vec_write([], ids)
            self._bump_generation(previous=int(self._read_meta("index_generation")))
            conn.execute("COMMIT")
        except BaseException:
            self._rollback()
            raise
        return len(ids)

    # ------------------------------------------------------------------ ricordi di sessione (F3)

    def register_session_memories(self, session_id: str, entries: Sequence[tuple[str, str]]) -> None:
        """Registra nel registry i ricordi di sessione (memory_id, file_path).

        Transazione piccola, senza bump di index_generation: l'inserimento dei
        ricordi resta compito di replace_file sul file scritto dal runner sotto
        run_dir/<session>/. Idempotente per la stessa sessione; un ricordo già
        registrato per un'altra sessione è un errore (un ricordo, una sessione).
        """
        conn = self._check_open()
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("INVALID_REQUEST: session_id vuota o non stringa")
        normalized = []
        for entry in entries:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise ValueError("INVALID_REQUEST: entry non è una coppia (memory_id, file_path)")
            memory_id, file_path = entry
            if not _is_sha256_hex(memory_id):
                raise ValueError(f"INVALID_REQUEST: memory_id non è un digest sha256: {memory_id!r}")
            if not isinstance(file_path, str) or not os.path.isabs(file_path):
                raise ValueError(f"INVALID_REQUEST: file_path non assoluto: {file_path!r}")
            normalized.append((memory_id, file_path))

        conn.execute("BEGIN IMMEDIATE")
        try:
            for memory_id, file_path in normalized:
                registered = conn.execute(
                    "SELECT session_id, file_path FROM session_memories WHERE memory_id = ?", (memory_id,)
                ).fetchone()
                if registered is not None:
                    if registered[0] != session_id:
                        raise ValueError(
                            f"SESSION_MEMORY_CONFLICT: {memory_id} già registrato per la sessione {registered[0]!r}"
                        )
                    if registered[1] != file_path:
                        raise ValueError(f"INVALID_REQUEST: path divergente per {memory_id} già registrato")
                    continue  # stessa sessione, stessi dati: idempotente
                indexed = conn.execute("SELECT file_path FROM memories WHERE id = ?", (memory_id,)).fetchone()
                if indexed is None:
                    raise ValueError(f"MEMORY_NOT_FOUND: {memory_id} non è nell'archivio")
                if indexed[0] != file_path:
                    raise ValueError(
                        f"INVALID_REQUEST: file_path {file_path!r} non corrisponde a quello indicizzato "
                        f"per {memory_id} ({indexed[0]!r})"
                    )
                conn.execute(
                    "INSERT INTO session_memories(session_id, memory_id, file_path, promoted, created_at) "
                    "VALUES (?, ?, ?, 0, ?)",
                    (session_id, memory_id, file_path, time.time()),
                )
            conn.execute("COMMIT")
        except BaseException:
            self._rollback()
            raise

    def mark_session_memory_promoted(self, memory_id: str) -> bool:
        """True solo alla prima promozione; False se assente o già promosso (idempotente)."""
        conn = self._check_open()
        if not _is_sha256_hex(memory_id):
            raise ValueError(f"INVALID_REQUEST: memory_id non è un digest sha256: {memory_id!r}")
        cursor = conn.execute(
            "UPDATE session_memories SET promoted = 1 WHERE memory_id = ? AND promoted = 0", (memory_id,)
        )
        return cursor.rowcount == 1

    def get_session_memory(self, memory_id: str) -> dict | None:
        """Registry del ricordo di sessione: {'session_id', 'promoted'} oppure None."""
        conn = self._check_open()
        if not _is_sha256_hex(memory_id):
            raise ValueError(f"INVALID_REQUEST: memory_id non è un digest sha256: {memory_id!r}")
        row = conn.execute(
            "SELECT session_id, promoted FROM session_memories WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            return None
        return {"session_id": row[0], "promoted": bool(row[1])}

    def purge_session_memories(self, session_id: str) -> int:
        """Rimuove atomicamente tutti i ricordi di una sessione e ritorna quanti.

        Un'unica transazione: registry → DELETE memories (cascade su archi e
        registry), mirror vec, righe files dei path della sessione solo se
        nessun altro ricordo le referenzia, bump di index_generation.
        Sessione inesistente: 0, nessun cambio.
        """
        conn = self._check_open()
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("INVALID_REQUEST: session_id vuota o non stringa")
        conn.execute("BEGIN IMMEDIATE")
        try:
            rows = conn.execute(
                "SELECT memory_id, file_path FROM session_memories WHERE session_id = ?", (session_id,)
            ).fetchall()
            if not rows:
                self._rollback()
                return 0
            ids = [memory_id for memory_id, _ in rows]
            paths = sorted({file_path for _, file_path in rows})
            for memory_id in ids:
                conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            if self.vector_backend_effective == "sqlite_vec":
                self._vec_write([], ids)
            for path in paths:
                conn.execute(
                    "DELETE FROM files WHERE file_path = ? AND NOT EXISTS "
                    "(SELECT 1 FROM memories WHERE file_path = files.file_path)",
                    (path,),
                )
            conn.execute("DELETE FROM session_memories WHERE session_id = ?", (session_id,))
            self._bump_generation(previous=int(self._read_meta("index_generation")))
            conn.execute("COMMIT")
        except BaseException:
            self._rollback()
            raise
        return len(ids)

    def get_memory(self, memory_id: str) -> MemoryView | None:
        conn = self._check_open()
        row = conn.execute(
            "SELECT payload_json, file_path, modality FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            return None
        return _view_from_row(row[0], memory_id, row[1], row[2])

    def reinforce_synapses(self, clicked_id: str, other_ids: list[str], delta: float = 0.25) -> None:
        """Rinforza simmetricamente gli archi del ricordo cliccato.

        Deduplica other_ids, esclude sé stesso, ignora ID rimossi (letti nella
        stessa transazione) e limita il peso a 1.0. Non tocca i BLOB embedding
        né la generazione dell'archivio: il grafo non è un cambio d'indice.
        """
        conn = self._check_open()
        if not _is_sha256_hex(clicked_id):
            raise ValueError(f"INVALID_REQUEST: clicked_id non è un digest sha256: {clicked_id!r}")
        if not isinstance(other_ids, list) or not all(isinstance(other_id, str) for other_id in other_ids):
            raise ValueError("INVALID_REQUEST: other_ids non è una lista di stringhe")
        if not math.isfinite(delta) or delta <= 0:
            raise ValueError(f"INVALID_REQUEST: delta non positivo o non finito: {delta!r}")
        unique_others = []
        for other_id in other_ids:
            if other_id != clicked_id and other_id not in unique_others:
                unique_others.append(other_id)

        conn.execute("BEGIN IMMEDIATE")
        try:
            if conn.execute("SELECT 1 FROM memories WHERE id = ?", (clicked_id,)).fetchone() is None:
                raise ValueError(f"MEMORY_NOT_FOUND: clicked_id {clicked_id} non è nell'archivio")
            now = time.time()
            for other_id in unique_others:
                if conn.execute("SELECT 1 FROM memories WHERE id = ?", (other_id,)).fetchone() is None:
                    continue
                for source_id, target_id in ((clicked_id, other_id), (other_id, clicked_id)):
                    row = conn.execute(
                        "SELECT weight FROM synaptic_edges WHERE source_id = ? AND target_id = ?",
                        (source_id, target_id),
                    ).fetchone()
                    new_weight = min(1.0, (row[0] if row else 0.0) + delta)
                    conn.execute(
                        "INSERT INTO synaptic_edges(source_id, target_id, weight, updated_at) VALUES (?, ?, ?, ?) "
                        "ON CONFLICT(source_id, target_id) DO UPDATE SET "
                        "weight = excluded.weight, updated_at = excluded.updated_at",
                        (source_id, target_id, new_weight, now),
                    )
            conn.execute("COMMIT")
        except BaseException:
            self._rollback()
            raise

    def hybrid_search(self, query_vec: list[float], settings: SearchSettings) -> list[ScoredMemory]:
        """Ricerca esatta sull'intero archivio; non applica le soglie UI.

        Il punteggio è calcolato su tutti i record (mai top-k prima della
        calibrazione): il backend NumPy fa il prodotto matrice-vettore, quello
        sqlite-vec i coseni equivalenti su righe complete. Restituisce al
        massimo settings.top_k risultati con winning_source=None.
        """
        conn = self._check_open()
        if not isinstance(settings, SearchSettings):
            raise ValueError("INVALID_REQUEST: settings non è un SearchSettings")
        if isinstance(settings.top_k, bool) or not isinstance(settings.top_k, int) or settings.top_k <= 0:
            raise ValueError(f"INVALID_REQUEST: top_k deve essere un intero positivo: {settings.top_k!r}")
        if settings.score_mode not in ("cosine", "zscore"):
            raise ValueError(f"INVALID_REQUEST: score_mode non ammesso: {settings.score_mode!r}")
        for name in ("gamma", "activation_threshold", "min_cosine", "min_final"):
            value = getattr(settings, name)
            if not math.isfinite(value):
                raise ValueError(f"INVALID_REQUEST: {name} non finito: {value!r}")
        if settings.gamma < 0:
            raise ValueError(f"INVALID_REQUEST: gamma negativo: {settings.gamma!r}")
        validate_vector(query_vec)

        records = self.all_memories()
        if not records:
            return []
        if self.vector_backend_effective == "sqlite_vec":
            query_blob = _encode_vector(list(query_vec))
            cosines_by_id = {}
            for memory_id, distance in conn.execute(
                "SELECT id, vec_distance_cosine(embedding, ?) FROM vec_memories", (query_blob,)
            ):
                if distance is None:
                    raise ValueError(f"CORRUPT_ROW: vettore di norma nulla per {memory_id}")
                cosines_by_id[memory_id] = min(1.0, max(-1.0, 1.0 - float(distance)))
            cosines = [cosines_by_id[record.memory.id] for record in records]
        else:
            cosines = ranking.cosine_scores(records, list(query_vec)).tolist()
        edges = [
            (row[0], row[1], float(row[2]))
            for row in conn.execute("SELECT source_id, target_id, weight FROM synaptic_edges")
        ]
        scored = ranking.rank_from_cosines(records, cosines, edges, settings)
        return scored[: settings.top_k]

    def all_memories(self) -> list[MemoryRecord]:
        """Snapshot consistente dei record correnti, ordinato per ID."""
        conn = self._check_open()
        records = []
        for payload_json, file_path, modality, blob in conn.execute(
            "SELECT payload_json, file_path, modality, embedding FROM memories ORDER BY id"
        ):
            row_id = json.loads(payload_json)["id"]
            records.append(
                MemoryRecord(_view_from_row(payload_json, row_id, file_path, modality), _decode_vector(blob))
            )
        return records

    def close(self) -> None:
        if self.conn is not None:
            with closing(self.conn):
                pass
            self.conn = None

    # ------------------------------------------------------------------ interni

    def _read_meta(self, key: str) -> str:
        conn = self._check_open()
        row = conn.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
        if row is None:
            raise ValueError(f"CORRUPT_ROW: metadata chiave {key!r} assente")
        return row[0]

    def _existing_ids(self, candidate_ids: set[str]) -> set[str]:
        found = set()
        for memory_id in candidate_ids:
            if self.conn.execute("SELECT 1 FROM memories WHERE id = ?", (memory_id,)).fetchone():
                found.add(memory_id)
        return found


def _payload_json_from_dict(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, allow_nan=False)


def _view_from_row(payload_json: str, memory_id: str, file_path: str, modality: str) -> MemoryView:
    payload = json.loads(payload_json)
    if payload.get("id") != memory_id or payload.get("file_path") != file_path or payload.get("modality") != modality:
        raise ValueError(f"CORRUPT_ROW: payload incoerente con le colonne per {memory_id}")
    try:
        return MemoryView(**payload)
    except TypeError as exc:
        raise ValueError(f"CORRUPT_ROW: payload non ricostruibile per {memory_id}: {exc}") from exc
