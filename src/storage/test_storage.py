"""Test ST1 — schema persistente e identità dello spazio (untracked_doc/02_AGENT_1_STORAGE.md).

Nuova apertura, riapertura, SPACE_MISMATCH senza distruzione, validazione dei
vettori prima del commit, formato BLOB little-endian float32 e scelta del
backend vettoriale. Tutti i test usano tmp_path, mai data/brain-real.db.
"""

from contextlib import closing
from dataclasses import asdict, replace
import hashlib
import json
import math
import sqlite3
import struct

import pytest

from src.contracts import EMBED_DIM, MemoryRecord, SpaceSpec, make_memory_id
from src.storage.db import BrainStorage

MOCK_SPEC = SpaceSpec("mock", "semantic-hash-mock", "1")
SPACE_JSON = json.dumps(asdict(MOCK_SPEC), sort_keys=True, separators=(",", ":"))


def _raw_conn(db_path, load_vec=False):
    # isolation_level=None: ogni statement autocommit, senza transazioni implicite
    # che closing() chiuderebbe con rollback.
    conn = sqlite3.connect(db_path, isolation_level=None)
    if load_vec:
        import sqlite_vec

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    return conn


def test_new_open_creates_schema_and_metadata(storage, db_path):
    assert storage.list_files() == []
    assert storage.file_hash("/percorso/inesistente.md") is None
    assert storage.index_generation() == 0
    assert storage.get_memory("f" * 64) is None
    assert storage.all_memories() == []

    with closing(_raw_conn(db_path)) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"metadata", "files", "memories", "synaptic_edges"} <= tables
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        # foreign_keys è per-connessione: si verifica su quella dello storage.
        assert storage.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        meta = dict(conn.execute("SELECT key, value FROM metadata"))
        assert meta["schema_version"] == "1"
        assert meta["index_generation"] == "0"
        assert meta["space"] == SPACE_JSON


def test_reopen_same_space_keeps_data(storage, db_path, record_factory):
    record = record_factory("/tmp/a.md", "section:one", "testo uno", axis=0)
    storage.replace_file(record.memory.file_path, record.memory.source_sha256, [record])
    generation = storage.index_generation()
    storage.close()

    reopened = BrainStorage(db_path, MOCK_SPEC, "numpy")
    try:
        assert reopened.index_generation() == generation
        assert reopened.file_hash(record.memory.file_path) == record.memory.source_sha256
        view = reopened.get_memory(record.memory.id)
        assert view is not None
        assert view == record.memory
        assert reopened.all_memories() == [MemoryRecord(record.memory, list(record.embedding))]
    finally:
        reopened.close()


def test_space_mismatch_rejected_without_destruction(db_path, record_factory):
    spec = SpaceSpec("mock", "semantic-hash-mock", "1")
    db = BrainStorage(db_path, spec, "numpy")
    record = record_factory("/tmp/a.md", "section:one", "testo uno", axis=0)
    db.replace_file(record.memory.file_path, record.memory.source_sha256, [record])
    db.close()

    other = SpaceSpec("mock", "semantic-hash-mock", "2")
    with pytest.raises(ValueError, match="SPACE_MISMATCH"):
        BrainStorage(db_path, other, "numpy")

    with closing(_raw_conn(db_path)) as conn:
        assert conn.execute("SELECT count(*) FROM memories").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM files").fetchone()[0] == 1

    reopened = BrainStorage(db_path, spec, "numpy")
    try:
        assert reopened.get_memory(record.memory.id) == record.memory
    finally:
        reopened.close()


def test_schema_version_incompatible_is_explicit(db_path):
    db = BrainStorage(db_path, MOCK_SPEC, "numpy")
    db.close()

    with closing(_raw_conn(db_path)) as conn:
        conn.execute("UPDATE metadata SET value='2' WHERE key='schema_version'")

    with pytest.raises(ValueError, match="SCHEMA_VERSION"):
        BrainStorage(db_path, MOCK_SPEC, "numpy")


def test_unrelated_sqlite_file_is_not_reinitialized(db_path):
    with closing(_raw_conn(db_path)) as conn:
        conn.execute("CREATE TABLE alieno(id INTEGER PRIMARY KEY)")

    with pytest.raises(ValueError, match="SCHEMA_INCOMPATIBLE"):
        BrainStorage(db_path, MOCK_SPEC, "numpy")


def test_invalid_vector_rejected_before_commit(storage, record_factory):
    base = record_factory("/tmp/a.md", "section:one", "testo uno", axis=0)
    bad_vectors = [
        [0.0] * (EMBED_DIM - 1) + [1.0, 0.0],  # dimensione sbagliata
        [float("nan")] * EMBED_DIM,
        [float("inf")] + [0.0] * (EMBED_DIM - 1),
        [0.5] * EMBED_DIM,  # norma invalida
    ]
    for vector in bad_vectors:
        with pytest.raises(ValueError):
            storage.replace_file(base.memory.file_path, base.memory.source_sha256, [replace(base, embedding=vector)])

    assert storage.all_memories() == []
    assert storage.index_generation() == 0
    assert storage.file_hash(base.memory.file_path) is None


def test_memory_identity_and_digests_validated(storage, record_factory):
    record = record_factory("/tmp/a.md", "section:one", "testo uno", axis=0)

    wrong_id = replace(record, memory=replace(record.memory, id="0" * 64))
    with pytest.raises(ValueError, match="MEMORY_ID_MISMATCH"):
        storage.replace_file(wrong_id.memory.file_path, wrong_id.memory.source_sha256, [wrong_id])

    upper_digest = replace(record, memory=replace(record.memory, content_sha256=record.memory.content_sha256.upper()))
    with pytest.raises(ValueError):
        storage.replace_file(upper_digest.memory.file_path, upper_digest.memory.source_sha256, [upper_digest])

    relative = replace(record, memory=replace(record.memory, file_path="tmp/a.md"))
    with pytest.raises(ValueError):
        storage.replace_file("tmp/a.md", relative.memory.source_sha256, [relative])

    assert storage.all_memories() == []


def test_source_hash_coherence_with_file_hash(storage, record_factory):
    record = record_factory("/tmp/a.md", "section:one", "testo uno", axis=0)
    other_hash = hashlib.sha256(b"contenuto diverso").hexdigest()
    with pytest.raises(ValueError, match="FILE_HASH_MISMATCH"):
        storage.replace_file(record.memory.file_path, other_hash, [record])
    assert storage.all_memories() == []


def test_invalid_file_hash_shape(storage, record_factory):
    record = record_factory("/tmp/a.md", "section:one", "testo uno", axis=0)
    with pytest.raises(ValueError):
        storage.replace_file(record.memory.file_path, "not-a-hash", [record])


def test_backend_selection_reports_effective(db_path):
    auto = BrainStorage(db_path, MOCK_SPEC, "auto")
    try:
        # sqlite-vec è installato nel venv: auto lo carica e lo riporta come effettivo.
        assert auto.vector_backend_effective == "sqlite_vec"
        assert auto.backend_fallback is False
    finally:
        auto.close()

    forced_numpy = BrainStorage(str(db_path) + "-2.db", MOCK_SPEC, "numpy")
    try:
        assert forced_numpy.vector_backend_effective == "numpy"
    finally:
        forced_numpy.close()

    forced_vec = BrainStorage(str(db_path) + "-3.db", MOCK_SPEC, "sqlite_vec")
    try:
        assert forced_vec.vector_backend_effective == "sqlite_vec"
    finally:
        forced_vec.close()


def test_invalid_backend_rejected(db_path):
    with pytest.raises(ValueError, match="vector_backend"):
        BrainStorage(db_path, MOCK_SPEC, "hnsw")


def test_sqlite_vec_explicit_failure_and_auto_fallback(db_path, monkeypatch):
    import src.storage.db as db_module

    def broken_loader(_conn):
        raise RuntimeError("estensione simulata indisponibile")

    monkeypatch.setattr(db_module, "_load_sqlite_vec", broken_loader)

    with pytest.raises(ValueError, match="SQLITE_VEC_UNAVAILABLE"):
        BrainStorage(db_path, MOCK_SPEC, "sqlite_vec")

    fallback = BrainStorage(db_path, MOCK_SPEC, "auto")
    try:
        assert fallback.vector_backend_effective == "numpy"
        assert fallback.backend_fallback is True
        assert "sqlite-vec" in fallback.backend_note
    finally:
        fallback.close()


def test_vec_mirror_follows_memories(db_path, record_factory):
    db = BrainStorage(db_path, MOCK_SPEC, "sqlite_vec")
    try:
        record = record_factory("/tmp/a.md", "section:one", "testo uno", axis=0)
        db.replace_file(record.memory.file_path, record.memory.source_sha256, [record])
        with closing(_raw_conn(db_path, load_vec=True)) as conn:
            memories = dict(conn.execute("SELECT id, embedding FROM memories"))
            vecs = dict(conn.execute("SELECT id, embedding FROM vec_memories"))
            assert set(vecs) == set(memories)

            def unpack(blob):
                return struct.unpack(f"<{EMBED_DIM}f", blob)

            assert all(unpack(vecs[mid]) == unpack(blob) for mid, blob in memories.items())
    finally:
        db.close()


def test_embedding_blob_format(storage, db_path, record_factory):
    record = record_factory("/tmp/a.md", "section:one", "testo uno", axis=0)
    storage.replace_file(record.memory.file_path, record.memory.source_sha256, [record])

    with closing(_raw_conn(db_path)) as conn:
        blob, payload_json = conn.execute("SELECT embedding, payload_json FROM memories").fetchone()

    assert len(blob) == 4 * EMBED_DIM
    decoded = struct.unpack(f"<{EMBED_DIM}f", blob)
    assert decoded[0] == pytest.approx(1.0)
    assert decoded[1] == 0.0
    norm = math.sqrt(sum(component * component for component in decoded))
    assert norm == pytest.approx(1.0, abs=1e-6)

    payload = json.loads(payload_json)
    assert set(payload) == set(asdict(record.memory))
    assert "embedding" not in payload
    assert payload["id"] == record.memory.id


# ------------------------------------------------------------------ ST2: atomicità e apprendimento


def _edge_weight(db_path, source_id, target_id):
    with closing(_raw_conn(db_path)) as conn:
        row = conn.execute(
            "SELECT weight FROM synaptic_edges WHERE source_id = ? AND target_id = ?",
            (source_id, target_id),
        ).fetchone()
    return None if row is None else row[0]


def test_replace_preserves_links_of_surviving_chunks(storage, record_factory, db_path):
    a = record_factory("/tmp/a.md", "section:one", "testo uno", axis=0)
    b = record_factory("/tmp/b.md", "section:two", "testo due", axis=1)
    storage.replace_file(a.memory.file_path, a.memory.source_sha256, [a])
    storage.replace_file(b.memory.file_path, b.memory.source_sha256, [b])
    storage.reinforce_synapses(a.memory.id, [b.memory.id])
    new_source_hash = hashlib.sha256(b"file con altra sezione cambiata").hexdigest()
    updated = replace(a, memory=replace(a.memory, source_sha256=new_source_hash))
    storage.replace_file(a.memory.file_path, new_source_hash, [updated])
    with sqlite3.connect(db_path) as conn:
        weight = conn.execute(
            "SELECT weight FROM synaptic_edges WHERE source_id=? AND target_id=?",
            (a.memory.id, b.memory.id),
        ).fetchone()[0]
    assert weight == 0.25


def test_failed_replace_leaves_previous_records_hash_and_edges(storage, record_factory, db_path):
    a = record_factory("/tmp/a.md", "section:one", "testo uno", axis=0)
    other = record_factory("/tmp/b.md", "section:two", "testo due", axis=1)
    storage.replace_file(a.memory.file_path, a.memory.source_sha256, [a])
    storage.replace_file(other.memory.file_path, other.memory.source_sha256, [other])
    storage.reinforce_synapses(a.memory.id, [other.memory.id])
    generation_before = storage.index_generation()

    # errore al secondo chunk: nessun cambiamento parziale
    corrupt = replace(other, embedding=[0.5] * EMBED_DIM)
    new_hash = hashlib.sha256(b"nuova versione").hexdigest()
    with pytest.raises(ValueError):
        storage.replace_file(a.memory.file_path, new_hash, [a, corrupt])

    assert storage.file_hash(a.memory.file_path) == a.memory.source_sha256
    assert storage.get_memory(a.memory.id) == a.memory
    assert storage.index_generation() == generation_before
    assert _edge_weight(db_path, a.memory.id, other.memory.id) == 0.25
    assert _edge_weight(db_path, other.memory.id, a.memory.id) == 0.25


def test_id_collision_inside_transaction_rolls_back(storage, db_path, record_factory):
    record = record_factory("/tmp/a.md", "section:one", "testo uno", axis=0)
    storage.replace_file(record.memory.file_path, record.memory.source_sha256, [record])
    generation_before = storage.index_generation()

    # riga contraffatta via SQL: id valido per il nuovo contenuto, payload col vecchio
    new_content = hashlib.sha256("contenuto cambiato".encode()).hexdigest()
    forged_id = make_memory_id(record.memory.file_path, record.memory.chunk_key, new_content)
    with closing(_raw_conn(db_path)) as conn:
        conn.execute(
            "INSERT INTO memories(id, file_path, modality, payload_json, embedding) VALUES (?, ?, ?, ?, ?)",
            (
                forged_id,
                record.memory.file_path,
                "markdown",
                json.dumps({**asdict(record.memory), "id": forged_id}),
                struct.pack("<256f", *([1.0] + [0.0] * 255)),
            ),
        )

    incoming = replace(record, memory=replace(record.memory, id=forged_id, content_sha256=new_content))
    with pytest.raises(ValueError, match="MEMORY_ID_MISMATCH"):
        storage.replace_file(record.memory.file_path, record.memory.source_sha256, [incoming])

    assert storage.get_memory(record.memory.id) == record.memory
    assert storage.index_generation() == generation_before


def test_empty_replacement_keeps_file_and_removes_chunks(storage, record_factory):
    first = record_factory("/tmp/a.md", "section:one", "testo uno", axis=0)
    second = record_factory("/tmp/a.md", "section:two", "testo due", axis=1)
    # i chunk dello stesso file condividono l'hash della fonte (così fa Ingestion)
    second = replace(second, memory=replace(second.memory, source_sha256=first.memory.source_sha256))
    storage.replace_file(first.memory.file_path, first.memory.source_sha256, [first, second])
    new_hash = hashlib.sha256(b"file svuotato").hexdigest()

    storage.replace_file(first.memory.file_path, new_hash, [])

    assert storage.file_hash(first.memory.file_path) == new_hash
    assert storage.get_memory(first.memory.id) is None
    assert storage.get_memory(second.memory.id) is None
    assert storage.list_files() == [first.memory.file_path]
    assert storage.index_generation() == 2


def test_surviving_chunk_keeps_original_created_at(storage, record_factory):
    record = record_factory("/tmp/a.md", "section:one", "testo uno", axis=0)
    storage.replace_file(record.memory.file_path, record.memory.source_sha256, [record])

    reloaded = replace(record, memory=replace(record.memory, created_at=99.0))
    storage.replace_file(record.memory.file_path, record.memory.source_sha256, [reloaded])

    assert storage.get_memory(record.memory.id).created_at == 1.0


def test_files_hash_and_generation_coherence(storage, record_factory):
    a = record_factory("/tmp/a.md", "section:one", "testo uno", axis=0)
    b = record_factory("/tmp/b.md", "section:two", "testo due", axis=1)
    assert storage.index_generation() == 0

    storage.replace_file(a.memory.file_path, a.memory.source_sha256, [a])
    assert storage.index_generation() == 1
    storage.replace_file(b.memory.file_path, b.memory.source_sha256, [b])
    assert storage.index_generation() == 2
    assert storage.list_files() == [a.memory.file_path, b.memory.file_path]

    storage.replace_file(a.memory.file_path, a.memory.source_sha256, [a])  # stessa versione, nuovo commit
    assert storage.index_generation() == 3
    assert storage.file_hash(a.memory.file_path) == a.memory.source_sha256


def test_two_connections_read_complete_snapshots(storage, db_path, record_factory):
    a = record_factory("/tmp/a.md", "section:one", "testo uno", axis=0)
    b = record_factory("/tmp/b.md", "section:two", "testo due", axis=1)
    storage.replace_file(a.memory.file_path, a.memory.source_sha256, [a])
    storage.replace_file(b.memory.file_path, b.memory.source_sha256, [b])
    storage.reinforce_synapses(a.memory.id, [b.memory.id])

    reader = BrainStorage(db_path, MOCK_SPEC, "numpy")
    try:
        assert {record.memory.id for record in reader.all_memories()} == {a.memory.id, b.memory.id}
        assert reader.index_generation() == storage.index_generation()
        assert reader.file_hash(a.memory.file_path) == a.memory.source_sha256
    finally:
        reader.close()


def test_delete_by_filepath_cleans_both_edge_directions(storage, db_path, record_factory):
    a = record_factory("/tmp/a.md", "section:one", "testo uno", axis=0)
    b = record_factory("/tmp/b.md", "section:two", "testo due", axis=1)
    storage.replace_file(a.memory.file_path, a.memory.source_sha256, [a])
    storage.replace_file(b.memory.file_path, b.memory.source_sha256, [b])
    storage.reinforce_synapses(a.memory.id, [b.memory.id])

    assert storage.delete_by_filepath(a.memory.file_path) == 1
    assert storage.get_memory(a.memory.id) is None
    assert storage.file_hash(a.memory.file_path) is None
    assert storage.list_files() == [b.memory.file_path]
    assert _edge_weight(db_path, a.memory.id, b.memory.id) is None
    assert _edge_weight(db_path, b.memory.id, a.memory.id) is None
    assert storage.get_memory(b.memory.id) == b.memory
    assert storage.index_generation() == 3

    assert storage.delete_by_filepath("/tmp/mai-indicizzato.md") == 0
    assert storage.index_generation() == 3


def test_reinforce_is_symmetric_cumulative_and_capped(storage, db_path, record_factory):
    a = record_factory("/tmp/a.md", "section:one", "testo uno", axis=0)
    b = record_factory("/tmp/b.md", "section:two", "testo due", axis=1)
    storage.replace_file(a.memory.file_path, a.memory.source_sha256, [a])
    storage.replace_file(b.memory.file_path, b.memory.source_sha256, [b])

    storage.reinforce_synapses(a.memory.id, [b.memory.id])
    assert _edge_weight(db_path, a.memory.id, b.memory.id) == 0.25
    assert _edge_weight(db_path, b.memory.id, a.memory.id) == 0.25

    storage.reinforce_synapses(a.memory.id, [b.memory.id])
    assert _edge_weight(db_path, a.memory.id, b.memory.id) == 0.5
    for _ in range(3):
        storage.reinforce_synapses(a.memory.id, [b.memory.id])
    assert _edge_weight(db_path, a.memory.id, b.memory.id) == 1.0

    storage.reinforce_synapses(a.memory.id, [b.memory.id])  # mai oltre 1.0
    assert _edge_weight(db_path, a.memory.id, b.memory.id) == 1.0


def test_reinforce_dedupes_ignores_self_and_missing(storage, db_path, record_factory):
    a = record_factory("/tmp/a.md", "section:one", "testo uno", axis=0)
    b = record_factory("/tmp/b.md", "section:two", "testo due", axis=1)
    c = record_factory("/tmp/c.md", "section:three", "testo tre", axis=2)
    for record in (a, b, c):
        storage.replace_file(record.memory.file_path, record.memory.source_sha256, [record])

    storage.reinforce_synapses(a.memory.id, [b.memory.id, b.memory.id, a.memory.id, "9" * 64])
    assert _edge_weight(db_path, a.memory.id, b.memory.id) == 0.25  # deduplicato: un solo incremento
    assert _edge_weight(db_path, a.memory.id, a.memory.id) is None  # self escluso
    assert _edge_weight(db_path, a.memory.id, c.memory.id) is None

    with pytest.raises(ValueError, match="MEMORY_NOT_FOUND"):
        storage.reinforce_synapses("8" * 64, [b.memory.id])


def test_reinforce_rejects_invalid_delta(storage, db_path, record_factory):
    a = record_factory("/tmp/a.md", "section:one", "testo uno", axis=0)
    b = record_factory("/tmp/b.md", "section:two", "testo due", axis=1)
    storage.replace_file(a.memory.file_path, a.memory.source_sha256, [a])
    storage.replace_file(b.memory.file_path, b.memory.source_sha256, [b])

    for delta in (0.0, -0.25, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            storage.reinforce_synapses(a.memory.id, [b.memory.id], delta=delta)
    assert _edge_weight(db_path, a.memory.id, b.memory.id) is None


def test_reinforce_leaves_embeddings_and_generation_untouched(storage, db_path, record_factory):
    a = record_factory("/tmp/a.md", "section:one", "testo uno", axis=0)
    b = record_factory("/tmp/b.md", "section:two", "testo due", axis=1)
    storage.replace_file(a.memory.file_path, a.memory.source_sha256, [a])
    storage.replace_file(b.memory.file_path, b.memory.source_sha256, [b])
    generation_before = storage.index_generation()
    with closing(_raw_conn(db_path)) as conn:
        blobs_before = conn.execute("SELECT id, embedding FROM memories ORDER BY id").fetchall()

    storage.reinforce_synapses(a.memory.id, [b.memory.id])

    with closing(_raw_conn(db_path)) as conn:
        assert conn.execute("SELECT id, embedding FROM memories ORDER BY id").fetchall() == blobs_before
    assert storage.index_generation() == generation_before
    assert {record.memory.id for record in storage.all_memories()} == {a.memory.id, b.memory.id}
