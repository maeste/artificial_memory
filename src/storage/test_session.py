"""Test F3-storage — ricordi di sessione: registry, promozione, purge atomica.

Spec [START] Enhancement F3-storage (Orchestratore, 2026-10-25): il runner
scrive file reali sotto run_dir/<session>/ e li indicizza con replace_file;
il registry session_memories rende quei ricordi cercabili e cancellabili
alla new_session. Tutto su tmp_path, mai data/.
"""

import hashlib
import os
import sqlite3
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pytest

from src.contracts import MemoryRecord, MemoryView, SearchSettings, SpaceSpec, make_memory_id
from src.storage.db import BrainStorage

MOCK_SPEC = SpaceSpec("mock", "semantic-hash-mock", "1")


def _raw_conn(db_path, load_vec=False):
    conn = sqlite3.connect(db_path, isolation_level=None)
    if load_vec:
        import sqlite_vec

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    return conn


def _record_on(path, key, text, axis):
    """Come record_factory ma SENZA risolvere il path: serve per testare la forma
    canonica del path, che il factory normalizza a resolve()."""
    source_hash = hashlib.sha256(text.encode()).hexdigest()
    content_hash = hashlib.sha256((key + "\n" + text).encode()).hexdigest()
    view = MemoryView(
        id=make_memory_id(path, key, content_hash),
        file_path=path,
        modality="markdown",
        chunk_key=key,
        content_sha256=content_hash,
        source_sha256=source_hash,
        context_header=f"{Path(path).name} > {key}",
        snippet_text=text,
        content_text=text,
        created_at=1.0,
    )
    vector = [0.0] * 256
    vector[axis] = 1.0
    return MemoryRecord(view, vector)


def _register_one(storage, record, session_id="sess-1"):
    storage.register_session_memories(session_id, [(record.memory.id, record.memory.file_path)])


def test_existing_db_without_session_table_opens_after_migration(db_path, record_factory):
    record = record_factory("/tmp/archivio.md", "chunk:one", "testo uno", axis=0)
    first = BrainStorage(db_path, MOCK_SPEC, "numpy")
    try:
        first.replace_file(record.memory.file_path, record.memory.source_sha256, [record])
    finally:
        first.close()

    with closing(_raw_conn(db_path)) as conn:
        conn.execute("DROP TABLE session_memories")  # DB pre-F3

    reopened = BrainStorage(db_path, MOCK_SPEC, "numpy")  # non deve sollevare né cancellare
    try:
        assert reopened.get_memory(record.memory.id) == record.memory
        assert reopened.index_generation() == 1
        with closing(_raw_conn(db_path)) as conn:
            columns = {row[1]: row[2] for row in conn.execute("PRAGMA table_info(session_memories)")}
            assert set(columns) == {"session_id", "memory_id", "file_path", "promoted", "created_at"}
            assert columns["promoted"].startswith("INTEGER")
    finally:
        reopened.close()


def test_register_makes_session_memories_searchable(storage, record_factory, tmp_path):
    record = record_factory(str(tmp_path / "run" / "s1" / "nota.md"), "chunk:one", "contenuto nota", axis=0)
    storage.replace_file(record.memory.file_path, record.memory.source_sha256, [record])

    _register_one(storage, record, "sess-1")

    assert storage.get_session_memory(record.memory.id) == {"session_id": "sess-1", "promoted": False}
    results = storage.hybrid_search(record.embedding, SearchSettings())
    assert results[0].memory.id == record.memory.id
    assert results[0].cosine_score == pytest.approx(1.0)


def test_register_does_not_bump_generation(storage, record_factory, tmp_path):
    record = record_factory(str(tmp_path / "run" / "s1" / "nota.md"), "chunk:one", "contenuto", axis=0)
    storage.replace_file(record.memory.file_path, record.memory.source_sha256, [record])
    generation = storage.index_generation()

    _register_one(storage, record)

    assert storage.index_generation() == generation


def test_register_rejects_unknown_memory(storage):
    with pytest.raises(ValueError, match="MEMORY_NOT_FOUND"):
        storage.register_session_memories("sess-1", [("e" * 64, "/tmp/inesistente.md")])
    assert storage.get_session_memory("e" * 64) is None


def test_register_rejects_malformed_entries(storage, record_factory, tmp_path):
    record = record_factory(str(tmp_path / "run" / "s1" / "nota.md"), "chunk:one", "contenuto", axis=0)
    storage.replace_file(record.memory.file_path, record.memory.source_sha256, [record])

    with pytest.raises(ValueError):
        storage.register_session_memories("", [(record.memory.id, record.memory.file_path)])
    with pytest.raises(ValueError):
        storage.register_session_memories("sess-1", [("corto", record.memory.file_path)])
    with pytest.raises(ValueError):
        storage.register_session_memories("sess-1", [(record.memory.id, "percorso/relativo.md")])
    with pytest.raises(ValueError):
        storage.register_session_memories("sess-1", [("solo-id",)])
    with pytest.raises(ValueError, match="non corrisponde"):
        storage.register_session_memories("sess-1", [(record.memory.id, "/tmp/altro-path.md")])

    assert storage.get_session_memory(record.memory.id) is None


def test_register_is_idempotent_for_same_session(storage, db_path, record_factory, tmp_path):
    record = record_factory(str(tmp_path / "run" / "s1" / "nota.md"), "chunk:one", "contenuto", axis=0)
    storage.replace_file(record.memory.file_path, record.memory.source_sha256, [record])

    _register_one(storage, record)
    _register_one(storage, record)  # ripetuto: nessun errore, nessuna riga duplicata

    assert storage.get_session_memory(record.memory.id) == {"session_id": "sess-1", "promoted": False}
    with closing(_raw_conn(db_path)) as conn:
        assert conn.execute("SELECT count(*) FROM session_memories").fetchone()[0] == 1


def test_register_rejects_same_memory_in_other_session(storage, db_path, record_factory, tmp_path):
    record = record_factory(str(tmp_path / "run" / "s1" / "nota.md"), "chunk:one", "contenuto", axis=0)
    storage.replace_file(record.memory.file_path, record.memory.source_sha256, [record])
    _register_one(storage, record, "sess-1")

    with pytest.raises(ValueError, match="SESSION_MEMORY_CONFLICT"):
        _register_one(storage, record, "sess-2")

    assert storage.get_session_memory(record.memory.id)["session_id"] == "sess-1"


def test_mark_promoted_is_idempotent(storage, record_factory, tmp_path):
    record = record_factory(str(tmp_path / "run" / "s1" / "nota.md"), "chunk:one", "contenuto", axis=0)

    assert storage.mark_session_memory_promoted(record.memory.id) is False  # assente

    storage.replace_file(record.memory.file_path, record.memory.source_sha256, [record])
    _register_one(storage, record)

    assert storage.mark_session_memory_promoted(record.memory.id) is True  # prima promozione
    assert storage.mark_session_memory_promoted(record.memory.id) is False  # già promosso
    assert storage.get_session_memory(record.memory.id) == {"session_id": "sess-1", "promoted": True}


def test_purge_removes_memories_files_registry_edges_and_bumps_generation(storage, db_path, record_factory, tmp_path):
    first = record_factory(str(tmp_path / "run" / "s1" / "uno.md"), "chunk:one", "uno", axis=0)
    second = record_factory(str(tmp_path / "run" / "s1" / "due.md"), "chunk:due", "due", axis=1)
    archive = record_factory("/tmp/archivio.md", "chunk:arch", "archivio", axis=2)
    for record in (first, second, archive):
        storage.replace_file(record.memory.file_path, record.memory.source_sha256, [record])
    storage.register_session_memories(
        "sess-1", [(first.memory.id, first.memory.file_path), (second.memory.id, second.memory.file_path)]
    )
    storage.reinforce_synapses(first.memory.id, [second.memory.id])
    generation = storage.index_generation()

    removed = storage.purge_session_memories("sess-1")

    assert removed == 2
    assert storage.get_memory(first.memory.id) is None
    assert storage.get_memory(second.memory.id) is None
    assert storage.file_hash(first.memory.file_path) is None
    assert storage.file_hash(second.memory.file_path) is None
    assert storage.get_session_memory(first.memory.id) is None
    assert storage.get_session_memory(second.memory.id) is None
    with closing(_raw_conn(db_path)) as conn:
        assert conn.execute("SELECT count(*) FROM synaptic_edges").fetchone()[0] == 0
    assert storage.get_memory(archive.memory.id) == archive.memory  # l'archivio resta
    assert storage.index_generation() == generation + 1


def test_purge_keeps_other_session_and_cleans_vec_mirror(db_path, record_factory, tmp_path):
    db = BrainStorage(db_path, MOCK_SPEC, "sqlite_vec")
    try:
        one = record_factory(str(tmp_path / "run" / "s1" / "uno.md"), "chunk:one", "uno", axis=0)
        two = record_factory(str(tmp_path / "run" / "s2" / "due.md"), "chunk:due", "due", axis=1)
        for record in (one, two):
            db.replace_file(record.memory.file_path, record.memory.source_sha256, [record])
        db.register_session_memories("sess-1", [(one.memory.id, one.memory.file_path)])
        db.register_session_memories("sess-2", [(two.memory.id, two.memory.file_path)])

        assert db.purge_session_memories("sess-1") == 1

        with closing(_raw_conn(db_path, load_vec=True)) as conn:
            vec_ids = {row[0] for row in conn.execute("SELECT id FROM vec_memories")}
        assert vec_ids == {two.memory.id}  # mirror pulito, il survivor resta
        assert db.get_session_memory(two.memory.id) == {"session_id": "sess-2", "promoted": False}
    finally:
        db.close()


def test_purge_nonexistent_session_returns_zero(storage):
    generation = storage.index_generation()
    assert storage.purge_session_memories("mai-esistita") == 0
    assert storage.index_generation() == generation


def test_purge_keeps_files_row_while_other_memory_references_path(storage, db_path, record_factory):
    kept = record_factory("/tmp/condiviso.md", "chunk:kept", "tenuto", axis=0)
    session_one = record_factory("/tmp/condiviso.md", "chunk:sessione", "di sessione", axis=1)
    session_one = _with_source_of(session_one, kept)
    storage.replace_file(kept.memory.file_path, kept.memory.source_sha256, [kept, session_one])
    _register_one(storage, session_one, "sess-1")

    assert storage.purge_session_memories("sess-1") == 1

    assert storage.get_memory(session_one.memory.id) is None
    assert storage.get_memory(kept.memory.id) == kept.memory  # il ricordo d'archivio resta
    assert storage.file_hash(kept.memory.file_path) == kept.memory.source_sha256  # riga files tenuta


def _with_source_of(record, donor):
    """Allinea source_sha256 di un chunk a quello del file donatore (come fa Ingestion)."""
    return MemoryRecord(replace(record.memory, source_sha256=donor.memory.source_sha256), record.embedding)


def test_identity_invariant_abspath_and_resolve_in_tmp_path(tmp_path, record_factory):
    # Invariante documentata (trappola F3): senza antenati symlink resolve() e
    # abspath coincidono e l'identità è unica qualunque forma usi il writer.
    target = tmp_path / "nota.md"
    target.write_text("contenuto", encoding="utf-8")
    resolved = str(target.resolve())
    assert resolved == os.path.abspath(resolved)

    record = record_factory(resolved, "chunk:one", "contenuto", axis=0)
    storage = BrainStorage(str(tmp_path / "brain.db"), MOCK_SPEC, "numpy")
    try:
        storage.replace_file(record.memory.file_path, record.memory.source_sha256, [record])
        assert storage.get_memory(record.memory.id) == record.memory
        assert storage.file_hash(str(target)) == record.memory.source_sha256
    finally:
        storage.close()


def test_identity_is_resolve_canonical_with_symlinked_ancestor(tmp_path):
    # Invariante documentata (trappola F3, allineata ai contratti 07 §3): l'identità
    # è Path.resolve() e lo storage canonizza i path con la STESSA regola. Le due
    # forme di un file con antenato symlink portano alla stessa identità e alla
    # stessa riga files; i file_path dei record devono però essere in forma
    # canonica risolta, altrimenti il record è rifiutato.
    real_dir = tmp_path / "reale"
    real_dir.mkdir()
    link_dir = tmp_path / "collegamento"
    os.symlink(real_dir, link_dir)
    target = link_dir / "nota.md"
    target.write_text("contenuto", encoding="utf-8")

    linked_path = str(target)  # forma abspath (attraverso il symlink)
    resolved_path = str(target.resolve())  # forma canonica
    assert linked_path != resolved_path

    canonical_record = _record_on(resolved_path, "chunk:one", "contenuto", axis=0)
    assert _record_on(linked_path, "chunk:one", "contenuto", axis=0).memory.id == canonical_record.memory.id
    storage = BrainStorage(str(tmp_path / "brain.db"), MOCK_SPEC, "numpy")
    try:
        # l'argomento di replace_file può usare qualunque forma: stessa fonte
        storage.replace_file(linked_path, canonical_record.memory.source_sha256, [canonical_record])
        assert storage.file_hash(linked_path) == storage.file_hash(resolved_path)
        storage.replace_file(resolved_path, canonical_record.memory.source_sha256, [canonical_record])
        assert len(storage.all_memories()) == 1

        # un record con file_path NON canonico è rifiutato (i contratti vogliono resolve())
        non_canonical = _record_on(linked_path, "chunk:due", "altro", axis=1)
        with pytest.raises(ValueError, match="file_path del record"):
            storage.replace_file(linked_path, non_canonical.memory.source_sha256, [non_canonical])
        assert storage.get_memory(non_canonical.memory.id) is None
    finally:
        storage.close()


def test_replace_file_accepts_runtime_style_session_paths(storage, record_factory, tmp_path):
    # Punto 3 della spec: replace_file regista già tutto per path sotto run_dir
    # (validazione, mirror vec, generazione): nessun attrito, nessuna API nuova.
    deep = tmp_path / "xdg" / "artificial-memory" / "run-123" / "sess-9" / "nota.md"
    record = record_factory(str(deep), "chunk:one", "contenuto di sessione", axis=0)

    storage.replace_file(record.memory.file_path, record.memory.source_sha256, [record])
    _register_one(storage, record, "sess-9")

    assert storage.file_hash(record.memory.file_path) == record.memory.source_sha256
    assert storage.get_session_memory(record.memory.id)["session_id"] == "sess-9"
    assert storage.hybrid_search(record.embedding, SearchSettings())[0].memory.id == record.memory.id
