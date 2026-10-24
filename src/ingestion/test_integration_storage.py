"""Integrazione G3 con il BrainStorage reale di Agent-1 ([UNBLOCK] 2026-10-18).

Gli stessi percorsi dei test con fake (test_watcher.py) girano contro lo
storage vero: validazione digest/ID interna a Storage, conservazione di
created_at/archi per gli ID sopravvissuti, rollback atomico. Il client resta
fake finché Agent-2 non dichiara [INTERFACE_READY].
"""

import hashlib
from pathlib import Path

import pytest

from src.config import default_space
from src.contracts import EMBED_DIM
from src.storage.db import BrainStorage

from src.ingestion.test_watcher import CharTokenizer, FakeEmbedClient
from src.ingestion.watcher import index_file, scan_directory


@pytest.fixture
def real_storage(tmp_path):
    storage = BrainStorage(str(tmp_path / "brain-mock.db"), default_space("mock", "1"))
    yield storage
    storage.close()


@pytest.fixture
def client():
    return FakeEmbedClient()


@pytest.fixture
def tokenizer():
    return CharTokenizer()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_index_file_real_storage_end_to_end(tmp_path, real_storage, client, tokenizer):
    source = tmp_path / "a.md"
    source.write_text("# Cache\nLa cache conserva i risultati.\n\n# Note\nAltro contenuto.\n")
    assert index_file(str(source), real_storage, client, tokenizer, str(tmp_path / "tmp")) is True
    assert real_storage.file_hash(str(source.resolve())) == sha256_file(source)
    records = [rec for rec in real_storage.all_memories() if rec.memory.file_path == str(source.resolve())]
    assert {rec.memory.chunk_key for rec in records} == {
        "section:Cache:0:part:0",
        "section:Note:0:part:0",
    }
    # secondo giro: invariato, zero inferenza
    calls = len(client.calls)
    assert index_file(str(source), real_storage, client, tokenizer, str(tmp_path / "tmp")) is False
    assert len(client.calls) == calls


def test_real_storage_preserves_surviving_ids_created_at_and_edges(tmp_path, real_storage, client, tokenizer):
    source = tmp_path / "a.md"

    def write(backup_body: str) -> None:
        source.write_text(f"# Cache\nContenuto cache stabile.\n\n# Backup\n{backup_body}\n")

    write("Versione backup 1.")
    assert index_file(str(source), real_storage, client, tokenizer, str(tmp_path / "tmp")) is True
    records = {rec.memory.chunk_key: rec for rec in real_storage.all_memories()}
    cache_id = records["section:Cache:0:part:0"].memory.id
    backup_id = records["section:Backup:0:part:0"].memory.id
    cache_created = records["section:Cache:0:part:0"].memory.created_at
    real_storage.reinforce_synapses(cache_id, [backup_id])

    write("Versione backup 2.")
    assert index_file(str(source), real_storage, client, tokenizer, str(tmp_path / "tmp")) is True
    after = {rec.memory.chunk_key: rec for rec in real_storage.all_memories()}
    assert after["section:Cache:0:part:0"].memory.id == cache_id
    assert after["section:Cache:0:part:0"].memory.created_at == cache_created
    assert "section:Backup:0:part:0" in after
    assert after["section:Backup:0:part:0"].memory.id != backup_id
    # l'arco del backup rimosso cade; il collegamento col nuovo backup si ricrea a livello logico
    assert real_storage.get_memory(cache_id) is not None
    assert real_storage.get_memory(backup_id) is None


def test_real_storage_rollback_on_embed_failure(tmp_path, real_storage, client, tokenizer):
    source = tmp_path / "a.md"
    source.write_text("# Titolo\nPrima versione.\n")
    assert index_file(str(source), real_storage, client, tokenizer, str(tmp_path / "tmp")) is True
    old_hash = real_storage.file_hash(str(source.resolve()))
    old_ids = {rec.memory.id for rec in real_storage.all_memories()}
    generation = real_storage.index_generation()

    source.write_text("# Titolo\nSeconda versione.\n")

    class Failing(FakeEmbedClient):
        def embed(self, request):
            self.calls.append(request)
            raise RuntimeError("MODEL_UNAVAILABLE")

    failing = Failing()
    with pytest.raises(RuntimeError):
        index_file(str(source), real_storage, failing, tokenizer, str(tmp_path / "tmp"))
    assert real_storage.file_hash(str(source.resolve())) == old_hash
    assert {rec.memory.id for rec in real_storage.all_memories()} == old_ids
    assert real_storage.index_generation() == generation


def test_scan_directory_real_storage_reconciliation(tmp_path, real_storage, client, tokenizer):
    watch = tmp_path / "watch"
    watch.mkdir()
    good = watch / "good.md"
    good.write_text("# Buono\nContenuto.\n")
    bad = watch / "bad.png"
    bad.write_bytes(b"non un immagine")

    results = scan_directory(str(watch), real_storage, client, tokenizer, str(tmp_path / "tmp"))
    assert results[str(good.resolve())] == "indexed"
    assert results[str(bad.resolve())] == "error:INVALID_MEDIA"
    assert real_storage.list_files() == [str(good.resolve())]

    # file cancellato a riposo: riconciliazione dal filesystem
    good.unlink()
    results = scan_directory(str(watch), real_storage, client, tokenizer, str(tmp_path / "tmp"))
    assert results[str(good.resolve())] == "deleted"
    assert real_storage.list_files() == []


def test_real_storage_accepts_empty_version(tmp_path, real_storage, client, tokenizer):
    source = tmp_path / "vuoto.md"
    source.write_text("")
    assert index_file(str(source), real_storage, client, tokenizer, str(tmp_path / "tmp")) is True
    assert real_storage.file_hash(str(source.resolve())) == sha256_file(source)
    assert real_storage.all_memories() == []
    assert client.calls == []


def test_real_storage_vectors_have_contract_dimension(tmp_path, real_storage, client, tokenizer):
    source = tmp_path / "a.md"
    source.write_text("# Titolo\nContenuto con vettore.\n")
    assert index_file(str(source), real_storage, client, tokenizer, str(tmp_path / "tmp")) is True
    for record in real_storage.all_memories():
        assert len(record.embedding) == EMBED_DIM
