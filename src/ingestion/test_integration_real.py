"""Integrazione completa sullo stack reale: EmbedServer (mock), EmbedClient, BrainStorage.

Il percorso reale di G3/G4 senza pesi: il backend è il SemanticHashMock
dichiarato (contratto D5 — il mock verifica cablaggio e protocollo REQ/REP,
non la qualità semantica, che resta della Fase 1).
"""

import time
import uuid

import pytest

from src.config import make_runtime_paths
from src.contracts import EmbedRequest, SearchSettings
from src.inference.client import EmbedClient
from src.inference.model import SemanticHashMock, mock_space
from src.inference.server import EmbedServer
from src.storage.db import BrainStorage

from src.ingestion.watcher import CharTokenizer, scan_directory


class CountingClient(EmbedClient):
    """Client reale che conta le richieste embed (per verificare lo skip)."""

    def __init__(self, endpoint: str, space_id: str, timeout_ms: int = 5000):
        super().__init__(endpoint, space_id, timeout_ms)
        self.embed_calls = 0

    def embed(self, req: EmbedRequest) -> list[float]:
        self.embed_calls += 1
        return super().embed(req)


@pytest.fixture
def stack(tmp_path):
    space = mock_space()
    paths = make_runtime_paths(
        run_id=f"ingest-test-{uuid.uuid4().hex[:8]}",
        db_path=str(tmp_path / "brain-mock.db"),
        watch_dir=str(tmp_path / "watch"),
        export_dir=str(tmp_path / "exports"),
    )
    server = EmbedServer(paths.embed_endpoint, space, lambda: SemanticHashMock(space))
    server.start()
    client = EmbedClient(paths.embed_endpoint, space.space_id, timeout_ms=5000)
    try:
        for _ in range(100):  # health finché il backend (mock, leggero) è pronto
            if client.health().get("ready"):
                break
            time.sleep(0.05)
        else:
            pytest.fail("server mock non pronto entro il timeout")
        storage = BrainStorage(paths.db_path, space)
        try:
            yield space, paths, storage, client
        finally:
            storage.close()
    finally:
        client.close()
        server.stop()


def test_full_stack_scan_indexes_and_retrieves_mechanically(tmp_path, stack):
    _space, _paths, storage, client = stack
    watch = tmp_path / "watch"
    watch.mkdir()
    (watch / "a.md").write_text("# Cache\nLa cache conserva i risultati.\n")
    (watch / "b.md").write_text("# Backup\nCopia giornaliera.\n")

    results = scan_directory(str(watch), storage, client, CharTokenizer(), str(tmp_path / "tmp"))
    assert set(results.values()) == {"indexed"}
    assert len(storage.list_files()) == 2
    assert storage.index_generation() == 2

    # ricerca meccanica: la query percorre client->server->storage reali
    query_vec = client.embed(EmbedRequest(request_id="q1", is_query=True, text_payload="cache"))
    scored = storage.hybrid_search(query_vec, SearchSettings(score_mode="cosine"))
    assert scored, "hybrid_search senza risultati sullo stack reale"
    expected_paths = {str((watch / "a.md").resolve()), str((watch / "b.md").resolve())}
    assert all(item.memory.file_path in expected_paths for item in scored)


def test_full_stack_second_scan_skips_inference(tmp_path, stack):
    space, paths, storage, _client = stack
    watch = tmp_path / "watch"
    watch.mkdir()
    (watch / "a.md").write_text("# Titolo\nContenuto iniziale.\n")
    counting = CountingClient(paths.embed_endpoint, space.space_id)
    try:
        results = scan_directory(str(watch), storage, counting, CharTokenizer(), str(tmp_path / "tmp"))
        assert set(results.values()) == {"indexed"}
        calls_first = counting.embed_calls
        assert calls_first > 0
        generation = storage.index_generation()

        results = scan_directory(str(watch), storage, counting, CharTokenizer(), str(tmp_path / "tmp"))
        assert set(results.values()) == {"unchanged"}
        assert counting.embed_calls == calls_first  # zero richieste nuove
        assert storage.index_generation() == generation
    finally:
        counting.close()


def test_full_stack_removal_and_reindex(tmp_path, stack):
    _space, _paths, storage, client = stack
    watch = tmp_path / "watch"
    watch.mkdir()
    source = watch / "a.md"
    source.write_text("# Titolo\nVersione 1.\n")
    scan_directory(str(watch), storage, client, CharTokenizer(), str(tmp_path / "tmp"))
    generation = storage.index_generation()

    source.unlink()
    results = scan_directory(str(watch), storage, client, CharTokenizer(), str(tmp_path / "tmp"))
    assert results[str(source.resolve())] == "deleted"
    assert storage.list_files() == []
    assert storage.all_memories() == []
    assert storage.index_generation() == generation + 1


def test_full_stack_timeout_maps_to_retry(tmp_path, stack):
    """Un TIMEOUT reale del client (zmq.Again -> EmbeddingError) viene ritentato."""
    _space, paths, storage, _client = stack
    watch = tmp_path / "watch"
    watch.mkdir()
    (watch / "a.md").write_text("# Titolo\nContenuto.\n")

    from src.inference.client import EmbeddingError

    class TimeoutFirstCall(CountingClient):
        def embed(self, req):
            if self.embed_calls == 0:
                self.embed_calls += 1
                raise EmbeddingError("TIMEOUT", "primo tentativo scaduto")
            return super().embed(req)

    flaky = TimeoutFirstCall(paths.embed_endpoint, _space.space_id)
    try:
        results = scan_directory(str(watch), storage, flaky, CharTokenizer(), str(tmp_path / "tmp"))
        assert results[str((watch / "a.md").resolve())] == "indexed"
        assert flaky.embed_calls >= 2  # tentativo fallito + retry riuscito
    finally:
        flaky.close()
