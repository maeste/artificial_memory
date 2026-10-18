"""Fake deterministici aderenti ai contratti 1.0 per i test dei Sensori.

Sviluppo parallelo: Storage (Agent-1) e Inferenza (Agent-2) non sono ancora
pronti; questi sostituti implementano le firme §4 usate dal runner. L'integrazione
reale avviene dopo gli [INTERFACE_READY] degli agenti proprietari. Il fake di
embedding è hash-deterministico e NON prova qualità semantica (08 §1).
"""

import hashlib
import math

import numpy as np

from src.config import MOCK_MODEL_ID, MOCK_REVISION
from src.contracts import (
    EMBED_DIM,
    EmbedRequest,
    MemoryView,
    ScoredMemory,
    SearchSettings,
    SpaceSpec,
)

TEST_SPACE = SpaceSpec(backend='mock', model_id=MOCK_MODEL_ID, revision=MOCK_REVISION)


def hash_unit_vector(key: str) -> list[float]:
    """Vettore unitario deterministico da una chiave testuale (solo per test)."""
    digest = hashlib.sha256(key.encode('utf-8')).digest()
    raw = np.frombuffer(digest * (EMBED_DIM // len(digest) + 1), dtype=np.uint8)[:EMBED_DIM]
    vec = raw.astype(np.float64) / 255.0
    norm = np.linalg.norm(vec)
    if norm == 0:
        vec = np.zeros(EMBED_DIM)
        vec[0] = 1.0
        norm = 1.0
    return (vec / norm).tolist()


def unit(vector: list[float]) -> list[float]:
    arr = np.asarray(vector, dtype=np.float64)
    return (arr / np.linalg.norm(arr)).tolist()


def cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    norm = np.linalg.norm(va) * np.linalg.norm(vb)
    if norm == 0:
        return 0.0
    return float(np.dot(va, vb) / norm)


class FakeEmbedClient:
    """EmbedClient finto §4: registry testo/path → vettore, fallimento iniettabile."""

    def __init__(self, vectors: dict[str, list[float]] | None = None, space: SpaceSpec = TEST_SPACE):
        self.vectors = dict(vectors or {})
        self.space = space
        self.expected_space_id = space.space_id
        self.requests: list[EmbedRequest] = []
        self.fail_with: Exception | None = None

    def embed(self, req: EmbedRequest) -> list[float]:
        self.requests.append(req)
        if self.fail_with is not None:
            raise self.fail_with
        key = req.text_payload if req.text_payload else req.audio_path
        assert key is not None, 'request senza payload'
        return list(self.vectors.get(key) or hash_unit_vector(key))

    def health(self) -> dict:
        return {'ready': True, 'space_id': self.expected_space_id, 'backend': self.space.backend, 'error': None}

    def close(self) -> None:
        return None


class FakeStorage:
    """BrainStorage finto §4: coseno reale sui record, boost e guasti iniettabili.

    Regola fake auto-consistente: z_score = 2*cosine, final_score = 2*cosine +
    synaptic_boost. Non applica soglie UI (contratto §4).
    """

    def __init__(self, space: SpaceSpec = TEST_SPACE):
        self.space = space
        self.records: dict[str, tuple[MemoryView, list[float]]] = {}
        self.boost: dict[str, float] = {}
        self.search_calls: list[tuple[list[float], SearchSettings]] = []
        self.reinforce_calls: list[tuple[str, list[str], float]] = []
        self.fail_on_search: Exception | None = None
        self.fail_on_reinforce: Exception | None = None
        self.index_gen = 0

    def add(self, memory: MemoryView, embedding: list[float]) -> None:
        self.records[memory.id] = (memory, unit(embedding))

    def hybrid_search(self, query_vec: list[float], settings: SearchSettings) -> list[ScoredMemory]:
        self.search_calls.append((list(query_vec), settings))
        if self.fail_on_search is not None:
            raise self.fail_on_search
        scored = []
        for memory, embedding in self.records.values():
            score = cosine(query_vec, embedding)
            boost = self.boost.get(memory.id, 0.0)
            scored.append(
                ScoredMemory(
                    memory=memory,
                    cosine_score=score,
                    z_score=score * 2.0,
                    synaptic_boost=boost,
                    final_score=score * 2.0 + boost,
                )
            )
        scored.sort(key=lambda item: (-item.final_score, -item.cosine_score, item.memory.id))
        return scored[: settings.top_k]

    def get_memory(self, memory_id: str) -> MemoryView | None:
        found = self.records.get(memory_id)
        return None if found is None else found[0]

    def all_memories(self) -> list:
        return [memory for memory, _ in self.records.values()]

    def reinforce_synapses(self, clicked_id: str, other_ids: list[str], delta: float = 0.25) -> None:
        if self.fail_on_reinforce is not None:
            raise self.fail_on_reinforce
        self.reinforce_calls.append((clicked_id, list(other_ids), delta))

    def index_generation(self) -> int:
        return self.index_gen

    def file_hash(self, file_path: str) -> str | None:
        return None

    def close(self) -> None:
        return None


class FakeClock:
    """Orologio iniettabile: il runner riceve clock=clock.time nei test."""

    def __init__(self, start: float = 1_000_000.0):
        self._now = float(start)

    def time(self) -> float:
        return self._now

    def advance(self, seconds: float) -> float:
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f'INVALID_REQUEST: avanzamento invalido: {seconds!r}')
        self._now += seconds
        return self._now
