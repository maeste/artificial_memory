"""Test ST3 — ranking esatto, calibrazione e propagazione (untracked_doc/02_AGENT_1_STORAGE.md).

Test puri con vettori controllati per la matematica, test DB per i pesi forniti
dal grafo, equivalenza fra backend NumPy e sqlite-vec. L'estensione è installata
nel venv: nessuno skip è accettato.
"""

import math
from dataclasses import replace

import numpy as np
import pytest

from src.contracts import MemoryRecord, MemoryView, SearchSettings, SpaceSpec, make_memory_id
from src.storage import ranking
from src.storage.db import BrainStorage

MOCK_SPEC = SpaceSpec("mock", "semantic-hash-mock", "1")
QUERY = [1.0] + [0.0] * 255


def aligned_vector(cos):
    """Vettore unitario con coseno `cos` rispetto a QUERY (asse 0)."""
    rest = math.sqrt(max(0.0, 1.0 - cos * cos) / 255.0)
    return [cos] + [rest] * 255


def make_record(path, key, cos, modality="markdown"):
    view = MemoryView(
        id=make_memory_id(path, key, "a" * 64),
        file_path=path,
        modality=modality,
        chunk_key=key,
        content_sha256="a" * 64,
        source_sha256="b" * 64,
        context_header=f"{path} > {key}",
        snippet_text=key,
        content_text=key,
        created_at=0.0,
    )
    return MemoryRecord(view, aligned_vector(cos))


def test_normalized_scores_small_group_uses_fixed_calibration():
    assert ranking.normalized_scores(np.array([0.54, 0.495])) == pytest.approx([1.6, 1.3])
    assert ranking.normalized_scores(np.array([0.51])) == pytest.approx([1.4])


def test_normalized_scores_large_group_uses_std_with_floor():
    # deviazione quasi zero: il floor 0.05 evita la divisione instabile
    assert ranking.normalized_scores(np.array([0.5, 0.5, 0.5])) == pytest.approx([0.0, 0.0, 0.0])
    assert ranking.normalized_scores(np.array([0.4, 0.5, 0.6])) == pytest.approx(
        [-1.224744871391589, 0.0, 1.224744871391589]
    )


def test_rank_scores_all_records_and_orders_by_final():
    records = [
        make_record("/tmp/a.md", "k:a", 0.54),
        make_record("/tmp/b.md", "k:b", 0.495),
        make_record("/tmp/c.md", "k:c", 0.51, modality="pdf"),
    ]
    scored = ranking.rank(records, QUERY, [], SearchSettings(top_k=10))

    assert [item.memory.id for item in scored] == [
        make_memory_id("/tmp/a.md", "k:a", "a" * 64),
        make_memory_id("/tmp/c.md", "k:c", "a" * 64),
        make_memory_id("/tmp/b.md", "k:b", "a" * 64),
    ]
    values = {item.memory.id: item for item in scored}
    assert values[records[0].memory.id].z_score == pytest.approx(1.6)
    assert values[records[1].memory.id].z_score == pytest.approx(1.3)
    assert values[records[2].memory.id].z_score == pytest.approx(1.4)
    assert all(item.synaptic_boost == 0.0 for item in scored)
    # Storage non assegna sorgenti live (contratti §5)
    assert all(item.matched_sources == [] and item.winning_source is None for item in scored)


def test_rank_cosine_mode_bases_on_cosine_but_keeps_z():
    records = [
        make_record("/tmp/a.md", "k:a", 0.54),
        make_record("/tmp/b.md", "k:b", 0.495),
        make_record("/tmp/c.md", "k:c", 0.51, modality="pdf"),
    ]
    scored = ranking.rank(records, QUERY, [], SearchSettings(top_k=10, score_mode="cosine"))
    values = {item.memory.id: item for item in scored}
    assert values[records[0].memory.id].final_score == pytest.approx(0.54)
    assert values[records[0].memory.id].z_score == pytest.approx(1.6)
    assert [item.memory.id for item in scored] == [records[0].memory.id, records[2].memory.id, records[1].memory.id]


def test_rank_empty_archive_returns_empty():
    assert ranking.rank([], QUERY, [], SearchSettings()) == []


def test_rank_tie_breaks_by_id_ascending():
    first = make_record("/tmp/x.md", "k:x", 0.5)
    second = make_record("/tmp/y.md", "k:y", 0.5)
    records = [second, first]  # ordine di ingresso inverso
    scored = ranking.rank(records, QUERY, [], SearchSettings(top_k=10))
    assert [item.memory.id for item in scored] == [first.memory.id, second.memory.id]


def test_boost_single_hop_flips_order_without_recirculation():
    records = [
        make_record("/tmp/a.md", "k:a", 0.54),  # base z 1.6
        make_record("/tmp/b.md", "k:b", 0.495),  # base z 1.3
        make_record("/tmp/c.md", "k:c", 0.51, modality="pdf"),  # base z 1.4
    ]
    settings = SearchSettings(top_k=10, gamma=0.6)

    plain = {item.memory.id: item for item in ranking.rank(records, QUERY, [], settings)}
    assert plain[records[1].memory.id].synaptic_boost == 0.0
    assert [item.memory.id for item in ranking.rank(records, QUERY, [], settings)] == [
        records[0].memory.id,
        records[2].memory.id,
        records[1].memory.id,
    ]

    bidirectional = [
        (records[0].memory.id, records[1].memory.id, 0.25),
        (records[1].memory.id, records[0].memory.id, 0.25),
    ]
    with_edges = {item.memory.id: item for item in ranking.rank(records, QUERY, bidirectional, settings)}
    # boost(B) = gamma * max(base(A) * w) = 0.6 * 1.6 * 0.25 = 0.24 → 1.54 > 1.4 di C
    assert with_edges[records[1].memory.id].synaptic_boost == pytest.approx(0.24)
    assert with_edges[records[1].memory.id].final_score == pytest.approx(1.54)
    # il verso opposto rinforza anche A: 0.6 * 1.3 * 0.25 = 0.195, senza sommare due volte
    assert with_edges[records[0].memory.id].synaptic_boost == pytest.approx(0.195)
    assert [item.memory.id for item in ranking.rank(records, QUERY, bidirectional, settings)] == [
        records[0].memory.id,
        records[1].memory.id,
        records[2].memory.id,
    ]


def test_boost_requires_active_seed():
    under_threshold = [
        make_record("/tmp/a.md", "k:a", 0.36),  # z = 0.4 < activation_threshold
        make_record("/tmp/b.md", "k:b", 0.495),
    ]
    edge = (
        under_threshold[0].memory.id,
        under_threshold[1].memory.id,
        0.25,
    )
    scored = {
        item.memory.id: item
        for item in ranking.rank(under_threshold, QUERY, [edge, edge[::-1]], SearchSettings(top_k=10))
    }
    assert scored[under_threshold[1].memory.id].synaptic_boost == 0.0

    below_min_cosine = [
        make_record("/tmp/a.md", "k:a", 0.54),  # z 1.6 ma cos 0.54 < min_cosine 0.6
        make_record("/tmp/b.md", "k:b", 0.495),
    ]
    edge2 = (below_min_cosine[0].memory.id, below_min_cosine[1].memory.id, 0.25)
    strict = SearchSettings(top_k=10, min_cosine=0.6)
    scored = {item.memory.id: item for item in ranking.rank(below_min_cosine, QUERY, [edge2, edge2[::-1]], strict)}
    assert scored[below_min_cosine[1].memory.id].synaptic_boost == 0.0


def test_gamma_zero_kills_boost():
    records = [
        make_record("/tmp/a.md", "k:a", 0.54),
        make_record("/tmp/b.md", "k:b", 0.495),
    ]
    edge = (records[0].memory.id, records[1].memory.id, 0.25)
    scored = ranking.rank(records, QUERY, [edge, edge[::-1]], SearchSettings(top_k=10, gamma=0.0))
    assert all(item.synaptic_boost == 0.0 for item in scored)


def test_rank_does_not_mutate_records():
    records = [
        make_record("/tmp/a.md", "k:a", 0.54),
        make_record("/tmp/b.md", "k:b", 0.495),
        make_record("/tmp/c.md", "k:c", 0.51, modality="pdf"),
    ]
    snapshot = [(record.memory, list(record.embedding)) for record in records]
    edge = (records[0].memory.id, records[1].memory.id, 0.25)
    ranking.rank(records, QUERY, [edge, edge[::-1]], SearchSettings(top_k=10))
    assert [(record.memory, record.embedding) for record in records] == snapshot


# ------------------------------------------------------------------ livelli DB


def test_hybrid_search_empty_db_returns_empty(db_path):
    db = BrainStorage(db_path, MOCK_SPEC, "numpy")
    try:
        assert db.hybrid_search(QUERY, SearchSettings()) == []
    finally:
        db.close()


def test_hybrid_search_validation_and_no_ui_thresholds(storage, record_factory):
    # asse 1: coseno 0 con QUERY su asse 0 → final negativo, comunque restituito
    low = record_factory("/tmp/low.md", "chunk:low", "basso", axis=1)
    storage.replace_file(low.memory.file_path, low.memory.source_sha256, [low])

    results = storage.hybrid_search(QUERY, SearchSettings(top_k=5))
    assert len(results) == 1
    assert results[0].final_score < 1.0  # min_final è soglia UI, non di storage

    with pytest.raises(ValueError):
        storage.hybrid_search([0.0] * 255, SearchSettings())
    with pytest.raises(ValueError):
        storage.hybrid_search([1.0] + [0.0] * 254 + [0.5], SearchSettings())
    with pytest.raises(ValueError):
        storage.hybrid_search(QUERY, SearchSettings(top_k=0))
    with pytest.raises(ValueError):
        storage.hybrid_search(QUERY, SearchSettings(score_mode="rocket"))
    with pytest.raises(ValueError):
        storage.hybrid_search(QUERY, SearchSettings(gamma=-0.1))


def test_db_supplies_weights_for_boost(storage, record_factory):
    a = record_factory("/tmp/a.md", "chunk:a", "alpha", axis=0)
    a = replace(a, embedding=aligned_vector(0.54))
    b = record_factory("/tmp/b.md", "chunk:b", "beta", axis=0)
    b = replace(b, embedding=aligned_vector(0.495))
    c = record_factory("/tmp/c.md", "chunk:c", "gamma", axis=0)
    c = MemoryRecord(replace(c.memory, modality="pdf"), aligned_vector(0.51))
    for record in (a, b, c):
        storage.replace_file(record.memory.file_path, record.memory.source_sha256, [record])
    settings = SearchSettings(top_k=10, gamma=0.6)

    assert [item.memory.id for item in storage.hybrid_search(QUERY, settings)] == [
        a.memory.id,
        c.memory.id,
        b.memory.id,
    ]

    storage.reinforce_synapses(a.memory.id, [b.memory.id], delta=0.25)
    with_edge = storage.hybrid_search(QUERY, settings)
    assert [item.memory.id for item in with_edge] == [a.memory.id, b.memory.id, c.memory.id]
    values = {item.memory.id: item for item in with_edge}
    assert values[b.memory.id].synaptic_boost == pytest.approx(0.24)
    assert values[b.memory.id].final_score == pytest.approx(1.54)
    assert values[c.memory.id].synaptic_boost == 0.0
    assert values[a.memory.id].matched_sources == []
    assert values[a.memory.id].winning_source is None


def test_hybrid_search_applies_top_k(storage, record_factory):
    for index in range(6):
        cos = 0.5 - index * 0.05
        record = make_record(f"/tmp/f{index}.md", f"chunk:{index}", cos)
        storage.replace_file(record.memory.file_path, record.memory.source_sha256, [record])

    results = storage.hybrid_search(QUERY, SearchSettings(top_k=2))
    assert len(results) == 2
    assert results[0].cosine_score >= results[1].cosine_score


def test_backend_equivalence_numpy_vs_sqlite_vec(tmp_path):
    # Per ogni gruppo std ≈ 0.31: l'errore f32 del coseno sqlite-vec (≤ ~3e-6,
    # misurato) è amplificato da 1/std nella calibrazione z → l'equivalenza 1e-5
    # sui punteggi regge con gruppi ben dispersi; gruppi quasi piatti (std al
    # floor 0.05) amplificherebbero oltre la tolleranza: NumPy è il riferimento.
    corpus = [
        ("/tmp/m0.md", "k:m0", "markdown", 0.95),
        ("/tmp/m1.md", "k:m1", "markdown", 0.20),
        ("/tmp/m2.md", "k:m2", "markdown", 0.575),
        ("/tmp/p0.pdf", "k:p0", "pdf", 0.90),
        ("/tmp/p1.pdf", "k:p1", "pdf", 0.15),
        ("/tmp/p2.pdf", "k:p2", "pdf", 0.525),
    ]
    results = {}
    for backend in ("numpy", "sqlite_vec"):
        db = BrainStorage(str(tmp_path / f"brain-{backend}.db"), MOCK_SPEC, backend)
        try:
            assert db.vector_backend_effective == backend  # nessuno skip: estensione installata
            for path, key, modality, cos in corpus:
                record = make_record(path, key, cos, modality=modality)
                db.replace_file(record.memory.file_path, record.memory.source_sha256, [record])
            first = db.get_memory(make_memory_id("/tmp/m0.md", "k:m0", "a" * 64))
            second = db.get_memory(make_memory_id("/tmp/p0.pdf", "k:p0", "a" * 64))
            db.reinforce_synapses(first.id, [second.id], delta=0.25)
            results[backend] = db.hybrid_search(QUERY, SearchSettings(top_k=10))
        finally:
            db.close()

    assert [item.memory.id for item in results["numpy"]] == [item.memory.id for item in results["sqlite_vec"]]
    for left, right in zip(results["numpy"], results["sqlite_vec"]):
        assert abs(left.cosine_score - right.cosine_score) <= 1e-5
        assert abs(left.z_score - right.z_score) <= 1e-5
        assert abs(left.final_score - right.final_score) <= 1e-5
        assert left.synaptic_boost == pytest.approx(right.synaptic_boost, abs=1e-5)
