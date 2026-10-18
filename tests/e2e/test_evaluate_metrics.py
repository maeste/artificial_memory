"""Test delle metriche normative di scripts/evaluate.py (08 §3, scope text-only)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))

from evaluate import (  # noqa: E402
    admissible,
    apply_thresholds,
    hits_for_query,
    metrics,
    select_calibrated,
)

FILE_NAMES = {'P1': 'f05.pdf', 'V1': 'f13.mp4', 'M1': 'f01.md'}


def _candidate(mode, mc, mf, hit, mrr, abst):
    return {
        'score_mode': mode, 'min_cosine': mc, 'min_final': mf,
        'metrics': {'agent_text': {'hit_at_5': hit, 'mrr_at_5': mrr, 'negative_abstention': abst}},
    }


def test_metrics_perfect():
    result = metrics([[True, False], [False, True, False], [True] * 5], [0, 0])
    assert result['hit_at_5'] == 1.0
    assert result['mrr_at_5'] == (1 + 0.5 + 1) / 3
    assert result['negative_abstention'] == 1.0


def test_metrics_no_hits():
    result = metrics([[False] * 5, [False] * 5], [2, 0])
    assert result['hit_at_5'] == 0.0
    assert result['mrr_at_5'] == 0.0
    assert result['negative_abstention'] == 0.5


def test_metrics_reject_empty_split():
    import pytest

    with pytest.raises(ValueError, match='INVALID_REQUEST'):
        metrics([], [])


def test_only_first_five_ranks_count():
    result = metrics([[False] * 5 + [True]], [0])
    assert result['hit_at_5'] == 0.0


def test_select_calibrated_prefers_admissible_then_mrr():
    low = _candidate('zscore', 0.30, 1.0, 1.0, 0.5, 0.9)
    high = _candidate('cosine', 0.40, 0.40, 1.0, 0.8, 0.9)
    inadmissible = _candidate('zscore', 0.20, 0.0, 1.0, 1.0, 0.6)
    assert select_calibrated([low, high, inadmissible]) is high
    assert select_calibrated([inadmissible]) is None


def test_admissible_requires_both_thresholds():
    ok = _candidate('zscore', 0.30, 1.0, 0.9, 0.9, 0.9)
    low_hit = _candidate('zscore', 0.20, 0.0, 0.7, 0.9, 0.9)
    low_abst = _candidate('zscore', 0.20, 0.0, 0.9, 0.9, 0.7)
    assert admissible(ok) and not admissible(low_hit) and not admissible(low_abst)


def test_hits_for_query_pdf_needs_exact_page():
    results = [
        {'memory': {'file_path': '/x/f05.pdf', 'page_num': 1}},
        {'memory': {'file_path': '/x/f05.pdf', 'page_num': 2}},
    ]
    assert hits_for_query(results, [{'file_id': 'P1', 'page_num': 2}], FILE_NAMES) == [False, True]


def test_hits_for_query_video_needs_half_overlap():
    results = [
        {'memory': {'file_path': '/x/f13.mp4', 'start_sec': 0.0, 'end_sec': 45.0}},
        {'memory': {'file_path': '/x/f13.mp4', 'start_sec': 40.0, 'end_sec': 85.0}},
        {'memory': {'file_path': '/x/f13.mp4', 'start_sec': 80.0, 'end_sec': 95.0}},
    ]
    hits = hits_for_query(results, [{'file_id': 'V1', 'start_sec': 40.0, 'end_sec': 85.0}], FILE_NAMES)
    assert hits == [False, True, False]


def test_hits_for_query_any_modality():
    assert hits_for_query([{'memory': {'file_path': '/x/f01.md'}}], [{'file_id': 'M1'}], FILE_NAMES) == [True]


def _result(memory_id, cosine, final, boost=0.0):
    return {
        'memory': {'id': memory_id, 'file_path': f'/x/{memory_id}.md'},
        'cosine_score': cosine, 'z_score': final, 'synaptic_boost': boost, 'final_score': final,
    }


def test_apply_thresholds_filters_and_sorts():
    results = [
        _result('a', 0.50, 2.0),
        _result('b', 0.10, 3.0),   # sotto min_cosine, nessun boost: escluso
        _result('c', 0.05, 1.8, boost=0.4),  # boost>0 ammesso anche con coseno basso
        _result('d', 0.45, 0.5),   # final sotto 1.0: escluso
    ]
    kept = apply_thresholds(results, min_cosine=0.25, min_final=1.0)
    assert [item['memory']['id'] for item in kept] == ['a', 'c']


def test_apply_thresholds_caps_at_five():
    results = [_result(f'm{i:02d}', 0.5, 2.0 - i * 0.01) for i in range(8)]
    assert len(apply_thresholds(results, 0.25, 1.0)) == 5
