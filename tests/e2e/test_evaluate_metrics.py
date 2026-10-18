"""Test delle metriche normative di scripts/evaluate.py (08 §3)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))

from evaluate import hits_for_query, metrics, select_calibrated  # noqa: E402


def _candidate(mode, mc, mf, hit_t, hit_a, ab_t, ab_a):
    def source(hit, abst):
        return {'hit_at_5': hit, 'mrr_at_5': hit, 'negative_abstention': abst}

    return {
        'score_mode': mode, 'min_cosine': mc, 'min_final': mf,
        'metrics': {'agent_text': source(hit_t, ab_t), 'mic_audio': source(hit_a, ab_a)},
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


def test_select_calibrated_prefers_amissible_then_mrr():
    low = _candidate('zscore', 0.30, 1.0, 1.0, 1.0, 0.9, 0.9)
    low['metrics']['agent_text']['mrr_at_5'] = 0.5
    high = _candidate('cosine', 0.40, 0.40, 1.0, 1.0, 0.9, 0.9)
    high['metrics']['mic_audio']['mrr_at_5'] = 0.6
    inadmissible = _candidate('zscore', 0.20, 0.0, 1.0, 0.5, 1.0, 1.0)
    assert select_calibrated([low, high, inadmissible]) is high
    assert select_calibrated([inadmissible]) is None


def test_hits_for_query_pdf_needs_exact_page(monkeypatch):
    monkeypatch.setattr('evaluate._FILE_NAME_CACHE',
                        {'P1': 'f05.pdf', 'V1': 'f13.mp4', 'M1': 'f01.md'})
    results = [
        {'memory': {'file_path': '/x/f05.pdf', 'page_num': 1}},
        {'memory': {'file_path': '/x/f05.pdf', 'page_num': 2}},
    ]
    assert hits_for_query(results, [{'file_id': 'P1', 'page_num': 2}]) == [False, True]


def test_hits_for_query_video_needs_half_overlap(monkeypatch):
    monkeypatch.setattr('evaluate._FILE_NAME_CACHE', {'V1': 'f13.mp4'})
    results = [
        {'memory': {'file_path': '/x/f13.mp4', 'start_sec': 0.0, 'end_sec': 45.0}},
        {'memory': {'file_path': '/x/f13.mp4', 'start_sec': 40.0, 'end_sec': 85.0}},
        {'memory': {'file_path': '/x/f13.mp4', 'start_sec': 80.0, 'end_sec': 95.0}},
    ]
    hits = hits_for_query(results, [{'file_id': 'V1', 'start_sec': 40.0, 'end_sec': 85.0}])
    assert hits == [False, True, False]


def test_hits_for_query_any_modality(monkeypatch):
    monkeypatch.setattr('evaluate._FILE_NAME_CACHE', {'M1': 'f01.md'})
    assert hits_for_query([{'memory': {'file_path': '/x/f01.md'}}], [{'file_id': 'M1'}]) == [True]
