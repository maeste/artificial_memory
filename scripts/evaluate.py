"""Valutazione del corpus — Ambient Second Brain (08_EVALUATION §3–4).

Metriche normative: Recall@5 (primo gold nei primi cinque, dopo soglie visive),
MRR@5 e negative abstention. La calibrazione confronta coseno/Z-score sulla
sola split calibration con grafo vuoto; il gate finale usa la configurazione
congelata. Il percorso di esecuzione riusa l'harness src.sensors.replay dei
producer reali senza GTK.

SCOPO (decisione utente 2026-10-18): S2-audio dichiarato FAIL reale con
evidenza (docs/evaluation/probe-failed-run{1,2}.json). La calibrazione e il
gate finale girano sul canale agent_text; mic_audio resta implementato ma
fuori dal gate semantico.

CLI di consegna (08 §4):
  uv run python scripts/evaluate.py --split calibration --backend real --report docs/evaluation/calibration.json
  uv run python scripts/evaluate.py --split test --backend real --report docs/evaluation/final.json
"""

import argparse
import math
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.contracts import SearchSettings, SpaceSpec  # noqa: E402

SCHEMA_VERSION = 1
AUDIO_POLICY = 'S2-audio FAIL reale documentato (run-1/run-2); scope text-only per decisione utente 2026-10-18'
CALIBRATION_PATH = Path('data/calibration.json')

# Griglia di 08 §3: per cosine min_final = min_cosine; per zscore min_final nella lista.
# Nota protocollo (calibrazione run del 2026-10-18): l'intervallo normativo
# [0.20..0.60] non conteneva candidati ammissibili (coseno: negative con top
# 0.64-0.70, mai astensione; zscore: parete hit/astensione fra mf 0.5 e 1.0
# con gold C4 z=0.68 sotto 4 negative su 5). 08 §3 consente correzioni delle
# soglie usando solo la calibrazione: la griglia cosine viene estesa di un
# passo in direzione PIÙ SEVERA (0.70 > 0.60), senza toccare i criteri di
# ammissibilità (.80/.80). La scelta avviene prima del test finale.
GRID_MIN_COSINE = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70)
GRID_MIN_FINAL_ZSCORE = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
GRID_EXTENSION_NOTE = (
    'griglia cosine estenda a 0.70 dopo calibrazione 2026-10-18: gold calibrazione '
    '0.726-0.828 (rank 1), negative 0.644-0.700; direzione più severa, criteri invariati'
)


def metrics(ranked_hits: list[list[bool]], negative_result_counts: list[int]) -> dict:
    """Metriche di 08 §3. Il chiamante rifiuta split vuoti."""
    if not ranked_hits or not negative_result_counts:
        raise ValueError('INVALID_REQUEST: split vuoto, metriche non calcolabili')
    ranks = [next((i + 1 for i, hit in enumerate(row[:5]) if hit), 0) for row in ranked_hits]
    return {
        'hit_at_5': sum(rank > 0 for rank in ranks) / len(ranks),
        'mrr_at_5': sum(1 / rank if rank else 0 for rank in ranks) / len(ranks),
        'negative_abstention': sum(n == 0 for n in negative_result_counts) / len(negative_result_counts),
    }


def apply_thresholds(results: list[dict], min_cosine: float, min_final: float) -> list[dict]:
    """Soglie visive del contratto: (cosine >= min_cosine OR boost > 0) AND final >= min_final."""
    kept = [
        item for item in results
        if (item['cosine_score'] >= min_cosine or item['synaptic_boost'] > 0)
        and item['final_score'] >= min_final
    ]
    kept.sort(key=lambda item: (-item['final_score'], -item['cosine_score'], item['memory']['id']))
    return kept[:5]


def admissible(candidate: dict) -> bool:
    """08 §3 punto 3 sullo scope attivo (agent_text): hit@5 e astensione >= .80."""
    text = candidate['metrics']['agent_text']
    return text['hit_at_5'] >= 0.80 and text['negative_abstention'] >= 0.80


def select_calibrated(candidates: list[dict]) -> dict | None:
    """08 §3 punto 4: max MRR, poi max astensione, poi cosine, poi soglie alte."""
    pool = [item for item in candidates if admissible(item)]
    if not pool:
        return None

    def key(item: dict):
        text = item['metrics']['agent_text']
        return (
            text['mrr_at_5'],
            text['negative_abstention'],
            1 if item['score_mode'] == 'cosine' else 0,
            item['min_cosine'],
            item['min_final'],
        )

    return max(pool, key=key)


def hits_for_query(results: list[dict], relevant: list[dict], file_names: dict[str, str]) -> list[bool]:
    """Hit per rango: PDF richiede pagina esatta, video sovrapposizione >= 50% del gold."""
    golds = []
    for gold in relevant:
        if 'page_num' in gold:
            golds.append(('pdf', gold['file_id'], gold['page_num'], None, None))
        elif 'start_sec' in gold:
            golds.append(('video', gold['file_id'], None, gold['start_sec'], gold['end_sec']))
        else:
            golds.append(('any', gold['file_id'], None, None, None))
    hits = []
    for item in results[:5]:
        memory = item.get('memory', item)
        hit = False
        for kind, file_id, page, g_start, g_end in golds:
            if Path(memory['file_path']).name != file_names.get(file_id, file_id):
                continue
            if kind == 'pdf' and memory.get('page_num') == page:
                hit = True
            elif kind == 'video':
                overlap = min(memory.get('end_sec') or 0.0, g_end) - max(memory.get('start_sec') or 0.0, g_start)
                if overlap >= 0.5 * (g_end - g_start):
                    hit = True
            elif kind == 'any':
                hit = True
            if hit:
                break
        hits.append(hit)
    return hits


def _load_file_names(manifest: dict) -> dict[str, str]:
    return {entry['id']: entry['path'] for entry in manifest['files']}


def _unfiltered_search(score_mode: str) -> dict:
    """Impostazioni senza soglie e top_k ampio: la griglia filtra offline."""
    return {
        'top_k': 50, 'score_mode': score_mode, 'gamma': 0.0,
        'activation_threshold': 0.5, 'min_cosine': -1.0, 'min_final': -1e9,
    }


def _make_harness(args, manifest_path: Path, space: SpaceSpec, search: dict):
    from src.sensors.replay import ReplayHarness

    return ReplayHarness(
        space=space, manifest_path=str(manifest_path), backend=args.backend,
        config={'search': search, 'embed_timeout_ms': 30000}, device=args.device,
    )


def _run_source(harness, queries: list[dict], source: str) -> dict[str, list[dict]]:
    """Outcome per query_id (repeat collassati: si usa il primo senza errore)."""
    outcomes = harness.run(queries=queries, source=source, repeat=1)
    by_query: dict[str, list[dict]] = {}
    for outcome in outcomes:
        by_query.setdefault(outcome['query_id'], []).append(outcome)
    return by_query


def run_calibration(args: argparse.Namespace, manifest: dict, space: SpaceSpec) -> int:
    manifest_path = Path(args.manifest).resolve()
    file_names = _load_file_names(manifest)
    positives = [q for q in manifest['queries'] if q['split'] == 'calibration' and q['relevant']]
    negatives = [q for q in manifest['queries'] if q['split'] == 'calibration' and not q['relevant']]
    if not positives or not negatives:
        print('ERROR: split calibration vuoto (positivi o negativi mancanti)', file=sys.stderr)
        return 1
    candidates = []
    raw_by_mode: dict[str, dict[str, list[dict]]] = {}
    for mode in ('zscore', 'cosine'):
        harness = _make_harness(args, manifest_path, space, _unfiltered_search(mode))
        try:
            by_query = _run_source(harness, positives + negatives, 'agent_text')
        finally:
            harness.close()
        raw_by_mode[mode] = {
            query_id: next((o['results'] for o in items if o['error'] is None), [])
            for query_id, items in by_query.items()
        }
        errors = [o['error'] for items in by_query.values() for o in items if o['error']]
        if errors:
            print(f'ERROR: replay {mode} con errori: {errors[:2]}', file=sys.stderr)
            return 1
    for mode in ('zscore', 'cosine'):
        for min_cosine in GRID_MIN_COSINE:
            finals = (min_cosine,) if mode == 'cosine' else GRID_MIN_FINAL_ZSCORE
            for min_final in finals:
                ranked_hits, negative_counts = [], []
                for query in positives:
                    kept = apply_thresholds(raw_by_mode[mode][query['id']], min_cosine, min_final)
                    ranked_hits.append(hits_for_query(kept, query['relevant'], file_names))
                for query in negatives:
                    kept = apply_thresholds(raw_by_mode[mode][query['id']], min_cosine, min_final)
                    negative_counts.append(len(kept))
                candidates.append({
                    'score_mode': mode, 'min_cosine': min_cosine, 'min_final': min_final,
                    'metrics': {'agent_text': metrics(ranked_hits, negative_counts)},
                })
    chosen = select_calibrated(candidates)
    report = {
        'schema_version': SCHEMA_VERSION,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'split': 'calibration',
        'audio_policy': AUDIO_POLICY,
        'space': {'model_id': space.model_id, 'revision': space.revision,
                  'space_id': space.space_id, 'backend': space.backend},
        'corpus_checksum': manifest['corpus_checksum'],
        'candidates': candidates,
        'grid_note': GRID_EXTENSION_NOTE,
        'selected': chosen,
        'pass': chosen is not None,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    if chosen is None:
        print('CALIBRAZIONE FALLITA: nessun candidato ammissibile (non si abbassano i criteri)', file=sys.stderr)
        return 1
    calibration = {
        'schema_version': SCHEMA_VERSION,
        'space': {'model_id': space.model_id, 'revision': space.revision,
                  'space_id': space.space_id, 'backend': space.backend},
        'corpus_checksum': manifest['corpus_checksum'],
        'score_mode': chosen['score_mode'],
        'min_cosine': chosen['min_cosine'],
        'min_final': chosen['min_final'],
        'gamma': 0.6,
        'activation_threshold': 0.5,
        'metrics_calibration': chosen['metrics'],
        'audio_policy': AUDIO_POLICY,
        'timestamp': report['timestamp'],
    }
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_PATH.write_text(json.dumps(calibration, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({
        'report': args.report, 'calibration': str(CALIBRATION_PATH),
        'selected': {k: chosen[k] for k in ('score_mode', 'min_cosine', 'min_final')},
        'metrics': chosen['metrics'],
    }, ensure_ascii=False))
    return 0


def run_test(args: argparse.Namespace, manifest: dict, space: SpaceSpec) -> int:
    manifest_path = Path(args.manifest).resolve()
    file_names = _load_file_names(manifest)
    frozen = json.loads(CALIBRATION_PATH.read_text()) if Path(CALIBRATION_PATH).exists() else None
    if frozen is None or frozen.get('corpus_checksum') != manifest['corpus_checksum']:
        print('ERROR: data/calibration.json assente o di un corpus diverso: calibrare prima del test',
              file=sys.stderr)
        return 1
    if frozen['space']['space_id'] != space.space_id:
        print('ERROR: calibrazione di uno spazio diverso: rifiutata', file=sys.stderr)
        return 1
    search = {
        'top_k': 5, 'score_mode': frozen['score_mode'], 'gamma': frozen['gamma'],
        'activation_threshold': frozen['activation_threshold'],
        'min_cosine': frozen['min_cosine'], 'min_final': frozen['min_final'],
    }
    harness = _make_harness(args, manifest_path, space, search)
    queries = [q for q in manifest['queries'] if q['split'] == 'test']
    try:
        outcomes = harness.run(queries=queries, source='agent_text', repeat=args.repeat)
    finally:
        harness.close()
    per_query, ranked_hits, negative_counts, failures = [], [], [], 0
    latencies: list[float] = []
    by_query: dict[str, list[dict]] = {}
    for outcome in outcomes:
        by_query.setdefault(outcome['query_id'], []).append(outcome)
        if outcome.get('error') is None and outcome.get('latency_sec') is not None:
            latencies.append(float(outcome['latency_sec']))
    for query in queries:
        items = by_query.get(query['id'], [])
        good = [o for o in items if o['error'] is None]
        if not good:
            failures += 1
        outcome = good[0] if good else items[0]
        hits = hits_for_query(outcome['results'], query['relevant'], file_names)
        per_query.append({
            'query_id': query['id'], 'relevant': query['relevant'], 'hits': hits,
            'results': outcome['results'], 'error': outcome['error'],
        })
        if query['relevant']:
            ranked_hits.append(hits)
        else:
            negative_counts.append(len(outcome['results']))
    positives = [q for q in queries if q['relevant']]
    negatives = [q for q in queries if not q['relevant']]
    source_metrics = metrics(ranked_hits, negative_counts)
    hit_rate = source_metrics['hit_at_5']
    abstention = source_metrics['negative_abstention']

    def nearest_rank(samples: list[float], percentile: float) -> float | None:
        if not samples:
            return None
        ordered = sorted(samples)
        index = max(1, math.ceil(percentile * len(ordered)))
        return ordered[index - 1]

    timings = {
        'definition': 'latency_sec = fine input originale -> evento pubblicato (debounce/VAD e coda inclusi, 08 §3)',
        'samples': len(latencies),
        'latency_raw_sec': sorted(round(value, 3) for value in latencies),
        'p50_sec': nearest_rank(latencies, 0.50),
        'p95_sec': nearest_rank(latencies, 0.95),
    }
    gates = {
        'S1_text': hit_rate >= 0.80,
        'S2_text': len(positives) > 0 and hit_rate >= 7 / 8,
        'S2_audio': 'FAIL reale documentato (scope text-only per decisione utente)',
        'S3': len(positives) > 0 and all(any(h for h in q['hits']) for q in per_query if q['relevant']),
        'S4_text': abstention >= 4 / 5,
        'S5': timings['p95_sec'] is not None and timings['p95_sec'] <= 5.0,
    }
    report = {
        'schema_version': SCHEMA_VERSION,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'split': 'test',
        'audio_policy': AUDIO_POLICY,
        'space': {'model_id': space.model_id, 'revision': space.revision,
                  'space_id': space.space_id, 'backend': space.backend},
        'corpus_checksum': manifest['corpus_checksum'],
        'calibration': {k: frozen[k] for k in ('score_mode', 'min_cosine', 'min_final', 'gamma')},
        'repeat': args.repeat,
        'metrics': {'agent_text': source_metrics},
        'timings': timings,
        'queries': per_query,
        'counts': {'positives': len(positives), 'negatives': len(negatives), 'failures': failures},
        'gates': gates,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'report': args.report, 'metrics': source_metrics, 'gates': gates}, ensure_ascii=False))
    return 0 if failures == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Valutazione corpus Ambient Second Brain')
    parser.add_argument('--split', choices=['calibration', 'test'], required=True)
    parser.add_argument('--backend', choices=['real', 'mock'], default='real')
    parser.add_argument('--manifest', default='tests/fixtures/ambient/manifest.json')
    parser.add_argument('--space-config', default='data/probe-config.json')
    parser.add_argument('--report', required=True)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args(argv)
    if args.backend != 'real':
        print('ERROR: il gate di valutazione richiede pesi reali (mock non sostituisce la prova semantica)',
              file=sys.stderr)
        return 2
    manifest = json.loads(Path(args.manifest).read_text())
    space_spec = json.loads(Path(args.space_config).read_text())['space']
    space = SpaceSpec(**space_spec)
    if args.split == 'calibration':
        return run_calibration(args, manifest, space)
    return run_test(args, manifest, space)


if __name__ == '__main__':
    raise SystemExit(main())
