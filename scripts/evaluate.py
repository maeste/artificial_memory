"""Valutazione del corpus — Ambient Second Brain (08_EVALUATION §3–4).

Metriche normative: Recall@5 (primo gold nei primi cinque, dopo soglie visive),
MRR@5 e negative abstention. La calibrazione confronta coseno/Z-score sulla
sola split calibration con grafo vuoto; il gate finale usa la configurazione
congelata. Il percorso di esecuzione riusa l'harness src.sensors.replay dei
producer reali senza GTK.

CLI di consegna (08 §4):
  uv run python scripts/evaluate.py --split calibration --backend real --report docs/evaluation/calibration.json
  uv run python scripts/evaluate.py --split test --backend real --report docs/evaluation/final.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.contracts import SearchSettings, SpaceSpec  # noqa: E402

SCHEMA_VERSION = 1


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


CALIBRATION_GRID = {
    'cosine': [(mode, mc, mc) for mode in ('cosine',) for mc in (0.20, 0.30, 0.40, 0.50, 0.60)],
    'zscore': [
        (mode, mc, mf)
        for mode in ('zscore',)
        for mc in (0.20, 0.30, 0.40, 0.50, 0.60)
        for mf in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
    ],
}


def admissible(candidate: dict) -> bool:
    """08 §3 punto 3: hit@5 >= .80 e astensione >= .80 sia testo sia audio."""
    for source in ('agent_text', 'mic_audio'):
        if candidate['metrics'][source]['hit_at_5'] < 0.80:
            return False
        if candidate['metrics'][source]['negative_abstention'] < 0.80:
            return False
    return True


def select_calibrated(candidates: list[dict]) -> dict | None:
    """08 §3 punto 4: max MRR medio, poi max astensione media, poi cosine, poi soglie alte."""
    pool = [item for item in candidates if admissible(item)]
    if not pool:
        return None

    def key(item: dict):
        mrr_mean = (
            item['metrics']['agent_text']['mrr_at_5'] + item['metrics']['mic_audio']['mrr_at_5']
        ) / 2
        abst_mean = (
            item['metrics']['agent_text']['negative_abstention']
            + item['metrics']['mic_audio']['negative_abstention']
        ) / 2
        return (
            mrr_mean,
            abst_mean,
            1 if item['score_mode'] == 'cosine' else 0,
            item['min_cosine'],
            item['min_final'],
        )

    return max(pool, key=key)


def search_settings_for(mode: str, min_cosine: float, min_final: float) -> SearchSettings:
    return SearchSettings(score_mode=mode, min_cosine=min_cosine, min_final=min_final, gamma=0.6)


def hits_for_query(results: list[dict], relevant: list[dict]) -> list[bool]:
    """Hit per rango: PDF richiede pagina esatta, video sovrapposizione >= 50% della durata gold."""
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
            if Path(memory['file_path']).name != _file_name(file_id):
                continue
            if kind == 'pdf' and memory.get('page_num') == page:
                hit = True
            elif kind == 'video':
                overlap = min(memory.get('end_sec', 0.0), g_end) - max(memory.get('start_sec', 0.0), g_start)
                if overlap >= 0.5 * (g_end - g_start):
                    hit = True
            elif kind == 'any':
                hit = True
            if hit:
                break
        hits.append(hit)
    return hits


_FILE_NAME_CACHE: dict[str, str] = {}


def _file_name(file_id: str) -> str:
    if not _FILE_NAME_CACHE:
        manifest_path = Path(_FILE_NAME_CACHE.get('__manifest', 'tests/fixtures/ambient/manifest.json'))
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            for entry in manifest['files']:
                _FILE_NAME_CACHE[entry['id']] = entry['path']
    return _FILE_NAME_CACHE.get(file_id, file_id)


def build_report(
    split: str, space: SpaceSpec, corpus_checksum: str, config: dict,
    per_source: dict, per_query: list[dict], timings: dict,
) -> dict:
    gates = {
        'S1': per_source['agent_text']['hit_at_5'] >= 0.80 and per_source['mic_audio']['hit_at_5'] >= 0.80,
        'S2': all(per_source[s]['hit_at_5'] >= (0.875 if split == 'test' else 1.0)
                  for s in ('agent_text', 'mic_audio')),
        'S3': bool(per_query) and all(any(h for h in q['hits']) for q in per_query),
        'S4': all(per_source[s]['negative_abstention'] >= 0.8 for s in ('agent_text', 'mic_audio')),
        'S5': timings.get('p95_total_sec') is not None and timings['p95_total_sec'] <= 5.0,
    }
    return {
        'schema_version': SCHEMA_VERSION,
        'split': split,
        'timestamp': config.get('timestamp'),
        'space': {'model_id': space.model_id, 'revision': space.revision, 'space_id': space.space_id,
                  'backend': space.backend},
        'corpus_checksum': corpus_checksum,
        'config': config,
        'metrics': per_source,
        'queries': per_query,
        'timings': timings,
        'gates': gates,
    }


def run_evaluation(args: argparse.Namespace) -> int:
    manifest = json.loads(Path(args.manifest).read_text())
    queries = [q for q in manifest['queries'] if q['split'] == args.split]
    if not queries:
        print(f'ERROR: split vuoto per {args.split}', file=sys.stderr)
        return 1
    try:
        sys.modules.pop('src.sensors.replay', None)
        from src.sensors.replay import ReplayHarness  # noqa: PLC0415
    except ImportError as exc:
        print(
            f'BLOCKER: harness src.sensors.replay non disponibile ({exc}); '
            'la valutazione end-to-end richiede il modulo Sensori (piano 05).',
            file=sys.stderr,
        )
        return 3
    space = SpaceSpec(**json.loads(Path(args.space_config).read_text())['space'])
    harness = ReplayHarness(space=space, manifest_path=str(Path(args.manifest)), backend=args.backend)
    per_source = {}
    per_query = []
    timings = {}
    for source in ('agent_text', 'mic_audio'):
        outcomes = harness.run(queries=queries, source=source, repeat=args.repeat)
        ranked_hits = []
        negative_counts = []
        for outcome in outcomes:
            hits = hits_for_query(outcome['results'], outcome['relevant'])
            per_query.append(
                {
                    'query_id': outcome['query_id'], 'source': source,
                    'relevant': outcome['relevant'], 'hits': hits,
                    'results': outcome['results'], 'error': outcome.get('error'),
                }
            )
            if outcome['relevant']:
                ranked_hits.append(hits)
            else:
                negative_counts.append(len(outcome['results']))
        per_source[source] = metrics(ranked_hits, negative_counts)
    report = build_report(
        split=args.split, space=space, corpus_checksum=manifest['corpus_checksum'],
        config={'repeat': args.repeat, 'backend': args.backend, 'score': vars(args.score_settings)
                if args.score_settings else None},
        per_source=per_source, per_query=per_query, timings=timings,
    )
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({'report': args.report, 'metrics': per_source, 'gates': report['gates']}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Valutazione corpus Ambient Second Brain')
    parser.add_argument('--split', choices=['calibration', 'test'], required=True)
    parser.add_argument('--backend', choices=['real', 'mock'], default='real')
    parser.add_argument('--manifest', default='tests/fixtures/ambient/manifest.json')
    parser.add_argument('--space-config', default='data/probe-config.json')
    parser.add_argument('--report', required=True)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--score-settings', default=None, help='JSON SearchSettings congelata')
    args = parser.parse_args(argv)
    if args.backend != 'real':
        print('ERROR: il gate di valutazione richiede pesi reali (mock non sostituisce la prova semantica)',
              file=sys.stderr)
        return 2
    return run_evaluation(args)


if __name__ == '__main__':
    raise SystemExit(main())
