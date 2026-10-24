#!/usr/bin/env python
"""Avvio del prototipo Ambient Second Brain (Orchestrator, O3).

  uv run python run_poc.py --watch-dir ./watch_dir --backend real \
      --agent-log /path/session.log --agent-format plain

Preflight -> runtime -> inferenza -> watcher/runner -> snapshot -> HUD.
Il backend è fisso per esecuzione; il mock impone DB distinto e badge visibile.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (  # noqa: E402
    AppConfig,
    default_data_layout,
    default_space,
    make_runtime_paths,
    new_run_id,
)
from src.contracts import SearchSettings  # noqa: E402
from src.launcher import Launcher  # noqa: E402
from src.preflight import run_preflight  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Avvia il prototipo Ambient Second Brain')
    parser.add_argument('--watch-dir', required=True)
    parser.add_argument('--backend', choices=['real', 'mock'], default='real')
    parser.add_argument('--agent-log', default=None)
    parser.add_argument('--agent-format', choices=['plain', 'claude-jsonl'], default='plain')
    parser.add_argument('--no-mic', action='store_true')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--db', default=None)
    parser.add_argument('--show-all', action='store_true')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--calibration', default=None, help='path data/calibration.json congelata')
    parser.add_argument('--transcription', choices=['off', 'local'], default='off')
    parser.add_argument('--asr-model-dir', default=None)
    parser.add_argument('--model-revision', default=None,
                        help='commit HF risolto per il backend real (o data/probe-config.json)')
    parser.add_argument('--runtime-root', default=None,
                        help='radice alternativa per la directory di esecuzione (default XDG_RUNTIME_DIR)')
    parser.add_argument('--transcribe-budget-sec', type=float, default=300.0)
    return parser.parse_args(argv)


def resolve_revision(args: argparse.Namespace) -> str:
    if args.model_revision:
        return args.model_revision
    probe_config = Path('data/probe-config.json')
    if probe_config.exists():
        resolved = json.loads(probe_config.read_text()).get('space', {}).get('revision')
        if resolved:
            return resolved
    raise SystemExit(
        'ERROR: revisione reale non risolta. Fornire --model-revision COMMIT oppure '
        'produrre data/probe-config.json dopo la risoluzione con Agent-2 (mai "main").'
    )


def build_search_settings(args: argparse.Namespace, calibration_path: str | None) -> SearchSettings:
    if calibration_path and Path(calibration_path).exists():
        frozen = json.loads(Path(calibration_path).read_text())
        return SearchSettings(
            score_mode=frozen['score_mode'], min_cosine=frozen['min_cosine'],
            min_final=frozen['min_final'], gamma=0.6, activation_threshold=0.5,
        )
    return SearchSettings()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    preflight = run_preflight(args.backend, args.transcription, args.asr_model_dir)
    if not preflight['ready']:
        print(json.dumps(preflight, indent=2, ensure_ascii=False), file=sys.stderr)
        print('Preflight: requisiti obbligatori mancanti, avvio negato.', file=sys.stderr)
        return 2

    root = Path(__file__).resolve().parent
    db_real, db_mock, export_dir = default_data_layout(str(root))
    db_path = args.db or (db_mock if args.backend == 'mock' else db_real)
    revision = '1' if args.backend == 'mock' else resolve_revision(args)
    space = default_space(args.backend, revision)
    run_id = new_run_id()
    paths = make_runtime_paths(run_id, db_path, args.watch_dir, export_dir,
                               runtime_root_override=args.runtime_root)
    config_path = Path(paths.run_dir) / 'config.json'
    config = AppConfig(space=space, paths=paths, search=build_search_settings(args, args.calibration),
                       device=args.device, mic_enabled=not args.no_mic)
    config_path.write_text(json.dumps({
        'space': {
            'backend': space.backend, 'model_id': space.model_id, 'revision': space.revision,
            'dimension': space.dimension, 'preprocessing': space.preprocessing,
        },
        'space_id': space.space_id,
        'paths': {
            'run_dir': paths.run_dir, 'embed_endpoint': paths.embed_endpoint,
            'events_endpoint': paths.events_endpoint, 'control_endpoint': paths.control_endpoint,
            'db_path': paths.db_path, 'watch_dir': paths.watch_dir, 'export_dir': paths.export_dir,
        },
        'search': {
            'top_k': config.search.top_k, 'score_mode': config.search.score_mode,
            'gamma': config.search.gamma, 'activation_threshold': config.search.activation_threshold,
            'min_cosine': config.search.min_cosine, 'min_final': config.search.min_final,
        },
        'device': config.device, 'mic_enabled': config.mic_enabled,
        'alpha': config.alpha, 'source_ttl_sec': config.source_ttl_sec,
        'command_timeout_ms': config.command_timeout_ms, 'embed_timeout_ms': config.embed_timeout_ms,
        'startup_timeout_sec': config.startup_timeout_sec, 'max_pins': config.max_pins,
        'session_limit_sec': config.session_limit_sec, 'session_max_inputs': config.session_max_inputs,
        'session_audio_max_bytes': config.session_audio_max_bytes,
        'show_all': args.show_all,
        'agent_log': args.agent_log, 'agent_format': args.agent_format,
        'transcription': args.transcription, 'asr_model_dir': args.asr_model_dir,
    }, indent=2, ensure_ascii=False), encoding='utf-8')

    extra_state = {
        'mic_enabled': config.mic_enabled, 'show_all': args.show_all,
        'transcription': args.transcription, 'agent_format': args.agent_format,
        'asr_model_dir': args.asr_model_dir,
    }
    base = Launcher(config_path=str(config_path), space=space, paths=paths,
                    headless=True, extra_state=extra_state)
    children = base.children
    if args.agent_log or args.no_mic:
        runner_extra = ['--agent-log', args.agent_log] if args.agent_log else []
        runner_extra += ['--agent-format', args.agent_format]
        if args.no_mic:
            runner_extra += ['--no-mic']
        children['runner'] = children['runner'] + runner_extra
    launcher = Launcher(config_path=str(config_path), space=space, paths=paths,
                        headless=args.headless, children=children, extra_state=extra_state)
    code = launcher.run()
    if launcher.error_report:
        print(f'launcher: {launcher.error_report}', file=sys.stderr)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
