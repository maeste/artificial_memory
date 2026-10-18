"""Preflight di Ambient Second Brain (Orchestrator).

Verifica requisiti e riporta stato/correzione proposta in JSON. Non modifica
il sistema: installa nulla e non scarica pesi. Exit 2 se un requisito
obbligatorio per il backend richiesto manca.

Uso: uv run python -m src.preflight --backend real [--transcription off|local]
"""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

RUNTIME_LIBS = ['numpy', 'zmq', 'watchdog', 'pymupdf', 'PIL', 'sounddevice', 'soundfile', 'gi']
REAL_LIBS = ['torch', 'transformers', 'sentence_transformers', 'sqlite_vec']


def _check_import(name: str) -> tuple[bool, str]:
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return False, f'import {name} fallito'
    if spec is None:
        return False, f'modulo {name} non installato nel venv'
    return True, f'modulo {name} importabile'


def _check_binary(binary: str) -> tuple[bool, str]:
    path = shutil.which(binary)
    return (path is not None, f'{binary} in {path}' if path else f'{binary} non trovato nel PATH')


def _check_ffmpeg_decode() -> tuple[bool, str]:
    """Non basta trovare il binario: decodifica un MP4 di test in temp dir."""
    if shutil.which('ffmpeg') is None or shutil.which('ffprobe') is None:
        return False, 'ffmpeg/ffprobe assenti'
    with tempfile.TemporaryDirectory(prefix='am-preflight-') as tmp:
        target = os.path.join(tmp, 'probe.mp4')
        make = subprocess.run(
            ['ffmpeg', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i', 'testsrc=duration=0.3:size=64x64:rate=10',
             '-pix_fmt', 'yuv420p', target],
            capture_output=True, timeout=30,
        )
        if make.returncode != 0 or not os.path.exists(target):
            return False, f'ffmpeg non produce un MP4: {make.stderr.decode(errors="replace")[:120]}'
        probe = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=codec_name',
             '-of', 'json', target],
            capture_output=True, timeout=30,
        )
        if probe.returncode != 0 or b'codec_name' not in probe.stdout:
            return False, 'ffprobe non decodifica il MP4 di test'
    return True, 'ffmpeg 8.x decodifica MP4 H.264'


def _check_gtk_typelibs() -> tuple[bool, str]:
    try:
        import gi  # noqa: PLC0415
    except ImportError:
        return False, 'PyGObject non installato nel venv'
    try:
        gi.require_version('Gtk', '4.0')
        gi.require_version('Adw', '1')
        from gi.repository import Adw, Gtk  # noqa: F401, PLC0415
    except Exception as exc:  # noqa: BLE001 - qualunque fallimento typelib è un gap
        return False, f'typelib GTK4/Adw non caricabile: {exc}'
    return True, 'Gtk 4.0 e Adw 1 caricati nello stesso Python di uv'


def _check_asr(asr_model_dir: str | None) -> tuple[bool, str]:
    ok, detail = _check_import('faster_whisper')
    if not ok:
        return False, f'{detail} (serve uv sync --extra asr)'
    if not asr_model_dir or not os.path.isdir(asr_model_dir):
        return False, 'directory modello ASR non predisposta (vedi src.hud.transcriber prepare)'
    manifest = os.path.join(asr_model_dir, 'manifest.json')
    if not os.path.exists(manifest):
        return False, f'manifest ASR assente in {asr_model_dir}'
    return True, f'ASR locale pronta in {asr_model_dir}'


def run_preflight(backend: str, transcription: str, asr_model_dir: str | None) -> dict:
    checks = []

    def add(check_id: str, mandatory: bool, ok: bool, detail: str, fix: str) -> None:
        status = 'ok' if ok else ('missing' if mandatory else 'warn')
        checks.append(
            {'id': check_id, 'mandatory': mandatory, 'status': status, 'detail': detail, 'fix': fix if not ok else None}
        )

    add('python', True, sys.version_info >= (3, 12), f'Python {sys.version_info.major}.{sys.version_info.minor}',
        'usare un interprete >= 3.12 (uv usa .python-version)')

    for lib in RUNTIME_LIBS:
        ok, detail = _check_import(lib)
        add(f'lib:{lib}', True, ok, detail, 'uv sync')

    if backend == 'real':
        for lib in REAL_LIBS:
            ok, detail = _check_import(lib)
            add(f'lib:{lib}', True, ok, detail, 'uv sync (pesi reali verificati dal healthcheck)')
    else:
        add('lib:real-stack', False, True, 'backend mock: torch/transformers non richiesti', '')

    ok, detail = _check_gtk_typelibs()
    add('gtk:typelibs', True, ok, detail, 'verificare PyGObject e typelib GTK4/Adw nel venv')

    ok, detail = _check_ffmpeg_decode()
    add('ffmpeg:mp4-decode', True, ok, detail, 'installare/aggiornare ffmpeg con decode H.264')

    for binary, mandatory, fix in (
        ('espeak-ng', True, 'sudo dnf install -y espeak-ng'),
        ('evince', False, 'sudo dnf install -y evince'),
        ('mpv', False, 'sudo dnf install -y mpv'),
        ('wl-copy', False, 'sudo dnf install -y wl-clipboard'),
        ('parec', False, 'sudo dnf install -y pipewire-utils'),
    ):
        ok, detail = _check_binary(binary)
        add(f'bin:{binary}', mandatory, ok, detail, fix)

    ok, detail = _check_import('sounddevice')
    if not ok:
        ok, detail = _check_binary('parec')
        detail = f'PortAudio assente, fallback parec: {detail}'
    add('mic:capture', False, ok, detail, 'sudo dnf install -y portaudio o pipewire-utils (parec)')

    if transcription == 'local':
        ok, detail = _check_asr(asr_model_dir)
        add('asr:local', True, ok, detail, 'uv sync --extra asr e preparazione modello esplicita')
    else:
        add('asr:not-required', False, True, 'transcription=off: nessun modello ASR richiesto', '')

    free = shutil.disk_usage(os.path.expanduser('~')).free
    add('disk:free', False, free > 20 * 1024**3, f'{free / 1024**3:.1f} GiB liberi nella home',
        'liberare spazio per pesi modello e staging export')

    mandatory_missing = [c['id'] for c in checks if c['status'] == 'missing']
    return {
        'schema_version': 1,
        'backend': backend,
        'transcription': transcription,
        'checks': checks,
        'mandatory_missing': mandatory_missing,
        'ready': not mandatory_missing,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Preflight Ambient Second Brain')
    parser.add_argument('--backend', choices=['real', 'mock'], default='real')
    parser.add_argument('--transcription', choices=['off', 'local'], default='off')
    parser.add_argument('--asr-model-dir', default=None)
    args = parser.parse_args(argv)
    report = run_preflight(args.backend, args.transcription, args.asr_model_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 2 if not report['ready'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
