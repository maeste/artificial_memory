"""Test del launcher con figli fake (Orchestrator, O3).

I figli fake implementano il wire contract (health, snapshot controllo) senza
modelli né GTK: la supervisione viene verificata senza backend reali.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import make_runtime_paths, new_run_id  # noqa: E402
from src.contracts import SpaceSpec  # noqa: E402
from src.launcher import Launcher  # noqa: E402

READY_INFERENCE = r"""
import json, sys, zmq, uuid
config = json.load(open(sys.argv[1]))
ctx = zmq.Context.instance()
rep = ctx.socket(zmq.REP)
rep.bind(config['paths']['embed_endpoint'])
while True:
    try:
        raw = rep.recv_string()
    except zmq.Again:
        continue
    kind = json.loads(raw)['type']
    if kind == 'health.request':
        body = {'schema_version': 1, 'type': 'health.response',
                'payload': {'ready': True, 'space_id': config['space_id'],
                            'backend': config['space']['backend'], 'error': None}}
    else:
        body = {'schema_version': 1, 'type': 'embed.response',
                'payload': {'request_id': json.loads(raw)['payload'].get('request_id', ''),
                            'success': False, 'space_id': config['space_id'],
                            'embedding': [], 'error_code': 'MODEL_UNAVAILABLE', 'error': 'fake'}}
    rep.send_string(json.dumps(body))
"""

NEVER_READY = r"""
import sys, time
time.sleep(float(sys.argv[2]) if len(sys.argv) > 2 else 120)
"""

RUNNER_FAKE = r"""
import json, sys, time, zmq
config = json.load(open(sys.argv[1]))
ctx = zmq.Context.instance()
pub = ctx.socket(zmq.PUB)
pub.bind(config['paths']['events_endpoint'])
rep = ctx.socket(zmq.REP)
rep.bind(config['paths']['control_endpoint'])
state = {'event_id': 'ev-1', 'sequence': 1, 'session_id': 'sess-1', 'space_id': config['space_id'],
         'backend': config['space']['backend'], 'emitted_at': time.time(), 'reason': 'status',
         'latest_inputs': [], 'results': [], 'pinned': [], 'sensor_status': {}, 'dropped_inputs': 0}
pub.send_string(json.dumps({'schema_version': 1, 'type': 'live.state', 'payload': state}))
while True:
    try:
        raw = rep.recv_string(flags=zmq.NOBLOCK)
    except zmq.Again:
        time.sleep(0.05)
        continue
    body = {'schema_version': 1, 'type': 'control.response',
            'payload': {'command_id': json.loads(raw)['payload']['command_id'], 'success': True,
                        'state': state, 'session_inputs': [], 'lease_expires_at': None,
                        'error_code': None, 'error': None}}
    rep.send_string(json.dumps(body))
"""

RUNNER_DIES = r"""
import sys, time
time.sleep(0.5)
sys.exit(3)
"""


@pytest.fixture
def fake_tree(tmp_path: Path) -> dict:
    scripts = {'inference': tmp_path / 'fake_inference.py', 'never': tmp_path / 'never_ready.py',
               'runner': tmp_path / 'fake_runner.py', 'runner_dies': tmp_path / 'runner_dies.py',
               'sleep': tmp_path / 'sleep.py'}
    scripts['inference'].write_text(READY_INFERENCE)
    scripts['never'].write_text(NEVER_READY)
    scripts['runner'].write_text(RUNNER_FAKE)
    scripts['runner_dies'].write_text(RUNNER_DIES)
    scripts['sleep'].write_text("import time\ntime.sleep(600)\n")
    return {key: str(value) for key, value in scripts.items()}


def make_launcher(tmp_path: Path, fake_tree: dict, *, inference='inference', runner='runner',
                  headless=True, startup_timeout=8.0) -> Launcher:
    space = SpaceSpec(backend='mock', model_id='semantic-hash-mock', revision='1')
    db_path = str(tmp_path / 'brain-test.db')
    paths = make_runtime_paths(new_run_id(), db_path, str(tmp_path / 'watch'),
                               str(tmp_path / 'exports'))
    config_path = Path(paths.run_dir) / 'config.json'
    config_path.write_text(json.dumps({
        'space': {'backend': space.backend, 'model_id': space.model_id,
                  'revision': space.revision, 'dimension': space.dimension,
                  'preprocessing': space.preprocessing},
        'space_id': space.space_id,
        'paths': {'run_dir': paths.run_dir, 'embed_endpoint': paths.embed_endpoint,
                  'events_endpoint': paths.events_endpoint, 'control_endpoint': paths.control_endpoint,
                  'db_path': paths.db_path, 'watch_dir': paths.watch_dir, 'export_dir': paths.export_dir},
    }))
    return Launcher(
        config_path=str(config_path), space=space, paths=paths, headless=headless,
        python_exe=sys.executable, startup_timeout_sec=startup_timeout,
        children={
            'inference': [sys.executable, fake_tree[inference], str(config_path)],
            'watcher': [sys.executable, fake_tree['sleep'], str(config_path)],
            'runner': [sys.executable, fake_tree[runner], str(config_path)],
            'hud': [sys.executable, fake_tree['sleep'], str(config_path)],
        },
    )


def _pids_alive(pids: list[int]) -> list[int]:
    alive = []
    for pid in pids:
        try:
            os.kill(pid, 0)
            alive.append(pid)
        except (ProcessLookupError, PermissionError):
            pass
    return alive


def test_inference_never_ready_times_out(tmp_path, fake_tree):
    launcher = make_launcher(tmp_path, fake_tree, inference='never', startup_timeout=2.0)
    code = launcher.run()
    assert code != 0
    assert launcher.error_report and 'inferenza' in launcher.error_report.lower()
    assert _pids_alive(launcher.child_pids.values()) == []


def test_headless_startup_and_sigint_teardown(tmp_path, fake_tree):
    launcher = make_launcher(tmp_path, fake_tree, headless=True)
    result = {}

    def run_it():
        result['code'] = launcher.run()

    import threading

    thread = threading.Thread(target=run_it)
    thread.start()
    deadline = time.time() + 10
    while not launcher.started_all and time.time() < deadline:
        time.sleep(0.05)
    assert launcher.started_all, 'i figli non sono partiti entro il deadline'
    state = json.loads((Path(launcher.paths.run_dir) / 'state.json').read_text())
    assert state['space_id']
    assert state['pids']['inference'] == launcher.child_pids['inference']
    launcher.request_shutdown()
    thread.join(timeout=15)
    assert result.get('code') == 0
    assert _pids_alive(list(launcher.child_pids.values())) == []
    assert not Path(launcher.paths.run_dir).exists()


def test_essential_child_death_is_fatal(tmp_path, fake_tree):
    launcher = make_launcher(tmp_path, fake_tree, runner='runner_dies')
    code = launcher.run()
    assert code != 0
    assert 'runner' in launcher.error_report
    assert _pids_alive(list(launcher.child_pids.values())) == []


def test_db_process_lock_blocks_second_launcher(tmp_path, fake_tree):
    first = make_launcher(tmp_path, fake_tree)
    import fcntl

    lock_path = first.paths.db_path + '.lock'
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    held = open(lock_path, 'w')
    fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        second = make_launcher(tmp_path, fake_tree)
        code = second.run()
        assert code != 0
        assert 'lock' in second.error_report.lower()
    finally:
        fcntl.flock(held.fileno(), fcntl.LOCK_UN)
        held.close()


def test_space_mismatch_on_health_stops_startup(tmp_path, fake_tree, monkeypatch):
    launcher = make_launcher(tmp_path, fake_tree)
    wrong = SpaceSpec(backend='mock', model_id='semantic-hash-mock', revision='1')
    object.__setattr__(launcher, '_expected_space_id', '0' * 64)
    code = launcher.run()
    assert code != 0
    assert 'space' in launcher.error_report.lower() or 'mismatch' in launcher.error_report.lower()
