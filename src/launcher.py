"""Launcher/supervisore di Ambient Second Brain (Orchestrator, O3).

Traduzione del pseudocodice normativo di 01_ORCHESTRATOR_PLAN §3:
preflight -> runtime -> inferenza (health finché ready) -> watcher e runner
-> snapshot runner -> HUD (salvo headless) -> supervisione -> teardown inverso.

I figli sono avviati con subprocess.Popen senza shell. La morte di un processo
indispensabile (inferenza/watcher/runner) termina il launcher con nome ed
errore; la chiusura dell'HUD non è fatale. Il DB è protetto da lock di
processo sul suo .lock, rilasciato dal sistema alla morte del launcher.
"""

import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import zmq

from src.contracts import (
    ControlResponse,
    HudCommand,
    RuntimePaths,
    SpaceSpec,
    decode_control_response,
    decode_health_response,
    pack_health,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEARDOWN_WAIT_SEC = 5.0
HEALTH_POLL_INTERVAL_SEC = 0.3
HEALTH_REQ_TIMEOUT_MS = 2000
ESSENTIAL = ('inference', 'watcher', 'runner')
START_ORDER = ('inference', 'watcher', 'runner', 'hud')


class Launcher:
    """Supervisore dei quattro processi locali. Testabile con figli fake."""

    def __init__(
        self,
        config_path: str,
        space: SpaceSpec,
        paths: RuntimePaths,
        headless: bool = True,
        python_exe: str | None = None,
        startup_timeout_sec: float = 300.0,
        children: dict[str, list[str]] | None = None,
        extra_state: dict | None = None,
    ):
        self.config_path = str(config_path)
        self.space = space
        self.paths = paths
        self.headless = headless
        self.python_exe = python_exe or sys_python()
        self.startup_timeout_sec = startup_timeout_sec
        self.children = children or self._default_children()
        self.extra_state = extra_state or {}
        self.child_pids: dict[str, int] = {}
        self.error_report: str | None = None
        self.started_all = False
        self._procs: dict[str, subprocess.Popen] = {}
        self._shutdown = threading.Event()
        self._lock_handle = None
        self._ctx = zmq.Context.instance()
        self._expected_space_id = space.space_id

    # ------------------------------------------------------------------ default

    def _default_children(self) -> dict[str, list[str]]:
        config = ['--config', self.config_path]
        children = {
            'inference': [self.python_exe, '-m', 'src.inference', 'serve'] + config,
            'watcher': [self.python_exe, '-m', 'src.ingestion.watcher'] + config,
            'runner': [self.python_exe, '-m', 'src.sensors.runner'] + config,
            'hud': [self.python_exe, '-m', 'src.hud.app'] + config,
        }
        return children

    # ------------------------------------------------------------------ run

    def run(self) -> int:
        self._install_signal_handlers()
        if not self._acquire_db_lock():
            return 2
        self._write_state()
        try:
            self._start_child('inference')
            if not self._wait_inference_ready():
                return 1
            self._start_child('watcher')
            self._start_child('runner')
            if not self._wait_runner_snapshot():
                return 1
            if not self.headless:
                self._start_child('hud')
            self._write_state()
            self.started_all = True
            return self._supervise()
        finally:
            self._teardown()

    def request_shutdown(self) -> None:
        self._shutdown.set()

    # ------------------------------------------------------------------ fasi

    def _acquire_db_lock(self) -> bool:
        lock_path = self.paths.db_path + '.lock'
        Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
        handle = open(lock_path, 'w')
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            self.error_report = f'DB lock occupato: {lock_path} appartiene a un altro launcher'
            return False
        self._lock_handle = handle
        return True

    def _start_child(self, role: str) -> None:
        argv = self.children[role]
        proc = subprocess.Popen(argv, cwd=str(PROJECT_ROOT), stdin=subprocess.DEVNULL)
        self._procs[role] = proc
        self.child_pids[role] = proc.pid
        self._write_state()

    def _wait_inference_ready(self) -> bool:
        deadline = time.monotonic() + self.startup_timeout_sec
        request = pack_health('health.request', {})
        while time.monotonic() < deadline and not self._shutdown.is_set():
            if self._procs['inference'].poll() is not None:
                code = self._procs['inference'].returncode
                self.error_report = f'inferenza terminata durante lo startup (codice {code})'
                return False
            req = self._ctx.socket(zmq.REQ)
            req.setsockopt(zmq.LINGER, 0)
            req.setsockopt(zmq.RCVTIMEO, HEALTH_REQ_TIMEOUT_MS)
            try:
                req.connect(self.paths.embed_endpoint)
                try:
                    req.send_string(request)
                    raw = req.recv_string()
                except zmq.Again:
                    continue
                health = decode_health_response(raw)
                if health['ready']:
                    if health['space_id'] != self._expected_space_id:
                        self.error_report = (
                            f'SPACE_MISMATCH: health.space_id={health["space_id"]} '
                            f'atteso={self._expected_space_id}'
                        )
                        return False
                    return True
            except (ValueError, zmq.ZMQError):
                time.sleep(HEALTH_POLL_INTERVAL_SEC)
            finally:
                req.close()
            time.sleep(HEALTH_POLL_INTERVAL_SEC)
        self.error_report = f'timeout startup inferenza dopo {self.startup_timeout_sec:.0f} s'
        return False

    def _wait_runner_snapshot(self) -> bool:
        deadline = time.monotonic() + self.startup_timeout_sec
        command_id = uuid.uuid4().hex
        while time.monotonic() < deadline and not self._shutdown.is_set():
            if self._procs['runner'].poll() is not None:
                code = self._procs['runner'].returncode
                self.error_report = f'runner terminato durante lo startup (codice {code})'
                return False
            response = self._request_snapshot(command_id)
            if response is not None and response.success and response.state is not None:
                return True
            time.sleep(HEALTH_POLL_INTERVAL_SEC)
        self.error_report = f'timeout snapshot runner dopo {self.startup_timeout_sec:.0f} s'
        return False

    def _request_snapshot(self, command_id: str) -> ControlResponse | None:
        command = HudCommand(command_id=command_id, session_id=None, action='snapshot')
        payload = {
            'schema_version': 1, 'type': 'hud.command',
            'payload': {
                'command_id': command.command_id, 'session_id': None, 'action': 'snapshot',
                'memory_id': None, 'enabled': None,
            },
        }
        req = self._ctx.socket(zmq.REQ)
        req.setsockopt(zmq.LINGER, 0)
        req.setsockopt(zmq.RCVTIMEO, HEALTH_REQ_TIMEOUT_MS)
        try:
            req.connect(self.paths.control_endpoint)
            req.send_string(json.dumps(payload))
            raw = req.recv_string()
            return decode_control_response(raw)
        except (zmq.Again, ValueError, zmq.ZMQError):
            return None
        finally:
            req.close()

    def _supervise(self) -> int:
        while not self._shutdown.is_set():
            for role in ESSENTIAL:
                proc = self._procs.get(role)
                if proc is not None and proc.poll() is not None:
                    self.error_report = f'{role} terminato con codice {proc.returncode}'
                    return 1
            hud = self._procs.get('hud')
            if hud is not None and hud.poll() is not None:
                self._procs.pop('hud', None)  # chiusura dell'HUD non è fatale
            time.sleep(0.3)
        return 0

    # ------------------------------------------------------------------ teardown

    def _teardown(self) -> None:
        procs = [self._procs.pop(role, None) for role in reversed(START_ORDER)]
        for proc in procs:
            if proc is not None and proc.poll() is None:
                proc.terminate()
        deadline = time.monotonic() + TEARDOWN_WAIT_SEC
        for proc in procs:
            if proc is None:
                continue
            try:
                proc.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
        shutil.rmtree(self.paths.run_dir, ignore_errors=True)
        if self._lock_handle is not None:
            try:
                fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            self._lock_handle.close()
            self._lock_handle = None

    # ------------------------------------------------------------------ stato

    def _write_state(self) -> None:
        state = {
            'schema_version': 1,
            'run_dir': self.paths.run_dir,
            'space_id': self.space.space_id,
            'backend': self.space.backend,
            'endpoints': {
                'embed': self.paths.embed_endpoint,
                'events': self.paths.events_endpoint,
                'control': self.paths.control_endpoint,
            },
            'db_path': self.paths.db_path,
            'watch_dir': self.paths.watch_dir,
            'headless': self.headless,
            'pids': dict(self.child_pids),
        }
        state.update(self.extra_state)
        path = Path(self.paths.run_dir) / 'state.json'
        path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding='utf-8')

    def _install_signal_handlers(self) -> None:
        def handler(signum, frame):  # noqa: ARG001
            self.request_shutdown()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except ValueError:
                pass  # non nel thread principale: shutdown esplicito via request_shutdown


def sys_python() -> str:
    """Interprete corrente SENZA resolve(): il symlink del venv va conservato,
    altrimenti si bypassa pyvenv.cfg e il figlio perde i site-packages."""
    return sys.executable if getattr(sys, 'executable', '') else 'python3'
