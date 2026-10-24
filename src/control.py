"""CLI di controllo di Ambient Second Brain (Orchestrator, O3).

  uv run python -m src.control --runtime PATH snapshot
  uv run python -m src.control --runtime PATH pin MEMORY_ID
  uv run python -m src.control --runtime PATH unpin MEMORY_ID
  uv run python -m src.control --runtime PATH new-session
  uv run python -m src.control --runtime PATH mic on|off

Genera UUID nuovi per comando e riusa lo stesso nel retry. L'output usa gli
stessi DTO dell'HUD (control.response decodificata, stampata come JSON).
"""

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import zmq

from src.contracts import ControlResponse, HudCommand, decode_control_response

COMMAND_TIMEOUT_MS = 2000


def load_endpoints(runtime: str) -> dict:
    state_path = Path(runtime) / 'state.json'
    if not state_path.exists():
        raise SystemExit(f'ERROR: state.json assente in {runtime}: runtime non attivo?')
    state = json.loads(state_path.read_text())
    return state['endpoints']


def send_command(endpoint: str, command: HudCommand) -> ControlResponse:
    """Invia il comando; al timeout ripete una volta con lo stesso command_id."""
    payload = {
        'schema_version': 1, 'type': 'hud.command',
        'payload': {
            'command_id': command.command_id, 'session_id': command.session_id,
            'action': command.action, 'memory_id': command.memory_id, 'enabled': command.enabled,
        },
    }
    ctx = zmq.Context.instance()
    last_error = 'TIMEOUT'
    for _ in range(2):
        req = ctx.socket(zmq.REQ)
        req.setsockopt(zmq.LINGER, 0)
        req.setsockopt(zmq.RCVTIMEO, COMMAND_TIMEOUT_MS)
        try:
            req.connect(endpoint)
            req.send_string(json.dumps(payload))
            return decode_control_response(req.recv_string())
        except zmq.Again:
            last_error = 'TIMEOUT'
        except ValueError as exc:
            last_error = f'INVALID_RESPONSE: {exc}'
        finally:
            req.close()
    raise SystemExit(f'ERROR: comando non completato ({last_error})')


def current_session(endpoint: str) -> str | None:
    discovery = HudCommand(command_id=uuid.uuid4().hex, session_id=None, action='snapshot')
    response = send_command(endpoint, discovery)
    if not response.success or response.state is None:
        raise SystemExit(f'ERROR: snapshot fallito: {response.error_code} {response.error}')
    return response.state.session_id


def print_response(response: ControlResponse) -> None:
    from dataclasses import asdict

    print(json.dumps(asdict(response), indent=2, ensure_ascii=False, default=str))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Controllo runtime Ambient Second Brain')
    parser.add_argument('--runtime', required=True, help='directory runtime dell\'esecuzione')
    parser.add_argument('command', choices=['snapshot', 'pin', 'unpin', 'new-session', 'mic'])
    parser.add_argument('target', nargs='?', help='MEMORY_ID per pin/unpin, on/off per mic')
    args = parser.parse_args(argv)
    endpoints = load_endpoints(args.runtime)
    endpoint = endpoints['control']

    if args.command == 'snapshot':
        command = HudCommand(command_id=uuid.uuid4().hex, session_id=None, action='snapshot')
    elif args.command == 'new-session':
        session = current_session(endpoint)
        command = HudCommand(command_id=uuid.uuid4().hex, session_id=session, action='new_session')
    elif args.command == 'mic':
        if args.target not in ('on', 'off'):
            raise SystemExit('ERROR: mic richiede on|off')
        session = current_session(endpoint)
        command = HudCommand(command_id=uuid.uuid4().hex, session_id=session, action='set_mic',
                             enabled=args.target == 'on')
    else:
        if not args.target:
            raise SystemExit(f'ERROR: {args.command} richiede un MEMORY_ID')
        session = current_session(endpoint)
        action = 'pin' if args.command == 'pin' else 'unpin'
        command = HudCommand(command_id=uuid.uuid4().hex, session_id=session, action=action,
                             memory_id=args.target)

    started = time.time()
    response = send_command(endpoint, command)
    print_response(response)
    if not response.success:
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
