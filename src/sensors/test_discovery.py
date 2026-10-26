"""Test discovery dei log agente e specifiche multi-tail (F2, piano multi-log).

Solo filesystem finto in tmp_path con HOME iniettabile: niente ricerche reali
nell'ambiente di test. build_log_specs è pura: stesso input → stesso output.
"""

import argparse
import os
from pathlib import Path

import pytest

from src.contracts import HudCommand
from src.sensors.test_sensors import FakeClock, RunnerHarness

# --------------------------------------------------------------------------- helper


def _touch(path: Path, mtime: float) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('', encoding='utf-8')
    os.utime(path, (mtime, mtime))
    return str(path)


def _args(**kwargs) -> argparse.Namespace:
    defaults = dict(
        agent_log=None, agent_format='plain', agent_logs=None,
        auto_last=False, project_cwd=None,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# --------------------------------------------------------------------------- discovery


class TestLastClaudeSession:
    def test_most_recent_wins_by_mtime(self, tmp_path):
        from src.sensors.agent_discovery import last_claude_session

        home = tmp_path / 'home'
        base = home / '.claude' / 'projects' / '-proj-x'
        _touch(base / 'old.jsonl', 1000)
        newest = _touch(base / 'new.jsonl', 2000)
        assert last_claude_session(project_cwd='/proj/x', home=str(home)) == newest

    def test_cwd_slug_maps_slashes_to_dash(self, tmp_path):
        from src.sensors.agent_discovery import last_claude_session

        home = tmp_path / 'home'
        slug = '-home-u-proj'  # /home/u/proj → -home-u-proj
        expected = _touch(home / '.claude' / 'projects' / slug / 's.jsonl', 1000)
        assert last_claude_session(project_cwd='/home/u/proj', home=str(home)) == expected

    def test_missing_dir_or_home_returns_none(self, tmp_path):
        from src.sensors.agent_discovery import last_claude_session

        assert last_claude_session(project_cwd='/nope', home=str(tmp_path)) is None
        assert last_claude_session(project_cwd='/x', home=str(tmp_path / 'missing')) is None

    def test_ignores_non_jsonl(self, tmp_path):
        from src.sensors.agent_discovery import last_claude_session

        home = tmp_path / 'home'
        base = home / '.claude' / 'projects' / '-p'
        _touch(base / 'notes.txt', 5000)
        assert last_claude_session(project_cwd='/p', home=str(home)) is None


class TestLastCodexSession:
    def test_recursive_rollout_most_recent(self, tmp_path):
        from src.sensors.agent_discovery import last_codex_session

        home = tmp_path / 'home'
        sessions = home / '.codex' / 'sessions'
        _touch(sessions / '2026' / '10' / '24' / 'rollout-aaa.jsonl', 1000)
        newest = _touch(sessions / '2026' / '10' / '25' / 'rollout-bbb.jsonl', 3000)
        _touch(sessions / '2026' / '10' / '23' / 'rollout-ccc.jsonl', 2000)
        assert last_codex_session(home=str(home)) == newest

    def test_ignores_non_rollout_files(self, tmp_path):
        from src.sensors.agent_discovery import last_codex_session

        home = tmp_path / 'home'
        sessions = home / '.codex' / 'sessions'
        _touch(sessions / 'notes.jsonl', 9000)
        rollout = _touch(sessions / 'r' / 'rollout-x.jsonl', 100)
        assert last_codex_session(home=str(home)) == rollout

    def test_missing_dir_returns_none(self, tmp_path):
        from src.sensors.agent_discovery import last_codex_session

        assert last_codex_session(home=str(tmp_path)) is None


class TestDiscoverAutoLast:
    def test_both_tools_found_deterministic_order(self, tmp_path):
        from src.sensors.agent_discovery import discover_auto_last

        home = tmp_path / 'home'
        claude = _touch(home / '.claude' / 'projects' / '-proj' / 's.jsonl', 1000)
        codex = _touch(home / '.codex' / 'sessions' / 'rollout-c.jsonl', 2000)
        specs = discover_auto_last(('claude', 'codex'), project_cwd='/proj', home=str(home))
        assert specs == [(claude, 'claude-jsonl'), (codex, 'codex-jsonl')]
        assert discover_auto_last(('codex', 'claude'), '/proj', str(home)) == [
            (codex, 'codex-jsonl'), (claude, 'claude-jsonl')
        ]

    def test_missing_tool_yields_the_other(self, tmp_path):
        from src.sensors.agent_discovery import discover_auto_last

        home = tmp_path / 'home'
        codex = _touch(home / '.codex' / 'sessions' / 'rollout-c.jsonl', 2000)
        specs = discover_auto_last(('claude', 'codex'), '/proj', str(home))
        assert specs == [(codex, 'codex-jsonl')]

    def test_unknown_tool_ignored_and_empty_safe(self, tmp_path):
        from src.sensors.agent_discovery import discover_auto_last

        assert discover_auto_last(('cursor',), '/proj', str(tmp_path)) == []
        assert discover_auto_last(('claude', 'codex'), '/nope', str(tmp_path)) == []


# --------------------------------------------------------------------------- build_log_specs


class TestBuildLogSpecs:
    def test_explicit_agent_log_default_format(self):
        from src.sensors.runner import build_log_specs

        assert build_log_specs(_args(agent_log='/logs/s.log')) == [('/logs/s.log', 'plain')]

    def test_agent_logs_parse_path_format(self):
        from src.sensors.runner import build_log_specs

        specs = build_log_specs(_args(agent_logs=['/a.log:claude-jsonl', '/b.log:codex-jsonl']))
        assert specs == [('/a.log', 'claude-jsonl'), ('/b.log', 'codex-jsonl')]

    def test_bare_entry_uses_agent_format(self):
        from src.sensors.runner import build_log_specs

        specs = build_log_specs(_args(agent_format='claude-jsonl', agent_logs=['/a.log']))
        assert specs == [('/a.log', 'claude-jsonl')]

    def test_agent_log_then_agent_logs_keeps_cli_order(self):
        from src.sensors.runner import build_log_specs

        specs = build_log_specs(_args(
            agent_log='/first.log', agent_format='claude-jsonl',
            agent_logs=['/second.log:codex-jsonl'],
        ))
        assert specs == [('/first.log', 'claude-jsonl'), ('/second.log', 'codex-jsonl')]

    def test_invalid_format_clear_error(self):
        from src.sensors.runner import build_log_specs

        with pytest.raises(ValueError, match='rtf'):
            build_log_specs(_args(agent_logs=['/a.log:rtf']))
        with pytest.raises(ValueError, match='formato'):
            build_log_specs(_args(agent_logs=['/a.log:']))  # formato vuoto dopo ':'

    def test_dedup_realpath_first_wins(self, tmp_path):
        from src.sensors.runner import build_log_specs

        real = tmp_path / 's.log'
        real.write_text('x', encoding='utf-8')
        dup = str(tmp_path / 'sub' / '..' / 's.log')
        specs = build_log_specs(_args(agent_logs=[f'{real}:claude-jsonl', f'{dup}:codex-jsonl']))
        assert specs == [(str(real), 'claude-jsonl')]  # primo vince, anche sul formato

    def test_auto_last_adds_discovered_file(self, tmp_path):
        from src.sensors.runner import build_log_specs

        home = tmp_path / 'home'
        explicit = tmp_path / 'explicit.log'
        explicit.write_text('x', encoding='utf-8')
        auto = _touch(home / '.claude' / 'projects' / '-proj' / 'auto.jsonl', 1000)
        args = _args(agent_log=str(explicit), agent_format='plain', auto_last=True, project_cwd='/proj')
        specs = build_log_specs(args, home=str(home))
        assert specs == [(str(explicit), 'plain'), (auto, 'claude-jsonl')]

    def test_auto_last_same_realpath_dedup_explicit_wins(self, tmp_path):
        from src.sensors.runner import build_log_specs

        home = tmp_path / 'home'
        log = tmp_path / 's.jsonl'
        log.write_text('x', encoding='utf-8')
        link = home / '.claude' / 'projects' / '-proj' / 'auto.jsonl'
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(log)  # la discovery trova lo stesso file esplicito
        args = _args(agent_log=str(log), agent_format='plain', auto_last=True, project_cwd='/proj')
        assert build_log_specs(args, home=str(home)) == [(str(log), 'plain')]

    def test_auto_last_disabled_or_empty(self, tmp_path):
        from src.sensors.runner import build_log_specs

        home = tmp_path / 'home'
        _touch(home / '.claude' / 'projects' / '-proj' / 's.jsonl', 1000)
        assert build_log_specs(_args(auto_last=False), home=str(home)) == []
        assert build_log_specs(_args(auto_last=True, project_cwd='/nothing'), home=str(home)) == []


# --------------------------------------------------------------------------- multi-tail


class TestMultiTail:
    """F2: un tail per file, session_provider condiviso, poll in sequenza."""

    @staticmethod
    def _append(path: Path, text: str) -> None:
        with open(path, 'a', encoding='utf-8') as handle:
            handle.write(text)

    def _tails(self, tmp_path, clock, format_a='plain', format_b='claude-jsonl'):
        from src.sensors.agent_sensor import AgentLogTail

        log_a = tmp_path / 'a.log'
        log_b = tmp_path / 'b.jsonl'
        provider = lambda: 'sess-live'  # noqa: E731 — provider condiviso come in runner main()
        tail_a = AgentLogTail(str(log_a), format_a, clock=clock.time, session_provider=provider)
        tail_b = AgentLogTail(str(log_b), format_b, clock=clock.time, session_provider=provider)
        return log_a, log_b, tail_a, tail_b

    def test_two_files_share_session_id(self, tmp_path):
        clock = FakeClock()
        log_a, log_b, tail_a, tail_b = self._tails(tmp_path, clock)
        self._append(log_a, 'domanda dal log A\n')
        self._append(log_b, '{"type":"user","message":{"content":"dal log B"}}\n')
        clock.advance(1.1)
        got = [item for item in (tail_a.poll(), tail_b.poll()) if item is not None]
        assert len(got) == 2
        assert {item.text for item in got} == {'domanda dal log A', 'dal log B'}
        assert all(item.session_id == 'sess-live' for item in got)

    def test_independent_debounce(self, tmp_path):
        clock = FakeClock()
        log_a, log_b, tail_a, tail_b = self._tails(tmp_path, clock)
        self._append(log_a, 'solo A\n')
        clock.advance(1.1)
        emitted = tail_a.poll()
        assert emitted is not None and emitted.text == 'solo A'
        assert tail_b.poll() is None  # B non ha righe: nessuna emissione
        self._append(log_b, '{"type":"user","message":{"content":"ora B"}}\n')
        clock.advance(1.1)
        emitted_b = tail_b.poll()
        assert emitted_b is not None and emitted_b.text == 'ora B'

    def test_malformed_in_one_tail_does_not_stop_the_other(self, tmp_path):
        clock = FakeClock()
        log_a, log_b, tail_a, tail_b = self._tails(tmp_path, clock,
                                                   format_a='claude-jsonl', format_b='claude-jsonl')
        self._append(log_a, '{"type":"user", rotta\n')
        self._append(log_a, '{"type":"user","message":{"content":"A dopo la riga rotta"}}\n')
        self._append(log_b, '{"type":"user","message":{"content":"B integra"}}\n')
        clock.advance(1.1)
        emitted_b = tail_b.poll()
        emitted_a = tail_a.poll()
        assert tail_a.diagnostics.get('malformed_lines') == 1
        assert emitted_a is not None and emitted_a.text == 'A dopo la riga rotta'
        assert emitted_b is not None and emitted_b.text == 'B integra'


# --------------------------------------------------------------------------- note sensor_status


class TestAgentLogsNote:
    """F2: il conteggio tail arriva all'HUD via note_sensor e sopravvive al reset."""

    def test_note_survives_new_session(self, tmp_path):
        harness = RunnerHarness(tmp_path)
        harness.service.note_sensor('agent_logs', '2')
        status = harness.service.status_event().sensor_status
        assert status['agent_logs'] == '2'

        harness.service.handle(
            HudCommand(command_id='ns1', session_id=harness.service.session_id, action='new_session')
        )
        assert harness.service.status_event().sensor_status['agent_logs'] == '2'

    def test_mic_status_not_shadowed_by_notes(self, tmp_path):
        harness = RunnerHarness(tmp_path)
        harness.service.note_sensor('mic', 'stale')
        status = harness.service.status_event().sensor_status
        assert status['mic'] == 'on'  # le chiavi per sorgente restano calcolate
