"""Test Sensori — EMA, fusione, TTL (L1), tail agente (L2), microfono e VAD (L3).

Red/green per task del piano 05_AGENT_4_SENSORS.md; fake in fakes.py aderenti
ai contratti 1.0. Nessun hardware: i test mic usano array PCM sintetici.
"""

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

from src.config import DEFAULT_SOURCE_TTL_SEC
from src.contracts import EMBED_DIM, HudCommand, LiveInput, MemoryView, SearchSettings, make_memory_id
from src.sensors.fakes import FakeClock, FakeEmbedClient, FakeStorage, cosine, unit

# --------------------------------------------------------------------------- helper


def digest(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


def make_memory(tmp_path, name: str, text: str = 'contenuto di prova') -> MemoryView:
    path = (tmp_path / name).resolve()
    content_sha = digest(f'{name}:{text}')
    return MemoryView(
        id=make_memory_id(str(path), 'section:test:0', content_sha),
        file_path=str(path),
        modality='markdown',
        chunk_key='section:test:0',
        content_sha256=content_sha,
        source_sha256=digest(name),
        context_header=name,
        snippet_text=text[:350],
        content_text=text,
        created_at=1000.0,
    )


def text_input(text: str, session: str = 's1', at: float = 1000.0) -> LiveInput:
    return LiveInput(id=f'in-{digest(text)[:12]}', session_id=session, source='agent_text',
                     occurred_at=at, text=text)


def mic_input(path: str, duration: float = 1.0, session: str = 's1', at: float = 1000.0) -> LiveInput:
    return LiveInput(id=f'in-{digest(path)[:12]}', session_id=session, source='mic_audio',
                     occurred_at=at, audio_path=path, duration_sec=duration)


def sysaudio_input(path: str, duration: float = 1.0, session: str = 's1', at: float = 1000.0) -> LiveInput:
    return LiveInput(id=f'in-{digest(path)[:12]}', session_id=session, source='system_audio',
                     occurred_at=at, audio_path=path, duration_sec=duration)


def snapshot_cmd(command_id: str = 'cmd-snap', session_id: str | None = None) -> HudCommand:
    return HudCommand(command_id=command_id, session_id=session_id, action='snapshot')


class RunnerHarness:
    """RunnerService con fake, clock controllato e alpha=1 (EMA = ultimo vettore)."""

    def __init__(self, tmp_path, storage=None, client=None, clock=None, settings=None, **kwargs):
        import os

        from src.contracts import RuntimePaths
        from src.sensors.runner import RunnerService

        self.tmp_path = tmp_path
        run_dir = tmp_path / 'run'
        run_dir.mkdir(exist_ok=True)
        self.paths = RuntimePaths(
            run_dir=str(run_dir),
            embed_endpoint=f'ipc://{run_dir}/embed.sock',
            events_endpoint=f'ipc://{run_dir}/events.sock',
            control_endpoint=f'ipc://{run_dir}/control.sock',
            db_path=str(tmp_path / 'brain-test.db'),
            watch_dir=str(tmp_path / 'watch'),
            export_dir=str(tmp_path / 'exports'),
        )
        os.makedirs(self.paths.watch_dir, exist_ok=True)
        os.makedirs(self.paths.export_dir, exist_ok=True)
        self.storage = storage if storage is not None else FakeStorage()
        self.client = client if client is not None else FakeEmbedClient()
        self.clock = clock if clock is not None else FakeClock()
        self.settings = settings if settings is not None else SearchSettings()
        self.service = RunnerService(
            self.storage, self.client, self.paths, self.settings,
            clock=self.clock.time, alpha=1.0, **kwargs,
        )

    def ingest_text(self, text: str, **kwargs):
        return self.service.ingest(text_input(text, session=self.service.session_id, **kwargs))

    def ingest_mic(self, name: str, duration: float = 1.0, **kwargs):
        return self.service.ingest(
            mic_input(self.mic_path(name), duration=duration, session=self.service.session_id, **kwargs)
        )

    def ingest_sysaudio(self, name: str, duration: float = 1.0, **kwargs):
        return self.service.ingest(
            sysaudio_input(self.mic_path(name), duration=duration, session=self.service.session_id, **kwargs)
        )

    def mic_path(self, name: str) -> str:
        return str((self.tmp_path / name).resolve())


# --------------------------------------------------------------------------- L1: EMA


class TestEMATracker:
    def test_first_vector_and_update(self):
        from src.sensors.ema_tracker import EMATracker

        a = [1.0] + [0.0] * 255
        b = [0.0, 1.0] + [0.0] * 254
        ema = EMATracker(alpha=0.35)
        assert ema.update(a) == a
        expected = np.array(a) * 0.65 + np.array(b) * 0.35
        expected /= np.linalg.norm(expected)
        assert np.allclose(ema.update(b), expected)

    @pytest.mark.parametrize('alpha', [0.0, -0.1, 1.0001, 2.0, float('nan'), float('inf')])
    def test_invalid_alpha_rejected(self, alpha):
        from src.sensors.ema_tracker import EMATracker

        with pytest.raises(ValueError):
            EMATracker(alpha=alpha)

    def test_alpha_bounds_accepted(self):
        from src.sensors.ema_tracker import EMATracker

        small = EMATracker(alpha=0.01)
        one = EMATracker(alpha=1.0)
        vec = unit([1.0, 2.0] + [0.5] * (EMBED_DIM - 2))
        assert small.update(vec) == vec  # primo vettore copiato per qualunque alpha
        assert one.update(vec) == vec

    def test_alpha_one_tracks_latest(self):
        from src.sensors.ema_tracker import EMATracker

        ema = EMATracker(alpha=1.0)
        first = unit([1.0] + [0.0] * 255)
        second = unit([0.0, 1.0] + [0.0] * 254)
        ema.update(first)
        assert ema.update(second) == second

    def test_zero_norm_rejected(self):
        from src.sensors.ema_tracker import EMATracker

        ema = EMATracker()
        with pytest.raises(ValueError, match='ZERO_NORM'):
            ema.update([0.0] * EMBED_DIM)
        # e lo stato resta assente: il prossimo update valido è un primo vettore
        vec = unit([3.0, 4.0] + [0.0] * 254)
        assert ema.update(vec) == vec

    def test_degenerate_update_rejected(self):
        from src.sensors.ema_tracker import EMATracker

        ema = EMATracker(alpha=0.35)
        a = unit([1.0] + [0.0] * 255)
        ema.update(a)
        # 0.65*a + 0.35*(-0.65/0.35 * a) = 0: norma nulla sull'aggiornamento
        degenerate = list(np.asarray([-x for x in a]) * (0.65 / 0.35))
        with pytest.raises(ValueError, match='ZERO_NORM'):
            ema.update(degenerate)

    @pytest.mark.parametrize('bad', [
        [float('nan')] + [0.0] * 255,
        [float('inf')] + [0.0] * 255,
        [0.0] * 254 + [float('nan')],
    ])
    def test_non_finite_rejected(self, bad):
        from src.sensors.ema_tracker import EMATracker

        with pytest.raises(ValueError):
            EMATracker().update(bad)

    def test_wrong_dimension_rejected(self):
        from src.sensors.ema_tracker import EMATracker

        with pytest.raises(ValueError, match='DIMENSION'):
            EMATracker().update([1.0] * 255)
        with pytest.raises(ValueError, match='DIMENSION'):
            EMATracker().update([1.0] * 257)

    def test_non_numeric_rejected(self):
        from src.sensors.ema_tracker import EMATracker

        with pytest.raises(ValueError):
            EMATracker().update(['x'] + [0.0] * 255)
        with pytest.raises(ValueError):
            EMATracker().update([True] + [0.0] * 255)

    def test_reset_clears_state(self):
        from src.sensors.ema_tracker import EMATracker

        ema = EMATracker(alpha=0.35)
        ema.update(unit([1.0] + [0.0] * 255))
        ema.reset()
        assert not ema.has_state
        vec = unit([0.0, 1.0] + [0.0] * 254)
        assert ema.update(vec) == vec  # di nuovo primo vettore copiato

    def test_reset_without_state_is_noop(self):
        from src.sensors.ema_tracker import EMATracker

        EMATracker().reset()

    def test_input_not_mutated(self):
        from src.sensors.ema_tracker import EMATracker

        ema = EMATracker(alpha=1.0)  # update(x) == x: ogni corruzione di stato è visibile
        arr = np.asarray(unit([1.0] + [0.0] * 255))
        snapshot = arr.copy()
        first = ema.update(arr.tolist())
        arr[0] = 99.0  # il chiamante modifica il suo array dopo l'update
        second_vec = unit([0.0, 1.0] + [0.0] * 254)
        ema.update(second_vec)
        assert first == snapshot.tolist()  # la prima risposta non cambia
        # lo stato interno non è stato corrotto dalla mutazione del chiamante
        assert ema.update(second_vec) == second_vec

    def test_update_normalizes(self):
        from src.sensors.ema_tracker import EMATracker

        ema = EMATracker()
        out = ema.update([2.0] + [0.0] * 255)  # non normalizzato in ingresso
        assert abs(np.linalg.norm(out) - 1.0) < 1e-12


# --------------------------------------------------------------------------- L1: fusione e TTL


class TestFusion:
    def test_dedup_winner_scores_and_matched_sources(self, tmp_path):
        harness = RunnerHarness(tmp_path)
        mem = make_memory(tmp_path, 'f01.md')
        text_vec = unit([1.0] + [0.0] * 255)
        mic_vec = unit([0.6, 0.8] + [0.0] * 254)
        harness.storage.add(mem, text_vec)
        harness.client.vectors['parliamo di cache'] = text_vec
        harness.client.vectors[harness.mic_path('voce-cache.wav')] = mic_vec

        harness.ingest_text('parliamo di cache')
        event = harness.ingest_mic('voce-cache.wav')

        merged = event.results[0]
        assert merged.memory.id == mem.id
        assert merged.matched_sources == ['agent_text', 'mic_audio']
        assert merged.winning_source == 'agent_text'  # punteggio più alto
        assert merged.cosine_score == pytest.approx(1.0)  # punteggi della sorgente vincente

    def test_tie_prefers_agent_text(self, tmp_path):
        harness = RunnerHarness(tmp_path)
        mem = make_memory(tmp_path, 'f02.md')
        same = unit([1.0] + [0.0] * 255)
        harness.storage.add(mem, same)
        harness.client.vectors['testo uguale'] = same
        harness.client.vectors[harness.mic_path('stesso.wav')] = same

        harness.ingest_text('testo uguale')
        event = harness.ingest_mic('stesso.wav')
        merged = event.results[0]
        assert merged.winning_source == 'agent_text'
        assert merged.matched_sources == ['agent_text', 'mic_audio']

    def test_mic_wins_when_higher_score(self, tmp_path):
        harness = RunnerHarness(tmp_path)
        mem = make_memory(tmp_path, 'f03.md')
        text_vec = unit([1.0, 0.0] + [0.0] * 254)
        mem_vec = unit([0.0, 1.0] + [0.0] * 254)
        mic_vec = unit([math.sqrt(0.5), math.sqrt(0.5)] + [0.0] * 254)
        harness.storage.add(mem, mem_vec)
        harness.client.vectors['generico'] = text_vec
        harness.client.vectors[harness.mic_path('voce.wav')] = mic_vec

        harness.ingest_text('generico')
        event = harness.ingest_mic('voce.wav')
        merged = event.results[0]
        assert merged.winning_source == 'mic_audio'
        assert abs(merged.cosine_score - cosine(mic_vec, mem_vec)) < 1e-9
        assert merged.matched_sources == ['agent_text', 'mic_audio']

    def test_thresholds_ordering_and_topk(self, tmp_path):
        settings = SearchSettings(top_k=2, min_cosine=0.25, min_final=1.0)
        storage = FakeStorage()
        harness = RunnerHarness(tmp_path, storage=storage, settings=settings)
        mem_a = make_memory(tmp_path, 'a.md')
        mem_b = make_memory(tmp_path, 'b.md')
        mem_c = make_memory(tmp_path, 'c.md')
        mem_d = make_memory(tmp_path, 'd.md')  # sotto soglia: escluso
        query = unit([1.0] + [0.0] * 255)
        harness.client.vectors['query'] = query
        # fake: final = 2*cos + boost
        harness.storage.add(mem_a, unit([0.9, 0.1] + [0.0] * 254))
        harness.storage.add(mem_b, unit([0.5, 0.5] + [0.0] * 254))
        harness.storage.add(mem_c, unit([0.99, 0.01] + [0.0] * 254))
        harness.storage.add(mem_d, unit([0.0, 1.0] + [0.0] * 254))

        event = harness.ingest_text('query')
        ids = [item.memory.id for item in event.results]
        assert len(event.results) == 2  # top_k dopo soglie
        assert ids[0] == mem_c.id  # final più alto
        assert ids[1] == mem_a.id
        assert mem_d.id not in ids  # cos ~0 < min_cosine e senza boost

    def test_boost_admits_low_cosine(self, tmp_path):
        harness = RunnerHarness(tmp_path)
        mem = make_memory(tmp_path, 'graph.md')
        query = unit([1.0] + [0.0] * 255)
        harness.client.vectors['target del grafo'] = query
        harness.storage.add(mem, unit([0.0, 1.0] + [0.0] * 254))  # cos ~0
        # fake final = 2*cos + boost: con boost 0.9 final < min_final → escluso
        harness.storage.boost[mem.id] = 0.9
        event = harness.ingest_text('target del grafo')
        assert all(item.memory.id != mem.id for item in event.results)

        # boost 1.0 → final = 1.0 >= min_final: ammesso nonostante cos < min_cosine
        harness.storage.boost[mem.id] = 1.0
        event = harness.service.ingest(
            text_input('target del grafo', session=harness.service.session_id, at=1001.0)
        )
        admitted = [item for item in event.results if item.memory.id == mem.id]
        assert len(admitted) == 1
        assert admitted[0].synaptic_boost == 1.0

    def test_show_all_skips_thresholds_marks_debug(self, tmp_path):
        storage = FakeStorage()
        harness = RunnerHarness(tmp_path, storage=storage, show_all=True)
        mem = make_memory(tmp_path, 'low.md')
        query = unit([1.0] + [0.0] * 255)
        harness.client.vectors['troppo lontano'] = query
        harness.storage.add(mem, unit([0.0, 1.0] + [0.0] * 254))  # cos ~0, boost 0

        event = harness.ingest_text('troppo lontano')
        assert [item.memory.id for item in event.results] == [mem.id]
        assert event.sensor_status.get('show_all') == 'DEBUG'

    def test_per_source_search_topk_x4_and_separate_ema(self, tmp_path):
        storage = FakeStorage()
        settings = SearchSettings(top_k=3)
        harness = RunnerHarness(tmp_path, storage=storage, settings=settings)
        text_vec = unit([1.0] + [0.0] * 255)
        mic_vec = unit([0.0, 1.0] + [0.0] * 254)
        harness.client.vectors['testo A'] = text_vec
        harness.client.vectors[harness.mic_path('voce B.wav')] = mic_vec

        harness.ingest_text('testo A')
        harness.ingest_mic('voce B.wav')

        assert len(storage.search_calls) >= 3  # 1 dopo testo, 2 dopo mic (entrambe le sorgenti)
        for _, call_settings in storage.search_calls:
            assert call_settings.top_k == settings.top_k * 4
        # la ricerca dell'altra sorgente usa la sua EMA: il mic non altera il testo
        assert tuple(storage.search_calls[0][0]) == tuple(text_vec)
        later_vecs = [tuple(vec) for vec, _ in storage.search_calls[1:]]
        assert tuple(text_vec) in later_vecs
        assert tuple(mic_vec) in later_vecs

    def test_failed_embedding_keeps_history_updates_diagnostics_only(self, tmp_path):
        harness = RunnerHarness(tmp_path)
        mem = make_memory(tmp_path, 'f04.md')
        harness.storage.add(mem, unit([1.0] + [0.0] * 255))
        harness.client.fail_with = TimeoutError('TIMEOUT')

        event = harness.ingest_text('input con modello giù')
        assert event.results == []  # nessun ranking nuovo
        assert event.sensor_status.get('agent_text_error') == 'TIMEOUT'

        # la storia conserva l'input anche se il modello è fallito (esportabile)
        response = harness.service.handle(snapshot_cmd())
        assert any(item.text == 'input con modello giù' for item in response.session_inputs)

        # EMA/last_seen non aggiornati: nessuna sorgente attiva
        harness.client.fail_with = None
        heartbeat = harness.service.status_event()
        assert heartbeat.results == []

    def test_ttl_expiry_recomputed_on_heartbeat(self, tmp_path):
        harness = RunnerHarness(tmp_path)
        mem = make_memory(tmp_path, 'f05.md')
        query = unit([1.0] + [0.0] * 255)
        harness.client.vectors['argomento'] = query
        harness.storage.add(mem, query)

        event = harness.ingest_text('argomento')
        assert len(event.results) == 1
        assert event.source_last_seen.get('agent_text') == pytest.approx(harness.clock.time())

        harness.clock.advance(DEFAULT_SOURCE_TTL_SEC - 1)
        still = harness.service.status_event()
        assert len(still.results) == 1  # entro TTL

        harness.clock.advance(2)  # oltre 120 s
        expired = harness.service.status_event()
        assert expired.results == []  # ricalcolato su heartbeat senza nuovo input

    def test_no_active_sources_keeps_tray(self, tmp_path):
        harness = RunnerHarness(tmp_path, max_pins=4)
        mem = make_memory(tmp_path, 'f06.md')
        query = unit([1.0] + [0.0] * 255)
        harness.client.vectors['pinnami'] = query
        harness.storage.add(mem, query)
        harness.ingest_text('pinnami')

        cmd = HudCommand(command_id='cmd-pin-1', session_id=harness.service.session_id,
                         action='pin', memory_id=mem.id)
        response = harness.service.handle(cmd)
        assert response.success

        harness.clock.advance(DEFAULT_SOURCE_TTL_SEC + 5)
        status = harness.service.status_event()
        assert status.results == []  # nessuna sorgente attiva
        assert len(status.pinned) == 1  # tray invariato


# --------------------------------------------------------------------------- L2: sensore agente


class TestThreeSources:
    """F1: system_audio come terza sorgente audio (query-only)."""

    def test_tie_three_sources_prefers_agent_text(self, tmp_path):
        harness = RunnerHarness(tmp_path)
        mem = make_memory(tmp_path, 'f07.md')
        same = unit([1.0] + [0.0] * 255)
        harness.storage.add(mem, same)
        harness.client.vectors['tripla'] = same
        harness.client.vectors[harness.mic_path('mic-t.wav')] = same
        harness.client.vectors[harness.mic_path('sys-t.wav')] = same

        harness.ingest_text('tripla')
        harness.ingest_mic('mic-t.wav')
        event = harness.ingest_sysaudio('sys-t.wav')
        merged = event.results[0]
        assert merged.winning_source == 'agent_text'
        # ordine canonico nei pareggi (SOURCE_ORDER): agent → mic → system
        assert merged.matched_sources == ['agent_text', 'mic_audio', 'system_audio']

    def test_system_audio_wins_when_higher_score(self, tmp_path):
        harness = RunnerHarness(tmp_path)
        mem = make_memory(tmp_path, 'f08.md')
        text_vec = unit([1.0, 0.0] + [0.0] * 254)
        mem_vec = unit([0.0, 1.0] + [0.0] * 254)
        sys_vec = unit([math.sqrt(0.5), math.sqrt(0.5)] + [0.0] * 254)
        harness.storage.add(mem, mem_vec)
        harness.client.vectors['generico sys'] = text_vec
        harness.client.vectors[harness.mic_path('sys-win.wav')] = sys_vec

        harness.ingest_text('generico sys')
        event = harness.ingest_sysaudio('sys-win.wav')
        merged = event.results[0]
        assert merged.winning_source == 'system_audio'
        assert merged.matched_sources == ['agent_text', 'system_audio']

    def test_ema_isolated_across_three_sources(self, tmp_path):
        storage = FakeStorage()
        harness = RunnerHarness(tmp_path, storage=storage)
        text_vec = unit([1.0] + [0.0] * 255)
        mic_vec = unit([0.0, 1.0] + [0.0] * 254)
        sys_vec = unit([0.0, 0.0, 1.0] + [0.0] * 253)
        harness.client.vectors['testo tre'] = text_vec
        harness.client.vectors[harness.mic_path('mic-tre.wav')] = mic_vec
        harness.client.vectors[harness.mic_path('sys-tre.wav')] = sys_vec

        harness.ingest_text('testo tre')
        harness.ingest_mic('mic-tre.wav')
        harness.ingest_sysaudio('sys-tre.wav')

        assert set(harness.service._last_seen) == {'agent_text', 'mic_audio', 'system_audio'}
        # l'ultimo giro di ricerca interroga una EMA per sorgente, senza contamini
        later = [tuple(vec) for vec, _ in storage.search_calls[-3:]]
        assert tuple(text_vec) in later
        assert tuple(mic_vec) in later
        assert tuple(sys_vec) in later


class TestEmbedRouting:
    """F1: _embed passa audio_path per OGNI sorgente audio (via FakeEmbedClient.requests)."""

    def test_mic_routes_audio_path(self, tmp_path):
        harness = RunnerHarness(tmp_path)
        harness.ingest_mic('voce-r.wav')
        req = harness.client.requests[-1]
        assert req.audio_path == harness.mic_path('voce-r.wav')
        assert req.text_payload == ''
        assert req.is_query is True

    def test_system_audio_routes_audio_path(self, tmp_path):
        harness = RunnerHarness(tmp_path)
        harness.ingest_sysaudio('sys-r.wav')
        req = harness.client.requests[-1]
        assert req.audio_path == harness.mic_path('sys-r.wav')
        assert req.text_payload == ''
        assert req.is_query is True

    def test_agent_text_routes_no_audio(self, tmp_path):
        harness = RunnerHarness(tmp_path)
        harness.ingest_text('solo testo')
        req = harness.client.requests[-1]
        assert req.audio_path is None
        assert req.text_payload == 'solo testo'

    @pytest.mark.parametrize('source', ['mic_audio', 'system_audio'])
    def test_audio_source_without_path_rejected(self, tmp_path, source):
        harness = RunnerHarness(tmp_path)
        item = LiveInput(id='in-x', session_id=harness.service.session_id,
                         source=source, occurred_at=1000.0)
        with pytest.raises(ValueError, match=f'{source} senza audio_path'):
            harness.service.ingest(item)

    @pytest.mark.parametrize('source', ['mic_audio', 'system_audio'])
    def test_audio_source_relative_path_rejected(self, tmp_path, source):
        harness = RunnerHarness(tmp_path)
        item = LiveInput(id='in-y', session_id=harness.service.session_id,
                         source=source, occurred_at=1000.0, audio_path='rel.wav')
        with pytest.raises(ValueError, match='audio_path relativo'):
            harness.service.ingest(item)


class TestParseAgentLine:
    def test_plain_strips_ansi_csi(self):
        from src.sensors.agent_sensor import parse_agent_line

        assert parse_agent_line('\x1b[31mrosso\x1b[0m piano', 'plain') == 'rosso piano'

    def test_plain_strips_osc(self):
        from src.sensors.agent_sensor import parse_agent_line

        assert parse_agent_line('\x1b]0;titolo\x07contenuto', 'plain') == 'contenuto'

    def test_plain_handles_backspace(self):
        from src.sensors.agent_sensor import parse_agent_line

        assert parse_agent_line('abc\x08\x08x', 'plain') == 'ax'

    def test_plain_keeps_last_repaint(self):
        from src.sensors.agent_sensor import parse_agent_line

        assert parse_agent_line('10%\r50%\rfine', 'plain') == 'fine'
        assert parse_agent_line('testo\r', 'plain') == 'testo'

    def test_plain_empty_is_none(self):
        from src.sensors.agent_sensor import parse_agent_line

        assert parse_agent_line('', 'plain') is None
        assert parse_agent_line('   \x1b[2K\n', 'plain') is None

    def test_unknown_format_rejected(self):
        from src.sensors.agent_sensor import parse_agent_line

        with pytest.raises(ValueError):
            parse_agent_line('x', 'rtf')

    def test_claude_user_string_content(self):
        from src.sensors.agent_sensor import parse_agent_line

        line = '{"type":"user","message":{"role":"user","content":"come ottimizzo la cache?"}}'
        assert parse_agent_line(line, 'claude-jsonl') == 'come ottimizzo la cache?'

    def test_claude_assistant_text_blocks(self):
        from src.sensors.agent_sensor import parse_agent_line

        line = ('{"type":"assistant","message":{"content":['
                '{"type":"text","text":"prima parte."},'
                '{"type":"text","text":"seconda parte."}]}}')
        assert parse_agent_line(line, 'claude-jsonl') == 'prima parte.\nseconda parte.'

    def test_claude_ignores_thinking_and_tool_blocks(self):
        from src.sensors.agent_sensor import parse_agent_line

        line = ('{"type":"assistant","message":{"content":['
                '{"type":"thinking","thinking":"ragionamento interno"},'
                '{"type":"tool_use","name":"Bash"},'
                '{"type":"text","text":"risposta visibile"}]}}')
        assert parse_agent_line(line, 'claude-jsonl') == 'risposta visibile'

    def test_claude_ignores_other_types(self):
        from src.sensors.agent_sensor import parse_agent_line

        assert parse_agent_line('{"type":"system","content":"boot"}', 'claude-jsonl') is None
        assert parse_agent_line('{"type":"summary","summary":"x"}', 'claude-jsonl') is None
        assert parse_agent_line(
            '{"type":"user","message":{"content":[{"type":"tool_result","content":"out"}]}}',
            'claude-jsonl',
        ) is None

    def test_claude_non_dict_is_none(self):
        from src.sensors.agent_sensor import parse_agent_line

        assert parse_agent_line('[1, 2, 3]', 'claude-jsonl') is None
        assert parse_agent_line('"solo stringa"', 'claude-jsonl') is None

    def test_claude_malformed_complete_line_raises(self):
        from src.sensors.agent_sensor import parse_agent_line

        with pytest.raises(ValueError, match='MALFORMED'):
            parse_agent_line('{"type":"user", broken', 'claude-jsonl')


class TestAgentLogTail:
    def _write(self, path: Path, text: str) -> None:
        with open(path, 'a', encoding='utf-8') as handle:
            handle.write(text)

    def _tail(self, path: Path, clock, **kwargs):
        from src.sensors.agent_sensor import AgentLogTail

        return AgentLogTail(str(path), 'plain', clock=clock.time, **kwargs)

    def test_initial_start_reads_from_end(self, tmp_path):
        log = tmp_path / 'session.log'
        self._write(log, 'conversazione storica\n')
        clock = FakeClock()
        tail = self._tail(log, clock)
        assert tail.poll() is None  # lo storico non diventa query live
        self._write(log, 'nuovo messaggio\n')
        clock.advance(1.1)
        live = tail.poll()
        assert live is not None and live.text == 'nuovo messaggio'
        assert live.source == 'agent_text'

    def test_replay_mode_reads_from_start(self, tmp_path):
        log = tmp_path / 'fixture.log'
        self._write(log, 'passato\n')
        clock = FakeClock()
        tail = self._tail(log, clock, start='beginning')
        clock.advance(1.1)
        live = tail.poll()
        assert live is not None and live.text == 'passato'

    def test_partial_line_stays_in_buffer(self, tmp_path):
        log = tmp_path / 'session.log'
        clock = FakeClock()
        tail = self._tail(log, clock)
        self._write(log, 'mezza ri')
        assert tail.poll() is None
        self._write(log, 'ga intera\n')
        clock.advance(1.1)
        live = tail.poll()
        assert live is not None and live.text == 'mezza riga intera'

    def test_debounce_batches_lines(self, tmp_path):
        log = tmp_path / 'session.log'
        clock = FakeClock()
        tail = self._tail(log, clock)
        self._write(log, 'una\n')
        self._write(log, 'due\n')
        assert tail.poll() is None  # debounce non scaduto
        clock.advance(1.1)
        live = tail.poll()
        assert live is not None and live.text == 'una\ndue'

    def test_null_and_banner_block_is_discarded(self, tmp_path):
        """Regressione 2026-10-24: un blocco PTY di NUL/banner non è una query."""
        log = tmp_path / 'session.log'
        clock = FakeClock()
        tail = self._tail(log, clock)
        junk = 'bob\x1b[K\x08\x08\x1b[K\x07' + '\x00' * 409 + '\n'
        self._write(log, junk)
        clock.advance(1.1)
        assert tail.poll() is None  # blocco senza contenuto reale: scartato
        self._write(log, 'Quanto durano i risultati nella cache?\n')
        clock.advance(1.1)
        live = tail.poll()
        assert live is not None and 'Quanto durano' in live.text

    def test_stable_block_not_resent(self, tmp_path):
        log = tmp_path / 'session.log'
        clock = FakeClock()
        tail = self._tail(log, clock)
        self._write(log, 'ciao\n')
        clock.advance(1.1)
        assert tail.poll() is not None
        # nessuna riga nuova: nessuna riemissione
        clock.advance(2.0)
        assert tail.poll() is None
        # la stessa riga riappended produce lo stesso blocco: non rinviata
        self._write(log, 'ciao\n')
        clock.advance(2.0)
        assert tail.poll() is None

    def test_ansi_and_repaint_cleaned_in_stream(self, tmp_path):
        log = tmp_path / 'session.log'
        clock = FakeClock()
        tail = self._tail(log, clock)
        self._write(log, '\x1b[31m$\x1b[0m \x1b]0;t\x07prompt 10%\r50%\rfatto\n')
        clock.advance(1.1)
        live = tail.poll()
        assert live is not None and live.text == 'fatto'  # ultimo repaint

    def test_truncation_reopens_from_zero(self, tmp_path):
        log = tmp_path / 'session.log'
        self._write(log, 'una\ndue\ntre\n')
        clock = FakeClock()
        tail = self._tail(log, clock)
        # troncamento: il file viene riscritto da capo
        log.write_text('reset\n', encoding='utf-8')
        clock.advance(1.1)
        live = tail.poll()
        assert live is not None and live.text == 'reset'

    def test_rotation_reopens_new_file(self, tmp_path):
        log = tmp_path / 'session.log'
        self._write(log, 'vecchio\n')
        clock = FakeClock()
        tail = self._tail(log, clock)
        # rotazione: nuovo inode con lo stesso percorso
        rotated = tmp_path / 'session.log.1'
        log.rename(rotated)
        self._write(log, 'nuovo file\n')
        clock.advance(1.1)
        live = tail.poll()
        assert live is not None and live.text == 'nuovo file'

    def test_session_id_untouched_across_rotation(self, tmp_path):
        log = tmp_path / 'session.log'
        self._write(log, 'prima\n')
        clock = FakeClock()
        tail = self._tail(log, clock, session_id='sess-1')
        log.rename(tmp_path / 'old.log')
        self._write(log, 'dopo rotazione\n')
        clock.advance(1.1)
        live = tail.poll()
        assert live is not None and live.session_id == 'sess-1'

    def test_max_chars_tail_kept(self, tmp_path):
        log = tmp_path / 'session.log'
        clock = FakeClock()
        tail = self._tail(log, clock)
        self._write(log, 'x' * 500 + '\n' + 'y' * 500 + '\n')
        clock.advance(1.1)
        live = tail.poll()
        assert live is not None
        assert len(live.text) == 800
        assert live.text.startswith('x' * 299)  # ultime 800 caratteri utili

    def test_jsonl_malformed_signaled_and_skipped(self, tmp_path):
        from src.sensors.agent_sensor import AgentLogTail

        log = tmp_path / 'session.jsonl'
        clock = FakeClock()
        tail = AgentLogTail(str(log), 'claude-jsonl', clock=clock.time)
        good1 = '{"type":"user","message":{"content":"prima"}}\n'
        bad = '{"type":"user", rotta\n'
        good2 = '{"type":"assistant","message":{"content":"seconda"}}\n'
        self._write(log, good1 + bad + good2)
        clock.advance(1.1)
        live = tail.poll()
        assert live is not None
        assert live.text == 'prima\nseconda'
        assert tail.diagnostics.get('malformed_lines') == 1

    def test_jsonl_ignored_lines_do_not_emit(self, tmp_path):
        from src.sensors.agent_sensor import AgentLogTail

        log = tmp_path / 'session.jsonl'
        clock = FakeClock()
        tail = AgentLogTail(str(log), 'claude-jsonl', clock=clock.time)
        self._write(log, '{"type":"system","content":"x"}\n')
        clock.advance(5.0)
        assert tail.poll() is None


# --------------------------------------------------------------------------- L3: microfono


def pcm_blocks(duration_sec: float, amplitude: int) -> list['np.ndarray']:
    """Blocchi PCM int16 da 30 ms (480 campioni) a ampiezza costante."""
    from src.sensors.mic_sensor import BLOCK_SAMPLES

    total = int(round(duration_sec / 0.03))
    blocks = []
    for _ in range(total):
        blocks.append(np.full(BLOCK_SAMPLES, amplitude, dtype=np.int16))
    return blocks


class TestVadSegmenter:
    def _feed(self, segmenter, blocks):
        out = []
        for block in blocks:
            out.extend(segmenter.feed(block))
        return out

    def test_calibration_from_silence(self):
        from src.sensors.mic_sensor import VadSegmenter

        seg = VadSegmenter()
        assert seg.threshold is None
        self._feed(seg, pcm_blocks(1.0, 0))  # un secondo di calibrazione
        assert seg.threshold == 0.008  # max(0.008, 3*~0)

    def test_silence_never_produces_segments(self):
        from src.sensors.mic_sensor import VadSegmenter

        seg = VadSegmenter()
        out = self._feed(seg, pcm_blocks(5.0, 0))
        assert out == []
        assert seg.flush() == []

    def test_short_speech_dropped(self):
        from src.sensors.mic_sensor import VadSegmenter

        seg = VadSegmenter()
        out = self._feed(seg, pcm_blocks(1.0, 0) + pcm_blocks(0.3, 3000) + pcm_blocks(0.7, 0))
        assert out == []  # 300 ms di parlato < 600 ms: eliminato

    def test_sentence_then_silence_emits(self):
        from src.sensors.mic_sensor import VadSegmenter

        seg = VadSegmenter()
        out = self._feed(seg, pcm_blocks(1.0, 0) + pcm_blocks(1.5, 3000) + pcm_blocks(0.7, 0))
        assert len(out) == 1
        assert out[0].speech_sec == pytest.approx(1.5, abs=0.031)
        assert out[0].duration_sec == pytest.approx(2.1, abs=0.031)

    def test_flush_emits_pending_speech(self):
        from src.sensors.mic_sensor import VadSegmenter

        seg = VadSegmenter()
        out = self._feed(seg, pcm_blocks(1.0, 0) + pcm_blocks(1.0, 3000))
        assert out == []  # nessuna pausa: ancora aperto
        flushed = seg.flush()
        assert len(flushed) == 1
        assert flushed[0].speech_sec == pytest.approx(1.0, abs=0.031)

    def test_continuous_speech_forces_emission_with_overlap(self):
        from src.sensors.mic_sensor import MAX_SEGMENT_SEC, VadSegmenter

        seg = VadSegmenter()
        out = self._feed(seg, pcm_blocks(1.0, 0) + pcm_blocks(13.0, 3000))
        assert len(out) == 1  # primo segmento forzato a 12 s
        assert out[0].duration_sec == pytest.approx(MAX_SEGMENT_SEC, abs=0.031)
        flushed = seg.flush()
        assert len(flushed) == 1  # la coda continua nel secondo segmento
        overlap_n = int(round(0.3 / 0.03)) * 480
        np.testing.assert_array_equal(flushed[0].samples[:overlap_n], out[0].samples[-overlap_n:])

    def test_noise_floor_updates_only_on_non_vocal(self):
        from src.sensors.mic_sensor import VadSegmenter

        seg = VadSegmenter()
        self._feed(seg, pcm_blocks(1.0, 300))  # calibrazione con rumore ampio
        first_threshold = seg.threshold
        assert first_threshold == pytest.approx(3 * 300 / 32768, abs=1e-4)

        self._feed(seg, pcm_blocks(2.0, 100))  # rumore più basso, frame non vocali
        assert seg.threshold < first_threshold  # il pavimento si adatta

        lowered = seg.threshold
        self._feed(seg, pcm_blocks(0.2, 8000))  # parlato: pavimento intoccato
        assert seg.threshold == lowered


class TestMicSensor:
    def _sensor(self, tmp_path, clock, **kwargs):
        from src.sensors.mic_sensor import MicSensor

        return MicSensor(
            audio_root=str(tmp_path / 'runtime'),
            session_provider=lambda: 'sess-live',
            clock=clock.time,
            **kwargs,
        )

    def test_wav_unique_closed_and_correct(self, tmp_path):
        clock = FakeClock()
        sensor = self._sensor(tmp_path, clock)
        inputs = []
        for block in pcm_blocks(1.0, 0) + pcm_blocks(1.5, 3000) + pcm_blocks(0.7, 0):
            inputs.extend(sensor.feed_block(block))
        assert len(inputs) == 1
        item = inputs[0]
        assert item.source == 'mic_audio'
        assert item.session_id == 'sess-live'
        assert item.duration_sec == pytest.approx(2.1, abs=0.031)
        import wave

        session_dir = tmp_path / 'runtime' / 'sess-live'
        assert item.audio_path.startswith(str(session_dir))
        with wave.open(item.audio_path, 'rb') as handle:
            assert handle.getnchannels() == 1
            assert handle.getframerate() == 16000
            assert handle.getsampwidth() == 2
            assert handle.getnframes() == int(item.duration_sec * 16000)

    def test_segments_never_overwrite(self, tmp_path):
        clock = FakeClock()
        sensor = self._sensor(tmp_path, clock)
        inputs = []
        blocks = (
            pcm_blocks(1.0, 0) + pcm_blocks(1.5, 3000) + pcm_blocks(0.7, 0)
            + pcm_blocks(1.5, 3000) + pcm_blocks(0.7, 0)
        )
        for block in blocks:
            inputs.extend(sensor.feed_block(block))
        paths = [item.audio_path for item in inputs]
        assert len(paths) == 2
        assert len(set(paths)) == 2  # WAV univoci, nessuna sovrascrittura
        for path in paths:
            assert Path(path).exists()

    def test_pause_flush_closes_pending(self, tmp_path):
        clock = FakeClock()
        sensor = self._sensor(tmp_path, clock)
        for block in pcm_blocks(1.0, 0) + pcm_blocks(1.0, 3000):
            sensor.feed_block(block)
        flushed = sensor.flush()  # pausa: flush controllato
        assert len(flushed) == 1
        assert Path(flushed[0].audio_path).exists()

    def test_calibration_blocks_do_not_create_segments(self, tmp_path):
        clock = FakeClock()
        sensor = self._sensor(tmp_path, clock)
        inputs = []
        for block in pcm_blocks(5.0, 0):
            inputs.extend(sensor.feed_block(block))
        assert inputs == []

    def test_system_audio_source_and_prefix(self, tmp_path):
        clock = FakeClock()
        sensor = self._sensor(tmp_path, clock, source='system_audio', file_prefix='sysaudio')
        inputs = []
        for block in pcm_blocks(1.0, 0) + pcm_blocks(1.5, 3000) + pcm_blocks(0.7, 0):
            inputs.extend(sensor.feed_block(block))
        assert len(inputs) == 1
        assert inputs[0].source == 'system_audio'
        name = Path(inputs[0].audio_path).name
        assert name.startswith('sysaudio-') and name.endswith('.wav')
        assert str(tmp_path / 'runtime' / 'sess-live') in inputs[0].audio_path

    def test_default_params_unchanged(self, tmp_path):
        """Regressione F1: i default restano mic_audio/'mic-' (replay e runner mic)."""
        clock = FakeClock()
        sensor = self._sensor(tmp_path, clock)
        inputs = []
        for block in pcm_blocks(1.0, 0) + pcm_blocks(1.5, 3000) + pcm_blocks(0.7, 0):
            inputs.extend(sensor.feed_block(block))
        assert len(inputs) == 1
        assert inputs[0].source == 'mic_audio'
        assert Path(inputs[0].audio_path).name.startswith('mic-')

    @pytest.mark.parametrize('source', ['agent_text', 'video_audio', ''])
    def test_invalid_source_rejected(self, tmp_path, source):
        with pytest.raises(ValueError, match='sorgente'):
            self._sensor(tmp_path, FakeClock(), source=source)


class FakePopen:
    """subprocess.Popen finto: registra argv, stdout leggibile, arresto pulito."""

    last_argv: tuple[str, ...] | None = None
    stdout_payload: bytes = b''

    def __init__(self, argv, executable=None, stdout=None, stderr=None):
        import io

        type(self).last_argv = tuple(argv)
        self.argv = tuple(argv)
        self.stdout_requested = stdout  # subprocess.PIPE nel chiamante reale
        self.stdout = io.BytesIO(type(self).stdout_payload)
        self.terminated = False

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.terminated = True


class TestAudioCapture:
    """F1: cattura di sistema via parec -d (monitor); mai il microfono per errore."""

    @staticmethod
    def _capture(callback, **kwargs):
        from src.sensors.mic_sensor import AudioCapture

        return AudioCapture(callback, **kwargs)

    def test_parec_device_uses_minus_d_and_skips_sounddevice(self, monkeypatch):
        from src.sensors import mic_sensor

        monkeypatch.setattr(mic_sensor.shutil, 'which', lambda name: '/usr/bin/parec')
        monkeypatch.setattr(mic_sensor.subprocess, 'Popen', FakePopen)
        FakePopen.stdout_payload = b'\x01\x00' * mic_sensor.BYTES_PER_BLOCK  # due blocchi completi
        blocks = []
        capture = self._capture(blocks.append, parec_device='@DEFAULT_MONITOR@')
        capture._try_sounddevice = lambda: pytest.fail('ramo sounddevice non deve essere tentato')
        capture.start()
        assert FakePopen.last_argv == (
            'parec', '--rate=16000', '--channels=1', '--format=s16le', '-d', '@DEFAULT_MONITOR@'
        )
        assert capture.backend == 'parec'
        assert capture.device_label == '@DEFAULT_MONITOR@'
        capture.stop()  # join del lettore: dopo, tutti i blocchi sono consegnati
        assert len(blocks) == 2
        assert int(blocks[0][0]) == 1

    def test_parec_without_device_records_mic_and_has_no_minus_d(self, monkeypatch):
        from src.sensors import mic_sensor

        monkeypatch.setattr(mic_sensor.shutil, 'which', lambda name: '/usr/bin/parec')
        monkeypatch.setattr(mic_sensor.subprocess, 'Popen', FakePopen)
        capture = self._capture(lambda block: None)
        capture._try_sounddevice = lambda: False  # nessun sounddevice nell'ambiente test
        capture.start()
        # ATTENZIONE (contratto Orchestrazione): parec SENZA -d registra il microfono
        assert '-d' not in FakePopen.last_argv
        assert capture.device_label == 'default'
        capture.stop()

    def test_parec_device_unavailable_raises_without_sounddevice(self, monkeypatch):
        from src.sensors import mic_sensor

        monkeypatch.setattr(mic_sensor.shutil, 'which', lambda name: None)
        capture = self._capture(lambda block: None, parec_device='@DEFAULT_MONITOR@')
        capture._try_sounddevice = lambda: pytest.fail('ramo sounddevice non deve essere tentato')
        with pytest.raises(RuntimeError, match='AUDIO_UNAVAILABLE'):
            capture.start()


class TestLiveInputQueue:
    def _item(self, path: str) -> LiveInput:
        return LiveInput(id=f'in-{digest(path)[:12]}', session_id='s1', source='mic_audio',
                         occurred_at=1.0, audio_path=path, duration_sec=1.0)

    def test_drop_oldest_calls_callback(self, tmp_path):
        from src.sensors.runner import LiveInputQueue

        paths = []
        for index in range(9):
            path = tmp_path / f'seg-{index}.wav'
            path.write_bytes(b'RIFF')
            paths.append(str(path))
        queue = LiveInputQueue(maxsize=8)
        dropped = []
        ok = True
        for index, path in enumerate(paths):
            ok = queue.put(self._item(path), on_drop=dropped.append)
        assert not ok  # il nono put ha scartato il più vecchio
        assert [item.audio_path for item in dropped] == [paths[0]]
        assert len(queue) == 8
        assert queue.dropped == 1
        # con callback custom la pulizia è del callback: il file resta al callback
        assert Path(paths[0]).exists()
        assert all(Path(p).exists() for p in paths[1:])

    def test_drop_deletes_wav_by_default(self, tmp_path):
        from src.sensors.runner import LiveInputQueue

        paths = []
        for index in range(9):
            path = tmp_path / f'seg-{index}.wav'
            path.write_bytes(b'RIFF')
            paths.append(str(path))
        queue = LiveInputQueue(maxsize=8)
        for path in paths:
            queue.put(self._item(path))
        assert queue.dropped == 1
        assert not Path(paths[0]).exists()  # WAV scartato non referenziato: eliminato
        assert all(Path(p).exists() for p in paths[1:])

    def test_drop_deletes_sysaudio_wav_by_default(self, tmp_path):
        from src.sensors.runner import LiveInputQueue, _delete_unreferenced_wav

        path = tmp_path / 'sys-drop.wav'
        path.write_bytes(b'RIFF')
        _delete_unreferenced_wav(
            LiveInput(id='in-sys-del', session_id='s1', source='system_audio',
                      occurred_at=1.0, audio_path=str(path), duration_sec=1.0)
        )
        assert not path.exists()  # F1: la pulizia copre anche la cattura di sistema

        queue = LiveInputQueue(maxsize=1)
        first = LiveInput(id='in-s1', session_id='s1', source='system_audio',
                          occurred_at=1.0, audio_path=str(path.parent / 'a.wav'), duration_sec=1.0)
        second = LiveInput(id='in-s2', session_id='s1', source='system_audio',
                           occurred_at=2.0, audio_path=str(path.parent / 'b.wav'), duration_sec=1.0)
        (path.parent / 'a.wav').write_bytes(b'RIFF')
        (path.parent / 'b.wav').write_bytes(b'RIFF')
        queue.put(first)
        queue.put(second)
        assert queue.dropped == 1
        assert not (path.parent / 'a.wav').exists()  # il più vecchio è stato rimosso

    def test_fifo_order_preserved(self, tmp_path):
        from src.sensors.runner import LiveInputQueue

        queue = LiveInputQueue(maxsize=8)
        for index in range(3):
            queue.put(self._item(f'/tmp/x{index}.wav'))
        batch = queue.drain()
        assert [item.audio_path for item in batch] == ['/tmp/x0.wav', '/tmp/x1.wav', '/tmp/x2.wav']
        assert len(queue) == 0
