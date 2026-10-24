"""Test sessione, comandi e lifecycle del runner (piano L4, contratti §5/§6).

Fake client/DB in fakes.py; nessun hardware. I test server usano socket IPC
reali in directory temporanee (livello «integrazione mock», 08 §1).
"""

import time
from pathlib import Path

import numpy as np
import pytest

from src.contracts import HudCommand, decode_control_response, decode_live_state
from src.sensors.test_sensors import (
    RunnerHarness,
    make_memory,
    mic_input,
    sysaudio_input,
    text_input,
)


def pin_cmd(session: str, memory_id: str, command_id: str) -> HudCommand:
    return HudCommand(command_id=command_id, session_id=session, action='pin', memory_id=memory_id)


def unpin_cmd(session: str, memory_id: str, command_id: str) -> HudCommand:
    return HudCommand(command_id=command_id, session_id=session, action='unpin', memory_id=memory_id)


def snap_cmd(session: str | None, command_id: str = 'snap'):
    return HudCommand(command_id=command_id, session_id=session, action='snapshot')


# --------------------------------------------------------------------------- pin e tray


class TestPin:
    def test_pin_a_then_b_reinforces_toward_survivors(self, tmp_path):
        harness = RunnerHarness(tmp_path, max_pins=4)
        mem_a = make_memory(tmp_path, 'a.md')
        mem_b = make_memory(tmp_path, 'b.md')
        harness.storage.add(mem_a, [1.0] + [0.0] * 255)
        harness.storage.add(mem_b, [0.0, 1.0] + [0.0] * 255)
        harness.ingest_text('contesto iniziale')

        session = harness.service.session_id
        response_a = harness.service.handle(pin_cmd(session, mem_a.id, 'pin-a'))
        assert response_a.success
        assert harness.storage.reinforce_calls == []  # nessun superstite: nessun rinforzo

        response_b = harness.service.handle(pin_cmd(session, mem_b.id, 'pin-b'))
        assert response_b.success
        assert harness.storage.reinforce_calls == [(mem_b.id, [mem_a.id], pytest.approx(0.25))]
        assert [p.memory.id for p in response_b.state.pinned] == [mem_a.id, mem_b.id]
        # lo snapshot al pin congela gli ultimi tre input in ordine cronologico
        assert response_b.state.pinned[1].snapshot.input_ids == [
            item.id for item in harness.service._history[-3:]
        ]

    def test_double_pin_is_idempotent(self, tmp_path):
        harness = RunnerHarness(tmp_path, max_pins=4)
        mem_a = make_memory(tmp_path, 'a.md')
        mem_b = make_memory(tmp_path, 'b.md')
        harness.storage.add(mem_a, [1.0] + [0.0] * 255)
        harness.storage.add(mem_b, [0.0, 1.0] + [0.0] * 255)
        session = harness.service.session_id

        first = harness.service.handle(pin_cmd(session, mem_a.id, 'pin-1'))
        second = harness.service.handle(pin_cmd(session, mem_a.id, 'pin-2'))
        assert first.success and second.success
        assert len(second.state.pinned) == 1
        assert harness.storage.reinforce_calls == []  # ripinnare non rinforza

    def test_retry_same_uuid_uses_stored_response(self, tmp_path):
        harness = RunnerHarness(tmp_path, max_pins=4)
        mem_a = make_memory(tmp_path, 'a.md')
        mem_b = make_memory(tmp_path, 'b.md')
        harness.storage.add(mem_a, [1.0] + [0.0] * 255)
        harness.storage.add(mem_b, [0.0, 1.0] + [0.0] * 255)
        session = harness.service.session_id
        command = pin_cmd(session, mem_a.id, 'retry-1')
        first = harness.service.handle(command)

        harness.storage.fail_on_reinforce = TimeoutError('DB giù')  # un retry non ripete la transazione
        second = harness.service.handle(command)
        assert second.success
        assert second.state.sequence == first.state.sequence
        assert second.command_id == first.command_id
        assert harness.storage.reinforce_calls == []

    def test_unpin_keeps_arcs(self, tmp_path):
        harness = RunnerHarness(tmp_path, max_pins=4)
        mem_a = make_memory(tmp_path, 'a.md')
        mem_b = make_memory(tmp_path, 'b.md')
        harness.storage.add(mem_a, [1.0] + [0.0] * 255)
        harness.storage.add(mem_b, [0.0, 1.0] + [0.0] * 255)
        session = harness.service.session_id
        harness.service.handle(pin_cmd(session, mem_a.id, 'p1'))
        harness.service.handle(pin_cmd(session, mem_b.id, 'p2'))
        calls = list(harness.storage.reinforce_calls)

        response = harness.service.handle(unpin_cmd(session, mem_a.id, 'u1'))
        assert response.success
        assert [p.memory.id for p in response.state.pinned] == [mem_b.id]
        assert harness.storage.reinforce_calls == calls  # unpin non tocca gli archi

        again = harness.service.handle(unpin_cmd(session, mem_a.id, 'u2'))
        assert again.success  # idempotente
        assert harness.storage.reinforce_calls == calls

    def test_pin_to_missing_memory(self, tmp_path):
        harness = RunnerHarness(tmp_path, max_pins=4)
        mem = make_memory(tmp_path, 'ghost.md')  # mai aggiunta allo storage
        session = harness.service.session_id
        response = harness.service.handle(pin_cmd(session, mem.id, 'p-ghost'))
        assert not response.success
        assert response.error_code == 'MEMORY_NOT_FOUND'
        assert response.state is None

    def test_pin_b_after_a_deleted(self, tmp_path):
        harness = RunnerHarness(tmp_path, max_pins=4)
        mem_a = make_memory(tmp_path, 'a.md')
        mem_b = make_memory(tmp_path, 'b.md')
        mem_c = make_memory(tmp_path, 'c.md')
        harness.storage.add(mem_a, [1.0] + [0.0] * 255)
        harness.storage.add(mem_b, [0.0, 1.0] + [0.0] * 255)
        harness.storage.add(mem_c, [0.5, 0.5] + [0.0] * 254)
        session = harness.service.session_id
        harness.service.handle(pin_cmd(session, mem_a.id, 'p1'))

        del harness.storage.records[mem_a.id]  # A cancellata dall'archivio
        response = harness.service.handle(pin_cmd(session, mem_b.id, 'p2'))
        assert response.success
        # rinforzo solo verso i sopravvissuti (A non c'è più: nessuna chiamata)
        assert harness.storage.reinforce_calls == []
        # il vecchio pin cancellato resta nel tray ma non blocca il nuovo
        assert [p.memory.id for p in response.state.pinned] == [mem_a.id, mem_b.id]

        response_c = harness.service.handle(pin_cmd(session, mem_c.id, 'p3'))
        assert response_c.success
        assert harness.storage.reinforce_calls == [
            (mem_c.id, [mem_b.id], pytest.approx(0.25))  # solo B è superstite
        ]

    def test_pin_limit_counts_survivors(self, tmp_path):
        harness = RunnerHarness(tmp_path, max_pins=2)
        mem_a = make_memory(tmp_path, 'a.md')
        mem_b = make_memory(tmp_path, 'b.md')
        mem_c = make_memory(tmp_path, 'c.md')
        for mem in (mem_a, mem_b, mem_c):
            harness.storage.add(mem, [0.3, 0.3] + [0.0] * 254)
        session = harness.service.session_id
        assert harness.service.handle(pin_cmd(session, mem_a.id, 'p1')).success
        assert harness.service.handle(pin_cmd(session, mem_b.id, 'p2')).success
        limited = harness.service.handle(pin_cmd(session, mem_c.id, 'p3'))
        assert not limited.success
        assert limited.error_code == 'PIN_LIMIT'

        del harness.storage.records[mem_b.id]  # B muore: C entra senza superare il limite
        admitted = harness.service.handle(pin_cmd(session, mem_c.id, 'p4'))
        assert admitted.success

    def test_db_failure_leaves_pin_out(self, tmp_path):
        harness = RunnerHarness(tmp_path, max_pins=4)
        mem_a = make_memory(tmp_path, 'a.md')
        mem_b = make_memory(tmp_path, 'b.md')
        harness.storage.add(mem_a, [1.0] + [0.0] * 255)
        harness.storage.add(mem_b, [0.0, 1.0] + [0.0] * 255)
        session = harness.service.session_id
        harness.service.handle(pin_cmd(session, mem_a.id, 'p1'))

        harness.storage.fail_on_reinforce = TimeoutError('DB giù')
        response = harness.service.handle(pin_cmd(session, mem_b.id, 'p2'))
        assert not response.success
        assert response.error_code == 'STORAGE_ERROR'
        assert len(harness.service.handle(snap_cmd(session, 's1')).state.pinned) == 1

    def test_snapshot_freezes_memory_view(self, tmp_path):
        harness = RunnerHarness(tmp_path, max_pins=4)
        mem = make_memory(tmp_path, 'freeze.md', text='versione pinnata')
        harness.storage.add(mem, [1.0] + [0.0] * 255)
        session = harness.service.session_id
        pinned_view = harness.service.handle(pin_cmd(session, mem.id, 'p1')).state.pinned[0].memory

        updated = make_memory(tmp_path, 'freeze.md', text='versione nuova')  # stesso file, nuovo sha
        harness.storage.records[mem.id] = (updated, [0.0, 1.0] + [0.0] * 254)
        current = harness.service.handle(snap_cmd(session, 's2')).state.pinned[0].memory
        assert current == pinned_view  # il pin congela la MemoryView vista al pin

    def test_no_arcs_from_visible_cards(self, tmp_path):
        harness = RunnerHarness(tmp_path)
        mem = make_memory(tmp_path, 'seen.md')
        vector = [1.0] + [0.0] * 255
        harness.storage.add(mem, vector)
        for index in range(3):
            harness.client.vectors[f'riga {index}'] = vector
            harness.ingest_text(f'riga {index}')
        assert len(harness.service.handle(snap_cmd(None, 'd1')).state.results) >= 1
        assert harness.storage.reinforce_calls == []  # la visualizzazione non crea archi


# --------------------------------------------------------------------------- sessione e reset


class TestSessionLifecycle:
    def test_snapshot_discovery_accepts_none(self, tmp_path):
        harness = RunnerHarness(tmp_path)
        harness.ingest_text('primo input')
        response = harness.service.handle(snap_cmd(None, 'disc-1'))
        assert response.success
        assert response.state.session_id == harness.service.session_id
        assert [item.text for item in response.session_inputs] == ['primo input']

    def test_stale_session_rejected(self, tmp_path):
        harness = RunnerHarness(tmp_path)
        old = harness.service.session_id
        harness.service.handle(HudCommand(command_id='ns', session_id=old, action='new_session'))
        harness.ingest_text('input ok')
        with pytest.raises(ValueError, match='STALE_SESSION'):
            harness.service.ingest(text_input('vecchio', session=old))
        response = harness.service.handle(pin_cmd(old, 'qualsiasi', 'p-old'))
        assert not response.success and response.error_code == 'STALE_SESSION'

    def test_new_session_resets_state_keeps_sequence(self, tmp_path):
        harness = RunnerHarness(tmp_path, max_pins=4)
        mem = make_memory(tmp_path, 'a.md')
        query = [1.0] + [0.0] * 255
        harness.client.vectors['argomento'] = query
        harness.storage.add(mem, query)
        harness.ingest_text('argomento')
        session = harness.service.session_id
        old_state = harness.service.handle(snap_cmd(session, 's1')).state
        harness.service.handle(pin_cmd(session, mem.id, 'p1'))

        response = harness.service.handle(
            HudCommand(command_id='ns1', session_id=session, action='new_session')
        )
        assert response.success
        new_state = response.state
        assert new_state.session_id != old_state.session_id
        assert new_state.sequence > old_state.sequence  # crescente attraverso il reset
        assert new_state.results == []  # tracker azzerati: nessuna sorgente attiva
        assert new_state.pinned == []
        assert new_state.latest_inputs == []
        assert response.lease_expires_at == pytest.approx(harness.clock.time() + 120.0)

    def test_input_after_reset_rejected_and_events_discardable(self, tmp_path):
        harness = RunnerHarness(tmp_path)
        old = harness.service.session_id
        session = harness.service.session_id
        harness.service.handle(
            HudCommand(command_id='ns1', session_id=session, action='new_session')
        )
        with pytest.raises(ValueError, match='STALE_SESSION'):
            harness.service.ingest(text_input('dopo reset', session=old))
        status = harness.service.status_event()
        assert status.sequence > 0

    def test_new_session_retry_is_deduplicated(self, tmp_path):
        harness = RunnerHarness(tmp_path)
        session = harness.service.session_id
        command = HudCommand(command_id='ns-dup', session_id=session, action='new_session')
        first = harness.service.handle(command)
        second = harness.service.handle(command)
        assert second.state.session_id == first.state.session_id  # nessun secondo reset

    def test_lease_on_reset_and_expired_retry(self, tmp_path):
        harness = RunnerHarness(tmp_path)
        session = harness.service.session_id
        import os

        audio_dir = os.path.join(harness.paths.run_dir, session)
        os.makedirs(audio_dir, exist_ok=True)
        response = harness.service.handle(
            HudCommand(command_id='ns1', session_id=session, action='new_session')
        )
        lease = response.lease_expires_at
        assert lease == pytest.approx(harness.clock.time() + 120.0)
        assert os.path.isdir(audio_dir)  # il lease rinvia la rimozione

        discovery = harness.service.handle(snap_cmd(None, 'd1'))
        assert discovery.lease_expires_at == lease  # il lease è visibile in discovery

        harness.clock.advance(121.0)
        harness.service.status_event()  # sweep alla scadenza
        assert not os.path.isdir(audio_dir)  # directory rimossa dopo il lease
        expired = harness.service.handle(snap_cmd(None, 'd2'))
        assert expired.lease_expires_at is None

    def test_session_input_limit_suspends_and_preserves(self, tmp_path):
        harness = RunnerHarness(tmp_path, session_max_inputs=3)
        for index in range(3):
            harness.ingest_text(f'input {index}')
        with pytest.raises(ValueError, match='SESSION_FULL'):
            harness.ingest_text('in sospeso')
        status = harness.service.status_event()
        assert status.sensor_status.get('session') == 'full'
        response = harness.service.handle(snap_cmd(None, 's1'))
        assert len(response.session_inputs) == 3  # gli input già presenti restano
        # pin/export/reset restano disponibili
        mem = make_memory(tmp_path, 'lim.md')
        harness.storage.add(mem, [1.0] + [0.0] * 255)
        assert harness.service.handle(
            pin_cmd(harness.service.session_id, mem.id, 'p1')
        ).success

    def test_index_generation_recomputed_without_ema_change(self, tmp_path):
        harness = RunnerHarness(tmp_path)
        mem = make_memory(tmp_path, 'idx.md')
        query = [1.0] + [0.0] * 255
        harness.client.vectors['cerca'] = query
        harness.storage.add(mem, query)
        harness.ingest_text('cerca')
        before_calls = len(harness.storage.search_calls)
        harness.storage.index_gen = 1  # nuovo commit dell'indice

        event = harness.service.status_event(reason='index')
        assert event.reason == 'index'
        assert len(event.results) == 1  # ricerca ricalcolata
        assert len(harness.storage.search_calls) > before_calls

    def test_latest_inputs_window_is_five_chronological(self, tmp_path):
        harness = RunnerHarness(tmp_path)
        for index in range(7):
            harness.ingest_text(f'messaggio {index}')
        event = harness.service.status_event()
        assert [item.text for item in event.latest_inputs] == [
            f'messaggio {index}' for index in range(2, 7)
        ]
        assert event.source_last_seen.get('agent_text') == pytest.approx(harness.clock.time())


# --------------------------------------------------------------------------- server IPC


class TestRunnerServer:
    def _server(self, tmp_path, **kwargs):
        import zmq

        from src.sensors.runner import RunnerServer

        harness = RunnerHarness(tmp_path, **kwargs)
        context = zmq.Context()
        server = RunnerServer(harness.service, context=context, heartbeat_sec=0.05)
        return harness, server, context

    def test_step_processes_commands_before_inputs(self, tmp_path):
        harness, server, context = self._server(tmp_path)
        try:
            mem = make_memory(tmp_path, 'prio.md')
            harness.storage.add(mem, [1.0] + [0.0] * 255)
            server.queue.put(text_input('input live', session=harness.service.session_id))
            command = pin_cmd(harness.service.session_id, mem.id, 'prio-cmd')
            response_slot = server.submit_async(command)

            server.step()  # un giro: prima i comandi, poi gli input
            assert response_slot['done'].is_set()
            assert response_slot['response'].success
            assert [p.memory.id for p in response_slot['response'].state.pinned] == [mem.id]
            assert len(server.queue) == 0  # l'input è stato ingoiato nello stesso giro
            # priorità: l'evento del comando precede quello dell'input
            assert response_slot['response'].state.sequence < server.last_event.sequence
            assert server.last_event.reason == 'input'
        finally:
            server.close()
            context.term()

    def test_pub_events_and_control_roundtrip(self, tmp_path):
        import zmq

        from src.contracts import pack

        harness, server, context = self._server(tmp_path)
        try:
            server.start()
            sub = context.socket(zmq.SUB)
            sub.subscribe(b'')
            sub.connect(harness.paths.events_endpoint)
            time.sleep(0.2)  # slow joiner

            mem = make_memory(tmp_path, 'ipc.md')
            harness.storage.add(mem, [1.0] + [0.0] * 255)
            server.queue.put(text_input('via ipc', session=harness.service.session_id))

            deadline = time.time() + 5
            event = None
            while time.time() < deadline:
                if sub.poll(200):
                    event = decode_live_state(sub.recv_string())
                    if event.reason == 'input':
                        break
            assert event is not None and event.reason == 'input'
            assert event.session_id == harness.service.session_id

            req = context.socket(zmq.REQ)
            req.setsockopt(zmq.RCVTIMEO, 4000)
            req.connect(harness.paths.control_endpoint)
            req.send_string(pack('hud.command', snap_cmd(None, 'ipc-snap')))
            response = decode_control_response(req.recv_string())
            assert response.success
            assert response.state.session_id == harness.service.session_id
            assert any(item.text == 'via ipc' for item in response.session_inputs)

            command = pin_cmd(harness.service.session_id, mem.id, 'ipc-pin')
            req.send_string(pack('hud.command', command))
            response = decode_control_response(req.recv_string())
            assert response.success
            assert [p.memory.id for p in response.state.pinned] == [mem.id]

            # retry stesso command_id via IPC: risposta memorizzata, stesso sequence
            req.send_string(pack('hud.command', command))
            retry = decode_control_response(req.recv_string())
            assert retry.state.sequence == response.state.sequence
            assert harness.storage.reinforce_calls == []  # tray vuoto: nessun rinforzo da ripetere
            sub.close()
            req.close()
        finally:
            server.close()
            context.term()

    def test_session_full_suspension_via_server(self, tmp_path):
        harness, server, context = self._server(tmp_path, session_max_inputs=1)
        try:
            session = harness.service.session_id
            server.queue.put(text_input('uno', session=session))
            server.step()
            server.queue.put(text_input('due', session=session))
            server.queue.put(text_input('tre', session=session))
            server.step()  # 'due' e 'tre' sospesi per SESSION_FULL
            response = harness.service.handle(snap_cmd(None, 'full-snap'))
            assert [item.text for item in response.session_inputs] == ['uno']
            status = harness.service.status_event()
            assert status.sensor_status.get('session') == 'full'
        finally:
            server.close()
            context.term()


# --------------------------------------------------------------------------- replay (L4)


def _write_wav_mono16k(path, samples: 'np.ndarray') -> None:
    import wave

    with wave.open(str(path), 'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(samples.astype('<i2').tobytes())


def _replay_manifest(tmp_path):
    """Corpus e manifest sintetici per il replay con backend mock."""
    import hashlib
    import json

    (tmp_path / 'm1.md').write_text('# Token\nIl token di accesso scade dopo quindici minuti.\n', encoding='utf-8')
    (tmp_path / 'm2.md').write_text('# Ricetta\nIl pane richiede farina acqua e lievito.\n', encoding='utf-8')
    samples = np.full(16000, 3000, dtype=np.int16)  # 1 s di tono costante
    _write_wav_mono16k(tmp_path / 'q1.wav', samples)
    manifest = {
        'schema_version': 1,
        'corpus_checksum': 'sintetico',
        'files': [
            {'id': 'M1', 'path': 'm1.md', 'sha256': hashlib.sha256(b'x').hexdigest()},
            {'id': 'M2', 'path': 'm2.md', 'sha256': hashlib.sha256(b'y').hexdigest()},
        ],
        'queries': [
            {'id': 'Q1', 'split': 'calibration', 'text': 'token di accesso',
             'audio_path': 'q1.wav',
             'audio_sha256': hashlib.sha256(b'q').hexdigest(),
             'relevant': [{'file_id': 'M1'}]},
        ],
    }
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest), encoding='utf-8')
    return str(path)


class TestReplayHarness:
    def _harness(self, tmp_path):
        from src.inference.model import mock_space
        from src.sensors.replay import ReplayHarness

        manifest_path = _replay_manifest(tmp_path)
        harness = ReplayHarness(
            mock_space(), manifest_path, 'mock', work_dir=str(tmp_path / 'work')
        )
        return harness

    def test_text_query_replays_with_debounce(self, tmp_path):
        harness = self._harness(tmp_path)
        try:
            queries = harness.manifest['queries']
            outcomes = harness.run(queries, source='agent_text', repeat=2)
            assert len(outcomes) == 2
            outcome = outcomes[0]
            assert outcome['query_id'] == 'Q1'
            assert outcome['error'] is None
            assert outcome['relevant'] == [{'file_id': 'M1'}]
            assert outcome['results'], 'il mock deve ritrovare il documento col testo'
            assert outcome['results'][0]['memory']['file_path'].endswith('m1.md')
            assert outcome['results'][0]['winning_source'] == 'agent_text'
        finally:
            harness.close()

    def test_audio_query_replays_through_vad(self, tmp_path):
        harness = self._harness(tmp_path)
        try:
            outcomes = harness.run(harness.manifest['queries'], source='mic_audio', repeat=1)
            assert len(outcomes) == 1
            outcome = outcomes[0]
            assert outcome['error'] is None  # pipeline completa: calibrazione, segmenti, evento
            # il mock non ha comprensione audio: astensione attesa, nessun crash
            assert isinstance(outcome['results'], list)
        finally:
            harness.close()

    def test_query_timeout_formula(self, tmp_path):
        harness = self._harness(tmp_path)
        try:
            query = harness.manifest['queries'][0]
            audio_timeout = harness._query_timeout(query, 'mic_audio')
            # 1 s di WAV + 1 segmento * (2*30+5) + 2
            assert audio_timeout == pytest.approx(1.0 + 65.0 + 2.0)
            assert harness._query_timeout(query, 'agent_text') == pytest.approx(67.0)
        finally:
            harness.close()


class TestMicCommand:
    def test_set_mic_pauses_only_mic(self, tmp_path):
        harness = RunnerHarness(tmp_path)
        session = harness.service.session_id
        mem = make_memory(tmp_path, 'text-live.md')
        harness.storage.add(mem, [1.0] + [0.0] * 255)
        harness.client.vectors['seconda frase col mic spento'] = [1.0] + [0.0] * 255
        harness.ingest_text('prima frase')

        response = harness.service.handle(
            HudCommand(command_id='mic-off', session_id=session, action='set_mic', enabled=False)
        )
        assert response.success
        assert response.state.sensor_status['mic'] == 'off'

        # la sorgente testuale resta attiva col mic in pausa
        event = harness.ingest_text('seconda frase col mic spento')
        assert event.results, 'il testo deve produrre risultati anche a mic spento'
        assert event.sensor_status['mic'] == 'off'

        back = harness.service.handle(
            HudCommand(command_id='mic-on', session_id=session, action='set_mic', enabled=True)
        )
        assert back.state.sensor_status['mic'] == 'on'


class TestSensorStatusSources:
    """F1: stato per sorgente — mic resta 'mic', la cattura di sistema è 'sysaudio'."""

    def test_default_both_audio_sources_on(self, tmp_path):
        harness = RunnerHarness(tmp_path)
        status = harness.service.status_event().sensor_status
        assert status['mic'] == 'on'
        assert status['sysaudio'] == 'on'

    def test_sensor_enabled_false_keeps_sysaudio_off(self, tmp_path):
        harness = RunnerHarness(tmp_path, sensor_enabled={'system_audio': False})
        status = harness.service.status_event().sensor_status
        assert status['sysaudio'] == 'off'
        assert status['mic'] == 'on'

    def test_no_mic_wiring_reflects_off_in_status(self, tmp_path):
        """F2 (richiesta Orchestratore): --no-mic → sensor_enabled mic_audio False → status off.

        Il dict è identico a quello costruito in runner main().
        """
        args_mic_off = {'mic_audio': not True, 'system_audio': False}
        harness = RunnerHarness(tmp_path, sensor_enabled=args_mic_off)
        status = harness.service.status_event().sensor_status
        assert status['mic'] == 'off'
        assert status['sysaudio'] == 'off'
        assert harness.service.mic_enabled is False

    def test_unknown_sensor_enabled_rejected(self, tmp_path):
        with pytest.raises(ValueError, match='sorgente non ammessa'):
            RunnerHarness(tmp_path, sensor_enabled={'radio': True})

    def test_set_mic_leaves_sysaudio_unchanged(self, tmp_path):
        harness = RunnerHarness(tmp_path)
        session = harness.service.session_id
        harness.service.handle(
            HudCommand(command_id='m-off', session_id=session, action='set_mic', enabled=False)
        )
        status = harness.service.status_event().sensor_status
        assert status['mic'] == 'off'
        assert status['sysaudio'] == 'on'  # il comando mic non tocca la cattura di sistema

    def test_mic_enabled_property_compat(self, tmp_path):
        """Compat/replay: la property mic_enabled continua a riflettere set_mic."""
        harness = RunnerHarness(tmp_path)
        assert harness.service.mic_enabled is True
        session = harness.service.session_id
        harness.service.handle(
            HudCommand(command_id='m-off', session_id=session, action='set_mic', enabled=False)
        )
        assert harness.service.mic_enabled is False
        assert harness.service.sensor_enabled('mic_audio') is False
        assert harness.service.sensor_enabled('system_audio') is True


class TestAudioBudget:
    """F1: il budget session_audio_max_bytes conta anche la cattura di sistema."""

    def test_budget_counts_mic_and_system_combined(self, tmp_path):
        harness = RunnerHarness(tmp_path, session_audio_max_bytes=1500)
        mic_file = harness.mic_path('budget-mic.wav')
        sys_file = harness.mic_path('budget-sys.wav')
        Path(mic_file).write_bytes(b'\0' * 1000)
        Path(sys_file).write_bytes(b'\0' * 904)
        harness.service.ingest(
            mic_input(mic_file, duration=0.03, session=harness.service.session_id)
        )
        assert harness.service._audio_bytes == 1000
        with pytest.raises(ValueError, match='SESSION_FULL'):
            harness.service.ingest(
                sysaudio_input(sys_file, duration=0.03, session=harness.service.session_id)
            )

    def test_system_audio_within_budget_passes(self, tmp_path):
        harness = RunnerHarness(tmp_path, session_audio_max_bytes=2000)
        sys_file = harness.mic_path('budget-ok.wav')
        Path(sys_file).write_bytes(b'\0' * 1000)
        harness.service.ingest(
            sysaudio_input(sys_file, duration=0.03, session=harness.service.session_id)
        )
        assert harness.service._audio_bytes == 1000

    def test_text_input_never_counts_audio_budget(self, tmp_path):
        harness = RunnerHarness(tmp_path, session_audio_max_bytes=100)
        harness.ingest_text('il testo non pesa sul budget audio')
        assert harness.service._audio_bytes == 0


class TestReplayLatency:
    def test_outcome_carries_latency_and_elapsed(self, tmp_path):
        from src.inference.model import mock_space
        from src.sensors.replay import ReplayHarness

        manifest_path = _replay_manifest(tmp_path)
        harness = ReplayHarness(mock_space(), manifest_path, 'mock', work_dir=str(tmp_path / 'work'))
        try:
            outcomes = harness.run(harness.manifest['queries'], source='agent_text', repeat=1)
            outcome = outcomes[0]
            # S5: dalla fine dell'input (riga originale scritta) all'evento pubblicato,
            # debounce incluso; il timeout non usa la latency (semantica invariata)
            assert outcome['error'] is None
            assert 'latency_sec' in outcome and 'elapsed_sec' in outcome
            assert outcome['latency_sec'] >= 0.9  # il debounce da 1 s è incluso
            assert outcome['elapsed_sec'] >= outcome['latency_sec']
        finally:
            harness.close()
