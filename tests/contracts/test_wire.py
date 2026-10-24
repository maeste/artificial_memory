"""Test del wire format e dei decoder dei contratti (Orchestrator, O1).

Ogni errore deve essere esplicito: nessuna coercizione silenziosa di tipo,
nessun campo sconosciuto accettato, nessun numero non finito tollerato.
"""

import dataclasses
import json

import pytest

from src.contracts import (
    EMBED_DIM,
    ContextSnapshot,
    ControlResponse,
    EmbedRequest,
    EmbedResponse,
    ExportSettings,
    HudCommand,
    LiveInput,
    LiveStateEvent,
    MemoryRecord,
    MemoryView,
    PinnedMemory,
    ScoredMemory,
    SearchSettings,
    SpaceSpec,
    decode_control_response,
    decode_embed_request,
    decode_embed_response,
    decode_health_response,
    decode_hud_command,
    decode_live_state,
    make_memory_id,
    pack,
    pack_health,
    unpack,
    validate_vector,
)

# --------------------------------------------------------------------------- fixture


def unit_vector(axis: int) -> list[float]:
    vec = [0.0] * EMBED_DIM
    vec[axis] = 1.0
    return vec


@pytest.fixture
def memory_view() -> MemoryView:
    return MemoryView(
        id='74a2476b67d62a26b8d1cfad91bb9957828442091e22b8fb7b889919c36bad52',
        file_path='/home/demo/watch_dir/f01.md',
        modality='markdown',
        chunk_key='section:Cache:0:part:0',
        content_sha256='57073c18c3938f8ca9eb0396911a27b01de78ced6d73c4811db9058aee579ddf',
        source_sha256='013b48308669da5143539e240e3b0a0603b34759f95622b8549f4d4dcea0fc3ca2',
        context_header='f01.md > Cache',
        snippet_text='La cache riduce gli accessi al database.',
        content_text='La cache riduce gli accessi al database.',
        created_at=1792300000.0,
    )


@pytest.fixture
def video_memory() -> MemoryView:
    return MemoryView(
        id='a'.rjust(64, 'a'),
        file_path='/home/demo/watch_dir/f13.mp4',
        modality='video',
        chunk_key='video:40000:85000',
        content_sha256='b'.rjust(64, 'b'),
        source_sha256='c'.rjust(64, 'c'),
        context_header='f13.mp4 > [40.0s, 85.0s)',
        snippet_text='Video 95 s, finestra [40,85)',
        content_text='',
        start_sec=40.0,
        end_sec=85.0,
        created_at=1792300000.0,
    )


@pytest.fixture
def scored(memory_view) -> ScoredMemory:
    return ScoredMemory(
        memory=memory_view,
        cosine_score=0.7,
        z_score=1.8,
        synaptic_boost=0.0,
        final_score=1.8,
        matched_sources=['agent_text'],
        winning_source='agent_text',
    )


@pytest.fixture
def live_state(scored, memory_view) -> LiveStateEvent:
    pin = PinnedMemory(
        memory=memory_view,
        pinned_at=1792300010.0,
        snapshot=ContextSnapshot(
            id='snap-1', session_id='sess-1', captured_at=1792300009.0, input_ids=['in-1', 'in-2', 'in-3']
        ),
    )
    inputs = [
        LiveInput(
            id=f'in-{i}',
            session_id='sess-1',
            source='agent_text',
            occurred_at=1792300000.0 + i,
            text=f'messaggio {i}',
        )
        for i in range(5)
    ]
    return LiveStateEvent(
        event_id='ev-1',
        sequence=42,
        session_id='sess-1',
        space_id='0'.rjust(64, '0'),
        backend='real',
        emitted_at=1792300010.0,
        reason='pin',
        latest_inputs=inputs,
        results=[scored],
        pinned=[pin],
        sensor_status={'agent_text': 'active', 'mic_audio': 'active'},
        source_last_seen={'agent_text': 1792300005.0, 'mic_audio': 1792300002.0},
        dropped_inputs=1,
    )


# --------------------------------------------------------------------------- pack/unpack base


def test_embed_roundtrip():
    req = EmbedRequest(request_id='q1', is_query=True, text_payload='cache')
    assert decode_embed_request(pack('embed.request', req)) == req


def test_unknown_protocol_is_rejected():
    raw = json.dumps({'schema_version': 99, 'type': 'embed.request', 'payload': {}})
    with pytest.raises(ValueError, match='SCHEMA_VERSION_MISMATCH'):
        unpack(raw, 'embed.request')


def test_relative_media_path_is_rejected():
    req = EmbedRequest(request_id='q1', is_query=True, image_paths=['relative.png'])
    with pytest.raises(ValueError):
        decode_embed_request(pack('embed.request', req))


def test_relative_audio_path_is_rejected():
    req = EmbedRequest(request_id='q1', is_query=True, audio_path='relative.wav')
    with pytest.raises(ValueError):
        decode_embed_request(pack('embed.request', req))


def test_wrong_kind_is_rejected():
    req = EmbedRequest(request_id='q1', is_query=True, text_payload='cache')
    with pytest.raises(ValueError, match='INVALID_ENVELOPE'):
        unpack(pack('embed.request', req), 'hud.command')


def test_non_dict_root_is_rejected():
    with pytest.raises(ValueError, match='INVALID_ENVELOPE'):
        unpack(json.dumps([1, 2]), 'embed.request')


def test_payload_not_dict_is_rejected():
    with pytest.raises(ValueError, match='INVALID_ENVELOPE'):
        unpack(json.dumps({'schema_version': 1, 'type': 'embed.request', 'payload': []}), 'embed.request')


def test_unicode_roundtrip_is_preserved():
    req = EmbedRequest(request_id='q1', is_query=True, text_payload='perché la	cache è calda')
    raw = pack('embed.request', req)
    assert '\\u' not in raw  # ensure_ascii=False
    assert decode_embed_request(raw) == req


# --------------------------------------------------------------------------- embed request/response


def test_query_with_text_and_audio_is_rejected():
    req = EmbedRequest(request_id='q1', is_query=True, text_payload='cache', audio_path='/abs/q.wav')
    with pytest.raises(ValueError, match='INVALID_REQUEST'):
        decode_embed_request(pack('embed.request', req))


def test_empty_payload_is_rejected():
    req = EmbedRequest(request_id='q1', is_query=True)
    with pytest.raises(ValueError, match='INVALID_REQUEST'):
        decode_embed_request(pack('embed.request', req))


def test_too_many_images_are_rejected():
    req = EmbedRequest(
        request_id='q1', is_query=False, text_payload='doc', image_paths=[f'/abs/i{i}.png' for i in range(4)]
    )
    with pytest.raises(ValueError, match='INVALID_REQUEST'):
        decode_embed_request(pack('embed.request', req))


def test_embed_response_roundtrip():
    resp = EmbedResponse(request_id='q1', success=True, space_id='sp', embedding=unit_vector(0))
    assert decode_embed_response(pack('embed.response', resp)) == resp


def test_error_response_has_no_embedding():
    resp = EmbedResponse(request_id='q1', success=False, space_id='sp', error_code='MODEL_UNAVAILABLE', error='ko')
    decoded = decode_embed_response(pack('embed.response', resp))
    assert decoded.success is False
    assert decoded.embedding == []
    assert decoded.error_code == 'MODEL_UNAVAILABLE'


def test_error_response_without_code_is_rejected():
    resp = EmbedResponse(request_id='q1', success=False, space_id='sp', error='ko')
    with pytest.raises(ValueError):
        decode_embed_response(pack('embed.response', resp))


def test_success_response_with_invalid_vector_is_rejected():
    vec = unit_vector(0)
    vec[1] = 1.0  # norma sbagliata
    resp = EmbedResponse(request_id='q1', success=True, space_id='sp', embedding=vec)
    with pytest.raises(ValueError, match='INVALID_VECTOR_NORM'):
        decode_embed_response(pack('embed.response', resp))


def test_health_envelope_roundtrip():
    raw = pack_health('health.request', {})
    value = json.loads(raw)
    assert value['type'] == 'health.request' and value['payload'] == {}
    raw_resp = pack_health('health.response', {'ready': True, 'space_id': 'sp', 'backend': 'real', 'error': None})
    assert json.loads(raw_resp)['payload']['ready'] is True


# --------------------------------------------------------------------------- live.state completo


def test_live_state_roundtrip_with_nested_results_pins_snapshot(live_state):
    decoded = decode_live_state(pack('live.state', live_state))
    assert decoded == live_state
    assert isinstance(decoded.results[0].memory, MemoryView)  # DTO annidato, non dict piatto
    assert isinstance(decoded.pinned[0].snapshot, ContextSnapshot)
    assert decoded.latest_inputs[-1].text == 'messaggio 4'
    assert len(decoded.latest_inputs) == 5


def test_live_state_rejects_unknown_field(live_state):
    payload = json.loads(pack('live.state', live_state))['payload']
    payload['surprise'] = 1
    raw = json.dumps({'schema_version': 1, 'type': 'live.state', 'payload': payload})
    with pytest.raises(ValueError, match='UNKNOWN_FIELD'):
        decode_live_state(raw)


def test_live_state_rejects_wrong_reason(live_state):
    payload = json.loads(pack('live.state', live_state))['payload']
    payload['reason'] = 'spontaneous'
    raw = json.dumps({'schema_version': 1, 'type': 'live.state', 'payload': payload})
    with pytest.raises(ValueError):
        decode_live_state(raw)


def test_live_state_rejects_six_latest_inputs(live_state):
    payload = json.loads(pack('live.state', live_state))['payload']
    payload['latest_inputs'].append(dict(payload['latest_inputs'][0], id='in-6'))
    raw = json.dumps({'schema_version': 1, 'type': 'live.state', 'payload': payload})
    with pytest.raises(ValueError, match='INVALID_REQUEST'):
        decode_live_state(raw)


def test_live_state_rejects_nan_in_scores(scored, live_state):
    event = dataclasses.replace(live_state, results=[dataclasses.replace(scored, cosine_score=float('nan'))])
    with pytest.raises(ValueError, match='NON_FINITE'):
        pack('live.state', event)


def test_cosine_out_of_range_is_rejected(scored, live_state):
    event = dataclasses.replace(live_state, results=[dataclasses.replace(scored, cosine_score=1.5)])
    with pytest.raises(ValueError, match='INVALID_REQUEST'):
        decode_live_state(pack('live.state', event))


def test_snapshot_with_four_inputs_is_rejected(live_state):
    pin = dataclasses.replace(
        live_state.pinned[0],
        snapshot=dataclasses.replace(live_state.pinned[0].snapshot, input_ids=['a', 'b', 'c', 'd']),
    )
    event = dataclasses.replace(live_state, pinned=[pin])
    with pytest.raises(ValueError, match='INVALID_REQUEST'):
        decode_live_state(pack('live.state', event))


# --------------------------------------------------------------------------- hud.command


def test_hud_command_roundtrip():
    cmd = HudCommand(command_id='c1', session_id='sess-1', action='pin', memory_id='m1')
    assert decode_hud_command(pack('hud.command', cmd)) == cmd


def test_pin_without_memory_id_is_rejected():
    cmd = HudCommand(command_id='c1', session_id='sess-1', action='pin')
    with pytest.raises(ValueError, match='INVALID_REQUEST'):
        decode_hud_command(pack('hud.command', cmd))


def test_promote_roundtrip_and_validation():
    cmd = HudCommand(command_id='c1', session_id='sess-1', action='promote', memory_id='m1')
    assert decode_hud_command(pack('hud.command', cmd)) == cmd


def test_promote_requires_memory_id():
    cmd = HudCommand(command_id='c1', session_id='sess-1', action='promote')
    with pytest.raises(ValueError, match='INVALID_REQUEST'):
        decode_hud_command(pack('hud.command', cmd))


def test_promote_forbids_enabled():
    payload = {'command_id': 'c1', 'session_id': 's', 'action': 'promote', 'memory_id': 'm1',
               'enabled': True}
    with pytest.raises(ValueError, match='INVALID_REQUEST'):
        decode_hud_command(json.dumps({'schema_version': 1, 'type': 'hud.command', 'payload': payload}))


def test_system_audio_input_roundtrip():
    inp = LiveInput(id='in-sys', session_id='s', source='system_audio',
                    occurred_at=1.0, audio_path='/abs/sys.wav', duration_sec=2.0)
    state = LiveStateEvent(
        event_id='e', sequence=1, session_id='s', space_id='x', backend='mock', emitted_at=1.0,
        reason='input', latest_inputs=[inp], results=[], pinned=[], sensor_status={'system_audio': 'on'},
        source_last_seen={'system_audio': 1.0}, dropped_inputs=0,
    )
    decoded = decode_live_state(pack('live.state', state))
    assert decoded.latest_inputs[0].source == 'system_audio'


def test_system_audio_without_audio_path_is_rejected():
    payload = {'id': 'in', 'session_id': 's', 'source': 'system_audio', 'occurred_at': 1.0}
    raw = json.dumps({'schema_version': 1, 'type': 'live.state',
                      'payload': {'event_id': 'e', 'sequence': 1, 'session_id': 's', 'space_id': 'x',
                                  'backend': 'mock', 'emitted_at': 1.0, 'reason': 'input',
                                  'latest_inputs': [payload], 'results': [], 'pinned': [],
                                  'sensor_status': {}, 'dropped_inputs': 0}})
    with pytest.raises(ValueError, match='system_audio'):
        decode_live_state(raw)


def test_memory_view_session_scope_roundtrip(memory_view):
    scoped = dataclasses.replace(memory_view, session_scope=True)
    state = LiveStateEvent(
        event_id='e', sequence=1, session_id='s', space_id='x', backend='mock', emitted_at=1.0,
        reason='input', latest_inputs=[], results=[ScoredMemory(memory=scoped, cosine_score=0.5,
                                                               z_score=0.5, synaptic_boost=0.0,
                                                               final_score=0.5)], pinned=[],
        sensor_status={}, dropped_inputs=0,
    )
    decoded = decode_live_state(pack('live.state', state))
    assert decoded.results[0].memory.session_scope is True


def test_memory_view_session_scope_defaults_false(memory_view):
    state = LiveStateEvent(
        event_id='e', sequence=1, session_id='s', space_id='x', backend='mock', emitted_at=1.0,
        reason='input', latest_inputs=[], results=[ScoredMemory(memory=memory_view, cosine_score=0.5,
                                                               z_score=0.5, synaptic_boost=0.0,
                                                               final_score=0.5)], pinned=[],
        sensor_status={}, dropped_inputs=0,
    )
    decoded = decode_live_state(pack('live.state', state))
    assert decoded.results[0].memory.session_scope is False


def test_set_mic_without_enabled_is_rejected():
    cmd = HudCommand(command_id='c1', session_id='sess-1', action='set_mic')
    with pytest.raises(ValueError, match='INVALID_REQUEST'):
        decode_hud_command(pack('hud.command', cmd))


def test_mutation_without_session_is_rejected():
    cmd = HudCommand(command_id='c1', session_id=None, action='new_session')
    with pytest.raises(ValueError, match='INVALID_REQUEST'):
        decode_hud_command(pack('hud.command', cmd))


def test_snapshot_accepts_null_session():
    cmd = HudCommand(command_id='c1', session_id=None, action='snapshot')
    assert decode_hud_command(pack('hud.command', cmd)) == cmd


def test_unknown_action_is_rejected():
    cmd = HudCommand(command_id='c1', session_id='sess-1', action='explode')
    with pytest.raises(ValueError):
        decode_hud_command(pack('hud.command', cmd))


# --------------------------------------------------------------------------- control.response


def test_control_response_roundtrip(live_state):
    resp = ControlResponse(
        command_id='c1', success=True, state=live_state, session_inputs=live_state.latest_inputs
    )
    decoded = decode_control_response(pack('control.response', resp))
    assert decoded == resp
    assert decoded.state.results[0].memory.modality == 'markdown'


def test_failed_control_response_requires_error_code():
    resp = ControlResponse(command_id='c1', success=False, state=None, error_code='STALE_SESSION', error='vecchia')
    decoded = decode_control_response(pack('control.response', resp))
    assert decoded.error_code == 'STALE_SESSION'


def test_failed_control_response_without_code_is_rejected():
    resp = ControlResponse(command_id='c1', success=False, state=None, error='boh')
    with pytest.raises(ValueError):
        decode_control_response(pack('control.response', resp))


# --------------------------------------------------------------------------- validazione vettori


def test_validate_vector_rejects_wrong_dimension():
    with pytest.raises(ValueError, match='INVALID_VECTOR_DIMENSION'):
        validate_vector([1.0] * 10)


def test_validate_vector_rejects_nan_and_inf():
    with pytest.raises(ValueError, match='NON_FINITE_VECTOR'):
        validate_vector([float('nan')] + [0.0] * (EMBED_DIM - 1))
    with pytest.raises(ValueError, match='NON_FINITE_VECTOR'):
        validate_vector([float('inf')] + [0.0] * (EMBED_DIM - 1))


def test_validate_vector_rejects_null_norm():
    with pytest.raises(ValueError, match='INVALID_VECTOR_NORM'):
        validate_vector([0.0] * EMBED_DIM)


def test_validate_vector_rejects_denormalized():
    with pytest.raises(ValueError, match='INVALID_VECTOR_NORM'):
        validate_vector([2.0] + [0.0] * (EMBED_DIM - 1))


def test_validate_vector_accepts_unit_vector():
    validate_vector(unit_vector(3))


def test_validate_vector_rejects_non_numbers():
    with pytest.raises(ValueError, match='INVALID_REQUEST'):
        validate_vector(['x'] + [0.0] * (EMBED_DIM - 1))  # type: ignore[list-item]


# --------------------------------------------------------------------------- numeri e tipi espliciti


def test_json_nan_literal_is_rejected(scored, live_state):
    event = dataclasses.replace(live_state, results=[dataclasses.replace(scored, final_score=float('nan'))])
    with pytest.raises(ValueError, match='NON_FINITE'):
        pack('live.state', event)


def test_json_inf_literal_is_rejected(scored, live_state):
    event = dataclasses.replace(live_state, results=[dataclasses.replace(scored, z_score=float('inf'))])
    with pytest.raises(ValueError, match='NON_FINITE'):
        pack('live.state', event)


def test_wrong_type_sequence_is_rejected(live_state):
    payload = json.loads(pack('live.state', live_state))['payload']
    payload['sequence'] = 'quarantadue'
    raw = json.dumps({'schema_version': 1, 'type': 'live.state', 'payload': payload})
    with pytest.raises(ValueError, match='INVALID_REQUEST'):
        decode_live_state(raw)


def test_bool_for_string_is_rejected(live_state):
    payload = json.loads(pack('live.state', live_state))['payload']
    payload['session_id'] = True
    raw = json.dumps({'schema_version': 1, 'type': 'live.state', 'payload': payload})
    with pytest.raises(ValueError, match='INVALID_REQUEST'):
        decode_live_state(raw)


def test_string_for_bool_is_rejected():
    payload = {'command_id': 'c1', 'session_id': 's', 'action': 'set_mic', 'memory_id': None, 'enabled': 'yes'}
    raw = json.dumps({'schema_version': 1, 'type': 'hud.command', 'payload': payload})
    with pytest.raises(ValueError, match='INVALID_REQUEST'):
        decode_hud_command(raw)


def test_nan_in_source_last_seen_is_rejected_via_json_constant(live_state):
    payload = json.loads(pack('live.state', live_state))['payload']
    raw = json.dumps(
        {'schema_version': 1, 'type': 'live.state', 'payload': payload}
    ).replace('"emitted_at": 1792300010.0', '"emitted_at": NaN')
    with pytest.raises(ValueError, match='NON_FINITE'):
        decode_live_state(raw)


# --------------------------------------------------------------------------- memory view e identità


def test_memory_view_rejects_relative_file_path(memory_view):
    bad = dataclasses.replace(memory_view, file_path='relative/f01.md')
    event = ScoredMemory(memory=bad, cosine_score=0.1, z_score=0.1, synaptic_boost=0.0, final_score=0.1)
    state = LiveStateEvent(
        event_id='e', sequence=1, session_id='s', space_id='x', backend='mock', emitted_at=1.0,
        reason='input', latest_inputs=[], results=[event], pinned=[],
        sensor_status={}, dropped_inputs=0,
    )
    with pytest.raises(ValueError, match='INVALID_REQUEST'):
        decode_live_state(pack('live.state', state))


def test_memory_view_rejects_invalid_interval(video_memory, scored):
    bad = dataclasses.replace(video_memory, start_sec=85.0, end_sec=40.0)
    entry = dataclasses.replace(scored, memory=bad)
    state = LiveStateEvent(
        event_id='e', sequence=1, session_id='s', space_id='x', backend='mock', emitted_at=1.0,
        reason='input', latest_inputs=[], results=[entry], pinned=[],
        sensor_status={}, dropped_inputs=0,
    )
    with pytest.raises(ValueError, match='INVALID_REQUEST'):
        decode_live_state(pack('live.state', state))


def test_memory_view_rejects_page_zero(memory_view, scored):
    bad = dataclasses.replace(memory_view, page_num=0)
    entry = dataclasses.replace(scored, memory=bad)
    state = LiveStateEvent(
        event_id='e', sequence=1, session_id='s', space_id='x', backend='mock', emitted_at=1.0,
        reason='input', latest_inputs=[], results=[entry], pinned=[],
        sensor_status={}, dropped_inputs=0,
    )
    with pytest.raises(ValueError, match='INVALID_REQUEST'):
        decode_live_state(pack('live.state', state))


def test_memory_view_rejects_long_snippet(memory_view, scored):
    bad = dataclasses.replace(memory_view, snippet_text='x' * 351)
    entry = dataclasses.replace(scored, memory=bad)
    state = LiveStateEvent(
        event_id='e', sequence=1, session_id='s', space_id='x', backend='mock', emitted_at=1.0,
        reason='input', latest_inputs=[], results=[entry], pinned=[],
        sensor_status={}, dropped_inputs=0,
    )
    with pytest.raises(ValueError, match='INVALID_REQUEST'):
        decode_live_state(pack('live.state', state))


def test_make_memory_id_is_deterministic_and_path_sensitive():
    a = make_memory_id('/tmp/a.md', 'section:one', 'c' * 64)
    assert a == make_memory_id('/tmp/a.md', 'section:one', 'c' * 64)
    assert a != make_memory_id('/tmp/b.md', 'section:one', 'c' * 64)
    assert a != make_memory_id('/tmp/a.md', 'section:two', 'c' * 64)
    assert len(a) == 64


def test_space_spec_id_is_deterministic_and_sensitive():
    real = SpaceSpec(backend='real', model_id='google/embeddinggemma-2', revision='abc123')
    same = SpaceSpec(backend='real', model_id='google/embeddinggemma-2', revision='abc123')
    other_rev = SpaceSpec(backend='real', model_id='google/embeddinggemma-2', revision='def456')
    mock = SpaceSpec(backend='mock', model_id='semantic-hash-mock', revision='1')
    assert real.space_id == same.space_id
    assert real.space_id != other_rev.space_id
    assert real.space_id != mock.space_id
    assert len(real.space_id) == 64


# --------------------------------------------------------------------------- export DTO


def test_export_settings_roundtrip_and_default_off():
    settings = ExportSettings()
    assert settings.transcription == 'off'
    assert settings.context_max_chars == 24000
    assert settings.max_transcribe_sec == 300.0


def test_decode_health_response_strict():
    good = pack_health('health.response', {'ready': True, 'space_id': 'sp', 'backend': 'real', 'error': None})
    parsed = decode_health_response(good)
    assert parsed == {'ready': True, 'space_id': 'sp', 'backend': 'real', 'error': None}
    bad = json.dumps(
        {'schema_version': 1, 'type': 'health.response', 'payload': {'ready': True, 'space_id': 'sp'}}
    )
    with pytest.raises(ValueError, match='INVALID_REQUEST'):
        decode_health_response(bad)
    wrong_backend = json.dumps(
        {
            'schema_version': 1, 'type': 'health.response',
            'payload': {'ready': True, 'space_id': 'sp', 'backend': 'cloud', 'error': None},
        }
    )
    with pytest.raises(ValueError):
        decode_health_response(wrong_backend)


def test_search_settings_defaults():
    settings = SearchSettings()
    assert settings.top_k == 5
    assert settings.score_mode == 'zscore'
    assert settings.gamma == 0.6
    assert settings.activation_threshold == 0.5
    assert settings.min_cosine == 0.25
    assert settings.min_final == 1.0


def test_memory_record_roundtrip_via_embed_flow(memory_view):
    record = MemoryRecord(memory=memory_view, embedding=unit_vector(5))
    assert record.memory.id == memory_view.id
    validate_vector(record.embedding)


# --------------------------------------------------------------------------- F1: system_audio


class TestSystemAudioWire:
    """F1: system_audio come terza sorgente (estensione contratti 2026-10-25)."""

    @staticmethod
    def _event_with_input(payload: dict, sensor_status: dict | None = None) -> str:
        return json.dumps(
            {
                'schema_version': 1,
                'type': 'live.state',
                'payload': {
                    'event_id': 'ev-sys',
                    'sequence': 1,
                    'session_id': 'sess-1',
                    'space_id': '0'.rjust(64, '0'),
                    'backend': 'mock',
                    'emitted_at': 1792300000.0,
                    'reason': 'input',
                    'latest_inputs': [payload],
                    'results': [],
                    'pinned': [],
                    'sensor_status': sensor_status or {},
                    'source_last_seen': {},
                    'dropped_inputs': 0,
                },
            },
            ensure_ascii=False,
        )

    def test_system_audio_input_roundtrip(self):
        raw = self._event_with_input(
            {
                'id': 'in-sys',
                'session_id': 'sess-1',
                'source': 'system_audio',
                'occurred_at': 1792300000.0,
                'audio_path': '/tmp/sys-1.wav',
                'duration_sec': 2.0,
                'text': '',
            }
        )
        event = decode_live_state(raw)
        assert [item.source for item in event.latest_inputs] == ['system_audio']
        assert event.latest_inputs[0].audio_path == '/tmp/sys-1.wav'
        assert event.latest_inputs[0].duration_sec == 2.0

    @pytest.mark.parametrize('source', ['mic_audio', 'system_audio'])
    def test_audio_source_without_path_rejected(self, source):
        raw = self._event_with_input(
            {'id': 'in-x', 'session_id': 'sess-1', 'source': source, 'occurred_at': 1792300000.0}
        )
        with pytest.raises(ValueError, match=f'{source} senza audio_path'):
            decode_live_state(raw)

    @pytest.mark.parametrize('source', ['mic_audio', 'system_audio'])
    def test_audio_source_status_key_roundtrip(self, source):
        raw = self._event_with_input(
            {
                'id': 'in-y',
                'session_id': 'sess-1',
                'source': source,
                'occurred_at': 1792300000.0,
                'audio_path': f'/tmp/{source}.wav',
            },
            sensor_status={source: 'on'},
        )
        event = decode_live_state(raw)
        # la sorgente è ammessa anche come chiave di sensor_status
        assert event.sensor_status.get(source) == 'on'
        assert event.latest_inputs[0].source == source

    def test_audio_modality_roundtrip(self, live_state, scored, memory_view):
        audio_memory = dataclasses.replace(memory_view, modality='audio', file_path='/home/demo/watch_dir/f07.wav')
        event = dataclasses.replace(live_state, results=[dataclasses.replace(scored, memory=audio_memory)])
        decoded = decode_live_state(pack('live.state', event))
        assert decoded.results[0].memory.modality == 'audio'
