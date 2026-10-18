"""Contratti tecnici condivisi — Ambient Second Brain (contratto 1.0).

Materializzazione di untracked_doc/07_TECHNICAL_CONTRACTS.md: DTO, wire format
JSON, decoder pubblici e validazione. Ogni modifica passa dall'Orchestratore,
che aggiorna prima il documento e poi questo modulo, avvisando i consumatori.

Regole generali dei decoder:
- nessuna coercizione silenziosa: tipo sbagliato, campo sconosciuto, literal
  non ammesso, numero non finito, path relativo, intervallo invalido o lista
  oltre i limitiproducono ValueError esplicita;
- tutti i path nei DTO sono assoluti;
- gli orari sono secondi Unix UTC, gli intervalli media sono [start, end).
"""

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Literal

SCHEMA_VERSION = 1
EMBED_DIM = 256

Source = Literal['agent_text', 'mic_audio']
Modality = Literal['markdown', 'pdf', 'image', 'video']
Backend = Literal['real', 'mock']

SOURCE_VALUES = frozenset({'agent_text', 'mic_audio'})
MODALITY_VALUES = frozenset({'markdown', 'pdf', 'image', 'video'})
BACKEND_VALUES = frozenset({'real', 'mock'})

# Limiti wire espliciti (i token del testo sono verificati dal backend col tokenizer).
MAX_IMAGE_PATHS = 3
MAX_TEXT_CHARS = 200_000
MAX_CONTEXT_HEADER_CHARS = 512
MAX_SNIPPET_CHARS = 350
MAX_LATEST_INPUTS = 5
MAX_SNAPSHOT_INPUT_IDS = 3
MAX_PINNED = 12
MAX_RESULTS = 32
MAX_SESSION_INPUTS = 2048
MAX_SENSOR_STATUS_KEYS = 64
MAX_WARNINGS = 256


# --------------------------------------------------------------------------- DTO


@dataclass(frozen=True)
class SpaceSpec:
    backend: Backend
    model_id: str
    revision: str
    dimension: int = EMBED_DIM
    preprocessing: str = 'eg2-poc-v1'

    @property
    def space_id(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(raw.encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class RuntimePaths:
    run_dir: str
    embed_endpoint: str
    events_endpoint: str
    control_endpoint: str
    db_path: str
    watch_dir: str
    export_dir: str


@dataclass(frozen=True)
class SearchSettings:
    top_k: int = 5
    score_mode: Literal['cosine', 'zscore'] = 'zscore'
    gamma: float = 0.6
    activation_threshold: float = 0.5
    min_cosine: float = 0.25
    min_final: float = 1.0


@dataclass(frozen=True)
class EmbedRequest:
    request_id: str
    is_query: bool
    text_payload: str = ''
    context_header: str = ''
    image_paths: list[str] = field(default_factory=list)
    audio_path: str | None = None


@dataclass(frozen=True)
class EmbedResponse:
    request_id: str
    success: bool
    space_id: str
    embedding: list[float] = field(default_factory=list)
    error_code: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class MemoryView:
    id: str
    file_path: str
    modality: Modality
    chunk_key: str
    content_sha256: str
    source_sha256: str
    context_header: str
    snippet_text: str
    content_text: str
    page_num: int | None = None
    start_sec: float | None = None
    end_sec: float | None = None
    created_at: float = 0.0


@dataclass(frozen=True)
class MemoryRecord:
    memory: MemoryView
    embedding: list[float]


@dataclass(frozen=True)
class ScoredMemory:
    memory: MemoryView
    cosine_score: float
    z_score: float
    synaptic_boost: float
    final_score: float
    matched_sources: list[Source] = field(default_factory=list)
    winning_source: Source | None = None


@dataclass(frozen=True)
class LiveInput:
    id: str
    session_id: str
    source: Source
    occurred_at: float
    text: str = ''
    audio_path: str | None = None
    duration_sec: float | None = None


@dataclass(frozen=True)
class ContextSnapshot:
    id: str
    session_id: str
    captured_at: float
    input_ids: list[str]


@dataclass(frozen=True)
class PinnedMemory:
    memory: MemoryView
    pinned_at: float
    snapshot: ContextSnapshot


@dataclass(frozen=True)
class LiveStateEvent:
    event_id: str
    sequence: int
    session_id: str
    space_id: str
    backend: Backend
    emitted_at: float
    reason: Literal['input', 'pin', 'unpin', 'reset', 'index', 'status']
    latest_inputs: list[LiveInput]
    results: list[ScoredMemory]
    pinned: list[PinnedMemory]
    sensor_status: dict[str, str]
    source_last_seen: dict[str, float] = field(default_factory=dict)
    dropped_inputs: int = 0


@dataclass(frozen=True)
class HudCommand:
    command_id: str
    session_id: str | None
    action: Literal['pin', 'unpin', 'snapshot', 'new_session', 'set_mic']
    memory_id: str | None = None
    enabled: bool | None = None


@dataclass(frozen=True)
class ControlResponse:
    command_id: str
    success: bool
    state: LiveStateEvent | None
    session_inputs: list[LiveInput] = field(default_factory=list)
    lease_expires_at: float | None = None
    error_code: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class ExportSettings:
    transcription: Literal['off', 'local'] = 'off'
    asr_model_dir: str | None = None
    language: str = 'it'
    context_max_chars: int = 24000
    max_transcribe_sec: float = 300.0


@dataclass(frozen=True)
class ExportResult:
    export_dir: str
    markdown_path: str
    manifest_path: str
    clipboard_status: Literal['copied', 'unavailable', 'failed']
    warnings: list[str]


# --------------------------------------------------------------------------- wire format


def pack(kind: str, payload: object) -> str:
    """Serializza un DTO dataclass in envelope JSON. Rifiuta float non finiti."""
    try:
        return json.dumps(
            {'schema_version': SCHEMA_VERSION, 'type': kind, 'payload': asdict(payload)},
            ensure_ascii=False,
            allow_nan=False,
        )
    except ValueError as exc:
        raise ValueError(f'NON_FINITE: valore numerico non serializzabile in {kind}: {exc}') from exc


def _reject_constant(literal: str) -> float:
    raise ValueError(f'NON_FINITE: literal JSON non ammesso: {literal}')


def unpack(raw: str, expected_kind: str) -> dict:
    value = json.loads(raw, parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError('INVALID_ENVELOPE: radice non è un oggetto')
    if value.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('SCHEMA_VERSION_MISMATCH')
    if value.get('type') != expected_kind or not isinstance(value.get('payload'), dict):
        raise ValueError('INVALID_ENVELOPE')
    return value['payload']


def pack_health(kind: Literal['health.request', 'health.response'], payload: dict) -> str:
    """Envelope health: payload dict, non dataclass."""
    if kind not in ('health.request', 'health.response'):
        raise ValueError(f'INVALID_REQUEST: kind health non valido: {kind!r}')
    try:
        return json.dumps(
            {'schema_version': SCHEMA_VERSION, 'type': kind, 'payload': payload},
            ensure_ascii=False,
            allow_nan=False,
        )
    except ValueError as exc:
        raise ValueError(f'NON_FINITE: valore numerico non serializzabile in {kind}: {exc}') from exc


def decode_health_response(raw: str) -> dict:
    """Parser dedicato health.response: {ready, space_id, backend, error}."""
    payload = unpack(raw, 'health.response')
    _check_keys(payload, {'ready', 'space_id', 'backend', 'error'})
    for mandatory in ('ready', 'space_id', 'backend'):
        if mandatory not in payload:
            raise ValueError(f'INVALID_REQUEST: campo health mancante: {mandatory}')
    ready = _as_bool(payload['ready'], 'health.ready')
    space_id = _as_nonempty_str(payload['space_id'], 'health.space_id')
    backend = _as_literal(payload['backend'], BACKEND_VALUES, 'health.backend')
    error = payload['error']
    if error is not None:
        error = _as_str(error, 'health.error')
    return {'ready': ready, 'space_id': space_id, 'backend': backend, 'error': error}


# --------------------------------------------------------------------------- validazione vettori


def validate_vector(vector: list[float]) -> None:
    if not isinstance(vector, (list, tuple)):
        raise ValueError('INVALID_REQUEST: il vettore non è una lista')
    if len(vector) != EMBED_DIM:
        raise ValueError('INVALID_VECTOR_DIMENSION')
    for component in vector:
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise ValueError(f'INVALID_REQUEST: componente non numerica: {component!r}')
        if not math.isfinite(component):
            raise ValueError('NON_FINITE_VECTOR')
    norm = math.sqrt(sum(component * component for component in vector))
    if abs(norm - 1.0) > 1e-4:
        raise ValueError('INVALID_VECTOR_NORM')


# --------------------------------------------------------------------------- helper di decode


def _check_keys(payload: dict, allowed: set[str]) -> None:
    unknown = set(payload.keys()) - allowed
    if unknown:
        raise ValueError(f'UNKNOWN_FIELD: {sorted(unknown)}')


def _optional(payload: dict, name: str):
    return payload.get(name)


def _as_str(value, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f'INVALID_REQUEST: {field_name} non è una stringa: {type(value).__name__}')
    return value


def _as_nonempty_str(value, field_name: str) -> str:
    text = _as_str(value, field_name)
    if not text:
        raise ValueError(f'INVALID_REQUEST: {field_name} è vuoto')
    return text


def _as_bool(value, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f'INVALID_REQUEST: {field_name} non è booleano: {type(value).__name__}')
    return value


def _as_int(value, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f'INVALID_REQUEST: {field_name} non è intero: {type(value).__name__}')
    return value


def _as_float(value, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'INVALID_REQUEST: {field_name} non è numerico: {type(value).__name__}')
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f'NON_FINITE: {field_name} non è finito')
    return number


def _as_literal(value, allowed: frozenset, field_name: str) -> str:
    text = _as_str(value, field_name)
    if text not in allowed:
        raise ValueError(f'INVALID_REQUEST: {field_name} non ammesso: {text!r}')
    return text


def _as_abs_path(value, field_name: str) -> str:
    text = _as_nonempty_str(value, field_name)
    if not os.path.isabs(text):
        raise ValueError(f'INVALID_REQUEST: {field_name} non è un path assoluto: {text!r}')
    return text


def _as_list(value, field_name: str, max_len: int) -> list:
    if not isinstance(value, list):
        raise ValueError(f'INVALID_REQUEST: {field_name} non è una lista: {type(value).__name__}')
    if len(value) > max_len:
        raise ValueError(f'INVALID_REQUEST: {field_name} oltre il limite di {max_len} elementi')
    return value


def _as_str_map(value, field_name: str, max_len: int) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f'INVALID_REQUEST: {field_name} non è un oggetto: {type(value).__name__}')
    if len(value) > max_len:
        raise ValueError(f'INVALID_REQUEST: {field_name} oltre il limite di {max_len} chiavi')
    result = {}
    for key, item in value.items():
        result[_as_str(key, f'{field_name}[chiave]')] = _as_str(item, f'{field_name}[{key}]')
    return result


def _bounded_text(value, field_name: str, max_chars: int) -> str:
    text = _as_str(value, field_name)
    if len(text) > max_chars:
        raise ValueError(f'INVALID_REQUEST: {field_name} oltre {max_chars} caratteri')
    return text


def _abs_path_list(value, field_name: str, max_len: int) -> list[str]:
    items = _as_list(value, field_name, max_len)
    return [_as_abs_path(item, f'{field_name}[{index}]') for index, item in enumerate(items)]


def _float_field(payload: dict, name: str, default: float | None = None) -> float | None:
    if name not in payload or payload[name] is None:
        if default is None and name not in payload:
            raise ValueError(f'INVALID_REQUEST: campo mancante {name}')
        return default
    return _as_float(payload[name], name)


# --------------------------------------------------------------------------- decoder DTO annidati


def _decode_memory_view(payload) -> MemoryView:
    if not isinstance(payload, dict):
        raise ValueError('INVALID_REQUEST: memory non è un oggetto')
    _check_keys(payload, {f.name for f in fields(MemoryView)})
    page_raw = _optional(payload, 'page_num')
    page_num = None if page_raw is None else _as_int(page_raw, 'memory.page_num')
    if page_num is not None and page_num < 1:
        raise ValueError(f'INVALID_REQUEST: memory.page_num < 1: {page_num}')
    start_raw = _optional(payload, 'start_sec')
    end_raw = _optional(payload, 'end_sec')
    start_sec = None if start_raw is None else _as_float(start_raw, 'memory.start_sec')
    end_sec = None if end_raw is None else _as_float(end_raw, 'memory.end_sec')
    if (start_sec is None) != (end_sec is None):
        raise ValueError('INVALID_REQUEST: intervallo con un solo estremo')
    if start_sec is not None and end_sec is not None and not (0 <= start_sec < end_sec):
        raise ValueError(f'INVALID_REQUEST: intervallo invalido [{start_sec}, {end_sec})')
    return MemoryView(
        id=_as_nonempty_str(payload['id'], 'memory.id'),
        file_path=_as_abs_path(payload['file_path'], 'memory.file_path'),
        modality=_as_literal(payload['modality'], MODALITY_VALUES, 'memory.modality'),
        chunk_key=_as_nonempty_str(payload['chunk_key'], 'memory.chunk_key'),
        content_sha256=_as_nonempty_str(payload['content_sha256'], 'memory.content_sha256'),
        source_sha256=_as_nonempty_str(payload['source_sha256'], 'memory.source_sha256'),
        context_header=_bounded_text(payload['context_header'], 'memory.context_header', MAX_CONTEXT_HEADER_CHARS),
        snippet_text=_bounded_text(payload['snippet_text'], 'memory.snippet_text', MAX_SNIPPET_CHARS),
        content_text=_bounded_text(payload['content_text'], 'memory.content_text', MAX_TEXT_CHARS),
        page_num=page_num,
        start_sec=start_sec,
        end_sec=end_sec,
        created_at=_as_float(payload.get('created_at', 0.0), 'memory.created_at'),
    )


def _decode_scored_memory(payload) -> ScoredMemory:
    if not isinstance(payload, dict):
        raise ValueError('INVALID_REQUEST: risultato non è un oggetto')
    _check_keys(payload, {f.name for f in fields(ScoredMemory)})
    cosine = _as_float(payload['cosine_score'], 'cosine_score')
    if not -1.0 <= cosine <= 1.0:
        raise ValueError(f'INVALID_REQUEST: cosine_score fuori range [-1,1]: {cosine}')
    synaptic = _as_float(payload['synaptic_boost'], 'synaptic_boost')
    if synaptic < 0:
        raise ValueError(f'INVALID_REQUEST: synaptic_boost negativo: {synaptic}')
    matched_raw = _as_list(payload.get('matched_sources', []), 'matched_sources', len(SOURCE_VALUES))
    matched: list[Source] = []
    for item in matched_raw:
        source = _as_literal(item, SOURCE_VALUES, 'matched_sources')
        if source in matched:
            raise ValueError(f'INVALID_REQUEST: matched_sources duplicato: {source}')
        matched.append(source)
    winning_raw = _optional(payload, 'winning_source')
    winning = None if winning_raw is None else _as_literal(winning_raw, SOURCE_VALUES, 'winning_source')
    if winning is not None and winning not in matched:
        raise ValueError(f'INVALID_REQUEST: winning_source {winning} assente da matched_sources')
    return ScoredMemory(
        memory=_decode_memory_view(payload['memory']),
        cosine_score=cosine,
        z_score=_as_float(payload['z_score'], 'z_score'),
        synaptic_boost=synaptic,
        final_score=_as_float(payload['final_score'], 'final_score'),
        matched_sources=matched,
        winning_source=winning,
    )


def _decode_live_input(payload) -> LiveInput:
    if not isinstance(payload, dict):
        raise ValueError('INVALID_REQUEST: input non è un oggetto')
    _check_keys(payload, {f.name for f in fields(LiveInput)})
    source = _as_literal(payload['source'], SOURCE_VALUES, 'source')
    text = _bounded_text(payload.get('text', ''), 'text', MAX_TEXT_CHARS)
    audio_raw = _optional(payload, 'audio_path')
    audio_path = None if audio_raw is None else _as_abs_path(audio_raw, 'audio_path')
    duration_raw = _optional(payload, 'duration_sec')
    duration_sec = None if duration_raw is None else _as_float(duration_raw, 'duration_sec')
    if duration_sec is not None and duration_sec < 0:
        raise ValueError(f'INVALID_REQUEST: duration_sec negativa: {duration_sec}')
    if source == 'mic_audio' and audio_path is None:
        raise ValueError('INVALID_REQUEST: input mic_audio senza audio_path')
    if source == 'agent_text' and not text:
        raise ValueError('INVALID_REQUEST: input agent_text senza testo')
    return LiveInput(
        id=_as_nonempty_str(payload['id'], 'id'),
        session_id=_as_nonempty_str(payload['session_id'], 'session_id'),
        source=source,
        occurred_at=_as_float(payload['occurred_at'], 'occurred_at'),
        text=text,
        audio_path=audio_path,
        duration_sec=duration_sec,
    )


def _decode_context_snapshot(payload) -> ContextSnapshot:
    if not isinstance(payload, dict):
        raise ValueError('INVALID_REQUEST: snapshot non è un oggetto')
    _check_keys(payload, {f.name for f in fields(ContextSnapshot)})
    ids_raw = _as_list(payload['input_ids'], 'input_ids', MAX_SNAPSHOT_INPUT_IDS)
    input_ids = [_as_nonempty_str(item, 'input_ids[]') for item in ids_raw]
    return ContextSnapshot(
        id=_as_nonempty_str(payload['id'], 'snapshot.id'),
        session_id=_as_nonempty_str(payload['session_id'], 'snapshot.session_id'),
        captured_at=_as_float(payload['captured_at'], 'snapshot.captured_at'),
        input_ids=input_ids,
    )


def _decode_pinned(payload) -> PinnedMemory:
    if not isinstance(payload, dict):
        raise ValueError('INVALID_REQUEST: pin non è un oggetto')
    _check_keys(payload, {f.name for f in fields(PinnedMemory)})
    return PinnedMemory(
        memory=_decode_memory_view(payload['memory']),
        pinned_at=_as_float(payload['pinned_at'], 'pinned_at'),
        snapshot=_decode_context_snapshot(payload['snapshot']),
    )


def _decode_live_state_event(payload) -> LiveStateEvent:
    _check_keys(payload, {f.name for f in fields(LiveStateEvent)})
    latest_raw = _as_list(payload['latest_inputs'], 'latest_inputs', MAX_LATEST_INPUTS)
    results_raw = _as_list(payload['results'], 'results', MAX_RESULTS)
    pinned_raw = _as_list(payload['pinned'], 'pinned', MAX_PINNED)
    source_last_seen_raw = payload.get('source_last_seen', {})
    if not isinstance(source_last_seen_raw, dict):
        raise ValueError('INVALID_REQUEST: source_last_seen non è un oggetto')
    source_last_seen = {}
    for key, value in source_last_seen_raw.items():
        source = _as_literal(key, SOURCE_VALUES, 'source_last_seen[chiave]')
        seen = _as_float(value, f'source_last_seen[{key}]')
        if seen < 0:
            raise ValueError(f'INVALID_REQUEST: source_last_seen[{key}] negativo')
        source_last_seen[source] = seen
    dropped = _as_int(payload.get('dropped_inputs', 0), 'dropped_inputs')
    if dropped < 0:
        raise ValueError(f'INVALID_REQUEST: dropped_inputs negativo: {dropped}')
    sequence = _as_int(payload['sequence'], 'sequence')
    if sequence < 0:
        raise ValueError(f'INVALID_REQUEST: sequence negativa: {sequence}')
    return LiveStateEvent(
        event_id=_as_nonempty_str(payload['event_id'], 'event_id'),
        sequence=sequence,
        session_id=_as_nonempty_str(payload['session_id'], 'session_id'),
        space_id=_as_nonempty_str(payload['space_id'], 'space_id'),
        backend=_as_literal(payload['backend'], BACKEND_VALUES, 'backend'),
        emitted_at=_as_float(payload['emitted_at'], 'emitted_at'),
        reason=_as_literal(payload['reason'], frozenset({'input', 'pin', 'unpin', 'reset', 'index', 'status'}), 'reason'),
        latest_inputs=[_decode_live_input(item) for item in latest_raw],
        results=[_decode_scored_memory(item) for item in results_raw],
        pinned=[_decode_pinned(item) for item in pinned_raw],
        sensor_status=_as_str_map(payload['sensor_status'], 'sensor_status', MAX_SENSOR_STATUS_KEYS),
        source_last_seen=source_last_seen,
        dropped_inputs=dropped,
    )


# --------------------------------------------------------------------------- decoder pubblici


def decode_embed_request(raw: str) -> EmbedRequest:
    payload = unpack(raw, 'embed.request')
    _check_keys(payload, {f.name for f in fields(EmbedRequest)})
    is_query = _as_bool(payload['is_query'], 'is_query')
    text = _bounded_text(payload.get('text_payload', ''), 'text_payload', MAX_TEXT_CHARS)
    image_paths = _abs_path_list(payload.get('image_paths', []), 'image_paths', MAX_IMAGE_PATHS)
    audio_raw = _optional(payload, 'audio_path')
    audio_path = None if audio_raw is None else _as_abs_path(audio_raw, 'audio_path')
    if is_query and text and audio_path is not None:
        raise ValueError('INVALID_REQUEST: query con testo e audio insieme')
    if not (text or image_paths or audio_path is not None):
        raise ValueError('INVALID_REQUEST: payload embedding vuoto')
    return EmbedRequest(
        request_id=_as_nonempty_str(payload['request_id'], 'request_id'),
        is_query=is_query,
        text_payload=text,
        context_header=_bounded_text(payload.get('context_header', ''), 'context_header', MAX_CONTEXT_HEADER_CHARS),
        image_paths=image_paths,
        audio_path=audio_path,
    )


def decode_embed_response(raw: str) -> EmbedResponse:
    payload = unpack(raw, 'embed.response')
    _check_keys(payload, {f.name for f in fields(EmbedResponse)})
    success = _as_bool(payload['success'], 'success')
    space_id = _as_nonempty_str(payload['space_id'], 'space_id')
    request_id = _as_str(payload.get('request_id', ''), 'request_id')
    embedding_raw = payload.get('embedding', [])
    error_code_raw = _optional(payload, 'error_code')
    error_raw = _optional(payload, 'error')
    if success:
        if not isinstance(embedding_raw, list):
            raise ValueError('INVALID_REQUEST: embedding non è una lista')
        try:
            validate_vector(embedding_raw)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        if error_code_raw is not None:
            raise ValueError('INVALID_REQUEST: risposta di successo con error_code')
        return EmbedResponse(
            request_id=request_id, success=True, space_id=space_id, embedding=[float(x) for x in embedding_raw]
        )
    if embedding_raw:
        raise ValueError('INVALID_REQUEST: risposta di errore con embedding')
    if error_code_raw is None or not _as_str(error_code_raw, 'error_code'):
        raise ValueError('INVALID_REQUEST: risposta di errore senza error_code')
    error = None if error_raw is None else _as_str(error_raw, 'error')
    return EmbedResponse(
        request_id=request_id, success=False, space_id=space_id, embedding=[],
        error_code=error_code_raw, error=error,
    )


def decode_live_state(raw: str) -> LiveStateEvent:
    return _decode_live_state_event(unpack(raw, 'live.state'))


def decode_hud_command(raw: str) -> HudCommand:
    payload = unpack(raw, 'hud.command')
    _check_keys(payload, {f.name for f in fields(HudCommand)})
    action = _as_literal(
        payload['action'], frozenset({'pin', 'unpin', 'snapshot', 'new_session', 'set_mic'}), 'action'
    )
    session_raw = _optional(payload, 'session_id')
    session_id = None if session_raw is None else _as_nonempty_str(session_raw, 'session_id')
    memory_raw = _optional(payload, 'memory_id')
    memory_id = None if memory_raw is None else _as_nonempty_str(memory_raw, 'memory_id')
    enabled_raw = _optional(payload, 'enabled')
    enabled = None if enabled_raw is None else _as_bool(enabled_raw, 'enabled')
    if action in ('pin', 'unpin') and memory_id is None:
        raise ValueError(f'INVALID_REQUEST: {action} richiede memory_id')
    if action == 'set_mic' and enabled is None:
        raise ValueError('INVALID_REQUEST: set_mic richiede enabled')
    if action != 'snapshot' and session_id is None:
        raise ValueError(f'INVALID_REQUEST: {action} richiede session_id (solo snapshot accetta None)')
    if action not in ('pin', 'unpin') and memory_id is not None:
        raise ValueError(f'INVALID_REQUEST: memory_id non ammesso per {action}')
    if action != 'set_mic' and enabled is not None:
        raise ValueError(f'INVALID_REQUEST: enabled non ammesso per {action}')
    return HudCommand(
        command_id=_as_nonempty_str(payload['command_id'], 'command_id'),
        session_id=session_id,
        action=action,
        memory_id=memory_id,
        enabled=enabled,
    )


def decode_control_response(raw: str) -> ControlResponse:
    payload = unpack(raw, 'control.response')
    _check_keys(payload, {f.name for f in fields(ControlResponse)})
    success = _as_bool(payload['success'], 'success')
    state_raw = _optional(payload, 'state')
    state = None if state_raw is None else _decode_live_state_event(state_raw)
    inputs_raw = _as_list(payload.get('session_inputs', []), 'session_inputs', MAX_SESSION_INPUTS)
    session_inputs = [_decode_live_input(item) for item in inputs_raw]
    lease_raw = _optional(payload, 'lease_expires_at')
    lease = None if lease_raw is None else _as_float(lease_raw, 'lease_expires_at')
    error_code_raw = _optional(payload, 'error_code')
    error_raw = _optional(payload, 'error')
    if not success:
        if error_code_raw is None or not _as_str(error_code_raw, 'error_code'):
            raise ValueError('INVALID_REQUEST: risposta fallita senza error_code')
    elif error_code_raw is not None:
        raise ValueError('INVALID_REQUEST: risposta di successo con error_code')
    error = None if error_raw is None else _as_str(error_raw, 'error')
    return ControlResponse(
        command_id=_as_nonempty_str(payload['command_id'], 'command_id'),
        success=success,
        state=state,
        session_inputs=session_inputs,
        lease_expires_at=lease,
        error_code=error_code_raw,
        error=error,
    )


# --------------------------------------------------------------------------- identità


def make_memory_id(file_path: str, chunk_key: str, content_sha256: str) -> str:
    canonical = str(os.path.abspath(file_path))
    raw = chr(0).join((canonical, chunk_key, content_sha256))
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()
