"""Runner dei sensori — fusione, sessione, comandi e lifecycle (piano 05, contratti §5/§6).

Il runner è il proprietario dello stato: due EMA, last_seen, storia di sessione,
pin, contatori. I producer (testo/microfono) accodano input; un worker unico
esegue inferenza, ricerca e comandi, così nessuna connessione/socket è toccata
da due thread. Il thread controllo possiede il REP ma non lo stato.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path

from src.config import (
    DEFAULT_ALPHA,
    DEFAULT_MAX_PINS,
    DEFAULT_SESSION_AUDIO_MAX_BYTES,
    DEFAULT_SESSION_LIMIT_SEC,
    DEFAULT_SESSION_MAX_INPUTS,
    DEFAULT_SOURCE_TTL_SEC,
)
from src.contracts import (
    AUDIO_SOURCES,
    MAX_LATEST_INPUTS,
    MAX_SNAPSHOT_INPUT_IDS,
    SOURCE_VALUES,
    ContextSnapshot,
    ControlResponse,
    EmbedRequest,
    HudCommand,
    LiveInput,
    LiveStateEvent,
    MemoryRecord,
    MemoryView,
    PinnedMemory,
    RuntimePaths,
    ScoredMemory,
    SearchSettings,
    SpaceSpec,
    decode_hud_command,
    make_memory_id,
    pack,
)
from src.sensors.ema_tracker import EMATracker

SOURCE_ORDER = ('agent_text', 'mic_audio', 'system_audio')
CANDIDATE_MULTIPLIER = 4  # candidati richiesti a Storage per sorgente (piano L1)
MAX_COMMAND_CACHE = 256
DEBUG_SHOW_ALL = 'DEBUG'
LEASE_SEC = 120.0  # finestra concessa all'HUD dopo new_session (contratti §6)
# Chiavi di stato per le sorgenti audio nell'evento live (HUD): mic resta 'mic'
# per compat, la cattura di sistema è 'sysaudio' (F1).
SENSOR_STATUS_KEYS = {'mic_audio': 'mic', 'system_audio': 'sysaudio'}
# Etichette di strumento per il front-matter del documento promosso (F3).
TOOL_LABELS = {
    'agent_text': 'agente-cli',
    'mic_audio': 'microfono',
    'system_audio': 'audio-di-sistema',
}
PROMOTION_BEFORE = 5  # input di storia precedenti nel contesto della promozione
PROMOTION_AFTER = 2  # input successivi

# Ordine canonico delle sorgenti nei pareggi di fusione (contratti §5).


def combine_results(
    results_by_source: dict[str, list[ScoredMemory]],
    settings: SearchSettings,
    *,
    show_all: bool = False,
) -> list[ScoredMemory]:
    """Fusione deterministica fra sorgenti (piano L1, contratti §5).

    Unione per memory ID con i punteggi della sorgente vincente (parità:
    agent_text prima di mic_audio); matched_sources raccoglie tutte le sorgenti
    che hanno restituito il candidato. Le soglie visive si applicano DOPO
    l'unione, ai punteggi della sorgente vincente; ordine final/coseno/ID e
    limite top_k. ``show_all`` salta le soglie ed è marcato DEBUG dall'HUD.
    """
    best: dict[str, tuple[ScoredMemory, str]] = {}
    matched: dict[str, list[str]] = {}
    for source in SOURCE_ORDER:
        for candidate in results_by_source.get(source, ()):
            key = candidate.memory.id
            sources = matched.setdefault(key, [])
            if source not in sources:
                sources.append(source)
            current = best.get(key)
            if current is None or candidate.cosine_score > current[0].cosine_score:
                best[key] = (candidate, source)
    merged = [
        replace(candidate, matched_sources=matched[key], winning_source=source)
        for key, (candidate, source) in best.items()
    ]
    if not show_all:
        merged = [
            item
            for item in merged
            if (item.cosine_score >= settings.min_cosine or item.synaptic_boost > 0)
            and item.final_score >= settings.min_final
        ]
    merged.sort(key=lambda item: (-item.final_score, -item.cosine_score, item.memory.id))
    return merged[: settings.top_k]


class RunnerService:
    """Stato live di sessione: input, EMA, retrieval, pin e comandi (§4/§6).

    Classe testabile senza hardware: clock e dipendenze sono iniettabili. Un
    solo thread alla volta tocca i metodi pubblici (il worker del server).
    """

    def __init__(
        self,
        storage,
        embed_client,
        paths,
        settings: SearchSettings,
        *,
        space: SpaceSpec | None = None,
        clock=None,
        alpha: float = DEFAULT_ALPHA,
        source_ttl_sec: float = DEFAULT_SOURCE_TTL_SEC,
        max_pins: int = DEFAULT_MAX_PINS,
        session_limit_sec: float = DEFAULT_SESSION_LIMIT_SEC,
        session_max_inputs: int = DEFAULT_SESSION_MAX_INPUTS,
        session_audio_max_bytes: int = DEFAULT_SESSION_AUDIO_MAX_BYTES,
        audio_root: str | None = None,
        show_all: bool = False,
        sensor_enabled: Mapping[str, bool] | None = None,
        archive_sources: frozenset[str] | None = None,
        transcriber=None,
    ):
        self.storage = storage
        self.embed_client = embed_client
        self.paths = paths
        self.settings = settings
        resolved = space if space is not None else getattr(storage, 'space', None)
        if resolved is None:
            raise ValueError(
                'INVALID_REQUEST: SpaceSpec non disponibile: passare space= al RunnerService'
            )
        self.space = resolved
        self.space_id = resolved.space_id
        self._clock = clock if clock is not None else time.time
        self.alpha = alpha
        self.source_ttl_sec = source_ttl_sec
        self.max_pins = max_pins
        self.session_limit_sec = session_limit_sec
        self.session_max_inputs = session_max_inputs
        self.session_audio_max_bytes = session_audio_max_bytes
        self.audio_root = audio_root or paths.run_dir
        self.show_all = show_all

        self.session_id = uuid.uuid4().hex
        self._trackers: dict[str, EMATracker] = {s: EMATracker(alpha=alpha) for s in SOURCE_ORDER}
        self._last_seen: dict[str, float] = {}
        self._history: list[LiveInput] = []
        self._pinned: list[PinnedMemory] = []
        self._sequence = 0
        self._sensor_status: dict[str, str] = {}
        self._dropped_inputs = 0
        # Stato on/off per sorgente (F1): default tutto attivo, override via
        # sensor_enabled (es. {'system_audio': False} senza --system-audio).
        self._enabled: dict[str, bool] = {source: True for source in SOURCE_ORDER}
        for source, enabled in (sensor_enabled or {}).items():
            if source not in self._enabled:
                raise ValueError(f'INVALID_REQUEST: sorgente non ammessa: {source!r}')
            self._enabled[source] = bool(enabled)
        # Sorgenti i cui input diventano ricordi di sessione indicizzati (F3):
        # default vuoto = nessuna archiviazione (comportamento pre-F3).
        self.archive_sources = frozenset(archive_sources or ())
        for source in self.archive_sources:
            if source not in SOURCE_VALUES:
                raise ValueError(f'INVALID_REQUEST: sorgente non ammessa: {source!r}')
        # Trascrizione live (F4): oggetto col protocollo .transcribe(wav_path)->str
        # (LocalTranscriber); None = off, gli input audio restano senza testo.
        self.transcriber = transcriber
        # Note esterne persistenti nel sensor_status (es. conteggio tail):
        # sopravvivono al reset di sessione, a differenza di _sensor_status.
        self._sensor_notes: dict[str, str] = {}
        self._session_suspended = False
        self._audio_bytes = 0
        self._started_at = self._clock()
        self._command_cache: OrderedDict[str, ControlResponse] = OrderedDict()
        self._lease_expires_at: float | None = None
        self._pending_cleanup: list[tuple[str, float]] = []

    # ------------------------------------------------------------------ input

    def ingest(self, item: LiveInput) -> LiveStateEvent:
        """Accetta un input live: storia, inferenza, EMA e retrieval (§4)."""
        self._validate_input(item)
        if item.session_id != self.session_id:
            raise ValueError(f'STALE_SESSION: input di sessione {item.session_id[:8]}…')
        if self._session_full(item):
            self._session_suspended = True
            raise ValueError('SESSION_FULL: limiti di sessione raggiunti, input sospesi')
        # Trascrizione live sincrona nel worker (F4): il transcript sostituisce
        # il testo dell'input PRIMA dell'append in storia.
        item = self._transcribe_live(item)
        # La storia registra l'input PRIMA dell'inferenza: resta esportabile
        # anche se il modello fallisce (piano L1).
        self._history.append(item)
        self._register_audio(item)
        for vector in self._embed_queries(item):
            try:
                self._trackers[item.source].update(vector)
            except ValueError as exc:
                self._sensor_status[f'{item.source}_error'] = str(exc)
            else:
                self._last_seen[item.source] = self._now()
                self._sensor_status.pop(f'{item.source}_error', None)
        # Embedding fallito: EMA/last_seen/ranking intatti, solo diagnostica.
        if item.source in self.archive_sources:
            # L'archiviazione è solo diagnostica: un fallimento non tocca la pipeline.
            try:
                self._index_session_memory(item)
            except Exception as exc:  # noqa: BLE001 — errore archivio: diagnostica (F3)
                code = getattr(exc, 'code', None) or str(exc) or type(exc).__name__
                self._sensor_status['archive_error'] = code
            else:
                self._sensor_status.pop('archive_error', None)
        return self._build_event('input')

    def status_event(self, reason: str = 'status') -> LiveStateEvent:
        """Heartbeat: TTL ricalcolato senza bisogno di un nuovo input (piano L1).

        ``reason='index'`` è la variante per un nuovo commit dell'indice: la
        ricerca viene ricalcolata senza alterare le EMA (contratti §6).
        """
        self._sweep_expired_cleanup()
        return self._build_event(reason)

    # ------------------------------------------------------------------ comandi

    def handle(self, cmd: HudCommand) -> ControlResponse:
        """Gestione idempotente: lo stesso command_id usa la risposta memorizzata (§6)."""
        cached = self._command_cache.get(cmd.command_id)
        if cached is not None:
            return cached
        response = self._dispatch(cmd)
        self._remember_response(cmd.command_id, response)
        return response

    def _dispatch(self, cmd: HudCommand) -> ControlResponse:
        if cmd.action == 'snapshot' and cmd.session_id is None:
            return self._snapshot_response(cmd)  # discovery: unica azione con session_id None
        if cmd.session_id != self.session_id:
            return self._error_response(cmd, 'STALE_SESSION', f'sessione non corrente: {cmd.session_id}')
        if cmd.action == 'snapshot':
            return self._snapshot_response(cmd)
        if cmd.action == 'pin':
            return self._pin(cmd)
        if cmd.action == 'unpin':
            return self._unpin(cmd)
        if cmd.action == 'new_session':
            return self._new_session(cmd)
        if cmd.action == 'set_mic':
            return self._set_mic(cmd)
        if cmd.action == 'promote':
            return self._promote(cmd)
        return self._error_response(cmd, 'INVALID_REQUEST', f'azione non gestita: {cmd.action}')

    def _error_response(self, cmd: HudCommand, code: str, message: str) -> ControlResponse:
        return ControlResponse(command_id=cmd.command_id, success=False, state=None,
                               error_code=code, error=message)

    def _snapshot_response(self, cmd: HudCommand) -> ControlResponse:
        self._sweep_expired_cleanup()
        state = self._build_event('status')
        return ControlResponse(
            command_id=cmd.command_id,
            success=True,
            state=state,
            session_inputs=list(self._history),
            lease_expires_at=self._active_lease(),
        )

    def _pin(self, cmd: HudCommand) -> ControlResponse:
        now = self._now()
        memory = self.storage.get_memory(cmd.memory_id)
        if memory is None:
            return self._error_response(cmd, 'MEMORY_NOT_FOUND', f'memoria assente: {cmd.memory_id}')
        existing = [p for p in self._pinned if p.memory.id == cmd.memory_id]
        if existing:
            if self.storage.get_memory(existing[0].memory.id) is not None:
                # Doppio pin: idempotente, nessun nuovo rinforzo (contratti §5).
                return ControlResponse(command_id=cmd.command_id, success=True,
                                       state=self._build_event('pin'))
            self._pinned = [p for p in self._pinned if p.memory.id != cmd.memory_id]
        survivors = [
            p for p in self._pinned if self.storage.get_memory(p.memory.id) is not None
        ]
        if len(survivors) >= self.max_pins:
            return self._error_response(cmd, 'PIN_LIMIT', f'massimo {self.max_pins} pin attivi')
        others = [p.memory.id for p in survivors]
        try:
            if others:
                self.storage.reinforce_synapses(cmd.memory_id, others)
        except Exception as exc:  # noqa: BLE001 — il DB può fallire: il pin non viene aggiunto
            return self._error_response(cmd, 'STORAGE_ERROR', f'rinforzo fallito: {exc}')
        snapshot = ContextSnapshot(
            id=uuid.uuid4().hex,
            session_id=self.session_id,
            captured_at=now,
            input_ids=[item.id for item in self._history[-MAX_SNAPSHOT_INPUT_IDS:]],
        )
        self._pinned.append(PinnedMemory(memory=memory, pinned_at=now, snapshot=snapshot))
        return ControlResponse(command_id=cmd.command_id, success=True,
                               state=self._build_event('pin'))

    def _unpin(self, cmd: HudCommand) -> ControlResponse:
        # Nessuna chiamata a Storage: gli archi appresi restano (contratti §5).
        self._pinned = [p for p in self._pinned if p.memory.id != cmd.memory_id]
        return ControlResponse(command_id=cmd.command_id, success=True,
                               state=self._build_event('unpin'))

    def _new_session(self, cmd: HudCommand) -> ControlResponse:
        now = self._now()
        # Il purge dei ricordi di sessione precede il reset: un fallimento è
        # STORAGE_ERROR e blocca il reset, senza perdita silenziosa (F3).
        try:
            self.storage.purge_session_memories(self.session_id)
        except Exception as exc:  # noqa: BLE001 — il DB può fallire: reset bloccato
            return self._error_response(cmd, 'STORAGE_ERROR',
                                        f'purge ricordi di sessione fallito: {exc}')
        old_audio_dir = self._audio_dir()
        if os.path.isdir(old_audio_dir):
            # Il reset rinvia la rimozione della vecchia directory audio alla
            # scadenza del lease (contratti §6).
            self._pending_cleanup.append((old_audio_dir, now + LEASE_SEC))
        self._lease_expires_at = now + LEASE_SEC
        self.session_id = uuid.uuid4().hex
        self._trackers = {s: EMATracker(alpha=self.alpha) for s in SOURCE_ORDER}
        self._last_seen.clear()
        self._history.clear()
        self._pinned.clear()
        self._sensor_status.clear()
        self._session_suspended = False
        self._audio_bytes = 0
        self._started_at = now
        # sequence resta crescente per tutta l'esecuzione, incluso reset (§6).
        return ControlResponse(command_id=cmd.command_id, success=True,
                               state=self._build_event('reset'),
                               lease_expires_at=self._lease_expires_at)

    def _set_mic(self, cmd: HudCommand) -> ControlResponse:
        self._enabled['mic_audio'] = bool(cmd.enabled)
        return ControlResponse(command_id=cmd.command_id, success=True,
                               state=self._build_event('status'))

    def _promote(self, cmd: HudCommand) -> ControlResponse:
        """Promuove un ricordo di sessione a documento di contesto (F3).

        Scrive il documento in watch_dir (front-matter, momento, contesto e
        WAV allegati) e marca il ricordo come promosso. Idempotente: un
        ricordo già promosso non viene riscritto. Una memoria fuori registry
        (ricordo di archivio) è NOT_SESSION_MEMORY.
        """
        memory = self.storage.get_memory(cmd.memory_id)
        if memory is None:
            return self._error_response(cmd, 'MEMORY_NOT_FOUND', f'memoria assente: {cmd.memory_id}')
        entry = self.storage.get_session_memory(cmd.memory_id)
        if entry is None:
            return self._error_response(cmd, 'NOT_SESSION_MEMORY',
                                        f'{cmd.memory_id[:12]}… è un ricordo di archivio')
        if not entry.get('promoted'):
            try:
                self._write_promotion(cmd.memory_id, memory)
            except Exception as exc:  # noqa: BLE001 — documento non scritto: non marcato
                return self._error_response(cmd, 'PROMOTE_FAILED', f'scritta promozione fallita: {exc}')
            try:
                self.storage.mark_session_memory_promoted(cmd.memory_id)
            except Exception as exc:  # noqa: BLE001 — il DB può fallire
                return self._error_response(cmd, 'STORAGE_ERROR', f'marca promozione fallita: {exc}')
        return ControlResponse(command_id=cmd.command_id, success=True,
                               state=self._build_event('status'))

    def _write_promotion(self, memory_id: str, memory: MemoryView) -> str:
        """Documento di contesto della promozione in watch_dir (F3).

        Front-matter (sessione/sorgente/strumento), momento completo del
        ricordo promosso, contesto circostante dalla storia e WAV allegati
        in assets/. Ritorna il path del documento; qualsiasi errore di
        scrittura si propaga (il ricordo resta da promuovere).
        """
        source_input = self._input_for_session_memory(memory)
        center_index = (
            self._history.index(source_input) if source_input is not None else len(self._history) - 1
        )
        window = self._history[max(0, center_index - PROMOTION_BEFORE):
                               min(len(self._history), center_index + PROMOTION_AFTER + 1)]
        directory = Path(self.paths.watch_dir) / f'promozione-{self.session_id[:8]}-{memory_id[:12]}'
        assets = directory / 'assets'
        assets.mkdir(parents=True, exist_ok=True)
        attachments: list[tuple[LiveInput, str]] = []
        for item in window:
            if item.source in AUDIO_SOURCES and item.audio_path and os.path.isfile(item.audio_path):
                rel = f'assets/{item.id}.wav'
                shutil.copyfile(item.audio_path, directory / rel)
                attachments.append((item, rel))
        sources = sorted({item.source for item in window})
        tools = sorted({TOOL_LABELS.get(item.source, item.source) for item in window})
        pin = next((p for p in self._pinned if p.memory.id == memory_id), None)
        lines = [
            '---',
            f'sessione: {self.session_id}',
            f'sorgente: {", ".join(sources)}',
            f'strumento: {", ".join(tools)}',
            f'memory_id: {memory_id}',
        ]
        if pin is not None:
            lines.append(f'pinnato_a: {pin.pinned_at:.0f}')
        lines += [
            '---',
            '',
            f'# Momento promosso: {memory.context_header}',
            '',
            memory.content_text or '(contenuto non testuale)',
            '',
            '## Contesto circostante',
            '',
        ]
        for item in window:
            if item.source in AUDIO_SOURCES:
                rel = next((rel for attached, rel in attachments if attached.id == item.id), None)
                line = f'- `{item.source}` ({(item.duration_sec or 0.0):.1f}s alle {item.occurred_at:.0f})'
                if rel:
                    line += f' — audio: [{rel}]({rel})'
                lines.append(line)
            else:
                lines.append(f'- `agent_text` {item.text}')
        path = directory / 'contesto.md'
        path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        return str(path)

    def _input_for_session_memory(self, memory: MemoryView) -> LiveInput | None:
        """L'input di storia che ha generato il ricordo di sessione (F3)."""
        for item in self._history:
            if item.source == 'agent_text':
                if memory.file_path.endswith(f'agent-{item.id}.md'):
                    return item
            elif item.audio_path and os.path.realpath(item.audio_path) == os.path.realpath(memory.file_path):
                return item
        return None

    def _index_session_memory(self, item: LiveInput) -> str:
        """Indicizza un input come ricordo di sessione cercabile (F3).

        File reale sotto run_dir/<session>/: il testo agente diventa un .md,
        l'audio riusa il WAV già scritto dal sensore. Embedding da documento
        (is_query=False), MemoryView con session_scope=True, poi replace_file
        e registro di sessione. Gli errori sono del chiamante (diagnostica).
        """
        session_dir = os.path.join(self.audio_root, self.session_id)
        os.makedirs(session_dir, exist_ok=True)
        if item.source == 'agent_text':
            file_path = os.path.join(session_dir, f'agent-{item.id}.md')
            with open(file_path, 'w', encoding='utf-8') as handle:
                handle.write(f'# Ricordo di sessione\n\n{item.text}\n')
            text_payload, audio_path = item.text, None
        else:
            file_path = item.audio_path or ''
            if item.text:
                # transcript disponibile (F4): il documento si embedda dal testo
                text_payload, audio_path = item.text, None
            else:
                text_payload, audio_path = '', item.audio_path
        request = EmbedRequest(
            request_id=uuid.uuid4().hex,
            is_query=False,
            text_payload=text_payload,
            audio_path=audio_path,
        )
        vector = self.embed_client.embed(request)
        content_sha = _sha256_file(file_path)
        canonical = str(Path(file_path).resolve())
        memory = MemoryView(
            id=make_memory_id(file_path, f'session:{item.source}', content_sha),
            file_path=canonical,
            modality='markdown' if item.source == 'agent_text' else 'audio',
            chunk_key=f'session:{item.source}',
            content_sha256=content_sha,
            source_sha256=content_sha,
            context_header=f'ricordo di sessione {self.session_id[:8]} ({item.source})',
            snippet_text=(item.text or f'audio {(item.duration_sec or 0.0):.1f}s')[:350],
            content_text=item.text or '',
            created_at=self._now(),
            session_scope=True,
        )
        self.storage.replace_file(file_path, content_sha,
                                  [MemoryRecord(memory=memory, embedding=vector)])
        self.storage.register_session_memories(self.session_id, [(memory.id, canonical)])
        return memory.id

    # ------------------------------------------------------------------ interno

    def _now(self) -> float:
        return self._clock()

    def _validate_input(self, item: LiveInput) -> None:
        if item.source not in SOURCE_VALUES:
            raise ValueError(f'INVALID_REQUEST: sorgente non ammessa: {item.source!r}')
        if not item.session_id:
            raise ValueError('INVALID_REQUEST: session_id vuoto')
        if item.source == 'agent_text' and not item.text:
            raise ValueError('INVALID_REQUEST: input agent_text senza testo')
        if item.source in AUDIO_SOURCES:
            if not item.audio_path:
                raise ValueError(f'INVALID_REQUEST: input {item.source} senza audio_path')
            if not os.path.isabs(item.audio_path):
                raise ValueError(f'INVALID_REQUEST: audio_path relativo: {item.audio_path}')

    def _session_full(self, item: LiveInput) -> bool:
        now = self._now()
        if now - self._started_at > self.session_limit_sec:
            return True
        if len(self._history) >= self.session_max_inputs:
            return True
        incoming = self._audio_size(item)
        return self._audio_bytes + incoming > self.session_audio_max_bytes

    def _register_audio(self, item: LiveInput) -> None:
        if item.source in AUDIO_SOURCES:
            self._audio_bytes += self._audio_size(item)

    @staticmethod
    def _audio_size(item: LiveInput) -> int:
        if item.source not in AUDIO_SOURCES or not item.audio_path:
            return 0
        try:
            return os.path.getsize(item.audio_path)
        except OSError:
            return int((item.duration_sec or 0.0) * 32000)  # PCM 16 kHz mono int16

    def _transcribe_live(self, item: LiveInput) -> LiveInput:
        """Trascrive un input audio muto se un transcriber è configurato (F4).

        Sincrono nel worker; un fallimento è diagnostica (transcribe_error) e
        l'input prosegue audio-only. Senza transcriber, per il testo già
        presente o senza audio_path: nessun cambiamento.
        """
        if self.transcriber is None or item.source not in AUDIO_SOURCES or item.text:
            return item
        if not item.audio_path:
            return item
        try:
            transcript = str(self.transcriber.transcribe(item.audio_path)).strip()
        except Exception as exc:  # noqa: BLE001 — l'ASR può fallire: si prosegue audio-only
            code = getattr(exc, 'code', None) or str(exc) or type(exc).__name__
            self._sensor_status['transcribe_error'] = code
            return item
        if not transcript:
            return item
        self._sensor_status.pop('transcribe_error', None)
        return replace(item, text=transcript)

    def _embed_queries(self, item: LiveInput) -> list[list[float]]:
        """Vettori query dell'input: audio E testo separati se trascritto (F4).

        Un input audio con transcript produce due query (prima l'audio, poi
        il testo), ciascuna con il proprio update EMA; i fallimenti sono
        per-query diagnostica e i vettori riusciti aggiornano comunque l'EMA.
        """
        if item.source in AUDIO_SOURCES:
            queries = [(item.audio_path, '')]
            if item.text:
                queries.append((None, item.text))
        else:
            queries = [(None, item.text or '')]
        vectors = []
        for audio_path, text_payload in queries:
            vector = self._embed_one(item.source, audio_path=audio_path, text_payload=text_payload)
            if vector is not None:
                vectors.append(vector)
        return vectors

    def _embed_one(self, source: str, *, audio_path: str | None, text_payload: str) -> list[float] | None:
        request = EmbedRequest(
            request_id=uuid.uuid4().hex,
            is_query=True,
            text_payload=text_payload,
            audio_path=audio_path,
        )
        try:
            return self.embed_client.embed(request)
        except Exception as exc:  # noqa: BLE001 — ogni fallimento inferenza è diagnostica
            code = getattr(exc, 'code', None) or str(exc) or type(exc).__name__
            self._sensor_status[f'{source}_error'] = code
            return None

    def _active_sources(self, now: float) -> list[str]:
        return [
            source
            for source in SOURCE_ORDER
            if source in self._last_seen and now - self._last_seen[source] <= self.source_ttl_sec
        ]

    def _retrieval(self, now: float) -> list[ScoredMemory]:
        active = self._active_sources(now)
        if not active:
            return []
        results_by_source: dict[str, list[ScoredMemory]] = {}
        for source in active:
            query = self._trackers[source].state
            if query is None:
                continue
            try:
                results_by_source[source] = self.storage.hybrid_search(
                    query, replace(self.settings, top_k=self.settings.top_k * CANDIDATE_MULTIPLIER)
                )
            except Exception as exc:  # noqa: BLE001 — la ricerca può fallire per sorgente
                self._sensor_status['search_error'] = str(exc) or type(exc).__name__
        return combine_results(results_by_source, self.settings, show_all=self.show_all)

    def _build_event(self, reason: str) -> LiveStateEvent:
        now = self._now()
        self._sequence += 1
        status = dict(self._sensor_status)
        status.update(self._sensor_notes)
        for source in SOURCE_ORDER:
            key = SENSOR_STATUS_KEYS.get(source)
            if key is not None:
                status[key] = 'on' if self._enabled[source] else 'off'
        if self.show_all:
            status['show_all'] = DEBUG_SHOW_ALL
        if self._session_suspended:
            status['session'] = 'full'
        return LiveStateEvent(
            event_id=uuid.uuid4().hex,
            sequence=self._sequence,
            session_id=self.session_id,
            space_id=self.space_id,
            backend=self.space.backend,
            emitted_at=now,
            reason=reason,  # type: ignore[argtype]
            latest_inputs=list(self._history[-MAX_LATEST_INPUTS:]),
            results=self._retrieval(now),
            pinned=list(self._pinned),
            sensor_status=status,
            source_last_seen=dict(self._last_seen),
            dropped_inputs=self._dropped_inputs,
        )

    def _remember_response(self, command_id: str, response: ControlResponse) -> None:
        self._command_cache[command_id] = response
        self._command_cache.move_to_end(command_id)
        while len(self._command_cache) > MAX_COMMAND_CACHE:
            self._command_cache.popitem(last=False)

    def _active_lease(self) -> float | None:
        if self._lease_expires_at is not None and self._now() >= self._lease_expires_at:
            return None
        return self._lease_expires_at

    def _sweep_expired_cleanup(self) -> None:
        now = self._now()
        remaining = []
        for path, expires_at in self._pending_cleanup:
            if now >= expires_at:
                shutil.rmtree(path, ignore_errors=True)
            else:
                remaining.append((path, expires_at))
        self._pending_cleanup = remaining
        if self._lease_expires_at is not None and now >= self._lease_expires_at:
            self._lease_expires_at = None

    def _audio_dir(self) -> str:
        return os.path.join(self.audio_root, self.session_id)

    # ------------------------------------------------------------------ per il server

    def note_dropped_input(self) -> None:
        """Il worker registra lo scarto per coda piena (contratti §6)."""
        self._dropped_inputs += 1

    def note_sensor(self, key: str, value: str) -> None:
        """Annotazione esterna nel sensor_status (es. 'agent_logs': numero tail).

        Sopravvive al reset di sessione; le chiavi per sorgente (mic/sysaudio)
        restano calcolate dallo stato interno in _build_event.
        """
        self._sensor_notes[key] = value

    def sensor_enabled(self, source: str) -> bool:
        """Stato on/off di una sorgente: i callback di cattura lo consultano."""
        return self._enabled.get(source, False)

    @property
    def mic_enabled(self) -> bool:
        return self._enabled['mic_audio']  # compat/replay (piano L3)

    @property
    def session_suspended(self) -> bool:
        return self._session_suspended


class LiveInputQueue:
    """Coda live limitata fra producer e worker (contratti §6).

    Massimo ``maxsize`` input (8 nel PoC); quando piena scarta l'input
    pendente più vecchio, incrementa il contatore e delega la pulizia del
    suo WAV al callback (il file non è referenziato: non è mai entrato in
    storia). L'ordine FIFO si conserva.
    """

    def __init__(self, maxsize: int = 8):
        if maxsize < 1:
            raise ValueError(f'INVALID_REQUEST: maxsize non valido: {maxsize}')
        self._maxsize = maxsize
        self._items: list[LiveInput] = []
        self.dropped = 0

    def put(self, item: LiveInput, on_drop: Callable[[LiveInput], None] | None = None) -> bool:
        """Accoda un input; False se la pressione ha scartato il più vecchio."""
        dropped = None
        if len(self._items) >= self._maxsize:
            dropped = self._items.pop(0)
            self.dropped += 1
        self._items.append(item)
        if dropped is not None:
            if on_drop is not None:
                on_drop(dropped)
            else:
                _delete_unreferenced_wav(dropped)
            return False
        return True

    def drain(self) -> list[LiveInput]:
        """Svuota la coda in ordine FIFO (il worker la consuma a lotto)."""
        items, self._items = self._items, []
        return items

    def __len__(self) -> int:
        return len(self._items)


def _delete_unreferenced_wav(item: LiveInput) -> None:
    """Elimina il WAV di un input scartato: non referenziato da storia o pin."""
    if item.source in AUDIO_SOURCES and item.audio_path:
        try:
            os.unlink(item.audio_path)
        except OSError:
            pass


def _sha256_file(path: str) -> str:
    """Digest del contenuto di un file, a blocchi (identità dei ricordi F3)."""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


class RunnerServer:
    """Collega RunnerService ai socket: PUB eventi, REP controllo, coda input.

    Il thread controllo possiede il REP ma non lo stato: inoltra i comandi
    alla coda e attende la risposta. Il worker unico possiede client
    inferenza, storage e stato; i comandi hanno priorità sui nuovi input fra
    un'inferenza e l'altra (contratti §6). Ogni pubblicazione è uno snapshot
    completo; heartbeat e index_generation ogni ``heartbeat_sec`` best effort.
    """

    def __init__(
        self,
        service: RunnerService,
        context=None,
        *,
        heartbeat_sec: float = 2.0,
        command_timeout_sec: float = 2.0,
        max_queue: int = 8,
    ):
        import queue as queue_mod
        import threading

        import zmq

        self._queue_mod = queue_mod
        self._threading = threading
        self._zmq = zmq
        self.service = service
        self.queue = LiveInputQueue(maxsize=max_queue)
        self.heartbeat_sec = heartbeat_sec
        self.command_timeout_sec = command_timeout_sec
        self._command_queue: 'queue_mod.Queue' = queue_mod.Queue()
        self._stop = threading.Event()
        self._last_heartbeat = 0.0
        self._last_generation: int | None = None
        self.last_event: LiveStateEvent | None = None
        self._worker: threading.Thread | None = None
        self._control: threading.Thread | None = None

    # ------------------------------------------------------------------ API

    def start(self) -> None:
        self._stop.clear()
        self._worker = self._threading.Thread(target=self._worker_loop, name='sensors-worker', daemon=True)
        self._control = self._threading.Thread(target=self._control_loop, name='sensors-control', daemon=True)
        self._worker.start()
        self._control.start()

    def close(self) -> None:
        self._stop.set()
        for thread in (self._worker, self._control):
            if thread is not None:
                thread.join(timeout=3)
        self._worker = None
        self._control = None

    def submit_async(self, cmd: HudCommand) -> dict:
        """Accoda un comando; il worker riempe 'response' e alza 'done'."""
        slot = {'done': self._threading.Event(), 'response': None}
        self._command_queue.put((cmd, slot))
        return slot

    def submit_command(self, cmd: HudCommand) -> ControlResponse:
        """Invia un comando e attende; al timeout risponde TIMEOUT senza invalidare."""
        slot = self.submit_async(cmd)
        if not slot['done'].wait(self.command_timeout_sec):
            return ControlResponse(
                command_id=cmd.command_id, success=False, state=None,
                error_code='TIMEOUT', error='worker non ha risposto in tempo',
            )
        return slot['response']

    def step(self) -> bool:
        """Un giro del worker: prima i comandi, poi gli input. Ritorna True se attivo."""
        worked = False
        while True:
            try:
                cmd, slot = self._command_queue.get_nowait()
            except self._queue_mod.Empty:
                break
            slot['response'] = self._execute(cmd)
            slot['done'].set()
            worked = True
        for item in self.queue.drain():
            try:
                event = self.service.ingest(item)
            except ValueError as exc:
                self._log_rejection(item, exc)
                continue
            self.last_event = event
            self._publish(event)
            worked = True
        return worked

    # ------------------------------------------------------------------ thread

    def _worker_loop(self) -> None:
        import time as time_mod

        pub = self._zmq.Context.instance().socket(self._zmq.PUB)
        pub.setsockopt(self._zmq.LINGER, 0)
        pub.bind(self.service.paths.events_endpoint)
        self._pub = pub
        try:
            while not self._stop.is_set():
                worked = self.step()
                now = time_mod.time()
                if now - self._last_heartbeat >= self.heartbeat_sec:
                    self._last_heartbeat = now
                    self._heartbeat()
                if not worked:
                    self._stop.wait(0.02)
        finally:
            pub.close(0)

    def _heartbeat(self) -> None:
        reason = 'status'
        try:
            generation = self.service.storage.index_generation()
            if self._last_generation is not None and generation != self._last_generation:
                reason = 'index'
            self._last_generation = generation
        except Exception:  # noqa: BLE001 — lettura best effort fra richieste
            pass
        event = self.service.status_event(reason=reason)
        self.last_event = event
        self._publish(event)

    def _control_loop(self) -> None:
        rep = self._zmq.Context.instance().socket(self._zmq.REP)
        rep.setsockopt(self._zmq.LINGER, 0)
        rep.setsockopt(self._zmq.RCVTIMEO, 100)
        rep.bind(self.service.paths.control_endpoint)
        try:
            while not self._stop.is_set():
                try:
                    raw = rep.recv_string()
                except self._zmq.Again:
                    continue
                response = self._serve(raw)
                rep.send_string(pack('control.response', response))
        finally:
            rep.close(0)

    def _serve(self, raw: str) -> ControlResponse:
        try:
            command = decode_hud_command(raw)
        except ValueError as exc:
            return ControlResponse(
                command_id=_extract_command_id(raw), success=False, state=None,
                error_code='INVALID_REQUEST', error=str(exc),
            )
        return self.submit_command(command)

    def _execute(self, cmd: HudCommand) -> ControlResponse:
        try:
            return self.service.handle(cmd)
        except Exception as exc:  # noqa: BLE001 — il REP non deve restare a metà ciclo
            return ControlResponse(
                command_id=cmd.command_id, success=False, state=None,
                error_code='INVALID_REQUEST', error=str(exc),
            )

    def _publish(self, event: LiveStateEvent) -> None:
        pub = getattr(self, '_pub', None)
        if pub is None:
            return  # step() diretto nei test senza socket
        try:
            pub.send_string(pack('live.state', event))
        except self._zmq.ZMQError:
            pass  # l'HUD recupera via controllo (contratti §6)

    def _log_rejection(self, item: LiveInput, exc: ValueError) -> None:
        print(f'sensors: input scartato ({exc}): {item.id}', file=sys.stderr)


def _extract_command_id(raw: str) -> str:
    try:
        value = json.loads(raw)
        command_id = value.get('payload', {}).get('command_id')
    except Exception:  # noqa: BLE001 — payload illeggibile
        command_id = None
    return command_id if isinstance(command_id, str) and command_id else 'unknown'


# --------------------------------------------------------------------------- CLI


def load_config(path: str | None) -> dict:
    """Legge il config JSON condiviso (SpaceSpec, RuntimePaths, SearchSettings, knob)."""
    if path is None:
        return {}
    import pathlib

    return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))


def build_runtime(config: dict) -> tuple[SpaceSpec, RuntimePaths, SearchSettings]:
    """Risolve space/paths/search dal config: chiavi assenti → default mock locale."""
    from src.config import MOCK_REVISION, default_data_layout, default_space, make_runtime_paths, new_run_id

    space_cfg = config.get('space')
    space = SpaceSpec(**space_cfg) if space_cfg else default_space('mock', MOCK_REVISION)
    paths_cfg = config.get('paths')
    if paths_cfg:
        paths = RuntimePaths(**paths_cfg)
    else:
        db_real, db_mock, export_dir = default_data_layout('.')
        db_path = db_mock if space.backend == 'mock' else db_real
        paths = make_runtime_paths(
            new_run_id(), db_path, config.get('watch_dir') or os.path.abspath('./watch_dir'), export_dir
        )
    search = SearchSettings(**(config.get('search') or {}))
    return space, paths, search


def build_log_specs(args, home: str | None = None) -> list[tuple[str, str]]:
    """Specifiche (path, formato) dei tail agente: PURA e deterministica (F2).

    --agent-log e --agent-logs sono espliciti e vincono su --auto-last in caso
    di stesso file (dedup su os.path.realpath, primo in ordine CLI). L'entry
    ``PATH[:FORMAT]`` senza formato usa --agent-format; un formato non ammesso
    è un errore esplicito prima di aprire qualunque tail.
    """
    from src.sensors.agent_discovery import discover_auto_last
    from src.sensors.agent_sensor import FORMATS

    explicit: list[tuple[str, str]] = []
    if getattr(args, 'agent_log', None):
        explicit.append((args.agent_log, args.agent_format))
    for entry in getattr(args, 'agent_logs', None) or []:
        path, sep, fmt = entry.partition(':')
        if sep:
            if fmt not in FORMATS:
                raise ValueError(
                    f"INVALID_REQUEST: formato non ammesso per {path!r}: {fmt!r} "
                    f"({'|'.join(FORMATS)})"
                )
            explicit.append((path, fmt))
        else:
            explicit.append((entry, args.agent_format))

    specs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for path, fmt in explicit:
        real = os.path.realpath(path)
        if real in seen:
            continue
        seen.add(real)
        specs.append((path, fmt))
    if getattr(args, 'auto_last', False):
        for path, fmt in discover_auto_last(('claude', 'codex'),
                                            getattr(args, 'project_cwd', None), home):
            real = os.path.realpath(path)
            if real in seen:
                continue
            seen.add(real)
            specs.append((path, fmt))
    return specs


def main(argv: list[str] | None = None) -> int:
    """Runner standalone:   python -m src.sensors.runner --config PATH ..."""
    import signal
    import threading

    parser = argparse.ArgumentParser(description='Runner live dei sensori (Agent-4)')
    parser.add_argument('--config', default=None, help='config JSON condiviso (launcher/bootstrap)')
    parser.add_argument('--agent-log', default=None, help='path esplicito del log agente')
    parser.add_argument('--agent-format', choices=('plain', 'claude-jsonl', 'codex-jsonl'),
                        default='plain')
    parser.add_argument(
        '--agent-logs', action='append', default=None, metavar='PATH[:FORMAT]',
        help='log agente aggiuntivo (ripetibile; default formato = --agent-format)',
    )
    parser.add_argument(
        '--auto-last', action='store_true',
        help='scopri le ultime sessioni claude/codex (dedup sui path espliciti)',
    )
    parser.add_argument(
        '--project-cwd', default=None,
        help='cwd del progetto per --auto-last (slug ~/.claude/projects)',
    )
    parser.add_argument('--no-mic', action='store_true', help='disattiva la cattura microfono')
    parser.add_argument(
        '--system-audio', action='store_true',
        help="cattura anche l'audio di sistema (monitor PipeWire via parec -d @DEFAULT_MONITOR@)",
    )
    parser.add_argument('--show-all', action='store_true', help='salta le soglie visive (DEBUG)')
    parser.add_argument(
        '--replay-log', action='store_true',
        help='diagnostica: il log agente è una fixture e va letto dall\'inizio',
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    # La trascrizione live si valida PRIMA di toccare runtime e socket:
    # una configurazione non disponibile è un exit 2 esplicito, senza mezzi
    # avvii (F4).
    live_mode = str(config.get('live_transcription', 'off'))
    if live_mode not in ('off', 'local'):
        print(f'runner: live_transcription non ammessa: {live_mode!r} (off|local)', file=sys.stderr)
        return 2
    transcriber = None
    if live_mode == 'local':
        from src.hud import transcriber as asr

        model_dir = config.get('asr_model_dir')
        ready, detail = asr.preflight_local(model_dir)
        if not ready:
            print(f'runner: trascrizione local non pronta ({detail}); avvio negato', file=sys.stderr)
            return 2
        transcriber = asr.LocalTranscriber(str(model_dir), str(config.get('asr_language', 'it')))
    space, paths, search = build_runtime(config)

    from src.inference.client import EmbedClient
    from src.storage.db import BrainStorage

    storage = BrainStorage(paths.db_path, space)
    client = EmbedClient(paths.embed_endpoint, space.space_id,
                         timeout_ms=int(config.get('embed_timeout_ms', 30000)))
    service = RunnerService(
        storage, client, paths, search, space=space,
        alpha=float(config.get('alpha', DEFAULT_ALPHA)),
        source_ttl_sec=float(config.get('source_ttl_sec', DEFAULT_SOURCE_TTL_SEC)),
        max_pins=int(config.get('max_pins', DEFAULT_MAX_PINS)),
        session_limit_sec=float(config.get('session_limit_sec', DEFAULT_SESSION_LIMIT_SEC)),
        session_max_inputs=int(config.get('session_max_inputs', DEFAULT_SESSION_MAX_INPUTS)),
        session_audio_max_bytes=int(config.get('session_audio_max_bytes', DEFAULT_SESSION_AUDIO_MAX_BYTES)),
        show_all=args.show_all,
        # Lo status rispecchia la realtà della cattura: --no-mic è 'off' (F2).
        sensor_enabled={'mic_audio': not args.no_mic, 'system_audio': args.system_audio},
        archive_sources=frozenset(config.get('archive_sources') or ()),
        transcriber=transcriber,
    )
    server = RunnerServer(service, command_timeout_sec=config.get('command_timeout_ms', 2000) / 1000)
    server.start()

    def enqueue(item: LiveInput) -> None:
        def on_drop(dropped: LiveInput) -> None:
            service.note_dropped_input()
            _delete_unreferenced_wav(dropped)

        server.queue.put(item, on_drop=on_drop)

    tails = []
    for log_path, log_format in build_log_specs(args):
        from src.sensors.agent_sensor import AgentLogTail

        tail = AgentLogTail(
            log_path, log_format,
            session_provider=lambda: service.session_id,
            start='beginning' if args.replay_log else 'end',
        )
        tails.append(tail)
        print(f'runner: tail su {log_path} ({log_format})', file=sys.stderr)
    if tails:
        service.note_sensor('agent_logs', str(len(tails)))

    captures = []

    def make_audio_callback(sensor, enabled: bool) -> Callable[[object], None]:
        """Callback speculare per ogni sorgente audio: attiva → feed, pausa → flush.

        ``enabled`` è lo stato di cattura della sorgente al momento dell'avvio
        (service.sensor_enabled(source)); il toggle a runtime passa dal flush
        controllato del sensore, non da qui."""

        def on_block(block) -> None:
            if enabled and not service.session_suspended:
                for item in sensor.feed_block(block):
                    enqueue(item)
            else:
                for item in sensor.flush():  # pausa: flush controllato, il testo resta attivo
                    enqueue(item)

        return on_block

    def run_capture(capture_obj, label: str) -> None:
        try:
            capture_obj.start()  # blocca nel thread dedicato finché non si ferma
        except Exception as exc:  # noqa: BLE001 — niente fallback finto: errore chiaro
            print(f'runner: {label} non disponibile ({exc}); solo sorgente testuale',
                  file=sys.stderr)

    if not args.no_mic or args.system_audio:
        from src.sensors.mic_sensor import AudioCapture, MicSensor

    if not args.no_mic:
        mic = MicSensor(audio_root=paths.run_dir, session_provider=lambda: service.session_id)
        mic_capture = AudioCapture(make_audio_callback(mic, service.sensor_enabled('mic_audio')))
        captures.append(mic_capture)
        threading.Thread(target=run_capture, args=(mic_capture, 'microfono'),
                         name='mic-capture', daemon=True).start()

    if args.system_audio:
        sysaudio = MicSensor(
            source='system_audio', file_prefix='sysaudio',
            audio_root=paths.run_dir, session_provider=lambda: service.session_id,
        )
        # @DEFAULT_MONITOR@: l'output del desktop, MAI il microfono (parec senza
        # -d registra il mic: il device è sempre esplicito qui).
        sys_capture = AudioCapture(
            make_audio_callback(sysaudio, service.sensor_enabled('system_audio')),
            parec_device='@DEFAULT_MONITOR@',
        )
        captures.append(sys_capture)
        threading.Thread(target=run_capture, args=(sys_capture, "cattura audio di sistema"),
                         name='sysaudio-capture', daemon=True).start()

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    print(f'runner: sessione {service.session_id[:8]}… eventi su {paths.events_endpoint}', file=sys.stderr)
    try:
        while True:
            time.sleep(0.05)
            for tail in tails:
                item = tail.poll()
                if item is not None:
                    enqueue(item)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        diagnostics = '; '.join(f'{tail.path}: {tail.diagnostics}' for tail in tails) or 'n/d'
        print(f'runner: arresto; diagnostica tail: {diagnostics}', file=sys.stderr)
        server.close()
        for capture_obj in captures:
            capture_obj.stop()
        client.close()
        storage.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
