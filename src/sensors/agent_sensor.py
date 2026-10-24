"""Sensore agente — tail incrementale e parsing plain/Claude JSONL (piano L2).

Legge solo i byte nuovi da un log esplicito, accumula righe complete, rimuove
ANSI/repaint (plain) o estrae i messaggi testuali (claude-jsonl) e li aggrega
con debounce. L'avvio iniziale legge dal fondo: una conversazione storica non
diventa query live. Nessuna ricerca automatica dell'ultimo file Claude globale.
"""

import json
import os
import re
import time
import uuid
from collections.abc import Callable

from src.contracts import LiveInput

SOURCE = 'agent_text'
FORMATS = ('plain', 'claude-jsonl')

ANSI_CSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
ANSI_OSC = re.compile(r'\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)')
MIN_REAL_CONTENT = 4


def _clean_plain(text: str) -> str | None:
    """Rimuove ANSI CSI/OSC, backspace, carriage-return e NUL da repaint; None se vuoto.

    Il controllo di contenuto reale (MIN_REAL_CONTENT) è applicato sul BLOCCO
    aggregato in poll(), non sulla riga: righe corte legittime ('una') devono
    potersi sommare a blocchi utili, mentre un blocco di soli NUL/banner della
    shell viene scartato (fix sessione manuale 2026-10-24: 409 NUL emessi come
    LiveInput dal bridge PTY).
    """
    text = ANSI_OSC.sub('', text)
    text = ANSI_CSI.sub('', text)
    text = text.replace('\x00', '')
    chars: list[str] = []
    for ch in text:
        if ch == '\b':
            if chars:
                chars.pop()
        else:
            chars.append(ch)
    segments = [''.join(chars).strip()] if not chars else ''.join(chars).split('\r')
    non_empty = [segment.strip() for segment in segments if segment.strip()]
    return non_empty[-1] if non_empty else None


def _parse_claude_line(line: str) -> str | None:
    try:
        value = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f'MALFORMED_LINE: riga JSONL non valida: {exc}') from exc
    if not isinstance(value, dict) or value.get('type') not in ('user', 'assistant'):
        return None  # metadati, heartbeat, summary, system: ignorati
    message = value.get('message')
    if not isinstance(message, dict):
        return None
    content = message.get('content')
    if isinstance(content, str):
        return content.strip() or None
    if isinstance(content, list):
        parts = []
        for block in content:
            # tool_use/tool_result/thinking ignorati: solo blocchi testuali (piano L2)
            if isinstance(block, dict) and block.get('type') == 'text':
                text = str(block.get('text', '')).strip()
                if text:
                    parts.append(text)
        return '\n'.join(parts) or None
    return None


def parse_agent_line(line: str, format: str) -> str | None:
    """Riga completa → testo utile per la query live, o None se ignorata (§4).

    ``plain``: pulizia ANSI/repaint. ``claude-jsonl``: oggetti user/assistant
    con ``message.content`` stringa o blocchi ``type='text'``. Una riga
    completa malformata solleva ValueError: il tail la segnala e la salta.
    """
    if format == 'plain':
        return _clean_plain(line)
    if format == 'claude-jsonl':
        return _parse_claude_line(line)
    raise ValueError(f"INVALID_REQUEST: formato non ammesso: {format!r} (plain|claude-jsonl)")


class AgentLogTail:
    """Tail incrementale di un log agente: byte nuovi, righe complete, debounce.

    ``start='end'`` (default) parte dal fondo: lo storico non diventa query.
    ``start='beginning'`` è la modalità diagnostica per test/fixture
    (``--replay-log``). Troncamento o sostituzione del file: riapertura da
    offset 0; session_id non cambia mai automaticamente.
    """

    def __init__(
        self,
        path: str,
        format: str,
        *,
        clock: Callable[[], float] | None = None,
        debounce_sec: float = 1.0,
        max_chars: int = 800,
        session_id: str = '',
        session_provider: Callable[[], str] | None = None,
        start: str = 'end',
    ):
        if format not in FORMATS:
            raise ValueError(f"INVALID_REQUEST: formato non ammesso: {format!r} (plain|claude-jsonl)")
        if start not in ('end', 'beginning'):
            raise ValueError(f'INVALID_REQUEST: avvio non ammesso: {start!r} (end|beginning)')
        self.path = path
        self.format = format
        self.session_id = session_id
        self._session_provider = session_provider
        self._clock = clock if clock is not None else time.time
        self._debounce_sec = debounce_sec
        self._max_chars = max_chars
        self._start_mode = start
        self._last_emit = self._clock()
        # Avvio iniziale: il fondo di un log esistente non diventa query live.
        # Un file che compare DOPO la creazione del tail appartiene alla
        # sessione live: si parte da 0, senza perdere il primo messaggio.
        try:
            stat = os.stat(path)
        except OSError:
            self._offset: int | None = None
            self._inode: int | None = None
        else:
            self._offset = 0 if start == 'beginning' else stat.st_size
            self._inode = stat.st_ino
        self._carry = b''
        self._pending: list[str] = []
        self._last_text: str | None = None
        self.diagnostics: dict[str, int] = {'malformed_lines': 0, 'bytes_read': 0}

    def poll(self) -> LiveInput | None:
        """Consuma il file e, scaduto il debounce, emette l'input aggregato."""
        now = self._clock()
        self._consume_file()
        if not self._pending or now - self._last_emit < self._debounce_sec:
            return None
        text = '\n'.join(self._pending)[-self._max_chars:]
        self._pending.clear()
        self._last_emit = now
        if not text or text == self._last_text:
            return None  # stesso blocco stabile: non rinviato (piano L2)
        if sum(1 for ch in text if ch.isalnum()) < MIN_REAL_CONTENT:
            return None  # blocco senza contenuto reale: banner/NUL/editing del PTY
        self._last_text = text
        return LiveInput(
            id=uuid.uuid4().hex,
            session_id=self._current_session(),
            source=SOURCE,
            occurred_at=now,
            text=text,
        )

    @property
    def has_pending(self) -> bool:
        return bool(self._pending)

    def _current_session(self) -> str:
        if self._session_provider is not None:
            return self._session_provider()
        return self.session_id

    def _consume_file(self) -> None:
        try:
            stat = os.stat(self.path)
        except FileNotFoundError:
            return  # il log può comparire più tardi: nessun errore
        except OSError:
            return
        if self._offset is None:
            # comparso dopo la creazione del tail: contenuto live, si parte da 0
            self._inode = stat.st_ino
            self._offset = 0
        if stat.st_ino != self._inode or stat.st_size < self._offset:
            # troncamento o rotazione: riapri da offset 0 (piano L2)
            self._inode = stat.st_ino
            self._offset = 0
            self._carry = b''
        if stat.st_size <= self._offset:
            return
        with open(self.path, 'rb') as handle:
            handle.seek(self._offset)
            chunk = handle.read()
        self._offset += len(chunk)
        self.diagnostics['bytes_read'] += len(chunk)
        self._carry += chunk
        while b'\n' in self._carry:
            raw, _, rest = self._carry.partition(b'\n')
            self._carry = rest
            self._handle_line(raw)

    def _handle_line(self, raw: bytes) -> None:
        try:
            line = raw.decode('utf-8')
        except UnicodeDecodeError:
            line = raw.decode('utf-8', errors='replace')
            self.diagnostics['undecodable_lines'] = self.diagnostics.get('undecodable_lines', 0) + 1
        try:
            text = parse_agent_line(line, self.format)
        except ValueError:
            self.diagnostics['malformed_lines'] = self.diagnostics.get('malformed_lines', 0) + 1
            return
        if text:
            self._pending.append(text)
