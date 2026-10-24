"""Logica di stato HUD, publisher fake e UI GTK (piano HUD H1/H4).

Le classi pure (HudState, FakePublisher, card_model, EventsSubscriber,
ControlWorker) sono verificabili senza display; la UI GTK si carica in modo
lazy e tocca i socket solo dal proprio thread. Nessuna variabile globale
mutabile: la configurazione vive negli oggetti.
"""

import queue
import threading
import uuid
from dataclasses import dataclass, field

import zmq

from src.config import DEFAULT_SOURCE_TTL_SEC
from src.contracts import (
    ControlResponse,
    HudCommand,
    LiveStateEvent,
    ScoredMemory,
    decode_live_state,
)

# «Backend in attesa» dopo questo silenzio (contratti §6: 6 s senza eventi).
BACKEND_WAIT_SEC = 6.0
# Polling SUB non bloccante sul thread GTK (piano H4: ogni 100 ms).
SUB_POLL_MS = 100

FORMAT_LABELS = {"markdown": "Markdown", "pdf": "PDF", "image": "Immagine", "video": "Video"}


@dataclass
class HudState:
    """Stato applicato della UI: ultimo evento coerente e regole di visibilità.

    `apply_event` scarta sequence non strettamente crescenti, incluse quelle
    di una sessione precedente (contratti §6). I DTO restano annidati: le card
    leggono `ScoredMemory.memory`, non campi piatti.
    """

    source_ttl_sec: float = DEFAULT_SOURCE_TTL_SEC
    stale_after_sec: float = BACKEND_WAIT_SEC
    session_id: str | None = None
    event: LiveStateEvent | None = field(default=None, repr=False)
    _last_sequence: int | None = field(default=None, repr=False)

    def apply_event(self, event: LiveStateEvent) -> bool:
        """Applica un evento SUB o lo scarta; True se applicato."""
        if self._last_sequence is not None and event.sequence <= self._last_sequence:
            return False
        self._last_sequence = event.sequence
        self.session_id = event.session_id
        self.event = event
        return True

    def apply_response(self, response) -> bool:
        """Applica lo stato di una risposta controllo (pin/unpin compreso).

        Le risposte controllo valgono come stato (contratti §6); le risposte
        fallite non toccano lo stato corrente.
        """
        if not response.success or response.state is None:
            return False
        return self.apply_event(response.state)

    @property
    def pinned(self):
        """Pin congelati dell'ultimo stato applicato (DTO PinnedMemory)."""
        if self.event is None:
            return []
        return list(self.event.pinned)

    @property
    def sensor_status(self) -> dict[str, str]:
        """Stato sensori riportato dal backend, opaco alla UI."""
        if self.event is None:
            return {}
        return dict(self.event.sensor_status)

    def visible_results(self, now: float) -> list[ScoredMemory]:
        """Risultati con winning_source ancora entro il TTL.

        Una card si nasconde quando scade la winning_source secondo
        source_last_seen, anche se altre matched_sources sono fresche: si
        attende il ricalcolo invece di mostrare il vecchio punteggio.
        """
        if self.event is None:
            return []
        visible = []
        for item in self.event.results:
            winning = item.winning_source
            if winning is None:
                visible.append(item)
                continue
            last_seen = self.event.source_last_seen.get(winning)
            if last_seen is None:
                continue
            if now - last_seen <= self.source_ttl_sec:
                visible.append(item)
        return visible

    def status(self, now: float) -> str:
        """'ok' con eventi freschi (lista anche vuota), altrimenti 'waiting_backend'."""
        if self.event is None or now - self.event.emitted_at > self.stale_after_sec:
            return "waiting_backend"
        return "ok"


class FakePublisher:
    """Generatore di eventi demo indipendente dal runner sensori.

    Produce LiveStateEvent validi per il decoder wire reale, così la demo
    desktop è indistinguibile dal flusso di Agent-4 a livello di contratto.
    """

    def __init__(
        self,
        *,
        session_id: str = "demo-session",
        space_id: str = "demo-space",
        backend: str = "mock",
        base_time: float = 1000.0,
        results: list[ScoredMemory] | None = None,
    ) -> None:
        self.session_id = session_id
        self.space_id = space_id
        self.backend = backend
        self.base_time = base_time
        self.results = results if results is not None else []
        self._sequence = 0
        self._input_count = 0

    def next_event(
        self,
        now: float | None = None,
        *,
        reason: str = "input",
        results: list[ScoredMemory] | None = None,
    ) -> LiveStateEvent:
        """Prossimo evento demo, con input fittizio e sequence crescente."""
        self._sequence += 1
        self._input_count += 1
        moment = self.base_time + self._sequence if now is None else now
        from src.contracts import LiveInput  # import locale: evita ricircolo a livello modulo

        return LiveStateEvent(
            event_id=str(uuid.uuid4()),
            sequence=self._sequence,
            session_id=self.session_id,
            space_id=self.space_id,
            backend=self.backend,
            emitted_at=moment,
            reason=reason,
            latest_inputs=[
                LiveInput(
                    id=f"demo-in-{self._input_count}",
                    session_id=self.session_id,
                    source="agent_text",
                    occurred_at=moment - 1.0,
                    text=f"domanda demo {self._input_count}",
                )
            ],
            results=self.results if results is None else results,
            pinned=[],
            sensor_status={"agent_text": "demo"},
            source_last_seen={"agent_text": moment},
            dropped_inputs=0,
        )


# --------------------------------------------------------------------------- H4: logica senza display


def card_model(item: ScoredMemory) -> dict:
    """Modello di card letto dal DTO annidato, mai da campi piatti."""
    memory = item.memory
    return {
        "memory_id": memory.id,
        "title": memory.context_header[:120],
        "snippet": memory.snippet_text,
        "format": FORMAT_LABELS[memory.modality],
        "matched": ", ".join(item.matched_sources),
        "score": f"{item.final_score:.2f}",
        "boost": item.synaptic_boost > 0,
    }


class EventsSubscriber:
    """SUB ZMQ pollato dal thread GTK: il socket vive ed è usato solo lì."""

    def __init__(self, endpoint: str):
        context = zmq.Context.instance()
        self._socket = context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket.setsockopt(zmq.RCVHWM, 64)
        self._socket.connect(endpoint)

    def poll(self) -> list[LiveStateEvent]:
        """Drain non bloccante degli eventi; i corrotti vengono scartati."""
        events = []
        while True:
            try:
                raw = self._socket.recv_string(zmq.NOBLOCK)
            except zmq.Again:
                break
            try:
                events.append(decode_live_state(raw))
            except ValueError:
                continue  # snapshot successivo corregge lo stato
        return events

    def close(self) -> None:
        self._socket.close(0)


class ControlWorker:
    """Coda di comandi su thread dedicato con REQ propri (src.control.send_command).

    Le risposte arrivano alla callback su thread worker: la UI la costruisce
    con GLib.idle_add per tornare al thread GTK. `send_command` ripete una
    volta con lo stesso command_id; un fallimento sistematico diventa una
    ControlResponse sintetica, mai un'eccezione nel loop.
    """

    def __init__(self, endpoint: str, on_response):
        self._endpoint = endpoint
        self._on_response = on_response
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, command: HudCommand) -> None:
        self._queue.put(command)

    def _loop(self) -> None:
        from src.control import send_command

        while True:
            command = self._queue.get()
            if command is None:
                break
            try:
                response = send_command(self._endpoint, command)
            except SystemExit as exc:
                response = ControlResponse(
                    command_id=command.command_id,
                    success=False,
                    state=None,
                    error_code="CONTROL_ERROR",
                    error=str(exc),
                )
            try:
                self._on_response(response)
            except Exception:
                pass  # un errore di UI non deve uccidere il worker

    def stop(self) -> None:
        self._queue.put(None)


def _default_open_file(memory):
    from src.hud import openers

    openers.open_memory_file(memory)


class HudLogic:
    """Percorso click->comando senza widget, testabile headless (fix flash H4).

    Gli handler GTK delegano qui: nessuna chiamata a GLib/GTK e nessun
    rebuild sincrono dentro un handler. La UI si ricostruisce solo su
    cambiamento reale (evento applicato, risposta controllo, scadenza di
    visibilità), mai a tempo: ricostruire la lista ogni 100 ms distrugge i
    bottoni fra press e release e il segnale «clicked» non arriva mai.
    """

    def __init__(self, *, settings, endpoints=None, worker=None, subscriber=None, demo=False, open_file=None):
        self.settings = settings
        self.demo = demo
        self.state = HudState()
        self.pending_commands: set[str] = set()
        self.pending_pins: dict[str, str] = {}  # command_id -> memory_id
        self.export_running = False
        self._demo_publisher = FakePublisher() if demo else None
        self._dirty = False
        self._view_signature = None
        self.open_file = open_file or _default_open_file
        if subscriber is not None:
            self.subscriber = subscriber
        else:
            self.subscriber = EventsSubscriber(endpoints["events"]) if endpoints else None
        if worker is not None:
            self.worker = worker
        elif endpoints is not None:
            self.worker = ControlWorker(endpoints["control"], self._on_worker_response)
        else:
            self.worker = None

    # ---------------------------------------------------------------- comandi

    def _send_current(self, action: str, *, memory_id=None, enabled=None):
        """Comando sulla sessione corrente; None senza worker (demo)."""
        if self.worker is None:
            return None
        command = HudCommand(
            command_id=uuid.uuid4().hex,
            session_id=self.state.session_id,
            action=action,
            memory_id=memory_id,
            enabled=enabled,
        )
        self.pending_commands.add(command.command_id)
        if action == "pin" and memory_id:
            self.pending_pins[command.command_id] = memory_id
        self.worker.submit(command)
        self.request_refresh()
        return command

    def start_discovery(self):
        """Snapshot di discovery: unico comando che accetta session_id None."""
        if self.worker is None:
            return None
        command = HudCommand(command_id=uuid.uuid4().hex, session_id=None, action="snapshot")
        self.pending_commands.add(command.command_id)
        self.worker.submit(command)
        self.request_refresh()
        return command

    def request_pin(self, memory_id: str):
        if memory_id in {p.memory.id for p in self.state.pinned}:
            return None
        if memory_id in self.pending_pins.values():
            return None
        return self._send_current("pin", memory_id=memory_id)

    def request_unpin(self, memory_id: str):
        return self._send_current("unpin", memory_id=memory_id)

    def request_mic_toggle(self):
        mic_on = (self.state.sensor_status.get("mic") or "on") != "off"
        return self._send_current("set_mic", enabled=not mic_on)

    def request_new_session(self):
        return self._send_current("new_session")

    def request_open(self, memory):
        self.open_file(memory)

    def mic_label(self) -> str:
        """Etichetta del bottone Mic: segue sensor_status dello snapshot.

        Default 'on' se assente, la stessa convenzione di request_mic_toggle;
        nessun aggiornamento ottimistico al click, segue lo stato applicato.
        """
        return f"Mic: {self.state.sensor_status.get('mic') or 'on'}"

    # ---------------------------------------------------------------- risposte ed eventi

    def _on_worker_response(self, response):
        """Hook del worker (thread controllo): la UI lo incarta in GLib.idle_add."""
        self.handle_response(response)

    def handle_response(self, response) -> bool:
        applied = self.state.apply_response(response)
        self.pending_commands.discard(response.command_id)
        self.pending_pins.pop(response.command_id, None)
        self.request_refresh()
        return applied

    def poll_events(self) -> bool:
        """Applica gli eventi SUB; True se qualcosa è cambiato."""
        if self.subscriber is None:
            return False
        changed = False
        for event in self.subscriber.poll():
            changed |= self.state.apply_event(event)
        return changed

    def view_signature_changed(self, now: float) -> bool:
        """True quando lista visibile o status cambiano (scadenze TTL comprese)."""
        signature = (
            self.state.status(now),
            tuple(item.memory.id for item in self.state.visible_results(now)),
        )
        if signature != self._view_signature:
            self._view_signature = signature
            return True
        return False

    def request_refresh(self):
        """Hook UI: default solo flag, la finestra lo mappa su un rebuild differito."""
        self._dirty = True

    def publish_toast(self, message: str):
        """Punto unico di pubblicazione toast: Adw.ToastOverlay, mai la window.

        ApplicationWindow non ha add_toast: senza overlay ogni fine export
        sollevava AttributeError e l'utente non vedeva mai l'esito.
        """
        toast = self.Adw.Toast.new(message)
        self.toast_overlay.add_toast(toast)


def export_outcome(result, error) -> tuple[str, str]:
    """(messaggio toast, testo pannello di stato) dall'esito dell'export.

    Nessun falso successo: un errore produce toast vuoto e stato esplicito.
    """
    if error is not None:
        return "", f"errore: EXPORT_FAILED: {error}"
    details = f"{result.markdown_path}\nclipboard: {result.clipboard_status}, warning: {len(result.warnings)}"
    return f"contesto pronto: {len(result.warnings)} warning — percorso in dettagli", details


# --------------------------------------------------------------------------- H4: UI GTK (solo desktop)


def _load_gi():
    """Import lazy di PyGObject: i test headless non toccano mai questa funzione."""
    import gi

    gi.require_version("Gtk", "4.0")
    gi.require_version("Adw", "1")
    from gi.repository import Adw, GLib, Gtk

    return Adw, GLib, Gtk


class HudWindow(HudLogic):
    """Finestra Adw (~380x720): header di stato, card live, tray pin, controlli.

    Il SUB è pollato dal timeout GLib sul thread GTK; il controllo e l'export
    girano su worker con callback rientranti via GLib.idle_add. La UI non si
    ricostruisce mai a tempo né dentro un handler: solo su cambiamento
    applicato, così i click non cadono su widget distrutti (fix flash).
    """

    def __init__(self, Adw, GLib, Gtk, *, endpoints=None, settings, demo=False):
        super().__init__(settings=settings, endpoints=endpoints, demo=demo)
        self.Adw, self.GLib, self.Gtk = Adw, GLib, Gtk
        self._build_window()

    # ---------------------------------------------------------------- costruzione UI

    def _build_window(self):
        Gtk, Adw = self.Gtk, self.Adw
        self.window = Adw.ApplicationWindow(
            title="Ambient Second Brain",
            default_width=380,
            default_height=720,
            resizable=True,
        )
        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        header = Adw.HeaderBar()
        self.header_label = Gtk.Label(label="in attesa del backend")
        header.set_title_widget(self.header_label)
        root.append(header)
        self.status_label = Gtk.Label(
            label="avvio…",
            wrap=True,
            xalign=0.0,
            margin_start=10,
            margin_end=10,
        )
        root.append(self.status_label)
        self.results_list = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        results_scroll = Gtk.ScrolledWindow(
            child=self.results_list,
            vexpand=True,
            margin_start=10,
            margin_end=10,
        )
        root.append(results_scroll)
        tray_label = Gtk.Label(label="Pin della sessione", xalign=0.0, margin_start=10)
        root.append(tray_label)
        self.tray_list = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        root.append(self.tray_list)
        controls = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=6, margin_start=10, margin_end=10, margin_bottom=10
        )
        self.mic_button = Gtk.Button(label="Mic: on")
        self.mic_button.connect("clicked", self._on_mic_toggle)
        controls.append(self.mic_button)
        self.session_button = Gtk.Button(label="Nuova sessione")
        self.session_button.connect("clicked", self._on_new_session)
        controls.append(self.session_button)
        root.append(controls)
        export_row = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=6, margin_start=10, margin_end=10, margin_bottom=10
        )
        self.transcription_selector = self.Gtk.DropDown(
            model=self.Gtk.StringList(strings=["Trascrizione: disattivata", "Trascrizione: locale"]),
            selected=0 if self.settings.transcription == "off" else 1,
        )
        export_row.append(self.transcription_selector)
        self.export_button = Gtk.Button(label="Prepara contesto")
        self.export_button.connect("clicked", self._on_export)
        export_row.append(self.export_button)
        root.append(export_row)
        self.toast_overlay = Adw.ToastOverlay()
        self.toast_overlay.set_child(root)
        self.window.set_content(self.toast_overlay)

    # ---------------------------------------------------------------- eventi e stato

    def start(self):
        """Avvia polling SUB, snapshot di discovery e (demo) publisher finto."""
        if self.demo:
            self.GLib.timeout_add(1500, self._on_demo_tick)
        if self.subscriber is not None:
            self.GLib.timeout_add(SUB_POLL_MS, self._on_sub_tick)
        self.start_discovery()  # snapshot senza sessione

    def request_refresh(self):
        """Rebuild differito: mai dentro il handler che sta gestendo il click."""
        self.GLib.idle_add(self.refresh, _now())

    def _on_sub_tick(self):
        now = _now()
        changed = self.poll_events()
        if changed or self._dirty or self.view_signature_changed(now):
            self._dirty = False
            self.refresh(now)
        return True  # continua il timeout

    def _on_demo_tick(self):
        if self.state.apply_event(self._demo_publisher.next_event()):
            self.refresh(_now())
        return True

    def _on_worker_response(self, response: ControlResponse):
        # Callback del worker (thread controllo): rientro nel thread GTK.
        self.GLib.idle_add(self._apply_response, response)

    def _apply_response(self, response: ControlResponse):
        self.handle_response(response)
        if not response.success:
            self.show_error(f"comando non riuscito: {response.error_code}")
        self.refresh(_now())
        return False  # idle una tantum

    # ---------------------------------------------------------------- azioni utente

    def _on_mic_toggle(self, _button):
        try:
            self.request_mic_toggle()
        except Exception as exc:  # nessun handler deve morire in silenzio
            self.show_error(str(exc))

    def _on_new_session(self, _button):
        # Nuova sessione: la storia temporanea riparte da zero; pin e export restano.
        try:
            self.request_new_session()
        except Exception as exc:
            self.show_error(str(exc))

    def _on_pin(self, _button, item: ScoredMemory):
        try:
            self.request_pin(item.memory.id)
        except Exception as exc:
            self.show_error(str(exc))

    def _on_unpin(self, _button, pinned_item):
        # Unpin non cancella l'apprendimento: solo il tray si aggiorna dallo stato.
        try:
            self.request_unpin(pinned_item.memory.id)
        except Exception as exc:
            self.show_error(str(exc))

    def _on_open(self, _button, item: ScoredMemory):
        try:
            self.request_open(item.memory)
        except (FileNotFoundError, RuntimeError, OSError) as exc:
            self.show_error(str(exc))

    def _on_export(self, _button):
        if self.export_running or self.worker is None:
            return
        # L'opzione si congela all'avvio dell'export (piano H3).
        frozen = replace_settings(
            self.settings,
            transcription="local" if self.transcription_selector.get_selected() == 1 else "off",
        )
        self.export_running = True
        self.export_button.set_sensitive(False)
        self.status_label.set_text("export: snapshot e staging…")
        thread = threading.Thread(target=self._export_job, args=(frozen,), daemon=True)
        thread.start()

    def _export_job(self, frozen):
        from src.hud.context_exporter import assemble_and_export_context

        def provider():
            command = HudCommand(command_id=uuid.uuid4().hex, session_id=None, action="snapshot")
            from src.control import send_command

            response = send_command(self._worker_endpoint(), command)
            if not response.success or response.state is None:
                raise RuntimeError(f"snapshot export fallito: {response.error_code}")
            return response.state, response.session_inputs, response.lease_expires_at

        try:
            result = assemble_and_export_context(
                self.state.event,
                _latest_session_inputs(self.state),
                _export_root(),
                frozen,
                lease_expires_at=None,
                snapshot_provider=provider,
            )
            self.GLib.idle_add(self._export_done, result, None)
        except Exception as exc:
            self.GLib.idle_add(self._export_done, None, exc)

    def _worker_endpoint(self):
        return self.worker._endpoint

    def _export_done(self, result, error):
        self.export_running = False
        self.export_button.set_sensitive(True)
        toast_message, status_text = export_outcome(result, error)
        self.status_label.set_text(status_text)
        if toast_message:
            self.publish_toast(toast_message)
        self.refresh(_now())
        return False

    def show_error(self, message: str):
        self.status_label.set_text(f"errore: {message}")

    # ---------------------------------------------------------------- aggiornamento vista

    def refresh(self, now):
        pinned_ids = {p.memory.id for p in self.state.pinned}
        self._refresh_header(now, pinned_ids)
        self._refresh_results(now, pinned_ids)
        self._refresh_tray()
        return False

    def _refresh_header(self, now, pinned_ids):
        status = self.state.status(now)
        event = self.state.event
        self.mic_button.set_label(self.mic_label())
        if status == "waiting_backend":
            self.header_label.set_text("in attesa del backend")
        elif event is not None:
            mic = self.state.sensor_status.get("mic", "?")
            agent = self.state.sensor_status.get("agent_text", "?")
            self.header_label.set_text(
                f"backend {event.backend} · mic {mic} · agente {agent} · scarti {event.dropped_inputs}"
            )
        self.mic_button.set_sensitive(self.worker is not None and self.state.session_id is not None)
        self.session_button.set_sensitive(self.worker is not None and self.state.session_id is not None)
        self.export_button.set_sensitive(
            self.worker is not None and self.state.session_id is not None and not self.export_running
        )

    def _refresh_results(self, now, pinned_ids):
        Gtk = self.Gtk
        while child := self.results_list.get_first_child():
            self.results_list.remove(child)
        status = self.state.status(now)
        if status == "waiting_backend":
            self.results_list.append(Gtk.Label(label="in attesa del backend…", wrap=True))
            return
        visible = self.state.visible_results(now)
        if not visible:
            # Lista vuota legittima (query negativa), distinta dall'attesa del backend.
            self.results_list.append(Gtk.Label(label="nessun ricordo pertinente in questo momento", wrap=True))
            return
        for item in visible:
            self.results_list.append(self._build_card(item, pinned_ids))

    def _build_card(self, item, pinned_ids):
        Gtk = self.Gtk
        model = card_model(item)
        card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        card.append(Gtk.Label(label=model["title"], xalign=0.0, wrap=True, ellipsize=3, width_chars=30, lines=1))
        snippet = Gtk.Label(label=model["snippet"], xalign=0.0, wrap=True, ellipsize=3)
        snippet.set_lines(3)
        card.append(snippet)
        badge = " · boost grafo" if model["boost"] else ""
        card.append(
            Gtk.Label(
                label=f"{model['format']} · {model['matched']} · score {model['score']}{badge}",
                xalign=0.0,
                wrap=True,
            )
        )
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        open_button = Gtk.Button(label="Apri")
        open_button.connect("clicked", self._on_open, item)
        row.append(open_button)
        pin_button = Gtk.Button(label="Pin")
        is_pinned = model["memory_id"] in pinned_ids
        is_pending = model["memory_id"] in set(self.pending_pins.values())
        pin_button.set_sensitive(not is_pinned and not is_pending)
        if is_pinned:
            pin_button.set_label("Pinnato")
        pin_button.connect("clicked", self._on_pin, item)
        row.append(pin_button)
        card.append(row)
        return card

    def _refresh_tray(self):
        Gtk = self.Gtk
        while child := self.tray_list.get_first_child():
            self.tray_list.remove(child)
        if not self.state.pinned:
            self.tray_list.append(Gtk.Label(label="(nessun pin)", xalign=0.0))
            return
        for pinned_item in self.state.pinned:
            row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
            row.append(Gtk.Label(label=pinned_item.memory.context_header[:120], xalign=0.0, ellipsize=3, hexpand=True))
            remove = Gtk.Button(label="Rimuovi")
            remove.connect("clicked", self._on_unpin, pinned_item)
            row.append(remove)
            self.tray_list.append(row)


def replace_settings(settings, **overrides):
    """Copia congelata delle ExportSettings: il cambio opzione vale dal prossimo export."""
    from dataclasses import replace

    return replace(settings, **overrides)


def _now():
    import time

    return time.time()


def _latest_session_inputs(state: HudState):
    """Input dell'ultimo stato: base per l'export se il provider non fornisce altro."""
    if state.event is None:
        return []
    return list(state.event.latest_inputs)


def _export_root():
    from pathlib import Path

    return str(Path("exports").resolve())


def main(argv: list[str] | None = None) -> int:
    """Avvio desktop: --runtime PATH (endpoints da state.json) oppure --demo."""
    import argparse
    import json
    from pathlib import Path

    from src.contracts import ExportSettings

    parser = argparse.ArgumentParser(prog="src.hud.app", description="HUD GTK di Ambient Second Brain")
    parser.add_argument("--runtime", help="directory runtime con state.json")
    parser.add_argument("--demo", action="store_true", help="demo senza backend, publisher fake")
    parser.add_argument("--transcription", choices=("off", "local"), default="off")
    parser.add_argument("--asr-model-dir", default=None)
    args = parser.parse_args(argv)
    if not args.demo and not args.runtime:
        parser.error("richiesto --runtime PATH oppure --demo")
    endpoints = None
    if args.runtime:
        state = json.loads((Path(args.runtime) / "state.json").read_text())
        endpoints = state["endpoints"]
    settings = ExportSettings(transcription=args.transcription, asr_model_dir=args.asr_model_dir)
    Adw, GLib, Gtk = _load_gi()

    app = Adw.Application(application_id="org.artificialmemory.Hud")
    window_holder = {}

    def on_activate(_app):
        window = HudWindow(Adw, GLib, Gtk, endpoints=endpoints, settings=settings, demo=args.demo)
        window_holder["window"] = window
        window.window.set_application(_app)
        window.window.present()
        window.start()

    app.connect("activate", on_activate)
    try:
        return app.run(None)
    finally:
        holder = window_holder.get("window")
        if holder and holder.subscriber is not None:
            holder.subscriber.close()
        if holder and holder.worker is not None:
            holder.worker.stop()


if __name__ == "__main__":
    raise SystemExit(main())
