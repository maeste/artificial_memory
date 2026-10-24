"""Test HUD H1 — openers e logica card senza display.

Nessuna GUI aperta: i test costruiscono argv ed eventi DTO senza eseguire
player né mostrare finestre (mark desktop riservato ai test GNOME in H4).
"""

import dataclasses
import hashlib
from dataclasses import replace

import pytest

from src.config import DEFAULT_SOURCE_TTL_SEC
from src.contracts import (
    ContextSnapshot,
    ControlResponse,
    ExportSettings,
    HudCommand,
    LiveInput,
    LiveStateEvent,
    MemoryView,
    PinnedMemory,
    ScoredMemory,
    decode_live_state,
    make_memory_id,
    pack,
)
from src.hud import app, openers


# --------------------------------------------------------------------------- fixture


@pytest.fixture
def memory_factory():
    """Costruisce MemoryView completa coi digest richiesti, con override per campo."""

    def make(**overrides) -> MemoryView:
        file_path = overrides.get("file_path", "/tmp/demo.md")
        chunk_key = overrides.get("chunk_key", "section:Demo:0:part:0")
        content_text = overrides.get("content_text", "La cache riduce gli accessi al database.")
        content_sha256 = hashlib.sha256(content_text.encode("utf-8")).hexdigest()
        defaults = dict(
            id=make_memory_id(file_path, chunk_key, content_sha256),
            file_path=file_path,
            modality="markdown",
            chunk_key=chunk_key,
            content_sha256=content_sha256,
            source_sha256=hashlib.sha256(b"sorgente-demo").hexdigest(),
            context_header="demo.md > Demo",
            snippet_text=content_text[:350],
            content_text=content_text,
            page_num=None,
            start_sec=None,
            end_sec=None,
            created_at=1000.0,
        )
        return MemoryView(**{**defaults, **overrides})

    return make


def make_result(memory: MemoryView, *, winning: str = "agent_text") -> ScoredMemory:
    matched = ["agent_text", "mic_audio"] if winning == "mic_audio" else ["agent_text"]
    return ScoredMemory(
        memory=memory,
        cosine_score=0.7,
        z_score=1.8,
        synaptic_boost=0.0,
        final_score=1.8,
        matched_sources=matched,
        winning_source=winning,
    )


def make_event(
    sequence: int,
    *,
    session_id: str = "s1",
    results: list[ScoredMemory] | None = None,
    pinned: list[PinnedMemory] | None = None,
    source_last_seen: dict[str, float] | None = None,
    emitted_at: float = 1000.0,
    reason: str = "input",
) -> LiveStateEvent:
    return LiveStateEvent(
        event_id=f"ev-{sequence}",
        sequence=sequence,
        session_id=session_id,
        space_id="space-demo",
        backend="mock",
        emitted_at=emitted_at,
        reason=reason,
        latest_inputs=[
            LiveInput(
                id="in-1",
                session_id=session_id,
                source="agent_text",
                occurred_at=emitted_at - 1.0,
                text="domanda demo",
            )
        ],
        results=results or [],
        pinned=pinned or [],
        sensor_status={"agent_text": "ok"},
        source_last_seen=source_last_seen if source_last_seen is not None else {"agent_text": emitted_at},
        dropped_inputs=0,
    )


def make_response(
    state: LiveStateEvent | None,
    *,
    command_id: str = "cmd-1",
    success: bool = True,
) -> ControlResponse:
    return ControlResponse(
        command_id=command_id,
        success=success,
        state=state,
        session_inputs=[],
        lease_expires_at=None,
        error_code=None if success else "STALE_SESSION",
        error=None if success else "sessione superata",
    )


def make_pin(memory: MemoryView) -> PinnedMemory:
    return PinnedMemory(
        memory=memory,
        pinned_at=1000.0,
        snapshot=ContextSnapshot(id="snap-1", session_id="s1", captured_at=999.0, input_ids=["in-1"]),
    )


# --------------------------------------------------------------------------- H1: openers


class TestOpenerBuildCommand:
    def test_video_seek_keeps_fraction(self, memory_factory):
        memory = memory_factory(modality="video", file_path="/tmp/a b.mp4", start_sec=12.5, end_sec=30.0)
        assert openers.build_open_command(memory) == ["mpv", "--start=12.5", "/tmp/a b.mp4"]

    def test_video_seek_integer_value_keeps_decimal(self, memory_factory):
        memory = memory_factory(modality="video", file_path="/tmp/v.mp4", start_sec=30.0, end_sec=45.0)
        assert openers.build_open_command(memory) == ["mpv", "--start=30.0", "/tmp/v.mp4"]

    def test_video_without_interval_opens_whole_file(self, memory_factory):
        memory = memory_factory(modality="video", file_path="/tmp/v.mp4", start_sec=None, end_sec=None)
        assert openers.build_open_command(memory) == ["mpv", "/tmp/v.mp4"]

    def test_pdf_page_index_is_user_facing(self, memory_factory):
        memory = memory_factory(modality="pdf", file_path="/tmp/doc.pdf", page_num=3)
        assert openers.build_open_command(memory) == ["evince", "--page-index=3", "/tmp/doc.pdf"]

    def test_pdf_first_page_is_one(self, memory_factory):
        memory = memory_factory(modality="pdf", file_path="/tmp/doc.pdf", page_num=1)
        assert openers.build_open_command(memory) == ["evince", "--page-index=1", "/tmp/doc.pdf"]

    def test_pdf_without_page_opens_document(self, memory_factory):
        memory = memory_factory(modality="pdf", file_path="/tmp/doc.pdf", page_num=None)
        assert openers.build_open_command(memory) == ["evince", "/tmp/doc.pdf"]

    def test_markdown_uses_xdg_open(self, memory_factory):
        memory = memory_factory(modality="markdown", file_path="/tmp/nota.md")
        assert openers.build_open_command(memory) == ["xdg-open", "/tmp/nota.md"]

    def test_image_uses_xdg_open(self, memory_factory):
        memory = memory_factory(modality="image", file_path="/tmp/img.png")
        assert openers.build_open_command(memory) == ["xdg-open", "/tmp/img.png"]

    def test_shell_metacharacters_stay_in_single_argument(self, memory_factory):
        path = "/tmp/a b; rm -rf x & `id` $(id) | ho.mp4"
        memory = memory_factory(modality="video", file_path=path, start_sec=1.0, end_sec=2.0)
        assert openers.build_open_command(memory) == ["mpv", "--start=1.0", path]


class TestOpenerLaunch:
    def test_missing_file_raises_visible_error(self, tmp_path, memory_factory):
        memory = memory_factory(modality="markdown", file_path=str(tmp_path / "assente.md"))
        with pytest.raises(FileNotFoundError, match="assente.md"):
            openers.open_memory_file(memory)

    def test_launches_built_argv_without_shell(self, tmp_path, memory_factory, monkeypatch):
        target = tmp_path / "nota.md"
        target.write_text("contenuto", encoding="utf-8")
        launched: dict = {}

        def fake_popen(argv, **kwargs):
            launched["argv"] = list(argv)
            launched["kwargs"] = kwargs
            return None

        monkeypatch.setattr(openers.subprocess, "Popen", fake_popen)
        monkeypatch.setattr(openers.shutil, "which", lambda name: f"/usr/bin/{name}")
        openers.open_memory_file(memory_factory(modality="markdown", file_path=str(target)))
        assert launched["argv"] == ["xdg-open", str(target)]
        assert launched["kwargs"].get("shell", False) is False

    def test_missing_tool_raises_clear_error(self, tmp_path, memory_factory, monkeypatch):
        target = tmp_path / "nota.md"
        target.write_text("contenuto", encoding="utf-8")
        monkeypatch.setattr(openers.shutil, "which", lambda name: None)
        memory = memory_factory(modality="markdown", file_path=str(target))
        with pytest.raises(RuntimeError, match="xdg-open"):
            openers.open_memory_file(memory)


class TestOpenerPreflight:
    def test_reports_missing_tools_by_name(self, monkeypatch):
        available = {"mpv": None, "evince": "/usr/bin/evince", "xdg-open": "/usr/bin/xdg-open"}
        monkeypatch.setattr(openers.shutil, "which", lambda name: available.get(name))
        assert openers.missing_open_tools() == ["mpv"]

    def test_all_present_is_empty(self, monkeypatch):
        monkeypatch.setattr(openers.shutil, "which", lambda name: f"/usr/bin/{name}")
        assert openers.missing_open_tools() == []


# --------------------------------------------------------------------------- H1: logica card


class TestHudStateSequences:
    def test_first_event_applies(self):
        state = app.HudState()
        assert state.apply_event(make_event(0)) is True
        assert state.session_id == "s1"

    def test_stale_sequence_same_session_discarded(self):
        state = app.HudState()
        state.apply_event(make_event(5))
        assert state.apply_event(make_event(4)) is False
        assert state.apply_event(make_event(5)) is False

    def test_old_session_lower_sequence_discarded(self):
        state = app.HudState()
        state.apply_event(make_event(10, session_id="s1"))
        assert state.apply_event(make_event(3, session_id="s0")) is False
        assert state.session_id == "s1"

    def test_new_session_higher_sequence_applied(self):
        state = app.HudState()
        state.apply_event(make_event(10, session_id="s1"))
        assert state.apply_event(make_event(11, session_id="s2")) is True
        assert state.session_id == "s2"

    def test_state_keeps_nested_dtos(self, memory_factory):
        memory = memory_factory()
        state = app.HudState()
        state.apply_event(make_event(0, results=[make_result(memory)]))
        visible = state.visible_results(1000.0)
        assert [item.memory.id for item in visible] == [memory.id]

    def test_pin_arrives_from_control_response(self, memory_factory):
        pin = make_pin(memory_factory())
        state = app.HudState()
        response = make_response(make_event(7, pinned=[pin], reason="pin"))
        assert state.apply_response(response) is True
        assert state.pinned == [pin]

    def test_late_control_response_does_not_rollback(self, memory_factory):
        pin_late = make_pin(memory_factory())
        state = app.HudState()
        state.apply_event(make_event(10))
        response = make_response(make_event(4, pinned=[pin_late]), command_id="cmd-late")
        assert state.apply_response(response) is False
        assert state.pinned == []

    def test_failed_response_ignored(self):
        state = app.HudState()
        assert state.apply_response(make_response(None, success=False)) is False
        assert state.session_id is None


class TestHudStateVisibility:
    def test_expired_winning_source_hides_card_even_with_other_fresh_match(self, memory_factory):
        memory_agent = memory_factory(file_path="/tmp/agente.md")
        memory_mic = memory_factory(file_path="/tmp/mic.md", content_text="parlato")
        mixed = replace(make_result(memory_mic, winning="mic_audio"), matched_sources=["agent_text", "mic_audio"])
        state = app.HudState()
        state.apply_event(
            make_event(
                0,
                results=[make_result(memory_agent), mixed],
                source_last_seen={"agent_text": 995.0, "mic_audio": 860.0},
            )
        )
        visible = state.visible_results(1000.0)
        assert [item.memory.id for item in visible] == [memory_agent.id]

    def test_winning_source_without_last_seen_is_hidden(self, memory_factory):
        memory_agent = memory_factory(file_path="/tmp/agente.md")
        memory_mic = memory_factory(file_path="/tmp/mic.md", content_text="parlato")
        state = app.HudState()
        state.apply_event(
            make_event(
                0,
                results=[make_result(memory_agent), make_result(memory_mic, winning="mic_audio")],
                source_last_seen={"agent_text": 995.0},
            )
        )
        assert [item.memory.id for item in state.visible_results(1000.0)] == [memory_agent.id]

    def test_ttl_boundary_is_inclusive(self, memory_factory):
        memory = memory_factory()
        state = app.HudState()
        state.apply_event(
            make_event(
                0, results=[make_result(memory)], source_last_seen={"agent_text": 1000.0 - DEFAULT_SOURCE_TTL_SEC}
            )
        )
        assert [item.memory.id for item in state.visible_results(1000.0)] == [memory.id]

    def test_no_results_before_first_event(self):
        assert app.HudState().visible_results(0.0) == []


class TestHudStateStatus:
    def test_waiting_backend_before_first_event(self):
        assert app.HudState().status(1000.0) == "waiting_backend"

    def test_fresh_empty_results_are_not_waiting(self):
        state = app.HudState()
        state.apply_event(make_event(0, results=[]))
        assert state.visible_results(1000.5) == []
        assert state.status(1000.5) == "ok"

    def test_silence_beyond_six_seconds_is_waiting(self):
        state = app.HudState()
        state.apply_event(make_event(0, emitted_at=1000.0))
        assert state.status(1006.0) == "ok"
        assert state.status(1006.5) == "waiting_backend"


# --------------------------------------------------------------------------- H1: publisher fake


class TestFakePublisher:
    def test_events_are_strictly_increasing_and_wire_valid(self):
        publisher = app.FakePublisher(base_time=1000.0)
        events = [publisher.next_event() for _ in range(3)]
        sequences = [event.sequence for event in events]
        assert sequences == sorted(set(sequences))
        for event in events:
            assert decode_live_state(pack("live.state", event)) == event


# --------------------------------------------------------------------------- H4: card, subscriber, worker


class TestCardModel:
    def test_reads_nested_dto(self, memory_factory):
        memory = memory_factory(
            modality="video",
            context_header="f13.mp4 > Scena",
            snippet_text="estratto breve",
            start_sec=1.0,
            end_sec=2.0,
        )
        model = app.card_model(make_result(memory))
        assert model["memory_id"] == memory.id
        assert model["title"] == "f13.mp4 > Scena"
        assert model["snippet"] == "estratto breve"
        assert model["format"] == "Video"
        assert model["matched"] == "agent_text"
        assert model["score"] == "1.80"
        assert model["boost"] is False

    def test_boost_flag_on_positive_synaptic(self, memory_factory):
        item = replace(make_result(memory_factory()), synaptic_boost=0.25)
        assert app.card_model(item)["boost"] is True


class TestEventsSubscriber:
    def test_poll_decodes_published_events(self, tmp_path):
        import time

        import zmq

        endpoint = f"ipc://{tmp_path}/events.sock"
        context = zmq.Context.instance()
        publisher = context.socket(zmq.PUB)
        publisher.bind(endpoint)
        subscriber = app.EventsSubscriber(endpoint)
        try:
            time.sleep(0.3)  # completare la subscribe prima di pubblicare
            event = make_event(1)
            publisher.send_string(pack("live.state", event))
            time.sleep(0.3)
            events = subscriber.poll()
            assert len(events) == 1
            assert events[0].sequence == 1 and events[0].session_id == "s1"
        finally:
            subscriber.close()
            publisher.close(0)

    def test_poll_skips_corrupt_payloads(self, tmp_path):
        import time

        import zmq

        endpoint = f"ipc://{tmp_path}/events.sock"
        context = zmq.Context.instance()
        publisher = context.socket(zmq.PUB)
        publisher.bind(endpoint)
        subscriber = app.EventsSubscriber(endpoint)
        try:
            time.sleep(0.3)
            publisher.send_string("non è json")
            publisher.send_string(pack("live.state", make_event(2)))
            time.sleep(0.3)
            events = subscriber.poll()
            assert [event.sequence for event in events] == [2]
        finally:
            subscriber.close()
            publisher.close(0)


class TestControlWorker:
    def test_delivers_response_through_callback(self, tmp_path):
        import json
        import threading

        import zmq

        endpoint = f"ipc://{tmp_path}/control.sock"
        context = zmq.Context.instance()
        rep = context.socket(zmq.REP)
        rep.bind(endpoint)
        stop = threading.Event()

        def serve():
            while not stop.is_set():
                if rep.poll(50):
                    request = json.loads(rep.recv_string())
                    response = make_response(make_event(9), command_id=request["payload"]["command_id"])
                    rep.send_string(pack("control.response", response))

        server = threading.Thread(target=serve)
        server.start()
        responses: list = []
        done = threading.Event()

        def on_response(response):
            responses.append(response)
            done.set()

        worker = app.ControlWorker(endpoint, on_response)
        try:
            command = HudCommand(command_id="cmd-worker-1", session_id=None, action="snapshot")
            worker.submit(command)
            assert done.wait(5.0), "risposta non arrivata"
        finally:
            worker.stop()
            stop.set()
            server.join(timeout=2.0)
            rep.close(0)
        assert responses[0].success and responses[0].state.sequence == 9
        assert responses[0].command_id == "cmd-worker-1"

    def test_unreachable_backend_becomes_failure_response(self, tmp_path):
        import threading

        endpoint = f"ipc://{tmp_path}/assente.sock"  # nessun REP: timeout × 2
        responses: list = []
        done = threading.Event()
        worker = app.ControlWorker(endpoint, lambda r: (responses.append(r), done.set()))
        try:
            command = HudCommand(command_id="cmd-worker-2", session_id=None, action="snapshot")
            worker.submit(command)
            assert done.wait(15.0), "risposta sintetica non arrivata"
        finally:
            worker.stop()
        assert responses[0].success is False
        assert responses[0].error_code == "CONTROL_ERROR"


# --------------------------------------------------------------------------- fix flash: click -> comando


class RecordingWorker:
    def __init__(self):
        self.commands: list[HudCommand] = []

    def submit(self, command):
        self.commands.append(command)


class StubSubscriber:
    def __init__(self, events):
        self.events = list(events)

    def poll(self):
        found, self.events = self.events, []
        return found


def make_logic(**overrides):
    worker = RecordingWorker()
    defaults = dict(settings=ExportSettings(), worker=worker, open_file=lambda memory: None)
    logic = app.HudLogic(**{**defaults, **overrides})
    return logic, worker


class TestClickToCommand:
    def test_pin_builds_command_with_current_session_and_marks_pending(self, memory_factory):
        memory = memory_factory()
        logic, worker = make_logic()
        logic.state.apply_event(make_event(1))
        command = logic.request_pin(memory.id)
        assert worker.commands == [command]
        assert command.action == "pin"
        assert command.session_id == "s1"
        assert command.memory_id == memory.id
        assert memory.id in logic.pending_pins.values()

    def test_pin_skipped_when_already_pinned(self, memory_factory):
        memory = memory_factory()
        logic, worker = make_logic()
        logic.state.apply_event(make_event(1, pinned=[make_pin(memory)], reason="pin"))
        assert logic.request_pin(memory.id) is None
        assert worker.commands == []

    def test_pin_skipped_while_pending(self, memory_factory):
        memory = memory_factory()
        logic, worker = make_logic()
        logic.state.apply_event(make_event(1))
        first = logic.request_pin(memory.id)
        assert first is not None
        assert logic.request_pin(memory.id) is None, "doppio click: un solo comando"
        assert len(worker.commands) == 1

    def test_response_clears_pending_pin_marker(self, memory_factory):
        memory = memory_factory()
        logic, _worker = make_logic()
        logic.state.apply_event(make_event(1))
        command = logic.request_pin(memory.id)
        response = make_response(make_event(2, pinned=[make_pin(memory)], reason="pin"), command_id=command.command_id)
        assert logic.handle_response(response) is True
        assert logic.pending_pins == {}, "il marker pendente si rilascia alla conferma"
        assert logic.request_pin(memory.id) is None, "ora risulta pinnato nello stato"

    def test_mic_toggle_reads_sensor_status(self):
        logic, worker = make_logic()
        logic.state.apply_event(make_event(1))  # sensor_status senza 'mic': on -> off
        command = logic.request_mic_toggle()
        assert command.action == "set_mic"
        assert command.enabled is False
        logic.state.apply_event(make_event(2, pinned=[], reason="status"))
        logic.state.event.sensor_status["mic"] = "off"
        command = logic.request_mic_toggle()
        assert command.enabled is True

    def test_open_missing_file_propagates_for_visible_error(self, tmp_path, memory_factory):
        logic, _worker = make_logic(open_file=None)  # apri il vero openers
        memory = memory_factory(file_path=str(tmp_path / "assente.md"))
        import pytest as _pytest

        with _pytest.raises(FileNotFoundError):
            logic.request_open(memory)

    def test_demo_without_worker_sends_nothing(self, memory_factory):
        logic = app.HudLogic(settings=ExportSettings(), worker=None)
        assert logic.request_pin("x") is None
        assert logic.request_unpin("x") is None
        assert logic.request_mic_toggle() is None
        assert logic.request_new_session() is None
        assert logic.start_discovery() is None


class TestTickRebuildsOnlyOnChange:
    def test_poll_events_reports_only_new_events(self):
        logic, _worker = make_logic(subscriber=StubSubscriber([make_event(1)]))
        assert logic.poll_events() is True
        assert logic.poll_events() is False, "nessun rebuild senza eventi nuovi"
        logic.subscriber = StubSubscriber([make_event(0)])
        assert logic.poll_events() is False, "evento vecchio scartato: niente rebuild"

    def test_view_signature_changes_only_on_real_change(self, memory_factory):
        memory = memory_factory()
        logic, _worker = make_logic()
        logic.state.apply_event(make_event(0, results=[make_result(memory)], source_last_seen={"agent_text": 1000.0}))
        assert logic.view_signature_changed(1005.0) is True, "prima osservazione: rebuild"
        assert logic.view_signature_changed(1005.0) is False, "niente rebuild a tick invariato"
        assert logic.view_signature_changed(1005.0 + DEFAULT_SOURCE_TTL_SEC + 1.0) is True, "card scaduta: rebuild"

    def test_discovery_uses_null_session(self):
        logic, worker = make_logic()
        command = logic.start_discovery()
        assert command.session_id is None and command.action == "snapshot"

    def test_mic_button_label_follows_sensor_status(self):
        logic, _worker = make_logic()
        logic.state.apply_event(make_event(1))  # sensor_status senza 'mic': default on
        assert logic.mic_label() == "Mic: on"
        logic.state.event.sensor_status["mic"] = "off"
        assert logic.mic_label() == "Mic: off"
        logic.state.event.sensor_status["mic"] = "error"
        assert logic.mic_label() == "Mic: error"


# --------------------------------------------------------------------------- fix toast overlay


class TestToastOverlay:
    def test_export_done_uses_toast_overlay(self):
        """Il toast va su un parent con add_toast (Adw.ToastOverlay), mai sulla window."""
        logic, _worker = make_logic()
        published: list = []
        logic.Adw = type(
            "AdwStub",
            (),
            {"Toast": type("Toast", (), {"new": staticmethod(lambda message: {"message": message})})},
        )
        logic.toast_overlay = type("OverlayStub", (), {"add_toast": lambda self, toast: published.append(toast)})()
        logic.publish_toast("contesto pronto: 0 warning")
        assert published == [{"message": "contesto pronto: 0 warning"}]

    def test_export_outcome_reports_path_clipboard_and_warnings(self):
        from src.contracts import ExportResult

        result = ExportResult(
            export_dir="/exports/s/e",
            markdown_path="/exports/s/e/context.md",
            manifest_path="/exports/s/e/manifest.json",
            clipboard_status="copied",
            warnings=["uno", "due"],
        )
        toast_message, status_text = app.export_outcome(result, None)
        assert "2 warning" in toast_message
        assert "/exports/s/e/context.md" in status_text
        assert "copied" in status_text

    def test_export_outcome_failure_is_explicit_not_a_fake_success(self):
        toast_message, status_text = app.export_outcome(None, RuntimeError("boom"))
        assert toast_message == ""
        assert "EXPORT_FAILED" in status_text
        assert "boom" in status_text


# --------------------------------------------------------------------------- F3-HUD: sistema/log, sessioni, promozioni


def session_memory(memory: MemoryView) -> MemoryView:
    """MemoryView di sessione: il campo additivo è ormai nei contratti."""
    return dataclasses.replace(memory, session_scope=True)


class TestHeaderBadges:
    def test_system_audio_and_logs_when_present(self):
        logic, _worker = make_logic()
        logic.state.apply_event(make_event(1))
        assert logic.header_badges() == ""
        logic.state.event.sensor_status["system_audio"] = "ok"
        logic.state.event.sensor_status["agent_logs"] = "3"
        assert logic.header_badges() == " · sistema ok · log 3"

    def test_partial_when_only_one_extra_source(self):
        logic, _worker = make_logic()
        logic.state.apply_event(make_event(1))
        logic.state.event.sensor_status["agent_logs"] = "7"
        assert logic.header_badges() == " · log 7"


class TestSessionCards:
    def test_card_model_flags_session_scope_with_badge_and_promote(self, memory_factory):
        model = app.card_model(make_result(session_memory(memory_factory())))
        assert model["session"] is True
        assert model["meta"].endswith(" · sessione")
        assert app.card_action_label(model, is_pinned=False) == "Promuovi"

    def test_regular_card_keeps_pin_and_no_badge(self, memory_factory):
        model = app.card_model(make_result(memory_factory()))
        assert model["session"] is False
        assert " · sessione" not in model["meta"]
        assert app.card_action_label(model, is_pinned=False) == "Pin"
        assert app.card_action_label(model, is_pinned=True) == "Pinnato"


class TestPromote:
    def test_request_promote_sends_command_and_marks_pending(self, memory_factory):
        memory = memory_factory()
        logic, worker = make_logic()
        logic.state.apply_event(make_event(1))
        command = logic.request_promote(memory.id)
        assert worker.commands == [command]
        assert command.action == "promote"
        assert command.memory_id == memory.id
        assert memory.id in logic.pending_promotes.values()

    def test_pending_promote_blocks_duplicate(self, memory_factory):
        memory = memory_factory()
        logic, worker = make_logic()
        logic.state.apply_event(make_event(1))
        assert logic.request_promote(memory.id) is not None
        assert logic.request_promote(memory.id) is None
        assert len(worker.commands) == 1

    def test_response_toasts_promote_outcome(self, memory_factory):
        memory = memory_factory()
        logic, _worker = make_logic()
        toasts: list[str] = []
        logic.notify = toasts.append
        logic.state.apply_event(make_event(1))
        command = logic.request_promote(memory.id)
        ok = make_response(make_event(2, reason="pin"), command_id=command.command_id)
        assert logic.resolve_response(ok) is True
        assert any("riuscita" in toast for toast in toasts)
        assert logic.pending_promotes == {}
        logic.state.apply_event(make_event(3))
        command = logic.request_promote(memory.id)
        ko = make_response(None, command_id=command.command_id, success=False)
        assert logic.resolve_response(ko) is False
        assert any("non riuscita" in toast for toast in toasts)

    def test_pin_response_does_not_toast_promote(self, memory_factory):
        memory = memory_factory()
        logic, _worker = make_logic()
        toasts: list[str] = []
        logic.notify = toasts.append
        logic.state.apply_event(make_event(1))
        command = logic.request_pin(memory.id)
        logic.resolve_response(
            make_response(make_event(2, pinned=[make_pin(memory)], reason="pin"), command_id=command.command_id)
        )
        assert toasts == []
