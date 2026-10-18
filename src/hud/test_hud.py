"""Test HUD H1 — openers e logica card senza display.

Nessuna GUI aperta: i test costruiscono argv ed eventi DTO senza eseguire
player né mostrare finestre (mark desktop riservato ai test GNOME in H4).
"""

import hashlib
from dataclasses import replace

import pytest

from src.config import DEFAULT_SOURCE_TTL_SEC
from src.contracts import (
    ContextSnapshot,
    ControlResponse,
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
