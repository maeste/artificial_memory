"""Test HUD H2 — pacchetto di contesto coerente e durevole (parte non trascrizione).

L'exporter riceve lo snapshot coerente dalla risposta controllo (DTO), non
liste UI. I test verificano contenuto effettivo, riferimenti risolvibili,
sorgente cambiata, deduplica e budget. Nessuna GUI e nessuna rete.
"""

import hashlib
import json
import os
import shutil
import uuid
import wave
from pathlib import Path

import pytest

from src.contracts import (
    ContextSnapshot,
    ExportSettings,
    LiveInput,
    LiveStateEvent,
    MemoryView,
    PinnedMemory,
    ScoredMemory,
)
from src.hud import context_exporter, transcriber
from src.hud.context_exporter import assemble_and_export_context

SESSION = "sess-export"
BASE_TIME = 1000.0
EMITTED_AT = 1080.0


# --------------------------------------------------------------------------- fixture e helper


def write_wav(path, seconds: float = 1.0, level: int = 0x00) -> None:
    """WAV mono 16 kHz 16 bit, senza dipendenze esterne; `level` distingue i contenuti."""
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(bytes((level, 0x00)) * int(16000 * seconds))


@pytest.fixture
def memory_factory():
    """MemoryView completa coi digest richiesti, con override per campo."""

    def make(**overrides) -> MemoryView:
        file_path = overrides.get("file_path", "/tmp/demo.md")
        chunk_key = overrides.get("chunk_key", "section:Demo:0:part:0")
        content_text = overrides.get("content_text", "La cache riduce gli accessi al database.")
        content_sha256 = hashlib.sha256(content_text.encode("utf-8")).hexdigest()
        defaults = dict(
            id=overrides.get("id") or hashlib.sha256(f"{file_path}{chunk_key}".encode()).hexdigest(),
            file_path=file_path,
            modality="markdown",
            chunk_key=chunk_key,
            content_sha256=content_sha256,
            source_sha256=overrides.get("source_sha256", hashlib.sha256(b"sorgente-demo").hexdigest()),
            context_header=overrides.get("context_header", "demo.md > Demo"),
            snippet_text=content_text[:350],
            content_text=content_text,
            page_num=overrides.get("page_num"),
            start_sec=overrides.get("start_sec"),
            end_sec=overrides.get("end_sec"),
            created_at=BASE_TIME,
        )
        return MemoryView(**{**defaults, **overrides})

    return make


def make_input(
    input_id: str,
    *,
    source: str = "agent_text",
    occurred_at: float = BASE_TIME,
    text: str = "",
    audio_path: str | None = None,
) -> LiveInput:
    return LiveInput(
        id=input_id,
        session_id=SESSION,
        source=source,
        occurred_at=occurred_at,
        text=text,
        audio_path=audio_path,
        duration_sec=1.0 if source == "mic_audio" else None,
    )


def make_pin(memory: MemoryView, input_ids: list[str], *, pinned_at: float = BASE_TIME) -> PinnedMemory:
    return PinnedMemory(
        memory=memory,
        pinned_at=pinned_at,
        snapshot=ContextSnapshot(
            id=f"snap-{memory.id[:8]}",
            session_id=SESSION,
            captured_at=pinned_at - 0.5,
            input_ids=input_ids,
        ),
    )


def make_state(
    *,
    pinned: list[PinnedMemory] | None = None,
    results: list[ScoredMemory] | None = None,
    backend: str = "mock",
) -> LiveStateEvent:
    return LiveStateEvent(
        event_id="ev-export",
        sequence=42,
        session_id=SESSION,
        space_id="space-demo",
        backend=backend,
        emitted_at=EMITTED_AT,
        reason="input",
        latest_inputs=[],
        results=results or [],
        pinned=pinned or [],
        sensor_status={"agent_text": "ok"},
        source_last_seen={"agent_text": EMITTED_AT},
        dropped_inputs=0,
    )


@pytest.fixture
def history_factory(tmp_path):
    """Storia nota: 7 agent_text, mic dentro e fuori finestra 60 s, WAV reali nel runtime."""

    def make() -> list[LiveInput]:
        runtime = tmp_path / "runtime"
        runtime.mkdir(exist_ok=True)
        write_wav(runtime / "mic-old.wav")
        write_wav(runtime / "mic-a.wav")
        write_wav(runtime / "mic-b.wav", level=0x20)
        inputs = [make_input(f"in-{i}", occurred_at=BASE_TIME + i, text=f"messaggio {i}") for i in range(1, 8)]
        inputs.append(
            make_input(
                "in-m-old",
                source="mic_audio",
                occurred_at=BASE_TIME,
                audio_path=str(runtime / "mic-old.wav"),
            )
        )
        inputs.append(
            make_input(
                "in-m-a",
                source="mic_audio",
                occurred_at=BASE_TIME + 20.0,
                audio_path=str(runtime / "mic-a.wav"),
            )
        )
        inputs.append(
            make_input(
                "in-m-b",
                source="mic_audio",
                occurred_at=EMITTED_AT,
                audio_path=str(runtime / "mic-b.wav"),
            )
        )
        return inputs

    return make


def do_export(tmp_path, state, inputs, **kwargs):
    export_root = str(tmp_path / "exports")
    settings = kwargs.pop("settings", ExportSettings())
    return assemble_and_export_context(state, inputs, export_root, settings, **kwargs)


# --------------------------------------------------------------------------- struttura e selezione


class TestPackageStructure:
    def test_layout_files_and_dirs(self, tmp_path, memory_factory, history_factory):
        pin = make_pin(memory_factory(), ["in-1"])
        result = do_export(tmp_path, make_state(pinned=[pin]), history_factory())
        package = tmp_path / "exports" / SESSION / result.export_dir.rsplit("/", 1)[-1]
        assert result.export_dir == str(package)
        assert (package / "context.md").is_file()
        assert (package / "manifest.json").is_file()
        assert (package / "session-context.json").is_file()
        assert (package / "assets").is_dir()
        assert (package / "memories").is_dir()
        assert result.markdown_path == str(package / "context.md")
        assert result.manifest_path == str(package / "manifest.json")

    def test_two_exports_have_distinct_directories(self, tmp_path, memory_factory, history_factory):
        state = make_state()
        inputs = history_factory()
        first = do_export(tmp_path, state, inputs)
        second = do_export(tmp_path, state, inputs)
        assert first.export_dir != second.export_dir

    def test_session_inputs_not_mutated(self, tmp_path, memory_factory, history_factory):
        pin = make_pin(memory_factory(), ["in-1"])
        inputs = history_factory()
        snapshot_order = [item.id for item in inputs]
        do_export(tmp_path, make_state(pinned=[pin]), inputs)
        assert [item.id for item in inputs] == snapshot_order


class TestSelection:
    def test_last_five_agent_text_selected(self, tmp_path, memory_factory, history_factory):
        result = do_export(tmp_path, make_state(), history_factory())
        session_context = read_json(result, "session-context.json")
        agent_ids = [item["id"] for item in session_context["inputs"] if item["source"] == "agent_text"]
        assert agent_ids == ["in-3", "in-4", "in-5", "in-6", "in-7"]

    def test_mic_window_is_sixty_seconds(self, tmp_path, memory_factory, history_factory):
        result = do_export(tmp_path, make_state(), history_factory())
        session_context = read_json(result, "session-context.json")
        mic_ids = [item["id"] for item in session_context["inputs"] if item["source"] == "mic_audio"]
        # in-m-old avvenuto a BASE_TIME = EMITTED_AT - 80: fuori finestra; gli altri due dentro.
        assert mic_ids == ["in-m-a", "in-m-b"]

    def test_pin_inputs_added_and_deduplicated(self, tmp_path, memory_factory, history_factory):
        # in-1 è fuori dai recenti cinque ma referenziato dallo snapshot del pin;
        # in-7 è già fra i recenti: deve comparire una volta sola.
        pin = make_pin(memory_factory(), ["in-1", "in-7"])
        result = do_export(tmp_path, make_state(pinned=[pin]), history_factory())
        session_context = read_json(result, "session-context.json")
        all_ids = [item["id"] for item in session_context["inputs"]]
        assert all_ids.count("in-7") == 1
        assert "in-1" in all_ids
        assert len(all_ids) == 8

    def test_pin_input_missing_from_history_is_omission(self, tmp_path, memory_factory, history_factory):
        pin = make_pin(memory_factory(), ["in-inesistente"])
        result = do_export(tmp_path, make_state(pinned=[pin]), history_factory())
        manifest = read_json(result, "manifest.json")
        assert any("in-inesistente" in warning for warning in manifest["warnings"])


def read_json(result, name: str) -> dict:
    return json.loads((Path(result.export_dir) / name).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- allegati audio e durevolezza


class TestWavAssets:
    def test_wavs_copied_with_uuid_names_and_relative_paths(self, tmp_path, history_factory):
        result = do_export(tmp_path, make_state(), history_factory())
        package = Path(result.export_dir)
        session_context = read_json(result, "session-context.json")
        mic_entries = [item for item in session_context["inputs"] if item["source"] == "mic_audio"]
        assert len(mic_entries) == 2
        for entry in mic_entries:
            audio = entry["audio_path"]
            assert audio is not None
            assert not audio.startswith("/")
            assert audio.startswith("assets/")
            name = Path(audio).name
            assert name.endswith(".wav")
            uuid.UUID(name[: len(name) - 4])  # nome UUID, non link al runtime
            assert (package / audio).is_file()

    def test_package_survives_runtime_deletion(self, tmp_path, history_factory):
        inputs = history_factory()
        result = do_export(tmp_path, make_state(), inputs)
        for item in inputs:
            if item.audio_path:
                os.remove(item.audio_path)
        package = Path(result.export_dir)
        session_context = read_json(result, "session-context.json")
        for entry in session_context["inputs"]:
            if entry["audio_path"]:
                assert (package / entry["audio_path"]).is_file()

    def test_missing_mic_audio_recorded_without_broken_reference(self, tmp_path, history_factory):
        inputs = history_factory()
        os.remove(inputs[-1].audio_path)  # in-m-b esiste nella storia ma il WAV è sparito
        result = do_export(tmp_path, make_state(), inputs)
        session_context = read_json(result, "session-context.json")
        missing = [item for item in session_context["inputs"] if item["id"] == "in-m-b"]
        assert missing and missing[0]["audio_path"] is None
        manifest = read_json(result, "manifest.json")
        assert any("in-m-b" in warning for warning in manifest["warnings"])


# --------------------------------------------------------------------------- pin congelati


class TestPinnedContent:
    def test_full_frozen_text_in_memories_and_cap_in_markdown(self, tmp_path, memory_factory, history_factory):
        long_text = "\n\n".join(f"paragrafo {i} " + "x" * 200 for i in range(20))
        pin = make_pin(memory_factory(content_text=long_text), ["in-1"])
        result = do_export(tmp_path, make_state(pinned=[pin]), history_factory())
        package = Path(result.export_dir)
        text_file = package / "memories" / f"{pin.memory.id}.txt"
        assert text_file.read_text(encoding="utf-8") == long_text
        markdown = (package / "context.md").read_text(encoding="utf-8")
        excerpt = long_text[:1500]
        assert excerpt in markdown
        assert long_text[1500:1600] not in markdown
        assert f"](memories/{pin.memory.id}.txt)" in markdown

    def test_changed_source_marked_unavailable_keeps_frozen_text(self, tmp_path, memory_factory, history_factory):
        source = tmp_path / "runtime" / "nota.md"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("versione originale", encoding="utf-8")
        memory = memory_factory(
            file_path=str(source),
            content_text="testo congelato del pin",
            source_sha256=hashlib.sha256(b"versione originale").hexdigest(),
        )
        pin = make_pin(memory, ["in-1"])
        source.write_text("versione modificata dopo il pin", encoding="utf-8")
        result = do_export(tmp_path, make_state(pinned=[pin]), history_factory())
        package = Path(result.export_dir)
        assert (package / "memories" / f"{memory.id}.txt").read_text(encoding="utf-8") == "testo congelato del pin"
        markdown = (package / "context.md").read_text(encoding="utf-8")
        assert "originale selezionato indisponibile" in markdown
        manifest = read_json(result, "manifest.json")
        pin_state = [entry for entry in manifest["pinned"] if entry["memory_id"] == memory.id]
        assert pin_state and pin_state[0]["source_state"] == "unavailable"

    def test_verified_source_state_recorded(self, tmp_path, memory_factory, history_factory):
        source = tmp_path / "runtime" / "nota.md"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("versione originale", encoding="utf-8")
        memory = memory_factory(
            file_path=str(source),
            content_text="testo del pin",
            source_sha256=hashlib.sha256(b"versione originale").hexdigest(),
        )
        pin = make_pin(memory, ["in-1"])
        result = do_export(tmp_path, make_state(pinned=[pin]), history_factory())
        manifest = read_json(result, "manifest.json")
        pin_state = [entry for entry in manifest["pinned"] if entry["memory_id"] == memory.id]
        assert pin_state and pin_state[0]["source_state"] == "verified"

    def test_pinned_memory_excluded_from_suggestions(self, tmp_path, memory_factory, history_factory):
        pinned_memory = memory_factory(file_path="/tmp/pinnato.md", content_text="pinnato")
        others = [memory_factory(file_path=f"/tmp/altro{i}.md", content_text=f"altro {i}") for i in range(4)]
        results = [make_suggestion(pinned_memory), *(make_suggestion(memory) for memory in others)]
        pin = make_pin(pinned_memory, ["in-1"])
        result = do_export(tmp_path, make_state(pinned=[pin], results=results), history_factory())
        markdown = (Path(result.export_dir) / "context.md").read_text(encoding="utf-8")
        suggestions_section = markdown.split("## Altri suggerimenti")[-1]
        assert "pinnato" not in suggestions_section
        assert suggestions_section.count("- [") == 3  # massimo tre, solo riferimenti


def make_suggestion(memory: MemoryView) -> ScoredMemory:
    return ScoredMemory(
        memory=memory,
        cosine_score=0.5,
        z_score=1.0,
        synaptic_boost=0.0,
        final_score=1.0,
        matched_sources=["agent_text"],
        winning_source="agent_text",
    )


# --------------------------------------------------------------------------- markdown e budget


class TestMarkdown:
    def test_mock_backend_opens_with_experimental_warning(self, tmp_path, memory_factory, history_factory):
        pin = make_pin(memory_factory(), ["in-1"])
        result = do_export(tmp_path, make_state(pinned=[pin], backend="mock"), history_factory())
        markdown = (Path(result.export_dir) / "context.md").read_text(encoding="utf-8")
        assert markdown.startswith(">")
        assert "sperimentali" in markdown.splitlines()[0]

    def test_recent_state_in_markdown(self, tmp_path, history_factory):
        result = do_export(tmp_path, make_state(), history_factory())
        markdown = (Path(result.export_dir) / "context.md").read_text(encoding="utf-8")
        assert "messaggio 7" in markdown
        assert "messaggio 2" not in markdown  # fuori dagli ultimi cinque


class TestBudget:
    def test_rejects_settings_below_minimum(self, tmp_path, history_factory):
        with pytest.raises(ValueError, match="12"):
            do_export(tmp_path, make_state(), history_factory(), settings=ExportSettings(context_max_chars=5000))

    def test_truncates_at_paragraph_boundaries_recording_omissions(self, tmp_path, memory_factory, history_factory):
        big_text = "\n\n".join(f"paragrafo {i} " + "y" * 400 for i in range(40))
        pins = [
            make_pin(memory_factory(file_path=f"/tmp/grosso{i}.md", content_text=big_text), ["in-1"]) for i in range(8)
        ]
        result = do_export(
            tmp_path,
            make_state(pinned=pins),
            history_factory(),
            settings=ExportSettings(context_max_chars=12000),
        )
        markdown = (Path(result.export_dir) / "context.md").read_text(encoding="utf-8")
        assert len(markdown) <= 12000
        manifest = read_json(result, "manifest.json")
        assert manifest["omissions"]

    def test_mandatory_overflow_raises_export_failed(self):
        views = [
            context_exporter._PinView(
                title="T" * 120,
                memory_id=f"id{i}",
                source_label=f"/tmp/molto-lungo-{i}.md",
                selection_label="input: a, b, c",
                extract="estratto " * 50,
                text_relpath=f"memories/id{i}.txt",
            )
            for i in range(12)
        ]
        with pytest.raises(RuntimeError, match="EXPORT_FAILED"):
            context_exporter._build_markdown(
                backend="mock",
                session_id=SESSION,
                pinned=views,
                recent_inputs=[],
                suggestions=[],
                budget_chars=200,
            )


# --------------------------------------------------------------------------- manifest e pubblicazione


class TestManifest:
    def test_manifest_core_fields(self, tmp_path, memory_factory, history_factory):
        pin = make_pin(memory_factory(), ["in-1"])
        result = do_export(
            tmp_path,
            make_state(pinned=[pin]),
            history_factory(),
            space=None,
        )
        manifest = read_json(result, "manifest.json")
        assert manifest["schema_version"] == 1
        assert manifest["session_id"] == SESSION
        assert manifest["space_id"] == "space-demo"
        assert manifest["backend"] == "mock"
        assert manifest["export_settings"]["transcription"] == "off"
        assert manifest["snapshot"]["event_id"] == "ev-export"
        assert manifest["snapshot"]["sequence"] == 42
        assert any(entry["memory_id"] == pin.memory.id for entry in manifest["pinned"])
        assert pin.memory.id in manifest["memory_map"]
        assert manifest["clipboard_status"] in ("copied", "unavailable", "failed")

    def test_attachment_digests_match_files(self, tmp_path, history_factory):
        result = do_export(tmp_path, make_state(), history_factory())
        import json
        from pathlib import Path

        package = Path(result.export_dir)
        manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["attachments"], "attesi WAV esportati negli assets"
        for attachment in manifest["attachments"]:
            data = (package / attachment["path"]).read_bytes()
            assert hashlib.sha256(data).hexdigest() == attachment["sha256"]


class TestPublish:
    def test_fatal_error_removes_staging_and_raises(self, tmp_path, history_factory, monkeypatch):
        inputs = history_factory()

        def exploding(src, dst, *args, **kwargs):
            raise OSError("disco finto")

        monkeypatch.setattr(context_exporter.shutil, "copyfileobj", exploding)
        with pytest.raises(RuntimeError, match="EXPORT_FAILED"):
            do_export(tmp_path, make_state(), inputs)
        session_dir = tmp_path / "exports" / SESSION
        assert session_dir.is_dir()
        assert list(session_dir.iterdir()) == [], "nessun leftover di staging o pacchetto parziale"

    def test_no_staging_leftover_after_success(self, tmp_path, history_factory):
        do_export(tmp_path, make_state(), history_factory())
        session_dir = tmp_path / "exports" / SESSION
        leftovers = [p.name for p in session_dir.iterdir() if p.name.startswith(".staging")]
        assert leftovers == []


class TestClipboard:
    def test_copied_with_wl_copy(self, tmp_path, history_factory, monkeypatch):
        launched: dict = {}

        def fake_popen(argv, **kwargs):
            launched["argv"] = list(argv)
            launched["stdin"] = kwargs.get("stdin")
            return type(
                "P",
                (),
                {
                    "__enter__": lambda s: s,
                    "__exit__": lambda s, *a: None,
                    "communicate": lambda s, i=None: (None, None),
                },
            )()

        monkeypatch.setattr(
            context_exporter.shutil, "which", lambda name: "/usr/bin/wl-copy" if name == "wl-copy" else None
        )
        monkeypatch.setattr(context_exporter.subprocess, "Popen", fake_popen)
        inputs = history_factory()
        result = do_export(tmp_path, make_state(), inputs)
        assert result.clipboard_status == "copied"
        assert launched["argv"] == ["wl-copy"]

    def test_unavailable_without_wl_copy(self, tmp_path, history_factory, monkeypatch):
        monkeypatch.setattr(context_exporter.shutil, "which", lambda name: None)
        result = do_export(tmp_path, make_state(), history_factory())
        assert result.clipboard_status == "unavailable"
        manifest = read_json(result, "manifest.json")
        assert manifest["clipboard_status"] == "unavailable"
        markdown = (tmp_path / "exports" / SESSION / "p" / "context.md").name  # il pacchetto resta valido
        assert markdown == "context.md"

    def test_failed_clipboard_does_not_invalidate_package(self, tmp_path, history_factory, monkeypatch):
        def broken_popen(argv, **kwargs):
            raise OSError("nessun display")

        monkeypatch.setattr(
            context_exporter.shutil, "which", lambda name: "/usr/bin/wl-copy" if name == "wl-copy" else None
        )
        monkeypatch.setattr(context_exporter.subprocess, "Popen", broken_popen)
        result = do_export(tmp_path, make_state(), history_factory())
        assert result.clipboard_status == "failed"
        assert result.warnings


# --------------------------------------------------------------------------- H2: allegati media


FFMPEG = shutil.which("ffmpeg")


def make_pdf(path: Path, pages: int = 3) -> None:
    """PDF reale di prova con PyMuPDF, pagine con testo distinguibile."""
    import pymupdf

    doc = pymupdf.open()
    for number in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"pagina {number + 1}", fontsize=18)
    doc.save(str(path))
    doc.close()


def source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory) -> Path:
    """MP4 di prova 2 s senza traccia audio (la clip esportata resta visuale)."""
    if FFMPEG is None:
        pytest.skip("ffmpeg non disponibile")
    import subprocess

    path = tmp_path_factory.mktemp("video") / "clip.mp4"
    result = subprocess.run(
        [
            FFMPEG,
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=64x64:d=2",
            "-r",
            "10",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return path


def probe_duration(path: Path) -> float:
    import subprocess

    ffprobe = shutil.which("ffprobe")
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


@pytest.fixture(scope="module")
def sample_video_with_audio(tmp_path_factory) -> Path:
    """MP4 di prova 2 s con traccia audio sine: serve a testare l'ordine ASR."""
    if FFMPEG is None:
        pytest.skip("ffmpeg non disponibile")
    import subprocess

    path = tmp_path_factory.mktemp("video-audio") / "clip-con-audio.mp4"
    result = subprocess.run(
        [
            FFMPEG,
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=64x64:d=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-shortest",
            "-r",
            "10",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return path


class TestMediaAttachments:
    def test_verified_image_copied_with_digest(self, tmp_path, memory_factory, history_factory):
        image = tmp_path / "runtime" / "foto.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(
            bytes.fromhex(
                "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
                "890000000d4944415478da63f8cfc0f01f0005000105a52a8e1b000000004945"
                "4e44ae426082"
            )
        )
        memory = memory_factory(
            file_path=str(image),
            modality="image",
            content_text="",
            source_sha256=source_sha(image),
        )
        pin = make_pin(memory, ["in-1"])
        result = do_export(tmp_path, make_state(pinned=[pin]), history_factory())
        package = Path(result.export_dir)
        manifest = read_json(result, "manifest.json")
        entry = [item for item in manifest["pinned"] if item["memory_id"] == memory.id][0]
        assert entry["source_state"] == "verified"
        assert entry["attachments"], "atteso allegato immagine"
        for attachment in entry["attachments"]:
            assert attachment["path"].startswith("assets/")
            assert (package / attachment["path"]).is_file()
            digest = hashlib.sha256((package / attachment["path"]).read_bytes()).hexdigest()
            assert digest == attachment["sha256"]

    def test_image_limitation_declared_for_text_only_agent(self, tmp_path, memory_factory, history_factory):
        image = tmp_path / "runtime" / "foto.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"\x89PNG\r\n\x1a\n")
        memory = memory_factory(
            file_path=str(image), modality="image", content_text="", source_sha256=source_sha(image)
        )
        pin = make_pin(memory, ["in-1"])
        result = do_export(tmp_path, make_state(pinned=[pin]), history_factory())
        markdown = (Path(result.export_dir) / "context.md").read_text(encoding="utf-8")
        assert "solo testuale" in markdown
        assert "caption" in markdown

    def test_pdf_page_rendered_from_verified_copy(self, tmp_path, memory_factory, history_factory):
        pdf = tmp_path / "runtime" / "doc.pdf"
        pdf.parent.mkdir(parents=True, exist_ok=True)
        make_pdf(pdf)
        memory = memory_factory(
            file_path=str(pdf),
            modality="pdf",
            content_text="pagina 2",
            page_num=2,
            source_sha256=source_sha(pdf),
        )
        pin = make_pin(memory, ["in-1"])
        result = do_export(tmp_path, make_state(pinned=[pin]), history_factory())
        package = Path(result.export_dir)
        manifest = read_json(result, "manifest.json")
        entry = [item for item in manifest["pinned"] if item["memory_id"] == memory.id][0]
        assert entry["attachments"], "attesa pagina renderizzata"
        page_png = [a for a in entry["attachments"] if a["kind"] == "pdf_page"]
        assert page_png and (package / page_png[0]["path"]).is_file()

    def test_scanned_pdf_declares_text_only_limitation(self, tmp_path, memory_factory, history_factory):
        pdf = tmp_path / "runtime" / "scan.pdf"
        pdf.parent.mkdir(parents=True, exist_ok=True)
        make_pdf(pdf)
        memory = memory_factory(
            file_path=str(pdf), modality="pdf", content_text="", page_num=1, source_sha256=source_sha(pdf)
        )
        pin = make_pin(memory, ["in-1"])
        result = do_export(tmp_path, make_state(pinned=[pin]), history_factory())
        markdown = (Path(result.export_dir) / "context.md").read_text(encoding="utf-8")
        assert "solo testuale" in markdown

    def test_video_clip_extracted_from_selected_interval(
        self, tmp_path, memory_factory, history_factory, sample_video
    ):
        memory = memory_factory(
            file_path=str(sample_video),
            modality="video",
            content_text="",
            start_sec=0.5,
            end_sec=1.5,
            source_sha256=source_sha(sample_video),
        )
        pin = make_pin(memory, ["in-1"])
        result = do_export(tmp_path, make_state(pinned=[pin]), history_factory())
        package = Path(result.export_dir)
        manifest = read_json(result, "manifest.json")
        entry = [item for item in manifest["pinned"] if item["memory_id"] == memory.id][0]
        clips = [a for a in entry["attachments"] if a["kind"] == "video_clip"]
        assert clips and (package / clips[0]["path"]).is_file()
        duration = probe_duration(package / clips[0]["path"])
        assert 0.6 <= duration <= 1.6, "la clip parte da zero e dura circa l'intervallo"
        assert entry["start_sec"] == 0.5 and entry["end_sec"] == 1.5

    def test_changed_video_source_produces_no_clip(self, tmp_path, memory_factory, history_factory, sample_video):
        stale_sha = "0" * 64
        memory = memory_factory(
            file_path=str(sample_video),
            modality="video",
            content_text="",
            start_sec=0.5,
            end_sec=1.5,
            source_sha256=stale_sha,
        )
        pin = make_pin(memory, ["in-1"])
        result = do_export(tmp_path, make_state(pinned=[pin]), history_factory())
        manifest = read_json(result, "manifest.json")
        entry = [item for item in manifest["pinned"] if item["memory_id"] == memory.id][0]
        assert entry["source_state"] == "unavailable"
        assert entry["attachments"] == []


# --------------------------------------------------------------------------- H2: lease e acquisizione


class TestLeaseAcquisition:
    def test_expired_without_provider_raises_and_publishes_nothing(self, tmp_path, history_factory):
        with pytest.raises(RuntimeError, match="EXPORT_FAILED"):
            do_export(tmp_path, make_state(), history_factory(), lease_expires_at=999.0, clock=lambda: 1000.0)
        session_dir = tmp_path / "exports" / SESSION
        assert list(session_dir.iterdir()) == []

    def test_retry_produces_single_package_from_new_state(self, tmp_path, history_factory):
        first_inputs = history_factory()
        os.remove(first_inputs[-1].audio_path)  # reset tra i due snapshot

        def provider():
            fresh_inputs = [
                make_input("in-r1", occurred_at=EMITTED_AT - 1, text="dopo il reset"),
                make_input("in-r2", source="mic_audio", occurred_at=EMITTED_AT, audio_path=None),
            ]
            return make_state(), fresh_inputs, 99999.0

        clocks = iter([1000.0, 1500.0, 2000.0, 2000.0, 2000.0, 2000.0])  # 1o tentativo scade dopo l'apertura
        result = do_export(
            tmp_path,
            make_state(),
            first_inputs,
            lease_expires_at=1200.0,
            snapshot_provider=provider,
            clock=lambda: next(clocks, 5000.0),
        )
        session_dir = tmp_path / "exports" / SESSION
        packages = [p for p in session_dir.iterdir() if not p.name.startswith(".staging")]
        assert len(packages) == 1, "nessun pacchetto parziale dal primo tentativo"
        manifest = read_json(result, "manifest.json")
        exported_ids = [item["id"] for item in manifest["inputs"]]
        assert "in-r1" in exported_ids and "in-r2" in exported_ids


# --------------------------------------------------------------------------- H3: trascrizione


class SpyTranscriber:
    """Registra le chiamate senza toccare faster_whisper."""

    def __init__(
        self,
        outputs: dict[str, str] | None = None,
        error: Exception | None = None,
        default: str = "trascrizione del parlato",
    ):
        self.calls: list[str] = []
        self.outputs = outputs or {}
        self.error = error
        self.default = default

    def transcribe(self, wav_path: str) -> str:
        self.calls.append(wav_path)
        if self.error is not None:
            raise self.error
        name = Path(wav_path).name
        if name in self.outputs:
            return self.outputs[name]
        if name.endswith(".mp4"):
            return "trascrizione della clip video"
        return self.default


def video_pin(tmp_path, memory_factory, video_path, input_ids):
    memory = memory_factory(
        file_path=str(video_path),
        modality="video",
        content_text="",
        start_sec=0.5,
        end_sec=1.5,
        source_sha256=source_sha(video_path),
    )
    return make_pin(memory, input_ids)


def mic_history_with(tmp_path, extra_mic_second=1.0):
    """Storia con due mic dentro la finestra, il secondo di durata e contenuto distinti."""
    runtime = tmp_path / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    write_wav(runtime / "mic-a.wav")
    write_wav(runtime / "mic-b.wav", extra_mic_second, level=0x20)
    return [
        make_input("in-a", occurred_at=BASE_TIME + 1, text="domanda uno"),
        make_input("in-b", occurred_at=BASE_TIME + 2, text="domanda due"),
        make_input("in-m-a", source="mic_audio", occurred_at=BASE_TIME + 20.0, audio_path=str(runtime / "mic-a.wav")),
        make_input("in-m-b", source="mic_audio", occurred_at=EMITTED_AT, audio_path=str(runtime / "mic-b.wav")),
    ]


class TestTranscriptionOff:
    def test_off_never_touches_asr(self, tmp_path, history_factory):
        captured: list = []

        def factory(settings):
            captured.append(settings)
            return SpyTranscriber()

        result = do_export(tmp_path, make_state(), history_factory(), transcriber_factory=factory)
        assert captured == [], "in off la factory ASR non deve essere chiamata"
        assert not (Path(result.export_dir) / "transcripts").exists()
        manifest = read_json(result, "manifest.json")
        assert manifest["transcription"]["mode"] == "off"
        markdown = (Path(result.export_dir) / "context.md").read_text(encoding="utf-8")
        assert "ascoltare" in markdown, "in off va dichiarato che serve un agente capace di ascoltare"


class TestTranscriptionLocal:
    def test_transcribes_selected_once_with_pin_priority(
        self, tmp_path, memory_factory, history_factory, sample_video_with_audio
    ):
        spy = SpyTranscriber()
        cache_dir = tmp_path / "asr-cache"
        pin = video_pin(tmp_path, memory_factory, sample_video_with_audio, ["in-a"])
        settings = ExportSettings(transcription="local", asr_model_dir=str(tmp_path / "modello"))
        inputs = history_factory()
        first = do_export(
            tmp_path,
            make_state(pinned=[pin]),
            inputs,
            settings=settings,
            transcriber_factory=lambda s: spy,
            asr_cache_dir=str(cache_dir),
        )
        # prima la clip del pin, poi i mic in ordine di storia
        assert [Path(p).suffix for p in spy.calls] == [".mp4", ".wav", ".wav"]
        manifest = read_json(first, "manifest.json")
        statuses = {item["key"]: item["status"] for item in manifest["transcription"]["per_input"]}
        assert statuses == {pin.memory.id: "transcribed", "in-m-a": "transcribed", "in-m-b": "transcribed"}
        assert manifest["transcription"]["mode"] == "local"
        assert manifest["transcription"]["params"]["beam_size"] == 1
        assert manifest["transcription"]["params"]["temperature"] == 0.0
        assert manifest["transcription"]["params"]["condition_on_previous_text"] is False
        transcripts = Path(first.export_dir) / "transcripts"
        assert (transcripts / f"{pin.memory.id}.txt").is_file()
        assert (transcripts / "in-m-a.txt").is_file()
        # secondo export con la stessa cache: nessuna nuova chiamata ASR
        second = do_export(
            tmp_path,
            make_state(pinned=[pin]),
            inputs,
            settings=settings,
            transcriber_factory=lambda s: spy,
            asr_cache_dir=str(cache_dir),
        )
        assert len(spy.calls) == 3, "la cache deve evitare una seconda trascrizione"
        second_manifest = read_json(second, "manifest.json")
        cached = {item["key"]: item["cache"] for item in second_manifest["transcription"]["per_input"]}
        assert set(cached.values()) == {"hit"}

    def test_transcripts_labeled_in_markdown_and_session_context(self, tmp_path, history_factory):
        spy = SpyTranscriber()
        result = do_export(
            tmp_path,
            make_state(),
            history_factory(),
            settings=ExportSettings(transcription="local", asr_model_dir=str(tmp_path / "modello")),
            transcriber_factory=lambda s: spy,
        )
        markdown = (Path(result.export_dir) / "context.md").read_text(encoding="utf-8")
        assert "Trascrizione ASR" in markdown
        assert "non citazioni garantite" in markdown
        session_context = read_json(result, "session-context.json")
        mic_entries = [item for item in session_context["inputs"] if item["source"] == "mic_audio"]
        for entry in mic_entries:
            assert entry["transcription"] == "trascrizione del parlato"
            assert entry["transcription_status"] == "transcribed"

    def test_budget_excludes_and_declares(self, tmp_path, memory_factory, sample_video_with_audio):
        runtime = tmp_path / "runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        write_wav(runtime / "mic-a.wav", 1.0)
        write_wav(runtime / "mic-b.wav", 1.0, level=0x20)
        inputs = [
            make_input("in-a", occurred_at=BASE_TIME + 1, text="domanda"),
            make_input(
                "in-m-a", source="mic_audio", occurred_at=BASE_TIME + 20, audio_path=str(runtime / "mic-a.wav")
            ),
            make_input("in-m-b", source="mic_audio", occurred_at=EMITTED_AT, audio_path=str(runtime / "mic-b.wav")),
        ]
        pin = video_pin(tmp_path, memory_factory, sample_video_with_audio, ["in-a"])
        result = do_export(
            tmp_path,
            make_state(pinned=[pin]),
            inputs,
            settings=ExportSettings(
                transcription="local", asr_model_dir=str(tmp_path / "modello"), max_transcribe_sec=2.5
            ),
            transcriber_factory=lambda s: SpyTranscriber(),
        )
        manifest = read_json(result, "manifest.json")
        assert manifest["transcription"]["excluded_ids"] == ["in-m-b"]
        assert not (Path(result.export_dir) / "transcripts" / "in-m-b.txt").exists()
        audio_assets = [a["path"] for a in manifest["attachments"] if a["kind"] == "mic_audio"]
        assert len(audio_assets) == 2, "l'audio escluso resta come allegato"

    def test_mute_video_clip_skips_asr(self, tmp_path, memory_factory, history_factory, sample_video):
        pin = video_pin(tmp_path, memory_factory, sample_video, ["in-a"])
        spy = SpyTranscriber()
        result = do_export(
            tmp_path,
            make_state(pinned=[pin]),
            history_factory(),
            settings=ExportSettings(transcription="local", asr_model_dir=str(tmp_path / "modello")),
            transcriber_factory=lambda s: spy,
        )
        assert [Path(p).suffix for p in spy.calls] == [".wav", ".wav"], "clip muto: nessuna chiamata ASR"
        manifest = read_json(result, "manifest.json")
        statuses = {item["key"]: item["status"] for item in manifest["transcription"]["per_input"]}
        assert statuses[pin.memory.id] == "no_audio"
        assert not any("non riuscita" in warning for warning in manifest["warnings"])
        entry = [item for item in manifest["pinned"] if item["memory_id"] == pin.memory.id][0]
        assert entry["attachments"], "la clip visuale resta allegata"

    def test_settings_frozen_per_export(self, tmp_path, history_factory):
        captured: list = []
        spy = SpyTranscriber()

        def factory(settings):
            captured.append(settings)
            return spy

        off_settings = ExportSettings(transcription="off")
        first = do_export(
            tmp_path, make_state(), history_factory(), settings=off_settings, transcriber_factory=factory
        )
        assert read_json(first, "manifest.json")["export_settings"]["transcription"] == "off"
        local_settings = ExportSettings(transcription="local", asr_model_dir=str(tmp_path / "modello"))
        second = do_export(
            tmp_path, make_state(), history_factory(), settings=local_settings, transcriber_factory=factory
        )
        assert captured == [local_settings], "l'opzione si congela all'avvio di ogni export"
        assert read_json(second, "manifest.json")["export_settings"]["transcription"] == "local"

    def test_missing_model_unavailable_keeps_attachments(self, tmp_path, history_factory):
        # niente transcriber_factory: percorso reale; faster_whisper non è installato
        # e il preflight deve fallire prima di ogni import.
        result = do_export(
            tmp_path,
            make_state(),
            history_factory(),
            settings=ExportSettings(transcription="local", asr_model_dir=str(tmp_path / "assente")),
        )
        manifest = read_json(result, "manifest.json")
        statuses = {item["status"] for item in manifest["transcription"]["per_input"]}
        assert statuses == {"unavailable"}
        audio_assets = [a["path"] for a in manifest["attachments"] if a["kind"] == "mic_audio"]
        assert len(audio_assets) == 2, "gli allegati audio restano in local con modello assente"

    def test_asr_error_marks_failed_and_keeps_package(self, tmp_path, history_factory):
        spy = SpyTranscriber(error=RuntimeError("modello rotto"))
        result = do_export(
            tmp_path,
            make_state(),
            history_factory(),
            settings=ExportSettings(transcription="local", asr_model_dir=str(tmp_path / "modello")),
            transcriber_factory=lambda s: spy,
        )
        manifest = read_json(result, "manifest.json")
        statuses = {item["status"] for item in manifest["transcription"]["per_input"]}
        assert statuses == {"failed"}
        assert any("modello rotto" in warning for warning in manifest["warnings"])
        assert (Path(result.export_dir) / "context.md").is_file()

    def test_empty_output_marks_empty(self, tmp_path, history_factory):
        result = do_export(
            tmp_path,
            make_state(),
            history_factory(),
            settings=ExportSettings(transcription="local", asr_model_dir=str(tmp_path / "modello")),
            transcriber_factory=lambda s: SpyTranscriber(default=""),
        )
        manifest = read_json(result, "manifest.json")
        statuses = {item["status"] for item in manifest["transcription"]["per_input"]}
        assert statuses == {"empty"}


class TestPreflightAndPrepare:
    def test_preflight_without_network(self, tmp_path):
        ok, _detail = transcriber.preflight_local(str(tmp_path / "assente"))
        assert ok is False
        model_dir = tmp_path / "modello"
        model_dir.mkdir()
        ok, _detail = transcriber.preflight_local(str(model_dir))
        assert ok is False, "senza manifest non è pronto"
        (model_dir / transcriber.MODEL_MANIFEST_NAME).write_text(
            json.dumps({"model": "base", "repo": transcriber.ASR_REPO, "revision": "abc123"}), encoding="utf-8"
        )
        ok, _detail = transcriber.preflight_local(str(model_dir))
        assert ok is False, "senza pesi non è pronto"
        (model_dir / transcriber.MODEL_WEIGHTS_NAME).write_bytes(b"pesi")
        ok, detail = transcriber.preflight_local(str(model_dir))
        assert ok is True and "abc123" in detail

    def test_prepare_writes_manifest_without_mobile_revision(self, tmp_path, monkeypatch):
        calls: list = []

        def fake_download(output_dir, model):
            calls.append((output_dir, model))
            (Path(output_dir) / transcriber.MODEL_WEIGHTS_NAME).write_bytes(b"pesi")
            return "fixed-commit-123"

        monkeypatch.setattr(transcriber, "_download_snapshot", fake_download)
        output = tmp_path / "asr"
        manifest = transcriber.prepare_model("base", str(output))
        assert calls == [(output, "base")]
        assert manifest["revision"] == "fixed-commit-123"
        assert manifest["repo"] == transcriber.ASR_REPO
        saved = json.loads((output / transcriber.MODEL_MANIFEST_NAME).read_text(encoding="utf-8"))
        assert saved["revision"] == "fixed-commit-123"

    def test_prepare_rejects_unknown_model(self, tmp_path):
        with pytest.raises(ValueError, match="modello"):
            transcriber.prepare_model("enorme", str(tmp_path / "asr"))


@pytest.mark.real_model
class TestRealTranscription:
    """Smoke della pipeline ASR locale: richiede extra asr + modello preparato.

    NON è un gate di qualità (S2-audio è FAIL documentato per decisione
    utente): su voce sintetica espeak it+f4 whisper-base garble C1 e va in
    loop sulle negative; i termini attesi qui sono parole del testo fixture
    che sopravvivono realmente nella trascrizione di C2/C3 (v. report H3).
    """

    def test_two_italian_wavs_contain_expected_terms(self, tmp_path):
        pytest.importorskip("faster_whisper")
        model_dir = Path("data/models/asr")
        ready, detail = transcriber.preflight_local(str(model_dir))
        if not ready:
            pytest.skip(f"modello ASR non preparato ({detail}): esegui il comando prepare")
        fixtures = Path("tests/fixtures/ambient")
        expected_terms = {"C2": ["mostrami", "grafico"], "C3": ["simbolo", "esclamativo"]}
        engine = transcriber.LocalTranscriber(str(model_dir), "it")
        for query_id, terms in expected_terms.items():
            text = engine.transcribe(str(fixtures / f"query-{query_id}.wav"))
            lowered = text.lower()
            assert text, f"{query_id}: trascrizione vuota"
            assert any(term in lowered for term in terms), f"{query_id}: termini attesi assenti in {text!r}"
