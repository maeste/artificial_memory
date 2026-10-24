"""Test del server REP con backend stub, senza pesi (Task I1) e del mapping
multimodale con modello finto (Task I2).

Verifica il ciclo recv/send del server reale: health con warmup, errori che
non rompono il loop, code di errore del backend propagati al client. I test
del builder intercettano l'argomento passato a model.encode senza pesi.
"""

import json
import os
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from src.contracts import EMBED_DIM, EmbedRequest, SpaceSpec, validate_vector
from src.inference.client import EmbedClient, EmbeddingError
from src.inference.model import EmbeddingGemmaBackend, build_model_input, project_output
from src.inference.server import EmbedServer

SPACE = SpaceSpec(backend="real", model_id="stub-model", revision="test-1")


class StubBackend:
    """Backend deterministico: verifica l'esistenza dei media, vettore unitario fisso.

    Se `fail_request_id` è impostato, fallisce solo per quella richiesta
    (il warmup ha request_id='warmup' e non deve fallire).
    """

    def __init__(self, fail_request_id: str | None = None):
        self.fail_request_id = fail_request_id
        self.calls = 0

    def embed(self, req: EmbedRequest) -> list[float]:
        self.calls += 1
        if self.fail_request_id is not None and req.request_id == self.fail_request_id:
            raise EmbeddingError("INPUT_TOO_LARGE", "testo troppo lungo")
        for path in req.image_paths:
            if not os.path.exists(path):
                raise EmbeddingError("INVALID_MEDIA", f"immagine inesistente: {path}")
        if req.audio_path is not None and not os.path.exists(req.audio_path):
            raise EmbeddingError("INVALID_MEDIA", f"audio inesistente: {req.audio_path}")
        vector = [0.0] * 256
        vector[0] = 1.0
        return vector


def _wait_ready(client: EmbedClient, timeout_sec: float = 5.0) -> dict:
    import time

    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        health = client.health()
        if health["ready"]:
            return health
        time.sleep(0.05)
    return client.health()


def test_embed_roundtrip(tmp_path):
    endpoint = f"ipc://{tmp_path}/embed-rt.sock"
    server = EmbedServer(endpoint, SPACE, StubBackend)
    server.start()
    try:
        client = EmbedClient(endpoint, expected_space_id=SPACE.space_id, timeout_ms=2000)
        vector = client.embed(EmbedRequest(request_id="r1", is_query=True, text_payload="query"))
        assert vector[0] == 1.0
        assert all(value == 0.0 for value in vector[1:])
        client.close()
    finally:
        server.stop()


def test_health_ready_after_load_and_warmup(tmp_path):
    endpoint = f"ipc://{tmp_path}/embed-health.sock"
    backend_holder: dict[str, StubBackend] = {}

    def factory() -> StubBackend:
        backend = StubBackend()
        backend_holder["backend"] = backend
        return backend

    server = EmbedServer(endpoint, SPACE, factory)
    server.start()
    try:
        client = EmbedClient(endpoint, expected_space_id=SPACE.space_id, timeout_ms=2000)
        health = _wait_ready(client)
        assert health == {
            "ready": True,
            "space_id": SPACE.space_id,
            "backend": "real",
            "error": None,
        }
        # il warmup è un embedding riuscito prima del primo client embed
        assert backend_holder["backend"].calls >= 1
        client.close()
    finally:
        server.stop()


def test_embed_before_ready_returns_model_unavailable(tmp_path):
    endpoint = f"ipc://{tmp_path}/embed-gate.sock"
    gate = threading.Event()

    def factory() -> StubBackend:
        gate.wait(timeout=5)
        return StubBackend()

    server = EmbedServer(endpoint, SPACE, factory)
    server.start()
    try:
        client = EmbedClient(endpoint, expected_space_id=SPACE.space_id, timeout_ms=2000)
        health = client.health()
        assert health["ready"] is False
        with pytest.raises(EmbeddingError) as excinfo:
            client.embed(EmbedRequest(request_id="r2", is_query=True, text_payload="query"))
        assert excinfo.value.code == "MODEL_UNAVAILABLE"
        gate.set()
        assert _wait_ready(client)["ready"] is True
        vector = client.embed(EmbedRequest(request_id="r3", is_query=True, text_payload="query"))
        assert vector[0] == 1.0
        client.close()
    finally:
        gate.set()
        server.stop()


def test_invalid_media_keeps_server_alive(tmp_path):
    endpoint = f"ipc://{tmp_path}/embed-media.sock"
    server = EmbedServer(endpoint, SPACE, StubBackend)
    server.start()
    try:
        client = EmbedClient(endpoint, expected_space_id=SPACE.space_id, timeout_ms=2000)
        _wait_ready(client)
        missing = str(tmp_path / "inesistente.wav")
        with pytest.raises(EmbeddingError) as excinfo:
            client.embed(EmbedRequest(request_id="r4", is_query=True, audio_path=missing))
        assert excinfo.value.code == "INVALID_MEDIA"
        vector = client.embed(EmbedRequest(request_id="r5", is_query=True, text_payload="ok"))
        assert vector[0] == 1.0
        client.close()
    finally:
        server.stop()


def test_backend_error_code_passthrough(tmp_path):
    endpoint = f"ipc://{tmp_path}/embed-err.sock"
    server = EmbedServer(endpoint, SPACE, lambda: StubBackend(fail_request_id="r6"))
    server.start()
    try:
        client = EmbedClient(endpoint, expected_space_id=SPACE.space_id, timeout_ms=2000)
        with pytest.raises(EmbeddingError) as excinfo:
            client.embed(EmbedRequest(request_id="r6", is_query=True, text_payload="query"))
        assert excinfo.value.code == "INPUT_TOO_LARGE"
        assert "testo troppo lungo" in str(excinfo.value)
        client.close()
    finally:
        server.stop()


def test_load_failure_reports_error_without_mock_fallback(tmp_path):
    endpoint = f"ipc://{tmp_path}/embed-fail.sock"

    def factory() -> StubBackend:
        raise RuntimeError("pesi non trovati")

    server = EmbedServer(endpoint, SPACE, factory)
    server.start()
    try:
        client = EmbedClient(endpoint, expected_space_id=SPACE.space_id, timeout_ms=2000)
        health = client.health()
        assert health["ready"] is False
        assert health["error"] is not None
        assert "pesi non trovati" in health["error"]
        with pytest.raises(EmbeddingError) as excinfo:
            client.embed(EmbedRequest(request_id="r7", is_query=True, text_payload="query"))
        assert excinfo.value.code == "MODEL_UNAVAILABLE"
        client.close()
    finally:
        server.stop()


def test_garbage_request_gets_error_and_loop_survives(tmp_path):
    endpoint = f"ipc://{tmp_path}/embed-garbage.sock"
    server = EmbedServer(endpoint, SPACE, StubBackend)
    server.start()
    try:
        client = EmbedClient(endpoint, expected_space_id=SPACE.space_id, timeout_ms=2000)
        _wait_ready(client)
        import zmq

        raw_socket = zmq.Context.instance().socket(zmq.REQ)
        raw_socket.setsockopt(zmq.LINGER, 0)
        raw_socket.setsockopt(zmq.RCVTIMEO, 2000)
        raw_socket.connect(endpoint)
        try:
            raw_socket.send_string("[]")
            reply = raw_socket.recv_string()
        finally:
            raw_socket.close(linger=0)
        assert "INVALID_REQUEST" in reply
        vector = client.embed(EmbedRequest(request_id="r8", is_query=True, text_payload="ok"))
        assert vector[0] == 1.0
        client.close()
    finally:
        server.stop()


def test_request_with_unreadable_id_gets_empty_id_in_error(tmp_path):
    endpoint = f"ipc://{tmp_path}/embed-noid.sock"
    server = EmbedServer(endpoint, SPACE, StubBackend)
    server.start()
    try:
        client = EmbedClient(endpoint, expected_space_id=SPACE.space_id, timeout_ms=2000)
        _wait_ready(client)
        import json
        import zmq

        raw_socket = zmq.Context.instance().socket(zmq.REQ)
        raw_socket.setsockopt(zmq.LINGER, 0)
        raw_socket.setsockopt(zmq.RCVTIMEO, 2000)
        raw_socket.connect(endpoint)
        envelope = json.dumps(
            {
                "schema_version": 1,
                "type": "embed.request",
                "payload": {"request_id": "r9", "is_query": True, "text_payload": ""},
            }
        )
        try:
            raw_socket.send_string(envelope)
            reply = raw_socket.recv_string()
        finally:
            raw_socket.close(linger=0)
        assert "INVALID_REQUEST" in reply
        payload = json.loads(reply)["payload"]
        assert payload["request_id"] == ""
        client.close()
    finally:
        server.stop()


# ----------------------------------------------------------------- Task I2: builder


class TestBuildModelInput:
    def test_query_text_gets_search_prefix(self):
        request = EmbedRequest(request_id="q", is_query=True, text_payload="latenza del database")
        assert build_model_input(request) == "task: search result | query: latenza del database"

    def test_doc_text_gets_title_prefix(self):
        request = EmbedRequest(
            request_id="d1",
            is_query=False,
            text_payload="La cache riduce i accessi.",
            context_header="note.md > Cache",
        )
        assert build_model_input(request) == "title: note.md > Cache | text: La cache riduce i accessi."

    def test_doc_without_header_uses_none(self):
        request = EmbedRequest(request_id="d2", is_query=False, text_payload="corpo")
        assert build_model_input(request) == "title: none | text: corpo"

    def test_audio_query_has_no_text_prefix(self):
        request = EmbedRequest(request_id="q2", is_query=True, audio_path="/abs/query.wav")
        assert build_model_input(request) == {"audio": "/abs/query.wav"}

    def test_doc_media_dict_placeholders_in_order(self):
        request = EmbedRequest(
            request_id="d3",
            is_query=False,
            text_payload="pagina di architettura",
            context_header="f07.pdf > pag 2",
            image_paths=["/abs/page2.png"],
            audio_path="/abs/window.wav",
        )
        model_input = build_model_input(request)
        assert model_input["image"] == ["/abs/page2.png"]
        assert model_input["audio"] == "/abs/window.wav"
        assert model_input["text"] == "title: f07.pdf > pag 2 | text: pagina di architettura <|image|> <|audio|>"

    def test_doc_images_only_order(self):
        request = EmbedRequest(
            request_id="d4",
            is_query=False,
            text_payload=" finestra video",
            image_paths=["/abs/f1.png", "/abs/f2.png", "/abs/f3.png"],
        )
        text = build_model_input(request)["text"]
        assert text.count("<|image|>") == 3
        assert text.endswith("<|image|> <|image|> <|image|>")
        assert "<|audio|>" not in text

    def test_query_with_images_rejected(self):
        request = EmbedRequest(request_id="q3", is_query=True, image_paths=["/abs/img.png"])
        with pytest.raises(EmbeddingError, match="INVALID_REQUEST"):
            build_model_input(request)


# ---------------------------------------------------------- Task I2: proiezione output


class TestProjectOutput:
    def test_normalizes_and_validates(self):
        raw = np.full(EMBED_DIM, 3.0, dtype=np.float32)
        result = project_output(raw)
        validate_vector(result)

    def test_accepts_batch_dimension(self):
        raw = np.zeros((1, EMBED_DIM), dtype=np.float32)
        raw[0, 0] = 2.0
        result = project_output(raw)
        assert len(result) == EMBED_DIM
        validate_vector(result)

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match="INVALID_MODEL_OUTPUT"):
            project_output(np.zeros((2, EMBED_DIM), dtype=np.float32))

    def test_rejects_non_finite(self):
        raw = np.full(EMBED_DIM, np.nan, dtype=np.float32)
        with pytest.raises(ValueError, match="INVALID_MODEL_OUTPUT"):
            project_output(raw)

    def test_rejects_tiny_norm(self):
        raw = np.full(EMBED_DIM, 1e-20, dtype=np.float32)
        with pytest.raises(ValueError, match="INVALID_VECTOR_NORM"):
            project_output(raw)


# ----------------------------------------------------------- Task I2: backend con finto modello


class StubTokenizer:
    """Tokenizer stub: un token per parola."""

    def __call__(self, text: str, **_):
        return {"input_ids": list(range(len(text.split())))}


class FakeEncodeModel:
    """Sostituto del modello SentenceTransformer: registra encode, senza pesi."""

    def __init__(self, output=None, with_audio=True, with_image=True):
        self.encode_calls: list[tuple[object, dict]] = []
        self.tokenizer = StubTokenizer()
        parts: dict[str, object] = {}
        if with_image:
            parts["image_processor"] = object()
        if with_audio:
            parts["feature_extractor"] = object()
        self.processor = SimpleNamespace(**parts)
        if output is None:
            self.output = np.zeros(EMBED_DIM, dtype=np.float32)
            self.output[0] = 1.0
        else:
            self.output = np.asarray(output, dtype=np.float32)

    def encode(self, model_input, **kwargs):
        self.encode_calls.append((model_input, kwargs))
        return self.output


def make_backend(monkeypatch, model, device="cpu"):
    monkeypatch.setattr(EmbeddingGemmaBackend, "_load_model", classmethod(lambda cls, space, dev: model))
    return EmbeddingGemmaBackend(SPACE, device)


def _write_wav(path, sr=16000, channels=1, seconds=0.5):
    import soundfile

    frames = int(sr * seconds)
    data = np.zeros((frames, channels), dtype=np.float32)
    soundfile.write(path, data, sr)


@pytest.fixture
def audio_files(tmp_path):
    good = str(tmp_path / "good.wav")
    wrong_sr = str(tmp_path / "wrong_sr.wav")
    stereo = str(tmp_path / "stereo.wav")
    too_long = str(tmp_path / "too_long.wav")
    _write_wav(good)
    _write_wav(wrong_sr, sr=44100)
    _write_wav(stereo, channels=2)
    _write_wav(too_long, seconds=46.0)
    return SimpleNamespace(good=good, wrong_sr=wrong_sr, stereo=stereo, too_long=too_long)


class TestEmbeddingGemmaBackend:
    def test_embed_passes_builder_input_and_mrl_kwargs(self, monkeypatch):
        model = FakeEncodeModel()
        backend = make_backend(monkeypatch, model)
        vector = backend.embed(EmbedRequest(request_id="q", is_query=True, text_payload="cache"))
        assert model.encode_calls == [
            ("task: search result | query: cache", {"truncate_dim": EMBED_DIM, "normalize_embeddings": True})
        ]
        validate_vector(vector)

    def test_backend_records_dtype_device_and_versions(self, monkeypatch):
        backend = make_backend(monkeypatch, FakeEncodeModel(), device="cpu")
        assert backend.info["device"] == "cpu"
        assert backend.info["dtype"] == "float32"
        assert backend.info["model_id"] == SPACE.model_id
        assert "torch" in backend.info and "sentence_transformers" in backend.info

    def test_oversize_text_rejected_without_truncation(self, monkeypatch):
        backend = make_backend(monkeypatch, FakeEncodeModel())
        request = EmbedRequest(request_id="big", is_query=True, text_payload="parola " * 6001)
        with pytest.raises(EmbeddingError) as excinfo:
            backend.embed(request)
        assert excinfo.value.code == "INPUT_TOO_LARGE"

    def test_missing_media_file_rejected(self, monkeypatch, tmp_path):
        backend = make_backend(monkeypatch, FakeEncodeModel())
        request = EmbedRequest(request_id="m", is_query=True, audio_path=str(tmp_path / "nope.wav"))
        with pytest.raises(EmbeddingError, match="INVALID_MEDIA"):
            backend.embed(request)

    def test_wrong_sample_rate_rejected(self, monkeypatch, audio_files):
        backend = make_backend(monkeypatch, FakeEncodeModel())
        request = EmbedRequest(request_id="sr", is_query=True, audio_path=audio_files.wrong_sr)
        with pytest.raises(EmbeddingError, match="INVALID_MEDIA"):
            backend.embed(request)

    def test_stereo_rejected(self, monkeypatch, audio_files):
        backend = make_backend(monkeypatch, FakeEncodeModel())
        request = EmbedRequest(request_id="st", is_query=True, audio_path=audio_files.stereo)
        with pytest.raises(EmbeddingError, match="INVALID_MEDIA"):
            backend.embed(request)

    def test_audio_over_45s_rejected(self, monkeypatch, audio_files):
        backend = make_backend(monkeypatch, FakeEncodeModel())
        request = EmbedRequest(request_id="long", is_query=True, audio_path=audio_files.too_long)
        with pytest.raises(EmbeddingError) as excinfo:
            backend.embed(request)
        assert excinfo.value.code == "INPUT_TOO_LARGE"

    def test_budget_function_limits(self):
        from src.inference.model import budget_check

        # ai limiti contrattuali: 6000 testo + 840 immagini + 1125 audio = 7965 <= 8192
        budget_check(6000, 3, 45.0)
        with pytest.raises(EmbeddingError) as excinfo:
            budget_check(6001, 0, 0.0)
        assert excinfo.value.code == "INPUT_TOO_LARGE"
        # difesa: oltre il budget totale anche senza superare il testo (8215 > 8192)
        with pytest.raises(EmbeddingError) as excinfo:
            budget_check(6000, 3, 55.0)
        assert excinfo.value.code == "INPUT_TOO_LARGE"

    def test_backend_without_audio_support_rejected(self, monkeypatch, audio_files):
        backend = make_backend(monkeypatch, FakeEncodeModel(with_audio=False))
        request = EmbedRequest(request_id="na", is_query=True, audio_path=audio_files.good)
        with pytest.raises(EmbeddingError) as excinfo:
            backend.embed(request)
        assert excinfo.value.code == "MODEL_UNAVAILABLE"

    def test_non_finite_model_output_rejected(self, monkeypatch):
        bad = np.full(EMBED_DIM, np.nan, dtype=np.float32)
        backend = make_backend(monkeypatch, FakeEncodeModel(output=bad))
        with pytest.raises(EmbeddingError) as excinfo:
            backend.embed(EmbedRequest(request_id="nf", is_query=True, text_payload="query"))
        assert excinfo.value.code == "MODEL_UNAVAILABLE"

    def test_no_automatic_retry_after_error(self, monkeypatch):
        model = FakeEncodeModel()
        backend = make_backend(monkeypatch, model)
        with pytest.raises(EmbeddingError):
            backend.embed(EmbedRequest(request_id="big2", is_query=True, text_payload="parola " * 6001))
        # il backend non ha richiamato encode dopo l'errore di budget
        assert model.encode_calls == []


# ------------------------------------------------------- prove real_model (pesi veri)

REAL_MODEL_ID = "google/embeddinggemma-2"
REAL_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
AMBIENT = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "fixtures", "ambient")
AMBIENT = os.path.abspath(AMBIENT)

pytestmark_real = pytest.mark.real_model


@pytest.mark.real_model
class TestRealModelSmoke:
    """Fumo del backend reale alla revisione pinnata: richiede i pesi in cache."""

    def test_load_records_dtype_device_and_modalities(self):
        import torch

        from src.inference.model import EmbeddingGemmaBackend as RealBackend

        space = SpaceSpec(backend="real", model_id=REAL_MODEL_ID, revision=REAL_REVISION)
        backend = RealBackend(space, device="cuda" if torch.cuda.is_available() else "cpu")
        assert backend.info["revision"] == REAL_REVISION
        assert backend.info["dtype"] in ("bfloat16", "float32")
        assert backend.info["dtype"] != "float16"
        assert set(backend.info["modalities"]) == {"text", "image", "audio"}

    def test_embed_text_audio_and_mixed(self):
        import soundfile

        from src.inference.model import EmbeddingGemmaBackend as RealBackend

        space = SpaceSpec(backend="real", model_id=REAL_MODEL_ID, revision=REAL_REVISION)
        backend = RealBackend(space)
        text_vector = backend.embed(EmbedRequest(request_id="t", is_query=True, text_payload="cache del database"))
        validate_vector(text_vector)
        audio_path = os.path.join(AMBIENT, "query-C1.wav")
        info = soundfile.info(audio_path)
        assert info.samplerate == 16000 and info.channels == 1
        audio_vector = backend.embed(EmbedRequest(request_id="a", is_query=True, audio_path=audio_path))
        validate_vector(audio_vector)
        frames = [os.path.join(AMBIENT, "_staging", "f13", "frames", f"frame_{n:05d}.png") for n in (10, 11, 12)]
        mixed_vector = backend.embed(
            EmbedRequest(
                request_id="m",
                is_query=False,
                text_payload="finestra video",
                context_header="f13.mp4 > 0-45000ms",
                image_paths=frames,
                audio_path=os.path.join(AMBIENT, "_staging", "f14", "window_0_45.wav"),
            )
        )
        validate_vector(mixed_vector)


# ------------------------------------------------------------------ Task I3: mock


class TestSemanticHashMock:
    def test_requires_mock_space(self):
        from src.inference.model import SemanticHashMock, mock_space

        with pytest.raises(EmbeddingError, match="INVALID_REQUEST"):
            SemanticHashMock(SPACE)
        SemanticHashMock(mock_space())

    def test_mock_identity_differs_from_real(self):
        from src.inference.model import mock_space

        real_space = SpaceSpec(backend="real", model_id=REAL_MODEL_ID, revision=REAL_REVISION)
        assert mock_space().space_id != real_space.space_id
        assert mock_space().model_id == "semantic-hash-mock"

    def test_embed_valid_unit_vector(self):
        from src.inference.model import SemanticHashMock, mock_space

        mock = SemanticHashMock(mock_space())
        vector = mock.embed(EmbedRequest(request_id="m1", is_query=True, text_payload="cache del database"))
        validate_vector(vector)

    def test_sentinel_and_path_properties(self):
        from src.inference.model import SemanticHashMock, mock_space

        mock = SemanticHashMock(mock_space())
        # payload completamente privo di feature: token sentinella, vettore comunque unitario
        empty = mock.embed(EmbedRequest(request_id="m2", is_query=True, text_payload=""))
        validate_vector(empty)
        # proprietà controllata: stesso basename -> stesso vettore, basename diverso -> diverso
        first = mock.embed(EmbedRequest(request_id="m3", is_query=True, audio_path="/x/y.wav"))
        second = mock.embed(EmbedRequest(request_id="m4", is_query=True, audio_path="/z/y.wav"))
        other = mock.embed(EmbedRequest(request_id="m5", is_query=True, audio_path="/z/other.wav"))
        assert first == second
        assert first != other


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

    def test_deterministic_within_process(self):
        from src.inference.model import SemanticHashMock, mock_space

        mock = SemanticHashMock(mock_space())
        request = EmbedRequest(request_id="m4", is_query=True, text_payload="Latenza del database")
        assert mock.embed(request) == mock.embed(request)

    def test_deterministic_across_processes(self):
        import subprocess
        import sys

        from src.inference.model import SemanticHashMock, mock_space

        mock = SemanticHashMock(mock_space())
        request = EmbedRequest(request_id="m5", is_query=True, text_payload="Latenza del database e cache")
        in_process = mock.embed(request)
        script = (
            "import json, sys;"
            "sys.path.insert(0, '.');"
            "from src.contracts import EmbedRequest;"
            "from src.inference.model import SemanticHashMock, mock_space;"
            "req = EmbedRequest(request_id='m5', is_query=True, text_payload='Latenza del database e cache');"
            "print(json.dumps(SemanticHashMock(mock_space()).embed(req)))"
        )
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
        assert json.loads(result.stdout) == in_process

    def test_overlapping_text_more_similar_than_disjoint(self):
        from src.inference.model import SemanticHashMock, mock_space

        mock = SemanticHashMock(mock_space())
        base = mock.embed(
            EmbedRequest(request_id="m6", is_query=True, text_payload="la cache riduce gli accessi al database")
        )
        overlap = mock.embed(EmbedRequest(request_id="m7", is_query=True, text_payload="la cache riduce gli accessi"))
        disjoint = mock.embed(
            EmbedRequest(request_id="m8", is_query=True, text_payload="ricetta della torta al cioccolato")
        )
        assert _dot(base, overlap) > _dot(base, disjoint)
        # nessuna pretesa di soglia: identico -> 1.0, tema generico non vincolato a >0.80
        same = mock.embed(
            EmbedRequest(request_id="m9", is_query=True, text_payload="la cache riduce gli accessi al database")
        )
        assert _dot(base, same) == pytest.approx(1.0, abs=1e-6)

    def test_media_markers_distinguish_modalities(self):
        from src.inference.model import SemanticHashMock, mock_space

        mock = SemanticHashMock(mock_space())
        text_only = mock.embed(EmbedRequest(request_id="m10", is_query=False, text_payload="descrizione"))
        with_image = mock.embed(
            EmbedRequest(request_id="m11", is_query=False, text_payload="descrizione", image_paths=["/a/b.png"])
        )
        similarity = _dot(text_only, with_image)
        assert -0.999 < similarity < 0.999


# ------------------------------------------------------------------ Task I4: CLI e probe

MANIFEST = os.path.join(AMBIENT, "manifest.json")
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _write_config(path, space_dict, device="auto"):
    config = {"schema_version": 1, "space": space_dict, "device": device}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(config, fh)
    return str(path)


def _mock_space_dict():
    return {
        "backend": "mock",
        "model_id": "semantic-hash-mock",
        "revision": "1",
        "dimension": 256,
        "preprocessing": "eg2-poc-v1",
    }


class TestProbeCli:
    def test_probe_mock_end_to_end_on_real_manifest(self, tmp_path):
        from src.inference.__main__ import run_probe

        config = _write_config(tmp_path / "config.json", _mock_space_dict())
        report = run_probe(
            config,
            MANIFEST,
            str(tmp_path / "report.json"),
            bank_path=str(tmp_path / "bank.json"),
        )
        assert report["counts"]["chunks"] == 26
        assert report["counts"]["chunk_errors"] == 0
        assert report["space"]["backend"] == "mock"
        assert report["gate_fase1_preliminare"]["applicabile"] is False
        # C1-C4 + T1-T8 con gold; N1-N10 diagnostica negativa
        assert report["metrics"]["text"]["n"] == 12
        assert report["metrics"]["audio"]["n"] == 12
        assert report["metrics"]["negative_diagnostica"]["n"] == 10
        assert report["times"]["load_sec"] >= 0
        # i tempi riguardano tutte le query embeddate (12 gold + 10 negative)
        assert report["times"]["query_text"]["n"] == 22
        assert report["times"]["query_audio"]["n"] == 22
        bank = json.load(open(tmp_path / "bank.json", encoding="utf-8"))
        assert bank["space"]["backend"] == "mock"
        assert len(bank["records"]) == 26
        assert len(bank["records"][0]["embedding"]) == 256
        assert bank["records"][0]["memory"]["id"]
        # ogni query ha top5 con punteggi e rango
        first = report["queries"][0]
        assert len(first["top5"]) == 5
        assert first["top5"][0]["rank"] == 1

    def test_probe_mock_calibration_split(self, tmp_path):
        from src.inference.__main__ import run_probe

        config = _write_config(tmp_path / "config.json", _mock_space_dict())
        report = run_probe(config, MANIFEST, str(tmp_path / "report.json"), split="calibration")
        assert report["metrics"]["text"]["n"] == 4
        assert report["metrics"]["audio"]["n"] == 4
        assert report["metrics"]["negative_diagnostica"]["n"] == 5

    def test_embed_cli_mock_subprocess(self, tmp_path):
        import subprocess
        import sys

        config = _write_config(tmp_path / "config.json", _mock_space_dict())
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.inference",
                "embed",
                "--config",
                config,
                "--text",
                "cache del database",
                "--query",
            ],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        )
        payload = json.loads(result.stdout)
        assert len(payload["embedding"]) == 256
        assert payload["latency_ms"] >= 0
        assert payload["space"]["backend"] == "mock"
        assert payload["request"]["is_query"] is True
        audio_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.inference",
                "embed",
                "--config",
                config,
                "--audio",
                os.path.join(AMBIENT, "query-N1.wav"),
            ],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        )
        audio_payload = json.loads(audio_result.stdout)
        assert audio_payload["request"]["audio_path"].endswith("query-N1.wav")

    def test_rejects_unknown_config_fields(self, tmp_path):
        from src.inference.__main__ import load_config

        config = tmp_path / "config.json"
        config.write_text(
            json.dumps({"schema_version": 1, "space": _mock_space_dict(), "surprise": 1}),
            encoding="utf-8",
        )
        with pytest.raises(SystemExit, match="campi sconosciuti"):
            load_config(str(config))

    @pytest.mark.real_model
    def test_probe_real_calibration_gate(self, tmp_path):
        from src.inference.__main__ import run_probe

        space_dict = {
            "backend": "real",
            "model_id": REAL_MODEL_ID,
            "revision": REAL_REVISION,
            "dimension": 256,
            "preprocessing": "eg2-poc-v1",
        }
        config = _write_config(tmp_path / "config.json", space_dict)
        report = run_probe(
            config,
            MANIFEST,
            str(tmp_path / "probe-real.json"),
            bank_path=str(tmp_path / "bank-real.json"),
            split="calibration",
            repeats=3,
        )
        assert report["backend_info"]["model_id"] == REAL_MODEL_ID
        assert report["counts"]["chunk_errors"] == 0
        gate = report["gate_fase1_preliminare"]
        assert gate["pass"] is True, "gate preliminare fallito: " + json.dumps(
            {
                "gate": gate,
                "metrics": report["metrics"],
                "queries": report["queries"],
            },
            ensure_ascii=False,
            indent=1,
        )
        # sanità di latenza inferenza (S5 pipeline completa si misura nel runner)
        assert report["times"]["query_text"]["p95_ms"] < 5000
        assert report["times"]["query_audio"]["p95_ms"] < 5000


class TestConfigSpaceIdCrossCheck:
    def test_accepts_orchestrator_superset_config(self, tmp_path):
        from src.inference.__main__ import load_config

        config = tmp_path / "config.json"
        space_dict = _mock_space_dict()
        full_space = SpaceSpec(**space_dict)
        config.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "space": space_dict,
                    "space_id": full_space.space_id,
                    "device": "auto",
                    "embed_timeout_ms": 30000,
                    "startup_timeout_sec": 300,
                }
            ),
            encoding="utf-8",
        )
        space, device = load_config(str(config))
        assert space.space_id == full_space.space_id
        assert device == "auto"

    def test_rejects_mismatched_space_id(self, tmp_path):
        from src.inference.__main__ import load_config

        config = tmp_path / "config.json"
        config.write_text(
            json.dumps({"schema_version": 1, "space": _mock_space_dict(), "space_id": "0" * 64}),
            encoding="utf-8",
        )
        with pytest.raises(SystemExit, match="space_id non corrisponde"):
            load_config(str(config))


# ------------------------------------------------------------------ Task serve


def _write_serve_config(tmp_path, space_dict, endpoint, device="auto"):
    config = {
        "schema_version": 1,
        "space": space_dict,
        "space_id": SpaceSpec(**space_dict).space_id,
        "paths": {"embed_endpoint": endpoint},
        "device": device,
        "embed_timeout_ms": 30000,
        "startup_timeout_sec": 300,
    }
    path = tmp_path / "serve-config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return str(path)


class TestServeCommand:
    def test_serve_subprocess_mock_ready_then_sigterm(self, tmp_path):
        import subprocess
        import sys
        import time as time_module

        endpoint = f"ipc://{tmp_path}/serve-mock.sock"
        config = _write_serve_config(tmp_path, _mock_space_dict(), endpoint)
        process = subprocess.Popen(
            [sys.executable, "-m", "src.inference", "serve", "--config", config],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            client = EmbedClient(endpoint, expected_space_id=SpaceSpec(**_mock_space_dict()).space_id, timeout_ms=2000)
            deadline = time_module.monotonic() + 20
            health = None
            while time_module.monotonic() < deadline:
                try:
                    health = client.health()
                    if health["ready"]:
                        break
                except EmbeddingError:
                    pass
                time_module.sleep(0.1)
            assert health is not None and health["ready"], f"health mai pronto: {health}"
            assert health["backend"] == "mock"
            vector = client.embed(EmbedRequest(request_id="s1", is_query=True, text_payload="cache"))
            assert len(vector) == 256
            client.close()
        finally:
            process.terminate()
            _, err = process.communicate(timeout=10)
        assert process.returncode == 0
        assert "modello pronto" in err
        assert "arresto completato" in err

    def test_serve_load_failure_exits_nonzero_without_mock_fallback(self, tmp_path, monkeypatch):
        from src.inference.__main__ import run_serve

        endpoint = f"ipc://{tmp_path}/serve-fail.sock"
        real_space_dict = {
            "backend": "real",
            "model_id": "modello-inesistente-test",
            "revision": "0" * 40,
        }
        config = _write_serve_config(tmp_path, real_space_dict, endpoint)

        def _explode(space, device):
            raise RuntimeError("pesi assenti (test)")

        monkeypatch.setattr(EmbeddingGemmaBackend, "_load_model", classmethod(_explode))
        result: dict = {}

        def runner():
            result["code"] = run_serve(config, poll_interval=0.05)

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        thread.join(timeout=15)
        assert not thread.is_alive(), "run_serve non è uscito dopo il fallimento di caricamento"
        assert result["code"] == 1

    def test_serve_rejects_config_without_embed_endpoint(self, tmp_path):
        from src.inference.__main__ import load_serve_config

        config = tmp_path / "cfg.json"
        config.write_text(
            json.dumps({"schema_version": 1, "space": _mock_space_dict(), "paths": {}}),
            encoding="utf-8",
        )
        with pytest.raises(SystemExit, match="embed_endpoint"):
            load_serve_config(str(config))


class TestServeIntegralConfig:
    """Contratto 07 §6: il config del launcher passa INTEGRALMENTE ai figli."""

    def _integral_config(self, tmp_path, endpoint, space_dict=None, **overrides):
        space_dict = space_dict or _mock_space_dict()
        valid_keys = {"backend", "model_id", "revision", "dimension", "preprocessing"}
        try:
            space_id = SpaceSpec(**{k: v for k, v in space_dict.items() if k in valid_keys}).space_id
        except TypeError:
            space_id = None
        config = {
            "schema_version": 1,
            "space": space_dict,
            "space_id": space_id,
            "paths": {
                "run_dir": "/run/user/1000/artificial-memory/run1",
                "embed_endpoint": endpoint,
                "events_endpoint": "ipc:///run/user/1000/artificial-memory/run1/events.sock",
                "control_endpoint": "ipc:///run/user/1000/artificial-memory/run1/control.sock",
                "db_path": "/abs/data/brain-real.db",
                "watch_dir": "/abs/watch",
                "export_dir": "/abs/exports",
            },
            "device": "auto",
            "embed_timeout_ms": 30000,
            "startup_timeout_sec": 300,
            # chiavi dei FIGLI ALTRI (runner/watcher/hud): accettate e ignorate da serve
            "agent_format": "claude-jsonl",
            "agent_log": "/abs/agent.log",
            "alpha": 0.35,
            "asr_model_dir": None,
            "command_timeout_ms": 2000,
            "max_pins": 12,
            "mic_enabled": True,
            "search": {"top_k": 5, "score_mode": "cosine"},
            "session_limit_sec": 3600,
            "session_max_inputs": 2000,
            "session_audio_max_bytes": 125829120,
            "show_all": False,
            "source_ttl_sec": 120,
            "transcription": "off",
        }
        config.update(overrides)
        path = tmp_path / "integral-config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return str(path)

    def test_integral_launcher_config_accepted(self, tmp_path):
        from src.inference.__main__ import load_serve_config

        space, endpoint, device = load_serve_config(
            self._integral_config(tmp_path, "ipc:///run/user/1000/am/run1/embed.sock")
        )
        assert space.model_id == "semantic-hash-mock"
        assert endpoint == "ipc:///run/user/1000/am/run1/embed.sock"
        assert device == "auto"

    def test_consumed_fields_still_strict(self, tmp_path):
        from src.inference.__main__ import load_serve_config

        with pytest.raises(SystemExit, match="embed_endpoint"):
            load_serve_config(self._integral_config(tmp_path, "ipc:///x/embed.sock", paths={"watch_dir": "/w"}))
        bad_space = _mock_space_dict() | {"surprise": True}
        with pytest.raises(SystemExit, match="campi sconosciuti"):
            load_serve_config(self._integral_config(tmp_path, "ipc:///x/embed.sock", space_dict=bad_space))
        with pytest.raises(SystemExit, match="space_id non corrisponde"):
            load_serve_config(self._integral_config(tmp_path, "ipc:///x/embed.sock", space_id="0" * 64))
