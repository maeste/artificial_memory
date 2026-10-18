"""Fixtures in-process per i test di src/inference.

Il fake server replica il protocollo REP del contratto (embed/health) senza
modello: vettori unitari deterministici e override di spazio, ritardo, envelope
malformed e vettori difettosi. Ogni socket nasce nel thread che lo usa.
"""

import hashlib
import json
import math
import threading
import time
import uuid

import pytest
import zmq

from src.contracts import EMBED_DIM, EmbedResponse, decode_embed_request, pack, pack_health, unpack


def deterministic_unit_vector(seed: str) -> list[float]:
    """Vettore unitario deterministico derivato da SHA-256 del seed."""
    values: list[float] = []
    digest = seed.encode("utf-8")
    while len(values) < EMBED_DIM:
        digest = hashlib.sha256(digest).digest()
        values.extend(byte - 127.5 for byte in digest)
    norm = math.sqrt(sum(value * value for value in values[:EMBED_DIM]))
    return [value / norm for value in values[:EMBED_DIM]]


class FakeServer:
    """REP socket conforme al contratto, senza backend reale."""

    def __init__(
        self,
        directory: str,
        space_id: str = "test-space",
        backend: str = "real",
        ready: bool = True,
        health_error: str | None = None,
        delay_ms: int = 0,
        malformed: str | None = None,
        wrong_request_id: bool = False,
        bad_vector: str | None = None,
        error_code: str | None = None,
    ):
        self.space_id = space_id
        self.backend = backend
        self.ready = ready
        self.health_error = health_error
        self.delay_ms = delay_ms
        self.malformed = malformed
        self.wrong_request_id = wrong_request_id
        self.bad_vector = bad_vector  # 'zero' | 'nan' | 'dim'
        self.error_code = error_code
        self.requests: list[str] = []
        self.replies_sent = 0
        self.endpoint = f"ipc://{directory}/embed-{uuid.uuid4().hex}.sock"
        self._stop = threading.Event()
        self._bound = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()
        if not self._bound.wait(timeout=5):
            raise RuntimeError("fake server: bind non completato")

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def wait_replies(self, count: int, timeout_sec: float = 3.0) -> bool:
        """Attende che il server abbia inviato `count` risposte (es. ritardi in corso)."""
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if self.replies_sent >= count:
                return True
            time.sleep(0.02)
        return False

    def _loop(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(self.endpoint)
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        self._bound.set()
        try:
            while not self._stop.is_set():
                if not poller.poll(timeout=50):
                    continue
                raw = socket.recv_string()
                self.requests.append(raw)
                if self.delay_ms:
                    # il ritardo vale per la prima richiesta poi si azzera: i test
                    # attendono wait_replies() prima del retry successivo
                    time.sleep(self.delay_ms / 1000.0)
                    self.delay_ms = 0
                socket.send_string(self._handle(raw))
                self.replies_sent += 1
        finally:
            socket.close(linger=0)
            context.term()

    def _handle(self, raw: str) -> str:
        if self.malformed is not None:
            return self.malformed
        try:
            payload = unpack(raw, "health.request")
            if payload != {}:
                raise ValueError("health.request con payload non vuoto")
            return pack_health(
                "health.response",
                {
                    "ready": self.ready,
                    "space_id": self.space_id,
                    "backend": self.backend,
                    "error": self.health_error,
                },
            )
        except ValueError:
            pass
        try:
            req = decode_embed_request(raw)
        except ValueError as exc:
            return pack(
                "embed.response",
                EmbedResponse(
                    request_id="",
                    success=False,
                    space_id=self.space_id,
                    error_code="INVALID_REQUEST",
                    error=str(exc),
                ),
            )
        request_id = req.request_id
        if self.wrong_request_id:
            request_id = f"sbagliato-{request_id}"
        if self.error_code is not None:
            return pack(
                "embed.response",
                EmbedResponse(
                    request_id=request_id,
                    success=False,
                    space_id=self.space_id,
                    error_code=self.error_code,
                    error=f"fake error: {self.error_code}",
                ),
            )
        vector = deterministic_unit_vector(req.request_id)
        if self.bad_vector == "zero":
            vector = [0.0] * EMBED_DIM
        elif self.bad_vector == "dim":
            vector = vector[:-1]
        elif self.bad_vector == "nan":
            # envelope con NaN letterale: pack rifiuterebbe, si costruisce a mano
            return json.dumps(
                {
                    "schema_version": 1,
                    "type": "embed.response",
                    "payload": {
                        "request_id": request_id,
                        "success": True,
                        "space_id": self.space_id,
                        "embedding": [float("nan")] * EMBED_DIM,
                        "error_code": None,
                        "error": None,
                    },
                }
            )
        return pack(
            "embed.response",
            EmbedResponse(request_id=request_id, success=True, space_id=self.space_id, embedding=vector),
        )


@pytest.fixture
def fake_server(tmp_path):
    """Factory: fake_server(**override) -> endpoint; i server vengono fermati al teardown."""

    servers: list[FakeServer] = []

    def factory(**kwargs) -> str:
        server = FakeServer(str(tmp_path), **kwargs)
        server.start()
        servers.append(server)
        return server.endpoint

    factory.servers = servers
    yield factory
    for server in servers:
        server.stop()
