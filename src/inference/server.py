"""Server REP dell'inferenza: proprietario del modello (contratto 1.0).

Il backend viene caricato in un thread dedicato; health risponde ready solo
dopo caricamento e warmup riuscito. Ogni messaggio ricevuto produce una
risposta: un errore non termina il ciclo recv/send. Nessun fallback al mock.
"""

import threading

import zmq

from src.contracts import (
    EmbedRequest,
    EmbedResponse,
    SpaceSpec,
    decode_embed_request,
    pack,
    pack_health,
    unpack,
)
from src.inference.client import EmbeddingError

_BIND_TIMEOUT_SEC = 5.0


class EmbedServer:
    """REP socket con backend iniettabile; testabile senza pesi."""

    def __init__(self, endpoint: str, space: SpaceSpec, backend_factory, on_ready=None):
        self._endpoint = endpoint
        self._space = space
        self._backend_factory = backend_factory
        self._on_ready = on_ready
        self._backend = None
        self._load_error: str | None = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._bound = threading.Event()
        self._serve_thread = threading.Thread(target=self._serve_loop, daemon=True)
        self._load_thread = threading.Thread(target=self._load, daemon=True)

    @property
    def space_id(self) -> str:
        return self._space.space_id

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def start(self) -> None:
        self._serve_thread.start()
        if not self._bound.wait(timeout=_BIND_TIMEOUT_SEC):
            raise RuntimeError("server inferenza: bind non completato")
        self._load_thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._serve_thread.join(timeout=2)
        self._load_thread.join(timeout=2)

    def _load(self) -> None:
        try:
            backend = self._backend_factory()
            backend.embed(EmbedRequest(request_id="warmup", is_query=True, text_payload="warmup"))
            self._backend = backend
            if self._on_ready is not None:
                # prima che health risponda ready: chi osserva ready vede il log già emesso
                try:
                    self._on_ready()
                except Exception:
                    pass
            self._ready.set()
        except Exception as exc:  # nessun fallback: l'errore resta visibile in health
            self._load_error = f"{type(exc).__name__}: {exc}"

    def _serve_loop(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(self._endpoint)
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        self._bound.set()
        try:
            while not self._stop.is_set():
                if not poller.poll(timeout=50):
                    continue
                raw = socket.recv_string()
                socket.send_string(self._handle(raw))
        finally:
            socket.close(linger=0)
            context.term()

    def _error_reply(self, request_id: str, code: str, message: str) -> str:
        return pack(
            "embed.response",
            EmbedResponse(
                request_id=request_id,
                success=False,
                space_id=self._space.space_id,
                error_code=code,
                error=message,
            ),
        )

    def _handle(self, raw: str) -> str:
        try:
            payload = unpack(raw, "health.request")
            if payload != {}:
                raise ValueError("health.request con payload non vuoto")
        except ValueError:
            pass
        else:
            return pack_health(
                "health.response",
                {
                    "ready": self._ready.is_set(),
                    "space_id": self._space.space_id,
                    "backend": self._space.backend,
                    "error": self._load_error,
                },
            )
        try:
            req = decode_embed_request(raw)
        except ValueError as exc:
            # request_id illeggibile: stringa vuota e INVALID_REQUEST (contratto §3)
            return self._error_reply("", "INVALID_REQUEST", str(exc))
        if not self._ready.is_set():
            return self._error_reply(
                req.request_id,
                "MODEL_UNAVAILABLE",
                self._load_error or "modello non ancora pronto",
            )
        try:
            vector = self._backend.embed(req)
        except EmbeddingError as exc:
            return self._error_reply(req.request_id, exc.code, exc.message)
        except Exception as exc:
            return self._error_reply(req.request_id, "MODEL_UNAVAILABLE", f"{type(exc).__name__}: {exc}")
        return pack(
            "embed.response",
            EmbedResponse(
                request_id=req.request_id,
                success=True,
                space_id=self._space.space_id,
                embedding=vector,
            ),
        )
