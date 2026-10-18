"""Client REQ del server di inferenza (contratto 1.0).

Utilizzato da un solo thread. Un timeout chiude il REQ con LINGER=0 e la
chiamata successiva ricrea il socket. Il client non ripete mai una richiesta
autonomamente: il chiamante può riprovare una volta con la stessa richiesta.
"""

import zmq

from src.contracts import (
    EmbedRequest,
    decode_embed_response,
    decode_health_response,
    pack,
    pack_health,
    validate_vector,
)


class EmbeddingError(Exception):
    """Errore del dominio embedding con codice del contratto."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class EmbedClient:
    """Client REQ seriale: embed, health e close idempotente."""

    def __init__(self, endpoint: str, expected_space_id: str, timeout_ms: int = 30000):
        self._endpoint = endpoint
        self._expected_space_id = expected_space_id
        self._timeout_ms = timeout_ms
        self._context = zmq.Context.instance()
        self._socket: zmq.Socket | None = None

    def _ensure_socket(self) -> zmq.Socket:
        if self._socket is None:
            socket = self._context.socket(zmq.REQ)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
            socket.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
            socket.connect(self._endpoint)
            self._socket = socket
        return self._socket

    def _reset_socket(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None

    def _exchange(self, raw_request: str) -> str:
        try:
            socket = self._ensure_socket()
            socket.send_string(raw_request)
            return socket.recv_string()
        except zmq.Again:
            self._reset_socket()
            raise EmbeddingError("TIMEOUT", "embedding request timed out")
        except zmq.ZMQError as exc:
            self._reset_socket()
            raise EmbeddingError("MODEL_UNAVAILABLE", f"trasporto ZMQ guasto: {exc}") from exc

    def embed(self, req: EmbedRequest) -> list[float]:
        raw = self._exchange(pack("embed.request", req))
        try:
            response = decode_embed_response(raw)
        except ValueError as exc:
            raise EmbeddingError("INVALID_RESPONSE", f"risposta non conforme: {exc}") from exc
        if response.request_id != req.request_id:
            raise EmbeddingError("INVALID_RESPONSE", f"request_id non corrisponde: {response.request_id!r}")
        if response.space_id != self._expected_space_id:
            raise EmbeddingError(
                "SPACE_MISMATCH",
                f"atteso {self._expected_space_id}, server ha {response.space_id}",
            )
        if not response.success:
            raise EmbeddingError(response.error_code or "MODEL_UNAVAILABLE", response.error or "")
        try:
            validate_vector(response.embedding)
        except ValueError as exc:
            raise EmbeddingError("INVALID_RESPONSE", f"vettore non valido: {exc}") from exc
        return response.embedding

    def health(self) -> dict:
        raw = self._exchange(pack_health("health.request", {}))
        try:
            return decode_health_response(raw)
        except ValueError as exc:
            raise EmbeddingError("INVALID_RESPONSE", f"health non conforme: {exc}") from exc

    def close(self) -> None:
        self._reset_socket()
