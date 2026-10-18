"""Test del client inferenza contro un server fake in-process (Task I1)."""

import math

import pytest
import zmq

from src.contracts import EMBED_DIM, EmbedRequest
from src.inference.client import EmbedClient, EmbeddingError

EXPECTED_SPACE = "expected-space"


def _valid_unit(vector: list[float]) -> bool:
    return (
        len(vector) == EMBED_DIM
        and all(math.isfinite(value) for value in vector)
        and abs(math.sqrt(sum(value * value for value in vector)) - 1.0) < 1e-4
    )


def test_wrong_space_is_rejected(fake_server):
    endpoint = fake_server(space_id="other-space")
    client = EmbedClient(endpoint, expected_space_id=EXPECTED_SPACE, timeout_ms=1000)
    with pytest.raises(EmbeddingError, match="SPACE_MISMATCH"):
        client.embed(EmbedRequest(request_id="q1", is_query=True, text_payload="cache"))
    client.close()


def test_embed_returns_valid_unit_vector(fake_server):
    endpoint = fake_server(space_id=EXPECTED_SPACE)
    client = EmbedClient(endpoint, expected_space_id=EXPECTED_SPACE, timeout_ms=2000)
    request = EmbedRequest(request_id="q2", is_query=True, text_payload="latenza database")
    vector = client.embed(request)
    assert _valid_unit(vector)
    # il server è deterministico: stessa richiesta, stesso vettore
    assert client.embed(request) == vector
    client.close()


def test_timeout_then_recovery(fake_server):
    endpoint = fake_server(space_id=EXPECTED_SPACE, delay_ms=500)
    client = EmbedClient(endpoint, expected_space_id=EXPECTED_SPACE, timeout_ms=100)
    request = EmbedRequest(request_id="q3", is_query=True, text_payload="lento")
    with pytest.raises(EmbeddingError) as excinfo:
        client.embed(request)
    assert excinfo.value.code == "TIMEOUT"
    assert "embedding request timed out" in str(excinfo.value)
    # la risposta in ritardo va nel vuoto (LINGER=0); si attende che il server sia libero
    assert fake_server.servers[-1].wait_replies(1)
    # il chiamante (non il client) riprova una volta con la stessa richiesta
    vector = client.embed(request)
    assert _valid_unit(vector)
    client.close()


def test_health_ready(fake_server):
    endpoint = fake_server(space_id=EXPECTED_SPACE, backend="real", ready=True)
    client = EmbedClient(endpoint, expected_space_id=EXPECTED_SPACE)
    assert client.health() == {
        "ready": True,
        "space_id": EXPECTED_SPACE,
        "backend": "real",
        "error": None,
    }
    client.close()


def test_health_not_ready(fake_server):
    endpoint = fake_server(space_id=EXPECTED_SPACE, backend="mock", ready=False, health_error="caricamento in corso")
    client = EmbedClient(endpoint, expected_space_id=EXPECTED_SPACE)
    health = client.health()
    assert health["ready"] is False
    assert health["error"] == "caricamento in corso"
    client.close()


def test_health_timeout_then_recovery(fake_server):
    endpoint = fake_server(space_id=EXPECTED_SPACE, delay_ms=500)
    client = EmbedClient(endpoint, expected_space_id=EXPECTED_SPACE, timeout_ms=100)
    with pytest.raises(EmbeddingError) as excinfo:
        client.health()
    assert excinfo.value.code == "TIMEOUT"
    assert fake_server.servers[-1].wait_replies(1)
    assert client.health()["ready"] is True
    client.close()


def test_malformed_reply_raises(fake_server):
    endpoint = fake_server(space_id=EXPECTED_SPACE, malformed="non e una risposta valida")
    client = EmbedClient(endpoint, expected_space_id=EXPECTED_SPACE, timeout_ms=2000)
    with pytest.raises(EmbeddingError) as excinfo:
        client.embed(EmbedRequest(request_id="q4", is_query=True, text_payload="testo"))
    assert excinfo.value.code == "INVALID_RESPONSE"
    client.close()


def test_wrong_request_id_raises(fake_server):
    endpoint = fake_server(space_id=EXPECTED_SPACE, wrong_request_id=True)
    client = EmbedClient(endpoint, expected_space_id=EXPECTED_SPACE, timeout_ms=2000)
    with pytest.raises(EmbeddingError, match="request_id"):
        client.embed(EmbedRequest(request_id="q5", is_query=True, text_payload="testo"))
    client.close()


def test_nan_vector_raises(fake_server):
    endpoint = fake_server(space_id=EXPECTED_SPACE, bad_vector="nan")
    client = EmbedClient(endpoint, expected_space_id=EXPECTED_SPACE, timeout_ms=2000)
    # il literal NaN è rifiutato dal parser del contratto prima della validazione
    with pytest.raises(EmbeddingError, match="INVALID_RESPONSE"):
        client.embed(EmbedRequest(request_id="q6", is_query=True, text_payload="testo"))
    client.close()


def test_zero_vector_raises(fake_server):
    endpoint = fake_server(space_id=EXPECTED_SPACE, bad_vector="zero")
    client = EmbedClient(endpoint, expected_space_id=EXPECTED_SPACE, timeout_ms=2000)
    with pytest.raises(EmbeddingError, match="INVALID_VECTOR_NORM"):
        client.embed(EmbedRequest(request_id="q7", is_query=True, text_payload="testo"))
    client.close()


def test_wrong_dimension_vector_raises(fake_server):
    endpoint = fake_server(space_id=EXPECTED_SPACE, bad_vector="dim")
    client = EmbedClient(endpoint, expected_space_id=EXPECTED_SPACE, timeout_ms=2000)
    with pytest.raises(EmbeddingError, match="INVALID_VECTOR_DIMENSION"):
        client.embed(EmbedRequest(request_id="q8", is_query=True, text_payload="testo"))
    client.close()


def test_server_error_code_propagates(fake_server):
    endpoint = fake_server(space_id=EXPECTED_SPACE, error_code="INPUT_TOO_LARGE")
    client = EmbedClient(endpoint, expected_space_id=EXPECTED_SPACE, timeout_ms=2000)
    with pytest.raises(EmbeddingError) as excinfo:
        client.embed(EmbedRequest(request_id="q9", is_query=True, text_payload="testo"))
    assert excinfo.value.code == "INPUT_TOO_LARGE"
    client.close()


def test_close_is_idempotent_and_reusable(fake_server):
    endpoint = fake_server(space_id=EXPECTED_SPACE)
    client = EmbedClient(endpoint, expected_space_id=EXPECTED_SPACE, timeout_ms=2000)
    request = EmbedRequest(request_id="q10", is_query=True, text_payload="prima")
    vector = client.embed(request)
    client.close()
    client.close()
    # dopo close il client ricrea il REQ: la stessa richiesta dà lo stesso vettore
    assert client.embed(request) == vector
    client.close()


def test_server_survives_garbage_request(fake_server):
    endpoint = fake_server(space_id=EXPECTED_SPACE)
    client = EmbedClient(endpoint, expected_space_id=EXPECTED_SPACE, timeout_ms=2000)
    raw_socket = zmq.Context.instance().socket(zmq.REQ)
    raw_socket.setsockopt(zmq.LINGER, 0)
    raw_socket.setsockopt(zmq.RCVTIMEO, 2000)
    raw_socket.connect(endpoint)
    try:
        raw_socket.send_string("{ envelope corrotto")
        reply = raw_socket.recv_string()
    finally:
        raw_socket.close(linger=0)
    assert "INVALID_REQUEST" in reply
    # un errore non termina il server: la richiesta successiva ha risposta
    vector = client.embed(EmbedRequest(request_id="q12", is_query=True, text_payload="ok"))
    assert _valid_unit(vector)
    client.close()
