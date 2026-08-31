"""End-to-end behavior of api/error_handlers.py, over real HTTP.

Unlike the rest of this codebase's route tests (which call handlers
directly — see test_compare_route.py's docstring), these go through
TestClient on purpose: the thing under test *is* the middleware and
exception-handler layer, and calling a route function directly bypasses
exactly the code these assertions are about.

The app here is a throwaway with toy routes rather than the real
create_app(), so no Postgres, Qdrant, or embedding model is needed to
verify the error contract.
"""

from __future__ import annotations

import logging

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel

from finvault.api.error_handlers import REQUEST_ID_HEADER, install_error_handlers
from finvault.errors import AccessDeniedError, AgentExecutionError, InvalidRequestError


class _Body(BaseModel):
    count: int


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    install_error_handlers(app)

    @app.get("/ok")
    def ok() -> dict:
        return {"status": "ok"}

    @app.get("/denied")
    def denied() -> dict:
        raise AccessDeniedError(
            "Role 'analyst' lacks clearance for restricted document doc-42",
            context={"document_id": "doc-42"},
        )

    @app.get("/unavailable")
    def unavailable() -> dict:
        raise AgentExecutionError("openrouter returned 402: insufficient credits at https://internal.example")

    @app.get("/invalid")
    def invalid() -> dict:
        raise InvalidRequestError("two ids required", user_message="At least two document_ids are required.")

    @app.get("/http-exception")
    def http_exception() -> dict:
        raise HTTPException(status_code=418, detail="teapot")

    @app.get("/bug")
    def bug() -> dict:
        raise RuntimeError("postgresql://finvault:hunter2@db:5432 connection reset")

    @app.post("/validated")
    def validated(body: _Body) -> dict:
        return {"count": body.count}

    # raise_server_exceptions=False so the unhandled-exception handler's
    # response is observable instead of the exception being re-raised into
    # the test, which is what the real server does for a client.
    return TestClient(app, raise_server_exceptions=False)


def test_a_domain_error_becomes_its_own_status_and_code(client: TestClient) -> None:
    response = client.get("/denied")
    assert response.status_code == 403
    error = response.json()["error"]
    assert error["code"] == "access_denied"
    assert error["retryable"] is False


def test_an_error_response_never_leaks_the_internal_message(client: TestClient) -> None:
    """The operator-facing message and the context both stay in the log.
    This is the assertion that keeps a document id, a DSN, or a prompt
    fragment out of a client's hands."""
    denied = client.get("/denied").json()["error"]
    assert "doc-42" not in denied["message"]
    assert denied["message"] == AccessDeniedError.user_message

    unavailable = client.get("/unavailable").json()["error"]
    assert "openrouter" not in unavailable["message"]
    assert "internal.example" not in unavailable["message"]


def test_an_unhandled_bug_is_an_opaque_500(client: TestClient) -> None:
    response = client.get("/bug")
    assert response.status_code == 500
    error = response.json()["error"]
    assert error["code"] == "internal_error"
    # The RuntimeError's message contains a credential-bearing DSN; none of
    # it may appear in the response.
    assert "hunter2" not in response.text
    assert "postgresql" not in response.text


def test_a_dependency_failure_tells_the_caller_it_is_worth_retrying(client: TestClient) -> None:
    response = client.get("/unavailable")
    assert response.status_code == 503
    assert response.json()["error"]["retryable"] is True


def test_a_per_instance_user_message_reaches_the_caller(client: TestClient) -> None:
    response = client.get("/invalid")
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "At least two document_ids are required."


def test_http_exceptions_and_validation_errors_use_the_same_envelope(client: TestClient) -> None:
    """One response shape for every failure means a client parses errors
    once, whether they came from us, from FastAPI, or from Pydantic."""
    teapot = client.get("/http-exception")
    assert teapot.status_code == 418
    assert teapot.json()["error"]["code"] == "http_418"

    invalid_body = client.post("/validated", json={"count": "not-a-number"})
    assert invalid_body.status_code == 422
    error = invalid_body.json()["error"]
    assert error["code"] == "validation_error"
    # The per-field detail is included: it describes the caller's own input.
    assert error["details"], error

    missing_route = client.get("/no-such-route")
    assert missing_route.status_code == 404
    assert "error" in missing_route.json()


def test_every_response_carries_a_request_id_in_body_and_header(client: TestClient) -> None:
    response = client.get("/denied")
    request_id = response.headers[REQUEST_ID_HEADER]
    assert request_id
    # Same id in both places, so a user quoting either one is enough to find
    # the request in the logs.
    assert response.json()["error"]["request_id"] == request_id

    # Successful responses get one too — needed to correlate a request that
    # *looked* fine with the warnings logged while serving it.
    assert client.get("/ok").headers[REQUEST_ID_HEADER]


def test_an_inbound_request_id_is_honored_when_it_is_safe(client: TestClient) -> None:
    """Lets a caller's own trace id follow the request through our logs."""
    response = client.get("/ok", headers={REQUEST_ID_HEADER: "caller-trace-123"})
    assert response.headers[REQUEST_ID_HEADER] == "caller-trace-123"


def test_a_hostile_inbound_request_id_is_replaced_not_sanitized(client: TestClient) -> None:
    """A request id goes straight into log records, so an id containing a
    newline could forge log lines. Rejected outright rather than cleaned."""
    forged = "abc\ninjected fake log line"
    response = client.get("/ok", headers={REQUEST_ID_HEADER: forged})
    assert response.headers[REQUEST_ID_HEADER] != forged
    assert "\n" not in response.headers[REQUEST_ID_HEADER]

    # Same for an unbounded one, which would otherwise bloat every record.
    long_id = "x" * 500
    assert client.get("/ok", headers={REQUEST_ID_HEADER: long_id}).headers[REQUEST_ID_HEADER] != long_id


def test_the_request_id_ties_the_response_to_its_log_records(client: TestClient, caplog) -> None:
    """The correlation guarantee the whole design rests on: the id a caller
    is handed is on the log record for the failure they hit."""
    with caplog.at_level(logging.WARNING):
        response = client.get("/denied")

    request_id = response.headers[REQUEST_ID_HEADER]
    failures = [r for r in caplog.records if r.getMessage() == "request_failed"]
    assert failures, caplog.records
    assert getattr(failures[0], "request_id", None) == request_id
    assert getattr(failures[0], "error_code", None) == "access_denied"
    # The context that was kept out of the response is present in the log.
    assert getattr(failures[0], "document_id", None) == "doc-42"


def test_expected_refusals_and_bugs_are_logged_at_different_levels(client: TestClient, caplog) -> None:
    with caplog.at_level(logging.DEBUG):
        client.get("/denied")
        client.get("/bug")

    by_event = {r.getMessage(): r for r in caplog.records}
    assert by_event["request_failed"].levelno == logging.WARNING
    assert by_event["unhandled_exception"].levelno == logging.ERROR
    # A bug gets a traceback attached; a refusal does not.
    assert by_event["unhandled_exception"].exc_info is not None
    assert by_event["request_failed"].exc_info is None
