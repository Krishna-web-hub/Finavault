"""Authentication failures, over real HTTP.

Every rejection here answers with the same code and the same wording. That
uniformity is the point: a 401 that varied by cause would tell a caller
probing tokens which part of their guess was wrong.
"""

from __future__ import annotations

import logging

import jwt
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from finvault.api.auth import create_access_token, get_current_user
from finvault.api.error_handlers import install_error_handlers
from finvault.config import settings
from finvault.models import Role, User


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    install_error_handlers(app)

    @app.get("/whoami")
    def whoami(user: User = Depends(get_current_user)) -> dict:
        return {"actor": user.id, "org_id": user.org_id}

    return TestClient(app, raise_server_exceptions=False)


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_a_valid_token_resolves_the_user(client: TestClient) -> None:
    user = User(username="alice", role=Role.ANALYST, org_id="org-a")
    response = client.get("/whoami", headers=_headers(create_access_token(user=user)))
    assert response.status_code == 200
    assert response.json() == {"actor": user.id, "org_id": "org-a"}


@pytest.mark.parametrize(
    "headers",
    [
        pytest.param({}, id="no_authorization_header"),
        pytest.param({"Authorization": "Bearer not-a-jwt"}, id="unparseable_token"),
        pytest.param({"Authorization": "Basic dXNlcjpwYXNz"}, id="wrong_scheme"),
    ],
)
def test_every_rejection_reports_the_same_code_and_message(client: TestClient, headers: dict) -> None:
    """A missing header used to answer `http_401` from HTTPBearer while a bad
    token answered `authentication_failed` — one condition, two codes."""
    response = client.get("/whoami", headers=headers)
    assert response.status_code == 401
    error = response.json()["error"]
    assert error["code"] == "authentication_failed"
    assert error["message"] == "Authentication failed. Provide a valid bearer token."


def test_a_token_signed_with_the_wrong_secret_is_rejected(client: TestClient) -> None:
    forged = jwt.encode({"sub": "x", "username": "x", "role": "analyst", "org_id": "org-a"}, "wrong-secret")
    assert client.get("/whoami", headers=_headers(forged)).status_code == 401


def test_a_verified_token_with_missing_claims_is_rejected(client: TestClient) -> None:
    """Signature valid, claims unusable — still a 401, not a 500 from a
    KeyError deeper in the request."""
    thin = jwt.encode({"sub": "x"}, settings.finvault_jwt_secret, algorithm="HS256")
    response = client.get("/whoami", headers=_headers(thin))
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "authentication_failed"


def test_the_reason_for_a_rejection_is_logged_but_never_returned(client: TestClient, caplog) -> None:
    """The distinction an operator needs (expired vs. forged vs. malformed)
    is exactly the distinction an attacker would use to probe."""
    with caplog.at_level(logging.WARNING):
        response = client.get("/whoami", headers=_headers("not-a-jwt"))

    failure = next(r for r in caplog.records if r.getMessage() == "request_failed")
    assert failure.jwt_error == "DecodeError"
    assert "DecodeError" not in response.text


def test_the_actor_is_bound_to_the_requests_log_records(client: TestClient, caplog) -> None:
    """Every record logged while serving an authenticated request names who
    made it, so an error found in the logs identifies its actor without a
    second lookup in the audit table."""
    user = User(username="alice", role=Role.ANALYST, org_id="org-a")
    with caplog.at_level(logging.INFO):
        client.get("/whoami", headers=_headers(create_access_token(user=user)))

    finished = next(r for r in caplog.records if r.getMessage() == "request_finished")
    assert finished.actor == user.id
    assert finished.org_id == "org-a"
