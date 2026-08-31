"""JWT authentication and role extraction.

Tokens carry `role` and `org_id` as claims — every downstream access-control
decision (retrieval clearance, ingestion clearance, org isolation) reads from
the User this module reconstructs from the verified token, never from a
client-supplied header.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from finvault.config import settings
from finvault.errors import AuthenticationError
from finvault.models import Role, User
from finvault.observability import add_request_context, get_logger
from finvault.security.rls import set_current_org

logger = get_logger(__name__)

# auto_error=False so a *missing* Authorization header reaches this module
# instead of being answered by HTTPBearer's own HTTPException. With
# auto_error=True, "no credentials" came back as code "http_401" while "bad
# credentials" came back as "authentication_failed" — two codes for one
# condition, and a client would have to match both.
_bearer = HTTPBearer(auto_error=False)
_ALGORITHM = "HS256"


def create_access_token(*, user: User, expires_minutes: int = 60) -> str:
    now = datetime.now(UTC)
    payload = {
        "sub": user.id,
        "username": user.username,
        "role": user.role.value,
        "org_id": user.org_id,
        "iat": now,
        "exp": now + timedelta(minutes=expires_minutes),
    }
    return jwt.encode(payload, settings.finvault_jwt_secret, algorithm=_ALGORITHM)


async def get_current_user(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> User:
    """Resolves the acting User from a verified JWT.

    `async` on purpose: it calls `add_request_context()`, and a contextvar
    set from the threadpool FastAPI uses for `def` dependencies would not
    survive back into the request — see that function's docstring.
    """
    if credentials is None:
        raise AuthenticationError("No Authorization header supplied", context={"reason": "missing_credentials"})

    try:
        payload = jwt.decode(credentials.credentials, settings.finvault_jwt_secret, algorithms=[_ALGORITHM])
    except jwt.PyJWTError as exc:
        # The specific PyJWT failure (expired vs. bad signature vs. wrong
        # algorithm) is a useful operator signal but a probing aid for an
        # attacker, so it goes in `context` — logged, never returned. The
        # client always gets the same generic 401 from AuthenticationError.
        raise AuthenticationError("JWT verification failed", context={"jwt_error": type(exc).__name__}) from exc
    try:
        user = User(
            id=payload["sub"], username=payload["username"], role=Role(payload["role"]), org_id=payload["org_id"]
        )
    except (KeyError, ValueError) as exc:
        raise AuthenticationError(
            "Token verified but claims are malformed",
            context={"claim_error": type(exc).__name__, "claims_present": sorted(payload)},
        ) from exc

    # Every later log record for this request carries who made it, so an
    # error found in the logs identifies its actor without re-reading the
    # audit table. Deliberately the opaque user id and org, never the raw
    # token or username-as-credential.
    add_request_context(actor=user.id, org_id=user.org_id)
    # Scopes this request's database transactions to the token's org, which
    # is what the Row Level Security policies read (security/rls.py). Set
    # from the *verified* token, never from a header or body — the whole
    # value of enforcing isolation in the database is lost if the org it
    # enforces came from the caller.
    set_current_org(user.org_id)
    return user
