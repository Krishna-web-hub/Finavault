"""The only place an exception becomes an HTTP response.

Before this module, fourteen `raise HTTPException(...)` calls scattered
through `routes.py` and `auth.py` each invented their own status code and
message wording, and an unhandled exception produced Starlette's default
`Internal Server Error` with nothing logged. Now routes raise the domain
errors from `errors.py` and the mapping happens here, once.

**Every error response has the same shape**, whatever produced it:

    {
      "error": {
        "code": "access_denied",          # stable; match on this, not text
        "message": "You do not have ...", # safe to show a user
        "retryable": false,               # is an identical retry worth it
        "request_id": "9f2c...":          # also in the X-Request-ID header
      }
    }

Four handlers cover the whole surface:

    FinVaultError          -> its own http_status/code/user_message
    HTTPException          -> re-shaped into the envelope (FastAPI's own
                              404s, and any third-party raise we don't own)
    RequestValidationError -> 422 plus Pydantic's per-field detail
    Exception              -> opaque 500, logged with a traceback

The last one is the important one: an exception that is not a
`FinVaultError` is by definition unforeseen, so its message may contain
anything — connection strings, prompt fragments, decrypted content. It is
never echoed to the caller. The `request_id` is what connects the 500 the
user saw to the traceback in the log.

Adding a new failure mode is therefore two steps, neither of them here:
define the exception in `errors.py`, raise it in the code that detects it.
"""

from __future__ import annotations

import re
import time

import jwt
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.datastructures import Headers, MutableHeaders
from starlette.exceptions import HTTPException

from finvault.api.rate_limit import EXEMPT_PATHS, RateLimiter, anonymous_limiter, enforce, identify
from finvault.config import settings
from finvault.errors import ClientError, FinVaultError, InternalError, PolicyError, RateLimitExceeded
from finvault.metrics import (
    http_request_duration_seconds,
    http_requests_in_flight,
    http_requests_total,
    record_error,
)
from finvault.observability import (
    bind_request_context,
    current_request_id,
    extra_fields,
    get_logger,
    log_exception,
    new_request_id,
)

logger = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"

# An inbound request id is caller-controlled and goes straight into logs, so
# it is constrained rather than trusted: anything with newlines or control
# characters could forge log lines, and an unbounded one could bloat every
# record for the request. Reject-and-replace rather than sanitize — a
# malformed id is not worth preserving.
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def _error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    retryable: bool = False,
    request_id: str | None = None,
    extra: dict | None = None,
    retry_after_seconds: int | None = None,
) -> JSONResponse:
    body = {"code": code, "message": message, "retryable": retryable, "request_id": request_id}
    if extra:
        body.update(extra)
    headers: dict[str, str] = {}
    if request_id:
        headers[REQUEST_ID_HEADER] = request_id
    if retry_after_seconds:
        # A 429 or 503 without Retry-After leaves a client guessing, and the
        # usual guess is "immediately" — which is how a rate limit turns into
        # a retry storm against the service that was already under pressure.
        headers["Retry-After"] = str(int(retry_after_seconds))
    return JSONResponse(status_code=status_code, content={"error": body}, headers=headers or None)


def _route_label(scope) -> str:
    """The matched route's path template, for use as a metric label.

    Starlette puts the matched `Route` in `scope["route"]` once routing has
    run, which — for this middleware — is by the time the response has been
    sent. Requests that matched nothing (404s, and probe traffic scanning
    for admin panels) collapse to a single "unmatched" series instead of one
    per URL an attacker invents.
    """
    route = scope.get("route")
    return getattr(route, "path", None) or "unmatched"


class RequestContextMiddleware:
    """Assigns a request id, binds it for the duration of the request, and
    logs one line when the request starts and one when it finishes.

    Written as raw ASGI rather than `BaseHTTPMiddleware` on purpose: this
    app streams Server-Sent Events from `POST /query/stream`, and
    `BaseHTTPMiddleware` wraps the response body in a way that has
    historically broken incremental delivery. Raw ASGI passes every message
    straight through.

    Unhandled exceptions are *not* logged here — `unhandled_exception_handler`
    below owns that, and duplicating it would put two tracebacks in the log
    for one failure. This only records that the request ended.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        inbound = Headers(scope=scope).get(REQUEST_ID_HEADER.lower())
        request_id = inbound if inbound and _SAFE_REQUEST_ID.match(inbound) else new_request_id()

        started = time.perf_counter()
        # http.response.start may never arrive if the app raises, so the
        # status is tracked as we go rather than read from a return value.
        state = {"status": 500}

        async def send_with_request_id(message) -> None:
            if message["type"] == "http.response.start":
                state["status"] = message["status"]
                MutableHeaders(scope=message).setdefault(REQUEST_ID_HEADER, request_id)
            await send(message)

        method = scope.get("method", "-")

        with bind_request_context(request_id=request_id, method=method, path=scope.get("path")):
            logger.info("request_started")
            http_requests_in_flight.inc()
            try:
                await self.app(scope, receive, send_with_request_id)
            finally:
                duration = time.perf_counter() - started
                http_requests_in_flight.dec()
                # The route *template* ("/documents/{item_id}"), never the raw
                # path. A metric labelled with the concrete path would create
                # one time series per document id — the classic way to melt a
                # Prometheus server. `_route_label` resolves it, and collapses
                # anything unmatched rather than admitting an arbitrary string.
                route = _route_label(scope)
                http_requests_total.labels(method=method, route=route, status_class=f"{state['status'] // 100}xx").inc()
                http_request_duration_seconds.labels(method=method, route=route).observe(duration)
                logger.info(
                    "request_finished",
                    extra=extra_fields(status=state["status"], duration_ms=round(duration * 1000, 2)),
                )


def _branch_of(exc: FinVaultError) -> str:
    """The metric label for an exception's branch.

    Two values, not four, because this is what an alert rule needs to
    distinguish: `expected` covers ClientError and PolicyError — the system
    correctly refusing — and `incident` covers everything else. The precise
    class is already on the `error_code` label for anyone drilling in.
    """
    return "expected" if isinstance(exc, (ClientError, PolicyError)) else "incident"


async def finvault_error_handler(request: Request, exc: FinVaultError) -> JSONResponse:
    """Every deliberate failure. The exception itself carries the status
    code, the client-facing wording, and whether a retry is worthwhile —
    this handler only serializes them, which is why adding an error type
    never requires editing this file.
    """
    log_exception(logger, exc, "request_failed")
    record_error(exc.code, branch=_branch_of(exc))
    return _error_response(
        status_code=exc.http_status,
        code=exc.code,
        message=exc.user_message,
        retryable=exc.retryable,
        request_id=current_request_id(),
        # RateLimitExceeded puts this in its context; every other error
        # leaves it absent and gets no header. Reading it from `context`
        # rather than special-casing the class keeps this handler generic —
        # any future error that knows how long to wait gets the header free.
        retry_after_seconds=exc.context.get("retry_after_seconds"),
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """`HTTPException` raised by the framework itself (unmatched route,
    method not allowed) or by a dependency we don't own. Application code
    should raise a `FinVaultError` instead; this exists so those responses
    still match the one envelope clients parse.

    Registered against **Starlette's** HTTPException, not FastAPI's. FastAPI
    subclasses it, so this catches both — whereas registering FastAPI's
    would miss the router's own 404 for an unmatched path, which is raised
    as the Starlette base class and would have fallen through to FastAPI's
    default `{"detail": ...}` body, quietly breaking the one-envelope
    promise for the single most common error a client hits.
    """
    logger.warning(
        "http_exception",
        extra=extra_fields(status=exc.status_code, detail=str(exc.detail)),
    )
    return _error_response(
        status_code=exc.status_code,
        code=f"http_{exc.status_code}",
        message=str(exc.detail),
        request_id=current_request_id(),
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Request body/query that failed Pydantic validation. The per-field
    detail is included because it describes the caller's own input — no
    internal state leaks through it — and without it a 422 is unactionable.
    """
    logger.warning("request_validation_failed", extra=extra_fields(error_count=len(exc.errors())))
    return _error_response(
        status_code=422,
        code="validation_error",
        message="The request body failed validation.",
        request_id=current_request_id(),
        # jsonable_encoder-safe: errors() can contain exception objects in
        # its `ctx`, which json.dumps would choke on.
        extra={"details": [{k: str(v) for k, v in err.items() if k in {"loc", "msg", "type"}} for err in exc.errors()]},
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Anything that is not a `FinVaultError` — i.e. a bug.

    Logged at ERROR with a full traceback, and answered with a generic 500
    that reveals nothing about `exc`. If you are here because a user
    reported an opaque error, take their `request_id` and search the logs
    for it: the traceback is on the `unhandled_exception` record.
    """
    log_exception(logger, exc, "unhandled_exception")
    fallback = InternalError()
    record_error(fallback.code, branch="incident")
    return _error_response(
        status_code=fallback.http_status,
        code=fallback.code,
        message=fallback.user_message,
        request_id=current_request_id(),
    )


class RateLimitMiddleware:
    """Applies the per-identity rate limit before a request reaches a route.

    Middleware rather than a route dependency, deliberately: a dependency
    runs after FastAPI has parsed and validated the body, so an oversized or
    malformed payload from a client already over its limit would still be
    fully processed. Here the check happens on the raw scope, before any of
    that work.

    It sits *inside* `RequestContextMiddleware` — see the note on
    registration order in `install_error_handlers` — so a 429 still carries
    a request id and still shows up in the HTTP metrics like any other
    response.

    The identity comes from the bearer token's subject when one is present.
    The token is decoded here without verifying the signature — see
    `_identity_from_scope` for why that is safe and why it is *only* safe
    for this.
    """

    def __init__(self, app, *, cache) -> None:  # type: ignore[no-untyped-def]
        self.app = app
        self._user_limiter = RateLimiter(cache)
        self._anonymous_limiter = anonymous_limiter(cache)

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or scope.get("path") in EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        actor = _identity_from_scope(headers)
        client = scope.get("client")
        identity, limiter_scope = identify(headers, client_host=client[0] if client else None, actor=actor)
        limiter = self._user_limiter if limiter_scope == "user" else self._anonymous_limiter

        try:
            enforce(limiter, identity=identity, scope=limiter_scope)
        except RateLimitExceeded as exc:
            # Handled here rather than allowed to propagate. Starlette's
            # ExceptionMiddleware — which dispatches everything registered
            # with add_exception_handler — sits *inside* the user middleware
            # stack, so an exception raised at this level sails straight past
            # it to ServerErrorMiddleware and comes back as an opaque 500.
            # (That is exactly what happened the first time this was written.)
            # Calling the handler directly produces the identical envelope,
            # status and Retry-After header a route-raised error would,
            # without duplicating any of that logic here.
            response = await finvault_error_handler(Request(scope, receive), exc)
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


def _identity_from_scope(headers: Headers) -> str | None:
    """The `sub` claim from the bearer token, without verifying it.

    Unverified is acceptable *here and nowhere else*, because the worst a
    forged claim achieves is choosing which rate-limit bucket to consume —
    and consuming someone else's bucket only ever restricts the attacker
    sooner. It grants no access: `api/auth.py` verifies the signature
    properly before any data is touched, and this value never reaches
    anything but a counter key.

    Doing it this way avoids running full JWT verification twice per request
    and keeps the limiter independent of the auth dependency, which by
    design has not run yet at middleware time.
    """
    authorization = headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    try:
        claims = jwt.decode(token, options={"verify_signature": False, "verify_exp": False})
    except jwt.PyJWTError:
        return None
    subject = claims.get("sub")
    return subject if isinstance(subject, str) and subject else None


def install_error_handlers(app: FastAPI, *, cache=None) -> None:
    """Wires the middleware and all four handlers. Called once from
    `create_app()`; nothing else should register exception handlers.

    **Registration order is the reverse of execution order.**
    `add_middleware` inserts at the front of the list, so the *last* one
    registered is the *outermost* one at runtime. The rate limiter is
    therefore registered first so that it runs inside the request-context
    middleware — otherwise a 429 would be produced before a request id
    exists and before the HTTP metrics wrap the call, making rate-limited
    requests invisible in exactly the logs and dashboards you would reach
    for when investigating them. (This was written backwards first; the
    symptom was 429s missing from `finvault_http_requests_total`.)
    """
    if cache is not None and settings.finvault_enable_rate_limit:
        app.add_middleware(RateLimitMiddleware, cache=cache)
    app.add_middleware(RequestContextMiddleware)
    app.add_exception_handler(FinVaultError, finvault_error_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)


__all__ = ["install_error_handlers", "RequestContextMiddleware", "RateLimitMiddleware", "REQUEST_ID_HEADER"]
