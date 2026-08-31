"""The single source of truth for every failure this system can produce.

**Read this file first when debugging an error.** Every exception FinVault
raises on purpose lives here, in one hierarchy, with one shape. Nothing
else in `src/finvault/` defines its own exception class — the older
per-module classes (`agents.base.AgentExecutionError`,
`security.access_control.AccessDeniedError`, …) are now thin re-exports of
the names below, so `except` clauses written against them still work.

The three questions a reader has, and where each is answered:

1. *What went wrong?*      -> the exception's class and `code`
2. *Whose fault is it?*    -> which of the four branches it sits in
3. *What do I do now?*     -> `retryable`, and the "Handling" line in the
                              class docstring

Four branches, deliberately shallow:

    FinVaultError                     every deliberate failure
    ├── ClientError                   the caller got it wrong -> 4xx
    │   ├── AuthenticationError       401  bad/expired token
    │   ├── AccessDeniedError         403  clearance or org isolation
    │   ├── NotFoundError             404  no such resource for this caller
    │   ├── InvalidRequestError       400  well-formed but unusable input
    │   ├── PayloadTooLargeError      413  upload over the configured cap
    │   ├── UnsupportedDocumentError  415  loader can't read this file type
    │   ├── RateLimitExceeded         429  too many requests for this identity
    │   └── ReviewQueueError          400  bad compliance-queue operation
    │       └── ReviewItemNotFoundError  404  unknown item, or another org's
    ├── PolicyError                   the system refused on purpose -> 403
    │   └── ExternalizationBlocked    classification may not reach the LLM
    ├── DependencyError               something we depend on failed -> 503
    │   ├── AgentExecutionError       the LLM call failed
    │   │   ├── TokenBudgetExceeded   per-request token ceiling hit
    │   │   └── UpstreamProtocolError HTTP 200 with an unusable body
    │   └── StorageError              Postgres / Qdrant / graph store failed
    └── InternalError                 our bug -> 500

`ClientError` and `PolicyError` are *expected* outcomes: they are logged at
WARNING, and their `user_message` is safe to return verbatim.
`DependencyError` and `InternalError` are *incidents*: they are logged at
ERROR with a traceback, and the caller sees only the generic
`user_message`, never `str(exc)` — see `api/error_handlers.py`, which is
the only place that turns any of these into an HTTP response.

Raising one:

    raise AccessDeniedError(
        f"Role '{role}' lacks clearance for {classification}",  # for the log
        context={"role": role, "classification": classification},
    )

The positional message is for operators and never reaches a client. Put
anything you would want to filter or group logs by in `context` — it is
emitted as structured log fields, so it must not contain plaintext
document content, secrets, or PII.
"""

from __future__ import annotations

from typing import Any


class FinVaultError(Exception):
    """Base class for every deliberate failure in FinVault.

    Catch this to mean "a known failure mode occurred". Anything that is
    *not* a `FinVaultError` escaping into the API layer is by definition a
    bug — `api/error_handlers.py` treats it as one and logs it accordingly.

    Class attributes are the contract; instances only carry the details:

    - `code`         stable, machine-readable, snake_case. Clients and log
                     queries match on this, so it never changes once shipped
                     even if the class is renamed.
    - `http_status`  what `api/error_handlers.py` returns for it.
    - `user_message` safe to show a caller: no internals, no `str(exc)`.
    - `retryable`    True only if retrying the identical request could
                     plausibly succeed without anything else changing.
    """

    # Distinct from InternalError's "internal_error" so that a bare
    # FinVaultError — which nothing should raise; pick a leaf class — is
    # visibly different in logs from a deliberate InternalError.
    code: str = "unexpected_error"
    http_status: int = 500
    user_message: str = "An unexpected error occurred. Please try again shortly."
    retryable: bool = False

    def __init__(
        self,
        message: str = "",
        *,
        context: dict[str, Any] | None = None,
        user_message: str | None = None,
    ) -> None:
        super().__init__(message or self.user_message)
        # Operator-facing detail. Distinct from `user_message` on purpose:
        # this may name internal components, IDs, and upstream errors.
        self.message = message or self.user_message
        # Structured log fields. Keep it small, non-sensitive, and stable —
        # these keys are what someone greps for at 3am.
        self.context: dict[str, Any] = dict(context or {})
        # Per-instance override for the rare case a caller can be told
        # something more useful than the class default.
        if user_message is not None:
            self.user_message = user_message

    def to_dict(self) -> dict[str, Any]:
        """The client-facing error body (minus `request_id`, which
        `api/error_handlers.py` adds because only it knows the request).
        """
        return {"code": self.code, "message": self.user_message, "retryable": self.retryable}

    def log_fields(self) -> dict[str, Any]:
        """Structured fields for one log record about this error."""
        return {"error_code": self.code, "error_type": type(self).__name__, **self.context}

    def __str__(self) -> str:
        if not self.context:
            return self.message
        details = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
        return f"{self.message} [{details}]"


# --------------------------------------------------------------------------
# ClientError — the caller got it wrong. Expected; not an incident.
# --------------------------------------------------------------------------


class ClientError(FinVaultError):
    """The request itself is the problem, so retrying it unchanged will
    fail identically. Logged at WARNING; `message` is safe to return
    because these are raised with caller-facing wording already.

    Handling: let it propagate to `api/error_handlers.py`. Do not wrap it
    in an `HTTPException` at the call site — that duplicates the mapping.
    """

    code = "bad_request"
    http_status = 400
    user_message = "The request could not be processed as submitted."


class AuthenticationError(ClientError):
    """No usable identity: missing, malformed, or expired token.

    Raised by: `api/auth.py`.
    Handling: 401. Never say *which* part of the token was wrong — that
    detail goes to the log, not the response.
    """

    code = "authentication_failed"
    http_status = 401
    user_message = "Authentication failed. Provide a valid bearer token."


class AccessDeniedError(ClientError):
    """Identity is known, but clearance or org membership forbids this.

    Raised by: `security/access_control.py`, and re-raised through every
    retrieval path before any plaintext is produced.
    Handling: 403. Retrieval paths *silently drop* inaccessible records
    rather than raising, so that a 403 never confirms a document exists —
    see `retrieval/retriever.py`. Raise this only where the caller already
    knows the resource identifier they supplied.
    """

    code = "access_denied"
    http_status = 403
    user_message = "You do not have clearance for this resource."


class NotFoundError(ClientError):
    """No such resource *for this caller* — genuinely absent and
    "exists but out of your org" are deliberately indistinguishable.

    Handling: 404.
    """

    code = "not_found"
    http_status = 404
    user_message = "The requested resource was not found."


class InvalidRequestError(ClientError):
    """Schema-valid but semantically unusable (e.g. one document id sent to
    an endpoint that compares two).

    Handling: 400. FastAPI/Pydantic already reject schema violations before
    a route runs; this is for the checks Pydantic cannot express.
    """

    code = "invalid_request"
    http_status = 400
    user_message = "The request was understood but could not be processed."


class PayloadTooLargeError(ClientError):
    """Upload exceeds `settings.finvault_max_upload_size_mb`.

    Raised by: `api/uploads.py`, both before reading (declared size) and
    while streaming (actual bytes) — a client can lie about the former.
    Handling: 413. The temp file is removed by the raiser before it
    propagates.
    """

    code = "payload_too_large"
    http_status = 413
    user_message = "The uploaded file exceeds the maximum allowed size."


class UnsupportedDocumentError(ClientError):
    """No loader for this file type.

    Raised by: `ingestion/loaders.py`.
    Handling: 415.
    """

    code = "unsupported_document_type"
    http_status = 415
    user_message = "This file type is not supported."


class RateLimitExceeded(ClientError):
    """The caller exceeded a request-rate window.

    Raised by: `api/rate_limit.py`.
    Handling: 429, with a `Retry-After` header derived from
    `context["retry_after_seconds"]` — `api/error_handlers.py` reads it from
    there, so raising this from anywhere gets the header for free.

    The one `ClientError` that is retryable: waiting is exactly what makes
    the identical request succeed. `retry_after_seconds` tells the client
    how long, which is the difference between a well-behaved backoff and a
    client hammering a limit it cannot see.
    """

    code = "rate_limit_exceeded"
    http_status = 429
    user_message = "Too many requests. Please slow down and try again shortly."
    retryable = True


class ReviewQueueError(ClientError):
    """An invalid operation against the compliance review queue — most
    often an illegal status transition (resolving an item back to
    "pending").

    Raised by: `security/review_queue.py`.
    Handling: 400. Kept as the branch's base so the pre-existing
    `except ReviewQueueError` call sites keep catching every queue failure,
    while the subclass below lets the API return the more accurate status.
    """

    code = "review_queue_error"
    http_status = 400
    user_message = "That review queue operation is not valid."


class ReviewItemNotFoundError(ReviewQueueError):
    """No review item with this id *in the caller's org*.

    Org mismatch is reported identically to genuine absence on purpose: a
    distinguishable 403 would confirm that another organization holds an
    item with that id.
    Handling: 404.
    """

    code = "review_item_not_found"
    http_status = 404
    user_message = "The requested review item was not found."


# --------------------------------------------------------------------------
# PolicyError — the system refused on purpose. Expected; not an incident.
# --------------------------------------------------------------------------


class PolicyError(FinVaultError):
    """A security or compliance rule forbade the operation. Nothing is
    broken; the refusal *is* the correct behavior, so this is logged at
    WARNING and is never retried.

    Distinct from `ClientError` because these fire deep inside the pipeline
    on data the caller never named — a document's classification, an
    answer's content — and are usually caught and converted into a blocked
    result rather than an HTTP error.
    """

    code = "policy_violation"
    http_status = 403
    user_message = "This request was blocked by policy."


class ExternalizationBlocked(PolicyError):
    """Content at this classification may not be placed in an LLM prompt.

    Raised by: `security/guardrails.enforce_externalization_policy`.
    Handling: caught in `agents/compliance_agent.py` and turned into a
    blocked `ComplianceVerdict` — it should not normally reach the API
    layer. If it does, 403 is correct.
    """

    code = "externalization_blocked"
    http_status = 403
    user_message = "This content cannot be processed due to its classification."


# --------------------------------------------------------------------------
# DependencyError — something we depend on failed. An incident.
# --------------------------------------------------------------------------


class DependencyError(FinVaultError):
    """An external system we call (LLM provider, Postgres, Qdrant) failed.

    Logged at ERROR with a traceback; the caller sees only `user_message`,
    because `str(exc)` here can carry upstream URLs, model names, and
    provider error bodies.

    Handling: **fail closed**. Every catch site must decline to answer
    rather than degrade to a guess — a missing retrieval result silently
    treated as "nothing found" would produce a confidently wrong answer,
    which is the single worst outcome for this system.
    """

    code = "dependency_unavailable"
    http_status = 503
    user_message = "A required service is temporarily unavailable. Please try again shortly."
    retryable = True


class AgentExecutionError(DependencyError):
    """The LLM call itself failed: auth, billing, rate limit, network, or a
    response the loop cannot proceed from.

    Raised by: `agents/base.py` only — no other module constructs this, so
    every instance comes from one of that file's five raise sites.
    Handling: fail closed. `agents/base.py` deliberately re-raises this
    (rather than returning it to the model as a tool-result string) when it
    surfaces from a nested agent, so a broken sub-agent can never be
    "worked around" by the parent model. Retries for transient causes
    already happened inside `_create_completion` before this was raised —
    do not add another retry loop on top.
    """

    code = "agent_execution_failed"
    http_status = 503
    user_message = "The assistant is temporarily unavailable and could not process this request."
    retryable = True


class TokenBudgetExceeded(AgentExecutionError):
    """The per-request `TokenBudget` shared across the whole
    Orchestrator -> Retriever -> Analyst chain ran out mid-request.

    Subclasses `AgentExecutionError` on purpose: every existing
    `except AgentExecutionError` handler gets identical fail-closed
    treatment without knowing budgets exist. Only the reported
    `block_reason` differs.
    Handling: not retryable — an identical request spends identically.
    Raise `finvault_max_tokens_per_request` or narrow the question.
    """

    code = "token_budget_exceeded"
    user_message = "This request exceeded its processing budget. Try a narrower question."
    retryable = False


class UpstreamProtocolError(AgentExecutionError):
    """HTTP 200 with a body the SDK accepted but we cannot use — `choices`
    missing or null, or a turn with neither text nor a tool call.

    Observed from moderation blocks and routing errors, notably when
    retrieved content contained an embedded prompt-injection attempt. Kept
    separate from its parent so these are countable in logs: a spike here
    usually means a provider-side filter, not an outage.
    Handling: same fail-closed path as `AgentExecutionError`.
    """

    code = "upstream_protocol_error"


class StorageError(DependencyError):
    """A persistence layer (Postgres, Qdrant, the graph store) failed.

    Handling: fail closed. Do not treat a failed read as an empty result —
    see the class docstring above for why.
    """

    code = "storage_unavailable"


# --------------------------------------------------------------------------
# InternalError — our bug. An incident.
# --------------------------------------------------------------------------


class InternalError(FinVaultError):
    """A bug in FinVault: an invariant we control was violated.

    Logged at ERROR with a traceback and reported as an opaque 500. If you
    are reading this in a log, the fix is a code change, not a config or
    infrastructure change.
    """

    code = "internal_error"
    http_status = 500


class ConfigurationError(InternalError):
    """The process is misconfigured (missing key, unusable DSN, absent
    credential).

    Handling: raise at startup where possible — a deployment that cannot
    work should fail immediately and visibly, not on the first user
    request. Not retryable at any layer.
    """

    code = "configuration_error"
    user_message = "The service is misconfigured. Please contact an administrator."


__all__ = [
    "FinVaultError",
    "ClientError",
    "AuthenticationError",
    "AccessDeniedError",
    "NotFoundError",
    "InvalidRequestError",
    "PayloadTooLargeError",
    "UnsupportedDocumentError",
    "RateLimitExceeded",
    "ReviewQueueError",
    "ReviewItemNotFoundError",
    "PolicyError",
    "ExternalizationBlocked",
    "DependencyError",
    "AgentExecutionError",
    "TokenBudgetExceeded",
    "UpstreamProtocolError",
    "StorageError",
    "InternalError",
    "ConfigurationError",
]
