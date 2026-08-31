"""Structured logging and per-request correlation.

Companion to `errors.py`: that file says *what* can go wrong, this one is
how you find out that it did. There was previously no logging anywhere in
`src/finvault/` — a failed request left nothing behind but an audit row
containing `str(exc)`, with no type, no traceback, and no way to connect
the API 503 to the agent call that caused it.

Two things live here:

**A request id that follows the work.** `RequestContextMiddleware` (see
`api/error_handlers.py`) stamps one per request and every log record
emitted while handling it carries the same `request_id`, so
`grep <id>` reconstructs the whole request — HTTP entry, orchestrator
step, LLM failure, HTTP exit. It is returned to the client in the
`X-Request-ID` response header and in the body of every error, so a bug
report that quotes it is enough to find the logs.

**One way to log an exception.** `log_exception()` picks the level from the
exception's branch in the `errors.py` hierarchy — expected refusals at
WARNING without a traceback, incidents at ERROR with one — so call sites
never decide that themselves and the levels stay consistent across the
codebase.

Usage:

    from finvault.observability import extra_fields, get_logger, log_exception

    logger = get_logger(__name__)
    logger.info("ingest_started", extra=extra_fields(document_id=doc.id))

    try:
        ...
    except FinVaultError as exc:
        log_exception(logger, exc, "ingest_failed", document_id=doc.id)

Always attach structured data with `extra_fields(...)` rather than a bare
`extra={...}`: it guards against field names that collide with stdlib
`LogRecord` attributes, which `logging` rejects by raising — turning a log
line into an exception, on an error path, where it does the most damage.

Context propagation caveat: `contextvars` follow `await` and
`asyncio.to_thread`, but **not** a bare `threading.Thread`. Anything that
starts a raw thread must carry the context across explicitly — see
`capture_context()` / `run_in_request_context()` and their use by the SSE
route in `api/routes.py`. That handoff covers every contextvar, not just
this module's log fields: the tenant scope in `security/rls.py` rides the
same mechanism, and losing it is a correctness bug, not a logging one.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

from finvault.errors import ClientError, FinVaultError, PolicyError

T = TypeVar("T")

# Fields carried alongside the current unit of work. Set once per request by
# RequestContextMiddleware and stamped onto every record by the log-record
# factory below, so no call site has to thread a request id through its
# arguments.
# `None` rather than `{}` as the default: a mutable default on a ContextVar
# is one shared object across every context that never sets one, so a single
# accidental in-place mutation anywhere would leak fields between requests.
# `_context()` normalizes the None away for readers.
_request_context: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "finvault_request_context", default=None
)


def _context() -> dict[str, Any]:
    return _request_context.get() or {}


# Attributes stdlib puts on every LogRecord. Anything NOT in here was added
# by a caller, and the JSON formatter emits it as a structured field.
_STDLIB_RECORD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",
    "asctime",
    "taskName",
    "fields",
}


def new_request_id() -> str:
    return uuid.uuid4().hex


def current_request_id() -> str | None:
    return _context().get("request_id")


def request_context() -> dict[str, Any]:
    """A copy of the current context — safe to hand to another thread."""
    return dict(_context())


@contextmanager
def bind_request_context(**fields: Any) -> Iterator[dict[str, Any]]:
    """Adds fields to every log record emitted inside the block.

    Merges with (rather than replaces) any enclosing context, and restores
    the previous context on exit even if the block raises.
    """
    merged = {**_context(), **{k: v for k, v in fields.items() if v is not None}}
    token = _request_context.set(merged)
    try:
        yield merged
    finally:
        _request_context.reset(token)


def add_request_context(**fields: Any) -> None:
    """Adds fields to the current context for the rest of the request, with
    no block to exit — for values discovered mid-request, such as the actor
    resolved from a token.

    Propagation caveat, and the reason `api/auth.get_current_user` is
    `async`: a `contextvars.ContextVar` set inside a worker thread is set on
    that thread's *copy* of the context and is discarded when it finishes.
    FastAPI runs `def` dependencies and `def` routes in a threadpool, so a
    call from one of those is silently lost. Call this only from async code
    running in the request's own task; the enclosing `bind_request_context`
    in `RequestContextMiddleware` still restores the prior context on exit,
    so nothing leaks between requests.
    """
    _request_context.set({**_context(), **{k: v for k, v in fields.items() if v is not None}})


def capture_context() -> contextvars.Context:
    """Snapshots every contextvar for handing to a raw `threading.Thread`.

    Call on the originating thread; pass the result to
    `run_in_request_context()` on the new one.

    Deliberately `contextvars.copy_context()` rather than a dict of the
    logging fields this module owns. A request carries more than log
    context — most importantly the tenant that `security/rls.py` scopes
    every database transaction to — and a handoff that restored only the
    log fields would give the new thread correctly correlated log lines and
    no tenant at all. That failure is close to invisible: with Row Level
    Security on, an unscoped transaction reads zero rows and writes trip the
    policy's WITH CHECK, so it surfaces as "the query found nothing" rather
    than as an error pointing back here.
    """
    return contextvars.copy_context()


def run_in_request_context(context: contextvars.Context, fn: Callable[[], T]) -> T:
    """Runs `fn` with a captured context restored — for work handed to a raw
    `threading.Thread`, which does not inherit contextvars the way
    `asyncio.to_thread` does. Capture with `capture_context()` on the
    originating thread, pass the result here on the new one.

    A `Context` may not be entered twice, so capture one per handoff rather
    than reusing a single snapshot across threads.
    """
    return context.run(fn)


_record_factory_installed = False


def _install_record_factory() -> None:
    """Stamps the current request context onto every LogRecord as it is
    created.

    A record factory rather than a `logging.Filter` on our handler, because
    a filter only runs for records reaching *that* handler: pytest's caplog,
    an APM agent's handler, or anything else attached later would see
    records with no `request_id`, and the correlation guarantee would hold
    only for our own console output. The factory runs before any handler
    exists, so every consumer sees the same fields.

    Called at import rather than from `configure_logging()` for the same
    reason — a library caller that never configures logging (a test, a
    notebook) still gets correlated records. It only ever *adds* attributes,
    never overwrites one a caller set explicitly, and is idempotent.
    """
    global _record_factory_installed
    if _record_factory_installed:
        return

    previous = logging.getLogRecordFactory()

    def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = previous(*args, **kwargs)
        for key, value in _context().items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return record

    logging.setLogRecordFactory(factory)
    _record_factory_installed = True


_install_record_factory()


class JsonFormatter(logging.Formatter):
    """One JSON object per line — the format log aggregators index without
    a custom parser. Use FINVAULT_LOG_FORMAT=text for a readable console
    during local development.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        # Supports a raw extra={"fields": {...}} call as well as the
        # extra_fields() form the codebase uses, so a caller who reaches for
        # the stdlib idiom directly still gets structured output.
        payload.update(getattr(record, "fields", {}) or {})
        payload.update(
            {k: v for k, v in record.__dict__.items() if k not in _STDLIB_RECORD_ATTRS and not k.startswith("_")}
        )
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable development output: the same fields, laid out for
    eyes instead of an indexer.
    """

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)-8s %(name)s %(message)s", datefmt="%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        fields = {
            **(getattr(record, "fields", {}) or {}),
            **{k: v for k, v in record.__dict__.items() if k not in _STDLIB_RECORD_ATTRS and not k.startswith("_")},
        }
        if fields:
            base += " | " + " ".join(f"{k}={v}" for k, v in fields.items())
        return base


_configured = False


def configure_logging(*, level: str | None = None, fmt: str | None = None, force: bool = False) -> None:
    """Installs FinVault's handler on the root logger.

    Called once from the FastAPI lifespan and from `scripts/`. Idempotent:
    repeated calls are no-ops unless `force=True`, so importing a module
    that configures logging can never duplicate handlers (and duplicate
    every line in the log).
    """
    global _configured
    if _configured and not force:
        return

    from finvault.config import settings

    level_name = (level or settings.finvault_log_level).upper()
    formatter = JsonFormatter() if (fmt or settings.finvault_log_format).lower() == "json" else TextFormatter()

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level_name)

    # Uvicorn installs its own handlers; let records reach ours instead so
    # access logs and application logs share one format and one request id.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """The logger for a module. Always `get_logger(__name__)` so the log's
    `logger` field points at the file that emitted it.
    """
    return logging.getLogger(name)


def extra_fields(**fields: Any) -> dict[str, Any]:
    """Builds the `extra=` dict for a log call:

        logger.info("ingest_started", extra=extra_fields(document_id=doc.id))

    Each field becomes a real attribute on the LogRecord, so it is visible
    to every handler and formatter — ours, an APM agent's, pytest's caplog —
    rather than being buried in a nested dict only our formatter knows how
    to unpack.

    A field whose name collides with a stdlib `LogRecord` attribute
    (`message`, `name`, `module`, `args`, …) is prefixed rather than
    dropped: passing one unprefixed makes `logging` raise
    "Attempt to overwrite %r in LogRecord", which would turn a log line
    into an exception — the last thing an error path needs.
    """
    safe = {k: v for k, v in fields.items() if k not in _STDLIB_RECORD_ATTRS and not k.startswith("_")}
    collisions = {f"field_{k}": v for k, v in fields.items() if k not in safe}
    return {**safe, **collisions}


def log_level_for(exc: BaseException) -> int:
    """Severity from the exception's branch in `errors.py`.

    A refused request is not an outage: `ClientError` and `PolicyError` are
    the system working correctly and would otherwise drown real incidents
    in ERROR-level noise. Everything else — including any non-FinVault
    exception, which by definition was unforeseen — is an incident.
    """
    if isinstance(exc, (ClientError, PolicyError)):
        return logging.WARNING
    return logging.ERROR


def log_exception(logger: logging.Logger, exc: BaseException, event: str, **fields: Any) -> None:
    """Logs `exc` at the level its branch warrants, with its `context`
    merged into the structured fields.

    Tracebacks are attached for incidents only. An expected refusal already
    says everything useful in its message and fields; a traceback there is
    noise that trains readers to skip stack traces.
    """
    level = log_level_for(exc)
    merged: dict[str, Any] = dict(fields)
    if isinstance(exc, FinVaultError):
        merged.update(exc.log_fields())
    else:
        merged.setdefault("error_type", type(exc).__name__)
    merged.setdefault("error_message", str(exc))
    # Pass the exception object rather than True: this stays correct when
    # called outside an `except` block (a fail-closed path that stored the
    # exception and logs it later), and omits the traceback entirely when
    # the exception was never raised, instead of printing "NoneType: None".
    attach_traceback = level >= logging.ERROR and exc.__traceback__ is not None
    logger.log(level, event, extra=extra_fields(**merged), exc_info=exc if attach_traceback else None)


__all__ = [
    "configure_logging",
    "get_logger",
    "log_exception",
    "extra_fields",
    "log_level_for",
    "bind_request_context",
    "add_request_context",
    "request_context",
    "capture_context",
    "run_in_request_context",
    "current_request_id",
    "new_request_id",
    "JsonFormatter",
    "TextFormatter",
]
