"""Tests for finvault/observability.py — the logging and correlation layer.

The behavior worth pinning here is not "does it log" but the three
properties error handling depends on: fields survive onto the record,
context follows the work (including across a raw thread), and the level is
chosen by the exception's branch rather than by the call site.
"""

from __future__ import annotations

import json
import logging
import threading

from finvault.errors import AccessDeniedError, AgentExecutionError, ExternalizationBlocked
from finvault.observability import (
    JsonFormatter,
    TextFormatter,
    add_request_context,
    bind_request_context,
    capture_context,
    current_request_id,
    extra_fields,
    get_logger,
    log_exception,
    run_in_request_context,
)
from finvault.security.rls import current_org, org_scope

logger = get_logger("finvault.tests.observability")


def _format_json(record: logging.LogRecord) -> dict:
    return json.loads(JsonFormatter().format(record))


def test_bound_context_appears_on_records_and_is_restored_afterwards(caplog) -> None:
    with caplog.at_level(logging.INFO):
        with bind_request_context(request_id="req-1", actor="user-7"):
            logger.info("inside")
        logger.info("outside")

    inside, outside = caplog.records
    assert (inside.request_id, inside.actor) == ("req-1", "user-7")
    # The context is unwound on exit, so one request's id can never leak
    # into the next request handled by the same worker.
    assert not hasattr(outside, "request_id")
    assert current_request_id() is None


def test_nested_context_merges_rather_than_replaces(caplog) -> None:
    with caplog.at_level(logging.INFO), bind_request_context(request_id="req-1"):  # noqa: SIM117
        # Deliberately a *nested* bind rather than one combined `with`: what
        # is being tested is that an inner bind merges with the enclosing
        # context instead of replacing it, and collapsing the two into a
        # single statement would not exercise that at all.
        with bind_request_context(session_id="s-9"):
            logger.info("nested")

    record = caplog.records[0]
    assert (record.request_id, record.session_id) == ("req-1", "s-9")


def test_add_request_context_extends_the_current_request(caplog) -> None:
    with caplog.at_level(logging.INFO), bind_request_context(request_id="req-1"):
        add_request_context(actor="user-7")
        logger.info("after_auth")

    assert caplog.records[0].actor == "user-7"


def test_context_can_be_carried_into_a_raw_thread() -> None:
    """A bare threading.Thread does not inherit contextvars, which is
    exactly the case POST /query/stream hits — the orchestrator runs in one.
    Without this, every log line from a streamed query would be orphaned.
    """
    seen: dict = {}

    with bind_request_context(request_id="req-stream"):
        context = capture_context()

        def worker() -> None:
            # Proof of the hazard: the raw thread starts with nothing.
            seen["without"] = current_request_id()
            run_in_request_context(context, lambda: seen.update(within=current_request_id()))

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

    assert seen["without"] is None
    assert seen["within"] == "req-stream"


def test_the_handoff_carries_every_contextvar_not_just_log_fields() -> None:
    """Regression: the handoff used to copy only this module's log-field
    dict, so the RLS tenant scope (security/rls.py) was silently dropped on
    the way into /query/stream's orchestrator thread. With Row Level
    Security enabled that is a correctness bug wearing a logging bug's
    clothes — `app.current_org` goes empty, reads match no rows and writes
    fail WITH CHECK — and a test asserting only on the request id passes
    happily through it. Assert on a contextvar this module does not own.
    """
    seen: dict = {}

    with bind_request_context(request_id="req-stream"), org_scope("org-a"):
        context = capture_context()

        def worker() -> None:
            seen["without"] = current_org()
            run_in_request_context(context, lambda: seen.update(within=current_org(), request_id=current_request_id()))

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

    assert seen["without"] is None
    assert seen["within"] == "org-a"
    assert seen["request_id"] == "req-stream"


def test_extra_fields_land_as_real_record_attributes(caplog) -> None:
    """Real attributes, not a nested dict — so any handler (an APM agent,
    caplog, a JSON formatter) can read them."""
    with caplog.at_level(logging.INFO):
        logger.info("ingest_started", extra=extra_fields(document_id="doc-1", chunks=12))

    record = caplog.records[0]
    assert (record.document_id, record.chunks) == ("doc-1", 12)


def test_a_field_named_like_a_logrecord_attribute_does_not_break_logging(caplog) -> None:
    """`extra={"module": ...}` makes logging raise. Since these calls sit on
    error paths, a colliding field name must never be the thing that turns a
    log line into a second exception."""
    with caplog.at_level(logging.INFO):
        logger.info("collision", extra=extra_fields(module="retrieval", filename="x.pdf", document_id="doc-1"))

    record = caplog.records[0]
    assert record.field_module == "retrieval"
    assert record.document_id == "doc-1"
    # The real LogRecord attribute is intact.
    assert record.module == "test_observability"


def test_json_output_carries_event_context_and_fields() -> None:
    with bind_request_context(request_id="req-1"):
        record = logger.makeRecord(
            logger.name, logging.INFO, "f.py", 1, "query_started", (), None, extra=extra_fields(session_id="s-1")
        )

    payload = _format_json(record)
    assert payload["event"] == "query_started"
    assert payload["level"] == "INFO"
    assert payload["logger"] == logger.name
    assert payload["session_id"] == "s-1"
    assert payload["request_id"] == "req-1"
    assert "timestamp" in payload


def test_text_output_renders_the_same_fields_readably() -> None:
    record = logger.makeRecord(
        logger.name, logging.WARNING, "f.py", 1, "query_blocked", (), None, extra=extra_fields(reason="policy")
    )
    line = TextFormatter().format(record)
    assert "query_blocked" in line
    assert "reason=policy" in line


def test_log_exception_picks_its_level_from_the_error_branch(caplog) -> None:
    with caplog.at_level(logging.DEBUG):
        log_exception(logger, AccessDeniedError("no clearance"), "denied")
        log_exception(logger, ExternalizationBlocked("restricted"), "blocked")
        log_exception(logger, AgentExecutionError("provider down"), "failed")
        log_exception(logger, ValueError("bug"), "crashed")

    levels = {r.getMessage(): r.levelno for r in caplog.records}
    assert levels["denied"] == logging.WARNING
    assert levels["blocked"] == logging.WARNING
    assert levels["failed"] == logging.ERROR
    assert levels["crashed"] == logging.ERROR


def test_log_exception_merges_the_error_context_into_the_record(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        log_exception(
            logger,
            AccessDeniedError("no clearance", context={"role": "analyst"}),
            "denied",
            document_id="doc-1",
        )

    record = caplog.records[0]
    assert record.error_code == "access_denied"
    assert record.error_type == "AccessDeniedError"
    assert record.role == "analyst"
    assert record.document_id == "doc-1"


def test_a_traceback_is_attached_to_incidents_only(caplog) -> None:
    with caplog.at_level(logging.DEBUG):
        try:
            raise AgentExecutionError("provider down")
        except AgentExecutionError as exc:
            log_exception(logger, exc, "failed")
        try:
            raise AccessDeniedError("no clearance")
        except AccessDeniedError as exc:
            log_exception(logger, exc, "denied")

    by_event = {r.getMessage(): r for r in caplog.records}
    assert by_event["failed"].exc_info is not None
    # An expected refusal says everything useful in its fields; a stack
    # trace there only trains readers to skip stack traces.
    assert by_event["denied"].exc_info is None
