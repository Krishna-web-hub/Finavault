"""Invariants of the exception taxonomy in finvault/errors.py.

These tests exist because the taxonomy's value is entirely in its
consistency: the moment two exceptions share a `code`, or one carries a
`user_message` that quotes internal state, the guarantees the rest of the
codebase relies on stop holding — and nothing else would catch it.
"""

from __future__ import annotations

import inspect
import logging

import pytest

from finvault import errors
from finvault.errors import (
    AccessDeniedError,
    AgentExecutionError,
    ClientError,
    DependencyError,
    FinVaultError,
    InternalError,
    PolicyError,
    ReviewItemNotFoundError,
    ReviewQueueError,
    TokenBudgetExceeded,
)
from finvault.observability import log_level_for


def _all_error_classes() -> list[type[FinVaultError]]:
    return [
        obj
        for _, obj in inspect.getmembers(errors, inspect.isclass)
        if issubclass(obj, FinVaultError) and obj.__module__ == errors.__name__
    ]


def test_every_error_class_is_exported() -> None:
    """An error missing from __all__ is invisible to `from ... import *`
    and, more importantly, to anyone reading the file's public surface."""
    assert sorted(cls.__name__ for cls in _all_error_classes()) == sorted(errors.__all__)


def test_error_codes_are_unique() -> None:
    """Codes are the stable identifier clients match on and log queries
    group by; two classes sharing one makes both unfilterable."""
    codes = [cls.code for cls in _all_error_classes()]
    duplicates = {code for code in codes if codes.count(code) > 1}
    assert duplicates == set(), f"duplicate error codes: {duplicates}"


def test_every_error_has_a_usable_http_status_and_message() -> None:
    for cls in _all_error_classes():
        assert 400 <= cls.http_status <= 599, cls.__name__
        assert cls.user_message.strip(), cls.__name__
        # A user-facing message that ends without punctuation is usually a
        # fragment meant for a log line, not a sentence meant for a person.
        assert cls.user_message.rstrip().endswith((".", "!", "?")), cls.__name__


def test_client_and_policy_errors_are_4xx_and_dependency_errors_are_5xx() -> None:
    """The branch a class sits in has to agree with the status it returns —
    a ClientError that answers 503 would be logged as an expected refusal
    while telling the caller to retry."""
    for cls in _all_error_classes():
        if issubclass(cls, (ClientError, PolicyError)):
            assert 400 <= cls.http_status < 500, cls.__name__
        elif issubclass(cls, (DependencyError, InternalError)):
            assert 500 <= cls.http_status < 600, cls.__name__


def test_expected_refusals_log_at_warning_and_incidents_at_error() -> None:
    assert log_level_for(AccessDeniedError()) == logging.WARNING
    assert log_level_for(errors.ExternalizationBlocked()) == logging.WARNING
    assert log_level_for(AgentExecutionError()) == logging.ERROR
    assert log_level_for(InternalError()) == logging.ERROR
    # An exception from outside this hierarchy is unforeseen by definition.
    assert log_level_for(ValueError("boom")) == logging.ERROR


def test_context_is_carried_into_the_message_and_the_log_fields() -> None:
    exc = AccessDeniedError("Role 'analyst' lacks clearance", context={"role": "analyst"})
    assert "role='analyst'" in str(exc)
    assert exc.log_fields() == {"error_code": "access_denied", "error_type": "AccessDeniedError", "role": "analyst"}


def test_context_never_reaches_the_client_body() -> None:
    """The whole point of the message/user_message split: operator detail
    goes to the log, the caller gets the class's safe wording."""
    exc = AccessDeniedError("Role 'analyst' lacks clearance for restricted", context={"document_id": "doc-42"})
    body = exc.to_dict()
    assert body == {"code": "access_denied", "message": AccessDeniedError.user_message, "retryable": False}
    assert "doc-42" not in body["message"]


def test_user_message_can_be_overridden_per_instance() -> None:
    exc = errors.UnsupportedDocumentError("bad type", user_message="Unsupported file type '.xyz'.")
    assert exc.to_dict()["message"] == "Unsupported file type '.xyz'."
    # The class default is untouched for every other instance.
    assert errors.UnsupportedDocumentError().user_message == "This file type is not supported."


def test_token_budget_exceeded_is_caught_as_an_agent_execution_error() -> None:
    """Load-bearing: every `except AgentExecutionError` fail-closed handler
    in the pipeline relies on this to treat a budget overrun identically."""
    with pytest.raises(AgentExecutionError):
        raise TokenBudgetExceeded("over budget")
    # ...but it is not retryable, unlike its parent — an identical request
    # spends identically.
    assert AgentExecutionError.retryable is True
    assert TokenBudgetExceeded.retryable is False


def test_review_item_not_found_is_a_404_within_the_review_queue_branch() -> None:
    """`except ReviewQueueError` still catches it, while the API reports the
    more accurate status."""
    with pytest.raises(ReviewQueueError):
        raise ReviewItemNotFoundError("no such item")
    assert ReviewItemNotFoundError.http_status == 404
    assert ReviewQueueError.http_status == 400


def test_legacy_import_paths_still_resolve_to_the_same_classes() -> None:
    """Modules that used to define these keep re-exporting them, so existing
    `except` clauses and imports elsewhere are unaffected by the move."""
    from finvault.agents.base import AgentExecutionError as from_base
    from finvault.security.access_control import AccessDeniedError as from_access
    from finvault.security.guardrails import ExternalizationBlocked as from_guardrails
    from finvault.security.review_queue import ReviewQueueError as from_queue

    assert from_base is AgentExecutionError
    assert from_access is AccessDeniedError
    assert from_guardrails is errors.ExternalizationBlocked
    assert from_queue is ReviewQueueError
