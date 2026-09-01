from __future__ import annotations

from finvault.agents.base import MAX_RATE_LIMIT_RETRIES
from finvault.agents.compliance_agent import ComplianceAgent
from finvault.config import settings
from finvault.models import Classification
from tests.fakes import FakeOpenAIClient, FakeResponse, make_rate_limit_error


def test_approve_when_response_mentions_block_in_a_benign_sentence() -> None:
    """Regression test: a naive `"BLOCK" in text` substring check false-
    positives on a legitimate approval that happens to contain the word
    (e.g. "this does not need to be blocked"). The fix anchors the parse to
    an explicit verdict word on the first line.
    """
    client = FakeOpenAIClient([FakeResponse.text("APPROVE\nThis answer does not need to be blocked or redacted.")])
    agent = ComplianceAgent(client=client)

    verdict = agent.review_output(
        question="What was revenue?", draft_answer="Revenue was $10M.", max_classification=Classification.INTERNAL
    )

    assert verdict.allowed is True


def test_blocks_on_explicit_block_verdict() -> None:
    client = FakeOpenAIClient([FakeResponse.text("BLOCK\nThis discloses client-identifying information.")])
    agent = ComplianceAgent(client=client)

    verdict = agent.review_output(question="q", draft_answer="a", max_classification=Classification.INTERNAL)

    assert verdict.allowed is False
    assert verdict.reason is not None


def test_fails_closed_on_api_exception() -> None:
    client = FakeOpenAIClient([RuntimeError("502 bad gateway")])
    agent = ComplianceAgent(client=client)

    verdict = agent.review_output(question="q", draft_answer="a", max_classification=Classification.INTERNAL)

    # An unreachable reviewer must not be silently treated as approval.
    assert verdict.allowed is False


def test_fails_closed_on_malformed_response() -> None:
    client = FakeOpenAIClient([FakeResponse.text("I think this is probably fine overall.")])
    agent = ComplianceAgent(client=client)

    verdict = agent.review_output(question="q", draft_answer="a", max_classification=Classification.INTERNAL)

    assert verdict.allowed is False


def test_fails_closed_on_response_with_no_choices() -> None:
    """Regression test for a real failure hit live: OpenRouter can return an
    HTTP 200 with `choices` missing/null (observed when the reviewed content
    contained an embedded prompt-injection attempt) instead of raising. That
    must fail closed like any other reviewer failure, not crash.
    """
    client = FakeOpenAIClient([FakeResponse(choices=[])])
    agent = ComplianceAgent(client=client)

    verdict = agent.review_output(question="q", draft_answer="a", max_classification=Classification.INTERNAL)

    assert verdict.allowed is False


def test_externalization_policy_blocks_before_any_llm_call() -> None:
    # No responses scripted at all — if the code tried to call the LLM here,
    # FakeCompletions would raise AssertionError("ran out of responses").
    client = FakeOpenAIClient([])
    agent = ComplianceAgent(client=client)

    verdict = agent.review_output(question="q", draft_answer="a", max_classification=Classification.RESTRICTED)

    assert verdict.allowed is False
    assert "restricted" in (verdict.reason or "")


# --- Rate limits vs. the reviewer -----------------------------------------
# Compliance runs LAST in a 5-8 call chain, so on a per-minute quota it is
# the call most likely to be limited — and blocking is its failure mode, so
# an unretried 429 turns an infrastructure hiccup into what looks like a
# compliance judgment about the content. Observed live against a 3 RPM
# Moonshot org: retrieval and analysis both succeeded and the correct answer
# was blocked for "manual review".


def test_reviewer_retries_a_rate_limit_instead_of_blocking_the_answer() -> None:
    client = FakeOpenAIClient(
        [make_rate_limit_error(), make_rate_limit_error(), FakeResponse.text("APPROVE\nNothing of concern.")]
    )
    agent = ComplianceAgent(client=client, sleep=lambda _: None)

    verdict = agent.review_output(
        question="What was revenue?", draft_answer="Revenue was $10M.", max_classification=Classification.INTERNAL
    )

    assert verdict.allowed is True
    assert len(client.chat.completions.calls) == 3


def test_reviewer_still_fails_closed_once_the_rate_limit_budget_is_exhausted() -> None:
    """Retries change how often the fail-closed path is reached, never what
    happens when it is. A reviewer that never answers must still block."""
    client = FakeOpenAIClient([make_rate_limit_error() for _ in range(MAX_RATE_LIMIT_RETRIES + 1)])
    agent = ComplianceAgent(client=client, sleep=lambda _: None)

    verdict = agent.review_output(question="q", draft_answer="a", max_classification=Classification.INTERNAL)

    assert verdict.allowed is False


def test_reviewer_is_given_room_for_a_reasoning_model_to_reach_its_verdict() -> None:
    """Regression test for a live failure on kimi-k3: the review call was
    capped at 200 tokens, which a thinking model spends on reasoning before
    emitting the verdict word. It came back with finish_reason "length" and
    empty content, so the reviewer failed closed and blocked a correct,
    fully-cited answer — an infrastructure limit surfacing to the operator
    as a compliance judgment about the content.

    The ceiling is nearly free to raise: it caps generation, it does not
    reserve spend, so a model that answers in one word still emits one word.
    """
    client = FakeOpenAIClient([FakeResponse.text("APPROVE\nNothing of concern.")])
    agent = ComplianceAgent(client=client)

    agent.review_output(question="q", draft_answer="a", max_classification=Classification.INTERNAL)

    sent = client.chat.completions.calls[0]["max_tokens"]
    assert sent == settings.finvault_compliance_review_max_tokens
    # A one-word verdict needs ~5 tokens; the budget must clear a reasoning
    # model's preamble by an order of magnitude, not by a little.
    assert sent >= 1024
