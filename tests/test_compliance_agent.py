from __future__ import annotations

from finvault.agents.compliance_agent import ComplianceAgent
from finvault.models import Classification
from tests.fakes import FakeOpenAIClient, FakeResponse


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
