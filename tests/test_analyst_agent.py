from __future__ import annotations

from finvault.agents.analyst_agent import AnalystAgent
from tests.fakes import FakeOpenAIClient, FakeResponse


def test_run_structured_parses_a_submit_answer_call() -> None:
    client = FakeOpenAIClient(
        [
            FakeResponse.tool_call(
                "submit_answer",
                {
                    "answer": "Revenue was $10 million in Q1.",
                    "citations": [{"document": "Q1 Report", "quoted_text": "Total revenue was $10 million"}],
                    "calculations": [],
                },
            )
        ]
    )
    agent = AnalystAgent(client=client)

    result = agent.run_structured("What was Q1 revenue?")

    assert result.answer == "Revenue was $10 million in Q1."
    assert len(result.citations) == 1
    assert result.citations[0].quoted_text == "Total revenue was $10 million"
    assert result.citations[0].document == "Q1 Report"


def test_run_structured_falls_back_to_plain_text_when_model_never_calls_submit_answer() -> None:
    """Some models (esp. free-tier) may ignore the submit_answer instruction
    and just answer directly. This must degrade gracefully — no citations to
    verify, not a crash.
    """
    client = FakeOpenAIClient([FakeResponse.text("Revenue was roughly $10 million, I believe.")])
    agent = AnalystAgent(client=client)

    result = agent.run_structured("What was Q1 revenue?")

    assert result.answer == "Revenue was roughly $10 million, I believe."
    assert result.citations == []


def test_run_returns_raw_json_string_when_submit_answer_is_called() -> None:
    client = FakeOpenAIClient([FakeResponse.tool_call("submit_answer", {"answer": "Fine.", "citations": []})])
    agent = AnalystAgent(client=client)

    raw = agent.run("question")

    assert '"answer": "Fine."' in raw or '"answer":"Fine."' in raw
