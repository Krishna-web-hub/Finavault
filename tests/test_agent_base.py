from __future__ import annotations

import pytest

from finvault.agents.base import Agent, AgentExecutionError, TokenBudget, TokenBudgetExceeded, ToolDefinition
from tests.fakes import FakeOpenAIClient, FakeResponse, make_bad_request_error, make_rate_limit_error


def _make_search_tool(handler=None):
    return ToolDefinition(
        name="search_documents",
        description="Search the corpus.",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        handler=handler or (lambda _: "some passages"),
    )


def test_agent_returns_final_text_when_no_tool_calls() -> None:
    client = FakeOpenAIClient([FakeResponse.text("The answer is 42.")])
    agent = Agent(name="test", system_prompt="You are a test agent.", client=client)

    assert agent.run("What is the answer?") == "The answer is 42."


def test_agent_raises_typed_error_on_api_failure_instead_of_crashing() -> None:
    client = FakeOpenAIClient([RuntimeError("connection reset")])
    agent = Agent(name="test", system_prompt="You are a test agent.", client=client)

    with pytest.raises(AgentExecutionError):
        agent.run("Anything")


def test_agent_raises_typed_error_on_response_with_no_choices() -> None:
    """Regression test for a real failure hit live: retrieved content
    containing an embedded prompt-injection attempt caused OpenRouter to
    return an HTTP 200 with `choices` missing/null instead of raising an
    exception — `response.choices[0]` then crashed with an unhandled
    TypeError, surfacing as a raw 500 all the way up through the API route.
    This must fail closed as a typed error instead.
    """
    client = FakeOpenAIClient([FakeResponse(choices=[])])
    agent = Agent(name="test", system_prompt="You are a test agent.", client=client)

    with pytest.raises(AgentExecutionError):
        agent.run("Anything")


def test_agent_raises_after_exhausting_empty_response_retries() -> None:
    """Regression test for a real failure mode seen against Claude Opus 5 via
    OpenRouter: a too-tight max_tokens budget can be entirely consumed by
    the model's default reasoning, leaving finish_reason="length" with empty
    content. That must eventually fail loudly, not return "" and let an
    empty answer silently pass through downstream compliance review as
    "clean". The agent retries a bounded number of times first (see the next
    test) — this exhausts all of them with empty responses.
    """
    client = FakeOpenAIClient(
        [
            FakeResponse.text("", finish_reason="length"),
            FakeResponse.text("", finish_reason="length"),
            FakeResponse.text("", finish_reason="length"),
        ]
    )
    agent = Agent(name="test", system_prompt="You are a test agent.", client=client)

    with pytest.raises(AgentExecutionError):
        agent.run("Anything")


def test_prompt_caching_disabled_by_default_sends_plain_system_string() -> None:
    client = FakeOpenAIClient([FakeResponse.text("ok")])
    agent = Agent(name="test", system_prompt="You are a test agent.", client=client)

    agent.run("hello")

    system_message = client.chat.completions.calls[0]["messages"][0]
    assert system_message["content"] == "You are a test agent."


def test_prompt_caching_enabled_marks_system_prompt_as_cacheable() -> None:
    client = FakeOpenAIClient([FakeResponse.text("ok")])
    agent = Agent(name="test", system_prompt="You are a test agent.", client=client, enable_prompt_caching=True)

    agent.run("hello")

    system_message = client.chat.completions.calls[0]["messages"][0]
    assert system_message["content"] == [
        {"type": "text", "text": "You are a test agent.", "cache_control": {"type": "ephemeral"}}
    ]


def test_transient_error_retries_and_succeeds() -> None:
    """Rate limits and connection errors are retryable — the agent should
    retry with backoff and succeed once the upstream recovers, rather than
    failing on the first transient blip.
    """
    client = FakeOpenAIClient([make_rate_limit_error(), make_rate_limit_error(), FakeResponse.text("recovered")])
    agent = Agent(name="test", system_prompt="You are a test agent.", client=client, sleep=lambda _: None)

    assert agent.run("Anything") == "recovered"
    assert len(client.chat.completions.calls) == 3


def test_transient_error_exhausts_retries_and_fails_closed() -> None:
    client = FakeOpenAIClient([make_rate_limit_error(), make_rate_limit_error(), make_rate_limit_error()])
    agent = Agent(name="test", system_prompt="You are a test agent.", client=client, sleep=lambda _: None)

    with pytest.raises(AgentExecutionError):
        agent.run("Anything")


def test_non_retryable_error_fails_immediately_without_retrying() -> None:
    """A bad request (e.g. malformed params) will fail identically on every
    retry — retrying just wastes time before the same failure. Only one
    call should ever be made.
    """
    client = FakeOpenAIClient([make_bad_request_error(), FakeResponse.text("should never be reached")])
    agent = Agent(name="test", system_prompt="You are a test agent.", client=client, sleep=lambda _: None)

    with pytest.raises(AgentExecutionError):
        agent.run("Anything")
    assert len(client.chat.completions.calls) == 1


def test_token_budget_charge_under_limit_does_not_raise() -> None:
    budget = TokenBudget(limit=100)
    budget.charge(60)
    budget.charge(30)
    assert budget.spent == 90


def test_token_budget_charge_over_limit_raises() -> None:
    budget = TokenBudget(limit=100)
    budget.charge(60)
    with pytest.raises(TokenBudgetExceeded):
        budget.charge(60)


def test_agent_run_charges_the_shared_budget() -> None:
    client = FakeOpenAIClient([FakeResponse.text("ok", total_tokens=42)])
    agent = Agent(name="test", system_prompt="You are a test agent.", client=client)
    budget = TokenBudget(limit=1000)

    agent.run("Anything", budget=budget)

    assert budget.spent == 42


def test_agent_run_fails_closed_when_budget_is_exhausted_mid_request() -> None:
    """TokenBudgetExceeded is a subclass of AgentExecutionError, so callers
    that only catch AgentExecutionError (e.g. Orchestrator.handle()) still
    fail closed correctly without needing to know about budgets.
    """
    client = FakeOpenAIClient([FakeResponse.text("ok", total_tokens=80)])
    agent = Agent(name="test", system_prompt="You are a test agent.", client=client)
    budget = TokenBudget(limit=50)

    with pytest.raises(TokenBudgetExceeded):
        agent.run("Anything", budget=budget)
    assert isinstance(TokenBudgetExceeded("x"), AgentExecutionError)


def test_agent_recovers_from_a_single_empty_response_via_nudge_retry() -> None:
    """Real failure mode seen against a free-tier model (nvidia/nemotron-3-super
    via OpenRouter, in the openai/gpt-oss-20b:free case): it occasionally
    returns a genuinely empty turn (no text, no tool call) after a tool
    result, instead of continuing. A nudge message asking it to continue
    resolves this in practice — this confirms the retry path, not just the
    eventual-failure path.
    """
    client = FakeOpenAIClient(
        [
            FakeResponse.text("", finish_reason="stop"),
            FakeResponse.text("Here is the answer.", finish_reason="stop"),
        ]
    )
    agent = Agent(name="test", system_prompt="You are a test agent.", client=client)

    assert agent.run("Anything") == "Here is the answer."
    # Confirm a nudge message was actually sent, not that it happened to
    # succeed for an unrelated reason.
    second_call_messages = client.chat.completions.calls[1]["messages"]
    assert "continue" in second_call_messages[-1]["content"].lower()


# --- require_tool_on_first_turn: a prompt instruction alone isn't reliable
# enough to make a free-tier model actually call a tool — see
# orchestrator.py's real motivation for this (observed live: the model
# skipped search_documents and answered directly whenever a question's
# phrasing mentioned a file format or a specific row). tool_choice forcing
# the API call itself is deterministic where the prompt wasn't.


def test_require_tool_on_first_turn_forces_tool_choice_on_the_first_request() -> None:
    client = FakeOpenAIClient(
        [
            FakeResponse.tool_call("search_documents", {"query": "revenue"}),
            FakeResponse.text("Here is the answer, grounded in what search_documents found."),
        ]
    )
    agent = Agent(
        name="test",
        system_prompt="You are a test agent.",
        client=client,
        tools=[_make_search_tool()],
        require_tool_on_first_turn="search_documents",
    )

    result = agent.run("Some ambiguous question")

    assert result == "Here is the answer, grounded in what search_documents found."
    first_call = client.chat.completions.calls[0]
    assert first_call["tool_choice"] == {"type": "function", "function": {"name": "search_documents"}}


def test_require_tool_on_first_turn_does_not_force_tool_choice_on_later_turns() -> None:
    client = FakeOpenAIClient(
        [
            FakeResponse.tool_call("search_documents", {"query": "revenue"}),
            FakeResponse.tool_call("search_documents", {"query": "revenue, reformulated"}),
            FakeResponse.text("Final answer after two searches."),
        ]
    )
    agent = Agent(
        name="test",
        system_prompt="You are a test agent.",
        client=client,
        tools=[_make_search_tool()],
        require_tool_on_first_turn="search_documents",
    )

    agent.run("Some ambiguous question")

    assert client.chat.completions.calls[0]["tool_choice"] == {
        "type": "function",
        "function": {"name": "search_documents"},
    }
    # Second and third calls (the model's own free choice each time) must
    # not have a forced tool_choice — only the very first call does.
    assert "tool_choice" not in client.chat.completions.calls[1]
    assert "tool_choice" not in client.chat.completions.calls[2]


def test_agent_without_require_tool_on_first_turn_never_sends_tool_choice() -> None:
    """Default behavior (every other agent in the codebase) is unaffected —
    tool_choice is only ever sent when this opt-in param is set."""
    client = FakeOpenAIClient([FakeResponse.text("Direct answer, no tools needed.")])
    agent = Agent(name="test", system_prompt="You are a test agent.", client=client, tools=[_make_search_tool()])

    agent.run("Anything")

    assert "tool_choice" not in client.chat.completions.calls[0]
