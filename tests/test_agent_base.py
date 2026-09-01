from __future__ import annotations

import pytest

from finvault.agents.base import Agent, AgentExecutionError, TokenBudget, TokenBudgetExceeded, ToolDefinition
from tests.fakes import (
    FakeOpenAIClient,
    FakeResponse,
    make_bad_request_error,
    make_connection_error,
    make_rate_limit_error,
)


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
    # _MAX_RATE_LIMIT_RETRIES + 1 failures: a 429 gets its own, more patient
    # budget than a connection error (see base.py) because it means "wait and
    # this will succeed", not "this is broken".
    client = FakeOpenAIClient([make_rate_limit_error() for _ in range(Agent._MAX_RATE_LIMIT_RETRIES + 1)])
    agent = Agent(name="test", system_prompt="You are a test agent.", client=client, sleep=lambda _: None)

    with pytest.raises(AgentExecutionError):
        agent.run("Anything")


def test_rate_limits_do_not_consume_the_connection_error_budget() -> None:
    """The two budgets are independent: a query that hits a per-minute quota
    several times must not arrive at its next genuine infra failure with its
    retries already spent."""
    calls = [make_rate_limit_error() for _ in range(Agent._MAX_RATE_LIMIT_RETRIES)]
    calls += [make_connection_error() for _ in range(Agent._MAX_TRANSIENT_RETRIES)]
    calls.append(FakeResponse.text("recovered"))
    client = FakeOpenAIClient(calls)
    agent = Agent(name="test", system_prompt="You are a test agent.", client=client, sleep=lambda _: None)

    assert agent.run("Anything") == "recovered"
    assert len(client.chat.completions.calls) == len(calls)


def test_rate_limit_backoff_is_patient_enough_for_a_per_minute_quota() -> None:
    """A 0.5s/1.0s backoff cannot outlast a 3-RPM window. Delays must reach
    the tens of seconds, and stay capped so a request can't stall forever."""
    slept: list[float] = []
    client = FakeOpenAIClient(
        [make_rate_limit_error() for _ in range(Agent._MAX_RATE_LIMIT_RETRIES)] + [FakeResponse.text("recovered")]
    )
    agent = Agent(name="test", system_prompt="You are a test agent.", client=client, sleep=slept.append)

    assert agent.run("Anything") == "recovered"
    assert max(slept) >= 20.0
    assert max(slept) <= Agent._MAX_RETRY_DELAY_SECONDS


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
# phrasing mentioned a file format or a specific row). tool_choice on the
# API call is a stronger hint than the prompt — but only a hint: several
# providers ignore it silently (see base.py), so the actual guarantee is
# the synthesized-call fallback covered at the bottom of this section.


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
    # "required", not {"type": "function", ...}: the explicit-function form
    # is ignored outright by several providers this deploys against.
    assert first_call["tool_choice"] == "required"


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

    assert client.chat.completions.calls[0]["tool_choice"] == "required"
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


# --- The guarantee itself. tool_choice is advisory in practice: measured
# live, qwen-2.5-72b ignores every form of it and kimi-k2.5 ignores the
# explicit-function form, both answering in plain text with finish_reason
# "stop" and no error. When that happens the agent must run the required
# tool itself, because the alternative is an answer composed from no
# retrieval at all — which reaches the user with zero citations, and so
# skips ComplianceAgent's citation verification entirely.


def test_forced_tool_runs_anyway_when_the_model_ignores_tool_choice() -> None:
    calls: list[dict] = []

    def handler(input_: dict) -> str:
        calls.append(input_)
        return "Passage: Q1 revenue was $158.4M."

    client = FakeOpenAIClient(
        [
            # The provider ignores tool_choice and asks for clarification.
            FakeResponse.text("Could you please specify which document you mean?"),
            FakeResponse.text("Q1 revenue was $158.4M."),
        ]
    )
    agent = Agent(
        name="test",
        system_prompt="You are a test agent.",
        client=client,
        tools=[
            ToolDefinition(
                name="search_documents",
                description="Search the corpus.",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                handler=handler,
            )
        ],
        require_tool_on_first_turn="search_documents",
    )

    result = agent.run("okay read the document and give the output")

    # The clarifying question must NOT be what the user gets back.
    assert result == "Q1 revenue was $158.4M."
    # The tool ran regardless, seeded with the raw user message.
    assert calls == [{"query": "okay read the document and give the output"}]
    # Its output was fed back into the conversation for the next turn.
    assert "Q1 revenue was $158.4M." in client.chat.completions.calls[1]["messages"][-1]["content"]


def test_forced_tool_fallback_only_applies_to_the_first_turn() -> None:
    """A plain-text answer on a LATER turn is a legitimate final answer, not
    a refusal to search — the fallback must not fire and loop forever."""
    calls: list[dict] = []

    def handler(input_: dict) -> str:
        calls.append(input_)
        return "Passage: some context."

    client = FakeOpenAIClient(
        [
            FakeResponse.tool_call("search_documents", {"query": "revenue"}),
            FakeResponse.text("Final answer."),
        ]
    )
    agent = Agent(
        name="test",
        system_prompt="You are a test agent.",
        client=client,
        tools=[
            ToolDefinition(
                name="search_documents",
                description="Search the corpus.",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                handler=handler,
            )
        ],
        require_tool_on_first_turn="search_documents",
    )

    assert agent.run("What was revenue?") == "Final answer."
    # Called once by the model, never re-run by the fallback.
    assert calls == [{"query": "revenue"}]


def test_sdk_internal_retries_are_disabled_so_one_policy_owns_retrying() -> None:
    """Regression test for a self-inflicted failure observed live. The OpenAI
    SDK retries twice by default, ~1s apart, with no idea why a call failed.
    Left on, each attempt made by complete_with_retries is really three HTTP
    requests: it burns a per-minute quota 3x faster than intended, collides
    with itself on an org capped at concurrent requests (Moonshot returned
    "max organization concurrency: 1" 429s that our own retries caused), and
    hides two of every three requests from llm_requests_total.

    The retry policy here is 429-aware; the SDK's is not. Only one of them
    should be retrying.
    """
    from finvault.agents.base import _client

    assert _client().max_retries == 0


def test_compliance_client_also_disables_sdk_retries() -> None:
    """The same amplification through the reviewer is worse: it is the last
    call in the chain, so it collides with whatever is still settling."""
    from finvault.agents.compliance_agent import ComplianceAgent

    agent = ComplianceAgent()

    assert agent._client.max_retries == 0
