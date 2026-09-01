"""Generic manual tool-use loop over an OpenAI-compatible chat completions
API, routed through OpenRouter (https://openrouter.ai) rather than calling
Anthropic directly. OpenRouter exposes the OpenAI chat-completions wire
format for every model it routes to (including Claude), so the `openai`
Python SDK talks to it by pointing `base_url` at OpenRouter's endpoint.

Deliberately holds no finance-specific logic — this is the reusable
scaffolding intended to outlive FinVault's RAG use case (see the plan's
"Roadmap beyond this build"). Domain behavior belongs in the system prompt
and tools a caller passes in, never here.

Prompt caching (settings.finvault_enable_prompt_caching, off by default) is
best-effort: it marks the system prompt with OpenRouter's documented
Anthropic-style `cache_control` passthrough, since every agent here sends
the same static system prompt on every call. This is NOT the native
Anthropic Messages API — there is no guarantee OpenRouter's proxy honors
the marker for whatever `finvault_model` is currently configured, and this
codebase does not surface cache_creation/cache_read token counts to verify
a hit. Treat it as a hint that may do nothing on some models/providers, not
a guaranteed cost/latency win.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from openai import APIConnectionError, InternalServerError, OpenAI, RateLimitError

from finvault.config import settings
from finvault.errors import AgentExecutionError, TokenBudgetExceeded, UpstreamProtocolError
from finvault.metrics import (
    agent_duration_seconds,
    agent_runs_total,
    forced_tool_synthesized_total,
    llm_requests_total,
    llm_retries_total,
    observe_duration,
    record_tokens,
    token_budget_exhausted_total,
)
from finvault.observability import extra_fields, get_logger, log_exception

logger = get_logger(__name__)

ToolHandler = Callable[[dict[str, Any]], str]

# The three exception types this module raises all live in finvault/errors.py
# now — one hierarchy for the whole system — and are re-exported here so
# `from finvault.agents.base import AgentExecutionError` still works. Read
# their docstrings there for the fail-closed contract every caller owes them:
#
#   AgentExecutionError   the LLM call failed (auth, billing, network, 5xx)
#   TokenBudgetExceeded   the per-request token ceiling was hit mid-chain
#   UpstreamProtocolError HTTP 200 with a body the loop cannot proceed from


@dataclass
class TokenBudget:
    """Shared, mutable token ceiling for one end-to-end request that may
    span several Agent instances (Orchestrator -> Retriever -> Analyst).
    Callers create one per request and pass the same instance through every
    `Agent.run()` call in that chain (see orchestrator.py) so usage
    accumulates across all of them, not per-agent.
    """

    limit: int
    spent: int = 0

    def charge(self, tokens: int) -> None:
        self.spent += tokens
        if self.spent > self.limit:
            raise TokenBudgetExceeded(
                f"Token budget exceeded: {self.spent}/{self.limit} tokens spent this request",
                context={"spent": self.spent, "limit": self.limit},
            )


# Retryable: transient infra conditions where a retry is likely to succeed.
# Deliberately excludes BadRequestError, AuthenticationError, PermissionDeniedError,
# NotFoundError — those indicate a genuinely broken request/config, and retrying
# them just burns time before failing the same way. APITimeoutError is a subclass
# of APIConnectionError so it's covered without listing it separately.
_RETRYABLE_EXCEPTIONS = (RateLimitError, APIConnectionError, InternalServerError)


# The SDK ships max_retries=2 and retries roughly 1s apart, blind to WHY the
# call failed. Left on, every deliberate attempt made by complete_with_retries
# is really three rapid-fire HTTP requests, which:
#   - burns a per-minute quota 3x faster than the policy below intends,
#   - collides with itself on an org capped at concurrent requests (observed
#     on Moonshot: "max organization concurrency: 1" 429s that our own
#     retries were generating), and
#   - undercounts real upstream traffic in llm_requests_total, because two
#     of every three requests are invisible to this code.
# One retry policy, in one place, that can tell a 429 from a 5xx.
SDK_INTERNAL_RETRIES = 0


def _client() -> OpenAI:
    api_key = settings.effective_api_key or "unconfigured"
    return OpenAI(
        base_url=settings.effective_base_url,
        api_key=api_key,
        timeout=settings.finvault_llm_timeout_seconds,
        max_retries=SDK_INTERNAL_RETRIES,
    )


# --- Retry policy, shared by every LLM call in the system -------------------
#
# At module level rather than on Agent, because ComplianceAgent makes its own
# single-shot review call without going through the tool loop. While this
# policy lived only inside Agent, that call was the one LLM request in the
# system with no retry at all — and it is the request that can least afford
# it: compliance runs LAST in a 5-8 call chain, so it is the most likely to
# meet a per-minute cap, and its failure mode is the harshest one here.
# A 429 there does not fail the request, it BLOCKS a correct answer for
# "manual review", which reads to an operator as a compliance judgment about
# the content rather than an infrastructure problem. Observed live on a
# 3 RPM Moonshot org: retrieval and analysis both succeeded, and the answer
# was blocked anyway.
#
# One policy in one place is the point — the duplication was the bug.

# Transient infra failures (connection, 5xx). A blip; a couple of quick
# retries either clears it or it is real.
MAX_TRANSIENT_RETRIES = 2
RETRY_BASE_DELAY_SECONDS = 0.5

# Rate limits get their own, far more patient budget. A 429 is not a failure
# signal like a 5xx — it is the upstream stating that the same request WILL
# succeed after a wait, so the only wrong move is giving up early. A budget
# sized for a blip (2 retries, 0.5s then 1.0s) cannot survive a per-minute
# quota: a small Moonshot org is capped at 3 RPM while one FinVault query
# makes ~5-8 sequential LLM calls, so mid-question 429s are routine rather
# than exceptional. Capped per-delay so a pathological upstream cannot stall
# a request indefinitely.
MAX_RATE_LIMIT_RETRIES = 5
RATE_LIMIT_BASE_DELAY_SECONDS = 4.0
MAX_RETRY_DELAY_SECONDS = 30.0


def complete_with_retries(
    *,
    client: OpenAI,
    sleep: Callable[[float], None],
    agent: str,
    model: str,
    **create_kwargs: Any,
) -> Any:
    """Issue one chat-completion request, retrying on transient upstream
    conditions. Raises AgentExecutionError once a budget is exhausted, so
    every caller fails closed on the same contract.
    """
    client_key = getattr(client, "api_key", None)
    if client_key == "unconfigured":
        raise AgentExecutionError(
            f"Agent '{agent}' cannot execute: LLM API key is unconfigured. "
            "Please provide an API key in the Streamlit sidebar or configure OPENROUTER_API_KEY / OPENAI_API_KEY."
        )

    # Two independent budgets: a rate limit does not consume the allowance
    # reserved for genuine infra failures, and vice versa.
    transient_attempts = 0
    rate_limit_attempts = 0
    while True:
        try:
            response = client.chat.completions.create(model=model, **create_kwargs)
            llm_requests_total.labels(agent=agent, model=model, outcome="success").inc()
            return response
        except _RETRYABLE_EXCEPTIONS as exc:
            llm_requests_total.labels(agent=agent, model=model, outcome="retryable_error").inc()
            llm_retries_total.labels(agent=agent, upstream_error=type(exc).__name__).inc()
            is_rate_limit = isinstance(exc, RateLimitError)
            if is_rate_limit:
                rate_limit_attempts += 1
                attempts, budget_limit = rate_limit_attempts, MAX_RATE_LIMIT_RETRIES
                delay = min(
                    RATE_LIMIT_BASE_DELAY_SECONDS * (2 ** (rate_limit_attempts - 1)),
                    MAX_RETRY_DELAY_SECONDS,
                )
            else:
                transient_attempts += 1
                attempts, budget_limit = transient_attempts, MAX_TRANSIENT_RETRIES
                delay = min(
                    RETRY_BASE_DELAY_SECONDS * (2 ** (transient_attempts - 1)),
                    MAX_RETRY_DELAY_SECONDS,
                )
            if attempts > budget_limit:
                raise AgentExecutionError(
                    f"Agent '{agent}' LLM request failed after {attempts} attempt(s) of a "
                    f"retryable error ({type(exc).__name__}): {exc}",
                    context={
                        "agent": agent,
                        "model": model,
                        "attempts": attempts,
                        "upstream_error": type(exc).__name__,
                        "rate_limited": is_rate_limit,
                    },
                ) from exc
            # WARNING, not ERROR: a retried transient failure that later
            # succeeds is not an incident. It is logged anyway because a
            # rising rate of these is the earliest warning of provider
            # trouble, and it is otherwise completely invisible.
            logger.warning(
                "llm_request_retrying",
                extra={
                    "fields": {
                        "agent": agent,
                        "model": model,
                        "attempt": attempts,
                        "delay_seconds": delay,
                        "upstream_error": type(exc).__name__,
                        "rate_limited": is_rate_limit,
                        "error_message": str(exc),
                    }
                },
            )
            sleep(delay)
        except Exception as exc:
            llm_requests_total.labels(agent=agent, model=model, outcome="error").inc()
            raise AgentExecutionError(
                f"Agent '{agent}' LLM request failed: {exc}",
                context={"agent": agent, "model": model, "upstream_error": type(exc).__name__},
            ) from exc
    # `while True` only exits via return or raise above; no fallthrough.


@dataclass
class ToolDefinition:
    """A tool this agent can call: the API-facing schema plus the local
    handler that executes it."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler

    def to_api_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


class Agent:
    """A named agent identity running a manual tool-use loop.

    A manual loop (rather than a higher-level framework) is used
    deliberately: it keeps every request/response, and every tool
    invocation, inspectable and audit-loggable by the caller — required for
    the compliance/audit posture of this system.
    """

    def __init__(
        self,
        *,
        name: str,
        system_prompt: str,
        tools: list[ToolDefinition] | None = None,
        model: str | None = None,
        client: OpenAI | None = None,
        max_iterations: int = 8,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
        enable_prompt_caching: bool | None = None,
        terminal_tool: str | None = None,
        final_tool: str | None = None,
        require_tool_on_first_turn: str | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.name = name
        self._system_prompt = system_prompt
        self._tools = {t.name: t for t in (tools or [])}
        self._model = model or settings.finvault_model
        self._client = client or _client()
        self._max_iterations = max_iterations
        self._max_tokens = max_tokens or settings.finvault_max_tokens
        # Injectable so tests can skip real backoff delays; production
        # callers never pass this.
        self._sleep = sleep or time.sleep
        # Two distinct short-circuit mechanisms, for two distinct needs:
        #  - terminal_tool: the MODEL's arguments to this call are themselves
        #    the final structured answer. The handler is never invoked; used
        #    when a caller wants the model to emit a JSON-schema-shaped
        #    result directly (see AnalystAgent's `submit_answer`).
        #  - final_tool: the tool's HANDLER runs normally, but its return
        #    value ends the loop immediately instead of being fed back to
        #    the model for further narration (see Orchestrator's `analyze`).
        # At most one is typically set per agent; both can coexist since they
        # trigger on different tool names.
        self._terminal_tool = terminal_tool
        self._final_tool = final_tool
        # A prompt instruction to "always call X first" is only ever a
        # request — a smaller/free-tier model will sometimes just respond
        # in plain text instead, reasoning (incorrectly) that its tools
        # "aren't suitable" for a question's surface phrasing (observed
        # live: a question mentioning "CSV" or a specific row number made
        # the Orchestrator skip search_documents entirely and answer from
        # nothing). tool_choice forcing a specific function on the request
        # itself is deterministic where the prompt alone isn't — used only
        # on the very first turn, so every later turn still lets the model
        # choose freely (reformulate, call a different tool, or finish).
        self._require_tool_on_first_turn = require_tool_on_first_turn
        # Opaque passthrough for provider-specific tuning (e.g. OpenRouter's
        # unified `reasoning` field to cap a reasoning model's thinking
        # budget) — kept generic and optional so this module makes no
        # assumptions about which provider/model is behind `finvault_model`.
        self._extra_body = extra_body
        self._enable_prompt_caching = (
            enable_prompt_caching if enable_prompt_caching is not None else settings.finvault_enable_prompt_caching
        )

    # Bounded retries for a model that returns a genuinely empty turn (no
    # text, no tool call) — observed against smaller/free-tier models that
    # occasionally stop mid-task without producing output. A nudge back into
    # the conversation resolves this more often than not; only fail closed
    # after repeated empty turns.
    _MAX_EMPTY_RESPONSE_RETRIES = 2

    # The retry policy itself lives at module level (see
    # complete_with_retries) so ComplianceAgent's single-shot review call
    # gets the identical behavior. Re-exported as class attributes because
    # they read as tuning knobs of the agent loop at every call site.
    _MAX_TRANSIENT_RETRIES = MAX_TRANSIENT_RETRIES
    _RETRY_BASE_DELAY_SECONDS = RETRY_BASE_DELAY_SECONDS
    _MAX_RATE_LIMIT_RETRIES = MAX_RATE_LIMIT_RETRIES
    _RATE_LIMIT_BASE_DELAY_SECONDS = RATE_LIMIT_BASE_DELAY_SECONDS
    _MAX_RETRY_DELAY_SECONDS = MAX_RETRY_DELAY_SECONDS

    def _create_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
        tool_choice: dict[str, Any] | str | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "max_tokens": self._max_tokens,
            "messages": messages,
            "tools": tool_schemas or None,
            "extra_body": self._extra_body,
        }
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        return complete_with_retries(
            client=self._client,
            sleep=self._sleep,
            agent=self.name,
            model=self._model,
            **kwargs,
        )

    def run(
        self,
        user_message: str,
        *,
        history: list[dict[str, str]] | None = None,
        budget: TokenBudget | None = None,
    ) -> str:
        """Runs the tool-use loop to a final answer.

        Wrapped in metrics here rather than inside `_run()` so one `run()`
        is one observation, however many LLM round-trips and tool calls the
        loop takes — "how long does the Analyst take?" is a question about
        the agent, not about its individual requests.
        """
        with observe_duration(agent_duration_seconds, agent=self.name):
            try:
                answer = self._run(user_message, history=history, budget=budget)
            except Exception:
                agent_runs_total.labels(agent=self.name, outcome="error").inc()
                raise
        agent_runs_total.labels(agent=self.name, outcome="success").inc()
        return answer

    def _run(
        self,
        user_message: str,
        *,
        history: list[dict[str, str]] | None = None,
        budget: TokenBudget | None = None,
    ) -> str:
        system_content: str | list[dict[str, Any]] = self._system_prompt
        if self._enable_prompt_caching:
            # Best-effort only — see the module docstring. Static system
            # prompt, sent unchanged on every call, is exactly the shape
            # this passthrough marker is meant for.
            system_content = [{"type": "text", "text": self._system_prompt, "cache_control": {"type": "ephemeral"}}]

        messages: list[dict[str, Any]] = [{"role": "system", "content": system_content}]
        # Prior turns (plain user/assistant text, not tool-call mechanics) —
        # see agents/session.py. Empty/None for every agent except the
        # Orchestrator's top-level agent, which is the only one a caller
        # currently threads conversation history through.
        messages.extend(history or [])
        messages.append({"role": "user", "content": user_message})
        tool_schemas = [t.to_api_schema() for t in self._tools.values()]
        empty_response_retries = 0

        for iteration in range(self._max_iterations):
            tool_choice = None
            forcing_this_turn = iteration == 0 and self._require_tool_on_first_turn is not None
            if forcing_this_turn:
                # "required" (call *some* tool), not the more precise
                # {"type": "function", "function": {"name": ...}} form.
                # Measured against the models actually served here: several
                # providers silently ignore the explicit-function form and
                # answer in plain text with finish_reason "stop", while
                # honoring "required" — qwen-2.5-72b ignores both, kimi-k2.5
                # ignores the explicit form and honors "required". Neither
                # errors, so there is no capability check that would catch
                # it; OpenRouter advertises `tool_choice` support for models
                # whose upstream provider does not implement it.
                # "required" lets the model pick the wrong tool, so it is a
                # hint, not the guarantee — the guarantee is the fallback
                # below, which does not depend on the provider at all.
                tool_choice = "required"
            response = self._create_completion(messages=messages, tool_schemas=tool_schemas, tool_choice=tool_choice)

            # Not every upstream failure raises: a moderation block or
            # routing error can come back as an HTTP 200 with a malformed
            # body (`choices` missing/null) that the SDK doesn't reject.
            # Observed in practice when retrieved content contained an
            # embedded prompt-injection attempt. Treat it the same as any
            # other request failure rather than crashing on the index below.
            if not response or not response.choices:
                raise UpstreamProtocolError(
                    f"Agent '{self.name}' received a response with no choices "
                    "(possible upstream moderation block or routing error).",
                    context={"agent": self.name, "model": self._model, "iteration": iteration},
                )

            if response.usage is not None:
                # Metered unconditionally, unlike the budget charge below:
                # cost is incurred whether or not this call chain happens to
                # carry a TokenBudget, and a spend dashboard that silently
                # omitted budget-less calls would under-report real spend.
                record_tokens(
                    agent=self.name,
                    model=self._model,
                    input_tokens=getattr(response.usage, "prompt_tokens", 0) or 0,
                    output_tokens=getattr(response.usage, "completion_tokens", 0) or 0,
                )

            if budget is not None and response.usage is not None:
                # Charged as soon as a response arrives, before any tool
                # dispatch — an overspend here must stop this call chain
                # immediately, not after another round of tool execution.
                try:
                    budget.charge(response.usage.total_tokens)
                except TokenBudgetExceeded:
                    token_budget_exhausted_total.labels(agent=self.name).inc()
                    raise

            choice = response.choices[0]
            message = choice.message
            messages.append(message.model_dump(exclude_none=True))

            if choice.finish_reason != "tool_calls" or not message.tool_calls:
                if forcing_this_turn:
                    # The provider ignored the forcing hint and answered
                    # directly. Left alone, this is the failure that matters
                    # most: the Orchestrator returns an answer composed from
                    # nothing, with zero citations — and because
                    # ComplianceAgent only verifies citations when some
                    # exist, an ungrounded answer is reviewed as "clean".
                    # So run the required tool ourselves rather than trusting
                    # the model to. Its output is appended as a user turn
                    # (not a synthetic assistant tool_call, which would need
                    # a fabricated tool_call_id that some providers reject),
                    # and the loop continues normally from there.
                    forced_name = self._require_tool_on_first_turn
                    assert forced_name is not None  # implied by forcing_this_turn
                    logger.warning(
                        "forced_tool_ignored_by_model",
                        extra=extra_fields(
                            agent=self.name,
                            model=self._model,
                            tool=forced_name,
                            finish_reason=choice.finish_reason,
                        ),
                    )
                    forced_tool_synthesized_total.labels(agent=self.name, tool=forced_name).inc()
                    forced_tool = self._tools[forced_name]
                    # The model produced no arguments, so fall back to the
                    # raw user message as the single required argument. A
                    # weaker search query than one the model would have
                    # written, but a grounded answer from a blunt query
                    # beats a fluent one from no retrieval at all.
                    required_args = forced_tool.input_schema.get("required") or []
                    forced_args = {required_args[0]: user_message} if required_args else {}
                    try:
                        forced_result = forced_tool.handler(forced_args)
                    except AgentExecutionError:
                        # Same fail-closed reasoning as the main dispatch
                        # path below: a sub-agent failure is a request
                        # failure, never a benign tool-result string.
                        raise
                    except Exception as exc:  # noqa: BLE001 — surface to the model, not the caller
                        log_exception(logger, exc, "forced_tool_execution_failed", agent=self.name, tool=forced_name)
                        forced_result = f"Error executing tool '{forced_name}': {exc}"
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"[System: {forced_name} was run automatically for you with "
                                f"{json.dumps(forced_args)}. You must ground your answer in these "
                                f"results rather than asking for clarification.]\n\n{forced_result}"
                            ),
                        }
                    )
                    continue

                if message.content and message.content.strip():
                    return message.content

                if empty_response_retries < self._MAX_EMPTY_RESPONSE_RETRIES:
                    # Some models (typically smaller/free-tier ones) return a
                    # genuinely empty turn — no text, no tool call — after
                    # seeing a tool result, instead of continuing the task or
                    # answering. A nudge back into the conversation resolves
                    # this more often than it doesn't.
                    empty_response_retries += 1
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "You stopped without calling a tool or providing an answer. "
                                "Either call one of your available tools to continue, or give "
                                "your final answer now."
                            ),
                        }
                    )
                    continue

                # Claude's thinking is on by default and shares the
                # max_tokens budget with the visible response (see the
                # claude-api skill notes on Opus 5) — a too-tight budget can
                # exhaust itself on reasoning and leave nothing for text,
                # which surfaces here as finish_reason "length" with empty
                # content. That is a truncation failure, not "the model had
                # nothing to say" — fail closed rather than returning "" and
                # letting an empty answer silently sail through compliance
                # review as trivially clean.
                raise UpstreamProtocolError(
                    f"Agent '{self.name}' produced no text after {empty_response_retries} nudge(s) "
                    f"(finish_reason={choice.finish_reason!r}); likely max_tokens too low for this "
                    "model's reasoning overhead, or the model is unreliable at this task — raise "
                    "FINVAULT_MAX_TOKENS or switch FINVAULT_MODEL.",
                    context={
                        "agent": self.name,
                        "model": self._model,
                        "finish_reason": choice.finish_reason,
                        "nudges": empty_response_retries,
                        "max_tokens": self._max_tokens,
                    },
                )

            final_tool_result: str | None = None
            for tool_call in message.tool_calls:
                try:
                    args = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError as exc:
                    # Handed back to the model to correct, so the request
                    # continues — but recorded, because a model that keeps
                    # emitting malformed arguments burns the iteration budget
                    # and eventually fails with a completely unrelated
                    # "exceeded max_iterations" message.
                    logger.info(
                        "tool_arguments_unparseable",
                        extra=extra_fields(agent=self.name, tool=tool_call.function.name, error_message=str(exc)),
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Error: invalid JSON arguments: {exc}",
                        }
                    )
                    continue

                if self._terminal_tool is not None and tool_call.function.name == self._terminal_tool:
                    # The model's arguments ARE the final answer — return
                    # immediately, skip the handler entirely, don't append a
                    # tool-result message or continue the loop. If other tool
                    # calls arrived in the same turn, they're intentionally
                    # dropped: a terminal call means the model is done.
                    return json.dumps(args)

                tool = self._tools.get(tool_call.function.name)
                if tool is None:
                    result = f"Error: unknown tool '{tool_call.function.name}'"
                else:
                    try:
                        result = tool.handler(args)
                    except AgentExecutionError:
                        # A fail-closed condition surfaced by a tool that
                        # itself wraps another agent (e.g. RetrieverAgent's
                        # own LLM call failing, or a shared TokenBudget
                        # exhausted mid-chain) must propagate as a request
                        # failure — not get swallowed into a benign-looking
                        # tool-result error string the model might work around.
                        raise
                    except Exception as exc:  # noqa: BLE001 — surface to the model, not the caller
                        # The model gets this as a tool result and will often
                        # recover, so it is not a request failure — but without
                        # this log the failure is invisible to operators, since
                        # the error string never leaves the message list.
                        log_exception(
                            logger,
                            exc,
                            "tool_execution_failed",
                            agent=self.name,
                            tool=tool_call.function.name,
                        )
                        result = f"Error executing tool '{tool_call.function.name}': {exc}"
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

                if self._final_tool is not None and tool_call.function.name == self._final_tool:
                    final_tool_result = result

            if final_tool_result is not None:
                return final_tool_result

        raise AgentExecutionError(
            f"Agent '{self.name}' exceeded max_iterations ({self._max_iterations}) without finishing",
            context={"agent": self.name, "model": self._model, "max_iterations": self._max_iterations},
        )
