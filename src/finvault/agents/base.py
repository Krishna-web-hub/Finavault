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


def _client() -> OpenAI:
    return OpenAI(base_url=settings.effective_base_url, api_key=settings.effective_api_key, timeout=60.0)


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

    # Bounded retries specifically for transient infra failures (rate limit,
    # connection, 5xx) — a different concern from the empty-response nudge
    # above, which is about the model's own output, not the request itself.
    _MAX_TRANSIENT_RETRIES = 2
    _RETRY_BASE_DELAY_SECONDS = 0.5

    def _create_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
        tool_choice: dict[str, Any] | None = None,
    ) -> Any:
        delay = self._RETRY_BASE_DELAY_SECONDS
        for attempt in range(self._MAX_TRANSIENT_RETRIES + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": self._model,
                    "max_tokens": self._max_tokens,
                    "messages": messages,
                    "tools": tool_schemas or None,
                    "extra_body": self._extra_body,
                }
                if tool_choice is not None:
                    kwargs["tool_choice"] = tool_choice
                response = self._client.chat.completions.create(**kwargs)
                llm_requests_total.labels(agent=self.name, model=self._model, outcome="success").inc()
                return response
            except _RETRYABLE_EXCEPTIONS as exc:
                llm_requests_total.labels(agent=self.name, model=self._model, outcome="retryable_error").inc()
                llm_retries_total.labels(agent=self.name, upstream_error=type(exc).__name__).inc()
                if attempt >= self._MAX_TRANSIENT_RETRIES:
                    raise AgentExecutionError(
                        f"Agent '{self.name}' LLM request failed after {attempt + 1} attempt(s) of a "
                        f"retryable error ({type(exc).__name__}): {exc}",
                        context={
                            "agent": self.name,
                            "model": self._model,
                            "attempts": attempt + 1,
                            "upstream_error": type(exc).__name__,
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
                            "agent": self.name,
                            "model": self._model,
                            "attempt": attempt + 1,
                            "delay_seconds": delay,
                            "upstream_error": type(exc).__name__,
                            "error_message": str(exc),
                        }
                    },
                )
                self._sleep(delay)
                delay *= 2
            except Exception as exc:
                llm_requests_total.labels(agent=self.name, model=self._model, outcome="error").inc()
                raise AgentExecutionError(
                    f"Agent '{self.name}' LLM request failed: {exc}",
                    context={"agent": self.name, "model": self._model, "upstream_error": type(exc).__name__},
                ) from exc
        # Unreachable — the loop always returns or raises above — but keeps
        # the type checker satisfied that every path produces a value.
        raise AssertionError("unreachable")

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
            if iteration == 0 and self._require_tool_on_first_turn is not None:
                tool_choice = {"type": "function", "function": {"name": self._require_tool_on_first_turn}}
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
