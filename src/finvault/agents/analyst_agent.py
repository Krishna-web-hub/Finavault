"""Analyst agent: grounded financial reasoning over retrieved context.

Arithmetic goes through `calculate`, a restricted AST-walking evaluator —
not Python's `eval()` — so a malicious or injected expression (e.g. arriving
via a prompt-injected document chunk) can't do anything beyond arithmetic on
literals. No names, calls, attributes, imports, or comprehensions are
reachable.

The final answer is structured (`submit_answer`, a terminal tool — see
agents/base.py) rather than free text: {answer, citations, calculations}.
This exists so the Orchestrator and Compliance agent can verify each
citation's quoted text actually appears in the retrieved context, instead of
trusting the model's claim that it cited something real.
"""

from __future__ import annotations

import ast
import json
import operator

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from finvault.agents.base import Agent, TokenBudget, ToolDefinition
from finvault.observability import extra_fields, get_logger
from finvault.security.guardrails import INJECTION_DEFENSE_INSTRUCTION

logger = get_logger(__name__)

SYSTEM_PROMPT = f"""You are the Analyst agent in a financial-document RAG system.
You receive retrieved document context and a user's question, and produce
grounded financial analysis: ratios, trend comparisons, risk observations.
Use the calculate tool for any arithmetic rather than computing it yourself —
financial figures must be exact. If the context doesn't contain enough
information to answer confidently, say so explicitly rather than guessing or
using outside knowledge.

When you have your final answer, call submit_answer rather than replying
with plain text. For every citation, `quoted_text` must be copied verbatim
from the retrieved context you were given — not paraphrased, not
reconstructed from memory — because it will be checked against that context
word-for-word. If a claim in your answer isn't backed by an exact quote you
can produce, don't cite it; say the context doesn't support it instead.

{INJECTION_DEFENSE_INSTRUCTION}
"""

_ALLOWED_OPERATORS: dict[type, object] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))  # type: ignore[operator]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.operand))  # type: ignore[operator]
    raise ValueError(f"Disallowed expression element: {ast.dump(node)}")


def calculate(input_: dict) -> str:
    expression = input_["expression"]
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
    except Exception as exc:  # noqa: BLE001 — surfaced to the model as a tool error, not raised
        # INFO, not an error level: a rejected expression is usually the
        # model trying something the sandbox forbids, and it recovers by
        # retrying. Logged because a *repeated* rejection is a prompt bug.
        logger.info(
            "calculator_rejected_expression",
            extra=extra_fields(error_type=type(exc).__name__, error_message=str(exc)),
        )
        return f"Error: could not evaluate expression: {exc}"
    return str(result)


class Citation(BaseModel):
    document: str = ""
    quoted_text: str


class AnalystAnswer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    calculations: list[str] = Field(default_factory=list)


_SUBMIT_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "description": "The final answer to the user's question."},
        "citations": {
            "type": "array",
            "description": "One entry per factual claim, backed by an exact quote from the retrieved context.",
            "items": {
                "type": "object",
                "properties": {
                    "document": {"type": "string", "description": "Source document title, if known."},
                    "quoted_text": {
                        "type": "string",
                        "description": "Exact substring copied from the retrieved context supporting this claim.",
                    },
                },
                "required": ["quoted_text"],
            },
        },
        "calculations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Expressions passed to the calculate tool and their results, if any arithmetic was used.",
        },
    },
    "required": ["answer", "citations"],
}


def _submit_answer_unreachable(_: dict) -> str:
    # Never actually invoked: base.py's terminal_tool mechanism intercepts a
    # call to this tool before any handler dispatch — the model's arguments
    # become the agent's return value directly. This handler only exists to
    # satisfy ToolDefinition's required `handler` field.
    return ""


class AnalystAgent:
    def __init__(self, *, model: str | None = None, client: OpenAI | None = None) -> None:
        self._agent = Agent(
            name="analyst",
            system_prompt=SYSTEM_PROMPT,
            model=model,
            client=client,
            terminal_tool="submit_answer",
            tools=[
                ToolDefinition(
                    name="calculate",
                    description=(
                        "Evaluate an arithmetic expression, e.g. a financial ratio. "
                        "Supports + - * / ** % and parentheses on numeric literals only."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"],
                    },
                    handler=calculate,
                ),
                ToolDefinition(
                    name="submit_answer",
                    description="Submit your final structured answer. Call this once you're done reasoning.",
                    input_schema=_SUBMIT_ANSWER_SCHEMA,
                    handler=_submit_answer_unreachable,
                ),
            ],
        )

    def run(self, prompt: str, *, budget: TokenBudget | None = None) -> str:
        return self._agent.run(prompt, budget=budget)

    def run_structured(self, prompt: str, *, budget: TokenBudget | None = None) -> AnalystAnswer:
        """Runs the agent and parses its output as a structured answer.

        Falls back to wrapping raw text as an answer with no citations if
        the model never called submit_answer (e.g. answered directly, or an
        unreliable model ignored the instruction) — degraded but still
        functional, matching this codebase's fail-closed-on-errors-but-not-
        on-model-quirks posture (see base.py's empty-response retry logic).
        """
        raw = self.run(prompt, budget=budget)
        try:
            return AnalystAnswer.model_validate_json(raw)
        except (ValidationError, json.JSONDecodeError) as exc:
            # Deliberate degradation, not a failure — but silent degradation
            # is how "why are there no citations?" becomes unanswerable, so
            # it is recorded. Compliance still refuses to verify citations
            # that are not there (see compliance_agent.py).
            logger.warning(
                "analyst_answer_unstructured",
                extra=extra_fields(error_type=type(exc).__name__, raw_length=len(raw)),
            )
            return AnalystAnswer(answer=raw, citations=[])
