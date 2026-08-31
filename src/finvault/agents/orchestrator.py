"""Orchestrator: routes a user question to the Retriever and Analyst agents,
then hands the draft answer to the Compliance agent unconditionally before
it can reach the user.

The Compliance step is Python control flow (`handle()` below), not a tool
the orchestrating LLM decides to invoke — that distinction is what makes it
an actual veto rather than a suggestion the model could skip.

`analyze` is configured as this agent's `final_tool` (see agents/base.py):
once the Orchestrator calls it, the Analyst's structured answer becomes the
Orchestrator's own final output immediately, rather than being handed back
to the Orchestrator model for further paraphrasing — which would otherwise
flatten the structured {answer, citations, calculations} payload back into
free text before Compliance ever sees the citations to verify.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

from openai import OpenAI
from pydantic import ValidationError

from finvault.agents.analyst_agent import AnalystAgent, AnalystAnswer
from finvault.agents.base import Agent, AgentExecutionError, TokenBudget, ToolDefinition
from finvault.agents.canvas_models import ExecutionStepNode, ExecutionStepStatus, KnowledgeGraphData
from finvault.agents.compliance_agent import ComplianceAgent, ComplianceVerdict
from finvault.agents.execution_events import ExecutionEvent, ExecutionEventBus
from finvault.agents.retriever_agent import RetrieverAgent
from finvault.agents.session import SessionStore
from finvault.config import settings
from finvault.models import User
from finvault.observability import extra_fields, get_logger, log_exception
from finvault.retrieval.graph_retriever import GraphRetriever
from finvault.retrieval.retriever import Retriever
from finvault.security.audit import AuditLog
from finvault.security.guardrails import scan_and_redact
from finvault.security.review_queue import ReviewQueue

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are the Orchestrator for a secure financial-document
assistant. You coordinate two specialist tools:
- search_documents: retrieves relevant passages from the document corpus for a query.
- analyze: given retrieved context and a question, produces grounded financial analysis.

For every user question: first call search_documents (call it more than once
with reformulated queries if the first results don't cover the question),
then call analyze with the retrieved context and the original question to
produce your answer. Do not answer from your own general knowledge of
finance — ground every claim in the retrieved context, and say so explicitly
if the context is insufficient to answer confidently.

search_documents works over every ingested document regardless of its
original file type (PDF, DOCX, TXT, MD, CSV) — the user's uploaded CSV
files are already indexed and searchable exactly like any other document.
Never decline to search, and never tell the user your tools "aren't
suitable," just because their question mentions a file format, a row
number, a specific record, or otherwise sounds technical — that phrasing
describes what they want to know, not a reason to skip search_documents.
If a question is genuinely too vague to form a search query from (e.g. "hi"
with no topic at all), ask a brief clarifying question instead of guessing
— but always attempt search_documents first when the question names any
topic, column, figure, or record to look for, however it's phrased.
"""

_PREVIEW_LIMIT = 160


def _preview(text: str) -> str:
    """Truncates a step payload for the execution trace — a preview, not
    the full content (the frontend renders this inline in a compact DAG
    node, not as a document viewer).
    """
    text = text.strip()
    return text if len(text) <= _PREVIEW_LIMIT else text[:_PREVIEW_LIMIT].rstrip() + "…"


T = TypeVar("T")


@dataclass
class OrchestratorResult:
    answer: str
    blocked: bool
    block_reason: str | None
    injection_flags: list[dict] = field(default_factory=list)
    guardrail_findings: list[str] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    review_item_id: str | None = None
    # execution_steps: real measured durations from the event bus (see
    # agents/execution_events.py), never estimates.
    # graph_data: entities/relationships extracted at ingest time (see
    # ingestion/extraction.py), scoped to just the documents this query
    # actually retrieved — empty unless a graph_retriever was configured
    # AND at least one document was retrieved. Typed against canvas_models
    # so a future populator can't silently drift from what the frontend
    # actually renders.
    execution_steps: list[ExecutionStepNode] = field(default_factory=list)
    graph_data: KnowledgeGraphData = field(default_factory=KnowledgeGraphData)


class Orchestrator:
    def __init__(
        self,
        *,
        retriever: Retriever | None = None,
        user: User,
        audit_log: AuditLog,
        model: str | None = None,
        compliance_agent: ComplianceAgent | None = None,
        client: OpenAI | None = None,
        retriever_agent: RetrieverAgent | None = None,
        analyst_agent: AnalystAgent | None = None,
        session_store: SessionStore | None = None,
        max_tokens_per_request: int | None = None,
        review_queue: ReviewQueue | None = None,
        graph_retriever: GraphRetriever | None = None,
    ) -> None:
        self._user = user
        self._audit_log = audit_log
        self._max_tokens_per_request = max_tokens_per_request or settings.finvault_max_tokens_per_request
        # Optional — see security/review_queue.py. None means a block still
        # behaves exactly as before this feature existed: a generic message,
        # nothing queued for follow-up.
        self._review_queue = review_queue
        # Optional — see retrieval/graph_retriever.py. None means graph_data
        # stays empty on every result, exactly as before this feature existed.
        self._graph_retriever = graph_retriever
        # retriever_agent/analyst_agent are injectable (mirroring
        # compliance_agent below) so tests can substitute controllable fakes
        # without standing up a full retrieval pipeline; production callers
        # just pass `retriever` and let these build normally.
        self._retriever_agent = retriever_agent or RetrieverAgent(retriever=retriever, user=user, model=model)  # type: ignore[arg-type]
        self._analyst_agent = analyst_agent or AnalystAgent(model=model)
        self._compliance_agent = compliance_agent or ComplianceAgent(model=model)
        # Optional — see agents/session.py. None means stateless, single-turn
        # behavior, exactly as before this feature existed.
        self._session_store = session_store
        self._injection_flags: list[dict] = []
        self._last_context: str = ""
        # Fresh per handle() call (see below) — referenced via self so these
        # closures, built once at construction time, always see the budget
        # for whichever request is currently in flight.
        self._current_budget: TokenBudget | None = None
        # Same per-call-freshness reasoning as _current_budget above — see
        # execution_bus property and _run_step below.
        self._current_bus: ExecutionEventBus = ExecutionEventBus()
        self._current_steps: list[ExecutionStepNode] = []

        def search_documents(input_: dict) -> str:
            def run() -> tuple[str, list[dict]]:
                return self._retriever_agent.run(input_["query"], budget=self._current_budget)

            result, flags = self._run_step(
                agent="retriever", action="search_documents", fn=run, preview=lambda r: _preview(r[0])
            )
            self._injection_flags.extend(flags)
            return result

        def analyze(input_: dict) -> str:
            self._last_context = input_["context"]

            def run() -> AnalystAnswer:
                return self._analyst_agent.run_structured(
                    f"Question: {input_['question']}\n\nRetrieved context:\n{input_['context']}",
                    budget=self._current_budget,
                )

            # scan_and_redact here, not just on the final answer: this is
            # the Analyst's raw pre-compliance draft, which can legitimately
            # contain PII the compliance step exists to strip before it
            # reaches the user. Previewing the raw draft in the execution
            # trace would create a second, unredacted place that PII
            # reaches — the same leak scan_and_redact is meant to prevent,
            # just via a different surface.
            structured = self._run_step(
                agent="analyst",
                action="analyze",
                fn=run,
                preview=lambda a: _preview(scan_and_redact(a.answer).redacted),
            )
            return structured.model_dump_json()

        self._agent = Agent(
            name="orchestrator",
            system_prompt=SYSTEM_PROMPT,
            model=model,
            client=client,
            final_tool="analyze",
            # A prompt instruction alone wasn't reliable enough: observed
            # live, a free-tier model skipped search_documents entirely and
            # answered directly (or refused) whenever the question's
            # surface phrasing mentioned a file format or a specific row —
            # reasoning, incorrectly, that its tools "weren't suitable".
            # Forcing the very first call is deterministic where the prompt
            # wasn't; every later call in the loop is still free (reformulate,
            # call analyze, or anything else).
            require_tool_on_first_turn="search_documents",
            tools=[
                ToolDefinition(
                    name="search_documents",
                    description="Retrieve relevant passages from the document corpus for a query.",
                    input_schema={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                    handler=search_documents,
                ),
                ToolDefinition(
                    name="analyze",
                    description="Produce grounded financial analysis from retrieved context and a question.",
                    input_schema={
                        "type": "object",
                        "properties": {"question": {"type": "string"}, "context": {"type": "string"}},
                        "required": ["question", "context"],
                    },
                    handler=analyze,
                ),
            ],
        )

    @property
    def execution_bus(self) -> ExecutionEventBus:
        """The bus for whichever request is currently in flight (or the
        most recently completed one) — a post-hoc inspection convenience
        (tests, mainly). For live streaming, a caller must construct its
        own bus, subscribe to it, and pass it to handle(event_bus=...) —
        subscribing here after handle() has already started (or worse,
        already returned) is too late; handle() only reads this property's
        underlying bus, it never lets a subscriber attach to it early.
        """
        return self._current_bus

    def _run_step(
        self,
        *,
        agent: str,
        action: str,
        fn: Callable[[], T],
        preview: Callable[[T], str],
        status_fn: Callable[[T], ExecutionStepStatus] | None = None,
        veto_reason_fn: Callable[[T], str | None] | None = None,
    ) -> T:
        """Runs one real sub-agent call, measuring actual wall-clock
        duration and publishing start/finish events — the only place
        ExecutionStepNode instances get constructed, so every step in the
        trace is backed by something that actually happened.
        """
        self._current_bus.publish(ExecutionEvent(type="step_started", agent=agent, action=action))
        start = time.perf_counter()
        try:
            result = fn()
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            # Logged here rather than only at the top of handle(): this is
            # the only frame that knows WHICH step failed and how long it
            # ran before it did. The exception is re-raised unchanged below,
            # so handle()'s own handler still owns the fail-closed decision.
            log_exception(
                logger, exc, "execution_step_failed", agent=agent, action=action, duration_ms=round(duration_ms, 2)
            )
            step = ExecutionStepNode(
                step_id=str(uuid.uuid4()),
                agent_name=agent,
                action=action,
                status="error",
                duration_ms=duration_ms,
                payload_preview=_preview(f"Error: {exc}"),
                timestamp=time.time(),
            )
            self._current_steps.append(step)
            self._current_bus.publish(ExecutionEvent(type="step_finished", agent=agent, action=action, step=step))
            raise

        duration_ms = (time.perf_counter() - start) * 1000
        status: ExecutionStepStatus = (status_fn or (lambda _: "success"))(result)
        step = ExecutionStepNode(
            step_id=str(uuid.uuid4()),
            agent_name=agent,
            action=action,
            status=status,
            duration_ms=duration_ms,
            payload_preview=preview(result),
            veto_reason=veto_reason_fn(result) if veto_reason_fn else None,
            timestamp=time.time(),
        )
        self._current_steps.append(step)
        self._current_bus.publish(ExecutionEvent(type="step_finished", agent=agent, action=action, step=step))
        return result

    def handle(
        self, question: str, *, session_id: str | None = None, event_bus: ExecutionEventBus | None = None
    ) -> OrchestratorResult:
        """`event_bus`: optional, for live streaming (Milestone 2's SSE
        route). Construct one, subscribe to it, *then* pass it in — a bus
        supplied this way is used as-is (not replaced), so every event
        published during this call reaches your subscriber in real time.
        Omit it (the default, used by every existing caller) and handle()
        creates its own fresh, private bus, exactly as before this param
        existed — nothing outside can observe events live, only inspect
        them after the fact via `execution_bus` or the returned result's
        `execution_steps`.
        """
        self._injection_flags = []
        self._last_context = ""
        self._current_budget = TokenBudget(limit=self._max_tokens_per_request)
        self._current_bus = event_bus or ExecutionEventBus()
        self._current_steps = []
        handle_start = time.perf_counter()
        self._audit_log.append(
            actor=self._user.id,
            action="query",
            resource="orchestrator",
            details={"question": question, "session_id": session_id},
        )

        history: list[dict[str, str]] | None = None
        if self._session_store is not None and session_id is not None:
            history = []
            for turn in self._session_store.get_history(session_id=session_id, user_id=self._user.id):
                history.append({"role": "user", "content": turn.question})
                history.append({"role": "assistant", "content": turn.answer})

        try:
            draft_raw = self._agent.run(question, history=history, budget=self._current_budget)
        except AgentExecutionError as exc:
            # Fail closed: an infra/billing/network failure at the orchestrator
            # level means no draft answer exists to review. Don't guess, don't
            # crash the caller (API route, script, whatever) — report a clean
            # unavailable result and log it, same as a compliance block.
            # TokenBudgetExceeded is a subclass of AgentExecutionError (see
            # base.py) so it's caught here too — distinguished only for a
            # more informative block_reason, not different handling.
            # block_reason comes from the exception's own `code` (see
            # finvault/errors.py) so the string the frontend displays, the
            # string in the logs, and the string in the audit row are the
            # same one — they used to be three independent literals.
            block_reason = exc.code
            log_exception(logger, exc, "orchestrator_failed", session_id=session_id, block_reason=block_reason)
            self._audit_log.append(
                actor=self._user.id, action="agent_failure", resource="orchestrator", details={"error": str(exc)}
            )
            # Whatever steps completed before the failure are already in
            # self._current_steps (appended by _run_step as each one
            # finished) — plus one final synthetic step marking where the
            # chain actually broke, with a real elapsed-time measurement
            # for the whole failed attempt, not a fabricated number.
            self._current_steps.append(
                ExecutionStepNode(
                    step_id=str(uuid.uuid4()),
                    agent_name="orchestrator",
                    action="run",
                    status="error",
                    duration_ms=(time.perf_counter() - handle_start) * 1000,
                    payload_preview=_preview(f"Error: {exc}"),
                    timestamp=time.time(),
                )
            )
            return OrchestratorResult(
                answer="The assistant is temporarily unavailable and could not process this request. Please try again shortly.",
                blocked=True,
                block_reason=block_reason,
                execution_steps=list(self._current_steps),
            )

        # draft_raw is the structured submit_answer JSON when the Orchestrator
        # actually called analyze (the normal path — see `final_tool` above);
        # if the model answered directly without calling analyze, it's plain
        # text, and this falls back to an answer with no citations to verify.
        try:
            parsed = AnalystAnswer.model_validate_json(draft_raw)
        except ValidationError:
            # Not an error: the model answered in plain text instead of
            # calling analyze. Recorded at INFO because it means the answer
            # reaching compliance has no citations to verify, which changes
            # how a downstream block should be read.
            logger.info("orchestrator_answer_unstructured", extra=extra_fields(session_id=session_id))
            parsed = AnalystAnswer(answer=draft_raw, citations=[])

        max_classification = self._retriever_agent.max_classification_seen
        citation_dicts = [c.model_dump() for c in parsed.citations]

        def run_compliance() -> ComplianceVerdict:
            return self._compliance_agent.review_output(
                question=question,
                draft_answer=parsed.answer,
                max_classification=max_classification,
                citations=citation_dicts,
                context=self._last_context,
            )

        # verdict.reason is designed to be safe to surface (see
        # compliance_agent.py) — never verdict.reviewable_answer, which is
        # deliberately reserved for the review queue (security/review_queue.py),
        # not for a user-facing execution trace.
        verdict = self._run_step(
            agent="compliance",
            action="review_output",
            fn=run_compliance,
            preview=lambda v: v.reason or "Approved — no compliance findings.",
            status_fn=lambda v: "success" if v.allowed else "vetoed",
            veto_reason_fn=lambda v: v.reason if not v.allowed else None,
        )

        self._audit_log.append(
            actor=self._user.id,
            action="compliance_review",
            resource="orchestrator",
            details={
                "allowed": verdict.allowed,
                "findings": verdict.findings,
                "injection_flags": self._injection_flags,
                "max_classification": max_classification.value,
                "reason": verdict.reason,
            },
        )

        if not verdict.allowed:
            review_item_id = None
            if self._review_queue is not None:
                item = self._review_queue.enqueue(
                    org_id=self._user.org_id,
                    user_id=self._user.id,
                    question=question,
                    draft_answer=verdict.reviewable_answer or "",
                    block_reason=verdict.reason,
                    findings=verdict.findings,
                    citations=citation_dicts,
                )
                review_item_id = item.id
                self._audit_log.append(
                    actor=self._user.id,
                    action="compliance_review_queued",
                    resource="orchestrator",
                    details={"review_item_id": review_item_id},
                )
            return OrchestratorResult(
                answer="This response was blocked by compliance review and requires manual handling.",
                blocked=True,
                block_reason=verdict.reason,
                injection_flags=self._injection_flags,
                guardrail_findings=verdict.findings,
                review_item_id=review_item_id,
                execution_steps=list(self._current_steps),
            )

        if self._session_store is not None and session_id is not None:
            # Only a compliance-approved answer ever enters history — see
            # agents/session.py's module docstring for why that's safe.
            self._session_store.append_turn(
                session_id=session_id, user_id=self._user.id, question=question, answer=verdict.redacted_answer
            )

        graph_data = KnowledgeGraphData()
        if self._graph_retriever is not None:
            retrieved_document_ids = self._retriever_agent.retrieved_document_ids
            if retrieved_document_ids:
                graph_data = self._graph_retriever.get_graph(user=self._user, document_ids=retrieved_document_ids)

        return OrchestratorResult(
            answer=verdict.redacted_answer,
            blocked=False,
            block_reason=None,
            injection_flags=self._injection_flags,
            guardrail_findings=verdict.findings,
            citations=citation_dicts,
            execution_steps=list(self._current_steps),
            graph_data=graph_data,
        )
