"""Tests for the SSE plumbing behind POST /query/stream.

No FastAPI TestClient/HTTP-level test here — this codebase has never had
route-level tests (main.py's lifespan connects to real Qdrant/Postgres and
would make a real paid LLM call through OpenRouter, none of which belong in
a unit test), and turning that around is a bigger change than this
milestone's scope. What IS tested here is the actual new, risky code: the
background-thread + thread-safe-queue bridge that lets a synchronous
Orchestrator.handle() feed a live event stream, replicated exactly as
routes.py wires it, plus the SSE wire-format helper.

Known gap: if routes.py's literal FastAPI decorator/StreamingResponse
wiring around this logic breaks, nothing here would catch it — only the
underlying mechanism is verified.
"""

from __future__ import annotations

import queue
import threading

from finvault.agents.analyst_agent import AnalystAgent
from finvault.agents.compliance_agent import ComplianceAgent
from finvault.agents.execution_events import ExecutionEvent, ExecutionEventBus
from finvault.agents.orchestrator import Orchestrator
from finvault.api.routes import _QUEUE_DONE, _sse_message
from finvault.errors import AgentExecutionError
from finvault.models import Role, User
from finvault.security.audit import InMemoryAuditLog
from tests.fakes import FakeOpenAIClient, FakeResponse
from tests.test_orchestrator import _RETRIEVED_CONTEXT, _FakeRetrieverAgent


def test_sse_message_wire_format() -> None:
    message = _sse_message("step_started", {"agent": "retriever", "action": "search_documents"})
    assert message == 'event: step_started\ndata: {"agent": "retriever", "action": "search_documents"}\n\n'


def _run_bridge(*, top_level_client: FakeOpenAIClient, analyst_client: FakeOpenAIClient) -> list:
    """Replicates routes.py's query_stream bridge exactly: background
    thread runs handle(), events cross into a queue.Queue via the bus, a
    consumer drains it — here synchronously (a plain loop instead of an
    async generator awaiting run_in_executor), which exercises the same
    thread-safety and ordering guarantees without needing asyncio.
    """
    audit_log = InMemoryAuditLog()
    user = User(username="tester", role=Role.ANALYST, org_id="org-a")
    orchestrator = Orchestrator(
        user=user,
        audit_log=audit_log,
        client=top_level_client,
        retriever_agent=_FakeRetrieverAgent(_RETRIEVED_CONTEXT),
        analyst_agent=AnalystAgent(client=analyst_client),
        compliance_agent=ComplianceAgent(semantic_review=False),
    )

    event_bus = ExecutionEventBus()
    event_queue: queue.Queue = queue.Queue()
    event_bus.subscribe(event_queue.put)

    def run_orchestrator() -> None:
        try:
            result = orchestrator.handle("What was Q1 revenue?", session_id="s1", event_bus=event_bus)
            event_queue.put(("done", result))
        except Exception as exc:
            event_queue.put(("error", str(exc)))
        finally:
            event_queue.put(_QUEUE_DONE)

    thread = threading.Thread(target=run_orchestrator, daemon=True)
    thread.start()

    drained: list = []
    while True:
        item = event_queue.get(timeout=5)
        if item is _QUEUE_DONE:
            break
        drained.append(item)
    thread.join(timeout=5)
    return drained


def test_bridge_delivers_events_in_order_then_a_done_tuple() -> None:
    top_level_client = FakeOpenAIClient(
        [
            FakeResponse.tool_call("search_documents", {"query": "Q1 revenue"}),
            FakeResponse.tool_call("analyze", {"question": "What was Q1 revenue?", "context": _RETRIEVED_CONTEXT}),
        ]
    )
    analyst_client = FakeOpenAIClient(
        [FakeResponse.tool_call("submit_answer", {"answer": "Q1 revenue was $10 million.", "citations": []})]
    )

    drained = _run_bridge(top_level_client=top_level_client, analyst_client=analyst_client)

    event_items = [item for item in drained if isinstance(item, ExecutionEvent)]
    terminal_items = [item for item in drained if isinstance(item, tuple)]

    assert [e.type for e in event_items] == [
        "step_started",
        "step_finished",
        "step_started",
        "step_finished",
        "step_started",
        "step_finished",
    ]
    # The terminal tuple always arrives last, after every step event.
    assert drained[-1] == terminal_items[-1]
    assert terminal_items[-1][0] == "done"
    result = terminal_items[-1][1]
    assert result.blocked is False
    assert result.answer == "Q1 revenue was $10 million."


def test_bridge_still_delivers_a_done_tuple_when_handle_fails_closed_internally() -> None:
    """AgentExecutionError is caught *inside* handle() (see orchestrator.py)
    and turned into a normal blocked OrchestratorResult — it never escapes
    handle() itself, so the bridge's own `except Exception` (reserved for
    something genuinely unexpected escaping handle(), not this) never
    fires. This proves handle()'s existing fail-closed contract survives
    being run through the bridge unchanged.
    """
    top_level_client = FakeOpenAIClient([RuntimeError("simulated network failure")])
    analyst_client = FakeOpenAIClient([])  # never reached

    drained = _run_bridge(top_level_client=top_level_client, analyst_client=analyst_client)

    terminal_items = [item for item in drained if isinstance(item, tuple)]
    assert len(terminal_items) == 1
    assert terminal_items[0][0] == "done"
    result = terminal_items[0][1]
    assert result.blocked is True
    # block_reason is now the exception's `code` from finvault/errors.py —
    # one vocabulary shared by the API error envelope, the logs, and this
    # field, instead of a literal invented at the raise site.
    assert result.block_reason == AgentExecutionError.code
