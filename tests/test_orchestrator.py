from __future__ import annotations

import pytest

from finvault.agents.analyst_agent import AnalystAgent
from finvault.agents.compliance_agent import ComplianceAgent
from finvault.agents.orchestrator import Orchestrator
from finvault.errors import AgentExecutionError
from finvault.models import Role, User
from finvault.retrieval.retriever import Retriever
from finvault.retrieval.vector_store import InMemoryVectorStore
from finvault.security.audit import InMemoryAuditLog
from finvault.security.encryption import EnvelopeEncryptor, LocalKeyProvider
from tests.fakes import FakeEmbeddingProvider, FakeOpenAIClient, FakeResponse, FakeRetrieverAgent


def _make_orchestrator(tmp_path, *, client: FakeOpenAIClient) -> tuple[Orchestrator, InMemoryAuditLog]:
    encryptor = EnvelopeEncryptor(LocalKeyProvider(tmp_path / "master.key"))
    vector_store = InMemoryVectorStore()
    audit_log = InMemoryAuditLog()
    retriever = Retriever(
        vector_store=vector_store, embedding_provider=FakeEmbeddingProvider(), encryptor=encryptor, audit_log=audit_log
    )
    user = User(username="tester", role=Role.ANALYST, org_id="org-a")
    orchestrator = Orchestrator(retriever=retriever, user=user, audit_log=audit_log, client=client)
    return orchestrator, audit_log


def test_orchestrator_fails_closed_when_top_level_agent_call_errors(tmp_path) -> None:
    # No scripted responses — the very first LLM call raises inside
    # FakeCompletions.create(), simulating a billing/network/API failure.
    client = FakeOpenAIClient([RuntimeError("simulated 402: out of credits")])
    orchestrator, audit_log = _make_orchestrator(tmp_path, client=client)

    result = orchestrator.handle("What was Q3 revenue?")

    assert result.blocked is True
    # block_reason is now the exception's `code` from finvault/errors.py —
    # one vocabulary shared by the API error envelope, the logs, and this
    # field, instead of a literal invented at the raise site.
    assert result.block_reason == AgentExecutionError.code
    # Must not have raised — the caller (API route, script, etc.) gets a
    # clean result, never a crash or a leaked stack trace.
    failure_entries = [e for e in audit_log.entries() if e.action == "agent_failure"]
    assert len(failure_entries) == 1


# Promoted to tests/fakes.py (as FakeRetrieverAgent) for reuse across test
# modules — aliased back to this historical name since it's referenced
# throughout this file, and test_query_stream.py imports it from here too.
_FakeRetrieverAgent = FakeRetrieverAgent

_RETRIEVED_CONTEXT = "Total revenue was $10 million in Q1, up from $8 million in the prior quarter."


def _make_orchestrator_with_injected_agents(
    *, analyst_client: FakeOpenAIClient
) -> tuple[Orchestrator, InMemoryAuditLog]:
    audit_log = InMemoryAuditLog()
    user = User(username="tester", role=Role.ANALYST, org_id="org-a")
    top_level_client = FakeOpenAIClient(
        [
            FakeResponse.tool_call("search_documents", {"query": "Q1 revenue"}),
            FakeResponse.tool_call("analyze", {"question": "What was Q1 revenue?", "context": _RETRIEVED_CONTEXT}),
        ]
    )
    orchestrator = Orchestrator(
        user=user,
        audit_log=audit_log,
        client=top_level_client,
        retriever_agent=_FakeRetrieverAgent(_RETRIEVED_CONTEXT),
        analyst_agent=AnalystAgent(client=analyst_client),
        compliance_agent=ComplianceAgent(semantic_review=False),
    )
    return orchestrator, audit_log


def test_orchestrator_allows_answer_with_verified_citations() -> None:
    analyst_client = FakeOpenAIClient(
        [
            FakeResponse.tool_call(
                "submit_answer",
                {
                    "answer": "Q1 revenue was $10 million.",
                    "citations": [{"document": "Q1 Report", "quoted_text": "Total revenue was $10 million"}],
                    "calculations": [],
                },
            )
        ]
    )
    orchestrator, _ = _make_orchestrator_with_injected_agents(analyst_client=analyst_client)

    result = orchestrator.handle("What was Q1 revenue?")

    assert result.blocked is False
    assert result.answer == "Q1 revenue was $10 million."
    assert result.citations == [{"document": "Q1 Report", "quoted_text": "Total revenue was $10 million"}]


def test_graph_data_stays_empty_without_a_graph_retriever_configured() -> None:
    """graph_retriever is optional (see Orchestrator.__init__) — omitting it
    must leave graph_data empty rather than fabricated, same reasoning as
    every other optional dependency in this class.
    """
    from finvault.agents.canvas_models import KnowledgeGraphData

    analyst_client = FakeOpenAIClient(
        [
            FakeResponse.tool_call(
                "submit_answer",
                {"answer": "Q1 revenue was $10 million.", "citations": [], "calculations": []},
            )
        ]
    )
    orchestrator, _ = _make_orchestrator_with_injected_agents(analyst_client=analyst_client)

    result = orchestrator.handle("What was Q1 revenue?")

    assert result.graph_data == KnowledgeGraphData()


def test_graph_data_stays_empty_when_no_documents_were_retrieved(tmp_path) -> None:
    """A graph_retriever can be configured while the retriever agent still
    reports no retrieved_document_ids (e.g. "No matching passages found.") —
    must not fall back to the user's *entire* org graph in that case.
    """
    from finvault.agents.canvas_models import KnowledgeGraphData
    from finvault.retrieval.graph_retriever import GraphRetriever
    from finvault.retrieval.graph_store import InMemoryGraphStore

    audit_log = InMemoryAuditLog()
    user = User(username="tester", role=Role.ANALYST, org_id="org-a")
    encryptor = EnvelopeEncryptor(LocalKeyProvider(tmp_path / "master.key"))
    graph_retriever = GraphRetriever(graph_store=InMemoryGraphStore(), encryptor=encryptor)
    top_level_client = FakeOpenAIClient(
        [
            FakeResponse.tool_call("search_documents", {"query": "Q1 revenue"}),
            FakeResponse.tool_call("analyze", {"question": "What was Q1 revenue?", "context": _RETRIEVED_CONTEXT}),
        ]
    )
    analyst_client = FakeOpenAIClient(
        [
            FakeResponse.tool_call(
                "submit_answer",
                {"answer": "Q1 revenue was $10 million.", "citations": [], "calculations": []},
            )
        ]
    )
    orchestrator = Orchestrator(
        user=user,
        audit_log=audit_log,
        client=top_level_client,
        retriever_agent=_FakeRetrieverAgent(_RETRIEVED_CONTEXT, retrieved_document_ids=set()),
        analyst_agent=AnalystAgent(client=analyst_client),
        compliance_agent=ComplianceAgent(semantic_review=False),
        graph_retriever=graph_retriever,
    )

    result = orchestrator.handle("What was Q1 revenue?")

    assert result.graph_data == KnowledgeGraphData()


def test_graph_data_is_scoped_to_the_documents_this_query_retrieved(tmp_path) -> None:
    from finvault.retrieval.graph_retriever import GraphRetriever
    from finvault.retrieval.graph_store import InMemoryGraphStore, node_aad
    from finvault.retrieval.graph_store import label_hash as graph_label_hash

    audit_log = InMemoryAuditLog()
    user = User(username="tester", role=Role.ANALYST, org_id="org-a")
    encryptor = EnvelopeEncryptor(LocalKeyProvider(tmp_path / "master.key"))
    graph_store = InMemoryGraphStore()

    retrieved_doc_id = "doc-in-this-query"
    other_doc_id = "doc-not-in-this-query"
    for org_id, doc_id, label in (("org-a", retrieved_doc_id, "Acme Capital"), ("org-a", other_doc_id, "Other Corp")):
        h = graph_label_hash(org_id, "company", label)
        graph_store.upsert_node(
            org_id=org_id,
            type="company",
            classification="internal",
            source_document_id=doc_id,
            node_label_hash=h,
            label_encrypted=encryptor.encrypt(label, aad=node_aad(org_id, "company", h)),
        )
    graph_retriever = GraphRetriever(graph_store=graph_store, encryptor=encryptor)

    top_level_client = FakeOpenAIClient(
        [
            FakeResponse.tool_call("search_documents", {"query": "Q1 revenue"}),
            FakeResponse.tool_call("analyze", {"question": "What was Q1 revenue?", "context": _RETRIEVED_CONTEXT}),
        ]
    )
    analyst_client = FakeOpenAIClient(
        [
            FakeResponse.tool_call(
                "submit_answer",
                {"answer": "Q1 revenue was $10 million.", "citations": [], "calculations": []},
            )
        ]
    )
    orchestrator = Orchestrator(
        user=user,
        audit_log=audit_log,
        client=top_level_client,
        retriever_agent=_FakeRetrieverAgent(_RETRIEVED_CONTEXT, retrieved_document_ids={retrieved_doc_id}),
        analyst_agent=AnalystAgent(client=analyst_client),
        compliance_agent=ComplianceAgent(semantic_review=False),
        graph_retriever=graph_retriever,
    )

    result = orchestrator.handle("What was Q1 revenue?")

    assert [n.label for n in result.graph_data.nodes] == ["Acme Capital"]


def test_externally_supplied_event_bus_receives_events_live() -> None:
    """The exact bug Milestone 2 had to fix: handle() used to always build
    its own fresh bus, discarding anything a caller subscribed to
    beforehand — meaning live streaming was structurally impossible. A
    caller must now be able to construct a bus, subscribe, pass it to
    handle(event_bus=...), and see every event as it's published.
    """
    from finvault.agents.execution_events import ExecutionEvent, ExecutionEventBus

    analyst_client = FakeOpenAIClient(
        [FakeResponse.tool_call("submit_answer", {"answer": "Q1 revenue was $10 million.", "citations": []})]
    )
    orchestrator, _ = _make_orchestrator_with_injected_agents(analyst_client=analyst_client)

    bus = ExecutionEventBus()
    received: list[ExecutionEvent] = []
    bus.subscribe(received.append)

    result = orchestrator.handle("What was Q1 revenue?", event_bus=bus)

    # Two events per step (started, finished) x 3 steps.
    assert [e.type for e in received] == [
        "step_started",
        "step_finished",
        "step_started",
        "step_finished",
        "step_started",
        "step_finished",
    ]
    assert [e.agent for e in received if e.type == "step_started"] == ["retriever", "analyst", "compliance"]
    # The bus passed in is the one actually used — not silently replaced.
    assert orchestrator.execution_bus is bus
    # And it's consistent with the synchronously-returned result.
    finished_steps = [e.step for e in received if e.type == "step_finished"]
    assert finished_steps == result.execution_steps


def test_execution_steps_reflect_the_real_three_step_pipeline() -> None:
    analyst_client = FakeOpenAIClient(
        [
            FakeResponse.tool_call(
                "submit_answer",
                {
                    "answer": "Q1 revenue was $10 million.",
                    "citations": [{"document": "Q1 Report", "quoted_text": "Total revenue was $10 million"}],
                    "calculations": [],
                },
            )
        ]
    )
    orchestrator, _ = _make_orchestrator_with_injected_agents(analyst_client=analyst_client)

    result = orchestrator.handle("What was Q1 revenue?")

    assert [s.agent_name for s in result.execution_steps] == ["retriever", "analyst", "compliance"]
    assert [s.action for s in result.execution_steps] == ["search_documents", "analyze", "review_output"]
    assert all(s.status == "success" for s in result.execution_steps)
    # Real measurements, not the old fabricated 120/340/90 literals — every
    # step ran a real (fake-client-backed, but real code path) call, so
    # each duration must be an actual non-negative measurement.
    assert all(isinstance(s.duration_ms, float) and s.duration_ms >= 0.0 for s in result.execution_steps)
    assert all(s.step_id for s in result.execution_steps)  # non-empty, unique per step
    assert len({s.step_id for s in result.execution_steps}) == 3
    # The retriever step previews the actual retrieved context, not a
    # canned string like the old "Retrieved context passages" placeholder.
    retriever_step = result.execution_steps[0]
    assert _RETRIEVED_CONTEXT[:50] in retriever_step.payload_preview


def test_execution_steps_measure_real_elapsed_time(monkeypatch) -> None:
    """Proves the timing mechanism is a real measurement, not another
    hardcoded constant — by controlling time.perf_counter() precisely and
    checking the computed duration matches the injected delta exactly.
    """
    import time as time_module

    from finvault.agents import orchestrator as orchestrator_module

    # handle() itself calls time.perf_counter() once up front (handle_start,
    # for the error-path synthetic step — unused here since nothing fails)
    # before the three _run_step start/end pairs: 7 calls total.
    fake_times = iter([0.0, 100.0, 100.25, 200.0, 200.5, 300.0, 300.1])
    monkeypatch.setattr(orchestrator_module.time, "perf_counter", lambda: next(fake_times))
    monkeypatch.setattr(orchestrator_module.time, "time", time_module.time)  # timestamp field unaffected

    analyst_client = FakeOpenAIClient(
        [FakeResponse.tool_call("submit_answer", {"answer": "Q1 revenue was $10 million.", "citations": []})]
    )
    orchestrator, _ = _make_orchestrator_with_injected_agents(analyst_client=analyst_client)

    result = orchestrator.handle("What was Q1 revenue?")

    durations = [s.duration_ms for s in result.execution_steps]
    assert durations == pytest.approx([250.0, 500.0, 100.0])


def test_analyst_step_preview_is_redacted_even_though_final_answer_already_was() -> None:
    """The Analyst's raw draft can contain PII the compliance step exists
    to strip. If the execution-trace preview showed the raw draft, PII
    would reach the user through a second, unredacted surface — this proves
    the preview goes through the same scan_and_redact as the real answer.
    """
    analyst_client = FakeOpenAIClient(
        [
            FakeResponse.tool_call(
                "submit_answer",
                {"answer": "Contact john.doe@example.com for the Q1 filing.", "citations": []},
            )
        ]
    )
    orchestrator, _ = _make_orchestrator_with_injected_agents(analyst_client=analyst_client)

    result = orchestrator.handle("Who do I contact about Q1?")

    analyst_step = next(s for s in result.execution_steps if s.agent_name == "analyst")
    assert "john.doe@example.com" not in analyst_step.payload_preview
    assert "[REDACTED:EMAIL]" in analyst_step.payload_preview


def test_compliance_step_is_vetoed_with_a_safe_veto_reason_not_the_raw_draft() -> None:
    analyst_client = FakeOpenAIClient(
        [
            FakeResponse.tool_call(
                "submit_answer",
                {
                    "answer": "Net income was $50 million.",
                    "citations": [{"document": "Q1 Report", "quoted_text": "Net income was $50 million"}],
                    "calculations": [],
                },
            )
        ]
    )
    orchestrator, _ = _make_orchestrator_with_injected_agents(analyst_client=analyst_client)

    result = orchestrator.handle("What was net income?")

    compliance_step = next(s for s in result.execution_steps if s.agent_name == "compliance")
    assert compliance_step.status == "vetoed"
    assert compliance_step.veto_reason is not None
    assert "citation" in compliance_step.veto_reason.lower()
    # The raw blocked draft answer must never appear in a user-visible trace
    # — that's what security/review_queue.py's reviewable_answer is for.
    assert "Net income was $50 million" not in compliance_step.payload_preview


def test_execution_steps_captured_before_a_top_level_failure_are_preserved() -> None:
    """When the top-level agent call itself fails (budget exhausted,
    infra error), whatever sub-agent steps already completed must still be
    in the trace, plus a final synthetic step marking where it broke —
    not an empty list that discards real diagnostic information.
    """
    analyst_client = FakeOpenAIClient(
        [FakeResponse.tool_call("submit_answer", {"answer": "ok", "citations": []}, total_tokens=100)]
    )
    top_level_client = FakeOpenAIClient(
        [
            FakeResponse.tool_call("search_documents", {"query": "q"}, total_tokens=20),
            FakeResponse.tool_call("analyze", {"question": "q", "context": _RETRIEVED_CONTEXT}, total_tokens=20),
        ]
    )
    audit_log = InMemoryAuditLog()
    user = User(username="tester", role=Role.ANALYST, org_id="org-a")
    orchestrator = Orchestrator(
        user=user,
        audit_log=audit_log,
        client=top_level_client,
        retriever_agent=_FakeRetrieverAgent(_RETRIEVED_CONTEXT),
        analyst_agent=AnalystAgent(client=analyst_client),
        compliance_agent=ComplianceAgent(semantic_review=False),
        max_tokens_per_request=100,
    )

    result = orchestrator.handle("q")

    assert result.blocked is True
    assert result.block_reason == "token_budget_exceeded"
    agents_in_trace = [s.agent_name for s in result.execution_steps]
    assert agents_in_trace[0] == "retriever"  # completed before the budget blew up
    assert agents_in_trace[-1] == "orchestrator"  # synthetic final error step
    assert result.execution_steps[-1].status == "error"
    assert result.execution_steps[-1].duration_ms > 0.0


def test_conversation_memory_threads_prior_turn_into_the_next_call() -> None:
    from finvault.agents.session import InMemorySessionStore

    session_store = InMemorySessionStore()
    analyst_client = FakeOpenAIClient(
        [
            FakeResponse.tool_call(
                "submit_answer",
                {
                    "answer": "Q1 revenue was $10 million.",
                    "citations": [{"document": "Q1 Report", "quoted_text": "Total revenue was $10 million"}],
                    "calculations": [],
                },
            ),
            FakeResponse.tool_call(
                "submit_answer",
                {
                    "answer": "It grew from $8 million the prior quarter.",
                    "citations": [{"document": "Q1 Report", "quoted_text": "up from $8 million"}],
                    "calculations": [],
                },
            ),
        ]
    )
    top_level_client = FakeOpenAIClient(
        [
            FakeResponse.tool_call("search_documents", {"query": "Q1 revenue"}),
            FakeResponse.tool_call("analyze", {"question": "What was Q1 revenue?", "context": _RETRIEVED_CONTEXT}),
            FakeResponse.tool_call("search_documents", {"query": "revenue growth"}),
            FakeResponse.tool_call("analyze", {"question": "How much did it grow?", "context": _RETRIEVED_CONTEXT}),
        ]
    )
    audit_log = InMemoryAuditLog()
    user = User(username="tester", role=Role.ANALYST, org_id="org-a")
    orchestrator = Orchestrator(
        user=user,
        audit_log=audit_log,
        client=top_level_client,
        retriever_agent=_FakeRetrieverAgent(_RETRIEVED_CONTEXT),
        analyst_agent=AnalystAgent(client=analyst_client),
        compliance_agent=ComplianceAgent(semantic_review=False),
        session_store=session_store,
    )

    first = orchestrator.handle("What was Q1 revenue?", session_id="session-1")
    assert first.blocked is False

    second = orchestrator.handle("How much did it grow?", session_id="session-1")
    assert second.blocked is False

    # The second turn's very first LLM call must already carry the first
    # turn as history, ahead of the new question.
    second_turn_first_call_messages = top_level_client.chat.completions.calls[2]["messages"]
    contents = [m.get("content") for m in second_turn_first_call_messages]
    assert "What was Q1 revenue?" in contents
    assert "Q1 revenue was $10 million." in contents
    assert contents[-1] == "How much did it grow?"

    stored = session_store.get_history(session_id="session-1", user_id=user.id)
    assert len(stored) == 2
    assert stored[0].question == "What was Q1 revenue?"
    assert stored[1].question == "How much did it grow?"


def test_blocked_turn_is_not_written_to_session_history() -> None:
    from finvault.agents.session import InMemorySessionStore

    session_store = InMemorySessionStore()
    analyst_client = FakeOpenAIClient(
        [
            FakeResponse.tool_call(
                "submit_answer",
                {
                    "answer": "Net income was $50 million.",
                    "citations": [{"document": "Q1 Report", "quoted_text": "Net income was $50 million"}],
                    "calculations": [],
                },
            )
        ]
    )
    top_level_client = FakeOpenAIClient(
        [
            FakeResponse.tool_call("search_documents", {"query": "net income"}),
            FakeResponse.tool_call("analyze", {"question": "What was net income?", "context": _RETRIEVED_CONTEXT}),
        ]
    )
    audit_log = InMemoryAuditLog()
    user = User(username="tester", role=Role.ANALYST, org_id="org-a")
    orchestrator = Orchestrator(
        user=user,
        audit_log=audit_log,
        client=top_level_client,
        retriever_agent=_FakeRetrieverAgent(_RETRIEVED_CONTEXT),
        analyst_agent=AnalystAgent(client=analyst_client),
        compliance_agent=ComplianceAgent(semantic_review=False),
        session_store=session_store,
    )

    result = orchestrator.handle("What was net income?", session_id="session-1")

    assert result.blocked is True
    assert session_store.get_history(session_id="session-1", user_id=user.id) == []


def test_orchestrator_fails_closed_when_shared_token_budget_is_exhausted_by_a_nested_agent() -> None:
    """The budget is shared across the whole Orchestrator -> Retriever ->
    Analyst chain. This scripts the overflow to happen inside the nested
    Analyst call specifically, to prove it propagates all the way back up
    through two levels of Agent tool-dispatch to Orchestrator.handle() —
    not just that an overflow at the top level is caught.
    """
    analyst_client = FakeOpenAIClient(
        [
            FakeResponse.tool_call(
                "submit_answer",
                {"answer": "Q1 revenue was $10 million.", "citations": [], "calculations": []},
                total_tokens=100,
            )
        ]
    )
    top_level_client = FakeOpenAIClient(
        [
            FakeResponse.tool_call("search_documents", {"query": "Q1 revenue"}, total_tokens=20),
            FakeResponse.tool_call(
                "analyze", {"question": "What was Q1 revenue?", "context": _RETRIEVED_CONTEXT}, total_tokens=20
            ),
        ]
    )
    audit_log = InMemoryAuditLog()
    user = User(username="tester", role=Role.ANALYST, org_id="org-a")
    orchestrator = Orchestrator(
        user=user,
        audit_log=audit_log,
        client=top_level_client,
        retriever_agent=_FakeRetrieverAgent(_RETRIEVED_CONTEXT),
        analyst_agent=AnalystAgent(client=analyst_client),
        compliance_agent=ComplianceAgent(semantic_review=False),
        max_tokens_per_request=100,
    )

    result = orchestrator.handle("What was Q1 revenue?")

    assert result.blocked is True
    assert result.block_reason == "token_budget_exceeded"
    failure_entries = [e for e in audit_log.entries() if e.action == "agent_failure"]
    assert len(failure_entries) == 1
    assert "budget" in failure_entries[0].details["error"].lower()


def test_blocked_response_is_enqueued_for_human_review() -> None:
    from finvault.security.review_queue import InMemoryReviewQueue

    review_queue = InMemoryReviewQueue()
    analyst_client = FakeOpenAIClient(
        [
            FakeResponse.tool_call(
                "submit_answer",
                {
                    "answer": "Net income was $50 million.",
                    "citations": [{"document": "Q1 Report", "quoted_text": "Net income was $50 million"}],
                    "calculations": [],
                },
            )
        ]
    )
    top_level_client = FakeOpenAIClient(
        [
            FakeResponse.tool_call("search_documents", {"query": "net income"}),
            FakeResponse.tool_call("analyze", {"question": "What was net income?", "context": _RETRIEVED_CONTEXT}),
        ]
    )
    audit_log = InMemoryAuditLog()
    user = User(username="tester", role=Role.ANALYST, org_id="org-a")
    orchestrator = Orchestrator(
        user=user,
        audit_log=audit_log,
        client=top_level_client,
        retriever_agent=_FakeRetrieverAgent(_RETRIEVED_CONTEXT),
        analyst_agent=AnalystAgent(client=analyst_client),
        compliance_agent=ComplianceAgent(semantic_review=False),
        review_queue=review_queue,
    )

    result = orchestrator.handle("What was net income?")

    assert result.blocked is True
    assert result.review_item_id is not None

    pending = review_queue.list_pending(org_id="org-a")
    assert len(pending) == 1
    item = pending[0]
    assert item.id == result.review_item_id
    assert item.question == "What was net income?"
    # The raw, unredacted draft is visible to the reviewer even though the
    # end user only ever saw the generic block message.
    assert item.draft_answer == "Net income was $50 million."
    assert item.citations == [{"document": "Q1 Report", "quoted_text": "Net income was $50 million"}]
    assert item.block_reason is not None


def test_orchestrator_blocks_answer_with_a_hallucinated_citation() -> None:
    """The citation's quoted_text doesn't appear anywhere in the context
    that was actually retrieved and handed to the Analyst — this must be
    caught and blocked, not passed through as a grounded claim.
    """
    analyst_client = FakeOpenAIClient(
        [
            FakeResponse.tool_call(
                "submit_answer",
                {
                    "answer": "Net income was $50 million.",
                    "citations": [{"document": "Q1 Report", "quoted_text": "Net income was $50 million"}],
                    "calculations": [],
                },
            )
        ]
    )
    orchestrator, audit_log = _make_orchestrator_with_injected_agents(analyst_client=analyst_client)

    result = orchestrator.handle("What was net income?")

    assert result.blocked is True
    assert result.block_reason is not None
    assert "citation" in result.block_reason.lower()
    review_entries = [e for e in audit_log.entries() if e.action == "compliance_review"]
    assert review_entries[-1].details["allowed"] is False
