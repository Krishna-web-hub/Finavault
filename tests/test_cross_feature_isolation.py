"""Cross-Feature Security & Multi-Tenant Isolation Test Suite.

Verifies multi-tenant isolation (`org_id`) and RBAC clearance across all 3
exclusive canvas features combined:
1. Knowledge Graph Canvas (`GraphRetriever` & `GET /graph`)
2. Real-Time DAG Execution Traces (`Orchestrator` & `execution_steps`)
3. Multi-Document Risk Heatmap (`ComparisonAgent` & `POST /documents/compare`)
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from finvault.agents.analyst_agent import AnalystAgent
from finvault.agents.compliance_agent import ComplianceAgent
from finvault.agents.orchestrator import Orchestrator
from finvault.api.routes import CompareRequest, compare_documents, get_graph
from finvault.errors import NotFoundError
from finvault.ingestion.pipeline import IngestionPipeline
from finvault.models import Classification, Role, User
from finvault.retrieval.graph_retriever import GraphRetriever
from finvault.retrieval.graph_store import InMemoryGraphStore, label_hash, node_aad
from finvault.retrieval.retriever import Retriever
from finvault.retrieval.vector_store import InMemoryVectorStore
from finvault.security.audit import InMemoryAuditLog
from finvault.security.encryption import EnvelopeEncryptor, LocalKeyProvider
from tests.fakes import FakeEmbeddingProvider, FakeOpenAIClient, FakeResponse
from tests.test_orchestrator import _FakeRetrieverAgent


def _setup_multi_tenant_env(tmp_path):
    key_provider = LocalKeyProvider(tmp_path / "master.key")
    encryptor = EnvelopeEncryptor(key_provider)
    embedding_provider = FakeEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    graph_store = InMemoryGraphStore()
    audit_log = InMemoryAuditLog()

    pipeline = IngestionPipeline(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        encryptor=encryptor,
        audit_log=audit_log,
        graph_store=graph_store,
    )
    retriever = Retriever(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        encryptor=encryptor,
        audit_log=audit_log,
    )
    graph_retriever = GraphRetriever(graph_store=graph_store, encryptor=encryptor)

    # Ingest document for Org A (Confidential)
    file_a = tmp_path / "doc_a.txt"
    file_a.write_text("Acme Capital Q3 revenue was $10M.")
    doc_a = pipeline.ingest_file(file_a, org_id="org-a", classification=Classification.CONFIDENTIAL)

    # Ingest document for Org B (Restricted)
    file_b = tmp_path / "doc_b.txt"
    file_b.write_text("Beta Corp Q3 revenue was $50M.")
    doc_b = pipeline.ingest_file(file_b, org_id="org-b", classification=Classification.RESTRICTED)

    # Insert explicit node into graph_store for Org B
    hb = label_hash("org-b", "company", "Beta Corp")
    graph_store.upsert_node(
        org_id="org-b",
        type="company",
        classification="restricted",
        source_document_id=doc_b.id,
        node_label_hash=hb,
        label_encrypted=encryptor.encrypt("Beta Corp", aad=node_aad("org-b", "company", hb)),
    )

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                retriever=retriever,
                graph_retriever=graph_retriever,
                audit_log=audit_log,
                review_queue=None,
                session_store=None,
            )
        )
    )

    return request, doc_a, doc_b, graph_retriever, audit_log, encryptor


def test_cross_feature_isolation_org_scoping_on_graph(tmp_path) -> None:
    request, _doc_a, _doc_b, _graph_retriever, _audit_log, _encryptor = _setup_multi_tenant_env(tmp_path)

    user_org_a = User(username="alice", role=Role.ANALYST, org_id="org-a")
    user_org_b = User(username="bob", role=Role.ADMIN, org_id="org-b")

    # User in Org A gets graph nodes only from Org A
    res_a = get_graph(request, user=user_org_a)
    assert not any(n.label == "Beta Corp" for n in res_a.nodes)

    # User in Org B sees Beta Corp
    res_b = get_graph(request, user=user_org_b)
    assert any(n.label == "Beta Corp" for n in res_b.nodes)


def test_cross_feature_isolation_rbac_clearance_on_graph(tmp_path) -> None:
    request, _doc_a, _doc_b, _graph_retriever, _audit_log, _encryptor = _setup_multi_tenant_env(tmp_path)

    # Viewer in Org B lacks clearance for RESTRICTED nodes
    viewer_org_b = User(username="bob_viewer", role=Role.VIEWER, org_id="org-b")
    res_viewer = get_graph(request, user=viewer_org_b)
    assert not any(n.label == "Beta Corp" for n in res_viewer.nodes)

    # Admin in Org B has clearance for RESTRICTED nodes
    admin_org_b = User(username="bob_admin", role=Role.ADMIN, org_id="org-b")
    res_admin = get_graph(request, user=admin_org_b)
    assert any(n.label == "Beta Corp" for n in res_admin.nodes)


def test_cross_feature_isolation_comparison_heatmap_org_leakage(tmp_path) -> None:
    request, doc_a, doc_b, _graph_retriever, _audit_log, _encryptor = _setup_multi_tenant_env(tmp_path)

    user_org_a = User(username="alice", role=Role.ANALYST, org_id="org-a")

    # User in Org A trying to compare Doc A (Org A) and Doc B (Org B)
    # Doc B is silently dropped -> less than 2 accessible docs -> returns 404
    with pytest.raises(NotFoundError) as exc_info:
        compare_documents(CompareRequest(document_ids=[doc_a.id, doc_b.id]), request, user=user_org_a)
    assert exc_info.value.http_status == 404


def test_cross_feature_isolation_execution_dag_trace_unredacted_previews(tmp_path) -> None:
    _request, _doc_a, _doc_b, _graph_retriever, audit_log, _encryptor = _setup_multi_tenant_env(tmp_path)

    user_org_a = User(username="alice", role=Role.ANALYST, org_id="org-a")
    retrieved_context = "Contact john.doe@example.com for Q3 filing."
    retriever_agent = _FakeRetrieverAgent(retrieved_context)

    top_level_client = FakeOpenAIClient(
        [
            FakeResponse.tool_call("search_documents", {"query": "Q3 filing"}),
            FakeResponse.tool_call("analyze", {"question": "Who do I contact?", "context": retrieved_context}),
        ]
    )
    analyst_client = FakeOpenAIClient(
        [
            FakeResponse.tool_call(
                "submit_answer",
                {"answer": "Contact john.doe@example.com for the Q1 filing.", "citations": []},
            )
        ]
    )
    compliance_agent = ComplianceAgent(semantic_review=False)

    orchestrator = Orchestrator(
        user=user_org_a,
        audit_log=audit_log,
        client=top_level_client,
        retriever_agent=retriever_agent,
        analyst_agent=AnalystAgent(client=analyst_client),
        compliance_agent=compliance_agent,
    )

    result = orchestrator.handle("Who do I contact?")

    # Verify execution DAG steps were logged
    assert len(result.execution_steps) >= 2

    # Verify step preview for Analyst was sanitized and does not contain raw email
    analyst_step = next(s for s in result.execution_steps if s.agent_name == "analyst")
    assert "john.doe@example.com" not in analyst_step.payload_preview
    assert "[REDACTED:EMAIL]" in analyst_step.payload_preview
