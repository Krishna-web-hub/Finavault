"""Test for GET /graph's route handler.

No FastAPI TestClient/HTTP-level test here, matching the rest of this
codebase's route-testing approach (see test_query_stream.py's docstring) —
main.py's lifespan connects to real Qdrant/Postgres and would make this a
heavier integration test than the route's own (trivial) logic warrants. The
route is a thin delegation to GraphRetriever.get_graph(user=...), already
covered thoroughly in isolation by test_graph_retriever.py — what's worth
testing here is that the route actually wires request.app.state.graph_retriever
through to that call rather than, say, forgetting the org-scoping `user` arg.
"""

from __future__ import annotations

from types import SimpleNamespace

from finvault.api.routes import get_graph
from finvault.models import Role, User
from finvault.retrieval.graph_retriever import GraphRetriever
from finvault.retrieval.graph_store import InMemoryGraphStore, label_hash, node_aad
from finvault.security.encryption import EnvelopeEncryptor, LocalKeyProvider


def _fake_request(graph_retriever: GraphRetriever) -> SimpleNamespace:
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(graph_retriever=graph_retriever)))


def test_get_graph_route_delegates_to_the_graph_retriever_for_the_caller(tmp_path) -> None:
    encryptor = EnvelopeEncryptor(LocalKeyProvider(tmp_path / "master.key"))
    graph_store = InMemoryGraphStore()
    h = label_hash("org-a", "company", "Acme Capital")
    graph_store.upsert_node(
        org_id="org-a",
        type="company",
        classification="internal",
        source_document_id="doc-1",
        node_label_hash=h,
        label_encrypted=encryptor.encrypt("Acme Capital", aad=node_aad("org-a", "company", h)),
    )
    graph_retriever = GraphRetriever(graph_store=graph_store, encryptor=encryptor)
    user = User(username="a", role=Role.ANALYST, org_id="org-a")

    result = get_graph(_fake_request(graph_retriever), user=user)

    assert [n.label for n in result.nodes] == ["Acme Capital"]


def test_get_graph_route_is_org_scoped(tmp_path) -> None:
    encryptor = EnvelopeEncryptor(LocalKeyProvider(tmp_path / "master.key"))
    graph_store = InMemoryGraphStore()
    h = label_hash("org-b", "company", "Other Corp")
    graph_store.upsert_node(
        org_id="org-b",
        type="company",
        classification="internal",
        source_document_id="doc-1",
        node_label_hash=h,
        label_encrypted=encryptor.encrypt("Other Corp", aad=node_aad("org-b", "company", h)),
    )
    graph_retriever = GraphRetriever(graph_store=graph_store, encryptor=encryptor)
    user = User(username="a", role=Role.ANALYST, org_id="org-a")

    result = get_graph(_fake_request(graph_retriever), user=user)

    assert result.nodes == []
