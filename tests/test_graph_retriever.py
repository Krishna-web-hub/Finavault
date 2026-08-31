from __future__ import annotations

from finvault.models import Role, User
from finvault.retrieval.graph_retriever import GraphRetriever
from finvault.retrieval.graph_store import InMemoryGraphStore, edge_aad, label_hash, node_aad
from finvault.security.encryption import EnvelopeEncryptor, LocalKeyProvider


def _make_encryptor(tmp_path):
    return EnvelopeEncryptor(LocalKeyProvider(tmp_path / "master.key"))


def _add_node(store, encryptor, *, org_id, type, label, classification, doc_id="doc-1"):
    h = label_hash(org_id, type, label)
    encrypted = encryptor.encrypt(label, aad=node_aad(org_id, type, h))
    return store.upsert_node(
        org_id=org_id,
        type=type,
        classification=classification,
        source_document_id=doc_id,
        node_label_hash=h,
        label_encrypted=encrypted,
    )


def _add_edge(store, encryptor, *, org_id, source, target, relation, classification, edge_id):
    encrypted = encryptor.encrypt(relation, aad=edge_aad(edge_id))
    return store.add_edge(
        id=edge_id,
        org_id=org_id,
        source_node_id=source.id,
        target_node_id=target.id,
        classification=classification,
        relation_encrypted=encrypted,
    )


def test_viewer_sees_public_and_internal_nodes_with_decrypted_labels(tmp_path) -> None:
    store = InMemoryGraphStore()
    encryptor = _make_encryptor(tmp_path)
    _add_node(store, encryptor, org_id="org-a", type="company", label="Acme Capital", classification="internal")

    retriever = GraphRetriever(graph_store=store, encryptor=encryptor)
    viewer = User(username="v", role=Role.VIEWER, org_id="org-a")

    graph = retriever.get_graph(user=viewer)

    assert len(graph.nodes) == 1
    assert graph.nodes[0].label == "Acme Capital"
    assert graph.nodes[0].type == "company"


def test_viewer_does_not_see_restricted_nodes(tmp_path) -> None:
    """Non-inference: a viewer must not learn a restricted entity exists at
    all — silent drop, not a denial (see graph_retriever.py's module docstring).
    """
    store = InMemoryGraphStore()
    encryptor = _make_encryptor(tmp_path)
    _add_node(
        store, encryptor, org_id="org-a", type="risk", label="Suspicious Activity Alert", classification="restricted"
    )

    retriever = GraphRetriever(graph_store=store, encryptor=encryptor)
    viewer = User(username="v", role=Role.VIEWER, org_id="org-a")

    graph = retriever.get_graph(user=viewer)

    assert graph.nodes == []


def test_compliance_officer_sees_restricted_nodes(tmp_path) -> None:
    store = InMemoryGraphStore()
    encryptor = _make_encryptor(tmp_path)
    _add_node(
        store, encryptor, org_id="org-a", type="risk", label="Suspicious Activity Alert", classification="restricted"
    )

    retriever = GraphRetriever(graph_store=store, encryptor=encryptor)
    officer = User(username="o", role=Role.COMPLIANCE_OFFICER, org_id="org-a")

    graph = retriever.get_graph(user=officer)

    assert len(graph.nodes) == 1
    assert graph.nodes[0].label == "Suspicious Activity Alert"


def test_edge_dropped_when_an_endpoint_node_is_above_clearance(tmp_path) -> None:
    """Even though the edge itself is internal, one endpoint is restricted
    and thus invisible to a viewer — showing the edge would leak that the
    hidden node exists.
    """
    store = InMemoryGraphStore()
    encryptor = _make_encryptor(tmp_path)
    company = _add_node(
        store, encryptor, org_id="org-a", type="company", label="Acme Capital", classification="internal"
    )
    risk = _add_node(
        store, encryptor, org_id="org-a", type="risk", label="Suspicious Activity Alert", classification="restricted"
    )
    _add_edge(
        store,
        encryptor,
        org_id="org-a",
        source=company,
        target=risk,
        relation="flagged_by",
        classification="internal",
        edge_id="edge-1",
    )

    retriever = GraphRetriever(graph_store=store, encryptor=encryptor)
    viewer = User(username="v", role=Role.VIEWER, org_id="org-a")

    graph = retriever.get_graph(user=viewer)

    assert len(graph.nodes) == 1
    assert graph.edges == []


def test_edge_dropped_when_its_own_classification_exceeds_clearance_even_if_both_endpoints_visible(tmp_path) -> None:
    """A relationship can be more sensitive than either endpoint alone (see
    db.py's comment on graph_edges.classification) — must be filtered on its
    own classification, not inherited from the nodes.
    """
    store = InMemoryGraphStore()
    encryptor = _make_encryptor(tmp_path)
    company = _add_node(
        store, encryptor, org_id="org-a", type="company", label="Acme Capital", classification="internal"
    )
    regulator = _add_node(
        store, encryptor, org_id="org-a", type="company", label="Regulator Y", classification="internal"
    )
    _add_edge(
        store,
        encryptor,
        org_id="org-a",
        source=company,
        target=regulator,
        relation="under_investigation_by",
        classification="restricted",
        edge_id="edge-1",
    )

    retriever = GraphRetriever(graph_store=store, encryptor=encryptor)
    viewer = User(username="v", role=Role.VIEWER, org_id="org-a")

    graph = retriever.get_graph(user=viewer)

    assert len(graph.nodes) == 2
    assert graph.edges == []


def test_visible_edge_has_decrypted_relation_label(tmp_path) -> None:
    store = InMemoryGraphStore()
    encryptor = _make_encryptor(tmp_path)
    company = _add_node(
        store, encryptor, org_id="org-a", type="company", label="Acme Capital", classification="internal"
    )
    metric = _add_node(store, encryptor, org_id="org-a", type="metric", label="Q3 Revenue", classification="internal")
    _add_edge(
        store,
        encryptor,
        org_id="org-a",
        source=company,
        target=metric,
        relation="reported",
        classification="internal",
        edge_id="edge-1",
    )

    retriever = GraphRetriever(graph_store=store, encryptor=encryptor)
    analyst = User(username="a", role=Role.ANALYST, org_id="org-a")

    graph = retriever.get_graph(user=analyst)

    assert len(graph.edges) == 1
    assert graph.edges[0].label == "reported"
    assert graph.edges[0].source == company.id
    assert graph.edges[0].target == metric.id


def test_graph_is_org_scoped(tmp_path) -> None:
    store = InMemoryGraphStore()
    encryptor = _make_encryptor(tmp_path)
    _add_node(store, encryptor, org_id="org-a", type="company", label="Acme Capital", classification="internal")
    _add_node(store, encryptor, org_id="org-b", type="company", label="Other Corp", classification="internal")

    retriever = GraphRetriever(graph_store=store, encryptor=encryptor)
    analyst = User(username="a", role=Role.ANALYST, org_id="org-a")

    graph = retriever.get_graph(user=analyst)

    assert len(graph.nodes) == 1
    assert graph.nodes[0].label == "Acme Capital"
