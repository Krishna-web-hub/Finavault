from __future__ import annotations

from finvault.retrieval.graph_store import InMemoryGraphStore, edge_aad, label_hash, node_aad
from finvault.security.encryption import EnvelopeEncryptor, LocalKeyProvider


def _make_encryptor(tmp_path):
    return EnvelopeEncryptor(LocalKeyProvider(tmp_path / "master.key"))


def test_upsert_node_creates_a_new_node_for_a_new_label(tmp_path) -> None:
    store = InMemoryGraphStore()
    encryptor = _make_encryptor(tmp_path)
    org_id = "org-a"
    h = label_hash(org_id, "company", "Acme Capital")
    encrypted = encryptor.encrypt("Acme Capital", aad=node_aad(org_id, "company", h))

    node = store.upsert_node(
        org_id=org_id,
        type="company",
        classification="internal",
        source_document_id="doc-1",
        node_label_hash=h,
        label_encrypted=encrypted,
    )

    nodes, _ = store.get_nodes_and_edges(org_id=org_id)
    assert len(nodes) == 1
    assert nodes[0].id == node.id


def test_upsert_node_dedupes_exact_match_on_org_type_and_label(tmp_path) -> None:
    store = InMemoryGraphStore()
    encryptor = _make_encryptor(tmp_path)
    org_id = "org-a"
    h = label_hash(org_id, "company", "Acme Capital")
    encrypted = encryptor.encrypt("Acme Capital", aad=node_aad(org_id, "company", h))

    first = store.upsert_node(
        org_id=org_id,
        type="company",
        classification="internal",
        source_document_id="doc-1",
        node_label_hash=h,
        label_encrypted=encrypted,
    )
    second = store.upsert_node(
        org_id=org_id,
        type="company",
        classification="internal",
        source_document_id="doc-2",
        node_label_hash=h,
        label_encrypted=encrypted,
    )

    assert first.id == second.id
    nodes, _ = store.get_nodes_and_edges(org_id=org_id)
    assert len(nodes) == 1


def test_upsert_node_does_not_dedupe_across_different_entity_types(tmp_path) -> None:
    """ "Acme" as a company and "Acme" as a metric label are different nodes
    — dedup is scoped to (org_id, type, label_hash), not label alone.
    """
    store = InMemoryGraphStore()
    encryptor = _make_encryptor(tmp_path)
    org_id = "org-a"
    h_company = label_hash(org_id, "company", "Acme")
    h_metric = label_hash(org_id, "metric", "Acme")

    store.upsert_node(
        org_id=org_id,
        type="company",
        classification="internal",
        source_document_id="doc-1",
        node_label_hash=h_company,
        label_encrypted=encryptor.encrypt("Acme", aad=node_aad(org_id, "company", h_company)),
    )
    store.upsert_node(
        org_id=org_id,
        type="metric",
        classification="internal",
        source_document_id="doc-1",
        node_label_hash=h_metric,
        label_encrypted=encryptor.encrypt("Acme", aad=node_aad(org_id, "metric", h_metric)),
    )

    nodes, _ = store.get_nodes_and_edges(org_id=org_id)
    assert len(nodes) == 2


def test_upsert_node_does_not_dedupe_across_orgs(tmp_path) -> None:
    store = InMemoryGraphStore()
    encryptor = _make_encryptor(tmp_path)

    for org_id in ("org-a", "org-b"):
        h = label_hash(org_id, "company", "Acme Capital")
        store.upsert_node(
            org_id=org_id,
            type="company",
            classification="internal",
            source_document_id="doc-1",
            node_label_hash=h,
            label_encrypted=encryptor.encrypt("Acme Capital", aad=node_aad(org_id, "company", h)),
        )

    nodes_a, _ = store.get_nodes_and_edges(org_id="org-a")
    nodes_b, _ = store.get_nodes_and_edges(org_id="org-b")
    assert len(nodes_a) == 1
    assert len(nodes_b) == 1
    assert nodes_a[0].id != nodes_b[0].id


def test_get_nodes_and_edges_is_org_scoped(tmp_path) -> None:
    store = InMemoryGraphStore()
    encryptor = _make_encryptor(tmp_path)

    h_a = label_hash("org-a", "company", "Acme A")
    node_a = store.upsert_node(
        org_id="org-a",
        type="company",
        classification="internal",
        source_document_id="doc-1",
        node_label_hash=h_a,
        label_encrypted=encryptor.encrypt("Acme A", aad=node_aad("org-a", "company", h_a)),
    )
    h_b = label_hash("org-b", "company", "Acme B")
    node_b = store.upsert_node(
        org_id="org-b",
        type="company",
        classification="internal",
        source_document_id="doc-1",
        node_label_hash=h_b,
        label_encrypted=encryptor.encrypt("Acme B", aad=node_aad("org-b", "company", h_b)),
    )
    store.add_edge(
        id="edge-a",
        org_id="org-a",
        source_node_id=node_a.id,
        target_node_id=node_a.id,
        classification="internal",
        relation_encrypted=encryptor.encrypt("self_ref", aad=edge_aad("edge-a")),
    )
    store.add_edge(
        id="edge-b",
        org_id="org-b",
        source_node_id=node_b.id,
        target_node_id=node_b.id,
        classification="internal",
        relation_encrypted=encryptor.encrypt("self_ref", aad=edge_aad("edge-b")),
    )

    nodes_a, edges_a = store.get_nodes_and_edges(org_id="org-a")
    assert [n.id for n in nodes_a] == [node_a.id]
    assert [e.id for e in edges_a] == ["edge-a"]


def test_upsert_node_fuzzy_dedupes_variations_of_same_entity(tmp_path) -> None:
    store = InMemoryGraphStore()
    encryptor = _make_encryptor(tmp_path)
    org_id = "org-a"

    # Both "Acme Capital Inc." and "Acme Capital, LLC" map to the same normalized label hash
    h1 = label_hash(org_id, "company", "Acme Capital Inc.")
    h2 = label_hash(org_id, "company", "Acme Capital, LLC")
    assert h1 == h2

    first = store.upsert_node(
        org_id=org_id,
        type="company",
        classification="internal",
        source_document_id="doc-1",
        node_label_hash=h1,
        label_encrypted=encryptor.encrypt("Acme Capital Inc.", aad=node_aad(org_id, "company", h1)),
    )
    second = store.upsert_node(
        org_id=org_id,
        type="company",
        classification="internal",
        source_document_id="doc-2",
        node_label_hash=h2,
        label_encrypted=encryptor.encrypt("Acme Capital, LLC", aad=node_aad(org_id, "company", h2)),
    )

    assert first.id == second.id
    nodes, _ = store.get_nodes_and_edges(org_id=org_id)
    assert len(nodes) == 1
