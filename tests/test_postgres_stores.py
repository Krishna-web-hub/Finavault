"""Integration tests for the Postgres-backed SessionStore and ReviewQueue.

Runs against a real Postgres instance (see docker-compose.yml) rather than
mocking SQLAlchemy — the whole point of these classes is correct SQL and
transaction behavior, which a mock can't verify. Skips cleanly if Postgres
isn't reachable, so the suite stays portable to a checkout without `docker
compose up`.

Every test uses a fresh uuid4 for session_id/org_id/item ids, so tests never
collide with each other or with any other data in this (shared, dev-only)
database — no truncation or cleanup step needed.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from finvault.agents.session import PostgresSessionStore
from finvault.config import settings
from finvault.db import init_db
from finvault.retrieval.graph_store import PostgresGraphStore, edge_aad, label_hash, node_aad
from finvault.security.encryption import EnvelopeEncryptor, LocalKeyProvider
from finvault.security.quarantine import PostgresQuarantineStore
from finvault.security.review_queue import PostgresReviewQueue, ReviewQueueError


@pytest.fixture(scope="module")
def engine():
    engine = create_engine(settings.postgres_dsn, future=True)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except OperationalError:
        pytest.skip(f"Postgres not reachable at {settings.postgres_dsn!r} — skipping Postgres-backed store tests")
    init_db(engine)
    return engine


# --- PostgresSessionStore ---


def test_append_and_get_history_roundtrip(engine) -> None:
    store = PostgresSessionStore(engine)
    session_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())

    store.append_turn(session_id=session_id, user_id=user_id, question="Q1", answer="A1")
    store.append_turn(session_id=session_id, user_id=user_id, question="Q2", answer="A2")

    history = store.get_history(session_id=session_id, user_id=user_id)

    assert [t.question for t in history] == ["Q1", "Q2"]
    assert [t.answer for t in history] == ["A1", "A2"]


def test_get_history_empty_for_unknown_session(engine) -> None:
    store = PostgresSessionStore(engine)
    assert store.get_history(session_id=str(uuid.uuid4()), user_id=str(uuid.uuid4())) == []


def test_history_isolated_between_users_for_the_same_session_id(engine) -> None:
    store = PostgresSessionStore(engine)
    session_id = str(uuid.uuid4())
    user_a, user_b = str(uuid.uuid4()), str(uuid.uuid4())

    store.append_turn(session_id=session_id, user_id=user_a, question="Q1", answer="A1")

    assert store.get_history(session_id=session_id, user_id=user_b) == []
    assert len(store.get_history(session_id=session_id, user_id=user_a)) == 1


def test_history_survives_a_fresh_engine_connection(engine) -> None:
    """Proves this is actually persisted server-side, not just cached in
    the SQLAlchemy Engine object — a second, independent engine pointed at
    the same database must see what the first one wrote.
    """
    session_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    PostgresSessionStore(engine).append_turn(session_id=session_id, user_id=user_id, question="Q1", answer="A1")

    second_engine = create_engine(settings.postgres_dsn, future=True)
    try:
        history = PostgresSessionStore(second_engine).get_history(session_id=session_id, user_id=user_id)
    finally:
        second_engine.dispose()

    assert len(history) == 1
    assert history[0].question == "Q1"


# --- PostgresReviewQueue ---


def _enqueue(queue: PostgresReviewQueue, *, org_id: str):
    return queue.enqueue(
        org_id=org_id,
        user_id=str(uuid.uuid4()),
        question="What was net income?",
        draft_answer="Net income was $50 million.",
        block_reason="citation verification failed",
        findings=[],
        citations=[{"document": "Q1 Report", "quoted_text": "Net income was $50 million"}],
    )


def test_enqueue_then_list_pending(engine) -> None:
    queue = PostgresReviewQueue(engine)
    org_id = str(uuid.uuid4())

    item = _enqueue(queue, org_id=org_id)

    pending = queue.list_pending(org_id=org_id)
    assert len(pending) == 1
    assert pending[0].id == item.id
    assert pending[0].status == "pending"
    assert pending[0].citations == [{"document": "Q1 Report", "quoted_text": "Net income was $50 million"}]


def test_list_pending_is_org_scoped(engine) -> None:
    queue = PostgresReviewQueue(engine)
    org_a, org_b = str(uuid.uuid4()), str(uuid.uuid4())

    _enqueue(queue, org_id=org_a)
    _enqueue(queue, org_id=org_b)

    assert len(queue.list_pending(org_id=org_a)) == 1
    assert len(queue.list_pending(org_id=org_b)) == 1


def test_resolve_released_removes_item_from_pending_list(engine) -> None:
    queue = PostgresReviewQueue(engine)
    org_id = str(uuid.uuid4())
    item = _enqueue(queue, org_id=org_id)

    resolved = queue.resolve(item.id, org_id=org_id, status="released", reviewed_by="officer-1", reviewer_note="ok")

    assert resolved.status == "released"
    assert resolved.reviewed_by == "officer-1"
    assert resolved.reviewer_note == "ok"
    assert queue.list_pending(org_id=org_id) == []
    assert queue.get(item.id).status == "released"


def test_resolve_denied_is_recorded(engine) -> None:
    queue = PostgresReviewQueue(engine)
    org_id = str(uuid.uuid4())
    item = _enqueue(queue, org_id=org_id)

    queue.resolve(item.id, org_id=org_id, status="denied", reviewed_by="officer-1")

    assert queue.get(item.id).status == "denied"


def test_resolve_unknown_item_raises(engine) -> None:
    queue = PostgresReviewQueue(engine)
    with pytest.raises(ReviewQueueError):
        queue.resolve(str(uuid.uuid4()), org_id=str(uuid.uuid4()), status="released", reviewed_by="officer-1")


def test_resolve_fails_for_wrong_org_same_as_missing_item(engine) -> None:
    queue = PostgresReviewQueue(engine)
    org_a, org_b = str(uuid.uuid4()), str(uuid.uuid4())
    item = _enqueue(queue, org_id=org_a)

    with pytest.raises(ReviewQueueError):
        queue.resolve(item.id, org_id=org_b, status="released", reviewed_by="officer-1")


def test_resolve_to_pending_is_rejected(engine) -> None:
    queue = PostgresReviewQueue(engine)
    org_id = str(uuid.uuid4())
    item = _enqueue(queue, org_id=org_id)

    with pytest.raises(ReviewQueueError):
        queue.resolve(item.id, org_id=org_id, status="pending", reviewed_by="officer-1")


# --- PostgresGraphStore ---


@pytest.fixture(scope="module")
def encryptor(tmp_path_factory):
    key_provider = LocalKeyProvider(tmp_path_factory.mktemp("keys") / "master.key")
    return EnvelopeEncryptor(key_provider)


def test_upsert_node_persists_and_dedupes_across_a_fresh_connection(engine, encryptor) -> None:
    store = PostgresGraphStore(engine)
    org_id = str(uuid.uuid4())
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

    second_store = PostgresGraphStore(create_engine(settings.postgres_dsn, future=True))
    second = second_store.upsert_node(
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


def test_get_nodes_and_edges_is_org_scoped(engine, encryptor) -> None:
    store = PostgresGraphStore(engine)
    org_a, org_b = str(uuid.uuid4()), str(uuid.uuid4())

    h_a = label_hash(org_a, "company", "Acme A")
    store.upsert_node(
        org_id=org_a,
        type="company",
        classification="internal",
        source_document_id="doc-1",
        node_label_hash=h_a,
        label_encrypted=encryptor.encrypt("Acme A", aad=node_aad(org_a, "company", h_a)),
    )
    h_b = label_hash(org_b, "company", "Acme B")
    store.upsert_node(
        org_id=org_b,
        type="company",
        classification="internal",
        source_document_id="doc-1",
        node_label_hash=h_b,
        label_encrypted=encryptor.encrypt("Acme B", aad=node_aad(org_b, "company", h_b)),
    )

    nodes_a, _ = store.get_nodes_and_edges(org_id=org_a)
    nodes_b, _ = store.get_nodes_and_edges(org_id=org_b)
    assert len(nodes_a) == 1
    assert len(nodes_b) == 1


def test_add_edge_persists_and_round_trips(engine, encryptor) -> None:
    store = PostgresGraphStore(engine)
    org_id = str(uuid.uuid4())
    h_source = label_hash(org_id, "company", "Acme Capital")
    source = store.upsert_node(
        org_id=org_id,
        type="company",
        classification="internal",
        source_document_id="doc-1",
        node_label_hash=h_source,
        label_encrypted=encryptor.encrypt("Acme Capital", aad=node_aad(org_id, "company", h_source)),
    )
    h_target = label_hash(org_id, "metric", "Q3 Revenue")
    target = store.upsert_node(
        org_id=org_id,
        type="metric",
        classification="internal",
        source_document_id="doc-1",
        node_label_hash=h_target,
        label_encrypted=encryptor.encrypt("Q3 Revenue", aad=node_aad(org_id, "metric", h_target)),
    )
    edge_id = str(uuid.uuid4())
    store.add_edge(
        id=edge_id,
        org_id=org_id,
        source_node_id=source.id,
        target_node_id=target.id,
        classification="internal",
        relation_encrypted=encryptor.encrypt("reported", aad=edge_aad(edge_id)),
    )

    _, edges = store.get_nodes_and_edges(org_id=org_id)
    assert len(edges) == 1
    assert edges[0].id == edge_id
    assert edges[0].source_node_id == source.id
    assert edges[0].target_node_id == target.id


# --- PostgresQuarantineStore ---
#
# The in-memory store (tests/test_quarantine.py) can't cover these: what's
# under test is the SQL itself — the insert-vs-update branch, the
# rowcount==0 path on release, and the org predicate on every statement.


def test_quarantine_roundtrip_and_org_isolation(engine) -> None:
    store = PostgresQuarantineStore(engine)
    org_id = str(uuid.uuid4())
    other_org = str(uuid.uuid4())
    document_id = str(uuid.uuid4())

    record = store.quarantine(document_id=document_id, org_id=org_id, reason="injection", actor="officer")

    assert record.status == "quarantined"
    assert store.quarantined_ids(org_id=org_id) == {document_id}
    assert store.quarantined_ids(org_id=other_org) == set()
    assert [r.document_id for r in store.list_quarantined(org_id=org_id)] == [document_id]


def test_release_clears_the_document_from_retrieval_exclusion(engine) -> None:
    store = PostgresQuarantineStore(engine)
    org_id = str(uuid.uuid4())
    document_id = str(uuid.uuid4())
    store.quarantine(document_id=document_id, org_id=org_id, reason=None, actor="officer")

    released = store.release(document_id=document_id, org_id=org_id, actor="officer")

    assert released is not None
    assert released.status == "released"
    assert released.released_by == "officer"
    assert store.quarantined_ids(org_id=org_id) == set()


def test_release_of_an_unknown_document_returns_none(engine) -> None:
    store = PostgresQuarantineStore(engine)
    assert store.release(document_id=str(uuid.uuid4()), org_id=str(uuid.uuid4()), actor="officer") is None


def test_another_org_cannot_release_our_quarantine(engine) -> None:
    store = PostgresQuarantineStore(engine)
    org_id = str(uuid.uuid4())
    document_id = str(uuid.uuid4())
    store.quarantine(document_id=document_id, org_id=org_id, reason=None, actor="officer")

    assert store.release(document_id=document_id, org_id=str(uuid.uuid4()), actor="intruder") is None
    assert store.quarantined_ids(org_id=org_id) == {document_id}


def test_requarantine_after_release_clears_stale_release_metadata(engine) -> None:
    """The UPDATE branch. Without explicitly nulling released_by/released_at,
    a re-quarantined row would still carry the previous release's metadata
    and read as released to anyone inspecting the table.
    """
    store = PostgresQuarantineStore(engine)
    org_id = str(uuid.uuid4())
    document_id = str(uuid.uuid4())
    store.quarantine(document_id=document_id, org_id=org_id, reason="first", actor="officer")
    store.release(document_id=document_id, org_id=org_id, actor="officer")

    store.quarantine(document_id=document_id, org_id=org_id, reason="second", actor="officer2")

    record = store.list_quarantined(org_id=org_id)[0]
    assert record.released_at is None
    assert record.released_by is None
    assert record.reason == "second"
    assert record.quarantined_by == "officer2"
