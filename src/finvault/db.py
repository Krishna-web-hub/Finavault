"""Shared PostgreSQL schema and engine.

Tables: `documents` (metadata + classification — the ACL primitive),
`audit_log` (the hash-chained, append-only audit trail), `sessions`
(conversation memory — see agents/session.py), `review_queue` (blocked
responses awaiting compliance follow-up — see security/review_queue.py),
and `graph_nodes`/`graph_edges` (extracted entities/relationships — see
ingestion/extraction.py and retrieval/graph_store.py).

Entity labels and relationship names are extracted *from* document content,
so they get the same envelope encryption as chunk text (security/encryption.py)
rather than being stored as plaintext — `label`/`relation` never appear as
literal columns here, only `*_encrypted` ciphertext payloads. The one
plaintext-derived exception is `label_hash`: a deterministic (unkeyed)
SHA-256 of the normalized label, used only so entity deduplication
(retrieval/graph_store.py) can find an existing node without decrypting
every row in the org on every insert. It is not a secrecy boundary — an
attacker with the hash and a guess could confirm a label via dictionary
attack — a deliberate, documented trade-off appropriate for this project's
current Phase 1 scope (see LocalKeyProvider's own docstring for the same
class of trade-off), not a substitute for real blind-indexing crypto.
"""

from __future__ import annotations

from sqlalchemy import JSON, Column, DateTime, Float, Integer, MetaData, String, Table, create_engine
from sqlalchemy.engine import Engine

from finvault.config import settings

metadata = MetaData()

documents_table = Table(
    "documents",
    metadata,
    Column("id", String, primary_key=True),
    Column("org_id", String, nullable=False, index=True),
    Column("title", String, nullable=False),
    Column("source_path", String, nullable=True),
    Column("classification", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

# Append-only by convention (no UPDATE/DELETE paths are exposed anywhere in
# this codebase) — entry_hash chains to prev_hash, so any out-of-band mutation
# is detectable via AuditLog.verify_chain().
audit_log_table = Table(
    "audit_log",
    metadata,
    Column("seq", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", Float, nullable=False),
    Column("actor", String, nullable=False),
    Column("action", String, nullable=False),
    Column("resource", String, nullable=False),
    Column("details", JSON, nullable=False),
    Column("prev_hash", String, nullable=False),
    Column("entry_hash", String, nullable=False),
)

# Turn order within a (session_id, user_id) pair is `id` insertion order —
# matching InMemorySessionStore's ordering guarantee (agents/session.py).
sessions_table = Table(
    "sessions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("session_id", String, nullable=False, index=True),
    Column("user_id", String, nullable=False, index=True),
    Column("question", String, nullable=False),
    Column("answer", String, nullable=False),
    Column("created_at", Float, nullable=False),
)

quarantined_documents_table = Table(
    "quarantined_documents",
    metadata,
    # document_id is the primary key: quarantine is per document, never per
    # chunk (see security/quarantine.py for why chunk-level is evadable).
    Column("document_id", String, primary_key=True),
    Column("org_id", String, nullable=False, index=True),
    Column("reason", String, nullable=True),
    Column("status", String, nullable=False),
    Column("quarantined_by", String, nullable=False),
    Column("quarantined_at", Float, nullable=False),
    Column("released_by", String, nullable=True),
    Column("released_at", Float, nullable=True),
)

review_queue_table = Table(
    "review_queue",
    metadata,
    Column("id", String, primary_key=True),
    Column("org_id", String, nullable=False, index=True),
    Column("user_id", String, nullable=False),
    Column("question", String, nullable=False),
    Column("draft_answer", String, nullable=False),
    Column("block_reason", String, nullable=True),
    Column("findings", JSON, nullable=False),
    Column("citations", JSON, nullable=False),
    Column("status", String, nullable=False, index=True),
    Column("created_at", Float, nullable=False),
    Column("reviewed_by", String, nullable=True),
    Column("reviewed_at", Float, nullable=True),
    Column("reviewer_note", String, nullable=True),
)

graph_nodes_table = Table(
    "graph_nodes",
    metadata,
    Column("id", String, primary_key=True),
    Column("org_id", String, nullable=False, index=True),
    Column("type", String, nullable=False),
    Column("classification", String, nullable=False, index=True),
    Column("source_document_id", String, nullable=True),
    # Dedup lookup key only — see module docstring. NOT the encryption key
    # and NOT sufficient to recover the label on its own.
    Column("label_hash", String, nullable=False, index=True),
    Column("label_encrypted", JSON, nullable=False),
    Column("details", JSON, nullable=False),
)

graph_edges_table = Table(
    "graph_edges",
    metadata,
    Column("id", String, primary_key=True),
    Column("org_id", String, nullable=False, index=True),
    Column("source_node_id", String, nullable=False, index=True),
    Column("target_node_id", String, nullable=False, index=True),
    # The classification of whichever document *asserted* this relationship
    # — not derived from the endpoint nodes' own classifications, since a
    # relationship between two otherwise-lower-tier entities can itself be
    # sensitive (e.g. "Company X -- under_investigation_by -- Regulator Y").
    Column("classification", String, nullable=False, index=True),
    Column("relation_encrypted", JSON, nullable=False),
    Column("weight", Float, nullable=False),
)

_engine: Engine | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(settings.postgres_dsn, future=True)
    return _engine


def init_db(engine: Engine | None = None) -> None:
    """Creates any missing tables directly from `metadata`.

    The development and test path only. It is idempotent and needs no
    Alembic config, which is what makes it right for a throwaway database —
    but it can only ever *create*: it will not add a column to a table that
    already exists, so a database built this way silently diverges from the
    models the moment one changes.

    Every other environment goes through `finvault-migrate` (see
    `finvault/migrate.py`), which applies the reviewed revisions in
    `migrations/versions/` and then installs the RLS policies. The two stay
    in step because CI autogenerates against the same `metadata` and fails
    if that produces a non-empty diff (see .github/workflows/ci.yml), so a
    Table edited here without a matching revision does not reach main.
    """
    metadata.create_all(engine or get_engine())
