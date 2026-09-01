#!/usr/bin/env python3
"""End-to-end demo: ingest sample finance documents, then run a few queries
through the full FinVault pipeline (retrieval -> analysis -> compliance),
printing the security decisions made along the way so the guardrails are
visible, not just the RAG answers.

Uses Qdrant/Postgres if reachable (matches `docker-compose up -d`), otherwise
falls back to in-memory equivalents so the demo runs with zero infra beyond
an ANTHROPIC_API_KEY (or an `ant auth login` profile).

One of the sample documents (restricted_memo.txt) deliberately contains an
embedded prompt-injection attempt, so this script also demonstrates the
guardrail catching it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

from finvault.agents.compliance_agent import ComplianceAgent
from finvault.agents.orchestrator import Orchestrator
from finvault.config import settings
from finvault.ingestion.embeddings import LocalEmbeddingProvider
from finvault.ingestion.pipeline import IngestionPipeline
from finvault.models import Classification, Role, User
from finvault.observability import configure_logging
from finvault.retrieval.retriever import Retriever
from finvault.retrieval.vector_store import InMemoryVectorStore, VectorStore
from finvault.security.audit import AuditLog, InMemoryAuditLog
from finvault.security.encryption import EnvelopeEncryptor, LocalKeyProvider
from finvault.security.rls import org_scope

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_docs"
ORG_ID = "acme-capital"


def _build_vector_store(dimension: int) -> VectorStore:
    try:
        from finvault.retrieval.vector_store import QdrantStore

        store = QdrantStore(url=settings.qdrant_url, collection=settings.qdrant_collection, dimension=dimension)
        print(f"[infra] connected to Qdrant at {settings.qdrant_url}")
        return store
    except Exception as exc:
        print(f"[infra] Qdrant unavailable ({exc}); falling back to in-memory vector store")
        return InMemoryVectorStore()


def _build_audit_log_and_engine() -> tuple[AuditLog, Engine | None]:
    """Returns the audit log plus the live DB engine (or None on fallback),
    so callers can also wire the same engine into IngestionPipeline for
    document-metadata persistence — using one Postgres connection for the
    audit log but not the other real table would be a partial connection,
    not "everything connected".
    """
    try:
        from sqlalchemy import text

        from finvault.db import get_engine, init_db
        from finvault.security.audit import PostgresAuditLog

        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        init_db(engine)
        print("[infra] connected to Postgres")
        return PostgresAuditLog(engine), engine
    except Exception as exc:
        print(f"[infra] Postgres unavailable ({exc}); falling back to in-memory audit log")
        return InMemoryAuditLog(), None


def main() -> None:
    # Text format, not the deployment default of JSON: this script's own
    # output is human-readable `print` lines, and interleaving JSON log
    # records with them would make the demo unreadable. The library logs
    # still surface — that is the point of configuring at all, since
    # without it a retry or a failed extraction would be invisible here.
    configure_logging(fmt="text")

    # No request means no org in context, and with Row Level Security on
    # that means every tenant table reads back empty (security/rls.py) —
    # the fail-closed direction, but silently confusing in a demo. Declaring
    # the scope explicitly is what every non-request entry point owes the
    # database, and this is the reference example of doing it.
    with org_scope(ORG_ID):
        _run_demo()


def _run_demo() -> None:
    print("=== FinVault demo: ingest -> query -> compliance review ===\n")

    key_provider = LocalKeyProvider(settings.finvault_master_key_path)
    encryptor = EnvelopeEncryptor(key_provider)
    print(f"[security] master key at {settings.finvault_master_key_path.resolve()}")

    embedding_provider = LocalEmbeddingProvider(settings.finvault_embedding_model)
    print(
        f"[security] embeddings run locally via '{settings.finvault_embedding_model}' "
        "— no document text is sent to a third-party embedding API"
    )

    vector_store = _build_vector_store(embedding_provider.dimension)
    audit_log, db_engine = _build_audit_log_and_engine()

    pipeline = IngestionPipeline(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        encryptor=encryptor,
        audit_log=audit_log,
        db_engine=db_engine,
    )
    retriever = Retriever(
        vector_store=vector_store, embedding_provider=embedding_provider, encryptor=encryptor, audit_log=audit_log
    )

    print("\n--- Ingesting sample documents ---")
    documents = {
        "quarterly_report.txt": Classification.INTERNAL,
        "compliance_policy.txt": Classification.CONFIDENTIAL,
        "restricted_memo.txt": Classification.RESTRICTED,
    }
    for filename, classification in documents.items():
        doc = pipeline.ingest_file(
            SAMPLE_DIR / filename, org_id=ORG_ID, classification=classification, actor="demo-script"
        )
        print(f"  ingested '{doc.title}' as {classification.value} -> document_id={doc.id}")

    analyst_user = User(username="dana-analyst", role=Role.ANALYST, org_id=ORG_ID)
    viewer_user = User(username="val-viewer", role=Role.VIEWER, org_id=ORG_ID)
    compliance_officer = User(username="cara-compliance", role=Role.COMPLIANCE_OFFICER, org_id=ORG_ID)

    questions = [
        (analyst_user, "What was total revenue and the year-over-year growth rate in the quarterly report?"),
        # A viewer lacks clearance for CONFIDENTIAL/RESTRICTED content — the
        # retriever should silently drop those chunks rather than answering.
        (viewer_user, "What does the restricted memo about client account 48213 say?"),
        (compliance_officer, "Summarize the AML compliance policy's customer due diligence requirements."),
        # A compliance officer *does* have clearance for the restricted memo,
        # which also contains the embedded injection attempt — this exercises
        # both the injection-detection heuristic and the untrusted-content
        # wrapping on genuinely authorized access.
        (compliance_officer, "What does the restricted memo about client account 48213 say?"),
    ]

    for user, question in questions:
        print(f"\n--- Query as {user.username} ({user.role.value}) ---")
        print(f"Q: {question}")
        orchestrator = Orchestrator(
            retriever=retriever, user=user, audit_log=audit_log, compliance_agent=ComplianceAgent()
        )
        result = orchestrator.handle(question)
        print(f"Blocked: {result.blocked}" + (f" ({result.block_reason})" if result.block_reason else ""))
        if result.injection_flags:
            print(f"Injection heuristics triggered: {result.injection_flags}")
        if result.guardrail_findings:
            print(f"Output guardrail redactions: {result.guardrail_findings}")
        print(f"A: {result.answer}")

    print("\n--- Audit trail ---")
    print(f"Chain intact: {audit_log.verify_chain()}")
    for entry in audit_log.entries()[-12:]:
        print(f"  [{entry.seq}] {entry.actor} {entry.action} {entry.resource} {entry.details}")


if __name__ == "__main__":
    main()
