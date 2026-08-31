#!/usr/bin/env python3
"""Offline retrieval evaluation: precision@k / recall@k over a small
hand-labeled query set, run against sample_docs/.

This is the harness identified in the ML-improvements review: retrieval
quality changes (embedding model swaps, chunking tweaks, the hybrid BM25
rerank added alongside this script) previously had no way to be measured
other than eyeballing answers. Ground truth here is document-level (which
source document is relevant for a query) rather than chunk-level, since
chunk boundaries are an implementation detail that shifts with chunk_size —
document-level relevance is stable across those changes.

Compares vector-only retrieval against hybrid (vector + BM25 rerank) so the
impact of the new rerank step is visible, not just its existence.

Uses a real LocalEmbeddingProvider (same as scripts/ingest_sample.py) and an
in-memory vector store/audit log — no Qdrant/Postgres required. Retrieval is
run as a COMPLIANCE_OFFICER (cleared for every classification tier used
here) so results reflect retrieval quality, not ACL filtering, which is
already covered by tests/test_pipeline.py.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from finvault.ingestion.embeddings import LocalEmbeddingProvider
from finvault.ingestion.pipeline import IngestionPipeline
from finvault.models import Classification, Role, User
from finvault.observability import configure_logging
from finvault.retrieval.retriever import Retriever
from finvault.retrieval.vector_store import InMemoryVectorStore
from finvault.security.audit import InMemoryAuditLog
from finvault.security.encryption import EnvelopeEncryptor, LocalKeyProvider

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_docs"
ORG_ID = "eval-org"
TOP_K = 3

# (source file, classification) — mirrors how these documents are actually
# classified in scripts/ingest_sample.py.
_DOCS = [
    ("quarterly_report.txt", Classification.INTERNAL),
    ("compliance_policy.txt", Classification.CONFIDENTIAL),
    ("restricted_memo.txt", Classification.RESTRICTED),
]


@dataclass(frozen=True)
class LabeledQuery:
    query: str
    relevant_document: str  # filename this query's answer should come from


_EVAL_SET = [
    LabeledQuery("What was total revenue in Q3 FY2026?", "quarterly_report.txt"),
    LabeledQuery("What was net income for the quarter?", "quarterly_report.txt"),
    LabeledQuery("What are the customer due diligence requirements for new clients?", "compliance_policy.txt"),
    LabeledQuery("What triggers enhanced due diligence procedures?", "compliance_policy.txt"),
    LabeledQuery("What happened with client account 48213?", "restricted_memo.txt"),
]


def _build_retriever(embedding_provider: LocalEmbeddingProvider, *, use_hybrid_rerank: bool) -> Retriever:
    key_provider = LocalKeyProvider(Path(".secrets/eval_master.key"))
    encryptor = EnvelopeEncryptor(key_provider)
    vector_store = InMemoryVectorStore()
    audit_log = InMemoryAuditLog()

    pipeline = IngestionPipeline(
        vector_store=vector_store, embedding_provider=embedding_provider, encryptor=encryptor, audit_log=audit_log
    )
    for filename, classification in _DOCS:
        pipeline.ingest_file(SAMPLE_DIR / filename, org_id=ORG_ID, classification=classification)

    return Retriever(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        encryptor=encryptor,
        audit_log=audit_log,
        use_hybrid_rerank=use_hybrid_rerank,
    )


def _evaluate(retriever: Retriever, user: User) -> tuple[float, float]:
    """Returns (mean precision@k, recall@k) across the eval set."""
    precisions: list[float] = []
    hits_at_k: list[bool] = []

    for item in _EVAL_SET:
        results = retriever.retrieve(item.query, user=user, top_k=TOP_K)
        retrieved_titles = [r.document_title for r in results]

        relevant_in_topk = sum(1 for title in retrieved_titles if title == item.relevant_document)
        precisions.append(relevant_in_topk / TOP_K)
        hits_at_k.append(item.relevant_document in retrieved_titles)

    mean_precision = sum(precisions) / len(precisions)
    recall = sum(hits_at_k) / len(hits_at_k)
    return mean_precision, recall


def main() -> None:
    # WARNING and above only: this script's signal is the precision/recall
    # table it prints, and per-query INFO records would bury it.
    configure_logging(level="WARNING", fmt="text")
    print(f"=== Retrieval eval: {len(_EVAL_SET)} labeled queries, top_k={TOP_K} ===\n")
    embedding_provider = LocalEmbeddingProvider("BAAI/bge-small-en-v1.5")
    officer = User(username="evaluator", role=Role.COMPLIANCE_OFFICER, org_id=ORG_ID)

    vector_only = _build_retriever(embedding_provider, use_hybrid_rerank=False)
    precision_v, recall_v = _evaluate(vector_only, officer)

    hybrid = _build_retriever(embedding_provider, use_hybrid_rerank=True)
    precision_h, recall_h = _evaluate(hybrid, officer)

    print(f"{'Mode':<20}{'Precision@' + str(TOP_K):<16}{'Recall@' + str(TOP_K):<12}")
    print(f"{'Vector-only':<20}{precision_v:<16.2f}{recall_v:<12.2f}")
    print(f"{'Hybrid (BM25+vec)':<20}{precision_h:<16.2f}{recall_h:<12.2f}")


if __name__ == "__main__":
    main()
