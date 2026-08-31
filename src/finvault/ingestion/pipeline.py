"""Ingestion pipeline: load -> chunk -> classify -> embed -> encrypt -> store.

Plaintext exists only transiently, in memory, between `load_text` and
`encryptor.encrypt` — it is never written to the vector store or the
documents table. Every ingest is audit-logged.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from sqlalchemy import insert
from sqlalchemy.engine import Engine

from finvault.agents.base import AgentExecutionError
from finvault.config import settings
from finvault.db import documents_table
from finvault.ingestion.chunking import chunk_text
from finvault.ingestion.classification import ClassificationSuggester
from finvault.ingestion.embeddings import EmbeddingProvider
from finvault.ingestion.extraction import ExtractionAgent
from finvault.ingestion.loaders import load_text
from finvault.models import Chunk, Classification, Document
from finvault.observability import get_logger, log_exception
from finvault.retrieval.graph_store import GraphStore, edge_aad, label_hash, node_aad
from finvault.retrieval.vector_store import VectorRecord, VectorStore
from finvault.security.audit import AuditLog
from finvault.security.encryption import EnvelopeEncryptor, chunk_aad

logger = get_logger(__name__)


class IngestionPipeline:
    def __init__(
        self,
        *,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        encryptor: EnvelopeEncryptor,
        audit_log: AuditLog,
        db_engine: Engine | None = None,
        classification_suggester: ClassificationSuggester | None = None,
        extraction_agent: ExtractionAgent | None = None,
        graph_store: GraphStore | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._embedding_provider = embedding_provider
        self._encryptor = encryptor
        self._audit_log = audit_log
        self._db_engine = db_engine
        # Optional and advisory only — see classification.py. When present,
        # its suggestion is logged alongside the caller-asserted tier for
        # drift-tracking; it never influences what actually gets stored.
        self._classification_suggester = classification_suggester
        # Optional — see ingestion/extraction.py. Both must be provided for
        # extraction to run; either omitted means ingestion behaves exactly
        # as before this feature existed (embed/encrypt/store only).
        self._extraction_agent = extraction_agent
        self._graph_store = graph_store

    def ingest_file(
        self,
        path: Path,
        *,
        org_id: str,
        classification: Classification,
        title: str | None = None,
        actor: str = "system",
    ) -> Document:
        text = load_text(path)
        document = Document(
            org_id=org_id,
            title=title or path.name,
            source_path=str(path),
            classification=classification,
        )
        self._persist_document(document)

        pieces = chunk_text(text)
        if not pieces:
            self._audit_log.append(
                actor=actor,
                action="ingest",
                resource=document.id,
                details={"chunks": 0, "warning": "no extractable text"},
            )
            return document

        if len(pieces) > settings.finvault_max_chunks_per_document:
            pieces = pieces[: settings.finvault_max_chunks_per_document]

        vectors = self._embedding_provider.embed(pieces)
        records: list[VectorRecord] = []
        # strict=True: one vector per chunk is an invariant of the embedding
        # call above. A mismatch would silently drop trailing chunks from the
        # index — a document that ingested "successfully" but is only
        # partially searchable, which is far worse than a loud failure.
        for index, (piece, vector) in enumerate(zip(pieces, vectors, strict=True)):
            chunk = Chunk(document_id=document.id, org_id=org_id, classification=classification, chunk_index=index)
            aad = chunk_aad(document.id, index)
            encrypted = self._encryptor.encrypt(piece, aad=aad)
            records.append(
                VectorRecord(
                    id=chunk.id,
                    vector=vector,
                    payload={
                        "document_id": document.id,
                        "org_id": org_id,
                        "classification": classification.value,
                        "chunk_index": index,
                        "document_title": document.title,
                        "embedding_model": self._embedding_provider.name,
                        "encrypted": encrypted.to_dict(),
                    },
                )
            )
        self._vector_store.upsert(records)

        details: dict[str, object] = {"chunks": len(records), "classification": classification.value, "org_id": org_id}
        if self._classification_suggester is not None:
            suggestion = self._classification_suggester.suggest(text[:50000])
            details["suggested_classification"] = suggestion.predicted.value
            details["suggested_classification_confidence"] = round(suggestion.confidence, 4)

        if self._extraction_agent is not None and self._graph_store is not None:
            try:
                extraction = self._extraction_agent.extract(text[:50000])
            except AgentExecutionError as exc:
                # ExtractionAgent.extract() already degrades to an empty
                # result for a malformed/no-tool-call response (see its own
                # docstring) — but a genuine agent-level failure (rate
                # limit, network error, etc.) propagates as
                # AgentExecutionError instead of being swallowed there. This
                # is the one further layer of defense that promise needs:
                # embed/encrypt/store must never be blocked by extraction
                # not working, and neither should it 500 the whole request
                # (found live: a 429 from the LLM provider's daily free-tier
                # quota crashed ingestion entirely before this fix).
                # Ingestion continues without a knowledge graph rather
                # than failing: the chunks are already embedded and stored,
                # and losing entity extraction degrades one feature instead
                # of the whole upload. WARNING (not ERROR) because the
                # user-visible operation succeeded — but it is logged, and
                # recorded on the audit row, so a graph that quietly stopped
                # growing is traceable to the day the quota ran out.
                log_exception(logger, exc, "extraction_failed_ingest_continuing", document_id=document.id)
                details["extraction_error"] = str(exc)
                extraction = None
            else:
                node_ids_by_label: dict[str, str] = {}
                for entity in extraction.entities:
                    node_label_hash = label_hash(org_id, entity.type, entity.label)
                    encrypted_label = self._encryptor.encrypt(
                        entity.label, aad=node_aad(org_id, entity.type, node_label_hash)
                    )
                    node = self._graph_store.upsert_node(
                        org_id=org_id,
                        type=entity.type,
                        classification=classification.value,
                        source_document_id=document.id,
                        node_label_hash=node_label_hash,
                        label_encrypted=encrypted_label,
                    )
                    node_ids_by_label[entity.label] = node.id
                for rel in extraction.relationships:
                    source_id = node_ids_by_label.get(rel.source_label)
                    target_id = node_ids_by_label.get(rel.target_label)
                    if not source_id or not target_id:
                        continue
                    edge_id = str(uuid4())
                    encrypted_relation = self._encryptor.encrypt(rel.relation, aad=edge_aad(edge_id))
                    self._graph_store.add_edge(
                        id=edge_id,
                        org_id=org_id,
                        source_node_id=source_id,
                        target_node_id=target_id,
                        classification=classification.value,
                        relation_encrypted=encrypted_relation,
                    )
                details["extracted_entities"] = len(extraction.entities)
                details["extracted_relationships"] = len(extraction.relationships)
                details["extraction_injection_flags"] = self._extraction_agent.last_injection_flags

        self._audit_log.append(actor=actor, action="ingest", resource=document.id, details=details)
        return document

    def _persist_document(self, document: Document) -> None:
        if self._db_engine is None:
            return
        with self._db_engine.begin() as conn:
            conn.execute(
                insert(documents_table).values(
                    id=document.id,
                    org_id=document.org_id,
                    title=document.title,
                    source_path=document.source_path,
                    classification=document.classification.value,
                    created_at=document.created_at,
                )
            )
