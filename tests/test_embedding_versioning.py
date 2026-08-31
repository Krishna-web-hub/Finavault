from __future__ import annotations

from finvault.ingestion.pipeline import IngestionPipeline
from finvault.models import Classification, Role, User
from finvault.retrieval.retriever import Retriever
from finvault.retrieval.vector_store import InMemoryVectorStore, VectorRecord
from finvault.security.audit import InMemoryAuditLog
from finvault.security.encryption import EnvelopeEncryptor, LocalKeyProvider, chunk_aad
from tests.fakes import FakeEmbeddingProvider


class _RenamedFakeEmbeddingProvider(FakeEmbeddingProvider):
    """Same vectors as FakeEmbeddingProvider, but reports a different model
    name — simulates the currently-configured model having changed since a
    chunk was ingested.
    """

    @property
    def name(self) -> str:
        return "a-different-model-v2"


def _setup(tmp_path):
    key_provider = LocalKeyProvider(tmp_path / "master.key")
    encryptor = EnvelopeEncryptor(key_provider)
    vector_store = InMemoryVectorStore()
    audit_log = InMemoryAuditLog()
    return key_provider, encryptor, vector_store, audit_log


def test_chunk_ingested_with_current_model_is_returned(tmp_path) -> None:
    _, encryptor, vector_store, audit_log = _setup(tmp_path)
    embedding_provider = FakeEmbeddingProvider()

    pipeline = IngestionPipeline(
        vector_store=vector_store, embedding_provider=embedding_provider, encryptor=encryptor, audit_log=audit_log
    )
    doc_path = tmp_path / "doc.txt"
    doc_path.write_text("Total revenue was $10 million in Q1.")
    pipeline.ingest_file(doc_path, org_id="org-a", classification=Classification.INTERNAL)

    retriever = Retriever(
        vector_store=vector_store, embedding_provider=embedding_provider, encryptor=encryptor, audit_log=audit_log
    )
    analyst = User(username="a", role=Role.ANALYST, org_id="org-a")
    hits = retriever.retrieve("revenue", user=analyst, top_k=5)

    assert len(hits) == 1
    retrieve_entries = [e for e in audit_log.entries() if e.action == "retrieve"]
    assert retrieve_entries[-1].details["embedding_model_mismatches"] == 0


def test_chunk_ingested_with_a_different_model_is_dropped_and_counted(tmp_path) -> None:
    _, encryptor, vector_store, audit_log = _setup(tmp_path)
    ingest_provider = FakeEmbeddingProvider()

    pipeline = IngestionPipeline(
        vector_store=vector_store, embedding_provider=ingest_provider, encryptor=encryptor, audit_log=audit_log
    )
    doc_path = tmp_path / "doc.txt"
    doc_path.write_text("Total revenue was $10 million in Q1.")
    pipeline.ingest_file(doc_path, org_id="org-a", classification=Classification.INTERNAL)

    # Query-time provider reports a different model name than the one that
    # actually produced the stored vectors.
    query_provider = _RenamedFakeEmbeddingProvider()
    retriever = Retriever(
        vector_store=vector_store, embedding_provider=query_provider, encryptor=encryptor, audit_log=audit_log
    )
    analyst = User(username="a", role=Role.ANALYST, org_id="org-a")
    hits = retriever.retrieve("revenue", user=analyst, top_k=5)

    assert hits == []
    retrieve_entries = [e for e in audit_log.entries() if e.action == "retrieve"]
    assert retrieve_entries[-1].details["embedding_model_mismatches"] == 1


def test_legacy_chunk_with_no_embedding_model_tag_is_still_returned(tmp_path) -> None:
    """Data ingested before this field existed has no embedding_model key at
    all — that must be treated as compatible, not as a mismatch, so upgrading
    doesn't silently drop an entire pre-existing corpus.
    """
    _, encryptor, vector_store, audit_log = _setup(tmp_path)
    embedding_provider = FakeEmbeddingProvider()

    plaintext = "Legacy chunk ingested before embedding_model tagging existed."
    encrypted = encryptor.encrypt(plaintext, aad=chunk_aad("doc-1", 0))
    vector_store.upsert(
        [
            VectorRecord(
                id="chunk-1",
                vector=embedding_provider.embed([plaintext])[0],
                payload={
                    "document_id": "doc-1",
                    "org_id": "org-a",
                    "classification": Classification.INTERNAL.value,
                    "chunk_index": 0,
                    "document_title": "legacy.txt",
                    "encrypted": encrypted.to_dict(),
                    # deliberately no "embedding_model" key
                },
            )
        ]
    )

    retriever = Retriever(
        vector_store=vector_store, embedding_provider=embedding_provider, encryptor=encryptor, audit_log=audit_log
    )
    analyst = User(username="a", role=Role.ANALYST, org_id="org-a")
    hits = retriever.retrieve("legacy chunk", user=analyst, top_k=5)

    assert len(hits) == 1
    assert hits[0].text == plaintext
