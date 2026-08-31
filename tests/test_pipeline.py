from __future__ import annotations

from finvault.ingestion.extraction import ExtractionAgent
from finvault.ingestion.pipeline import IngestionPipeline
from finvault.models import Classification, Role, User
from finvault.retrieval.graph_retriever import GraphRetriever
from finvault.retrieval.graph_store import InMemoryGraphStore
from finvault.retrieval.retriever import Retriever
from finvault.retrieval.vector_store import InMemoryVectorStore
from finvault.security.audit import InMemoryAuditLog
from finvault.security.encryption import EnvelopeEncryptor, LocalKeyProvider
from tests.fakes import FakeEmbeddingProvider, FakeOpenAIClient, FakeResponse, make_rate_limit_error


def _make_pipeline_and_retriever(tmp_path):
    key_provider = LocalKeyProvider(tmp_path / "master.key")
    encryptor = EnvelopeEncryptor(key_provider)
    embedding_provider = FakeEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    audit_log = InMemoryAuditLog()

    pipeline = IngestionPipeline(
        vector_store=vector_store, embedding_provider=embedding_provider, encryptor=encryptor, audit_log=audit_log
    )
    retriever = Retriever(
        vector_store=vector_store, embedding_provider=embedding_provider, encryptor=encryptor, audit_log=audit_log
    )
    return pipeline, retriever, audit_log


def test_ingest_then_retrieve_round_trips_plaintext(tmp_path) -> None:
    pipeline, retriever, _ = _make_pipeline_and_retriever(tmp_path)

    doc_path = tmp_path / "doc.txt"
    doc_path.write_text("Total revenue was $10 million in Q1.\n\nNet income was $2 million.")

    document = pipeline.ingest_file(doc_path, org_id="org-a", classification=Classification.INTERNAL)

    analyst = User(username="a", role=Role.ANALYST, org_id="org-a")
    hits = retriever.retrieve("revenue", user=analyst, top_k=5)

    assert hits, "expected at least one retrieved chunk"
    assert any("Total revenue" in h.text for h in hits)
    assert all(h.chunk.document_id == document.id for h in hits)


def test_retrieval_drops_chunks_above_user_clearance(tmp_path) -> None:
    pipeline, retriever, _ = _make_pipeline_and_retriever(tmp_path)

    doc_path = tmp_path / "restricted.txt"
    doc_path.write_text("This is a highly sensitive restricted paragraph about client X.")
    pipeline.ingest_file(doc_path, org_id="org-a", classification=Classification.RESTRICTED)

    viewer = User(username="v", role=Role.VIEWER, org_id="org-a")
    hits = retriever.retrieve("sensitive paragraph", user=viewer, top_k=5)

    # A viewer lacks clearance for RESTRICTED content — it must be dropped
    # silently, not surfaced as a denial (see retriever.py docstring: whether
    # a matching restricted document even exists shouldn't be inferable).
    assert hits == []


def test_retrieval_respects_org_isolation(tmp_path) -> None:
    pipeline, retriever, _ = _make_pipeline_and_retriever(tmp_path)

    doc_path = tmp_path / "doc.txt"
    doc_path.write_text("Org A's confidential revenue figures for the quarter.")
    pipeline.ingest_file(doc_path, org_id="org-a", classification=Classification.INTERNAL)

    other_org_analyst = User(username="b", role=Role.ANALYST, org_id="org-b")
    hits = retriever.retrieve("revenue figures", user=other_org_analyst, top_k=5)

    assert hits == []


def test_ingest_logs_to_audit_trail(tmp_path) -> None:
    pipeline, _, audit_log = _make_pipeline_and_retriever(tmp_path)

    doc_path = tmp_path / "doc.txt"
    doc_path.write_text("Some financial content to ingest.")
    document = pipeline.ingest_file(
        doc_path, org_id="org-a", classification=Classification.CONFIDENTIAL, actor="tester"
    )

    ingest_entries = [e for e in audit_log.entries() if e.action == "ingest" and e.resource == document.id]
    assert len(ingest_entries) == 1
    assert ingest_entries[0].details["classification"] == "confidential"
    assert audit_log.verify_chain() is True


def test_ingest_with_extraction_populates_the_graph_store_and_is_visible_via_graph_retriever(tmp_path) -> None:
    key_provider = LocalKeyProvider(tmp_path / "master.key")
    encryptor = EnvelopeEncryptor(key_provider)
    embedding_provider = FakeEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    audit_log = InMemoryAuditLog()
    graph_store = InMemoryGraphStore()
    extraction_client = FakeOpenAIClient(
        [
            FakeResponse.tool_call(
                "submit_extraction",
                {
                    "entities": [
                        {"label": "Acme Capital", "type": "company"},
                        {"label": "Q3 Revenue", "type": "metric"},
                    ],
                    "relationships": [
                        {"source_label": "Acme Capital", "target_label": "Q3 Revenue", "relation": "reported"}
                    ],
                },
            )
        ]
    )
    extraction_agent = ExtractionAgent(client=extraction_client)

    pipeline = IngestionPipeline(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        encryptor=encryptor,
        audit_log=audit_log,
        extraction_agent=extraction_agent,
        graph_store=graph_store,
    )

    doc_path = tmp_path / "doc.txt"
    doc_path.write_text("Acme Capital reported Q3 Revenue of $10 million.")
    document = pipeline.ingest_file(doc_path, org_id="org-a", classification=Classification.INTERNAL)

    graph_retriever = GraphRetriever(graph_store=graph_store, encryptor=encryptor)
    analyst = User(username="a", role=Role.ANALYST, org_id="org-a")
    graph = graph_retriever.get_graph(user=analyst)

    labels = {n.label for n in graph.nodes}
    assert labels == {"Acme Capital", "Q3 Revenue"}
    assert all(n.source_document_id == document.id for n in graph.nodes)
    assert len(graph.edges) == 1
    assert graph.edges[0].label == "reported"

    ingest_entries = [e for e in audit_log.entries() if e.action == "ingest" and e.resource == document.id]
    assert ingest_entries[0].details["extracted_entities"] == 2
    assert ingest_entries[0].details["extracted_relationships"] == 1


def test_ingest_survives_extraction_agent_raising_agent_execution_error(tmp_path) -> None:
    """Found live: a 429 rate-limit from the LLM provider inside
    ExtractionAgent.extract() propagated as AgentExecutionError all the way
    out of ingest_file(), 500ing the whole request — even though chunking,
    embedding, and encryption had already succeeded. extract() only
    swallows malformed-output cases (ValidationError/JSONDecodeError) on its
    own; a real agent-level failure needs this further layer, matching
    extraction.py's own documented promise that ingestion must never be
    blocked by extraction not working.
    """
    key_provider = LocalKeyProvider(tmp_path / "master.key")
    encryptor = EnvelopeEncryptor(key_provider)
    embedding_provider = FakeEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    audit_log = InMemoryAuditLog()
    graph_store = InMemoryGraphStore()
    failing_client = FakeOpenAIClient([make_rate_limit_error(), make_rate_limit_error(), make_rate_limit_error()])
    extraction_agent = ExtractionAgent(client=failing_client)

    pipeline = IngestionPipeline(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        encryptor=encryptor,
        audit_log=audit_log,
        extraction_agent=extraction_agent,
        graph_store=graph_store,
    )

    doc_path = tmp_path / "doc.txt"
    doc_path.write_text("Acme Capital reported Q3 Revenue of $10 million.")

    # Must not raise — the whole point of this test.
    document = pipeline.ingest_file(doc_path, org_id="org-a", classification=Classification.INTERNAL)

    # Chunking/embedding/encryption still happened normally.
    analyst = User(username="a", role=Role.ANALYST, org_id="org-a")
    retriever = Retriever(
        vector_store=vector_store, embedding_provider=embedding_provider, encryptor=encryptor, audit_log=audit_log
    )
    hits = retriever.retrieve("revenue", user=analyst, top_k=5)
    assert hits and hits[0].chunk.document_id == document.id

    # Graph population was skipped, not half-applied.
    graph_retriever = GraphRetriever(graph_store=graph_store, encryptor=encryptor)
    assert graph_retriever.get_graph(user=analyst).nodes == []

    ingest_entries = [e for e in audit_log.entries() if e.action == "ingest" and e.resource == document.id]
    assert "extraction_error" in ingest_entries[0].details
    assert "extracted_entities" not in ingest_entries[0].details


def test_get_document_text_reassembles_the_full_document_in_order(tmp_path) -> None:
    pipeline, retriever, _ = _make_pipeline_and_retriever(tmp_path)

    doc_path = tmp_path / "doc.txt"
    doc_path.write_text("First paragraph about revenue.\n\nSecond paragraph about net income.")
    document = pipeline.ingest_file(doc_path, org_id="org-a", classification=Classification.INTERNAL)

    analyst = User(username="a", role=Role.ANALYST, org_id="org-a")
    retrieved = retriever.get_document_text(document.id, user=analyst)

    assert retrieved is not None
    assert retrieved.document_id == document.id
    assert retrieved.title == document.title
    assert "First paragraph about revenue." in retrieved.text
    assert "Second paragraph about net income." in retrieved.text
    assert retrieved.text.index("First paragraph") < retrieved.text.index("Second paragraph")


def test_get_document_text_returns_none_for_a_classification_the_user_lacks_clearance_for(tmp_path) -> None:
    pipeline, retriever, _ = _make_pipeline_and_retriever(tmp_path)

    doc_path = tmp_path / "restricted.txt"
    doc_path.write_text("Highly sensitive restricted content.")
    document = pipeline.ingest_file(doc_path, org_id="org-a", classification=Classification.RESTRICTED)

    viewer = User(username="v", role=Role.VIEWER, org_id="org-a")
    assert retriever.get_document_text(document.id, user=viewer) is None


def test_get_document_text_returns_none_across_org_boundaries(tmp_path) -> None:
    pipeline, retriever, _ = _make_pipeline_and_retriever(tmp_path)

    doc_path = tmp_path / "doc.txt"
    doc_path.write_text("Org A's confidential content.")
    document = pipeline.ingest_file(doc_path, org_id="org-a", classification=Classification.INTERNAL)

    other_org_analyst = User(username="b", role=Role.ANALYST, org_id="org-b")
    assert retriever.get_document_text(document.id, user=other_org_analyst) is None


def test_get_document_text_returns_none_for_an_unknown_document_id(tmp_path) -> None:
    _, retriever, _ = _make_pipeline_and_retriever(tmp_path)
    analyst = User(username="a", role=Role.ANALYST, org_id="org-a")
    assert retriever.get_document_text("no-such-document", user=analyst) is None


def test_ingest_without_extraction_agent_configured_skips_extraction_entirely(tmp_path) -> None:
    """Extraction is fully optional — omitting extraction_agent/graph_store
    from the constructor must not change ingest behavior at all.
    """
    pipeline, _, audit_log = _make_pipeline_and_retriever(tmp_path)

    doc_path = tmp_path / "doc.txt"
    doc_path.write_text("Acme Capital reported Q3 Revenue of $10 million.")
    document = pipeline.ingest_file(doc_path, org_id="org-a", classification=Classification.INTERNAL)

    ingest_entries = [e for e in audit_log.entries() if e.action == "ingest" and e.resource == document.id]
    assert "extracted_entities" not in ingest_entries[0].details


def test_csv_chunking_and_large_dataset_safety(tmp_path) -> None:
    """CSV files with single newlines must chunk properly without generating
    oversized mega-chunks or crashing embedding.
    """
    from finvault.ingestion.chunking import chunk_text

    csv_data = "\n".join([f"col1_{i},col2_{i},col3_{i},value_{i}" for i in range(2000)])
    chunks = chunk_text(csv_data, chunk_size=500, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 700 for c in chunks)
