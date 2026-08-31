"""Tests for POST /documents/compare's route handler.

No FastAPI TestClient/HTTP-level test here, matching the rest of this
codebase's route-testing approach (see test_query_stream.py's docstring).
The route's own logic — validating document_ids, dropping inaccessible
documents, and failing closed on an agent error — is what's worth testing;
ComparisonAgent's extraction/scoring logic is already covered in isolation
by test_comparison_agent.py, and Retriever.get_document_text by
test_pipeline.py.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from finvault.agents.comparison_agent import ComparisonAgent
from finvault.api.routes import CompareRequest, compare_documents
from finvault.errors import InvalidRequestError, NotFoundError
from finvault.ingestion.pipeline import IngestionPipeline
from finvault.models import Classification, Role, User
from finvault.retrieval.retriever import Retriever
from finvault.retrieval.vector_store import InMemoryVectorStore
from finvault.security.audit import InMemoryAuditLog
from finvault.security.encryption import EnvelopeEncryptor, LocalKeyProvider
from tests.fakes import FakeEmbeddingProvider, FakeOpenAIClient, FakeResponse


def _make_request_with_two_documents(tmp_path):
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

    doc_a = pipeline.ingest_file(
        _write(tmp_path, "a.txt", "Q3 revenue was $10 million."), org_id="org-a", classification=Classification.INTERNAL
    )
    doc_b = pipeline.ingest_file(
        _write(tmp_path, "b.txt", "Q3 revenue was $11 million."), org_id="org-a", classification=Classification.INTERNAL
    )

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(retriever=retriever, audit_log=audit_log)))
    return request, doc_a, doc_b, audit_log


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_compare_rejects_fewer_than_two_document_ids(tmp_path) -> None:
    request, doc_a, _doc_b, _audit_log = _make_request_with_two_documents(tmp_path)
    user = User(username="a", role=Role.ANALYST, org_id="org-a")

    # The route raises the domain error, not an HTTPException — the status
    # code lives on the exception class (see finvault/errors.py) and is
    # applied by api/error_handlers.py, so asserting on both here pins the
    # contract the API actually exposes.
    with pytest.raises(InvalidRequestError) as exc_info:
        compare_documents(CompareRequest(document_ids=[doc_a.id]), request, user=user)
    assert exc_info.value.http_status == 400
    assert exc_info.value.code == "invalid_request"


def test_compare_404s_when_fewer_than_two_documents_are_found_or_accessible(tmp_path) -> None:
    request, doc_a, _doc_b, _audit_log = _make_request_with_two_documents(tmp_path)
    user = User(username="a", role=Role.ANALYST, org_id="org-a")

    with pytest.raises(NotFoundError) as exc_info:
        compare_documents(CompareRequest(document_ids=[doc_a.id, "no-such-document"]), request, user=user)
    assert exc_info.value.http_status == 404


def test_compare_silently_drops_a_document_from_another_org(tmp_path, monkeypatch) -> None:
    request, doc_a, doc_b, _audit_log = _make_request_with_two_documents(tmp_path)
    other_org_user = User(username="b", role=Role.ANALYST, org_id="org-b")

    # Neither document belongs to org-b, so both get silently dropped —
    # fewer than two accessible documents remain, same 404 as "not found".
    with pytest.raises(NotFoundError) as exc_info:
        compare_documents(CompareRequest(document_ids=[doc_a.id, doc_b.id]), request, user=other_org_user)
    assert exc_info.value.http_status == 404
    # Not AccessDeniedError: a 403 here would confirm the documents exist.
    assert exc_info.value.code == "not_found"


def test_compare_returns_a_real_heatmap_and_audit_logs_the_comparison(tmp_path, monkeypatch) -> None:
    request, doc_a, doc_b, audit_log = _make_request_with_two_documents(tmp_path)
    user = User(username="a", role=Role.ANALYST, org_id="org-a")

    scripted_client = FakeOpenAIClient(
        [
            FakeResponse.tool_call(
                "submit_comparison",
                {
                    "metrics": [
                        {
                            "metric_name": "Q3 Revenue",
                            "values": [
                                {"document_title": "a.txt", "display_value": "$10M", "raw_value": 10_000_000},
                                {"document_title": "b.txt", "display_value": "$11M", "raw_value": 11_000_000},
                            ],
                        }
                    ]
                },
            )
        ]
    )
    monkeypatch.setattr("finvault.api.routes.ComparisonAgent", lambda: ComparisonAgent(client=scripted_client))

    heatmap = compare_documents(CompareRequest(document_ids=[doc_a.id, doc_b.id]), request, user=user)

    assert set(heatmap.documents) == {"a.txt", "b.txt"}
    assert heatmap.metrics == ["Q3 Revenue"]
    assert len(heatmap.cells) == 2

    compare_entries = [e for e in audit_log.entries() if e.action == "compare_documents"]
    assert len(compare_entries) == 1
    assert compare_entries[0].details["compared"] == 2
    assert compare_entries[0].details["metrics_found"] == 1
