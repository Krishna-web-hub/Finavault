from __future__ import annotations

from finvault.agents.retriever_agent import RetrieverAgent
from finvault.ingestion.pipeline import IngestionPipeline
from finvault.models import Classification, Role, User
from finvault.retrieval.retriever import Retriever
from finvault.retrieval.vector_store import InMemoryVectorStore
from finvault.security.audit import InMemoryAuditLog
from finvault.security.encryption import EnvelopeEncryptor, LocalKeyProvider
from tests.fakes import FakeEmbeddingProvider, FakeOpenAIClient, FakeResponse

SECRET_SENTENCE = "Client account 48213 wired $2.4 million through a high-risk jurisdiction."


def test_restricted_content_is_withheld_from_the_llm_prompt(tmp_path) -> None:
    """Regression test for a real gap found live: a compliance officer has
    ACL clearance to *retrieve* restricted content, but that's a separate
    question from whether the content is permitted to ever reach the LLM.
    Previously restricted chunk text was forwarded into the tool result
    unconditionally and only the *final answer* was gated — meaning
    restricted content was already sent to the LLM provider by the time
    anything blocked it. This confirms the fix: the raw text never leaves
    the retrieval layer for a classification outside the externalization
    allowlist, even for a user who is fully authorized to read it.
    """
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

    doc_path = tmp_path / "restricted.txt"
    doc_path.write_text(SECRET_SENTENCE)
    pipeline.ingest_file(doc_path, org_id="org-a", classification=Classification.RESTRICTED)

    # Compliance officer: fully cleared by RBAC to retrieve RESTRICTED content.
    compliance_officer = User(username="cara", role=Role.COMPLIANCE_OFFICER, org_id="org-a")

    client = FakeOpenAIClient(
        [
            FakeResponse.tool_call("search_documents", {"query": "client account 48213"}),
            FakeResponse.text("Reported back to the orchestrator."),
        ]
    )
    retriever_agent = RetrieverAgent(retriever=retriever, user=compliance_officer, client=client)

    result, _flags = retriever_agent.run("What does the memo about client account 48213 say?")

    assert result == "Reported back to the orchestrator."

    # The tool result sent back to the model (the last message of the second
    # request) must not contain the actual secret text — only a withheld notice.
    second_call_messages = client.chat.completions.calls[1]["messages"]
    tool_result_content = second_call_messages[-1]["content"]
    assert SECRET_SENTENCE not in tool_result_content
    assert "withheld" in tool_result_content.lower()

    # ACL clearance still means the retriever *saw* the chunk (tracked for
    # the Orchestrator's second, defense-in-depth gate) — this isn't a
    # retrieval denial, only an externalization one.
    assert retriever_agent.max_classification_seen == Classification.RESTRICTED
