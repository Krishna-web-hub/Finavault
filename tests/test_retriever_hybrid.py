from __future__ import annotations

from typing import ClassVar

from finvault.ingestion.embeddings import EmbeddingProvider
from finvault.ingestion.pipeline import IngestionPipeline
from finvault.models import Classification, Role, User
from finvault.retrieval.reranker import Reranker
from finvault.retrieval.retriever import Retriever, _reciprocal_rank_fusion, _tokenize
from finvault.retrieval.vector_store import InMemoryVectorStore
from finvault.security.audit import InMemoryAuditLog
from finvault.security.encryption import EnvelopeEncryptor, LocalKeyProvider

# --- Unit tests for the pure fusion helpers (no embeddings involved) ---


def test_tokenize_lowercases_and_splits_on_word_boundaries() -> None:
    assert _tokenize("Client Account #48213 wired $2.4 million.") == [
        "client",
        "account",
        "48213",
        "wired",
        "2",
        "4",
        "million",
    ]


def test_reciprocal_rank_fusion_rewards_agreement_across_rankings() -> None:
    # "a" is #1 in both rankings -> should fuse to the top score.
    fused = _reciprocal_rank_fusion(["a", "b", "c"], ["a", "c", "b"])
    assert max(fused, key=fused.get) == "a"


def test_reciprocal_rank_fusion_ignores_ids_absent_from_a_ranking() -> None:
    fused = _reciprocal_rank_fusion(["a", "b"], ["b"])
    # "a" only contributes from the first ranking; "b" contributes from both.
    assert fused["b"] > fused["a"]


# --- Integration test: BM25 measurably changes the final order relative to
# a deliberately adversarial vector ranking (vector score says the lexical
# match is the least relevant of four candidates). ---

_QUERY = "zzzqueryzzz"
_TARGET_TEXT = "The exact phrase zzzqueryzzz appears here. zzzqueryzzz repeated for emphasis zzzqueryzzz."
_FILLER_W = "Unrelated filler paragraph about weather patterns and travel."
_FILLER_X = "Unrelated filler paragraph about gardening tips and recipes."
_FILLER_Z = "Unrelated filler paragraph about sports scores and standings."


class FixedVectorEmbeddingProvider(EmbeddingProvider):
    """Deterministic fake keyed by exact text, so the test can hand-pick
    vector similarity independently of lexical content — specifically to
    make the vector ranking rank the true lexical match *last*, so any
    improvement after fusion is attributable to the BM25 signal, not chance.
    """

    _VECTORS: ClassVar[dict[str, list[float]]] = {
        _QUERY: [1.0, 0.0, 0.0, 0.0],
        _FILLER_W: [0.9, 0.0, 0.0, 0.1],  # closest to query by cosine
        _FILLER_X: [0.8, 0.0, 0.0, 0.2],
        _FILLER_Z: [0.7, 0.0, 0.0, 0.3],
        _TARGET_TEXT: [0.0, 1.0, 0.0, 0.0],  # orthogonal to query -> worst vector rank
    }

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._VECTORS[t] for t in texts]

    @property
    def dimension(self) -> int:
        return 4

    @property
    def name(self) -> str:
        return "fixed-vector-fake"


def _make_retriever(tmp_path, *, use_hybrid_rerank: bool, reranker: Reranker | None = None) -> Retriever:
    key_provider = LocalKeyProvider(tmp_path / "master.key")
    encryptor = EnvelopeEncryptor(key_provider)
    embedding_provider = FixedVectorEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    audit_log = InMemoryAuditLog()

    pipeline = IngestionPipeline(
        vector_store=vector_store, embedding_provider=embedding_provider, encryptor=encryptor, audit_log=audit_log
    )
    for text in (_FILLER_W, _FILLER_X, _FILLER_Z, _TARGET_TEXT):
        doc_path = tmp_path / f"{hash(text)}.txt"
        doc_path.write_text(text)
        pipeline.ingest_file(doc_path, org_id="org-a", classification=Classification.INTERNAL)

    return Retriever(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        encryptor=encryptor,
        audit_log=audit_log,
        use_hybrid_rerank=use_hybrid_rerank,
        reranker=reranker,
    )


def test_vector_only_ranks_the_lexical_match_last(tmp_path) -> None:
    retriever = _make_retriever(tmp_path, use_hybrid_rerank=False)
    analyst = User(username="a", role=Role.ANALYST, org_id="org-a")

    hits = retriever.retrieve(_QUERY, user=analyst, top_k=10)

    assert [h.text for h in hits][-1] == _TARGET_TEXT


def test_hybrid_rerank_lifts_the_lexical_match_out_of_last_place(tmp_path) -> None:
    retriever = _make_retriever(tmp_path, use_hybrid_rerank=True)
    analyst = User(username="a", role=Role.ANALYST, org_id="org-a")

    hits = retriever.retrieve(_QUERY, user=analyst, top_k=10)

    ranked_texts = [h.text for h in hits]
    assert ranked_texts[-1] != _TARGET_TEXT, (
        "BM25 fusion should move the exact lexical match out of last place, "
        "even though the (deliberately adversarial) vector score ranked it worst"
    )


def test_hybrid_rerank_is_logged_in_the_audit_trail(tmp_path) -> None:
    key_provider = LocalKeyProvider(tmp_path / "master.key")
    encryptor = EnvelopeEncryptor(key_provider)
    embedding_provider = FixedVectorEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    audit_log = InMemoryAuditLog()

    pipeline = IngestionPipeline(
        vector_store=vector_store, embedding_provider=embedding_provider, encryptor=encryptor, audit_log=audit_log
    )
    for text in (_FILLER_W, _TARGET_TEXT):
        doc_path = tmp_path / f"{hash(text)}.txt"
        doc_path.write_text(text)
        pipeline.ingest_file(doc_path, org_id="org-a", classification=Classification.INTERNAL)

    retriever = Retriever(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        encryptor=encryptor,
        audit_log=audit_log,
        use_hybrid_rerank=True,
    )
    analyst = User(username="a", role=Role.ANALYST, org_id="org-a")
    retriever.retrieve(_QUERY, user=analyst, top_k=10)

    retrieve_entries = [e for e in audit_log.entries() if e.action == "retrieve"]
    assert retrieve_entries[-1].details["hybrid_reranked"] is True


# --- Integration test: an optional cross-encoder signal can decisively
# change the winner relative to vector+BM25 fusion alone. ---


class FakeStrongReranker(Reranker):
    """Deterministic fake keyed by exact text — stands in for a real
    cross-encoder so this test doesn't need to download a model."""

    def __init__(self, scores_by_text: dict[str, float]) -> None:
        self._scores_by_text = scores_by_text

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [self._scores_by_text[t] for t in texts]


def test_cross_encoder_signal_can_change_the_winner(tmp_path) -> None:
    # With only vector + BM25, _FILLER_W wins: it has the best vector rank
    # (0) and a decent BM25 rank (1, tied-zero group ordered by original
    # position), edging out _TARGET_TEXT despite BM25 lifting it off last
    # place (see test_hybrid_rerank_lifts_the_lexical_match_out_of_last_place).
    two_signal = _make_retriever(tmp_path, use_hybrid_rerank=True)
    analyst = User(username="a", role=Role.ANALYST, org_id="org-a")
    two_signal_top = two_signal.retrieve(_QUERY, user=analyst, top_k=10)[0].text
    assert two_signal_top == _FILLER_W

    # A cross-encoder that strongly favors the true lexical match and
    # strongly disfavors the prior winner should flip the final ranking.
    reranker = FakeStrongReranker({_FILLER_W: 1.0, _FILLER_X: 3.0, _FILLER_Z: 2.0, _TARGET_TEXT: 10.0})
    three_signal = _make_retriever(tmp_path, use_hybrid_rerank=True, reranker=reranker)
    three_signal_top = three_signal.retrieve(_QUERY, user=analyst, top_k=10)[0].text

    assert three_signal_top == _TARGET_TEXT
