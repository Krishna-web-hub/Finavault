"""Search -> ACL filter -> decrypt -> hybrid rerank.

This is the only place decrypted chunk plaintext is produced during a query.
Chunks the requesting user lacks clearance for are silently dropped from the
result set (not surfaced as a denial) so that whether a matching restricted
document even exists isn't itself information a lower-clearance user can infer.

Hybrid reranking (BM25 lexical scoring fused with the vector similarity
score) deliberately runs only over this already-decrypted, already
ACL-filtered candidate set — in memory, per query — rather than maintaining
a separate persisted lexical index over the corpus. A standing BM25 index
needs a term/document matrix derived from plaintext, which would itself be
a new at-rest artifact leaking content structure; reranking an ephemeral
candidate set that's already been authorized and decrypted for this request
adds the lexical signal without adding a new place plaintext-derived data
lives outside memory.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from finvault.ingestion.embeddings import EmbeddingProvider
from finvault.models import Chunk, Classification, RetrievedChunk, RetrievedDocument, User
from finvault.retrieval.reranker import Reranker
from finvault.retrieval.vector_store import VectorStore
from finvault.security.access_control import check_clearance
from finvault.security.audit import AuditLog
from finvault.security.encryption import EncryptedPayload, EnvelopeEncryptor, chunk_aad
from finvault.security.quarantine import QuarantineStore

# Reciprocal Rank Fusion constant — the standard default from the RRF
# literature (Cormack et al.), not tuned for this corpus. Larger k flattens
# the influence of any single ranking's top position; smaller k weights a
# #1 rank in either signal more heavily.
_RRF_K = 60

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _reciprocal_rank_fusion(*rankings: list[str], k: int = _RRF_K) -> dict[str, float]:
    """Each ranking is a list of ids in descending-relevance order. Returns a
    fused score per id (higher is better). An id absent from one ranking
    simply doesn't get that ranking's contribution — not penalized further.
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            fused[item_id] = fused.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return fused


class Retriever:
    def __init__(
        self,
        *,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        encryptor: EnvelopeEncryptor,
        audit_log: AuditLog,
        use_hybrid_rerank: bool = True,
        reranker: Reranker | None = None,
        quarantine_store: QuarantineStore | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._embedding_provider = embedding_provider
        self._encryptor = encryptor
        self._audit_log = audit_log
        self._use_hybrid_rerank = use_hybrid_rerank
        # Optional third fusion signal (see retrieval/reranker.py) — None by
        # default, since a cross-encoder adds a neural forward pass per
        # candidate rather than BM25's near-zero cost. Callers opt in
        # explicitly (see api/main.py's FINVAULT_ENABLE_CROSS_ENCODER_RERANK).
        self._reranker = reranker
        # Optional — see security/quarantine.py. None means no document is
        # ever excluded, exactly as before this feature existed.
        self._quarantine_store = quarantine_store

    def retrieve(self, query: str, *, user: User, top_k: int = 5) -> list[RetrievedChunk]:
        query_vector = self._embedding_provider.embed([query])[0]
        hits = self._vector_store.search(query_vector, top_k=top_k, org_id=user.org_id)

        # Fetched once per call rather than per hit: retrieval filters a
        # whole page of results, so one query beats top_k of them.
        quarantined = self._quarantined_ids(org_id=user.org_id)

        results: list[RetrievedChunk] = []
        denied = 0
        model_mismatches = 0
        quarantined_dropped = 0
        for hit in hits:
            payload = hit.payload
            classification = Classification(payload["classification"])
            if not check_clearance(user.role, classification):
                denied += 1
                continue

            # Dropped before the ACL-cleared text is ever decrypted, let
            # alone assembled into a prompt. Unlike an over-classification
            # withholding (see agents/retriever_agent.py), no marker is
            # emitted in its place: a quarantined document is one an
            # operator has judged hostile, and naming it back into the
            # model's context is the thing quarantine exists to stop.
            if payload["document_id"] in quarantined:
                quarantined_dropped += 1
                continue

            # A chunk embedded by a different model than the one currently
            # configured lives in an incompatible vector space — same
            # dimension doesn't mean comparable distances. Missing tag (data
            # ingested before this field existed) is treated as compatible,
            # not as a mismatch, so this doesn't silently drop pre-existing
            # corpora on upgrade.
            stored_model = payload.get("embedding_model")
            if stored_model is not None and stored_model != self._embedding_provider.name:
                model_mismatches += 1
                continue

            chunk = Chunk(
                id=hit.id,
                document_id=payload["document_id"],
                org_id=payload["org_id"],
                classification=classification,
                chunk_index=payload["chunk_index"],
            )
            encrypted = EncryptedPayload.from_dict(payload["encrypted"])
            aad = chunk_aad(chunk.document_id, chunk.chunk_index)
            text = self._encryptor.decrypt(encrypted, aad=aad)

            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    document_title=payload.get("document_title", ""),
                    text=text,
                    score=hit.score,
                )
            )

        hybrid_applied = False
        if self._use_hybrid_rerank and len(results) > 1:
            results = self._hybrid_rerank(query, results)
            hybrid_applied = True

        self._audit_log.append(
            actor=user.id,
            action="retrieve",
            resource="vector_store",
            details={
                "query": query,
                "returned": len(results),
                "denied_by_clearance": denied,
                "quarantined_dropped": quarantined_dropped,
                "embedding_model_mismatches": model_mismatches,
                "hybrid_reranked": hybrid_applied,
            },
        )
        return results

    def get_document_text(self, document_id: str, *, user: User) -> RetrievedDocument | None:
        """Reassembles one document's full plaintext from its chunks, in
        chunk_index order — for whole-document operations (e.g. comparison,
        agents/comparison_agent.py) where a handful of semantically-retrieved
        chunks isn't the right unit of context.

        Every chunk of a document shares the classification it was ingested
        at (see ingestion/pipeline.py), so in practice this is one clearance
        check, not many — but it's still applied per chunk, and any denial
        fails the whole document closed rather than assembling partial text.
        Same non-inference posture as retrieve(): returns None (not found,
        wrong org, or ACL-denied) rather than distinguishing those cases.
        """
        # Quarantine applies to every path that yields plaintext, not just
        # semantic retrieval — the comparison route reaches documents by id,
        # which would otherwise walk straight around the filter in
        # retrieve(). Returns None like any other denial: same
        # non-inference posture as the docstring describes.
        if document_id in self._quarantined_ids(org_id=user.org_id):
            return None

        hits = self._vector_store.get_by_document(document_id, org_id=user.org_id)
        if not hits:
            return None

        ordered = sorted(hits, key=lambda h: h.payload["chunk_index"])
        title = ordered[0].payload.get("document_title", "")
        pieces: list[str] = []
        for hit in ordered:
            payload = hit.payload
            classification = Classification(payload["classification"])
            if not check_clearance(user.role, classification):
                return None

            encrypted = EncryptedPayload.from_dict(payload["encrypted"])
            aad = chunk_aad(document_id, payload["chunk_index"])
            pieces.append(self._encryptor.decrypt(encrypted, aad=aad))

        self._audit_log.append(
            actor=user.id, action="get_document_text", resource=document_id, details={"chunks": len(pieces)}
        )
        return RetrievedDocument(document_id=document_id, title=title, text="\n\n".join(pieces))

    def _quarantined_ids(self, *, org_id: str) -> set[str]:
        """Empty set when no store is configured, so every call site can
        treat quarantine as an ordinary filter instead of branching on
        whether the feature is wired up.
        """
        if self._quarantine_store is None:
            return set()
        return self._quarantine_store.quarantined_ids(org_id=org_id)

    def _hybrid_rerank(self, query: str, results: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Fuses the existing vector-similarity order with a BM25 lexical
        ranking, and optionally a cross-encoder ranking, over the same
        (already decrypted, already ACL-filtered) candidate set — via
        reciprocal rank fusion. Candidate ids are unique chunk ids, so ties
        in any ranking don't collide across chunks.
        """
        vector_ranked_ids = [r.chunk.id for r in results]

        bm25 = BM25Okapi([_tokenize(r.text) for r in results])
        bm25_scores = bm25.get_scores(_tokenize(query))
        bm25_ranked_ids = [
            results[i].chunk.id for i in sorted(range(len(results)), key=lambda i: bm25_scores[i], reverse=True)
        ]

        rankings = [vector_ranked_ids, bm25_ranked_ids]
        if self._reranker is not None:
            cross_encoder_scores = self._reranker.score(query, [r.text for r in results])
            cross_encoder_ranked_ids = [
                results[i].chunk.id
                for i in sorted(range(len(results)), key=lambda i: cross_encoder_scores[i], reverse=True)
            ]
            rankings.append(cross_encoder_ranked_ids)

        fused_scores = _reciprocal_rank_fusion(*rankings)
        reranked = sorted(results, key=lambda r: fused_scores[r.chunk.id], reverse=True)
        return [r.model_copy(update={"score": fused_scores[r.chunk.id]}) for r in reranked]
