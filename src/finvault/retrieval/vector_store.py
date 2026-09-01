"""Vector store interface + Qdrant implementation.

Every payload stored here is either ciphertext (the `encrypted` field — see
security/encryption.py) or non-sensitive routing metadata (org_id,
classification, chunk_index, document title). A compromised or misconfigured
Qdrant instance never exposes plaintext document content.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class VectorRecord:
    id: str
    vector: list[float]
    payload: dict[str, Any]


@dataclass
class VectorSearchHit:
    id: str
    score: float
    payload: dict[str, Any]


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, records: list[VectorRecord]) -> None: ...

    @abstractmethod
    def search(self, query_vector: list[float], *, top_k: int, org_id: str) -> list[VectorSearchHit]: ...

    @abstractmethod
    def get_by_document(self, document_id: str, *, org_id: str) -> list[VectorSearchHit]:
        """Every chunk belonging to one document — an exact lookup, not a
        similarity search (score is meaningless here; see Retriever.
        get_document_text(), the only caller). Used for whole-document
        operations like comparison, where retrieval's usual top-k semantic
        search isn't the right tool."""
        ...


class QdrantStore(VectorStore):
    """Self-hostable vector store (see docker-compose.yml). Swap for a
    managed Qdrant Cloud instance or another VectorStore implementation in
    production — retrieval.py and ingestion/pipeline.py depend only on this
    interface.
    """

    def __init__(self, *, url: str, collection: str, dimension: int) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        if url.startswith("http://") or url.startswith("https://"):
            self._client = QdrantClient(url=url)
        elif url in (":memory:", "memory"):
            self._client = QdrantClient(location=":memory:")
        else:
            self._client = QdrantClient(path=url)

        self._collection = collection
        existing = {c.name for c in self._client.get_collections().collections}
        if collection not in existing:
            self._client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )

    def upsert(self, records: list[VectorRecord]) -> None:
        from qdrant_client.models import PointStruct

        points = [PointStruct(id=r.id, vector=r.vector, payload=r.payload) for r in records]
        self._client.upsert(collection_name=self._collection, points=points)

    def search(self, query_vector: list[float], *, top_k: int, org_id: str) -> list[VectorSearchHit]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        # Org isolation enforced at the query level, in addition to the
        # per-hit classification/ACL check applied downstream in
        # retrieval/retriever.py — defense in depth, not either/or.
        results = self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=Filter(must=[FieldCondition(key="org_id", match=MatchValue(value=org_id))]),
        )
        return [VectorSearchHit(id=str(r.id), score=r.score, payload=r.payload or {}) for r in results]

    def get_by_document(self, document_id: str, *, org_id: str) -> list[VectorSearchHit]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        points, _ = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="document_id", match=MatchValue(value=document_id)),
                    FieldCondition(key="org_id", match=MatchValue(value=org_id)),
                ]
            ),
            limit=10_000,  # generous upper bound on one document's chunk count
            with_payload=True,
            with_vectors=False,
        )
        return [VectorSearchHit(id=str(p.id), score=1.0, payload=p.payload or {}) for p in points]


class InMemoryVectorStore(VectorStore):
    """Reference implementation for tests and scripted demos — brute-force
    cosine similarity, no external service required.
    """

    def __init__(self) -> None:
        self._records: dict[str, VectorRecord] = {}

    def upsert(self, records: list[VectorRecord]) -> None:
        for record in records:
            self._records[record.id] = record

    def search(self, query_vector: list[float], *, top_k: int, org_id: str) -> list[VectorSearchHit]:
        import math

        def cosine(a: list[float], b: list[float]) -> float:
            # strict=True: comparing vectors of different dimension means two
            # embedding models are mixed in one collection — see retriever.py's
            # embedding_model mismatch filter for why that must never be scored.
            dot = sum(x * y for x, y in zip(a, b, strict=True))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

        scored = [
            VectorSearchHit(id=r.id, score=cosine(query_vector, r.vector), payload=r.payload)
            for r in self._records.values()
            if r.payload.get("org_id") == org_id
        ]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]

    def get_by_document(self, document_id: str, *, org_id: str) -> list[VectorSearchHit]:
        return [
            VectorSearchHit(id=r.id, score=1.0, payload=r.payload)
            for r in self._records.values()
            if r.payload.get("document_id") == document_id and r.payload.get("org_id") == org_id
        ]
