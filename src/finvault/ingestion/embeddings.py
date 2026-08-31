"""Embedding providers.

LocalEmbeddingProvider runs entirely on-device via sentence-transformers —
no document or query text is ever sent to a third-party embedding API. This
is the default, and the reason the embedding step doesn't count as data
leaving the system boundary (see plan's "Key design decisions").
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from finvault.cache import Cache, digest
from finvault.metrics import record_cache
from finvault.observability import extra_fields, get_logger

logger = get_logger(__name__)


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier for the concrete model in use — stored alongside
        every ingested vector (see ingestion/pipeline.py) so a later model
        swap can be detected at query time instead of silently comparing
        vectors from two incompatible embedding spaces (see
        retrieval/retriever.py's embedding_model mismatch filter).
        """
        ...


class LocalEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # deferred: heavy import

        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        # get_embedding_dimension() replaced get_sentence_embedding_dimension()
        # in sentence-transformers 5.x; fall back for older pinned versions.
        get_dimension = (
            getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        )
        self._dimension = get_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def name(self) -> str:
        return self._model_name


class CachedEmbeddingProvider(EmbeddingProvider):
    """Wraps any provider with a shared embedding cache.

    An embedding is a pure function of (model, text), so a cache hit is
    exactly the value the model would have produced — unlike a cached
    *answer*, this needs no tenant or clearance scoping to be correct. Two
    orgs embedding the same sentence should get the same vector, and do.

    It earns its place on ingestion of overlapping corpora (the same
    boilerplate paragraph across a year of filings) and on repeated queries,
    where it removes a forward pass from the request's critical path.

    Cache keys are keyed digests, not plain hashes of the text: a plain
    digest would let anyone who can read the cache confirm that a guessed
    sentence had been embedded — see `cache.py` for why that trade-off is
    refused here even though `db.py` accepts it for `label_hash`.

    Batches are handled per item rather than all-or-nothing, so a batch of
    fifty with forty cached costs ten embeddings, not fifty.
    """

    def __init__(self, inner: EmbeddingProvider, cache: Cache, *, ttl_seconds: int) -> None:
        self._inner = inner
        self._cache = cache
        self._ttl = ttl_seconds

    def _key(self, text: str) -> str:
        # The model name is part of the key, so swapping models cannot serve
        # vectors from the previous model's incompatible space — the same
        # hazard retriever.py guards against for stored chunks.
        return f"fv:emb:{digest(self._inner.name, text)}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float] | None] = [None] * len(texts)
        misses: list[int] = []

        for index, text in enumerate(texts):
            cached = self._cache.get(self._key(text))
            if isinstance(cached, list):
                results[index] = cached
                record_cache("embedding", result="hit")
            else:
                misses.append(index)
                record_cache("embedding", result="miss")

        if misses:
            computed = self._inner.embed([texts[i] for i in misses])
            for index, vector in zip(misses, computed, strict=True):
                results[index] = vector
                self._cache.set(self._key(texts[index]), vector, ttl_seconds=self._ttl)

        if misses and len(misses) < len(texts):
            logger.debug(
                "embedding_cache_partial",
                extra=extra_fields(requested=len(texts), computed=len(misses)),
            )
        # Every slot is filled by construction: it was either a hit or is in
        # `misses`, and `computed` is the same length as `misses`.
        return [vector for vector in results if vector is not None]

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    @property
    def name(self) -> str:
        # Deliberately the inner provider's name, not a wrapped one: this is
        # stored on every vector at ingest and compared at query time, so a
        # corpus ingested with caching on must stay readable with it off.
        return self._inner.name
