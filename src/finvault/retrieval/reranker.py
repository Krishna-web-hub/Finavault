"""Optional cross-encoder reranking signal.

Unlike an embedding model (which encodes query and document independently,
then compares them by vector geometry), a cross-encoder scores a
(query, document) pair jointly in one forward pass — generally more
accurate, but one inference call per candidate instead of one per query.

Like the BM25 signal in retriever.py, this only ever runs over the
candidate set that's already been ACL-filtered and decrypted for the
current request — no new persisted artifact, no new place plaintext-derived
data lives outside memory.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Reranker(ABC):
    @abstractmethod
    def score(self, query: str, texts: list[str]) -> list[float]:
        """Returns one relevance score per text, higher is more relevant.
        Scores are only meaningful relative to each other for a given query
        — not comparable across calls or against vector/BM25 scores.
        """
        ...


class LocalCrossEncoderReranker(Reranker):
    """Runs entirely on-device via sentence-transformers' CrossEncoder — same
    "no third-party API, no document text leaves the system boundary"
    property as LocalEmbeddingProvider.
    """

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder  # deferred: heavy import

        self._model = CrossEncoder(model_name)

    def score(self, query: str, texts: list[str]) -> list[float]:
        pairs = [(query, text) for text in texts]
        return self._model.predict(pairs).tolist()
