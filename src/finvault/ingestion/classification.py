"""Advisory sensitivity-tier classifier: embedding-similarity-to-exemplars.

Suggests a Classification for a document's text by comparing its embedding
against a small set of hand-written exemplar sentences per tier, via cosine
similarity to each tier's exemplar centroid. This is a lightweight zero-shot
classifier (no training data, reuses whatever EmbeddingProvider ingestion
already has), not a trained model.

Advisory only: see ingestion/pipeline.py, where the caller-supplied
classification remains authoritative and the suggestion is only logged
alongside it for drift-tracking. Under-classifying a restricted document is
far more costly than an unnecessary human double-check, so this must never
auto-assign a tier.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from finvault.ingestion.embeddings import EmbeddingProvider
from finvault.models import Classification

# Hand-written exemplars per tier, drawn from the kind of phrasing that
# actually distinguishes these documents in practice (see sample_docs/) —
# explicit distribution markings ("not for external release", "executive
# committee ... only") are a strong signal humans already write into these
# documents, so leaning on them is a reasonable zero-shot heuristic.
_EXEMPLARS: dict[Classification, list[str]] = {
    Classification.PUBLIC: [
        "This is a public marketing brochure describing our services to prospective clients.",
        "Press release announcing quarterly results to the general public.",
        "General product information available to anyone on our public website.",
    ],
    Classification.INTERNAL: [
        "Internal quarterly report prepared for internal distribution only, not for external release.",
        "Internal memo summarizing team performance for employees.",
        "Company-wide internal announcement about office policy.",
    ],
    Classification.CONFIDENTIAL: [
        "Confidential compliance policy for compliance and legal personnel only.",
        "Confidential due diligence procedures restricted to authorized staff.",
        "This document contains confidential business strategy not to be shared outside the firm.",
    ],
    Classification.RESTRICTED: [
        "Restricted internal memo for executive committee and compliance officer access only, do not distribute.",
        "This matter is subject to attorney-client privilege and must not be disclosed.",
        "Restricted review of a specific client account under investigation for suspicious activity.",
    ],
}


def _cosine(a: list[float], b: list[float]) -> float:
    # strict=True: two vectors of different length is a bug (an embedding
    # model mismatch), and silently scoring the shorter overlap would
    # produce a plausible-looking similarity from incomparable inputs.
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _centroid(vectors: list[list[float]]) -> list[float]:
    dim = len(vectors[0])
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]


@dataclass(frozen=True)
class ClassificationSuggestion:
    predicted: Classification
    # Predicted tier's share of total similarity mass across all tiers, in
    # [0, 1] — a relative confidence, not a calibrated probability.
    confidence: float
    # Raw cosine similarity per tier (keyed by tier value), kept for
    # auditing/debugging so a reviewer can see how close the call was.
    scores: dict[str, float] = field(default_factory=dict)


class ClassificationSuggester:
    """Advisory sensitivity-tier suggestion. Never assigns a classification —
    callers decide what, if anything, to do with the suggestion.
    """

    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        self._embedding_provider = embedding_provider
        self._centroids: dict[Classification, list[float]] | None = None

    def _ensure_centroids(self) -> dict[Classification, list[float]]:
        if self._centroids is None:
            self._centroids = {
                tier: _centroid(self._embedding_provider.embed(texts)) for tier, texts in _EXEMPLARS.items()
            }
        return self._centroids

    def suggest(self, text: str) -> ClassificationSuggestion:
        centroids = self._ensure_centroids()
        doc_vector = self._embedding_provider.embed([text])[0]

        scores = {tier: _cosine(doc_vector, centroid) for tier, centroid in centroids.items()}
        predicted = max(scores, key=scores.get)  # type: ignore[arg-type]

        # Cosine similarity can be negative; shift to non-negative before
        # normalizing into a confidence share so a tier that's merely "less
        # negative" than the others doesn't read as a strong signal.
        shifted = {tier: max(score, 0.0) for tier, score in scores.items()}
        total = sum(shifted.values())
        confidence = (shifted[predicted] / total) if total > 0 else 1 / len(scores)

        return ClassificationSuggestion(
            predicted=predicted,
            confidence=confidence,
            scores={tier.value: score for tier, score in scores.items()},
        )
