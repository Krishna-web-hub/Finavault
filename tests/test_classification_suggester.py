from __future__ import annotations

from finvault.ingestion.classification import ClassificationSuggester
from finvault.ingestion.embeddings import EmbeddingProvider
from finvault.models import Classification

# The real exemplar sentences in classification.py are written using these
# exact marker words ("public", "internal", "confidential", "restricted").
# A fake that embeds text as marker-word counts lets tests exercise the
# suggester's ranking/confidence logic deterministically, without a real
# embedding model.
_MARKERS = ["public", "internal", "confidential", "restricted"]


class MarkerCountEmbeddingProvider(EmbeddingProvider):
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[text.lower().count(marker) for marker in _MARKERS] for text in texts]

    @property
    def dimension(self) -> int:
        return len(_MARKERS)

    @property
    def name(self) -> str:
        return "marker-count-fake"


def test_suggests_restricted_for_restricted_style_text() -> None:
    suggester = ClassificationSuggester(MarkerCountEmbeddingProvider())
    result = suggester.suggest(
        "Restricted internal memo for executive committee only. Do not distribute this restricted material."
    )
    assert result.predicted == Classification.RESTRICTED


def test_suggests_public_for_public_style_text() -> None:
    suggester = ClassificationSuggester(MarkerCountEmbeddingProvider())
    result = suggester.suggest("This is a public brochure for anyone to read, publicly available to the public.")
    assert result.predicted == Classification.PUBLIC


def test_suggests_confidential_for_confidential_style_text() -> None:
    suggester = ClassificationSuggester(MarkerCountEmbeddingProvider())
    result = suggester.suggest("Confidential compliance policy, confidential and for legal personnel only.")
    assert result.predicted == Classification.CONFIDENTIAL


def test_confidence_falls_back_evenly_when_text_has_no_signal() -> None:
    suggester = ClassificationSuggester(MarkerCountEmbeddingProvider())
    result = suggester.suggest("Quarterly figures and balance sheet highlights for the period.")
    # No marker word present at all -> every tier's cosine similarity is 0.0,
    # so confidence must fall back to an even split rather than divide by zero.
    assert result.confidence == 1 / len(Classification)


def test_scores_are_reported_per_tier() -> None:
    suggester = ClassificationSuggester(MarkerCountEmbeddingProvider())
    result = suggester.suggest("Restricted material for compliance officers.")
    assert set(result.scores.keys()) == {"public", "internal", "confidential", "restricted"}


def test_suggestion_never_raises_for_empty_text() -> None:
    suggester = ClassificationSuggester(MarkerCountEmbeddingProvider())
    result = suggester.suggest("")
    assert result.predicted in list(Classification)
