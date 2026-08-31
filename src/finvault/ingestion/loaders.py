"""Document loaders — extract plain text from source files.

Plaintext returned here is transient: the pipeline chunks, embeds, and
encrypts it before anything touches disk or the vector store (see
ingestion/pipeline.py).
"""

from __future__ import annotations

from pathlib import Path

from finvault.errors import UnsupportedDocumentError

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".md", ".csv"}


def load_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix == ".docx":
        return _load_docx(path)
    if suffix == ".csv":
        return _load_csv(path)
    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="replace")
    # A typed error rather than ValueError: this reaches the API as a 415
    # with the supported list in the message, instead of being swallowed by
    # some generic `except ValueError` upstream or surfacing as a 500.
    raise UnsupportedDocumentError(
        f"Unsupported file type: {suffix} (supported: {sorted(SUPPORTED_SUFFIXES)})",
        context={"suffix": suffix},
        user_message=f"Unsupported file type '{suffix}'. Supported types: {', '.join(sorted(SUPPORTED_SUFFIXES))}.",
    )


def _load_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def _load_docx(path: Path) -> str:
    from docx import Document as DocxDocument

    doc = DocxDocument(str(path))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _load_csv(path: Path) -> str:
    # A structured profile (schema, per-column stats, sample rows), not the
    # raw row dump — see ingestion/tabular.py's module docstring for why:
    # raw rows retrieve badly against natural-language questions and drove a
    # real token-budget exhaustion (repeated failed reformulated searches).
    from finvault.ingestion.tabular import summarize_csv

    raw = path.read_text(encoding="utf-8", errors="replace")
    return summarize_csv(raw)
