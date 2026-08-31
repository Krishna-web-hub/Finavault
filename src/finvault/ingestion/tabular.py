"""CSV summarization: turns raw tabular text into a natural-language-shaped
profile (schema, per-column stats, sample rows) instead of a raw row dump.

Why this exists: naive paragraph/line chunking plus semantic embedding of
raw CSV rows retrieves badly against natural-language questions — a row
like "891,0,3,Heikkinen,female,26,..." has almost no semantic overlap with
"what does this dataset show", so the retriever agent ends up reformulating
and re-searching repeatedly without ever finding a good match, burning
through the shared token budget before an answer is ever produced (see
agents/retriever_agent.py's max_iterations and agents/base.py's
TokenBudget — this was reproduced live: 23k tokens spent by the retriever
alone, on a vague question against a raw-ingested CSV, before the request
failed closed). A structured summary is prose-shaped and answers the kind
of questions people actually ask about a dataset ("how many rows", "what
columns", "what's the average X") directly from retrieved context, the
same way a paragraph of financial narrative does.

This is a summary, not a substitute for the data: exact per-row lookups
("what was passenger 5's fare") are explicitly out of scope — that's a
structured-query problem (text-to-SQL or similar), not a RAG-over-prose
one, and isn't what this pipeline is for.
"""

from __future__ import annotations

import csv
import io

_MAX_COLUMNS_PROFILED = 50
_MAX_SAMPLE_ROWS = 5
_MAX_CATEGORY_VALUES_LISTED = 8


def summarize_csv(text: str) -> str:
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return "This CSV file is empty."

    header, data_rows = rows[0], rows[1:]
    if not data_rows:
        return f"This CSV file has a header row only, with {len(header)} column(s): {', '.join(header)}. No data rows."

    truncated = len(header) > _MAX_COLUMNS_PROFILED
    profiled_header = header[:_MAX_COLUMNS_PROFILED]

    columns: dict[str, list[str]] = {name: [] for name in profiled_header}
    for row in data_rows:
        for i, name in enumerate(profiled_header):
            columns[name].append(row[i] if i < len(row) else "")

    parts: list[str] = [
        f"This is a tabular dataset (CSV) with {len(data_rows)} rows and {len(header)} columns."
        + (f" Only the first {_MAX_COLUMNS_PROFILED} columns are profiled below." if truncated else ""),
        "Columns: " + ", ".join(header) + ".",
    ]

    for name in profiled_header:
        parts.append(_profile_column(name, columns[name]))

    parts.append("Sample rows:")
    for row in data_rows[:_MAX_SAMPLE_ROWS]:
        parts.append(", ".join(f"{name}={row[i] if i < len(row) else ''}" for i, name in enumerate(profiled_header)))

    return "\n\n".join(parts)


def _profile_column(name: str, values: list[str]) -> str:
    non_empty = [v for v in values if v.strip() != ""]
    missing = len(values) - len(non_empty)
    missing_note = f", {missing} missing" if missing else ""

    numeric = _try_numeric(non_empty)
    if numeric:
        lo, hi = min(numeric), max(numeric)
        mean = sum(numeric) / len(numeric)
        return f"Column '{name}' (numeric): min={_fmt(lo)}, max={_fmt(hi)}, mean={_fmt(mean)}{missing_note}."

    unique = sorted(set(non_empty))
    if len(unique) <= _MAX_CATEGORY_VALUES_LISTED:
        counts = {v: non_empty.count(v) for v in unique}
        breakdown = ", ".join(f"{v}={counts[v]}" for v in unique)
        return f"Column '{name}' (categorical): {breakdown}{missing_note}."
    return f"Column '{name}': {len(unique)} unique values{missing_note}."


def _try_numeric(values: list[str]) -> list[float] | None:
    """None means "not a numeric column" (at least one non-numeric value) —
    distinguished from an empty list, which means every value was blank."""
    if not values:
        return None
    result = []
    for v in values:
        try:
            result.append(float(v))
        except ValueError:
            # Deliberately silent, and the one except in this codebase that
            # logs nothing: this IS the numeric-column test, so a
            # non-numeric value is the expected answer "no", not a failure.
            # Logging it would emit a record per non-numeric cell.
            return None
    return result


def _fmt(n: float) -> str:
    return f"{n:.4g}"
