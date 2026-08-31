from __future__ import annotations

from finvault.ingestion.loaders import load_text


def test_load_text_summarizes_csv_instead_of_dumping_raw_rows(tmp_path) -> None:
    """.csv now routes through ingestion/tabular.py's summarize_csv rather
    than a raw read_text passthrough — see that module's docstring for why
    (raw rows retrieve badly and drove a real token-budget exhaustion).
    """
    path = tmp_path / "data.csv"
    path.write_text("name,age\nAlice,30\nBob,40\n")

    text = load_text(path)

    assert "2 rows" in text
    assert "Column 'age' (numeric)" in text
    # The raw row text should not appear verbatim as a CSV line — it's been
    # reshaped into the "name=Alice, age=30" sample-row format instead.
    assert "Alice,30" not in text
    assert "name=Alice, age=30" in text


def test_load_text_still_reads_txt_and_md_verbatim(tmp_path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("Plain narrative text, unchanged.")

    assert load_text(path) == "Plain narrative text, unchanged."
