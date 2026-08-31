from __future__ import annotations

from finvault.ingestion.tabular import summarize_csv


def test_summarize_csv_reports_row_and_column_counts() -> None:
    csv_text = "name,age\nAlice,30\nBob,40\n"
    summary = summarize_csv(csv_text)
    assert "2 rows" in summary
    assert "2 columns" in summary
    assert "Columns: name, age" in summary


def test_summarize_csv_profiles_numeric_columns_with_real_stats() -> None:
    csv_text = "age\n10\n20\n30\n"
    summary = summarize_csv(csv_text)
    assert "Column 'age' (numeric): min=10, max=30, mean=20" in summary


def test_summarize_csv_profiles_low_cardinality_columns_as_categorical_with_counts() -> None:
    csv_text = "sex\nmale\nfemale\nmale\n"
    summary = summarize_csv(csv_text)
    assert "Column 'sex' (categorical): female=1, male=2" in summary


def test_summarize_csv_reports_unique_count_for_high_cardinality_text_columns() -> None:
    csv_text = "name\n" + "\n".join(f"person-{i}" for i in range(20))
    summary = summarize_csv(csv_text)
    assert "Column 'name': 20 unique values" in summary


def test_summarize_csv_reports_missing_values() -> None:
    csv_text = "age\n10\n\n30\n"
    summary = summarize_csv(csv_text)
    assert "1 missing" in summary


def test_summarize_csv_includes_sample_rows() -> None:
    csv_text = "name,age\nAlice,30\nBob,40\n"
    summary = summarize_csv(csv_text)
    assert "name=Alice, age=30" in summary
    assert "name=Bob, age=40" in summary


def test_summarize_csv_handles_an_empty_file() -> None:
    assert summarize_csv("") == "This CSV file is empty."


def test_summarize_csv_handles_a_header_only_file() -> None:
    summary = summarize_csv("name,age\n")
    assert "header row only" in summary
    assert "name, age" in summary


def test_summarize_csv_caps_profiled_columns_for_very_wide_files() -> None:
    header = ",".join(f"col{i}" for i in range(60))
    row = ",".join(str(i) for i in range(60))
    summary = summarize_csv(f"{header}\n{row}\n")
    assert "Only the first 50 columns are profiled" in summary
    # The full column list is still shown (col50..col59 legitimately appear
    # there) — it's the detailed per-column stat line that's capped.
    assert "Column 'col49'" in summary
    assert "Column 'col50'" not in summary


def test_summarize_csv_output_is_dramatically_smaller_than_a_large_raw_dataset() -> None:
    """The whole point: a summary that stays small (and thus embeds/retrieves
    well) regardless of how many rows the source file has, unlike a raw
    row-per-line dump which grows linearly with row count.
    """
    header = "id,category,value"
    rows = [f"{i},cat{i % 5},{i * 1.5}" for i in range(5000)]
    csv_text = header + "\n" + "\n".join(rows)

    summary = summarize_csv(csv_text)

    assert len(summary) < len(csv_text) / 10
    assert "5000 rows" in summary
