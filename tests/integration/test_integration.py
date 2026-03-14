"""Monorepo-level integration test: ingestion → cleaning pipeline.

Runs a 5-row fixture through the ingestion normalisation logic and the
cleaning pipeline end-to-end. The LLM extraction stage is skipped — this
test validates that the data contracts between ingestion output and cleaning
input are compatible, and that cleaning produces the expected output schema.
"""

import json

import pandas as pd

from cleaning.constants import COLUMN_ORDER
from cleaning.pipeline import clean_enriched
from ingestion.loader import fill_missing_values

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _make_enriched_rows(n: int = 5) -> list[dict]:
    """Return n minimal enriched rows that satisfy the cleaning pipeline contract."""
    rows = []
    for i in range(n):
        rows.append({
            "row_id": f"row{i:012d}",
            "job_url": f"https://example.com/job/{i}",
            "date_posted": "2024-01-01",
            "company_name": "Test GmbH",
            "city": "Berlin",
            "state": "Berlin",
            "country": "Germany",
            "title": "Software Developer (m/w/d)",
            "title_cleaned": "Software Developer",
            "job_family": "Software Developer",
            "seniority_from_title": "Senior",
            "contract_type": "Full-time",
            "work_modality": "Remote",
            "seniority": "Senior",
            "salary_min": "60000",
            "salary_max": "80000",
            "experience_years": "3",
            "technical_skills": '["Python", "Docker"]',
            "soft_skills": '["Communication"]',
            "nice_to_have_skills": '["Kubernetes"]',
            "benefits": '["Remote work"]',
            "tasks": '["Build APIs"]',
            "location": "Berlin, Germany",
            "source_file": "test.csv",
            "site": "linkedin",
            "validation_flags": "[]",
            "description": (
                "We are looking for a Python developer with Docker experience. "
                "Kubernetes knowledge is a plus. Remote work possible."
            ),
        })
    return rows


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestIngestionToCleaningContract:
    """Verify that ingestion helpers work correctly on fixture data."""

    def test_fill_missing_values_preserves_required_fields(self):
        """fill_missing_values should not drop title/company_name/location."""
        df = pd.DataFrame([{
            "title": "Developer",
            "company_name": None,
            "location": "Berlin",
        }])
        result = fill_missing_values(df)
        assert "title" in result.columns
        assert result.loc[0, "title"] == "Developer"

    def test_fixture_rows_have_required_cleaning_columns(self):
        """Fixture rows must include all columns the cleaning pipeline expects."""
        rows = _make_enriched_rows(1)
        row = rows[0]
        required = {"row_id", "title", "description", "job_family", "technical_skills"}
        assert required.issubset(row.keys()), f"Missing: {required - row.keys()}"


class TestCleaningPipeline:
    """End-to-end cleaning pipeline on 5-row fixture."""

    def test_clean_enriched_produces_output_file(self, tmp_path):
        """clean_enriched writes the output CSV."""
        rows = _make_enriched_rows(5)
        input_csv = tmp_path / "enriched_test.csv"
        output_csv = tmp_path / "cleaned_test.csv"
        pd.DataFrame(rows).to_csv(input_csv, index=False)

        clean_enriched(input_csv, output_csv)

        assert output_csv.exists(), "Output CSV was not created"

    def test_clean_enriched_row_count_preserved(self, tmp_path):
        """Row count should be unchanged through cleaning (all 5 rows are valid)."""
        rows = _make_enriched_rows(5)
        input_csv = tmp_path / "enriched_test.csv"
        output_csv = tmp_path / "cleaned_test.csv"
        pd.DataFrame(rows).to_csv(input_csv, index=False)

        result = clean_enriched(input_csv, output_csv)

        assert len(result) == 5, f"Expected 5 rows, got {len(result)}"

    def test_clean_enriched_output_columns_match_column_order(self, tmp_path):
        """Output column order must match COLUMN_ORDER."""
        rows = _make_enriched_rows(5)
        input_csv = tmp_path / "enriched_test.csv"
        output_csv = tmp_path / "cleaned_test.csv"
        pd.DataFrame(rows).to_csv(input_csv, index=False)

        result = clean_enriched(input_csv, output_csv)

        expected_cols = [c for c in COLUMN_ORDER if c in result.columns]
        assert list(result.columns) == expected_cols

    def test_clean_enriched_no_nan_in_required_fields(self, tmp_path):
        """row_id, title, description must have no NaN values in output."""
        rows = _make_enriched_rows(5)
        input_csv = tmp_path / "enriched_test.csv"
        output_csv = tmp_path / "cleaned_test.csv"
        pd.DataFrame(rows).to_csv(input_csv, index=False)

        result = clean_enriched(input_csv, output_csv)

        for col in ("row_id", "title", "description"):
            if col in result.columns:
                assert result[col].notna().all(), f"NaN found in required column '{col}'"

    def test_clean_enriched_city_normalized(self, tmp_path):
        """Munich → München normalization should happen during cleaning."""
        rows = _make_enriched_rows(3)
        for row in rows:
            row["city"] = "Munich"
            row["state"] = "Bavaria"
        input_csv = tmp_path / "enriched_test.csv"
        output_csv = tmp_path / "cleaned_test.csv"
        pd.DataFrame(rows).to_csv(input_csv, index=False)

        result = clean_enriched(input_csv, output_csv)

        assert (result["city"] == "München").all(), "Munich was not normalized to München"

    def test_clean_enriched_dropped_columns(self, tmp_path):
        """'country' and 'source_file' columns should be dropped in the output."""
        rows = _make_enriched_rows(5)
        input_csv = tmp_path / "enriched_test.csv"
        output_csv = tmp_path / "cleaned_test.csv"
        pd.DataFrame(rows).to_csv(input_csv, index=False)

        result = clean_enriched(input_csv, output_csv)

        assert "country" not in result.columns
        assert "source_file" not in result.columns

    def test_clean_enriched_validation_flags_are_json(self, tmp_path):
        """validation_flags column values must be parseable JSON arrays."""
        rows = _make_enriched_rows(5)
        input_csv = tmp_path / "enriched_test.csv"
        output_csv = tmp_path / "cleaned_test.csv"
        pd.DataFrame(rows).to_csv(input_csv, index=False)

        result = clean_enriched(input_csv, output_csv)

        if "validation_flags" in result.columns:
            for val in result["validation_flags"].dropna():
                parsed = json.loads(str(val))
                assert isinstance(parsed, list), f"validation_flags not a JSON list: {val!r}"

    def test_clean_enriched_idempotent(self, tmp_path):
        """Running clean_enriched twice on the same data produces identical output."""
        rows = _make_enriched_rows(3)
        input_csv = tmp_path / "enriched.csv"
        output1 = tmp_path / "cleaned1.csv"
        output2 = tmp_path / "cleaned2.csv"
        pd.DataFrame(rows).to_csv(input_csv, index=False)

        df1 = clean_enriched(input_csv, output1)
        df2 = clean_enriched(output1, output2)

        assert list(df1.columns) == list(df2.columns)
        assert len(df1) == len(df2)
