"""Tests for extraction/preprocessing/validate_input.py."""


import pandas as pd
import pytest

from extraction.checkpoint import Checkpoint
from extraction.preprocessing.validate_input import validate_input

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_BASE_ROW = {
    "row_id": "abc123def456",
    "source_file": "jobs_1.csv",
    "title": "Software Engineer",
    "site": "linkedin",
    "job_url": "https://linkedin.com/jobs/view/12345",
    "company_name": "ACME GmbH",
    "location": "Munich, Bavaria, Germany",
    "date_posted": "2025-01-15",
    "description": "Wir suchen einen erfahrenen Software Engineer mit Python-Kenntnissen. " * 10,
}


@pytest.fixture
def cp(tmp_path):
    checkpoint = Checkpoint(tmp_path / "test.db")
    yield checkpoint
    checkpoint.close()


@pytest.fixture
def cfg(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    return {
        "paths": {"reports_dir": reports_dir},
        "validation": {
            "min_description_length": 250,
            "date_anomaly_cutoff": "2024-01-01",
        },
    }


def _make_df(overrides: dict) -> pd.DataFrame:
    row = {**_BASE_ROW, **overrides}
    return pd.DataFrame([row])


def _get_flags(df: pd.DataFrame, col: str = "input_flags") -> list[str]:
    return df[col].iloc[0]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_valid_row_has_no_flags(cp, cfg):
    df = _make_df({})
    result, report = validate_input(df, cfg, cp)
    assert _get_flags(result) == []


# ---------------------------------------------------------------------------
# Parametrize: each flag type
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("override,expected_flag", [
    ({"title": ""}, "missing_title"),
    ({"title": "   "}, "missing_title"),
    ({"company_name": ""}, "missing_company"),
    ({"job_url": "ftp://bad-url"}, "invalid_url"),
    ({"job_url": "not-a-url"}, "invalid_url"),
    ({"description": "short"}, "short_description"),
    ({"date_posted": "not-a-date"}, "invalid_date"),
    ({"date_posted": "1992-02-14"}, "date_anomaly"),
    (
        {"description": "Datenschutz Einstellungen Ihres Browsers bitte aktivieren sie javascript"},
        "privacy_wall",
    ),
])
def test_flag_detected(override, expected_flag, cp, cfg):
    df = _make_df(override)
    # Register the row in checkpoint so mark_skipped can find it
    cp.register_rows([{"row_id": df["row_id"].iloc[0], "file_path": "jobs.csv"}])
    result, _ = validate_input(df, cfg, cp)
    assert expected_flag in _get_flags(result)


# ---------------------------------------------------------------------------
# Side effects
# ---------------------------------------------------------------------------

def test_privacy_wall_marks_skipped_in_checkpoint(cp, cfg):
    df = _make_df({
        "row_id": "priv_row_001",
        "description": "Datenschutz Einstellungen Ihres Browsers bitte aktivieren sie javascript",
    })
    cp.register_rows([{"row_id": "priv_row_001", "file_path": "jobs.csv"}])
    validate_input(df, cfg, cp)

    pending = cp.get_pending("loaded")
    assert "priv_row_001" not in pending


def test_date_anomaly_sets_date_to_na(cp, cfg):
    df = _make_df({"date_posted": "1992-02-14"})
    result, _ = validate_input(df, cfg, cp)
    assert result["date_posted"].iloc[0] is None or pd.isna(result["date_posted"].iloc[0])


def test_valid_date_not_modified(cp, cfg):
    df = _make_df({"date_posted": "2025-03-01"})
    result, _ = validate_input(df, cfg, cp)
    assert result["date_posted"].iloc[0] == "2025-03-01"


# ---------------------------------------------------------------------------
# Report structure
# ---------------------------------------------------------------------------

def test_report_json_structure(cp, cfg):
    df = _make_df({})
    _, report = validate_input(df, cfg, cp)

    assert "total_rows" in report
    assert "valid_rows" in report
    assert "flagged_counts" in report
    assert "description_length_stats" in report
    assert "company_count" in report

    stats = report["description_length_stats"]
    assert all(k in stats for k in ("min", "max", "mean", "median"))


def test_report_json_written_to_disk(cp, cfg):
    df = _make_df({})
    validate_input(df, cfg, cp)

    report_path = cfg["paths"]["reports_dir"] / "input_quality.json"
    assert report_path.exists()


def test_report_total_rows_correct(cp, cfg):
    rows = [
        {**_BASE_ROW, "row_id": f"row_{i}", "job_url": f"https://x.com/{i}"}
        for i in range(7)
    ]
    df = pd.DataFrame(rows)
    _, report = validate_input(df, cfg, cp)
    assert report["total_rows"] == 7


# ---------------------------------------------------------------------------
# Multiple flags on one row
# ---------------------------------------------------------------------------

def test_multiple_flags_on_one_row(cp, cfg):
    df = _make_df({"title": "", "company_name": "", "job_url": "bad"})
    result, _ = validate_input(df, cfg, cp)
    flags = _get_flags(result)
    assert "missing_title" in flags
    assert "missing_company" in flags
    assert "invalid_url" in flags
