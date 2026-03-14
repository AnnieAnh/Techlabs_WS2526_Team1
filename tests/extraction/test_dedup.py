"""Tests for extraction/dedup/row_dedup.py."""


import pandas as pd
import pytest

from extraction.checkpoint import Checkpoint
from extraction.dedup.row_dedup import _company_casing_score, deduplicate_rows

_COLS = ["row_id", "source_file", "title", "site", "job_url", "company_name",
         "location", "date_posted", "description", "input_flags"]


def _row(
    row_id: str,
    title: str | None = None,
    company: str | None = None,
    location: str = "Berlin, Germany",
    url: str | None = None,
) -> dict:
    """Create a unique row by default; override only the fields you care about."""
    return {
        "row_id": row_id,
        "source_file": "jobs_1.csv",
        "title": title if title is not None else f"Engineer {row_id}",
        "site": "linkedin",
        "job_url": url or f"https://x.com/{row_id}",
        "company_name": company if company is not None else f"Company {row_id}",
        "location": location,
        "date_posted": "2025-01-01",
        "description": "A" * 300,
        "input_flags": [],
    }


@pytest.fixture
def cp(tmp_path):
    checkpoint = Checkpoint(tmp_path / "test.db")
    yield checkpoint
    checkpoint.close()


@pytest.fixture
def cfg(tmp_path):
    deduped = tmp_path / "deduped"
    reports = tmp_path / "reports"
    deduped.mkdir()
    reports.mkdir()
    return {"paths": {"deduped_dir": deduped, "reports_dir": reports}}


def _register(cp: Checkpoint, rows: list[dict]) -> None:
    cp.register_rows([{"row_id": r["row_id"], "file_path": "jobs.csv"} for r in rows])


def test_same_url_keeps_first(cp, cfg):
    rows = [
        _row("r1", url="https://x.com/same"),
        _row("r2", url="https://x.com/same"),  # duplicate URL
    ]
    _register(cp, rows)
    df, _ = deduplicate_rows(pd.DataFrame(rows), cp, cfg)
    assert "r1" in df["row_id"].values
    assert "r2" not in df["row_id"].values


def test_different_urls_both_kept(cp, cfg):
    rows = [
        _row("r1", url="https://x.com/1"),
        _row("r2", url="https://x.com/2"),
    ]
    _register(cp, rows)
    df, _ = deduplicate_rows(pd.DataFrame(rows), cp, cfg)
    assert len(df) == 2


def test_url_dupe_marked_skipped(cp, cfg):
    rows = [
        _row("r1", url="https://x.com/same"),
        _row("r2", url="https://x.com/same"),
    ]
    _register(cp, rows)
    deduplicate_rows(pd.DataFrame(rows), cp, cfg)
    pending = cp.get_pending("loaded")
    assert "r2" not in pending  # r2 was skipped


def test_same_title_company_location_different_url_keeps_first(cp, cfg):
    rows = [
        _row("r1", title="Dev", company="ACME", location="Berlin",
             url="https://x.com/1"),
        _row("r2", title="Dev", company="ACME", location="Berlin",
             url="https://x.com/2"),  # different URL, same posting
    ]
    _register(cp, rows)
    df, _ = deduplicate_rows(pd.DataFrame(rows), cp, cfg)
    assert "r1" in df["row_id"].values
    assert "r2" not in df["row_id"].values


def test_same_title_company_different_location_keeps_both(cp, cfg):
    """Multi-location postings (same job, different city) must NOT be deduped."""
    rows = [
        _row("r1", title="Dev", company="ACME", location="Berlin, Germany",
             url="https://x.com/1"),
        _row("r2", title="Dev", company="ACME", location="Munich, Bavaria, Germany",
             url="https://x.com/2"),
    ]
    _register(cp, rows)
    df, _ = deduplicate_rows(pd.DataFrame(rows), cp, cfg)
    assert len(df) == 2
    assert "r1" in df["row_id"].values
    assert "r2" in df["row_id"].values


def test_case_insensitive_composite_dedup(cp, cfg):
    rows = [
        _row("r1", title="Software Engineer", company="ACME", location="Berlin",
             url="https://x.com/1"),
        _row("r2", title="software engineer", company="acme", location="berlin",
             url="https://x.com/2"),
    ]
    _register(cp, rows)
    df, _ = deduplicate_rows(pd.DataFrame(rows), cp, cfg)
    assert len(df) == 1


def test_report_counts_correct(cp, cfg):
    rows = [
        _row("r1", url="https://x.com/same"),
        _row("r2", url="https://x.com/same"),  # URL dupe
        _row("r3", title="Dev", company="X", location="Berlin", url="https://x.com/3"),
        _row("r4", title="Dev", company="X", location="Berlin", url="https://x.com/4"),
        _row("r5"),  # unique
    ]
    _register(cp, rows)
    df, report = deduplicate_rows(pd.DataFrame(rows), cp, cfg)

    assert report["before"] == 5
    assert report["pass_1_removed"] == 1   # r2
    assert report["pass_2_removed"] == 1   # r4
    assert report["after"] == 3
    assert report["removed"] == 2


def test_report_json_written(cp, cfg):
    rows = [_row("r1"), _row("r2")]
    _register(cp, rows)
    deduplicate_rows(pd.DataFrame(rows), cp, cfg)
    report_path = cfg["paths"]["reports_dir"] / "dedup_report.json"
    assert report_path.exists()


def test_deduped_csv_written(cp, cfg):
    rows = [_row("r1"), _row("r2")]
    _register(cp, rows)
    deduplicate_rows(pd.DataFrame(rows), cp, cfg)
    csv_files = list(cfg["paths"]["deduped_dir"].glob("deduped_*.csv"))
    assert len(csv_files) == 1


def test_no_duplicates_all_kept(cp, cfg):
    rows = [_row(f"r{i}") for i in range(5)]
    _register(cp, rows)
    df, report = deduplicate_rows(pd.DataFrame(rows), cp, cfg)
    assert len(df) == 5
    assert report["removed"] == 0


# ---------------------------------------------------------------------------
# Company name casing tie-breaking (Item 15)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name, expected_positive", [
    ("Acme GmbH", True),       # proper casing → positive score
    ("acme gmbh", False),      # all-lowercase → negative/zero score
    ("ACME GMBH", True),       # all-caps → positive (caps start)
    ("", False),               # empty → 0
])
def test_company_casing_score(name, expected_positive):
    score = _company_casing_score(name)
    if expected_positive:
        assert score > 0
    else:
        assert score <= 0


def test_casing_tiebreak_prefers_proper_case(cp, cfg):
    """When two rows differ only in company casing, proper case is kept."""
    rows = [
        _row("r1", title="Dev", company="acme gmbh", location="Berlin",
             url="https://x.com/1"),
        _row("r2", title="Dev", company="Acme GmbH", location="Berlin",
             url="https://x.com/2"),
    ]
    _register(cp, rows)
    df, _ = deduplicate_rows(pd.DataFrame(rows), cp, cfg)
    assert len(df) == 1
    # The properly-cased "Acme GmbH" should win
    assert df.iloc[0]["company_name"] == "Acme GmbH"
