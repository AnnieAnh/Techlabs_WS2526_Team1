"""Tests for extraction/reporting/quality.py."""

import json
from pathlib import Path

import pandas as pd
import pytest

from extraction.reporting.quality import generate_quality_report

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(tmp_path: Path) -> dict:
    return {"paths": {"reports_dir": tmp_path / "reports"}}


def _result(row_id: str, **data_overrides) -> dict:
    """Return an extraction result dict with sensible defaults (new schema)."""
    data: dict = {
        "contract_type": "Full-time",
        "work_modality": "Hybrid",
        "seniority": "Senior",
        "salary_min": 80_000,
        "salary_max": 100_000,
        "technical_skills": ["Python", "SQL"],
        "soft_skills": ["Communication"],
        "nice_to_have_skills": ["Docker"],
        "experience_years": 3,
    }
    data.update(data_overrides)
    return {"row_id": row_id, "data": data, "validation_flags": []}


def _df(n: int = 4, **col_overrides) -> pd.DataFrame:
    """Build a test DataFrame with regex_* prefixed Tier-1 columns.

    At step 6 (validate) where quality report runs, Tier-1 columns still
    have the regex_ prefix. Callers pass shorthand names (e.g. contract_type)
    which are auto-prefixed with regex_.
    """
    # Map shorthand names to regex_ prefixed names for Tier-1 columns
    _TIER1_PREFIX_MAP = {
        "contract_type": "regex_contract_type",
        "work_modality": "regex_work_modality",
        "seniority_from_title": "regex_seniority_from_title",
    }
    rows = []
    for i in range(n):
        row = {
            "row_id": f"r{i}",
            "city": "Munich" if i % 2 == 0 else None,
            "state": "Bavaria" if i % 2 == 0 else "Berlin",
            "country": "Germany",
            "regex_contract_type": "Full-time",
            "regex_work_modality": "Hybrid",
            "regex_seniority_from_title": "Senior",
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    for col, values in col_overrides.items():
        # Auto-prefix Tier-1 columns with regex_
        actual_col = _TIER1_PREFIX_MAP.get(col, col)
        if isinstance(values, list):
            df[actual_col] = values[:n]
        else:
            df[actual_col] = values
    return df


# ---------------------------------------------------------------------------
# Report structure
# ---------------------------------------------------------------------------


def test_report_has_all_required_sections(tmp_path):
    report = generate_quality_report([_result("r1")], None, _cfg(tmp_path))
    for key in (
        "summary", "field_coverage", "distributions", "top_skills",
        "top_soft_skills", "hallucination_summary", "benefit_coverage",
        "salary_stats", "location_stats", "validation_summary", "quality_concerns",
    ):
        assert key in report, f"Missing key: {key}"


def test_report_summary_total_rows(tmp_path):
    results = [_result(f"r{i}") for i in range(7)]
    report = generate_quality_report(results, None, _cfg(tmp_path))
    assert report["summary"]["total_rows"] == 7


def test_report_empty_results(tmp_path):
    report = generate_quality_report([], None, _cfg(tmp_path))
    assert report["summary"]["total_rows"] == 0
    assert report["top_skills"] == []
    assert report["quality_concerns"] == []


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------


def test_report_saves_json(tmp_path):
    generate_quality_report([_result("r1")], None, _cfg(tmp_path), save=True)
    path = tmp_path / "reports" / "quality_report.json"
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["summary"]["total_rows"] == 1


def test_report_saves_markdown(tmp_path):
    generate_quality_report([_result("r1")], None, _cfg(tmp_path), save=True)
    md = tmp_path / "reports" / "quality_report.md"
    assert md.exists()
    content = md.read_text(encoding="utf-8")
    assert "# Pipeline Quality Report" in content
    assert "Field Coverage" in content


def test_report_no_save_skips_files(tmp_path):
    generate_quality_report([_result("r1")], None, _cfg(tmp_path), save=False)
    assert not (tmp_path / "reports" / "quality_report.json").exists()


# ---------------------------------------------------------------------------
# Field coverage
# ---------------------------------------------------------------------------


def test_field_coverage_full_row(tmp_path):
    results = [_result("r1")]
    report = generate_quality_report(results, None, _cfg(tmp_path))
    cov = report["field_coverage"]
    assert cov.get("contract_type") == 100.0
    assert cov.get("seniority") == 100.0
    assert cov.get("technical_skills") == 100.0


def test_field_coverage_null_not_counted(tmp_path):
    results = [
        _result("r1", seniority="Senior", salary_min=None),
        _result("r2", seniority=None, salary_min=60_000),
    ]
    report = generate_quality_report(results, None, _cfg(tmp_path))
    cov = report["field_coverage"]
    assert cov.get("seniority") == 50.0
    assert cov.get("salary_min") == 50.0


def test_field_coverage_empty_list_not_counted(tmp_path):
    results = [
        _result("r1", technical_skills=[]),
        _result("r2", technical_skills=["Python"]),
    ]
    report = generate_quality_report(results, None, _cfg(tmp_path))
    assert report["field_coverage"].get("technical_skills") == 50.0


def test_field_coverage_na_string_not_counted(tmp_path):
    results = [
        _result("r1", work_modality=None),
        _result("r2", work_modality="Remote"),
    ]
    report = generate_quality_report(results, None, _cfg(tmp_path))
    assert report["field_coverage"].get("work_modality") == 50.0


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------


def test_distributions_contract_type(tmp_path):
    """Distributions now read Tier-1 fields from the DataFrame (BUG-1 fix)."""
    results = [_result("r1"), _result("r2"), _result("r3")]
    df = _df(3, contract_type=["Full-time", "Full-time", "Part-time"])
    report = generate_quality_report(results, df, _cfg(tmp_path))
    dist = report["distributions"]["contract_type"]
    assert dist["Full-time"] == 2
    assert dist["Part-time"] == 1


def test_distributions_null_counted_as_null(tmp_path):
    results = [_result("r1"), _result("r2")]
    df = _df(2, contract_type=[None, "Full-time"])
    report = generate_quality_report(results, df, _cfg(tmp_path))
    dist = report["distributions"]["contract_type"]
    assert dist.get("null") == 1
    assert dist.get("Full-time") == 1


def test_distributions_ordered_by_frequency(tmp_path):
    results = [_result(f"r{i}") for i in range(4)]
    df = _df(4, contract_type=["Full-time", "Full-time", "Full-time", "Part-time"])
    report = generate_quality_report(results, df, _cfg(tmp_path))
    dist = report["distributions"]["contract_type"]
    values = list(dist.keys())
    assert values[0] == "Full-time"  # most frequent first


def test_distributions_empty_when_no_df(tmp_path):
    """Distributions return empty dicts when df is None (graceful fallback)."""
    results = [_result("r1")]
    report = generate_quality_report(results, None, _cfg(tmp_path))
    assert report["distributions"]["contract_type"] == {}
    assert report["distributions"]["work_modality"] == {}
    assert report["distributions"]["seniority"] == {}


def test_distributions_seniority_reads_from_title_column(tmp_path):
    """Seniority distribution reads from seniority_from_title DataFrame column."""
    results = [_result("r1"), _result("r2"), _result("r3")]
    df = _df(3, seniority_from_title=["Senior", "Junior", "Senior"])
    report = generate_quality_report(results, df, _cfg(tmp_path))
    dist = report["distributions"]["seniority"]
    assert dist["Senior"] == 2
    assert dist["Junior"] == 1


# ---------------------------------------------------------------------------
# Top skills
# ---------------------------------------------------------------------------


def test_top_skills_ordered_by_frequency(tmp_path):
    results = [
        _result("r1", technical_skills=["Python", "SQL"]),
        _result("r2", technical_skills=["Python", "Docker"]),
        _result("r3", technical_skills=["Python"]),
    ]
    report = generate_quality_report(results, None, _cfg(tmp_path))
    top = report["top_skills"]
    assert top[0]["skill"] == "Python"
    assert top[0]["count"] == 3


def test_top_skills_limited_to_n(tmp_path):
    # Create results with 60 unique skills
    unique_skills = [f"Skill{i}" for i in range(60)]
    results = [_result("r1", technical_skills=unique_skills)]
    report = generate_quality_report(results, None, _cfg(tmp_path))
    assert len(report["top_skills"]) <= 50


def test_top_skills_empty_when_no_skills(tmp_path):
    results = [_result("r1", technical_skills=[])]
    report = generate_quality_report(results, None, _cfg(tmp_path))
    assert report["top_skills"] == []


# ---------------------------------------------------------------------------
# Salary stats
# ---------------------------------------------------------------------------


def test_salary_stats_basic(tmp_path):
    results = [
        _result("r1", salary_min=60_000, salary_max=80_000),
        _result("r2", salary_min=80_000, salary_max=100_000),
        _result("r3", salary_min=None, salary_max=None),
    ]
    report = generate_quality_report(results, None, _cfg(tmp_path))
    sal = report["salary_stats"]
    assert sal["rows_with_salary"] == 2
    assert sal["salary_min"]["min"] == 60_000
    assert sal["salary_min"]["max"] == 80_000
    assert sal["salary_min"]["count"] == 2


def test_salary_stats_no_salary_rows(tmp_path):
    results = [_result("r1", salary_min=None, salary_max=None)]
    report = generate_quality_report(results, None, _cfg(tmp_path))
    assert report["salary_stats"]["rows_with_salary"] == 0
    assert report["salary_stats"]["salary_min"]["count"] == 0


def test_salary_coverage_pct(tmp_path):
    results = [
        _result("r1", salary_min=70_000),
        _result("r2", salary_min=None),
    ]
    report = generate_quality_report(results, None, _cfg(tmp_path))
    assert report["salary_stats"]["coverage_pct"] == 50.0


# ---------------------------------------------------------------------------
# Location stats
# ---------------------------------------------------------------------------


def test_location_stats_none_df(tmp_path):
    report = generate_quality_report([_result("r1")], None, _cfg(tmp_path))
    assert report["location_stats"] == {}


def test_location_stats_with_df(tmp_path):
    report = generate_quality_report([_result(f"r{i}") for i in range(4)], _df(4), _cfg(tmp_path))
    loc = report["location_stats"]
    assert "top_cities" in loc
    assert "Munich" in loc["top_cities"]
    assert "top_states" in loc
    assert "country_distribution" in loc


def test_location_stats_excludes_na_from_top_cities(tmp_path):
    report = generate_quality_report([_result(f"r{i}") for i in range(4)], _df(4), _cfg(tmp_path))
    assert None not in report["location_stats"].get("top_cities", {})


# ---------------------------------------------------------------------------
# Validation summary
# ---------------------------------------------------------------------------


def test_validation_summary_with_flags(tmp_path):
    results = [
        {
            "row_id": "r1",
            "data": {},
            "validation_flags": [
                {
                    "field": "salary_min", "rule": "min_greater_than_max",
                    "severity": "error", "message": "x",
                }
            ],
        },
        _result("r2"),  # no flags
    ]
    report = generate_quality_report(results, None, _cfg(tmp_path))
    vs = report["validation_summary"]
    assert vs["rows_with_flags"] == 1
    assert vs["rows_clean"] == 1
    assert "min_greater_than_max" in vs["flags_by_rule"]
    assert vs["flags_by_severity"].get("error") == 1


def test_validation_summary_no_flags(tmp_path):
    results = [_result("r1")]
    report = generate_quality_report(results, None, _cfg(tmp_path))
    vs = report["validation_summary"]
    assert vs["rows_with_flags"] == 0
    assert vs["rows_clean"] == 1
    assert vs["clean_pct"] == 100.0


# ---------------------------------------------------------------------------
# Quality concerns
# ---------------------------------------------------------------------------


def test_quality_concerns_low_coverage(tmp_path):
    results = [
        {
            "row_id": "r1",
            "data": {
                "contract_type": None,
                "seniority": None,
                "salary_min": None,
                "technical_skills": [],
                "description_language": "EN",  # only this has coverage
            },
            "validation_flags": [],
        }
    ]
    report = generate_quality_report(results, None, _cfg(tmp_path))
    concern_fields = [c.split(":")[0] for c in report["quality_concerns"]]
    assert "contract_type" in concern_fields or "seniority" in concern_fields


def test_quality_concerns_high_coverage_no_flag(tmp_path):
    results = [_result(f"r{i}") for i in range(5)]
    report = generate_quality_report(results, None, _cfg(tmp_path))
    # All fields are 100% covered — no concerns expected
    assert report["quality_concerns"] == []


# ---------------------------------------------------------------------------
# Markdown content
# ---------------------------------------------------------------------------


def test_markdown_contains_key_sections(tmp_path):
    results = [_result("r1")]
    generate_quality_report(results, None, _cfg(tmp_path), save=True)
    md = (tmp_path / "reports" / "quality_report.md").read_text(encoding="utf-8")
    assert "## Field Coverage" in md
    assert "## Top 20 Technical Skills" in md
    assert "## Salary Statistics" in md


def test_markdown_has_top_skill(tmp_path):
    results = [_result("r1", technical_skills=["Python", "Rust"])]
    generate_quality_report(results, None, _cfg(tmp_path), save=True)
    md = (tmp_path / "reports" / "quality_report.md").read_text(encoding="utf-8")
    assert "Python" in md


# ---------------------------------------------------------------------------
# New sections: top_soft_skills, hallucination_summary, benefit_coverage
# ---------------------------------------------------------------------------


def test_top_soft_skills_populated(tmp_path):
    """E7: top_soft_skills returns non-empty list when soft_skills data is present."""
    results = [
        _result("r1", soft_skills=["Communication", "Teamwork"]),
        _result("r2", soft_skills=["Communication", "Leadership"]),
        _result("r3", soft_skills=["Teamwork"]),
    ]
    report = generate_quality_report(results, None, _cfg(tmp_path))
    top_soft = report["top_soft_skills"]
    assert len(top_soft) > 0
    # Communication appeared twice — should be first
    assert top_soft[0]["skill"] == "Communication"
    assert top_soft[0]["count"] == 2


def test_top_soft_skills_empty_when_no_soft_skills(tmp_path):
    """E7: top_soft_skills is empty when no soft_skills in data."""
    results = [_result("r1", soft_skills=[])]
    report = generate_quality_report(results, None, _cfg(tmp_path))
    assert report["top_soft_skills"] == []


def test_hallucination_summary_populated(tmp_path):
    """E7: hallucination_summary has expected keys and counts skill flags."""
    results = [
        {
            "row_id": "r1",
            "data": {"technical_skills": ["Python"]},
            "validation_flags": [
                {
                    "rule": "skill_not_in_description",
                    "field": "technical_skills",
                    "severity": "warning",
                    "message": "'Python' not grounded in description",
                }
            ],
        },
        _result("r2"),  # no flags
    ]
    report = generate_quality_report(results, None, _cfg(tmp_path))
    hal = report["hallucination_summary"]
    assert "total_skill_flags" in hal
    assert "high_hallucination_rows" in hal
    assert "top_falsely_flagged" in hal
    assert hal["total_skill_flags"] == 1
    assert hal["high_hallucination_rows"] == 0
    assert "Python" in hal["top_falsely_flagged"]


def test_hallucination_summary_empty_no_flags(tmp_path):
    """E7: hallucination_summary shows zeros when no hallucination flags."""
    results = [_result("r1")]
    report = generate_quality_report(results, None, _cfg(tmp_path))
    hal = report["hallucination_summary"]
    assert hal["total_skill_flags"] == 0
    assert hal["high_hallucination_rows"] == 0


def test_benefit_coverage_populated(tmp_path):
    """E7: benefit_coverage tracks % of rows with non-empty benefits."""
    results = [
        _result("r1", benefits=["Remote work", "Gym"]),
        _result("r2", benefits=[]),
        _result("r3", benefits=["Bonus"]),
    ]
    report = generate_quality_report(results, None, _cfg(tmp_path))
    bc = report["benefit_coverage"]
    assert "rows_with_benefits_pct" in bc
    # 2 out of 3 rows have benefits → 66.7%
    assert bc["rows_with_benefits_pct"] == pytest.approx(66.7, abs=0.5)


def test_markdown_contains_new_sections(tmp_path):
    """E7: Markdown report contains new section headers."""
    results = [
        _result("r1", soft_skills=["Communication"]),
        {
            "row_id": "r2",
            "data": {"technical_skills": ["Go"]},
            "validation_flags": [
                {
                    "rule": "skill_not_in_description",
                    "field": "technical_skills",
                    "severity": "warning",
                    "message": "Skill 'Go' not found in description",
                }
            ],
        },
    ]
    generate_quality_report(results, None, _cfg(tmp_path), save=True)
    md = (tmp_path / "reports" / "quality_report.md").read_text(encoding="utf-8")
    assert "## Top 20 Soft Skills" in md
    assert "## Hallucination Summary" in md
    assert "## Benefit Coverage" in md
