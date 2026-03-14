"""Tests for extraction/exporter.py."""

import pandas as pd

from extraction.exporter import _column_order, export_enriched_csv, merge_results

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _df(n: int = 3) -> pd.DataFrame:
    """Base DataFrame including regex_ Tier 1 columns (as produced by regex_extract stage)."""
    return pd.DataFrame([
        {
            "row_id": f"r{i}",
            "source_file": "jobs_1.csv" if i < 2 else "jobs_2.csv",
            "title": f"Engineer {i}",
            "job_url": f"https://example.com/{i}",
            "company_name": "ACME",
            "location": "Munich, Bavaria, Germany",
            "date_posted": "2025-01-01",
            "city": "Munich",
            "state": "Bavaria",
            "country": "Germany",
            # Tier 1 regex_ columns (promoted to canonical names by merge_results)
            "regex_contract_type": "Full-time",
            "regex_work_modality": "Hybrid",
            "regex_salary_min": 70_000 + i * 5_000,
            "regex_salary_max": 90_000,
            "regex_experience_years": 3,
            "regex_seniority_from_title": "Senior",
            "regex_languages": [],
            "regex_education_level": None,
        }
        for i in range(n)
    ])


def _result(row_id: str, **data_overrides) -> dict:
    """Tier 2 LLM extraction result."""
    data: dict = {
        "technical_skills": ["Python", "SQL"],
        "soft_skills": ["communication"],
        "nice_to_have_skills": ["Docker"],
        "benefits": ["30 days vacation"],
        "tasks": ["Build APIs", "Code review"],
        "job_family": "Backend Developer",
        "job_summary": "Backend role in Munich.",
    }
    data.update(data_overrides)
    return {"row_id": row_id, "data": data, "validation_flags": []}


# ---------------------------------------------------------------------------
# merge_results
# ---------------------------------------------------------------------------


def test_merge_adds_tier2_columns():
    df = _df(2)
    results = [_result("r0"), _result("r1")]
    enriched = merge_results(df, results)
    assert "technical_skills" in enriched.columns
    assert "job_family" in enriched.columns


def test_merge_promotes_tier1_columns():
    df = _df(2)
    enriched = merge_results(df, [_result("r0"), _result("r1")])
    # regex_ prefix removed; canonical name present
    assert "contract_type" in enriched.columns
    assert "salary_min" in enriched.columns
    assert "regex_contract_type" not in enriched.columns


def test_merge_correct_tier2_values():
    df = _df(2)
    results = [
        _result("r0", technical_skills=["Python"]),
        _result("r1", technical_skills=["Java"]),
    ]
    enriched = merge_results(df, results)
    assert enriched.loc[enriched["row_id"] == "r0", "technical_skills"].iloc[0] == ["Python"]
    assert enriched.loc[enriched["row_id"] == "r1", "technical_skills"].iloc[0] == ["Java"]


def test_merge_correct_tier1_values():
    df = _df(2)
    enriched = merge_results(df, [_result("r0"), _result("r1")])
    r0_salary = enriched.loc[enriched["row_id"] == "r0", "salary_min"].iloc[0]
    assert r0_salary == 70_000  # from _df fixture


def test_merge_preserves_original_columns():
    df = _df(2)
    results = [_result("r0"), _result("r1")]
    enriched = merge_results(df, results)
    for col in ("title", "job_url", "city", "state"):
        assert col in enriched.columns


def test_merge_adds_validation_flags():
    df = _df(1)
    r = _result("r0")
    r["validation_flags"] = [{"rule": "min_greater_than_max", "severity": "error", "message": "x"}]
    enriched = merge_results(df, [r])
    flags = enriched.loc[enriched["row_id"] == "r0", "validation_flags"].iloc[0]
    assert isinstance(flags, list)
    assert len(flags) == 1


def test_merge_row_count_unchanged():
    df = _df(3)
    results = [_result(f"r{i}") for i in range(3)]
    enriched = merge_results(df, results)
    assert len(enriched) == 3


def test_merge_empty_results():
    df = _df(2)
    enriched = merge_results(df, [])
    assert len(enriched) == 2
    # Tier 2 columns added as None when no results
    assert "technical_skills" in enriched.columns
    # Tier 1 columns promoted from regex_ columns
    assert "contract_type" in enriched.columns


def test_merge_partial_results():
    df = _df(3)
    # Only r0 and r1 have extraction results — r2 gets nulls for Tier 2
    results = [_result("r0"), _result("r1")]
    enriched = merge_results(df, results)
    assert len(enriched) == 3
    r2_skills = enriched.loc[enriched["row_id"] == "r2", "technical_skills"]
    assert r2_skills.isna().iloc[0]


def test_merge_ignores_result_with_no_row_id():
    df = _df(1)
    results = [{"data": {"technical_skills": ["Go"]}, "validation_flags": []}]
    enriched = merge_results(df, results)
    # No row_id → not merged; original row preserved
    assert len(enriched) == 1


# ---------------------------------------------------------------------------
# Column ordering
# ---------------------------------------------------------------------------


def test_column_order_puts_row_id_first():
    df = _df(1)
    enriched = merge_results(df, [_result("r0")])
    order = _column_order(enriched)
    assert order[0] == "row_id"


def test_column_order_original_before_extracted():
    df = _df(1)
    enriched = merge_results(df, [_result("r0")])
    order = _column_order(enriched)
    city_pos = order.index("city") if "city" in order else 999
    skills_pos = order.index("technical_skills") if "technical_skills" in order else 999
    assert city_pos < skills_pos


def test_column_order_includes_all_columns():
    df = _df(1)
    enriched = merge_results(df, [_result("r0")])
    order = _column_order(enriched)
    assert set(order) == set(enriched.columns)


# ---------------------------------------------------------------------------
# export_enriched_csv
# ---------------------------------------------------------------------------


def test_export_creates_combined_csv(tmp_path):
    cfg = {"paths": {"extracted_dir": tmp_path / "extracted"}}
    df = _df(3)
    enriched = merge_results(df, [_result(f"r{i}") for i in range(3)])
    exported = export_enriched_csv(enriched, cfg)
    assert "combined" in exported
    assert exported["combined"].exists()


def test_export_combined_has_correct_row_count(tmp_path):
    cfg = {"paths": {"extracted_dir": tmp_path / "extracted"}}
    df = _df(3)
    enriched = merge_results(df, [_result(f"r{i}") for i in range(3)])
    exported = export_enriched_csv(enriched, cfg)
    import pandas as pd
    loaded = pd.read_csv(exported["combined"])
    assert len(loaded) == 3


def test_export_per_source_file(tmp_path):
    cfg = {"paths": {"extracted_dir": tmp_path / "extracted"}}
    df = _df(3)  # 2 rows in jobs_1.csv, 1 row in jobs_2.csv
    enriched = merge_results(df, [_result(f"r{i}") for i in range(3)])
    exported = export_enriched_csv(enriched, cfg)
    assert "jobs_1.csv" in exported
    assert "jobs_2.csv" in exported
    assert exported["jobs_1.csv"].exists()


def test_export_per_file_correct_row_count(tmp_path):
    cfg = {"paths": {"extracted_dir": tmp_path / "extracted"}}
    df = _df(3)
    enriched = merge_results(df, [_result(f"r{i}") for i in range(3)])
    exported = export_enriched_csv(enriched, cfg)
    import pandas as pd
    f1 = pd.read_csv(exported["jobs_1.csv"])
    assert len(f1) == 2
    f2 = pd.read_csv(exported["jobs_2.csv"])
    assert len(f2) == 1


def test_export_creates_directory(tmp_path):
    cfg = {"paths": {"extracted_dir": tmp_path / "nested" / "extracted"}}
    df = _df(1)
    enriched = merge_results(df, [_result("r0")])
    exported = export_enriched_csv(enriched, cfg)
    assert exported["combined"].exists()


def test_export_utf8_encoding(tmp_path):
    cfg = {"paths": {"extracted_dir": tmp_path / "extracted"}}
    df = _df(1)
    df.loc[0, "company_name"] = "Müller GmbH"
    enriched = merge_results(df, [_result("r0")])
    exported = export_enriched_csv(enriched, cfg)
    content = exported["combined"].read_text(encoding="utf-8")
    assert "Müller" in content
