"""Tests for extraction/reporting/evaluation.py."""

import json

import pandas as pd
import pytest

from extraction.reporting.evaluation import evaluate, save_accuracy_report

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result(row_id: str, **data_overrides) -> dict:
    data: dict = {
        "contract_type": "Full-time",
        "work_modality": "Hybrid",
        "seniority": "Senior",
        "salary_min": 80_000,
        "salary_max": 100_000,
        "experience_years": 3,
        "technical_skills": ["Python", "SQL"],
        "soft_skills": ["Teamwork"],
        "nice_to_have_skills": ["Docker"],
        "benefits": ["Flexible hours"],
        "tasks": ["Code review"],
    }
    data.update(data_overrides)
    return {"row_id": row_id, "data": data, "validation_flags": []}


def _golden(row_id: str, **overrides) -> dict:
    base: dict = {
        "row_id": row_id,
        "contract_type": "Full-time",
        "work_modality": "Hybrid",
        "seniority": "Senior",
        "salary_min": 80_000,
        "salary_max": 100_000,
        "experience_years": 3,
        "technical_skills": ["Python", "SQL"],
        "soft_skills": ["Teamwork"],
        "nice_to_have_skills": ["Docker"],
        "benefits": ["Flexible hours"],
        "tasks": ["Code review"],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Report structure
# ---------------------------------------------------------------------------


def test_evaluate_returns_matched_rows():
    results = [_result("r1"), _result("r2")]
    golden = [_golden("r1"), _golden("r2")]
    report = evaluate(results, golden)
    assert report["matched_rows"] == 2


def test_evaluate_no_common_ids():
    results = [_result("r1")]
    golden = [_golden("r99")]
    report = evaluate(results, golden)
    assert report["matched_rows"] == 0


def test_evaluate_has_all_field_sections():
    results = [_result("r1")]
    golden = [_golden("r1")]
    report = evaluate(results, golden)
    for field in ("contract_type", "salary_min", "technical_skills", "experience_years"):
        assert field in report["fields"], f"Missing field: {field}"


def test_evaluate_partial_id_overlap():
    results = [_result("r1"), _result("r2"), _result("r3")]
    golden = [_golden("r1"), _golden("r2")]
    report = evaluate(results, golden)
    assert report["matched_rows"] == 2
    assert report["total_pipeline_rows"] == 3
    assert report["total_golden_rows"] == 2


def test_evaluate_empty_results():
    report = evaluate([], [])
    assert report["matched_rows"] == 0


# ---------------------------------------------------------------------------
# Categorical accuracy
# ---------------------------------------------------------------------------


def test_categorical_perfect_accuracy():
    results = [_result("r1", contract_type="Full-time")]
    golden = [_golden("r1", contract_type="Full-time")]
    report = evaluate(results, golden)
    assert report["fields"]["contract_type"]["exact_match_pct"] == 100.0


def test_categorical_zero_accuracy():
    results = [_result("r1", contract_type="Part-time")]
    golden = [_golden("r1", contract_type="Full-time")]
    report = evaluate(results, golden)
    assert report["fields"]["contract_type"]["exact_match_pct"] == 0.0


def test_categorical_partial_accuracy():
    results = [
        _result("r1", contract_type="Full-time"),
        _result("r2", contract_type="Part-time"),
    ]
    golden = [
        _golden("r1", contract_type="Full-time"),
        _golden("r2", contract_type="Full-time"),
    ]
    report = evaluate(results, golden)
    assert report["fields"]["contract_type"]["exact_match_pct"] == 50.0


def test_categorical_confusion_matrix():
    results = [_result("r1", contract_type="Part-time")]
    golden = [_golden("r1", contract_type="Full-time")]
    report = evaluate(results, golden)
    cm = report["fields"]["contract_type"]["confusion_matrix"]
    assert cm["Full-time"]["Part-time"] == 1


def test_categorical_top_mismatches():
    results = [_result("r1", seniority="Mid")]
    golden = [_golden("r1", seniority="Senior")]
    report = evaluate(results, golden)
    mismatches = report["fields"]["seniority"]["top_mismatches"]
    assert len(mismatches) >= 1
    assert mismatches[0]["true"] == "Senior"
    assert mismatches[0]["predicted"] == "Mid"


def test_categorical_null_null_counts_as_match():
    results = [_result("r1", seniority=None)]
    golden = [_golden("r1", seniority=None)]
    report = evaluate(results, golden)
    assert report["fields"]["seniority"]["exact_match_pct"] == 100.0


def test_categorical_multiple_rows_confusion():
    results = [
        _result("r1", seniority="Senior"),
        _result("r2", seniority="Mid"),
        _result("r3", seniority="Senior"),
    ]
    golden = [
        _golden("r1", seniority="Senior"),
        _golden("r2", seniority="Senior"),
        _golden("r3", seniority="Mid"),
    ]
    report = evaluate(results, golden)
    # 1/3 correct (only r1 matches)
    assert report["fields"]["seniority"]["exact_match_pct"] == pytest.approx(33.3, abs=0.1)


# ---------------------------------------------------------------------------
# Numeric accuracy
# ---------------------------------------------------------------------------


def test_numeric_exact_match():
    results = [_result("r1", salary_min=80_000)]
    golden = [_golden("r1", salary_min=80_000)]
    report = evaluate(results, golden)
    assert report["fields"]["salary_min"]["exact_match_pct"] == 100.0


def test_numeric_within_10_pct():
    results = [_result("r1", salary_min=80_000)]
    golden = [_golden("r1", salary_min=85_000)]
    report = evaluate(results, golden)
    # 80k vs 85k: diff = 5k, 5k/85k ≈ 5.9% < 10%
    assert report["fields"]["salary_min"]["within_10pct_pct"] == 100.0
    assert report["fields"]["salary_min"]["exact_match_pct"] == 0.0


def test_numeric_outside_10_pct():
    results = [_result("r1", salary_min=60_000)]
    golden = [_golden("r1", salary_min=80_000)]
    report = evaluate(results, golden)
    # 60k vs 80k: diff = 20k, 20k/80k = 25% > 10%
    assert report["fields"]["salary_min"]["within_10pct_pct"] == 0.0


def test_numeric_both_null():
    results = [_result("r1", salary_min=None)]
    golden = [_golden("r1", salary_min=None)]
    report = evaluate(results, golden)
    assert report["fields"]["salary_min"]["both_null"] == 1
    assert report["fields"]["salary_min"]["exact_match_pct"] == 100.0


def test_numeric_one_null_one_value():
    results = [_result("r1", salary_min=None)]
    golden = [_golden("r1", salary_min=80_000)]
    report = evaluate(results, golden)
    # One is null, one has value — cannot compare, counts as mismatch
    assert report["fields"]["salary_min"]["exact_match_pct"] == 0.0


def test_numeric_experience_field():
    results = [_result("r1", experience_years=3)]
    golden = [_golden("r1", experience_years=3)]
    report = evaluate(results, golden)
    assert report["fields"]["experience_years"]["exact_match_pct"] == 100.0


# ---------------------------------------------------------------------------
# List accuracy
# ---------------------------------------------------------------------------


def test_list_perfect_match():
    results = [_result("r1", technical_skills=["Python", "SQL"])]
    golden = [_golden("r1", technical_skills=["Python", "SQL"])]
    report = evaluate(results, golden)
    assert report["fields"]["technical_skills"]["avg_jaccard_pct"] == 100.0
    assert report["fields"]["technical_skills"]["avg_precision_pct"] == 100.0
    assert report["fields"]["technical_skills"]["avg_recall_pct"] == 100.0


def test_list_partial_overlap():
    results = [_result("r1", technical_skills=["Python", "Docker"])]
    golden = [_golden("r1", technical_skills=["Python", "SQL"])]
    report = evaluate(results, golden)
    # Intersection: {python}, Union: {python, docker, sql} → Jaccard = 1/3
    assert report["fields"]["technical_skills"]["avg_jaccard_pct"] == pytest.approx(33.3, abs=0.2)


def test_list_empty_both():
    results = [_result("r1", technical_skills=[])]
    golden = [_golden("r1", technical_skills=[])]
    report = evaluate(results, golden)
    assert report["fields"]["technical_skills"]["avg_jaccard_pct"] == 100.0


def test_list_case_insensitive():
    results = [_result("r1", technical_skills=["python", "SQL"])]
    golden = [_golden("r1", technical_skills=["Python", "sql"])]
    report = evaluate(results, golden)
    assert report["fields"]["technical_skills"]["avg_jaccard_pct"] == 100.0


def test_list_precision_recall():
    # Pipeline predicts 3 skills, golden has 2 — one overlap
    results = [_result("r1", technical_skills=["Python", "Docker", "Java"])]
    golden = [_golden("r1", technical_skills=["Python", "SQL"])]
    report = evaluate(results, golden)
    f = report["fields"]["technical_skills"]
    # Precision: 1 correct / 3 predicted = 33.3%
    assert f["avg_precision_pct"] == pytest.approx(33.3, abs=0.2)
    # Recall: 1 correct / 2 golden = 50%
    assert f["avg_recall_pct"] == pytest.approx(50.0, abs=0.2)


def test_list_pipeline_empty_golden_not():
    results = [_result("r1", technical_skills=[])]
    golden = [_golden("r1", technical_skills=["Python", "SQL"])]
    report = evaluate(results, golden)
    assert report["fields"]["technical_skills"]["avg_jaccard_pct"] == 0.0
    assert report["fields"]["technical_skills"]["avg_recall_pct"] == 0.0


# ---------------------------------------------------------------------------
# Save report
# ---------------------------------------------------------------------------


def test_save_accuracy_report(tmp_path):
    cfg = {"paths": {"reports_dir": tmp_path / "reports"}}
    results = [_result("r1")]
    golden = [_golden("r1")]
    report = evaluate(results, golden)
    path = save_accuracy_report(report, cfg)
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["matched_rows"] == 1
    assert "fields" in data


def test_save_accuracy_report_creates_dir(tmp_path):
    cfg = {"paths": {"reports_dir": tmp_path / "nested" / "reports"}}
    report = evaluate([_result("r1")], [_golden("r1")])
    path = save_accuracy_report(report, cfg)
    assert path.exists()


# ---------------------------------------------------------------------------
# Fix: List fields as JSON strings (CSV-loaded golden set)
# ---------------------------------------------------------------------------


def test_list_golden_json_string():
    """Golden set loaded from CSV has list fields as JSON strings."""
    results = [_result("r1", technical_skills=["Python", "SQL"])]
    golden = [_golden("r1", technical_skills='["Python", "SQL"]')]
    report = evaluate(results, golden)
    assert report["fields"]["technical_skills"]["avg_jaccard_pct"] == 100.0


def test_list_both_json_strings():
    """Both pipeline and golden as JSON strings still compare correctly."""
    results = [{"row_id": "r1", "data": {"technical_skills": '["Python", "Docker"]'}}]
    golden = [_golden("r1", technical_skills='["Python", "SQL"]')]
    report = evaluate(results, golden)
    # Intersection: {python}, Union: {python, docker, sql}
    assert report["fields"]["technical_skills"]["avg_jaccard_pct"] == pytest.approx(33.3, abs=0.2)


def test_list_golden_empty_string():
    """Golden set empty list as string '[]' or 'NA' handled correctly."""
    results = [_result("r1", benefits=[])]
    golden = [_golden("r1", benefits="[]")]
    report = evaluate(results, golden)
    assert report["fields"]["benefits"]["avg_jaccard_pct"] == 100.0


# ---------------------------------------------------------------------------
# Fix: Tier-1 fields sourced from DataFrame
# ---------------------------------------------------------------------------


def test_tier1_from_dataframe():
    """Tier-1 fields sourced from DataFrame when data dict lacks them.

    At step 6 (validate), Tier-1 columns still have the regex_ prefix.
    _TIER1_DF_CANDIDATES maps evaluation field names to regex_* DataFrame columns.
    """
    # Pipeline result with NO Tier-1 fields in data (realistic LLM-only output)
    results = [{"row_id": "r1", "data": {
        "technical_skills": ["Python"],
        "soft_skills": [],
        "nice_to_have_skills": [],
        "benefits": [],
        "tasks": [],
    }}]
    golden = [_golden("r1", contract_type="Full-time", seniority="Senior",
                       salary_min=80000, experience_years=3)]
    # DataFrame with Tier-1 fields using regex_ prefix (as they appear at step 6)
    df = pd.DataFrame([{
        "row_id": "r1",
        "regex_contract_type": "Full-time",
        "regex_work_modality": "Hybrid",
        "regex_seniority_from_title": "Senior",
        "regex_salary_min": 80000,
        "regex_salary_max": 100000,
        "regex_experience_years": 3,
    }])
    report = evaluate(results, golden, df=df)
    assert report["fields"]["contract_type"]["exact_match_pct"] == 100.0
    assert report["fields"]["seniority"]["exact_match_pct"] == 100.0
    assert report["fields"]["salary_min"]["exact_match_pct"] == 100.0


def test_tier1_df_seniority_mapping():
    """regex_seniority_from_title in DataFrame maps to seniority in evaluation."""
    results = [{"row_id": "r1", "data": {}}]
    golden = [_golden("r1", seniority="Junior")]
    df = pd.DataFrame([{"row_id": "r1", "regex_seniority_from_title": "Junior"}])
    report = evaluate(results, golden, df=df)
    assert report["fields"]["seniority"]["exact_match_pct"] == 100.0


def test_evaluate_without_df_backwards_compatible():
    """Calling evaluate() without df still works (backwards compatible)."""
    results = [_result("r1")]
    golden = [_golden("r1")]
    report = evaluate(results, golden)
    assert report["matched_rows"] == 1
    assert report["fields"]["contract_type"]["exact_match_pct"] == 100.0


def test_tier1_from_plain_columns():
    """Tier-1 fields found via unprefixed column names (post-merge DataFrame)."""
    results = [{"row_id": "r1", "data": {}}]
    golden = [_golden("r1", contract_type="Full-time", seniority="Senior")]
    df = pd.DataFrame([{
        "row_id": "r1",
        "contract_type": "Full-time",
        "seniority_from_title": "Senior",
        "salary_min": 80000,
    }])
    report = evaluate(results, golden, df=df)
    assert report["fields"]["contract_type"]["exact_match_pct"] == 100.0
    assert report["fields"]["seniority"]["exact_match_pct"] == 100.0
    assert report["fields"]["salary_min"]["exact_match_pct"] == 100.0


# ---------------------------------------------------------------------------
# Fix: Golden set empty/NA normalisation
# ---------------------------------------------------------------------------


def test_categorical_golden_empty_string_matches_pipeline_none():
    """Golden set empty string should match pipeline None (both = not specified)."""
    results = [_result("r1", seniority=None)]
    golden = [_golden("r1", seniority="")]
    report = evaluate(results, golden)
    assert report["fields"]["seniority"]["exact_match_pct"] == 100.0


def test_categorical_golden_na_matches_pipeline_none():
    """Golden set 'NA' string should match pipeline None."""
    results = [_result("r1", contract_type=None)]
    golden = [_golden("r1", contract_type="NA")]
    report = evaluate(results, golden)
    assert report["fields"]["contract_type"]["exact_match_pct"] == 100.0


def test_numeric_golden_empty_string_matches_pipeline_none():
    """Golden set empty string for salary should count as both_null with pipeline None."""
    results = [_result("r1", salary_min=None)]
    golden = [_golden("r1", salary_min="")]
    report = evaluate(results, golden)
    assert report["fields"]["salary_min"]["both_null"] == 1
    assert report["fields"]["salary_min"]["exact_match_pct"] == 100.0


def test_numeric_golden_na_matches_pipeline_none():
    """Golden set 'NA' for experience should count as both_null."""
    results = [_result("r1", experience_years=None)]
    golden = [_golden("r1", experience_years="NA")]
    report = evaluate(results, golden)
    assert report["fields"]["experience_years"]["both_null"] == 1
    assert report["fields"]["experience_years"]["exact_match_pct"] == 100.0


def test_golden_normalisation_does_not_affect_real_values():
    """Real golden values should not be normalised away."""
    results = [_result("r1", seniority="Senior", salary_min=80000)]
    golden = [_golden("r1", seniority="Senior", salary_min=80000)]
    report = evaluate(results, golden)
    assert report["fields"]["seniority"]["exact_match_pct"] == 100.0
    assert report["fields"]["salary_min"]["exact_match_pct"] == 100.0
