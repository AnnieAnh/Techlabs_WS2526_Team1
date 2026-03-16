"""Tests for the cleaning pipeline."""

import json
from pathlib import Path

import pandas as pd
import pytest

from cleaning.benefit_categorizer import benefit_category_set
from cleaning.categorical_remapper import (
    fix_company_name_na,
    fix_residual_gender_markers,
    normalize_company_casing,
    remap_categoricals,
)
from cleaning.constants import COLUMN_ORDER
from cleaning.location_cleaner import normalize_city_names
from cleaning.missing_values import fix_numeric_columns, standardize_missing_values
from cleaning.output_formatter import assert_invariants, drop_columns_and_rows, reorder_columns
from cleaning.pipeline import clean_enriched
from cleaning.quality_flagger import flag_description_quality
from cleaning.skill_normalizer import (
    fix_cpp_inference,
    normalize_skill_casing,
    re_verify_skills_post_clean,
)
from cleaning.validation_fixer import fix_validation_flags

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_FAMILIES = {
    "Software Developer", "Frontend Developer", "Backend Developer",
    "Fullstack Developer", "Data Engineer", "Data Scientist", "ML Engineer",
    "DevOps Engineer", "Cloud Engineer", "IT Consultant", "Other",
    "QA Engineer", "Solution Architect", "Solutions Architect",
}


def _df(**cols) -> pd.DataFrame:
    """Build a one-row DataFrame with the given column values."""
    data = {k: [v] for k, v in cols.items()}
    return pd.DataFrame(data)


def _enriched_row(**overrides) -> pd.DataFrame:
    """Return a minimal valid enriched row with sane defaults."""
    base = {
        "row_id": "abc123def456",
        "job_url": "https://example.com/job/1",
        "date_posted": "2024-01-15",
        "company_name": "Acme GmbH",
        "city": "Berlin",
        "state": "Berlin",
        "country": "Germany",
        "title": "Senior Developer (m/w/d)",
        "title_cleaned": "Senior Developer",
        "job_family": "Software Developer",
        "seniority_from_title": "Senior",
        "contract_type": "Full-time",
        "work_modality": "Hybrid",
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
        "source_file": "combined_jobs_1.csv",
        "site": "linkedin",
        "validation_flags": "[]",
        "description": "We are looking for a Senior Developer...",
    }
    base.update(overrides)
    return pd.DataFrame({k: [v] for k, v in base.items()})


# ---------------------------------------------------------------------------
# Missing value standardization
# ---------------------------------------------------------------------------

def test_standardize_missing_values_list_cols():
    """Empty list columns become '[]'."""
    df = _df(technical_skills=None, soft_skills="", nice_to_have_skills="nan")
    result = standardize_missing_values(df)
    assert result.loc[0, "technical_skills"] == "[]"
    assert result.loc[0, "soft_skills"] == "[]"
    assert result.loc[0, "nice_to_have_skills"] == "[]"


def test_standardize_missing_values_string_cols():
    """Empty/None/sentinel string columns become None."""
    df = _df(job_family="", seniority_from_title=None, contract_type="NA")
    result = standardize_missing_values(df)
    assert result.loc[0, "job_family"] is None
    assert result.loc[0, "seniority_from_title"] is None
    assert result.loc[0, "contract_type"] is None


def test_standardize_valid_list_json_preserved():
    """Valid JSON list strings are preserved."""
    df = _df(technical_skills='["Python", "SQL"]')
    result = standardize_missing_values(df)
    assert result.loc[0, "technical_skills"] == '["Python", "SQL"]'


def test_standardize_python_repr_converted_to_json():
    """Python repr lists are converted to JSON."""
    df = _df(technical_skills="['Python', 'SQL']")
    result = standardize_missing_values(df)
    parsed = json.loads(result.loc[0, "technical_skills"])
    assert parsed == ["Python", "SQL"]


def test_standardize_deduplicates_list_case_insensitive():
    """Case-insensitive duplicates in list columns are removed (keeps first occurrence)."""
    df = _df(benefits='["Flexibles Arbeitszeitmodell", "Remote", "flexibles arbeitszeitmodell"]')
    result = standardize_missing_values(df)
    parsed = json.loads(result.loc[0, "benefits"])
    assert parsed == ["Flexibles Arbeitszeitmodell", "Remote"]


# ---------------------------------------------------------------------------
# Numeric column normalization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("60000", 60000),
    ("60000.0", 60000),
    ("50.000", 50000),     # German number format
    ("", None),
    ("NA", None),
    (None, None),
])
def test_fix_numeric_salary(raw, expected):
    """Salary strings are converted to int or None."""
    df = _df(salary_min=raw)
    result = fix_numeric_columns(df)
    val = result.loc[0, "salary_min"]
    if expected is None:
        assert val is None or (isinstance(val, float) and pd.isna(val))
    else:
        assert val == expected


@pytest.mark.parametrize("raw, expected", [
    ("500000", None),    # above ceiling
    ("5000", None),      # below floor
    ("100000", 100000),
])
def test_fix_numeric_salary_outliers(raw, expected):
    """Salary outliers outside [10k, 300k] are nulled to None."""
    df = _df(salary_max=raw)
    result = fix_numeric_columns(df)
    val = result.loc[0, "salary_max"]
    if expected is None:
        assert val is None or (isinstance(val, float) and pd.isna(val))
    else:
        assert val == expected


def test_fix_numeric_experience():
    """experience_years is converted to int."""
    df = _df(experience_years="5.0")
    result = fix_numeric_columns(df)
    assert result.loc[0, "experience_years"] == 5


# ---------------------------------------------------------------------------
# City name normalization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw_city, state, expected_city", [
    ("Munich", "Bavaria", "München"),
    ("Cologne", "North Rhine-Westphalia", "Köln"),
    ("Nuremberg", "Bavaria", "Nürnberg"),
    ("Frankfurt", "Hesse", "Frankfurt am Main"),
    ("Frankfurt", "Brandenburg", "Frankfurt (Oder)"),
    ("Frankfurt", "Bavaria", "Frankfurt am Main"),   # unknown state → am Main
    ("Berlin", "Berlin", "Berlin"),                  # unchanged
])
def test_normalize_city_names(raw_city, state, expected_city):
    """City aliases and Frankfurt state-aware normalization."""
    df = _df(city=raw_city, state=state)
    result = normalize_city_names(df)
    assert result.loc[0, "city"] == expected_city


# ---------------------------------------------------------------------------
# Categorical remapping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("Web Developer", "Fullstack Developer"),
    ("Machine Learning Engineer", "ML Engineer"),
    ("DevSecOps Engineer", "DevOps Engineer"),
    ("Software Developer", "Software Developer"),  # already canonical
    (None, None),
])
def test_remap_job_family(raw, expected):
    """job_family values are remapped to canonical forms."""
    df = _df(job_family=raw)
    result = remap_categoricals(df)
    val = result.loc[0, "job_family"]
    if expected is None:
        assert val is None or (isinstance(val, float) and pd.isna(val))
    else:
        assert val == expected


def test_remap_contract_type():
    """'Permanent' → 'Full-time'."""
    df = _df(contract_type="Permanent")
    result = remap_categoricals(df)
    assert result.loc[0, "contract_type"] == "Full-time"


def test_remap_seniority_from_title():
    """'Entry Level' → 'Junior'."""
    df = _df(seniority_from_title="Entry Level")
    result = remap_categoricals(df)
    assert result.loc[0, "seniority_from_title"] == "Junior"


# ---------------------------------------------------------------------------
# Validation flag format conversion
# ---------------------------------------------------------------------------

def test_fix_validation_flags_empty():
    """Empty/None/NaN → '[]'."""
    df = _df(validation_flags="")
    result = fix_validation_flags(df)
    assert result.loc[0, "validation_flags"] == "[]"


def test_fix_validation_flags_python_repr():
    """Python repr flags are converted to JSON."""
    py_repr = (
        "[{'rule': 'salary_too_high', 'severity': 'error',"
        " 'field': 'salary_max', 'message': 'too high', 'row_id': 'abc'}]"
    )
    df = _df(validation_flags=py_repr)
    result = fix_validation_flags(df)
    parsed = json.loads(result.loc[0, "validation_flags"])
    assert isinstance(parsed, list)
    assert parsed[0]["rule"] == "salary_too_high"


def test_fix_validation_flags_already_json():
    """Already-JSON flags pass through unchanged."""
    flags_json = json.dumps([{"rule": "skill_not_in_description", "severity": "warning"}])
    df = _df(validation_flags=flags_json)
    result = fix_validation_flags(df)
    parsed = json.loads(result.loc[0, "validation_flags"])
    assert parsed[0]["rule"] == "skill_not_in_description"


# ---------------------------------------------------------------------------
# Column and row dropping
# ---------------------------------------------------------------------------

def test_drop_columns_removes_unused():
    """country, location, source_file are dropped."""
    df = _enriched_row()
    result = drop_columns_and_rows(df, _VALID_FAMILIES)
    assert "country" not in result.columns
    assert "location" not in result.columns
    assert "source_file" not in result.columns


def test_drop_rows_empty_job_family():
    """Rows with empty-string job_family are dropped."""
    df = _enriched_row(job_family="")
    result = drop_columns_and_rows(df, _VALID_FAMILIES)
    assert len(result) == 0


def test_drop_rows_none_job_family_kept():
    """Rows with job_family=None are NOT dropped (None is valid unknown placeholder)."""
    df = _enriched_row(job_family=None)
    result = drop_columns_and_rows(df, _VALID_FAMILIES)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Column reordering
# ---------------------------------------------------------------------------

def test_reorder_columns_order():
    """Columns in output match COLUMN_ORDER exactly."""
    df = _enriched_row()
    df = drop_columns_and_rows(df, _VALID_FAMILIES)
    result = reorder_columns(df)
    expected_cols = [c for c in COLUMN_ORDER if c in result.columns]
    assert list(result.columns) == expected_cols


def test_reorder_columns_drops_extras():
    """Extra columns not in COLUMN_ORDER are dropped."""
    df = _enriched_row()
    df["extra_column"] = "should be dropped"
    result = reorder_columns(df)
    assert "extra_column" not in result.columns


# ---------------------------------------------------------------------------
# Invariant assertions
# ---------------------------------------------------------------------------

def test_assert_invariants_passes_clean_data():
    """Clean data passes all assertions."""
    df = _enriched_row()
    df = standardize_missing_values(df)
    df = fix_numeric_columns(df)
    df = fix_validation_flags(df)
    df = drop_columns_and_rows(df, _VALID_FAMILIES)
    df = reorder_columns(df)
    # Should not raise
    assert_invariants(df, _VALID_FAMILIES)


def test_assert_invariants_fails_empty_categorical():
    """Empty categorical raises AssertionError."""
    df = _enriched_row(job_family="")
    # job_family is empty string — should fail
    with pytest.raises(AssertionError, match="empty strings in 'job_family'"):
        assert_invariants(df, _VALID_FAMILIES)


def test_assert_invariants_fails_invalid_job_family():
    """Job family not in valid set raises AssertionError."""
    df = _enriched_row(job_family="Wizard Developer")
    with pytest.raises(AssertionError, match="unmapped job_family"):
        assert_invariants(df, _VALID_FAMILIES)


def test_assert_invariants_fails_bad_json_list():
    """Non-JSON list column raises AssertionError."""
    df = _enriched_row(technical_skills="Python, SQL")  # not JSON
    with pytest.raises(AssertionError, match="non-JSON-array values"):
        assert_invariants(df, _VALID_FAMILIES)


def test_assert_invariants_fails_salary_inversion():
    """salary_min > salary_max raises AssertionError."""
    df = _enriched_row(salary_min="90000", salary_max="60000")
    with pytest.raises(AssertionError, match="salary_min > salary_max"):
        assert_invariants(df, _VALID_FAMILIES)


def test_assert_invariants_fails_duplicate_row_ids():
    """Duplicate row_ids raise AssertionError."""
    row = _enriched_row()
    df = pd.concat([row, row], ignore_index=True)
    with pytest.raises(AssertionError, match="duplicate row_ids"):
        assert_invariants(df, _VALID_FAMILIES)


# ---------------------------------------------------------------------------
# Round-trip: full clean_enriched pipeline
# ---------------------------------------------------------------------------

def test_clean_enriched_round_trip(tmp_path: Path):
    """Round-trip: writing a CSV and running clean_enriched produces valid output."""
    # Build a messy but parseable enriched CSV
    rows = []
    for i in range(5):
        rows.append({
            "row_id": f"row{i:012d}",
            "job_url": f"https://example.com/{i}",
            "date_posted": "2024-01-01",
            "company_name": "Test GmbH",
            "city": "Munich",
            "state": "Bavaria",
            "country": "Germany",
            "title": "Developer (m/w/d)",
            "title_cleaned": "Developer",
            "job_family": "Software Developer",
            "seniority_from_title": "Senior",
            "contract_type": "Full-time",
            "work_modality": "Remote",
            "seniority": "Senior",
            "salary_min": "60000",
            "salary_max": "80000",
            "experience_years": "3",
            "technical_skills": '["Python"]',
            "soft_skills": "[]",
            "nice_to_have_skills": "[]",
            "benefits": "[]",
            "tasks": "[]",
            "location": "Munich, Bavaria, Germany",
            "source_file": "test.csv",
            "site": "linkedin",
            "validation_flags": "[]",
            "description": "Python developer role in Munich.",
        })

    input_csv = tmp_path / "enriched_test.csv"
    output_csv = tmp_path / "cleaned_test.csv"
    pd.DataFrame(rows).to_csv(input_csv, index=False)

    result = clean_enriched(input_csv, output_csv)

    # Output file exists
    assert output_csv.exists()

    # city was normalised
    assert (result["city"] == "München").all()

    # country and location columns dropped
    assert "country" not in result.columns
    assert "location" not in result.columns

    # Columns in correct order
    expected = [c for c in COLUMN_ORDER if c in result.columns]
    assert list(result.columns) == expected

    # No invariant failures (use test-local valid families set)
    assert_invariants(result, _VALID_FAMILIES)


def test_clean_enriched_idempotent(tmp_path: Path):
    """Running clean_enriched twice on already-clean data produces identical output."""
    row = {
        "row_id": "aabbccddeeff",
        "job_url": "https://example.com/1",
        "date_posted": "2024-01-01",
        "company_name": "Test GmbH",
        "city": "München",
        "state": "Bavaria",
        "country": "Germany",
        "title": "Developer",
        "title_cleaned": "Developer",
        "job_family": "Software Developer",
        "seniority_from_title": "Senior",
        "contract_type": "Full-time",
        "work_modality": "Remote",
        "seniority": "Senior",
        "salary_min": "60000",
        "salary_max": "80000",
        "experience_years": "3",
        "technical_skills": '["Python"]',
        "soft_skills": "[]",
        "nice_to_have_skills": "[]",
        "benefits": "[]",
        "tasks": "[]",
        "location": "München, Bavaria, Germany",
        "source_file": "test.csv",
        "site": "linkedin",
        "validation_flags": "[]",
        "description": "A developer role.",
    }
    input1 = tmp_path / "enriched1.csv"
    output1 = tmp_path / "cleaned1.csv"
    output2 = tmp_path / "cleaned2.csv"

    pd.DataFrame([row]).to_csv(input1, index=False)
    df1 = clean_enriched(input1, output1)
    df2 = clean_enriched(output1, output2)

    # Same shape and values
    assert list(df1.columns) == list(df2.columns)
    assert len(df1) == len(df2)
    for col in df1.columns:
        assert df1[col].iloc[0] == df2[col].iloc[0], f"Column {col!r} differs after second run"


# ---------------------------------------------------------------------------
# C++ inference correction
# ---------------------------------------------------------------------------


def test_fix_cpp_inference_genuine_cpp():
    """Description contains 'c++' → row unchanged."""
    row = pd.Series({
        "description": "We use C++ for low-level development.",
        "technical_skills": '["C++", "Python"]',
        "nice_to_have_skills": '[]',
    })
    result = fix_cpp_inference(row)
    assert json.loads(result["technical_skills"]) == ["C++", "Python"]


def test_fix_cpp_inference_bare_c_only():
    """No C++ but bare C in description → C++ replaced by C in skill lists."""
    row = pd.Series({
        "description": "Experience with the C programming language is required.",
        "technical_skills": '["C++", "Python"]',
        "nice_to_have_skills": '["C++"]',
    })
    result = fix_cpp_inference(row)
    tech = json.loads(result["technical_skills"])
    nice = json.loads(result["nice_to_have_skills"])
    assert "C" in tech
    assert "C++" not in tech
    assert "C" in nice
    assert "C++" not in nice


def test_fix_cpp_inference_neither():
    """No C++ and no bare C → C++ removed from skill lists entirely."""
    row = pd.Series({
        "description": "Experience with Python and Java required.",
        "technical_skills": '["C++", "Python"]',
        "nice_to_have_skills": '[]',
    })
    result = fix_cpp_inference(row)
    tech = json.loads(result["technical_skills"])
    assert "C++" not in tech
    assert "Python" in tech


def test_fix_cpp_inference_lowercase_c_no_false_positive():
    """Lowercase 'c' in words like 'scrum' must NOT trigger bare-C detection."""
    row = pd.Series({
        "description": "Scrum ceremonies and agile practices required.",
        "technical_skills": '["C++", "Python"]',
        "nice_to_have_skills": '[]',
    })
    result = fix_cpp_inference(row)
    tech = json.loads(result["technical_skills"])
    # No bare uppercase C in description → C++ should be removed entirely
    assert "C++" not in tech
    assert "C" not in tech
    assert "Python" in tech


def test_fix_cpp_inference_uppercase_c_detected():
    """Uppercase 'C' at word boundary correctly detects the C language."""
    row = pd.Series({
        "description": "Programming in C and embedded systems.",
        "technical_skills": '["C++", "Python"]',
        "nice_to_have_skills": '[]',
    })
    result = fix_cpp_inference(row)
    tech = json.loads(result["technical_skills"])
    assert "C" in tech
    assert "C++" not in tech


# ---------------------------------------------------------------------------
# Company name NA normalization
# ---------------------------------------------------------------------------


def test_fix_company_name_na_lowercase():
    """'na' (lowercase) → None."""
    df = _df(company_name="na")
    result = fix_company_name_na(df)
    val = result.loc[0, "company_name"]
    assert val is None or (isinstance(val, float) and pd.isna(val))


def test_fix_company_name_na_uppercase_to_none():
    """'NA' (uppercase) is also converted to None."""
    df = _df(company_name="NA")
    result = fix_company_name_na(df)
    val = result.loc[0, "company_name"]
    assert val is None or (isinstance(val, float) and pd.isna(val))


def test_fix_company_name_na_real_company_unchanged():
    """Real company names are not modified."""
    df = _df(company_name="Acme GmbH")
    result = fix_company_name_na(df)
    assert result.loc[0, "company_name"] == "Acme GmbH"


# ---------------------------------------------------------------------------
# Residual gender marker stripping
# ---------------------------------------------------------------------------


def test_fix_residual_gender_markers_gn():
    """'(gn)' residual stripped from title_cleaned."""
    df = _df(title_cleaned="Software Engineer (gn)")
    result = fix_residual_gender_markers(df)
    assert result.loc[0, "title_cleaned"] == "Software Engineer"


def test_fix_residual_gender_markers_comma():
    """'(m,w,d)' residual stripped from title_cleaned."""
    df = _df(title_cleaned="Backend Developer (m,w,d)")
    result = fix_residual_gender_markers(df)
    assert result.loc[0, "title_cleaned"] == "Backend Developer"


def test_fix_residual_gender_markers_all_genders():
    """'(all genders)' residual stripped from title_cleaned."""
    df = _df(title_cleaned="Data Engineer (all genders)")
    result = fix_residual_gender_markers(df)
    assert result.loc[0, "title_cleaned"] == "Data Engineer"


# ---------------------------------------------------------------------------
# Skill casing normalization
# ---------------------------------------------------------------------------


def test_normalize_skill_casing_most_frequent_wins():
    """Most-frequent capitalization variant wins per lowercase group."""
    # "GIT" appears twice, "git" once → canonical should be "GIT"
    df = pd.DataFrame({
        "technical_skills": ['["GIT", "Python"]', '["GIT"]', '["git"]'],
    })
    result = normalize_skill_casing(df)
    for row_val in result["technical_skills"]:
        skills = json.loads(row_val)
        git_variants = [s for s in skills if s.lower() == "git"]
        assert all(s == "GIT" for s in git_variants), f"Expected 'GIT', got {git_variants}"


def test_normalize_skill_casing_no_change_if_consistent():
    """Consistently cased skills are not changed."""
    df = pd.DataFrame({"technical_skills": ['["Python", "Docker"]', '["Python"]']})
    result = normalize_skill_casing(df)
    for row_val in result["technical_skills"]:
        skills = json.loads(row_val)
        assert all(s in ("Python", "Docker") for s in skills)


# ---------------------------------------------------------------------------
# Company name casing normalization
# ---------------------------------------------------------------------------


def test_normalize_company_casing_most_frequent_wins():
    """Most-frequent capitalization variant wins."""
    df = pd.DataFrame({
        # "Acme GmbH" appears 3 times, "acme gmbh" appears 1 time → canonical = "Acme GmbH"
        "company_name": ["acme gmbh", "Acme GmbH", "Acme GmbH", "Acme GmbH"],
    })
    result = normalize_company_casing(df)
    assert (result["company_name"] == "Acme GmbH").all()


# ---------------------------------------------------------------------------
# Extended invariant assertions
# ---------------------------------------------------------------------------


def test_assert_invariants_fails_lowercase_na_company():
    """lowercase 'na' in company_name raises AssertionError."""
    df = _enriched_row(company_name="na")
    with pytest.raises(AssertionError, match="sentinel string values in company_name"):
        assert_invariants(df, _VALID_FAMILIES)


def test_assert_invariants_fails_gender_residual_in_title():
    """Residual gender marker in title_cleaned raises AssertionError."""
    df = _enriched_row(title_cleaned="Software Engineer (gn)")
    with pytest.raises(AssertionError, match="residual gender markers in title_cleaned"):
        assert_invariants(df, _VALID_FAMILIES)


# ---------------------------------------------------------------------------
# Benefit categorization
# ---------------------------------------------------------------------------


def test_benefit_category_set_remote():
    """'Remote work' benefit → benefit_categories includes 'remote_work'."""
    result = json.loads(benefit_category_set('["Remote work", "Flexible hours"]'))
    assert "remote_work" in result
    assert "flexible_hours" in result


def test_benefit_category_set_empty():
    """Empty benefits list → empty category list."""
    result = json.loads(benefit_category_set("[]"))
    assert result == []


def test_benefit_category_set_unknown():
    """Unknown benefit → 'other' category."""
    result = json.loads(benefit_category_set('["Mystery Perk"]'))
    assert result == ["other"]


@pytest.mark.parametrize("benefit, expected_category", [
    ("Sabbatical option", "time_off"),
    ("Deutschlandticket", "mobility"),
    ("JobRad leasing", "mobility"),
    ("Weihnachtsgeld", "compensation"),
    ("Urlaubsgeld", "time_off"),  # "urlaub" substring matches time_off
    ("13. Gehalt", "compensation"),
    ("Betriebliche Altersvorsorge (bAV)", "retirement"),
    ("Vermögenswirksame Leistungen", "retirement"),
    ("Fortbildungsbudget", "education"),
    ("Mentoring programme", "education"),
    ("Obstkorb und Getränke", "food"),
    ("Sommerfest", "social"),
    ("Weihnachtsfeier", "social"),
    ("Corporate Benefits Rabatte", "perks"),
    ("Kinderbetreuung/Kita-Zuschuss", "perks"),
    ("Dog-friendly office", "perks"),
    ("Work-Life Balance Programm", "flexible_hours"),
    ("4-Tage-Woche", "flexible_hours"),
    ("Hybrid work model", "remote_work"),
    ("Yoga and meditation", "health"),
    ("Ergonomie am Arbeitsplatz", "health"),
])
def test_benefit_category_expanded_keywords(benefit, expected_category):
    """New benefit keywords map to the correct category."""
    result = json.loads(benefit_category_set(json.dumps([benefit])))
    assert expected_category in result


# ---------------------------------------------------------------------------
# Description quality flagging
# ---------------------------------------------------------------------------


def test_flag_quality_concatenated():
    """>10 camelCase transitions → 'concatenated'."""
    # Each pair like "aA" contributes 1 transition; 12 pairs = 12 transitions (>10)
    text = "aAbBcCdDeEfFgGhHiIjJkKlL" * 2
    df = pd.DataFrame({"description": [text]})
    result = flag_description_quality(df)
    assert result.loc[0, "description_quality"] == "concatenated"


def test_flag_quality_clean():
    """Normal plain text → 'clean'."""
    df = pd.DataFrame({"description": ["We are looking for a senior developer."]})
    result = flag_description_quality(df)
    assert result.loc[0, "description_quality"] == "clean"


def test_flag_quality_no_description_col():
    """Missing description column → DataFrame returned unchanged."""
    df = pd.DataFrame({"title": ["Developer"]})
    result = flag_description_quality(df)
    assert "description_quality" not in result.columns


# ---------------------------------------------------------------------------
# Post-clean skill re-verification
# ---------------------------------------------------------------------------


def test_re_verify_skills_post_clean_flags_missing_skill():
    """Skill absent from description gets skill_not_in_description flag."""
    row = _enriched_row(
        technical_skills='["Python", "Kubernetes"]',
        description="Python scripting required.",
        validation_flags="[]",
    ).iloc[0].to_dict()
    df = pd.DataFrame([row])
    result = re_verify_skills_post_clean(df)
    flags = json.loads(result.iloc[0]["validation_flags"])
    rules = [f["rule"] for f in flags]
    assert "skill_not_in_description" in rules


def test_re_verify_skills_post_clean_no_flag_when_present():
    """Skill found in description → no flag added."""
    row = _enriched_row(
        technical_skills='["Python"]',
        nice_to_have_skills="[]",
        description="We need Python experience.",
        validation_flags="[]",
    ).iloc[0].to_dict()
    df = pd.DataFrame([row])
    result = re_verify_skills_post_clean(df)
    flags = json.loads(result.iloc[0]["validation_flags"])
    assert not any(f["rule"] == "skill_not_in_description" for f in flags)


def test_re_verify_skills_post_clean_variant_match():
    """Skill variant (e.g. 'nodejs' for 'Node.js') counts as present."""
    row = _enriched_row(
        technical_skills='["Node.js"]',
        nice_to_have_skills="[]",
        description="Experience with nodejs required.",
        validation_flags="[]",
    ).iloc[0].to_dict()
    df = pd.DataFrame([row])
    result = re_verify_skills_post_clean(df)
    flags = json.loads(result.iloc[0]["validation_flags"])
    assert not any(f["rule"] == "skill_not_in_description" for f in flags)


def test_re_verify_skills_post_clean_replaces_stale_flags():
    """Stale extraction-time skill flags are replaced, not appended."""
    stale_flags = json.dumps([
        {"field": "technical_skills", "rule": "skill_not_in_description",
         "severity": "warning", "message": "Skill 'OldSkill' not found in description"},
        {"field": "salary_min", "rule": "below_floor",
         "severity": "warning", "message": "Salary below floor"},
    ])
    row = _enriched_row(
        technical_skills='["Python"]',
        nice_to_have_skills="[]",
        description="Python scripting required.",
        validation_flags=stale_flags,
    ).iloc[0].to_dict()
    df = pd.DataFrame([row])
    result = re_verify_skills_post_clean(df)
    flags = json.loads(result.iloc[0]["validation_flags"])
    rules = [f["rule"] for f in flags]
    # Stale skill flag for "OldSkill" must be gone
    skill_flags = [f for f in flags if f["rule"] == "skill_not_in_description"]
    assert not any(f.get("message", "").find("OldSkill") >= 0 for f in skill_flags)
    # Non-skill flags (salary) must be preserved
    assert "below_floor" in rules


def test_re_verify_skills_post_clean_preserves_non_skill_flags():
    """Non-skill flags from extraction are always preserved."""
    existing_flags = json.dumps([
        {"field": "salary_min", "rule": "below_floor",
         "severity": "warning", "message": "Salary below floor"},
        {"field": "work_modality", "rule": "remote_but_onsite_text",
         "severity": "warning", "message": "Mismatch"},
    ])
    row = _enriched_row(
        technical_skills="[]",
        nice_to_have_skills="[]",
        description="A developer role.",
        validation_flags=existing_flags,
    ).iloc[0].to_dict()
    df = pd.DataFrame([row])
    result = re_verify_skills_post_clean(df)
    flags = json.loads(result.iloc[0]["validation_flags"])
    rules = [f["rule"] for f in flags]
    assert "below_floor" in rules
    assert "remote_but_onsite_text" in rules


# ---------------------------------------------------------------------------
# validation_flags: Python list input handling
# ---------------------------------------------------------------------------


def test_fix_validation_flags_python_list_input():
    """Python list objects (from merge_results) are serialized correctly."""
    flags_list = [{"rule": "skill_not_in_description", "message": "Skill 'Go' not found"}]
    df = _df(validation_flags=flags_list)
    result = fix_validation_flags(df)
    parsed = json.loads(result.loc[0, "validation_flags"])
    assert isinstance(parsed, list)
    assert parsed[0]["rule"] == "skill_not_in_description"
    assert "'Go'" in parsed[0]["message"]


def test_fix_validation_flags_empty_list_input():
    """Empty Python list → '[]'."""
    df = _df(validation_flags=[])
    result = fix_validation_flags(df)
    assert result.loc[0, "validation_flags"] == "[]"


def test_fix_validation_flags_list_with_embedded_quotes():
    """List with embedded quotes in values is preserved correctly."""
    flags_list = [
        {"rule": "skill_not_in_description", "message": "Skill 'C++' not grounded"},
        {"rule": "high_hallucination_rate", "message": "3 of 5 skills flagged"},
    ]
    df = _df(validation_flags=flags_list)
    result = fix_validation_flags(df)
    parsed = json.loads(result.loc[0, "validation_flags"])
    assert len(parsed) == 2
    assert "'C++'" in parsed[0]["message"]
