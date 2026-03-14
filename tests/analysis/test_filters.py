"""Tests for analysis/filters.py."""

import pandas as pd

from analysis.filters import (
    explode_json_col,
    filter_by_job_family,
    filter_by_seniority,
    filter_remote,
    filter_salary_known,
    salary_df,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df(**cols) -> pd.DataFrame:
    return pd.DataFrame({k: [v] if not isinstance(v, list) else v for k, v in cols.items()})


def _jobs(**overrides) -> pd.DataFrame:
    """Build a small multi-row DataFrame for filter tests."""
    base = {
        "job_family": ["Software Developer", "Data Scientist", "Frontend Developer"],
        "seniority_from_title": ["Senior", "Junior", "Mid"],
        "work_modality": ["Remote", "Hybrid", "On-site"],
        "salary_min": [70000, 50000, None],
        "salary_max": [90000, 65000, None],
    }
    base.update(overrides)
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# filter_by_job_family
# ---------------------------------------------------------------------------

def test_filter_by_job_family_exact_match():
    df = _jobs()
    result = filter_by_job_family(df, "Software Developer")
    assert len(result) == 1
    assert result.iloc[0]["job_family"] == "Software Developer"


def test_filter_by_job_family_case_insensitive():
    df = _jobs()
    result = filter_by_job_family(df, "software developer")
    assert len(result) == 1


def test_filter_by_job_family_no_match_returns_empty():
    df = _jobs()
    result = filter_by_job_family(df, "DevOps Engineer")
    assert len(result) == 0


# ---------------------------------------------------------------------------
# filter_by_seniority
# ---------------------------------------------------------------------------

def test_filter_by_seniority_matches_senior():
    df = _jobs()
    result = filter_by_seniority(df, "Senior")
    assert len(result) == 1
    assert result.iloc[0]["seniority_from_title"] == "Senior"


def test_filter_by_seniority_case_insensitive():
    df = _jobs()
    result = filter_by_seniority(df, "junior")
    assert len(result) == 1


# ---------------------------------------------------------------------------
# filter_remote
# ---------------------------------------------------------------------------

def test_filter_remote_returns_only_remote_rows():
    df = _jobs()
    result = filter_remote(df)
    assert len(result) == 1
    assert result.iloc[0]["work_modality"] == "Remote"


def test_filter_remote_empty_when_no_remote():
    df = _df(work_modality=["Hybrid", "On-site"])
    result = filter_remote(df)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# filter_salary_known
# ---------------------------------------------------------------------------

def test_filter_salary_known_keeps_numeric_rows():
    df = _jobs()
    result = filter_salary_known(df)
    # Only the first two rows have numeric salary strings
    assert len(result) == 2


def test_filter_salary_known_drops_na_values():
    df = _df(salary_min=[None, None], salary_max=[None, None])
    result = filter_salary_known(df)
    assert len(result) == 0


def test_filter_salary_known_drops_none_values():
    df = _df(salary_min=[None, 60000], salary_max=[None, 80000])
    result = filter_salary_known(df)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# salary_df
# ---------------------------------------------------------------------------

def test_salary_df_casts_to_int():
    df = _jobs()
    result = salary_df(df)
    assert result["salary_min"].dtype in ("int64", "int32", int)
    assert result["salary_max"].dtype in ("int64", "int32", int)


def test_salary_df_adds_salary_mid():
    df = _df(salary_min=[70000], salary_max=[90000])
    result = salary_df(df)
    assert "salary_mid" in result.columns
    assert result.iloc[0]["salary_mid"] == 80000


def test_salary_df_drops_na_rows():
    df = _df(salary_min=[None, 60000], salary_max=[None, 80000])
    result = salary_df(df)
    assert len(result) == 1
    assert result.iloc[0]["salary_mid"] == 70000


# ---------------------------------------------------------------------------
# explode_json_col
# ---------------------------------------------------------------------------

def test_explode_json_col_valid_json_list():
    df = _df(row_id=["r1"], skills=['["Python", "SQL"]'])
    result = explode_json_col(df, "skills")
    assert len(result) == 2
    assert set(result["skills"].tolist()) == {"Python", "SQL"}


def test_explode_json_col_python_repr_list():
    """Python-repr list strings (ast.literal_eval fallback) are parsed correctly."""
    df = _df(row_id=["r1"], skills=["['Python', 'Docker']"])
    result = explode_json_col(df, "skills")
    assert len(result) == 2
    assert set(result["skills"].tolist()) == {"Python", "Docker"}


def test_explode_json_col_malformed_drops_row():
    """Malformed JSON produces an empty list — the row is dropped."""
    df = _df(row_id=["r1", "r2"], skills=["not_valid_json", '["Java"]'])
    result = explode_json_col(df, "skills")
    # r1 parses to [] → dropped; r2 → 1 row
    assert len(result) == 1
    assert result.iloc[0]["skills"] == "Java"


def test_explode_json_col_null_salary_filter():
    """Null salary values (NA string or None) are not included after explode."""
    # Exploding salary_min would not include empty-list rows
    result = explode_json_col(
        _df(row_id=["r1", "r2"], skills=['["Python"]', None]), "skills"
    )
    # None parses to [] → dropped
    assert len(result) == 1
    assert result.iloc[0]["skills"] == "Python"


def test_explode_json_col_preserves_other_columns():
    df = _df(row_id=["r1"], company=["Acme"], skills=['["Python", "SQL"]'])
    result = explode_json_col(df, "skills")
    assert "row_id" in result.columns
    assert "company" in result.columns
    # Both exploded rows carry the original metadata
    assert (result["row_id"] == "r1").all()
    assert (result["company"] == "Acme").all()


def test_explode_json_col_empty_list_drops_row():
    df = _df(row_id=["r1"], skills=["[]"])
    result = explode_json_col(df, "skills")
    assert len(result) == 0
