"""Tests for analysis/utils.py."""

import json

import pandas as pd
import pytest

from analysis.utils import load_enriched, parse_flags, parse_json_col

# ---------------------------------------------------------------------------
# parse_flags
# ---------------------------------------------------------------------------

def _make_flag(rule: str = "missing_field", severity: str = "warning",
               field: str = "salary", message: str = "salary missing") -> dict:
    return {"rule": rule, "severity": severity, "field": field, "message": message}


def test_parse_flags_returns_one_row_per_flag():
    flags = [_make_flag("r1"), _make_flag("r2")]
    df = pd.DataFrame([{"row_id": "abc", "validation_flags": json.dumps(flags)}])
    result = parse_flags(df)
    assert len(result) == 2
    assert set(result["rule"].tolist()) == {"r1", "r2"}


def test_parse_flags_includes_row_id():
    flags = [_make_flag()]
    df = pd.DataFrame([{"row_id": "xyz123", "validation_flags": json.dumps(flags)}])
    result = parse_flags(df)
    assert result.iloc[0]["row_id"] == "xyz123"


def test_parse_flags_empty_flags_produces_empty_dataframe():
    df = pd.DataFrame([{"row_id": "abc", "validation_flags": "[]"}])
    result = parse_flags(df)
    assert len(result) == 0
    # Schema columns still present
    for col in ("row_id", "rule", "severity", "field", "message"):
        assert col in result.columns


def test_parse_flags_malformed_json_is_skipped():
    """Rows with unparseable validation_flags contribute no flag rows."""
    df = pd.DataFrame([
        {"row_id": "a", "validation_flags": "not_json"},
        {"row_id": "b", "validation_flags": json.dumps([_make_flag()])},
    ])
    result = parse_flags(df)
    assert len(result) == 1
    assert result.iloc[0]["row_id"] == "b"


def test_parse_flags_multiple_rows():
    df = pd.DataFrame([
        {"row_id": "r1", "validation_flags": json.dumps([_make_flag("rule_a")])},
        {"row_id": "r2", "validation_flags": "[]"},
        {"row_id": "r3", "validation_flags": json.dumps(
            [_make_flag("rule_b"), _make_flag("rule_c")]
        )},
    ])
    result = parse_flags(df)
    assert len(result) == 3
    assert set(result["row_id"].tolist()) == {"r1", "r3"}


# ---------------------------------------------------------------------------
# load_enriched
# ---------------------------------------------------------------------------

def test_load_enriched_with_explicit_path(tmp_path):
    """load_enriched reads from a path provided as argument."""
    csv_path = tmp_path / "cleaned_jobs.csv"
    df = pd.DataFrame({
        "row_id": ["abc123"],
        "title": ["Software Engineer"],
        "salary_min": ["70000"],
        "salary_max": ["90000"],
    })
    df.to_csv(csv_path, index=False, encoding="utf-8")

    result = load_enriched(str(csv_path))
    assert len(result) == 1
    assert result.iloc[0]["title"] == "Software Engineer"


def test_load_enriched_converts_na_string_to_none(tmp_path):
    """load_enriched uses read_csv_safe, so 'NA' strings become NaN/missing."""
    csv_path = tmp_path / "cleaned_jobs.csv"
    df = pd.DataFrame({"row_id": ["r1"], "salary_min": ["NA"]})
    df.to_csv(csv_path, index=False, encoding="utf-8")

    result = load_enriched(str(csv_path))
    assert pd.isna(result.iloc[0]["salary_min"])


def test_load_enriched_missing_file_raises(tmp_path):
    """load_enriched raises an error if the file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        load_enriched(str(tmp_path / "nonexistent.csv"))


# ---------------------------------------------------------------------------
# parse_json_col
# ---------------------------------------------------------------------------

def test_parse_json_col_valid_json():
    df = pd.DataFrame({"skills": ['["Python", "SQL"]', '["Java"]']})
    result = parse_json_col(df, "skills")
    assert result.iloc[0] == ["Python", "SQL"]
    assert result.iloc[1] == ["Java"]


def test_parse_json_col_python_repr_fallback():
    """Python-repr list strings are parsed via ast.literal_eval fallback."""
    df = pd.DataFrame({"skills": ["['Python', 'Docker']"]})
    result = parse_json_col(df, "skills")
    assert result.iloc[0] == ["Python", "Docker"]


def test_parse_json_col_malformed_returns_empty_list():
    df = pd.DataFrame({"skills": ["not_valid", "{}"]})
    result = parse_json_col(df, "skills")
    assert result.iloc[0] == []
    assert result.iloc[1] == []  # dict, not list → empty list


def test_parse_json_col_none_returns_empty_list():
    df = pd.DataFrame({"skills": [None, "[]"]})
    result = parse_json_col(df, "skills")
    assert result.iloc[0] == []
    assert result.iloc[1] == []


def test_parse_json_col_already_list_passthrough():
    """If column already contains Python lists, they pass through unchanged."""
    df = pd.DataFrame({"skills": [["Python", "SQL"]]})
    result = parse_json_col(df, "skills")
    assert result.iloc[0] == ["Python", "SQL"]
