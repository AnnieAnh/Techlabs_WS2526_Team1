"""Final column ordering, row dropping, and invariant assertions for the cleaning pipeline."""

import json
import logging
from pathlib import Path

import pandas as pd
import yaml

from cleaning.constants import _LIST_COLUMNS, ANALYSIS_COLUMN_ORDER, COLUMN_ORDER
from shared.constants import MISSING_SENTINELS

logger = logging.getLogger("pipeline.output_formatter")


def _load_valid_job_families(config_path: Path | None = None) -> set[str]:
    """Load canonical job family names from job_families.yaml.

    Resolved relative to __file__ so it works regardless of CWD.

    Args:
        config_path: Override path to job_families.yaml (for testing).

    Returns:
        Set of canonical job family strings.
    """
    path = config_path or (
        Path(__file__).parent.parent / "extraction" / "config" / "job_families.yaml"
    )
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return set(data["families"])


def drop_columns_and_rows(df: pd.DataFrame, valid_families: set[str]) -> pd.DataFrame:
    """Drop unneeded columns and parse-failure rows.

    Drops: country, location, source_file, title_original (if present).
    Drops rows where job_family is empty string (parse failures that weren't
    caught by the "NA" standardisation — e.g. blank string from LLM).

    Args:
        df: Cleaned DataFrame.
        valid_families: Set of canonical job family names (unused here, kept for API compat).

    Returns:
        DataFrame with unneeded columns and parse-failure rows removed.
    """
    df = df.copy()

    _drop_candidates = ("country", "location", "source_file", "title_original")
    cols_to_drop = [c for c in _drop_candidates if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    if "job_family" in df.columns:
        empty_mask = df["job_family"].str.strip() == ""
        n_empty = empty_mask.sum()
        if n_empty:
            logger.warning("Dropping %d rows with empty job_family (parse failures)", n_empty)
            df = df[~empty_mask].copy()

    return df


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder to canonical COLUMN_ORDER; drop any extra columns.

    Args:
        df: DataFrame to reorder.

    Returns:
        DataFrame containing only COLUMN_ORDER columns in order.
    """
    present = [c for c in COLUMN_ORDER if c in df.columns]
    return df[present].copy()


def to_analysis_ready(df: pd.DataFrame) -> pd.DataFrame:
    """Produce a lean analysis-ready DataFrame by dropping QA-only columns.

    Drops: validation_flags, description_quality, description.
    These columns are diagnostic — useful for debugging but not for analysis.

    Args:
        df: Fully cleaned DataFrame (already reordered).

    Returns:
        DataFrame with only analysis-relevant columns.
    """
    present = [c for c in ANALYSIS_COLUMN_ORDER if c in df.columns]
    return df[present].copy()


def assert_invariants(df: pd.DataFrame, valid_families: set[str]) -> None:
    """Assert final data quality invariants.

    Raises AssertionError with a descriptive message if any invariant fails.

    Args:
        df: Fully cleaned DataFrame.
        valid_families: Set of canonical job family strings.
    """
    # 1. No empty strings in key categorical columns
    for col in ("job_family", "contract_type", "work_modality", "seniority_from_title"):
        if col not in df.columns:
            continue
        empties = (df[col].str.strip() == "").sum()
        assert empties == 0, f"{empties} empty strings in '{col}'"

    # 2. All job_family values must be in the valid enum (or None for unknowns)
    if "job_family" in df.columns and valid_families:
        valid_mask = df["job_family"].isin(valid_families) | df["job_family"].isna()
        invalid = df[~valid_mask]["job_family"].unique()
        assert len(invalid) == 0, f"Invalid job_family values: {list(invalid)[:10]}"

    # 3. All list columns must contain valid JSON arrays
    for col in _LIST_COLUMNS:
        if col not in df.columns:
            continue
        bad_count = 0
        for val in df[col]:
            try:
                parsed = json.loads(val)
                if not isinstance(parsed, list):
                    bad_count += 1
            except (json.JSONDecodeError, TypeError, ValueError):
                bad_count += 1
        assert bad_count == 0, f"{bad_count} non-JSON-array values in '{col}'"

    # 4. Salary sanity: where both min and max are present, min <= max
    if "salary_min" in df.columns and "salary_max" in df.columns:
        has_both = df["salary_min"].notna() & df["salary_max"].notna()
        if has_both.any():
            sub = df[has_both].copy()
            mins = sub["salary_min"].astype(int)
            maxs = sub["salary_max"].astype(int)
            inversions = (mins > maxs).sum()
            assert inversions == 0, f"{inversions} rows where salary_min > salary_max"

    # 5. Unique row_ids
    if "row_id" in df.columns:
        dupes = df["row_id"].duplicated().sum()
        assert dupes == 0, f"{dupes} duplicate row_ids"

    # 6. No sentinel strings in company_name (should be None after standardization)
    if "company_name" in df.columns:
        na_string_mask = df["company_name"].str.strip().isin(MISSING_SENTINELS)
        assert na_string_mask.sum() == 0, (
            f"{int(na_string_mask.sum())} sentinel string values in company_name "
            f"(should be None)"
        )

    # 7. No residual gender markers in title_cleaned
    if "title_cleaned" in df.columns:
        residual_mask = df["title_cleaned"].str.contains(
            r"\((?:gn|m,w,d|m,f,d|all\s+genders?)\)",
            case=False,
            regex=True,
            na=False,
        )
        assert residual_mask.sum() == 0, (
            f"{int(residual_mask.sum())} residual gender markers in title_cleaned"
        )

    # 8. No duplicate entries within a row (case-insensitive)
    for col in _LIST_COLUMNS:
        if col not in df.columns:
            continue
        for idx, val in df[col].items():
            try:
                skills = json.loads(val)
                if isinstance(skills, list):
                    lower_skills = [s.lower() for s in skills if isinstance(s, str)]
                    assert len(lower_skills) == len(set(lower_skills)), (
                        f"Duplicate entries (case-insensitive) in row {idx}, column '{col}'"
                    )
            except (json.JSONDecodeError, TypeError):
                pass

    # 9. No empty-string elements in JSON list columns
    for col in _LIST_COLUMNS:
        if col not in df.columns:
            continue
        for idx, val in df[col].items():
            try:
                skills = json.loads(val)
                if isinstance(skills, list):
                    empty_entries = [s for s in skills if isinstance(s, str) and s.strip() == ""]
                    assert not empty_entries, (
                        f"Empty string entry in row {idx}, column '{col}'"
                    )
            except (json.JSONDecodeError, TypeError):
                pass
