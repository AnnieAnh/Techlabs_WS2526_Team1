"""DataFrame filter helpers for analysis notebooks."""

import pandas as pd

from shared.json_utils import explode_json_column


def filter_by_job_family(df: pd.DataFrame, family: str) -> pd.DataFrame:
    """Return rows where job_family matches (case-insensitive).

    Args:
        df: Source DataFrame with a 'job_family' column.
        family: Job family name to filter by.

    Returns:
        Filtered DataFrame.
    """
    return df[df["job_family"].str.lower() == family.lower()]


def filter_by_seniority(df: pd.DataFrame, seniority: str) -> pd.DataFrame:
    """Return rows where seniority_from_title matches (case-insensitive).

    Args:
        df: Source DataFrame with a 'seniority_from_title' column.
        seniority: Seniority level to filter by.

    Returns:
        Filtered DataFrame.
    """
    return df[df["seniority_from_title"].str.lower() == seniority.lower()]


def filter_remote(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows where work_modality is 'Remote' (case-insensitive).

    Args:
        df: Source DataFrame with a 'work_modality' column.

    Returns:
        Filtered DataFrame.
    """
    return df[df["work_modality"].str.lower() == "remote"]


def filter_salary_known(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows where both salary_min and salary_max are valid numeric values.

    Expects salary columns to have been through fix_numeric_columns (int or None).

    Args:
        df: Source DataFrame with 'salary_min' and 'salary_max' columns.

    Returns:
        Filtered DataFrame (copy).
    """
    mask = df["salary_min"].notna() & df["salary_max"].notna()
    return df[mask].copy()


def salary_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return salary-known rows with salary_min/max as int columns and salary_mid derived.

    Args:
        df: Source DataFrame.

    Returns:
        Filtered DataFrame with integer salary columns and a 'salary_mid' column added.
    """
    out = filter_salary_known(df).copy()
    out["salary_min"] = pd.to_numeric(out["salary_min"]).astype(int)
    out["salary_max"] = pd.to_numeric(out["salary_max"]).astype(int)
    out["salary_mid"] = (out["salary_min"] + out["salary_max"]) // 2
    return out


def explode_json_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Explode a JSON-list column into one row per element.

    Args:
        df: Source DataFrame.
        col: Column name containing JSON arrays (e.g. '["Python", "Docker"]').

    Returns:
        Exploded DataFrame with one element per row; null elements dropped.
    """
    return explode_json_column(df, col)
