"""DataFrame filter helpers for analysis notebooks."""

import pandas as pd

from shared.json_utils import explode_json_column, parse_json_list


def exclude_future_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with the sentinel future date (2027-01-01) used for missing dates.

    Args:
        df: Source DataFrame with a 'date_posted' column.

    Returns:
        Filtered DataFrame without future-dated sentinel rows.
    """
    return df[df["date_posted"] != "2027-01-01"].copy()


def exclude_other_family(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where job_family is 'Other' (meaningless in cross-family comparisons).

    Args:
        df: Source DataFrame with a 'job_family' column.

    Returns:
        Filtered DataFrame without 'Other' job family rows.
    """
    return df[df["job_family"] != "Other"].copy()


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
        seniority: Seniority level to filter by (e.g. 'Junior', 'Senior', 'Lead').

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

    Works after fix_numeric_columns has coerced salary columns to numeric values or None.
    Uses pd.notna() to detect which rows have valid salary values.

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
        Filtered DataFrame with int64 salary columns and a 'salary_mid' column
        (floor of the midpoint: (min + max) // 2).
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


def filter_by_modality(df: pd.DataFrame, modality: str) -> pd.DataFrame:
    """Return rows where work_modality matches (case-insensitive).

    Args:
        df: Source DataFrame with a 'work_modality' column.
        modality: One of 'Remote', 'Hybrid', 'Onsite'.

    Returns:
        Filtered DataFrame.
    """
    return df[df["work_modality"].str.lower() == modality.lower()]


def filter_by_city(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """Return rows for a specific city (case-insensitive).

    Args:
        df: Source DataFrame with a 'city' column.
        city: City name (e.g. 'Berlin', 'München').

    Returns:
        Filtered DataFrame.
    """
    return df[df["city"].str.lower() == city.lower()]


def filter_by_state(df: pd.DataFrame, state: str) -> pd.DataFrame:
    """Return rows for a specific German federal state (case-insensitive).

    Args:
        df: Source DataFrame with a 'state' column.
        state: State name (e.g. 'Bavaria', 'Berlin').

    Returns:
        Filtered DataFrame.
    """
    return df[df["state"].str.lower() == state.lower()]


def filter_description_quality(
    df: pd.DataFrame, quality: str = "clean"
) -> pd.DataFrame:
    """Return rows matching a description quality label.

    Args:
        df: Source DataFrame with a 'description_quality' column.
        quality: One of 'clean' or 'concatenated'.

    Returns:
        Filtered DataFrame.
    """
    return df[df["description_quality"] == quality]


def rows_with_skill(df: pd.DataFrame, skill: str, col: str = "technical_skills") -> pd.DataFrame:
    """Return rows where a specific skill appears in a JSON-list column.

    Performs a fast case-insensitive string containment check rather than
    full JSON parsing — suitable for exploratory filtering.

    Args:
        df: Source DataFrame.
        skill: Skill name to search for (e.g. 'Python', 'React').
        col: Column to search in. Defaults to 'technical_skills'.

    Returns:
        Filtered DataFrame of rows mentioning the skill.
    """
    pattern = skill.lower()
    mask = df[col].apply(
        lambda v: pattern in str(v).lower() if pd.notna(v) else False
    )
    return df[mask]


def parse_list_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Parse a JSON-list column to a Series of Python lists.

    Convenience wrapper around shared.json_utils.parse_json_list for use
    directly on a DataFrame column without going through utils.parse_json_col.

    Args:
        df: Source DataFrame.
        col: Column containing JSON array strings.

    Returns:
        Series of Python lists (empty list for unparseable values).
    """
    return df[col].apply(parse_json_list)
