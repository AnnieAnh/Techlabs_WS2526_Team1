"""Shared data utilities for analysis notebooks.

Usage in notebooks:
    from analysis.utils import load_enriched, parse_json_col, parse_flags
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from shared.io import read_csv_safe
from shared.json_utils import parse_json_list


def load_enriched(path: str | None = None) -> pd.DataFrame:
    """Load the cleaned enriched jobs CSV.

    Converts "NA" strings to None via read_csv_safe so callers can use
    .isna() / .notna() for missing-value checks.

    Args:
        path: Path to the CSV file. Defaults to data/cleaning/cleaned_jobs.csv
              relative to the repository root.

    Returns:
        DataFrame with all enriched columns (None for missing values).
    """
    if path is not None:
        p = Path(path)
    else:
        # src/analysis/ → src/ → repo root → data/cleaning/cleaned_jobs.csv
        p = Path(__file__).parent.parent.parent / "data" / "cleaning" / "cleaned_jobs.csv"
    return read_csv_safe(p)


def parse_json_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Parse a JSON list column into Python lists.

    Args:
        df: DataFrame containing the column.
        col: Column name with JSON list strings.

    Returns:
        Series of Python lists (empty list for unparseable values).
    """
    return df[col].apply(parse_json_list)


def parse_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Expand validation_flags JSON into a one-row-per-flag DataFrame.

    Args:
        df: DataFrame with a 'validation_flags' column (JSON strings).

    Returns:
        DataFrame with columns: row_id, rule, severity, field, message.
        Empty DataFrame if no flags exist.
    """
    rows = []
    for _, row in df.iterrows():
        try:
            flags = json.loads(str(row.get("validation_flags", "[]")))
        except (json.JSONDecodeError, TypeError):
            flags = []
        for flag in flags:
            rows.append({"row_id": row.get("row_id"), **flag})
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["row_id", "rule", "severity", "field", "message"]
    )


def notebook_init() -> pd.DataFrame:
    """Standard notebook setup: apply style, load data, configure figures directory.

    Call this once at the top of each analysis notebook.
    Sets charts.FIGURES_DIR so chart functions can save figures automatically.

    Returns:
        The loaded cleaned jobs DataFrame.
    """
    from analysis import charts as _charts
    from analysis.style import set_style

    set_style()
    df = load_enriched()

    figures_dir = Path(__file__).parent.parent.parent / "data" / "analysis" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    _charts.FIGURES_DIR = figures_dir

    print(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df
