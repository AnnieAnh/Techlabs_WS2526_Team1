"""NA standardization and numeric column fixing for the cleaning pipeline."""

import json
import logging
import re

import pandas as pd

from cleaning.constants import (
    _LIST_COLUMNS,
    _SALARY_COLUMNS,
    _SALARY_MAX_CEILING,
    _SALARY_MIN_FLOOR,
    _STRING_EXTRACTED_COLUMNS,
)
from shared.constants import MISSING_SENTINELS as _MISSING_SENTINELS
from shared.json_utils import parse_json_list

logger = logging.getLogger("pipeline.missing_values")

# Regex: German thousands format — digits separated by dots in groups of 3
_GERMAN_FMT = re.compile(r"^\d{1,3}(\.\d{3})+$")


def standardize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise missing values to None.

    List columns: empty/None/NaN → "[]"; lists are also deduplicated
    case-insensitively (preserving first occurrence and original casing).
    String extracted columns: None/NaN/empty/sentinel strings → None.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with standardized missing values (None instead of "NA").
    """
    df = df.copy()

    def _safe_list(v: str) -> str:
        """Parse JSON list, deduplicate case-insensitively, re-serialize."""
        parsed = parse_json_list(v)
        seen: set[str] = set()
        deduped: list = []
        for item in parsed:
            key = item.lower() if isinstance(item, str) else str(item)
            if key not in seen:
                seen.add(key)
                deduped.append(item)
        return json.dumps(deduped, ensure_ascii=False)

    for col in _LIST_COLUMNS:
        if col not in df.columns:
            continue
        df[col] = df[col].apply(
            lambda v: "[]"
            if (v is None or isinstance(v, float) or str(v).strip() in _MISSING_SENTINELS
                or str(v).strip() == "[]")
            else str(v)
        )
        df[col] = df[col].apply(_safe_list)

    for col in _STRING_EXTRACTED_COLUMNS:
        if col not in df.columns:
            continue
        df[col] = df[col].apply(
            lambda v: None
            if (v is None or (isinstance(v, float) and pd.isna(v))
                or str(v).strip() in _MISSING_SENTINELS)
            else str(v).strip()
        )
    return df


def fix_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert salary/experience to numeric; null salary outliers.

    Handles German number format (50.000 = 50,000) by stripping dots in salary
    fields before parsing. Salary values outside [10k, 300k] are set to None.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with numeric columns normalized (int or None).
    """
    df = df.copy()

    def _to_int_or_none(val: object, col: str) -> int | None:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        s = str(val).strip()
        if s in _MISSING_SENTINELS:
            return None
        if _GERMAN_FMT.match(s):
            s_clean = s.replace(".", "")
        else:
            s_clean = s.replace(",", "")
        try:
            n = int(float(s_clean))
        except (ValueError, OverflowError):
            return None
        if col in _SALARY_COLUMNS:
            if n < _SALARY_MIN_FLOOR or n > _SALARY_MAX_CEILING:
                return None
        return n

    for col in _SALARY_COLUMNS + ["experience_years"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda v, c=col: _to_int_or_none(v, c))
    return df
