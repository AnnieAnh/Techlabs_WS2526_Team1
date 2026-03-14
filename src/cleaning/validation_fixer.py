"""Validation flag format conversion for the cleaning pipeline."""

import json
import logging
import re

import pandas as pd

from shared.constants import MISSING_SENTINELS

logger = logging.getLogger("pipeline.validation_fixer")

# Python repr uses single quotes; JSON requires double quotes.
# This regex converts single-quoted strings to double-quoted when safe.
_SINGLE_TO_DOUBLE_RE = re.compile(r"'([^']*?)'")


def _python_repr_to_json(s: str) -> str | None:
    """Attempt to convert a Python repr string to valid JSON.

    Handles the common case where validation_flags were serialized with
    str() instead of json.dumps() (single quotes instead of double quotes,
    True/False/None instead of true/false/null).

    Returns the parsed JSON string if successful, None otherwise.
    """
    converted = _SINGLE_TO_DOUBLE_RE.sub(r'"\1"', s)
    converted = converted.replace("True", "true").replace("False", "false").replace("None", "null")
    try:
        parsed = json.loads(converted)
        if isinstance(parsed, list):
            return json.dumps(parsed, ensure_ascii=False)
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def fix_validation_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Convert validation_flags from Python-repr strings to JSON strings.

    Args:
        df: DataFrame with validation_flags column.

    Returns:
        DataFrame with validation_flags as valid JSON arrays.
    """
    if "validation_flags" not in df.columns:
        return df

    df = df.copy()

    def _to_json_flags(val: object) -> str:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "[]"
        if isinstance(val, list):
            return json.dumps(val, ensure_ascii=False)
        s = str(val).strip()
        if s in MISSING_SENTINELS:
            return "[]"
        # Try direct JSON parse first (normal case)
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return json.dumps(parsed, ensure_ascii=False)
        except (json.JSONDecodeError, ValueError):
            pass
        # Fallback: convert Python repr (single quotes, True/False/None) to JSON
        result = _python_repr_to_json(s)
        if result is not None:
            return result
        logger.warning("Could not parse validation_flags value: %r", s[:120])
        return "[]"

    df["validation_flags"] = df["validation_flags"].apply(_to_json_flags)
    return df
