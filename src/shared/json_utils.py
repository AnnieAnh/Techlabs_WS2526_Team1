"""Shared JSON list parsing utilities for all pipeline phases.

Single source of truth for parsing JSON-encoded list columns produced by the
LLM extraction step. All three pipeline phases (extraction, cleaning, analysis)
use these functions instead of local implementations.
"""

import json
from ast import literal_eval

import pandas as pd


def parse_json_list(value: object) -> list:
    """Parse a JSON or Python-repr list string into a Python list.

    Tries json.loads first, falls back to ast.literal_eval for Python-style
    list strings the LLM occasionally produces (e.g. ``"['Python', 'SQL']"``).

    Args:
        value: A string, list, or any value to parse.

    Returns:
        A Python list, or an empty list for any unparseable value.
    """
    if isinstance(value, list):
        return value
    try:
        result = json.loads(str(value))
        return result if isinstance(result, list) else []
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    try:
        result = literal_eval(str(value))
        return result if isinstance(result, list) else []
    except Exception:
        return []


def parse_json_column(df: pd.DataFrame, col: str) -> pd.Series:
    """Parse a DataFrame column of JSON list strings into Python lists.

    Args:
        df: DataFrame containing the column.
        col: Column name with JSON list strings.

    Returns:
        Series of Python lists (empty list for unparseable values).
    """
    return df[col].apply(parse_json_list)


def explode_json_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Parse and explode a JSON list column into one row per element.

    Args:
        df: Source DataFrame.
        col: Column name containing JSON arrays.

    Returns:
        Exploded DataFrame with one element per row; null elements dropped.
    """
    out = df.copy()
    out[col] = out[col].apply(parse_json_list)
    return out.explode(col).dropna(subset=[col])
