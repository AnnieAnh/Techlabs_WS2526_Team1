"""Date parsing utilities for ingestion pipeline."""

import re
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from shared.constants import MISSING_SENTINELS


def parse_date_to_exact(
    value: Any,
    reference_date: datetime | None = None,
) -> str | None:
    """Convert date_posted value to ISO date string (YYYY-MM-DD).

    Handles:
        - ISO dates (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
        - Unix timestamps (seconds or milliseconds)
        - Relative strings: "X days ago", "Today", "Yesterday", etc.

    Args:
        value: Raw date value (string, int, or None).
        reference_date: Reference for relative dates; defaults to now.

    Returns:
        ISO date string or None if unparseable.
    """
    if pd.isna(value) or value is None:
        return None

    val = str(value).strip()
    if val in MISSING_SENTINELS:
        return None

    ref = reference_date or datetime.now()

    # Already ISO format
    iso_match = re.match(r"^(\d{4}-\d{2}-\d{2})", val)
    if iso_match:
        return iso_match.group(1)

    # Unix timestamp (milliseconds or seconds)
    if val.isdigit() or (val.startswith("-") and val[1:].isdigit()):
        ts = int(val)
        divisor = 1000 if ts > 1e12 else 1
        dt = datetime.fromtimestamp(ts / divisor, tz=UTC)
        return dt.strftime("%Y-%m-%d")

    # Relative strings
    val_lower = val.lower()
    if val_lower in ("today", "just now", "just posted"):
        return ref.strftime("%Y-%m-%d")
    if val_lower == "yesterday":
        return (ref - timedelta(days=1)).strftime("%Y-%m-%d")
    if re.match(r"^\d+\s*hours?\s+ago$", val_lower):
        return ref.strftime("%Y-%m-%d")

    # X day(s)/week(s)/month(s) ago
    match = re.match(r"^(\d+)\s*(day|week|month)s?\s+ago$", val_lower)
    if match:
        n = int(match.group(1))
        unit = match.group(2)
        if unit == "day":
            return (ref - timedelta(days=n)).strftime("%Y-%m-%d")
        if unit == "week":
            return (ref - timedelta(weeks=n)).strftime("%Y-%m-%d")
        if unit == "month":
            return (ref - timedelta(days=n * 30)).strftime("%Y-%m-%d")

    return None


def normalize_date_posted(
    df: pd.DataFrame,
    reference_date: datetime | None = None,
) -> pd.DataFrame:
    """Convert date_posted column to ISO date strings (YYYY-MM-DD).

    Args:
        df: DataFrame with date_posted column.
        reference_date: Reference for relative dates.

    Returns:
        DataFrame with date_posted normalized to YYYY-MM-DD strings (or None).
    """
    if "date_posted" not in df.columns:
        return df
    df = df.copy()
    ref = reference_date or datetime.now()
    df["date_posted"] = df["date_posted"].apply(
        lambda x: parse_date_to_exact(x, ref)
    )
    return df
