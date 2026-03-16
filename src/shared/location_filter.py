"""Coarse non-German location filter.

Provides a regex-based heuristic for detecting clearly non-German job locations.
Currently unused — available for future pipeline integration.
"""

import re

import pandas as pd

# Patterns that strongly suggest a non-German job location.
# Intentionally broad — false positives here only mean the extraction
# pipeline has to do more location-stage work (it will correct them).
_NON_GERMAN_PATTERNS = re.compile(
    r"\b(?:United States|USA|Prague|Brussels|Metropolitan Area)\b"
    r"|\bDACH\b"
    r"|,\s*[A-Z]{2}\s*$",
    re.IGNORECASE,
)


def is_likely_non_german(location: str) -> bool:
    """Return True if the location string suggests a non-German job posting.

    Args:
        location: Raw location string from a job posting.

    Returns:
        True if the location is likely outside Germany, False otherwise.
    """
    if pd.isna(location) or location is None:
        return False
    return bool(_NON_GERMAN_PATTERNS.search(str(location)))
