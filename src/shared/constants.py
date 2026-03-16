"""Shared constants for all pipeline components.

Import from here rather than defining locally in each module.
"""

# ---------------------------------------------------------------------------
# Salary thresholds
# ---------------------------------------------------------------------------

# Canonical floor for annual salary (EUR). Captures legitimate part-time IT
# roles in Germany. The validator's "below_floor" warning uses this as its
# fallback when cfg["validation"] does not override it.
SALARY_MIN_FLOOR: int = 10_000
SALARY_MAX_CEILING: int = 300_000

# Values below this are flagged as possible monthly (rather than annual) salaries.
SALARY_MONTHLY_THRESHOLD: int = 5_000

# ---------------------------------------------------------------------------
# Missing value sentinels
# ---------------------------------------------------------------------------

# Canonical set of string values that represent missing data.
# Use MISSING_SENTINELS for membership tests instead of inline string literals
# scattered across modules.
# Note: "null" included to cover location_parser._FALSY_STRINGS usage.
MISSING_SENTINELS: frozenset[str] = frozenset({
    "",
    "nan",
    "NaN",
    "None",
    "none",
    "NA",
    "na",
    "N/A",
    "n/a",
    "null",
})
