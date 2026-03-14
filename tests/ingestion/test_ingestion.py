"""Tests for the ingestion pipeline."""

from datetime import datetime

import numpy as np
import pandas as pd

from ingestion.date_parser import parse_date_to_exact
from ingestion.loader import fill_missing_values, normalize_whitespace

# ---------------------------------------------------------------------------
# Company name fill value
# ---------------------------------------------------------------------------


def test_fill_company_name_none():
    """NaN company_name is filled with None (NA boundary is at CSV I/O)."""
    df = pd.DataFrame({"company_name": [np.nan, "Acme GmbH", np.nan]})
    result = fill_missing_values(df)
    assert result.loc[0, "company_name"] is None or pd.isna(result.loc[0, "company_name"])
    assert result.loc[1, "company_name"] == "Acme GmbH"
    assert result.loc[2, "company_name"] is None or pd.isna(result.loc[2, "company_name"])


def test_fill_company_name_not_string_na():
    """The fill value must be None/NaN, not the string 'NA'."""
    df = pd.DataFrame({"company_name": [np.nan]})
    result = fill_missing_values(df)
    assert result.loc[0, "company_name"] != "NA"
    assert result.loc[0, "company_name"] is None or pd.isna(result.loc[0, "company_name"])


# ---------------------------------------------------------------------------
# Pinned reference date
# ---------------------------------------------------------------------------


def test_parse_date_pinned_reference():
    """Same relative date + same pinned reference date → same ISO output."""
    ref = datetime(2026, 2, 15)
    result1 = parse_date_to_exact("2 days ago", reference_date=ref)
    result2 = parse_date_to_exact("2 days ago", reference_date=ref)
    assert result1 == result2
    assert result1 == "2026-02-13"


def test_parse_date_different_references_differ():
    """Same relative date + different reference dates → different output."""
    ref_a = datetime(2026, 2, 15)
    ref_b = datetime(2026, 2, 20)
    assert parse_date_to_exact("3 days ago", reference_date=ref_a) != parse_date_to_exact(
        "3 days ago", reference_date=ref_b
    )


# ---------------------------------------------------------------------------
# normalize_whitespace
# ---------------------------------------------------------------------------


def test_normalize_whitespace_strips_all_string_cols():
    """normalize_whitespace strips leading/trailing whitespace from string cols."""
    df = pd.DataFrame({
        "title": ["  Senior Developer  "],
        "company_name": [" Acme GmbH "],
        "location": ["  Berlin, Germany"],
    })
    result = normalize_whitespace(df)
    assert result.loc[0, "title"] == "Senior Developer"
    assert result.loc[0, "company_name"] == "Acme GmbH"
    assert result.loc[0, "location"] == "Berlin, Germany"


def test_normalize_whitespace_non_string_cols_untouched():
    """normalize_whitespace leaves non-string (numeric) columns unchanged."""
    df = pd.DataFrame({
        "title": ["  Developer  "],
        "salary": [60000],
    })
    result = normalize_whitespace(df)
    assert result.loc[0, "salary"] == 60000


