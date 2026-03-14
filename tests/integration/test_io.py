"""Tests for shared/io.py — read_csv_safe and write_csv_safe contract."""

from pathlib import Path

import pandas as pd

from shared.io import read_csv_safe, write_csv_safe


def test_read_csv_safe_converts_NA_string_to_na(tmp_path: Path) -> None:
    """'NA' in a CSV cell is converted to NaN by read_csv_safe (not kept as string).

    Verifies the boundary contract: 'NA' strings become missing (NaN) in-memory.
    Without keep_default_na=False, pandas would treat "NA" as NaN already, but
    read_csv_safe explicitly applies .replace("NA", None) to catch it even when
    keep_default_na=False is set.
    """
    p = tmp_path / "test.csv"
    p.write_text("company_name,title\nNA,Engineer\nActual Corp,Developer\n", encoding="utf-8")
    df = read_csv_safe(p)
    assert pd.isna(df.loc[0, "company_name"]), "'NA' should become NaN/missing"
    assert df.loc[1, "company_name"] == "Actual Corp"


def test_read_csv_safe_preserves_non_NA_strings(tmp_path: Path) -> None:
    """read_csv_safe must not corrupt non-NA string values."""
    p = tmp_path / "test.csv"
    p.write_text("name,value\nNA Technologies,100\nTechCorp,200\n", encoding="utf-8")
    df = read_csv_safe(p)
    # "NA Technologies" is NOT exactly "NA" — it should be kept as-is.
    assert df.loc[0, "name"] == "NA Technologies"
    assert df.loc[1, "name"] == "TechCorp"


def test_write_then_read_roundtrip_None(tmp_path: Path) -> None:
    """write_csv_safe → read_csv_safe is identity for missing values.

    None/NaN in memory → 'NA' on disk → NaN/missing back in memory.
    """
    p = tmp_path / "roundtrip.csv"
    original = pd.DataFrame({"company_name": [None, "TechCorp"], "salary": [None, 75000]})
    write_csv_safe(original, p)

    loaded = read_csv_safe(p)
    assert pd.isna(loaded.loc[0, "company_name"]), "None should round-trip to NaN"
    assert loaded.loc[1, "company_name"] == "TechCorp"
    assert pd.isna(loaded.loc[0, "salary"]), "None salary should round-trip to NaN"
    assert float(loaded.loc[1, "salary"]) == 75000.0


def test_read_csv_safe_utf8_encoding(tmp_path: Path) -> None:
    """read_csv_safe handles UTF-8 encoded files with German characters."""
    p = tmp_path / "german.csv"
    p.write_text("city,count\nMünchen,10\nKöln,5\n", encoding="utf-8")
    df = read_csv_safe(p)
    assert df.loc[0, "city"] == "München"
    assert df.loc[1, "city"] == "Köln"
