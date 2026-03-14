"""Tests for extraction/dedup/description_dedup.py."""

import json

import pandas as pd
import pytest

from extraction.dedup.description_dedup import (
    _hash_description,
    find_near_duplicates,
    group_rows_by_description,
    propagate_group_results,
)


@pytest.fixture
def deduped_dir(tmp_path):
    d = tmp_path / "deduped"
    d.mkdir()
    return d


def _make_df(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "row_id": "r000",
        "title": "Engineer",
        "company_name": "ACME",
        "description": "Some job description text that is long enough",
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


def test_identical_descriptions_same_hash():
    assert _hash_description("same text") == _hash_description("same text")


def test_different_descriptions_different_hash():
    assert _hash_description("text one") != _hash_description("text two")


def test_whitespace_normalized_same_hash():
    assert _hash_description("hello   world") == _hash_description("hello world")
    assert _hash_description("  leading") == _hash_description("leading")
    assert _hash_description("trailing  ") == _hash_description("trailing")


def test_different_whitespace_but_different_content_different_hash():
    assert _hash_description("abc def") != _hash_description("abc xyz")


def test_identical_descriptions_grouped(deduped_dir):
    desc = "Same description text repeated here " * 5
    df = _make_df([
        {"row_id": "r1", "description": desc},
        {"row_id": "r2", "description": desc},
        {"row_id": "r3", "description": desc},
    ])
    groups = group_rows_by_description(df, deduped_dir)

    # All 3 rows have the same description → 1 group
    assert len(groups) == 1
    group = next(iter(groups.values()))
    assert group["count"] == 3
    assert set(group["member_row_ids"]) == {"r1", "r2", "r3"}


def test_different_descriptions_separate_groups(deduped_dir):
    df = _make_df([
        {"row_id": "r1", "description": "Description A " * 5},
        {"row_id": "r2", "description": "Description B " * 5},
        {"row_id": "r3", "description": "Description C " * 5},
    ])
    groups = group_rows_by_description(df, deduped_dir)
    assert len(groups) == 3


def test_representative_is_first_in_order(deduped_dir):
    desc = "Shared description text here " * 5
    df = _make_df([
        {"row_id": "r1", "description": desc},
        {"row_id": "r2", "description": desc},
    ])
    groups = group_rows_by_description(df, deduped_dir)
    group = next(iter(groups.values()))
    assert group["representative_row_id"] == "r1"  # first in DataFrame order


def test_representative_selection_deterministic(deduped_dir, tmp_path):
    """Same data → same representative every time."""
    desc = "Deterministic description text here " * 5
    df = _make_df([
        {"row_id": "r1", "description": desc},
        {"row_id": "r2", "description": desc},
    ])
    dir1 = tmp_path / "d1"
    dir1.mkdir()
    dir2 = tmp_path / "d2"
    dir2.mkdir()

    g1 = group_rows_by_description(df, dir1)
    g2 = group_rows_by_description(df, dir2)

    rep1 = next(iter(g1.values()))["representative_row_id"]
    rep2 = next(iter(g2.values()))["representative_row_id"]
    assert rep1 == rep2


def test_groups_json_written(deduped_dir):
    df = _make_df([{"row_id": "r1", "description": "Some description " * 10}])
    group_rows_by_description(df, deduped_dir)
    out_path = deduped_dir / "description_groups.json"
    assert out_path.exists()
    with open(out_path, encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict)
    assert len(data) == 1


def test_single_row_per_description_count_1(deduped_dir):
    df = _make_df([
        {"row_id": "r1", "description": "Unique A " * 5},
        {"row_id": "r2", "description": "Unique B " * 5},
    ])
    groups = group_rows_by_description(df, deduped_dir)
    for group in groups.values():
        assert group["count"] == 1


def test_fan_out_copies_value_to_members(deduped_dir):
    desc = "Shared description " * 5
    df = _make_df([
        {"row_id": "r1", "description": desc},
        {"row_id": "r2", "description": desc},
    ])
    df["extracted_value"] = None
    df.loc[df["row_id"] == "r1", "extracted_value"] = "Python"  # rep has value

    groups = group_rows_by_description(df, deduped_dir)
    result = propagate_group_results(df, groups, "extracted_value")

    assert result.loc[result["row_id"] == "r2", "extracted_value"].iloc[0] == "Python"


def test_fan_out_does_not_overwrite_rep(deduped_dir):
    desc = "Shared description " * 5
    df = _make_df([
        {"row_id": "r1", "description": desc},
        {"row_id": "r2", "description": desc},
    ])
    df["extracted_value"] = None
    df.loc[df["row_id"] == "r1", "extracted_value"] = "Python"

    groups = group_rows_by_description(df, deduped_dir)
    result = propagate_group_results(df, groups, "extracted_value")

    assert result.loc[result["row_id"] == "r1", "extracted_value"].iloc[0] == "Python"


def test_fan_out_missing_column_returns_df_unchanged(deduped_dir):
    df = _make_df([{"row_id": "r1", "description": "Desc " * 5}])
    groups = group_rows_by_description(df, deduped_dir)
    result = propagate_group_results(df, groups, "nonexistent_col")
    assert "nonexistent_col" not in result.columns


# ---------------------------------------------------------------------------
# find_near_duplicates (MinHash LSH pass)
# ---------------------------------------------------------------------------


def _make_similar(base: str, suffix: str) -> str:
    """Add a short suffix to create a slightly different but very similar text."""
    return base + " " + suffix


def test_near_duplicates_identical_descriptions():
    """Identical descriptions must be detected as near-duplicates."""
    text = "We are looking for a Python developer with experience in Django and REST APIs."
    df = _make_df([
        {"row_id": "r1", "description": text},
        {"row_id": "r2", "description": text},
    ])
    result = find_near_duplicates(df)
    # One of the two must map to the other as representative
    assert len(result) >= 1
    mapped_ids = set(result.keys())
    assert mapped_ids.issubset({"r1", "r2"})


def test_near_duplicates_very_similar_descriptions():
    """Descriptions sharing ≥95% of their unique tokens are flagged as near-dups."""
    # Build 100 unique words — change only 1 in r2 so Jaccard ≥ 0.99
    words = [f"token{i}" for i in range(100)]
    base = " ".join(words)
    almost_same = " ".join(words[:-1] + ["different"])
    df = _make_df([
        {"row_id": "r1", "description": base},
        {"row_id": "r2", "description": almost_same},
    ])
    result = find_near_duplicates(df, threshold=0.90)
    # r2 should map back to r1 (first-seen representative)
    assert "r2" in result
    assert result["r2"] == "r1"


def test_near_duplicates_distinct_descriptions_not_flagged():
    """Completely different descriptions must not be flagged as near-duplicates."""
    df = _make_df([
        {"row_id": "r1", "description": "Python backend developer Django REST API"},
        {"row_id": "r2", "description": "Sales manager B2B enterprise accounts revenue quota"},
    ])
    result = find_near_duplicates(df)
    assert result == {}


def test_near_duplicates_empty_dataframe():
    df = _make_df([])
    result = find_near_duplicates(df)
    assert result == {}


def test_near_duplicates_single_row():
    df = _make_df([{"row_id": "r1", "description": "Python developer"}])
    result = find_near_duplicates(df)
    assert result == {}


def test_near_duplicates_returns_dict():
    df = _make_df([{"row_id": "r1", "description": "Any text"}])
    result = find_near_duplicates(df)
    assert isinstance(result, dict)
