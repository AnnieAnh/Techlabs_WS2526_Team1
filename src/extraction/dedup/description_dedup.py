"""Description dedup: group identical descriptions to avoid duplicate LLM extractions.

Two dedup passes:
  Pass 1 (exact): SHA-256 hash of normalized description text.
                  Called by steps/deduplicate.py.
  Pass 2 (near):  MinHash LSH at 95% Jaccard similarity (requires datasketch).
                  Called by steps/deduplicate.py after group_rows_by_description().
                  Falls back silently if datasketch is not installed.
"""

import hashlib
import json
import logging
import re
from pathlib import Path

import pandas as pd

try:
    from datasketch import MinHash, MinHashLSH

    _DATASKETCH_AVAILABLE = True
except ImportError:
    _DATASKETCH_AVAILABLE = False

logger = logging.getLogger("pipeline.description_dedup")


def _normalize_description(text: str) -> str:
    """Collapse all whitespace to single spaces for consistent hashing.

    Args:
        text: Raw description string.

    Returns:
        Whitespace-normalized string.
    """
    return re.sub(r"\s+", " ", text).strip()


def _hash_description(text: str) -> str:
    """Return SHA-256 hex digest of the normalized description.

    Args:
        text: Raw description string (will be normalized before hashing).

    Returns:
        Hex string of SHA-256 digest.
    """
    normalized = _normalize_description(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _minhash_description(text: str, num_perm: int = 128) -> "MinHash":
    """Build a MinHash signature for a description string."""
    m = MinHash(num_perm=num_perm)
    for token in _normalize_description(text).lower().split():
        m.update(token.encode("utf-8"))
    return m


def find_near_duplicates(
    df: pd.DataFrame,
    threshold: float = 0.95,
    num_perm: int = 128,
) -> dict[str, str]:
    """Find near-duplicate descriptions using MinHash LSH (Pass 2).

    Identifies pairs with ≥ threshold Jaccard similarity and maps each
    near-duplicate row_id to the first-seen representative row_id.

    Returns an empty dict if datasketch is not installed.

    Args:
        df: DataFrame with 'row_id' and 'description' columns.
        threshold: Minimum Jaccard similarity to consider near-duplicates.
        num_perm: Number of permutations for MinHash (higher = more accurate).

    Returns:
        Dict mapping near-duplicate row_id → canonical representative row_id.
        Rows not mapped are their own representative.
    """
    if not _DATASKETCH_AVAILABLE:
        logger.debug("datasketch not installed — skipping MinHash near-duplicate pass")
        return {}

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes: dict[str, MinHash] = {}
    near_dup_map: dict[str, str] = {}

    for _, row in df.iterrows():
        row_id = str(row["row_id"])
        mh = _minhash_description(str(row["description"]), num_perm)
        minhashes[row_id] = mh

        try:
            candidates = lsh.query(mh)
        except Exception:
            candidates = []

        if candidates:
            # Map this row to the first-seen representative
            representative = candidates[0]
            near_dup_map[row_id] = near_dup_map.get(representative, representative)
        else:
            try:
                lsh.insert(row_id, mh)
            except ValueError:
                pass  # Already inserted (shouldn't happen — but be safe)

    # Resolve transitive chains: if A→B and B→C, resolve to A→C, B→C
    changed = True
    while changed:
        changed = False
        for dup_id, rep_id in list(near_dup_map.items()):
            if rep_id in near_dup_map:
                near_dup_map[dup_id] = near_dup_map[rep_id]
                changed = True

    if near_dup_map:
        unique_reps = len(set(near_dup_map.values()))
        logger.info(
            "MinHash near-duplicate pass: %d near-dups found → %d clusters at threshold=%.2f",
            len(near_dup_map),
            unique_reps,
            threshold,
        )
    return near_dup_map


def group_rows_by_description(df: pd.DataFrame, deduped_dir: Path) -> dict:
    """Hash descriptions and group rows that share an identical description.

    For each group with more than one row, one representative is chosen (the first
    row in DataFrame order). The others are 'fan-out targets' — they will receive
    extraction results copied from the representative after LLM extraction runs.

    Results are saved to deduped_dir/description_groups.json.

    Args:
        df: Deduped DataFrame (output of deduplicate_rows).
        deduped_dir: Directory to save the description_groups.json file.

    Returns:
        Dict mapping description hash → {representative_row_id, member_row_ids, ...}.
    """
    logger.info("Grouping rows by description hash ...")

    df = df.copy()
    df["_desc_hash"] = df["description"].apply(_hash_description)

    groups: dict[str, dict] = {}
    for desc_hash, group_df in df.groupby("_desc_hash", sort=False):
        hash_key = str(desc_hash)
        row_ids = group_df["row_id"].tolist()
        representative = row_ids[0]

        first_row = group_df.iloc[0]
        groups[hash_key] = {
            "representative_row_id": representative,
            "member_row_ids": row_ids,
            "count": len(row_ids),
            "title": str(first_row.get("title", "")),
            "company": str(first_row.get("company_name", "")),
        }

    multi = sum(1 for g in groups.values() if g["count"] > 1)
    saved = sum(g["count"] - 1 for g in groups.values() if g["count"] > 1)
    total = len(df)
    pct = saved / total * 100 if total > 0 else 0.0

    logger.info("Found %d unique descriptions out of %d rows", len(groups), total)
    logger.info(
        "LLM extraction needed for %d descriptions (saving %d API calls, %.1f%% cost reduction)",
        len(groups),
        saved,
        pct,
    )

    if groups:
        largest = max(groups.values(), key=lambda g: g["count"])
        if largest["count"] > 1:
            logger.info(
                "Largest group: '%s' by %s — %d identical postings",
                largest["title"],
                largest["company"],
                largest["count"],
            )

    out_path = deduped_dir / "description_groups.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(groups, f, indent=2, ensure_ascii=False)
    logger.info(
        "Description groups saved to %s (%d groups, %d multi)", out_path, len(groups), multi
    )

    return groups


def propagate_group_results(
    df: pd.DataFrame,
    groups: dict,
    extracted_col: str,
) -> pd.DataFrame:
    """Copy extraction results from each group's representative to all group members.

    After LLM extraction runs on representative rows, this function propagates
    the extracted value to every other row in the same description group.

    Args:
        df: DataFrame with extraction results (extracted_col may be missing for non-reps).
        groups: Groups dict from group_rows_by_description().
        extracted_col: Column name containing the extracted value to fan out.

    Returns:
        DataFrame with extracted_col filled in for all rows.
    """
    if extracted_col not in df.columns:
        logger.warning("Column '%s' not found in DataFrame — fan-out skipped", extracted_col)
        return df

    df = df.copy()
    row_id_to_value: dict[str, object] = dict(
        zip(df["row_id"], df[extracted_col], strict=False)
    )

    for group in groups.values():
        rep_id = group["representative_row_id"]
        rep_value = row_id_to_value.get(rep_id)
        if rep_value is None:
            continue
        for member_id in group["member_row_ids"]:
            if member_id != rep_id:
                row_id_to_value[member_id] = rep_value

    df[extracted_col] = df["row_id"].map(row_id_to_value)
    return df
