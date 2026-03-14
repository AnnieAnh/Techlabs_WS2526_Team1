"""Step 3: Deduplicate — filter, URL dedup, composite dedup, grouping, MinHash.

Pre-filter: privacy-wall and invalid-URL rows (flagged in Step 2) are removed
BEFORE description grouping. This prevents garbage text from poisoning the
MinHash LSH groups sent to the LLM.

Dedup uses ``title_cleaned`` (set by step 2) for the composite key, so
``"Senior Developer (m/w/d)"`` and ``"Senior Developer"`` match correctly.

Pass 4 (MinHash near-dup) uses description_dedup.py for near-duplicate grouping
via LSH at 95% Jaccard similarity.
"""

import logging

import pandas as pd

from extraction.checkpoint import Checkpoint
from extraction.dedup.description_dedup import find_near_duplicates, group_rows_by_description
from extraction.dedup.row_dedup import deduplicate_rows
from pipeline_state import PipelineState
from shared.schemas import dedup_output_schema, validate_step_output

logger = logging.getLogger("pipeline.deduplicate")


def _filter_flagged_rows(df: pd.DataFrame, logger: logging.Logger) -> tuple[pd.DataFrame, dict]:
    """Remove rows flagged as privacy_wall or invalid_url by Step 2's validate_input.

    These rows contain garbage text (cookie consent screens, error pages) that
    would poison MinHash LSH groups if left in.

    Returns:
        Tuple of (filtered DataFrame, dict with removal counts).
    """
    import json as _json

    filter_counts = {"privacy_wall_removed": 0, "invalid_url_removed": 0}

    if "input_flags" not in df.columns:
        return df, filter_counts

    def _has_flag(flags_val: object, flag_name: str) -> bool:
        if not flags_val:
            return False
        if isinstance(flags_val, list):
            flags_list = flags_val
        elif isinstance(flags_val, str):
            try:
                flags_list = _json.loads(flags_val)
            except (ValueError, TypeError):
                return flag_name in flags_val
        else:
            return False
        return any(
            (isinstance(f, dict) and f.get("rule") == flag_name)
            or (isinstance(f, str) and f == flag_name)
            for f in flags_list
        )

    privacy_mask = df["input_flags"].apply(lambda x: _has_flag(x, "privacy_wall"))
    invalid_mask = df["input_flags"].apply(lambda x: _has_flag(x, "invalid_url"))

    filter_counts["privacy_wall_removed"] = int(privacy_mask.sum())
    filter_counts["invalid_url_removed"] = int(invalid_mask.sum())

    combined_mask = privacy_mask | invalid_mask
    n_removed = int(combined_mask.sum())

    if n_removed:
        logger.info(
            "Filtered %d flagged rows (privacy_wall=%d, invalid_url=%d)",
            n_removed,
            filter_counts["privacy_wall_removed"],
            filter_counts["invalid_url_removed"],
        )
        df = df[~combined_mask].copy()

    return df, filter_counts


def run_deduplicate(state: PipelineState, cfg: dict) -> None:
    """Filter flagged rows, then remove duplicates and group descriptions.

    Passes in order:
    0. Filter rows flagged privacy_wall or invalid_url (prevents garbage in groups).
    1. URL exact dedup (job_url identity).
    2. Composite dedup: lower(title_cleaned) + lower(company_name) + lower(location).
    3. Description grouping: SHA-256 hash of normalized description text.
    4. Near-duplicate grouping: MinHash LSH at 95% Jaccard similarity — catches
       postings that differ only by date, city, or minor formatting.

    Sets ``state.description_groups`` (for extract step) and
    ``state.dedup_report`` (for logging / debugging).

    Args:
        state: Mutable pipeline state — reads and modifies ``state.df``.
        cfg: Pipeline config dict (used for checkpoint db and output paths).
    """
    df = state.require_df("deduplicate")
    before = len(df)

    logger.info("=" * 70)
    logger.info("Step 3: Deduplicate (%d rows)", before)
    logger.info("=" * 70)

    cp = Checkpoint(cfg["paths"]["checkpoint_db"])

    # — Pass 0: Filter privacy-wall and invalid-URL rows
    df, filter_counts = _filter_flagged_rows(df, logger)

    # — Pass 1 + 2: URL dedup + composite key dedup (uses title_cleaned if present)
    df, dedup_report = deduplicate_rows(df, cp, cfg)

    # — Pass 3: Description grouping (SHA-256 exact match)
    groups = group_rows_by_description(df, cfg["paths"]["deduped_dir"])
    reps = sum(1 for g in groups.values() if g["count"] > 0)
    logger.info(
        "Description grouping: %d groups (%d representatives)", len(groups), reps
    )

    # — Pass 4: Near-duplicate grouping (MinHash LSH, 95% Jaccard)
    near_dup_map = find_near_duplicates(df, threshold=0.95, num_perm=128)
    if near_dup_map:
        # Build reverse lookup: row_id → group_hash for O(1) lookups
        row_to_group: dict[str, str] = {}
        for group_hash, group in groups.items():
            for member_id in group["member_row_ids"]:
                row_to_group[member_id] = group_hash

        merged = 0
        for dup_id, rep_id in near_dup_map.items():
            rep_group_hash = row_to_group.get(rep_id)
            dup_group_hash = row_to_group.get(dup_id)
            if rep_group_hash is None or dup_group_hash is None:
                continue
            if rep_group_hash == dup_group_hash:
                continue  # already in the same group

            # Remove dup from its original group
            src_group = groups[dup_group_hash]
            if dup_id in src_group["member_row_ids"]:
                src_group["member_row_ids"].remove(dup_id)
                src_group["count"] -= 1

            # Add dup to rep's group
            dst_group = groups[rep_group_hash]
            if dup_id not in dst_group["member_row_ids"]:
                dst_group["member_row_ids"].append(dup_id)
                dst_group["count"] += 1

            # Update reverse lookup
            row_to_group[dup_id] = rep_group_hash
            merged += 1

        # Clean up empty groups
        empty_hashes = [h for h, g in groups.items() if g["count"] == 0]
        for h in empty_hashes:
            del groups[h]

        logger.info(
            "Near-duplicate pass: %d row_ids merged into existing groups (%d empty groups removed)",
            merged,
            len(empty_hashes),
        )

    after = len(df)
    logger.info(
        "Dedup complete: %d → %d rows (%d removed)",
        before,
        after,
        before - after,
        extra={
            "event": "dedup_complete",
            "rows_before": before,
            "rows_after": after,
            "rows_removed": before - after,
        },
    )

    validate_step_output(df, dedup_output_schema, "deduplicate")
    state.df = df
    state.description_groups = groups
    dedup_report.update(filter_counts)
    state.dedup_report = dedup_report
