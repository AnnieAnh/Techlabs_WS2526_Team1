"""Cross-file deduplication: URL exact match, then title+company+location match."""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from extraction.checkpoint import Checkpoint
from shared.io import write_csv_safe

logger = logging.getLogger("pipeline.dedup")


def _company_casing_score(name: object) -> int:
    """Score company name by proper casing quality (higher = better).

    Prefers names with uppercase letters at word starts and mixed case
    over all-lowercase or ALL-UPPERCASE variants.

    Args:
        name: Company name string (or non-string, returning 0).

    Returns:
        Integer score — higher means better casing.
    """
    if not isinstance(name, str) or not name.strip():
        return 0
    words = name.split()
    score = 0
    for word in words:
        if not word:
            continue
        if word[0].isupper():
            score += 2
        if not word.isupper() and not word.islower():
            score += 1
        if word.islower():
            score -= 1
    return score


def deduplicate_rows(
    df: pd.DataFrame,
    checkpoint: Checkpoint,
    config: dict,
) -> tuple[pd.DataFrame, dict]:
    """Remove duplicate job postings in two passes.

    Pass 1: Drop rows with identical job_url (keep first occurrence).
    Pass 2: Drop rows with identical lower(title)+lower(company)+lower(location)
            on different URLs (keep first occurrence).

    Important: rows with same title+company but DIFFERENT location are kept — they
    represent the same job posted to multiple cities and are legitimate distinct rows
    for location analysis.

    Args:
        df: Validated DataFrame (output of validate_input).
        checkpoint: Checkpoint instance for state tracking.
        config: Pipeline config dict (used for output directory).

    Returns:
        Tuple of (deduped DataFrame, report dict).
    """
    logger.info("=== STAGE: Cross-File Deduplication ===")
    start = time.monotonic()

    before = len(df)
    deduped_dir: Path = config["paths"]["deduped_dir"]
    reports_dir: Path = config["paths"]["reports_dir"]

    url_dupes_mask = df.duplicated(subset=["job_url"], keep="first")
    url_dupe_ids = df.loc[url_dupes_mask, "row_id"].tolist()
    df = df[~url_dupes_mask].copy()
    pass1_removed = len(url_dupe_ids)

    for row_id in url_dupe_ids:
        checkpoint.mark_skipped(row_id)

    logger.info("Pass 1 (URL dedup): removed %d exact URL duplicates", pass1_removed)

    title_col = "title_cleaned" if "title_cleaned" in df.columns else "title"
    df["_title_lower"] = df[title_col].str.lower().str.strip()
    df["_company_lower"] = df["company_name"].str.lower().str.strip()
    df["_location_lower"] = df["location"].str.lower().str.strip()

    # Sort by casing score (descending) so duplicated(keep="first") keeps the
    # properly-cased company name variant rather than an arbitrary first-seen row.
    df["_casing_score"] = df["company_name"].apply(_company_casing_score)
    df = df.sort_values("_casing_score", ascending=False, kind="stable").copy()

    composite_key = ["_title_lower", "_company_lower", "_location_lower"]
    composite_dupes_mask = df.duplicated(subset=composite_key, keep="first")
    composite_dupe_ids = df.loc[composite_dupes_mask, "row_id"].tolist()
    df = df[~composite_dupes_mask].copy()

    for row_id in composite_dupe_ids:
        checkpoint.mark_skipped(row_id)

    pass2_removed = len(composite_dupe_ids)
    logger.info("Pass 2 (title+company+location dedup): removed %d additional", pass2_removed)

    df = df.drop(columns=["_title_lower", "_company_lower", "_location_lower", "_casing_score"])

    after = len(df)
    total_removed = before - after
    pct = (total_removed / before * 100) if before > 0 else 0.0

    logger.info(
        "Dedup complete: %d -> %d (%d removed, %.1f%% reduction)",
        before,
        after,
        total_removed,
        pct,
    )

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = deduped_dir / f"deduped_{timestamp}.csv"
    write_csv_safe(df, out_path)
    logger.info("Deduped data saved to %s", out_path)

    report = {
        "before": before,
        "after": after,
        "removed": total_removed,
        "removal_percent": round(pct, 2),
        "pass_1_removed": pass1_removed,
        "pass_2_removed": pass2_removed,
    }

    report_path = reports_dir / "dedup_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Dedup report saved to %s", report_path)

    elapsed = time.monotonic() - start
    logger.info("Stage Cross-File Deduplication complete: %d rows, %.1fs", after, elapsed)
    return df, report
