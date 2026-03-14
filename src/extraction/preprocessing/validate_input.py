"""Input validation: flag quality issues in loaded rows, generate quality report."""

import json
import logging
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from extraction.checkpoint import Checkpoint
from shared.constants import MISSING_SENTINELS

logger = logging.getLogger("pipeline.validate_input")

# Substrings that identify privacy-wall / cookie-wall placeholder descriptions
_PRIVACY_WALL_MARKERS = [
    "datenschutz einstellungen",
    "cookie-einstellungen",
    "privacy settings",
    "bitte aktivieren sie javascript",
    "please enable javascript",
    "diese seite benötigt javascript",
]


def _is_privacy_wall(description: str) -> bool:
    """Return True if the description matches a known privacy/cookie wall pattern."""
    lower = description.lower()
    return any(marker in lower for marker in _PRIVACY_WALL_MARKERS)


def _parse_date(value: str) -> date | None:
    """Try multiple date formats and return a date object, or None on failure."""
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except (ValueError, TypeError):
            pass
    return None


def validate_input(
    df: pd.DataFrame,
    config: dict,
    checkpoint: Checkpoint,
) -> tuple[pd.DataFrame, dict]:
    """Validate each row against quality rules, flag issues, and update the checkpoint.

    Rules applied (each adds a flag string to 'input_flags'):
      - missing_title:       title is empty
      - missing_company:     company_name is empty
      - invalid_url:         job_url does not start with 'http'
      - short_description:   description shorter than min_description_length
      - privacy_wall:        description matches known cookie/privacy wall patterns
      - invalid_date:        date_posted is not parseable as a date
      - date_anomaly:        date_posted is parseable but before date_anomaly_cutoff

    Side effects:
      - Rows flagged as 'privacy_wall' are marked as skipped in the checkpoint.
      - Rows with 'date_anomaly' have their date_posted set to None.
      - A quality report JSON is written to config['paths']['reports_dir'].

    Args:
        df: Loaded DataFrame (output of load_all_csvs).
        config: Pipeline config dict.
        checkpoint: Checkpoint instance for state tracking.

    Returns:
        Tuple of (annotated DataFrame with 'input_flags' column, report dict).
    """
    logger.info("=== STAGE: Input Validation ===")
    start = time.monotonic()

    val_cfg = config["validation"]
    min_len: int = val_cfg["min_description_length"]
    anomaly_cutoff: date = _parse_date(str(val_cfg["date_anomaly_cutoff"]))  # type: ignore[arg-type, assignment]
    reports_dir: Path = config["paths"]["reports_dir"]

    flags_list: list[list[str]] = []

    for _, row in df.iterrows():
        flags: list[str] = []

        if not str(row.get("title", "")).strip():
            flags.append("missing_title")

        if not str(row.get("company_name", "")).strip():
            flags.append("missing_company")

        if not str(row.get("job_url", "")).startswith("http"):
            flags.append("invalid_url")

        desc = str(row.get("description", ""))
        if _is_privacy_wall(desc):
            flags.append("privacy_wall")
        elif len(desc) < min_len:
            flags.append("short_description")

        raw_date = str(row.get("date_posted", ""))
        parsed_date = _parse_date(raw_date)
        if parsed_date is None and raw_date not in MISSING_SENTINELS:
            flags.append("invalid_date")
        elif parsed_date is not None and anomaly_cutoff and parsed_date < anomaly_cutoff:
            flags.append("date_anomaly")

        flags_list.append(flags)

    df = df.copy()
    df["input_flags"] = flags_list

    privacy_mask = df["input_flags"].apply(lambda f: "privacy_wall" in f)
    anomaly_mask = df["input_flags"].apply(lambda f: "date_anomaly" in f)

    skipped_count = 0
    for row_id in df.loc[privacy_mask, "row_id"]:
        checkpoint.mark_skipped(row_id)
        skipped_count += 1

    if anomaly_mask.any():
        df.loc[anomaly_mask, "date_posted"] = None

    all_flags = [f for flags in flags_list for f in flags]
    flag_counts: dict[str, int] = {}
    for flag in set(all_flags):
        count = all_flags.count(flag)
        flag_counts[flag] = count
        log_fn = logger.warning if flag in ("privacy_wall", "short_description") else logger.info
        log_fn("  %d rows flagged: %s", count, flag)

    flagged_rows = df["input_flags"].apply(bool).sum()
    valid_rows = len(df) - flagged_rows
    logger.info(
        "Validation complete: %d valid, %d flagged, %d excluded (privacy_wall)",
        valid_rows,
        flagged_rows,
        skipped_count,
    )

    desc_lengths = df["description"].str.len()
    report = {
        "total_rows": len(df),
        "valid_rows": int(valid_rows),
        "flagged_counts": flag_counts,
        "excluded_privacy_wall": skipped_count,
        "description_length_stats": {
            "min": int(desc_lengths.min()),
            "max": int(desc_lengths.max()),
            "mean": round(float(desc_lengths.mean()), 1),
            "median": round(float(desc_lengths.median()), 1),
        },
        "company_count": int(df["company_name"].nunique()),
    }

    report_path = reports_dir / "input_quality.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Quality report saved to %s", report_path)

    elapsed = time.monotonic() - start
    logger.info("Stage Input Validation complete: %d rows, %.1fs", len(df), elapsed)
    return df, report
