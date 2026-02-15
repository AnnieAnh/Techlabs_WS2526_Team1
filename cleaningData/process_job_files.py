"""
Job listings data processing pipeline.

Combines raw job data from Indeed and LinkedIn CSV files into a unified schema,
performs deduplication, normalizes dates, and outputs a cleaned dataset with
a processing log for auditability.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent

# Input/output paths
RAW_DATA_DIR = _SCRIPT_DIR / "RawData"
PROCESSED_DIR = _SCRIPT_DIR / "Processed"
INDEED_PATH = RAW_DATA_DIR / "Raw_Jobs_INDEED.csv"
LINKEDIN_1_PATH = RAW_DATA_DIR / "Raw_Jobs_LINKEDIN_1.csv"
LINKEDIN_2_PATH = RAW_DATA_DIR / "Raw_Jobs_LINKEDIN_2.csv"
OUTPUT_PATH = PROCESSED_DIR / "combined_jobs.csv"
PROCESSING_LOG_PATH = PROCESSED_DIR / "processing_log.json"

# Target schema columns (output)
TARGET_COLUMNS = [
    "title",
    "site",
    "job_url",
    "company_name",
    "location",
    "date_posted",
    "description",
]


@dataclass
class SourceConfig:
    """Configuration for a single data source (file + column mapping)."""

    name: str
    file_path: Path
    site_label: str
    column_mapping: dict[str, str]  # target_col -> source_col


# Source-specific column mappings (target schema key -> source column name)
INDEED_CONFIG = SourceConfig(
    name="Indeed",
    file_path=INDEED_PATH,
    site_label="indeed",
    column_mapping={
        "title": "title",
        "job_url": "job_url",
        "company_name": "company",
        "location": "location",
        "date_posted": "date_posted",
        "description": "description",
    },
)

LINKEDIN_1_CONFIG = SourceConfig(
    name="LinkedIn #1",
    file_path=LINKEDIN_1_PATH,
    site_label="linkedin",
    column_mapping={
        "title": "title",
        "job_url": "job_url",
        "company_name": "company",
        "location": "location",
        "date_posted": "posted_date",
        "description": "description",
    },
)

LINKEDIN_2_CONFIG = SourceConfig(
    name="LinkedIn #2",
    file_path=LINKEDIN_2_PATH,
    site_label="linkedin",
    column_mapping={
        "title": "title",
        "job_url": "link",
        "company_name": "company",
        "location": "location_job",
        "date_posted": "time",
        "description": "description",
    },
)

# Deduplication subset (columns used to identify duplicates)
DEDUP_SUBSET = ["job_url", "title", "company_name"]


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def _validate_input_files(*paths: Path) -> None:
    """Raise FileNotFoundError if any path does not exist or is not a file."""
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing input file(s): {[str(p) for p in missing]}"
        )


def normalize_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from all string columns."""
    string_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in string_cols:
        if df[col].dtype == "object":
            df[col] = df[col].str.strip()
    return df


def deduplicate(
    df: pd.DataFrame,
    subset: list[str],
    source_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Remove duplicate rows based on a subset of columns.

    Keeps the first occurrence of each duplicate. Returns the deduplicated
    DataFrame and a statistics dict for logging.

    Args:
        df: Input DataFrame.
        subset: Column names used to identify duplicates.
        source_name: Human-readable name for logging.

    Returns:
        Tuple of (deduplicated DataFrame, stats dict).
    """
    rows_before = len(df)
    df_deduped = df.drop_duplicates(subset=subset, keep="first")
    rows_after = len(df_deduped)
    removed = rows_before - rows_after
    removal_pct = (removed / rows_before * 100) if rows_before > 0 else 0.0

    stats = {
        "source": source_name,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "duplicates_removed": removed,
        "removal_percentage": round(removal_pct, 2),
    }

    logger.info(
        "[%s] Removed %d duplicates (%.1f%%) | %d → %d records",
        source_name,
        removed,
        removal_pct,
        rows_before,
        rows_after,
    )
    return df_deduped, stats


def ensure_string_type(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert specified columns to string, replacing 'nan'/'None' with NaN."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .replace("nan", np.nan)
                .replace("None", np.nan)
            )
    return df


def parse_date_to_exact(
    value: Any,
    reference_date: datetime | None = None,
) -> str | None:
    """
    Convert date_posted value to ISO date string (YYYY-MM-DD).

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
    if val in ("", "nan", "None"):
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
        dt = datetime.utcfromtimestamp(ts / divisor)
        return dt.strftime("%Y-%m-%d")

    # Relative strings
    val_lower = val.lower()
    if val_lower in ("today", "just now", "just posted"):
        return ref.strftime("%Y-%m-%d")
    if val_lower == "yesterday":
        return (ref - timedelta(days=1)).strftime("%Y-%m-%d")
    if re.match(r"^\d+\s*hours?\s+ago$", val_lower):
        return ref.strftime("%Y-%m-%d")

    # X day(s)/week(s)/month(s) ago (handles both singular and plural)
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
    """Convert date_posted column to ISO date strings (YYYY-MM-DD)."""
    if "date_posted" not in df.columns:
        return df
    df = df.copy()
    ref = reference_date or datetime.now()
    df["date_posted"] = df["date_posted"].apply(
        lambda x: parse_date_to_exact(x, ref)
    )
    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values with placeholders.

    - company_name: "na"
    - date_posted: oldest date in the dataset (min of non-null dates)
    """
    df = df.copy()

    if "company_name" in df.columns:
        df["company_name"] = df["company_name"].fillna("na")
        logger.info("Filled missing company_name with 'na'.")

    if "date_posted" in df.columns:
        valid_dates = df["date_posted"].dropna()
        if len(valid_dates) > 0:
            oldest_date = valid_dates.min()
            df["date_posted"] = df["date_posted"].fillna(oldest_date)
            logger.info(
                "Filled missing date_posted with oldest date: %s.",
                oldest_date,
            )
        else:
            logger.warning(
                "No valid dates in date_posted; missing values left as-is."
            )

    return df


def check_type_consistency(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    columns: list[str],
) -> None:
    """
    Verify that specified columns have matching dtypes in both DataFrames.

    Raises:
        ValueError: If any column has incompatible types.
    """
    mismatches = []
    for col in columns:
        if col in df1.columns and col in df2.columns:
            if df1[col].dtype != df2[col].dtype:
                mismatches.append(
                    f"  {col}: {df1[col].dtype} vs {df2[col].dtype}"
                )

    if mismatches:
        logger.error("Type mismatches found:\n%s", "\n".join(mismatches))
        raise ValueError("Fix type mismatches before merging.")

    logger.info("Data types consistent across files.")


def load_and_normalize_source(
    config: SourceConfig,
    processing_log: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Load a single source file, map to target schema, and deduplicate.

    Args:
        config: Source configuration (path + column mapping).
        processing_log: List to append deduplication stats to.

    Returns:
        Normalized DataFrame with TARGET_COLUMNS.
    """
    logger.info("=" * 70)
    logger.info("Processing: %s", config.name)
    logger.info("=" * 70)

    df = pd.read_csv(config.file_path)
    logger.info("Loaded %d records from %s", len(df), config.name)

    # Map to target schema
    normalized = pd.DataFrame(
        {
            target_col: df[config.column_mapping[target_col]]
            for target_col in TARGET_COLUMNS
            if target_col != "site"
        }
    )
    normalized["site"] = config.site_label

    # Reorder to match TARGET_COLUMNS
    normalized = normalized[TARGET_COLUMNS]

    normalized = normalize_whitespace(normalized)
    normalized, stats = deduplicate(
        normalized,
        subset=DEDUP_SUBSET,
        source_name=config.name,
    )
    stats["file"] = config.file_path.name
    processing_log.append(stats)

    return normalized


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------


def run_pipeline() -> None:
    """
    Execute the full job data processing pipeline.

    Loads Indeed and LinkedIn sources, merges, deduplicates, normalizes dates,
    performs quality checks, and writes output + processing log.
    """
    _validate_input_files(INDEED_PATH, LINKEDIN_1_PATH, LINKEDIN_2_PATH)

    processing_log: list[dict[str, Any]] = []
    run_timestamp = datetime.now().isoformat()

    # Step 1: Load and normalize Indeed
    indeed_df = load_and_normalize_source(INDEED_CONFIG, processing_log)

    # Step 2: Load and normalize LinkedIn #1
    linkedin1_df = load_and_normalize_source(LINKEDIN_1_CONFIG, processing_log)

    # Step 3: Load and normalize LinkedIn #2
    linkedin2_df = load_and_normalize_source(LINKEDIN_2_CONFIG, processing_log)

    # Step 4: Merge LinkedIn files
    logger.info("=" * 70)
    logger.info("Merging LinkedIn files")
    logger.info("=" * 70)

    string_columns = TARGET_COLUMNS
    linkedin1_df = ensure_string_type(linkedin1_df, string_columns)
    linkedin2_df = ensure_string_type(linkedin2_df, string_columns)
    check_type_consistency(linkedin1_df, linkedin2_df, string_columns)

    linkedin_combined = pd.concat(
        [linkedin1_df, linkedin2_df],
        ignore_index=True,
    )
    logger.info("Combined LinkedIn records: %d", len(linkedin_combined))

    linkedin_combined, combined_stats = deduplicate(
        linkedin_combined,
        subset=DEDUP_SUBSET,
        source_name="LinkedIn Combined",
    )
    combined_stats["file"] = "merged (LinkedIn #1 + LinkedIn #2)"
    processing_log.append(combined_stats)

    # Step 5: Merge with Indeed
    logger.info("=" * 70)
    logger.info("Merging LinkedIn + Indeed")
    logger.info("=" * 70)

    indeed_df = ensure_string_type(indeed_df, string_columns)
    linkedin_combined = ensure_string_type(linkedin_combined, string_columns)
    check_type_consistency(indeed_df, linkedin_combined, string_columns)

    final_df = pd.concat(
        [linkedin_combined, indeed_df],
        ignore_index=True,
    )
    logger.info("Final dataset: %d records", len(final_df))

    final_df = normalize_date_posted(final_df)
    logger.info("Normalized date_posted to exact dates.")

    # Fill missing values: company_name -> "na", date_posted -> oldest date
    final_df = fill_missing_values(final_df)

    # Step 6: Quality report
    logger.info("=" * 70)
    logger.info("Data quality report")
    logger.info("=" * 70)

    logger.info(
        "Source breakdown:\n%s",
        final_df["site"].value_counts().to_string(),
    )

    missing = final_df.isnull().sum()
    missing_pct = (missing / len(final_df) * 100).round(1)
    logger.info("Missing values per column:")
    for col in TARGET_COLUMNS:
        logger.info(
            "  %s: %d (%.1f%%)",
            col,
            missing[col],
            missing_pct[col],
        )

    logger.info(
        "Sample records:\n%s",
        final_df.head(3)[["title", "site", "company_name"]].to_string(
            index=False
        ),
    )

    # Step 7: Save output
    logger.info("=" * 70)
    logger.info("Saving output")
    logger.info("=" * 70)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)
    logger.info(
        "Saved combined jobs to: %s (final row count: %d)",
        OUTPUT_PATH,
        len(final_df),
    )

    total_removed = sum(s["duplicates_removed"] for s in processing_log)
    report = {
        "run_timestamp": run_timestamp,
        "processing_log": processing_log,
        "summary": {
            "total_deduplication_steps": len(processing_log),
            "total_duplicates_removed_across_all_steps": total_removed,
            "final_output_rows": len(final_df),
            "output_file": OUTPUT_PATH.name,
        },
    }

    with open(PROCESSING_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("Saved processing log to: %s", PROCESSING_LOG_PATH)


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline()
