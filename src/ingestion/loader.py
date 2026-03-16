"""Source file loader and normalizer for ingestion pipeline."""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

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


def normalize_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from all string columns.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with whitespace stripped from string columns.
    """
    df = df.copy()
    string_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in string_cols:
        df[col] = df[col].str.strip()
    return df


def ensure_string_type(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert specified columns to string, replacing 'nan'/'None' with NaN.

    Args:
        df: Input DataFrame.
        columns: Column names to coerce.

    Returns:
        DataFrame with specified columns coerced to string,
        with 'nan'/'None' artifacts replaced by NaN.
    """
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


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing company_name values with None (NA boundary is at CSV I/O).

    Args:
        df: DataFrame with optional company_name column.

    Returns:
        DataFrame with company_name NaN → None.
    """
    df = df.copy()
    if "company_name" in df.columns:
        df["company_name"] = df["company_name"].where(
            df["company_name"].notna(), other=None,  # type: ignore[call-overload]
        )
    return df


def load_and_normalize_source(config: SourceConfig) -> pd.DataFrame:
    """Load a single source file and map to target schema.

    Args:
        config: Source configuration (path + column mapping).

    Returns:
        Normalized DataFrame with TARGET_COLUMNS.
    """
    logger.info("=" * 70)
    logger.info("Processing: %s", config.name)
    logger.info("=" * 70)

    df = pd.read_csv(
        config.file_path, encoding="utf-8", keep_default_na=False, na_values=[]
    )
    logger.info("Loaded %d records from %s", len(df), config.name)

    normalized = pd.DataFrame(
        {
            target_col: df[config.column_mapping[target_col]]
            for target_col in TARGET_COLUMNS
            if target_col != "site"
        }
    )
    normalized["site"] = config.site_label
    normalized = normalized[TARGET_COLUMNS]
    normalized = normalize_whitespace(normalized)
    return normalized
