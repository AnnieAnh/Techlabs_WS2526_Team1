"""Shared I/O utilities for the monorepo.

Provides safe CSV reading with the project-standard NA handling applied
consistently across all components (ingestion, extraction, cleaning).
"""

from pathlib import Path
from typing import Any

import pandas as pd

from shared.constants import MISSING_SENTINELS


def read_csv_safe(
    path: Path | str,
    fallback_encoding: str | None = "latin-1",
    **kwargs,
) -> pd.DataFrame:
    """Read a CSV with safe NA handling and optional encoding fallback.

    Disables pandas' default NA inference (keep_default_na=False), then explicitly
    converts MISSING_SENTINELS strings (including 'NA') to Python None.
    Tries UTF-8 first; falls back to ``fallback_encoding`` on UnicodeDecodeError.

    Args:
        path: Path to the CSV file.
        fallback_encoding: Encoding to try if UTF-8 decoding fails. Set to
            ``None`` to disable fallback (raises UnicodeDecodeError instead).
        **kwargs: Additional keyword arguments forwarded to pd.read_csv.

    Returns:
        DataFrame with all MISSING_SENTINELS values converted to None.
    """
    defaults: dict[str, Any] = dict(encoding="utf-8", keep_default_na=False, na_values=[])
    defaults.update(kwargs)
    try:
        df = pd.read_csv(path, **defaults)
    except UnicodeDecodeError:
        if fallback_encoding is None:
            raise
        fallback_defaults: dict[str, Any] = dict(defaults)
        fallback_defaults["encoding"] = fallback_encoding
        df = pd.read_csv(path, **fallback_defaults)
    # Replace sentinel strings ("NA", "nan", "None", …) with Python None.
    # df.replace("NA", None) produces float NaN in some pandas versions because
    # pandas treats None as NaN internally. Per-column apply guarantees object
    # columns store Python None, not float('nan'), so downstream .to_dict("records")
    # returns None (falsy) instead of NaN (truthy), keeping (val or "") patterns safe.
    for col in df.columns:
        df[col] = df[col].apply(
            lambda v: None if isinstance(v, str) and v in MISSING_SENTINELS else v
        )
    return df


def write_csv_safe(df: pd.DataFrame, path: Path | str, **kwargs) -> None:
    """Write a DataFrame to CSV, converting None/NaN back to 'NA' string.

    Ensures consistent NA representation in output CSVs regardless of whether
    values were stored as Python None or pandas NaN internally.

    Args:
        df: DataFrame to write.
        path: Destination path for the CSV file.
        **kwargs: Additional keyword arguments forwarded to DataFrame.to_csv.
    """
    defaults: dict[str, Any] = dict(index=False, encoding="utf-8", na_rep="NA")
    defaults.update(kwargs)
    df.to_csv(path, **defaults)
