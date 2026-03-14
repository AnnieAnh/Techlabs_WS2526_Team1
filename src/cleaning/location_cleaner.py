"""City name normalization for the cleaning pipeline.

City aliases are loaded lazily on first use to avoid import-time file I/O.
"""

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger("pipeline.location_cleaner")

# Lazy-loaded — populated on first call to normalize_city_names()
_CITY_ALIASES: dict[str, str] | None = None


def _load_city_aliases() -> dict[str, str]:
    """Load city alias map from cleaning/config/city_name_map.yaml."""
    path = Path(__file__).parent / "config" / "city_name_map.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["aliases"]


def normalize_city_names(df: pd.DataFrame) -> pd.DataFrame:
    """Map English/variant city names to German canonical forms.

    Frankfurt is state-aware: Hesse → Frankfurt am Main,
    Brandenburg → Frankfurt (Oder). All other entries with city="Frankfurt"
    default to Frankfurt am Main.

    City aliases are loaded from config on the first call (lazy loading).

    Args:
        df: DataFrame with 'city' (and optionally 'state') columns.

    Returns:
        DataFrame with city names normalized to German canonical forms.
    """
    global _CITY_ALIASES
    if _CITY_ALIASES is None:
        _CITY_ALIASES = _load_city_aliases()
        logger.debug("City alias map loaded: %d entries", len(_CITY_ALIASES))

    df = df.copy()

    if "city" not in df.columns:
        return df

    # General aliases (apply before Frankfurt special case)
    df["city"] = df["city"].replace(_CITY_ALIASES)

    # Frankfurt state-aware normalisation
    if "state" in df.columns:
        frankfurt_mask = df["city"] == "Frankfurt"
        df.loc[frankfurt_mask & (df["state"] == "Brandenburg"), "city"] = "Frankfurt (Oder)"
        df.loc[frankfurt_mask & (df["state"] != "Brandenburg"), "city"] = "Frankfurt am Main"
    else:
        df.loc[df["city"] == "Frankfurt", "city"] = "Frankfurt am Main"

    return df
