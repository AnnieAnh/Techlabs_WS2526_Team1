"""Categorical field remapping and company/title cleanup for the cleaning pipeline."""

import logging
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

from shared.constants import MISSING_SENTINELS

logger = logging.getLogger("pipeline.categorical_remapper")


def _load_config(filename: str) -> dict:
    """Load a YAML config file from cleaning/config/."""
    path = Path(__file__).parent / "config" / filename
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


_JOB_FAMILY_REMAP_CFG = _load_config("job_family_remap.yaml")
_JOB_FAMILY_REMAP: dict[str, str] = _JOB_FAMILY_REMAP_CFG["remap"]
_CONTRACT_TYPE_REMAP: dict[str, str] = _JOB_FAMILY_REMAP_CFG["contract_type_remap"]
_SENIORITY_REMAP: dict[str, str] = _JOB_FAMILY_REMAP_CFG["seniority_remap"]

def _load_valid_job_families() -> frozenset[str]:
    path = Path(__file__).parent.parent / "extraction" / "config" / "job_families.yaml"
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return frozenset(data.get("families", []) + ["Other"])


_VALID_JOB_FAMILIES: frozenset[str] = _load_valid_job_families()

# Residual gender marker pattern for title_cleaned
_RESIDUAL_GENDER_RE = re.compile(
    r"\s*\((?:gn|m,w,d|m,f,d|all\s+genders?)\)\s*",
    re.IGNORECASE,
)


def remap_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Remap out-of-spec categorical values to canonical forms.

    Args:
        df: DataFrame with job_family, contract_type, seniority_from_title columns.

    Returns:
        DataFrame with remapped categorical values.
    """
    df = df.copy()

    if "job_family" in df.columns:
        df["job_family"] = df["job_family"].replace(_JOB_FAMILY_REMAP)
        # Fall back to "Other" for any value still not in the canonical set
        unknown_mask = df["job_family"].notna() & ~df["job_family"].isin(_VALID_JOB_FAMILIES)
        unknown_vals = sorted(set(df.loc[unknown_mask, "job_family"].dropna()))
        if unknown_vals:
            logger.warning(
                "Auto-mapped %d unknown job_family value(s) to 'Other': %s",
                len(unknown_vals),
                unknown_vals,
            )
            df.loc[unknown_mask, "job_family"] = "Other"

    if "contract_type" in df.columns:
        df["contract_type"] = df["contract_type"].replace(_CONTRACT_TYPE_REMAP)

    if "seniority_from_title" in df.columns:
        df["seniority_from_title"] = df["seniority_from_title"].replace(_SENIORITY_REMAP)

    return df


def fix_company_name_na(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize placeholder 'na'/'NA' company_name values to None.

    Args:
        df: DataFrame with company_name column.

    Returns:
        DataFrame with normalized company_name values.
    """
    if "company_name" not in df.columns:
        return df
    df = df.copy()
    mask = df["company_name"].str.strip().isin(MISSING_SENTINELS)
    n_fixed = int(mask.sum())
    df.loc[mask, "company_name"] = None
    if n_fixed:
        logger.info("Normalized %d 'na'/'NA' company_name values to None", n_fixed)
    return df


def fix_residual_gender_markers(df: pd.DataFrame) -> pd.DataFrame:
    """Strip residual gender markers (gn, m,w,d, m,f,d, all genders) from title_cleaned.

    Args:
        df: DataFrame with title_cleaned column.

    Returns:
        DataFrame with gender markers removed from title_cleaned.
    """
    if "title_cleaned" not in df.columns:
        return df
    df = df.copy()

    def _strip(title: object) -> str:
        cleaned = _RESIDUAL_GENDER_RE.sub(" ", str(title))
        return re.sub(r"\s+", " ", cleaned).strip()

    before = df["title_cleaned"].copy()
    df["title_cleaned"] = df["title_cleaned"].apply(_strip)
    n_fixed = int((df["title_cleaned"] != before).sum())
    if n_fixed:
        logger.info("Stripped residual gender markers from %d title_cleaned values", n_fixed)
    return df


def normalize_company_casing(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize company_name casing — most-frequent capitalisation variant wins.

    Args:
        df: DataFrame with company_name column.

    Returns:
        DataFrame with company_name casing normalized.
    """
    if "company_name" not in df.columns:
        return df
    df = df.copy()

    company_counter: Counter[str] = Counter(df["company_name"].dropna().astype(str))

    canonical_map: dict[str, str] = {}
    for name in (n for n, _ in company_counter.most_common()):
        key = str(name).lower()
        if key not in canonical_map:
            canonical_map[key] = name

    replace_map = {
        name: canonical_map[str(name).lower()]
        for name in company_counter
        if canonical_map.get(str(name).lower()) != name
    }
    if replace_map:
        df["company_name"] = df["company_name"].replace(replace_map)
        logger.info("%d company casing variants normalized", len(replace_map))
    return df
