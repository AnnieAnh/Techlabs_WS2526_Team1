"""Final enriched CSV export — merge all pipeline outputs into one DataFrame."""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from shared.io import write_csv_safe
from shared.schemas import extraction_output_schema

logger = logging.getLogger("pipeline.exporter")

# Preferred column output order
_ORIGINAL_COLS = [
    "row_id", "source_file", "title", "site", "job_url",
    "company_name", "location", "date_posted",
]
_PARSED_COLS = ["city", "state", "country"]
_NORMALIZED_COLS = ["title_cleaned", "job_family", "seniority_from_title"]
# Tier 1: sourced from regex_extractor.py (regex_* prefix dropped at export time)
_TIER1_COLS = [
    "contract_type", "work_modality",
    "salary_min", "salary_max",
    "experience_years",
    "seniority_from_title",
    "languages",
    "education_level",
]
# Tier 2: sourced from LLM extraction
_TIER2_COLS = [
    "technical_skills", "soft_skills", "nice_to_have_skills", "benefits", "tasks",
    "job_family", "job_summary",
]
# Evidence columns: full evidence objects with source quotes (prompt v2.0)
_EVIDENCE_COLS = [
    "technical_skills_evidence", "nice_to_have_skills_evidence",
    "benefits_evidence", "tasks_evidence",
]
_EXTRACTED_COLS = _TIER1_COLS + _TIER2_COLS
_META_COLS = ["validation_flags", "input_flags", "description"]


def merge_results(
    df: pd.DataFrame,
    results: list[dict[str, Any]],
) -> pd.DataFrame:
    """Merge extraction results and validation flags into the main DataFrame.

    Args:
        df: Main DataFrame with original + parsed + normalized columns and row_id.
        results: List of extraction result dicts with 'row_id', 'data',
                 and optionally 'validation_flags'.

    Returns:
        Enriched DataFrame with all extraction fields added as columns.
    """
    logger.info("=== Merging extraction results into DataFrame (%d rows) ===", len(df))

    # Promote regex_* Tier 1 columns to canonical names (drop regex_ prefix)
    _TIER1_CANONICAL = [
        "contract_type", "work_modality", "salary_min", "salary_max",
        "experience_years", "seniority_from_title", "languages", "education_level",
    ]
    for col in _TIER1_CANONICAL:
        regex_col = f"regex_{col}"
        if regex_col in df.columns and col not in df.columns:
            df = df.rename(columns={regex_col: col})
        elif regex_col in df.columns:
            # Canonical column already exists — drop the regex_ duplicate
            df = df.drop(columns=[regex_col])

    # Drop any remaining regex_* columns not in the canonical list
    leftover_regex = [c for c in df.columns if c.startswith("regex_")]
    if leftover_regex:
        df = df.drop(columns=leftover_regex)

    extracted: dict[str, dict] = {}
    for r in results:
        row_id = r.get("row_id")
        if not row_id:
            continue
        data = r.get("data") or {}
        flags = r.get("validation_flags") or []
        extracted[row_id] = {**data, "validation_flags": flags}

    if extracted:
        ext_df = pd.DataFrame.from_dict(extracted, orient="index")
        ext_df.index.name = "row_id"
        ext_df = ext_df.reset_index()
        enriched = df.merge(ext_df, on="row_id", how="left")
    else:
        enriched = df.copy()
        for col in _TIER2_COLS + ["validation_flags"]:
            if col not in enriched.columns:
                enriched[col] = None

    matched = enriched["row_id"].isin(extracted).sum()
    logger.info(
        "Merged %d/%d rows with extraction results (%d unmatched)",
        matched,
        len(enriched),
        len(enriched) - matched,
    )
    return enriched


def _column_order(df: pd.DataFrame) -> list[str]:
    """Return columns in preferred output order, with any extras appended."""
    ordered: list[str] = []
    all_groups = (
        _ORIGINAL_COLS, _PARSED_COLS, _NORMALIZED_COLS,
        _EXTRACTED_COLS, _EVIDENCE_COLS, _META_COLS,
    )
    for col_group in all_groups:
        for col in col_group:
            if col in df.columns and col not in ordered:
                ordered.append(col)
    for col in df.columns:
        if col not in ordered:
            ordered.append(col)
    return ordered


def export_enriched_csv(
    df: pd.DataFrame,
    cfg: dict,
) -> dict[str, Path]:
    """Export enriched DataFrame as per-source-file CSVs and a combined CSV.

    Args:
        df: Enriched DataFrame (output of merge_results()).
        cfg: Pipeline config (reads cfg["paths"]["extracted_dir"]).

    Returns:
        Dict mapping source file name (or "combined") to the exported Path.
    """
    extracted_dir = Path(cfg.get("paths", {}).get("extracted_dir", "data/extracted"))
    extracted_dir.mkdir(parents=True, exist_ok=True)

    # Fill categorical empty strings with None (write_csv_safe handles None → "NA")
    categorical_cols = ["contract_type", "work_modality", "education_level"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].replace("", None)

    # Serialise list columns as JSON strings (not Python repr)
    array_cols = [
        "technical_skills", "soft_skills", "nice_to_have_skills", "benefits", "tasks",
        "languages",
        # Evidence columns preserve source quotes from prompt v2.0
        "technical_skills_evidence", "nice_to_have_skills_evidence",
        "benefits_evidence", "tasks_evidence",
    ]
    for col in array_cols:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else x
            )

    col_order = _column_order(df)
    df_ordered = df[col_order]

    exported: dict[str, Path] = {}

    if "source_file" in df_ordered.columns:
        for source_file, group in df_ordered.groupby("source_file"):
            out_path = extracted_dir / f"enriched_{source_file}"
            write_csv_safe(group, out_path)
            size_mb = out_path.stat().st_size / 1_048_576
            logger.info(
                "Exported %s: %d rows, %d cols → %s (%.1f MB)",
                source_file,
                len(group),
                len(group.columns),
                out_path,
                size_mb,
            )
            exported[str(source_file)] = out_path

    combined_path = extracted_dir / "enriched_combined.csv"
    try:
        extraction_output_schema.validate(df_ordered)
        logger.info("Schema validation passed (%d rows)", len(df_ordered))
    except Exception as exc:
        logger.warning("Schema validation failed: %s", exc)
        raise
    write_csv_safe(df_ordered, combined_path)
    size_mb = combined_path.stat().st_size / 1_048_576
    logger.info(
        "Exported combined: %d rows, %d cols → %s (%.1f MB)",
        len(df_ordered),
        len(df_ordered.columns),
        combined_path,
        size_mb,
    )
    exported["combined"] = combined_path

    return exported
