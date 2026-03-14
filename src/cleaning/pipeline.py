"""Cleaning pipeline orchestrator.

Chains all cleaning modules in the correct order to produce an analysis-ready CSV.

Entry points:
  clean(df) -> pd.DataFrame          — data quality fixes only
  enrich(df) -> pd.DataFrame         — feature engineering / derived columns
  clean_enriched(input, output) -> pd.DataFrame  — full pipeline: load → clean → enrich → save

Dual correction pipeline note:
  Several cleaning steps (categorical remapping, C++ inference fix, skill casing) duplicate
  corrections already applied by extraction/post_extraction.py before CSV export.
  These cleaning steps are SAFETY NETS — they ensure correctness even when an older enriched
  CSV is fed directly to cleaning without going through the extractor pipeline.
  Under normal operation the extractor corrections run first and the cleaning steps are
  redundant but harmless.
"""

import logging
from pathlib import Path

import pandas as pd

from cleaning.benefit_categorizer import benefit_category_set
from cleaning.categorical_remapper import (
    fix_company_name_na,
    fix_residual_gender_markers,
    normalize_company_casing,
    remap_categoricals,
)
from cleaning.location_cleaner import normalize_city_names
from cleaning.missing_values import (
    fix_numeric_columns,
    standardize_missing_values,
)
from cleaning.output_formatter import (
    _load_valid_job_families,
    assert_invariants,
    drop_columns_and_rows,
    reorder_columns,
    to_analysis_ready,
)
from cleaning.quality_flagger import (
    compute_skill_frequencies,
    flag_description_quality,
)
from cleaning.skill_normalizer import (
    fix_cpp_inference,
    normalize_skill_casing,
    re_verify_skills_post_clean,
)
from cleaning.soft_skill_normalizer import normalize_soft_skills
from cleaning.validation_fixer import fix_validation_flags
from shared.io import read_csv_safe, write_csv_safe
from shared.schemas import cleaning_output_schema

logger = logging.getLogger("pipeline.pipeline")


def clean(df: pd.DataFrame, standalone: bool = False) -> pd.DataFrame:
    """Apply data quality fixes — no new column creation.

    Covers: safety net country filter, missing value standardization, numeric
    normalization, city name normalization, categorical remapping, validation flag
    cleanup, company name fix, residual gender marker stripping, C++ inference fix,
    skill casing normalization, and company casing normalization.

    Args:
        df: Input DataFrame (enriched CSV, post-extraction).
        standalone: If True, run skill normalization (C++ fix, casing). When run
            via orchestrate.py, these are already done in the validate step so
            they are skipped (standalone=False) to avoid redundant work.

    Returns:
        Cleaned DataFrame.
    """
    # Safety net: keep only Germany rows
    if "country" in df.columns:
        n_before = len(df)
        df = df[df["country"] == "Germany"].copy()
        n_dropped = n_before - len(df)
        if n_dropped:
            logger.info("Dropped %d non-Germany rows (safety net filter)", n_dropped)

    # Standardize missing values
    df = standardize_missing_values(df)
    logger.info("Missing values standardized")

    # Numeric columns to int-or-None; salary outlier nulling
    df = fix_numeric_columns(df)
    logger.info("Numeric columns fixed")

    # City name normalization
    df = normalize_city_names(df)
    logger.info("City names normalized")

    # Remap categorical values
    df = remap_categoricals(df)
    logger.info("Categoricals remapped")

    # Convert validation_flags to JSON
    df = fix_validation_flags(df)
    logger.info("validation_flags converted to JSON")

    # Fix placeholder "na" in company_name
    df = fix_company_name_na(df)

    # Strip residual gender markers from title_cleaned
    df = fix_residual_gender_markers(df)

    # Fix hallucinated C++ inference and normalize skill casing.
    # Gated behind standalone=True — when run via orchestrate.py, these are
    # already done in the validate step (post_extraction.py).
    if standalone:
        df = df.apply(fix_cpp_inference, axis=1)
        logger.info("C++ inference corrected (standalone)")
        df = normalize_skill_casing(df)
        logger.info("Skill casing normalized (standalone)")
    else:
        logger.debug("Skipping C++ fix and skill casing (already done in validate step)")

    # Normalize company_name casing
    df = normalize_company_casing(df)

    return df


def enrich(
    df: pd.DataFrame,
    freq_path: Path | None = None,
    standalone: bool = False,
) -> pd.DataFrame:
    """Apply feature engineering / derived columns.

    Covers: benefit categorization, soft skill normalization, description quality
    flagging, post-clean skill re-verification, and skill frequency labeling.

    Args:
        df: Cleaned DataFrame (output of clean()).
        freq_path: Optional path to write skill_frequencies.csv. If None, skipped.
        standalone: If True, run skill re-verification. When run via orchestrate.py,
            skill verification was already done in the validate step.

    Returns:
        Enriched DataFrame with derived columns added.
    """
    # Derive benefit categories
    df["benefit_categories"] = df["benefits"].apply(benefit_category_set)
    logger.info("Benefit categories computed")

    # Normalize soft skills + add soft_skill_categories
    df = normalize_soft_skills(df)
    logger.info("Soft skills normalized; soft_skill_categories added")

    # Flag description quality
    df = flag_description_quality(df)
    logger.info("Description quality flagged")

    # Re-verify skills against description after C++ fix and skill casing.
    # Gated behind standalone=True — orchestrate.py already runs verification
    # in the validate step.
    if standalone:
        df = re_verify_skills_post_clean(df)
        logger.info("Post-clean skill verification complete (standalone)")
    else:
        logger.debug("Skipping post-clean skill verification (already done in validate step)")

    # Compute and optionally save skill frequency report
    freq = compute_skill_frequencies(df, "technical_skills")
    if freq_path is not None:
        write_csv_safe(pd.DataFrame(freq), freq_path)
        logger.info("Skill frequencies written to %s (%d skills)", freq_path, len(freq))

    return df


def clean_enriched(
    input_path: Path | str,
    output_path: Path | str,
) -> pd.DataFrame:
    """Read an enriched CSV, apply all cleaning steps, and write the result.

    Orchestrates: load → clean() → enrich() → validate → save.

    Args:
        input_path: Path to the enriched CSV produced by the pipeline exporter.
        output_path: Destination path for the cleaned CSV.

    Returns:
        The cleaned DataFrame (also written to output_path).
    """
    logger.info("=== STAGE: pipeline.clean_enriched ===")

    input_path = Path(input_path)
    output_path = Path(output_path)

    valid_families = _load_valid_job_families()

    df = read_csv_safe(input_path)
    logger.info("Loaded %d rows × %d columns from %s", len(df), len(df.columns), input_path)

    # Standalone mode: run all normalization steps as safety net
    df = clean(df, standalone=True)
    df = enrich(df, freq_path=output_path.parent / "skill_frequencies.csv", standalone=True)

    # Drop unneeded columns and parse-failure rows
    df = drop_columns_and_rows(df, valid_families)
    logger.info("Unused columns/rows dropped; %d rows remain", len(df))

    # Reorder columns
    df = reorder_columns(df)
    logger.info("Columns reordered (%d columns)", len(df.columns))

    # Save debug dump before asserting — preserves state on invariant failure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path = output_path.with_suffix(".debug.csv")
    write_csv_safe(df, debug_path)
    logger.info("Debug output saved to %s", debug_path)

    try:
        assert_invariants(df, valid_families)
    except AssertionError as e:
        logger.error("INVARIANT FAILED: %s — debug output at %s", e, debug_path)
        raise
    logger.info("All invariants passed")

    try:
        cleaning_output_schema.validate(df)
        logger.info("Schema validation passed (%d rows)", len(df))
    except Exception as exc:
        logger.warning("Schema validation failed: %s", exc)
        raise

    # Promote debug file to final output (full QA version)
    debug_path.rename(output_path)
    logger.info("Stage clean_enriched complete: %d rows written to %s", len(df), output_path)

    # Write lean analysis-ready version (no validation_flags, description_quality, description)
    analysis_df = to_analysis_ready(df)
    analysis_path = output_path.with_name("analysis_ready.csv")
    write_csv_safe(analysis_df, analysis_path)
    logger.info(
        "Analysis-ready output: %d rows × %d columns written to %s",
        len(analysis_df),
        len(analysis_df.columns),
        analysis_path,
    )

    return df


if __name__ == "__main__":
    import sys as _sys

    _default_input = "data/extraction/extracted/enriched_combined.csv"
    _input = _sys.argv[1] if len(_sys.argv) > 1 else _default_input
    _output = _sys.argv[2] if len(_sys.argv) > 2 else "data/cleaning/cleaned_jobs.csv"
    clean_enriched(_input, _output)
