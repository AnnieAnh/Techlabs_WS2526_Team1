"""Step 7: Clean + Enrich — merge extraction results, fix data, add enrichment columns.

Cleaning and enrichment share a data dependency and run sequentially; splitting
them into micro-steps would add orchestration complexity with no benefit.
"""

import json
import logging
from pathlib import Path

from cleaning.benefit_categorizer import benefit_category_set
from cleaning.categorical_remapper import (
    fix_company_name_na,
    fix_residual_gender_markers,
    normalize_company_casing,
    remap_categoricals,
)
from cleaning.location_cleaner import normalize_city_names
from cleaning.missing_values import fix_numeric_columns, standardize_missing_values
from cleaning.quality_flagger import flag_description_quality
from cleaning.soft_skill_normalizer import normalize_soft_skills
from cleaning.validation_fixer import fix_validation_flags
from extraction.exporter import merge_results
from pipeline_state import PipelineState
from shared.io import write_csv_safe
from shared.schemas import clean_enrich_output_schema, validate_step_output

logger = logging.getLogger("pipeline.clean_enrich")


def run_clean_enrich(state: PipelineState, cfg: dict) -> None:
    """Merge extraction results into df, clean data, and add enrichment columns.

    Operations in order:
    1. merge_results — merge LLM extraction data into df (promotes regex_* columns)
    2. standardize_missing_values, fix_numeric_columns — NA normalization, float cast
       (second pass: handles LLM columns that don't exist before merge_results;
       first pass ran at end of step 2 for pre-LLM columns)
    3. normalize_city_names — apply city alias map (e.g. Munich → München)
    4. remap_categoricals, fix_company_name_na, fix_residual_gender_markers,
       normalize_company_casing — categorical cleaning
    5. fix_validation_flags — resolve/relabel validation flags
    6. benefit_category_set, normalize_soft_skills — enrichment columns
    7. flag_description_quality — tags descriptions as 'concatenated'
       (HTML-strip artefacts) or 'clean'

    Args:
        state: Mutable pipeline state — reads df + extraction_results, modifies df.
        cfg: Pipeline config dict.
    """
    # Note: fix_cpp_inference and normalize_skill_casing are deliberately
    # omitted here — they run in the validate step (step 6).
    # re_verify_skills_post_clean is standalone-only (clean_enriched()).
    df = state.require_df("clean_enrich")

    logger.info("=" * 70)
    logger.info("Step 7: Clean + Enrich (%d rows)", len(df))
    logger.info("=" * 70)

    # Load extraction results from state or disk
    results = state.extraction_results
    if state.no_llm:
        results = []
        logger.info("--no-llm: skipping extraction results, LLM columns will be None")
    elif results is None:
        results_path = Path(cfg["paths"]["extracted_dir"]) / "extraction_results.json"
        if not results_path.exists():
            logger.warning(
                "No extraction_results.json — merging without LLM data"
            )
            results = []
        else:
            with open(results_path, encoding="utf-8") as f:
                results = json.load(f)

    # — 1. Merge extraction results into df
    df = merge_results(df, results)

    # — 2. Standardize missing values + fix numeric columns
    df = standardize_missing_values(df)
    df = fix_numeric_columns(df)

    # — 3. Normalize city names
    df = normalize_city_names(df)

    # — 4. Categorical remapping + company name fixes
    df = remap_categoricals(df)
    df = fix_company_name_na(df)
    df = fix_residual_gender_markers(df)
    df = normalize_company_casing(df)

    # — 5. Fix validation flags
    df = fix_validation_flags(df)

    # — 6. Enrichment columns
    if "benefits" in df.columns:
        df["benefit_categories"] = df["benefits"].apply(benefit_category_set)
    df = normalize_soft_skills(df)

    # — 7. Description quality flags
    df = flag_description_quality(df)

    validate_step_output(df, clean_enrich_output_schema, "clean_enrich")

    # Write intermediate artifact so export can resume independently
    enriched_path = Path(cfg["paths"]["extracted_dir"]) / "enriched_cleaned.csv"
    enriched_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv_safe(df, enriched_path)

    rows = len(df)
    cols = len(df.columns)
    logger.info(
        "Clean + Enrich complete: %d rows, %d columns → %s",
        rows,
        cols,
        enriched_path,
        extra={"event": "clean_enrich_done", "rows": rows, "columns": cols},
    )

    state.df = df
