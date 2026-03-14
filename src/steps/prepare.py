"""Step 2: Prepare — validate input, parse locations, normalize titles.

All operations are deterministic, rule-based transforms that prepare data
for the expensive steps. Grouping them means cheap operations (seconds)
run together, separately from extraction (minutes + API cost).

Regex field extraction lives in its own Step 4 (regex_extract) so it runs
on clean, filtered, deduped rows — fewer rows to process.

Title normalization runs BEFORE dedup (step 3) so the composite dedup key
uses normalized titles (e.g. ``"Senior Developer (m/w/d)"`` and
``"Senior Developer"`` match correctly).
"""

import logging

import numpy as np

from cleaning.missing_values import standardize_missing_values
from extraction.checkpoint import Checkpoint
from extraction.preprocessing.location_parser import (
    is_non_german,
    load_geo_config,
    parse_all_locations,
)
from extraction.preprocessing.title_normalizer import load_title_translations, normalize_all_titles
from extraction.preprocessing.validate_input import validate_input
from pipeline_state import PipelineState
from shared.schemas import prepare_output_schema, validate_step_output

logger = logging.getLogger("pipeline.prepare")


def run_prepare(state: PipelineState, cfg: dict) -> None:
    """Validate, geo-parse, and title-normalize every row.

    Steps (in order):
    1. Input validation — flags rows with missing fields, short descriptions,
       privacy walls, date anomalies.
    2. Location parsing — extracts city/state/country from the raw location
       string, then removes non-German rows.
    3. Title normalization — strips gender markers (m/w/d), translates German
       titles to English, writes ``title_cleaned`` column.
    4. Early NA standardization (first pass) — normalizes pre-LLM columns so
       later steps see consistent NA representations. A second pass runs in
       step 7 after LLM-extracted columns are merged in.

    Args:
        state: Mutable pipeline state — reads and modifies ``state.df``.
        cfg: Pipeline config dict (from ``src/extraction/config/settings.yaml``).
    """
    df = state.require_df("prepare")

    logger.info("=" * 70)
    logger.info("Step 2: Prepare (%d rows)", len(df))
    logger.info("=" * 70)

    cp = Checkpoint(cfg["paths"]["checkpoint_db"])

    # — 1. Input validation
    df, _val_report = validate_input(df, cfg, cp)
    n_flagged = df["input_flags"].apply(bool).sum()
    logger.info("Validation: %d rows flagged", n_flagged)

    # — 2. Location parsing + non-German filter
    geo_config = load_geo_config()
    df, loc_report = parse_all_locations(df, geo_config, cp, cfg["paths"]["reports_dir"])
    fallback_pct = loc_report["fallback_percent"]
    logger.info(
        "Location parsing: %d rows (%.1f%% fallback)",
        len(df),
        fallback_pct,
        extra={"event": "location_parse_done", "fallback_percent": fallback_pct},
    )
    if fallback_pct > 2.0:
        logger.warning("Fallback rate %.1f%% exceeds 2%% threshold", fallback_pct)

    _is_non_german_vec = np.vectorize(is_non_german)
    non_german = _is_non_german_vec(
        df["location"].to_numpy(dtype=str, na_value=""),
        df["state"].to_numpy(dtype=str, na_value=""),
        df["country"].to_numpy(dtype=str, na_value=""),
    )
    n_excluded = int(non_german.sum())
    if n_excluded:
        excluded_countries = df.loc[non_german, "country"].value_counts().to_dict()  # type: ignore[union-attr]
        logger.info("Excluding %d non-German rows: %s", n_excluded, excluded_countries)
        df = df[~non_german].copy()
    logger.info("Rows after country filter: %d", len(df))

    # — 3. Title normalization (runs BEFORE dedup so title_cleaned is used for key)
    translations = load_title_translations()
    df, title_report = normalize_all_titles(df, translations, cp, cfg["paths"]["reports_dir"])
    logger.info(
        "Title normalization: %d → %d unique titles",
        title_report["unique_titles_before"],
        title_report["unique_titles_after"],
        extra={"event": "title_normalize_done", **title_report},
    )

    # — 4. Early NA standardization (first pass — pre-LLM columns only).
    # Normalizes company_name, location, and other basic string columns so that
    # later steps see consistent NA representations. A second pass runs in step 7
    # after merge_results populates LLM-extracted columns.
    df = standardize_missing_values(df)
    logger.debug("Early NA standardization complete")

    validate_step_output(df, prepare_output_schema, "prepare")
    state.df = df
