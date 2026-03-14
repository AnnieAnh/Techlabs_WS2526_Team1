"""Step 4: Regex Extract — deterministic regex field extraction on clean, deduped rows.

Runs after dedup (step 3) so regex extraction processes only filtered, deduped rows.

Requires ``title_cleaned``, ``description``, and ``row_id`` columns from step 2.
Adds 8 ``regex_*`` columns to the DataFrame.
"""

import logging

from extraction.preprocessing.regex_extractor import extract_regex_fields
from pipeline_state import PipelineState
from shared.schemas import regex_extract_output_schema, validate_step_output

logger = logging.getLogger("pipeline.regex_extract")

_REGEX_FIELDS = (
    "contract_type",
    "work_modality",
    "salary_min",
    "salary_max",
    "experience_years",
    "seniority_from_title",
    "languages",
    "education_level",
)


def run_regex_extract(state: PipelineState, cfg: dict) -> None:
    """Run deterministic regex field extraction on every row.

    Extracts 8 fields from ``description`` and ``title_cleaned`` using
    pattern-based rules. Each field is stored as ``regex_<field_name>``.

    Args:
        state: Mutable pipeline state — reads and modifies ``state.df``.
        cfg: Pipeline config dict (unused but required by step interface).
    """
    df = state.require_df("regex_extract")

    logger.info("=" * 70)
    logger.info("Step 4: Regex Extract (%d rows)", len(df))
    logger.info("=" * 70)

    # Schema guard: these columns must exist from step 2
    required = {"title_cleaned", "description", "row_id"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Step 4 (regex_extract) requires columns: {missing}")

    rows = df.to_dict("records")
    regex_rows = [
        extract_regex_fields(
            str(r.get("description", "")),
            str(r.get("title_cleaned") or r.get("title", "")),
        )
        for r in rows
    ]

    for key in _REGEX_FIELDS:
        df[f"regex_{key}"] = [r[key] for r in regex_rows]

    logger.info("Regex pre-extraction: %d rows tagged with %d fields", len(df), len(_REGEX_FIELDS))

    validate_step_output(df, regex_extract_output_schema, "regex_extract")
    state.df = df
