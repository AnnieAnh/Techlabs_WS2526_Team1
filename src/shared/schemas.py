"""Pandera DataFrame schemas for pipeline step boundaries.

Schemas validate the DataFrame contract at each step boundary. When a step
breaks the contract, the next step fails immediately with a clear
"column X missing or invalid" error rather than producing silently wrong output.

Usage::

    from shared.schemas import validate_step_output, prepare_output_schema

    # At the end of a step:
    validate_step_output(state.df, prepare_output_schema, "prepare")

Available schemas (in pipeline order):
    ingestion_output_schema      — after ingestion (combined_jobs.csv)
    prepare_output_schema        — after prepare step
    dedup_output_schema          — after deduplicate step (same contract, fewer rows)
    regex_extract_output_schema  — after regex_extract step (adds regex_* columns)
    clean_enrich_output_schema   — after clean_enrich step
    export_output_schema         — after export step (strict final column set)

Standalone pipeline schemas (used by extraction/exporter.py and cleaning/pipeline.py):
    extraction_output_schema — enriched_combined.csv
    cleaning_output_schema   — cleaned_jobs.csv
"""

import logging

import pandas as pd
import pandera as pa

logger = logging.getLogger("pipeline.schemas")

# ---------------------------------------------------------------------------
# Step 1: Ingest output
# (columns produced by ingestion/pipeline.py → combined_jobs.csv)
# ---------------------------------------------------------------------------

ingestion_output_schema = pa.DataFrameSchema(
    {
        "title": pa.Column(pa.String, nullable=True),
        "company_name": pa.Column(pa.String, nullable=True),
        "job_url": pa.Column(
            pa.String,
            nullable=False,
            checks=pa.Check.str_startswith("http"),
        ),
        "location": pa.Column(pa.String, nullable=True),
        "date_posted": pa.Column(pa.String, nullable=True),
        "description": pa.Column(
            pa.String,
            nullable=False,
            checks=pa.Check.str_length(min_value=1),
        ),
        "site": pa.Column(
            pa.String,
            nullable=False,
            checks=pa.Check.isin(["linkedin", "indeed"]),
        ),
    },
    coerce=False,
    strict=False,  # extra columns allowed
)

# ---------------------------------------------------------------------------
# Step 2: Prepare output
# (adds row_id, title_cleaned, city, state, country columns)
# ---------------------------------------------------------------------------

prepare_output_schema = pa.DataFrameSchema(
    {
        "row_id": pa.Column(
            pa.String,
            nullable=False,
            unique=True,
        ),
        "title_cleaned": pa.Column(pa.String, nullable=True),
        "city": pa.Column(pa.String, nullable=True),
        "state": pa.Column(pa.String, nullable=True),
        "country": pa.Column(pa.String, nullable=True),
        "description": pa.Column(pa.String, nullable=False),
    },
    coerce=False,
    strict=False,
)

# ---------------------------------------------------------------------------
# Step 3: Deduplicate output
# (same contract as prepare, just fewer rows)
# ---------------------------------------------------------------------------

dedup_output_schema = prepare_output_schema

# ---------------------------------------------------------------------------
# Step 4: Regex Extract output
# (adds regex_* columns to the dedup output)
# ---------------------------------------------------------------------------

regex_extract_output_schema = pa.DataFrameSchema(
    {
        "row_id": pa.Column(
            pa.String,
            nullable=False,
            unique=True,
        ),
        "title_cleaned": pa.Column(pa.String, nullable=True),
        "description": pa.Column(pa.String, nullable=False),
        "regex_salary_min": pa.Column(nullable=True),
    },
    coerce=False,
    strict=False,
)

# ---------------------------------------------------------------------------
# Steps 5 & 6: Extract and Validate output
# These steps operate on state.extraction_results (list of dicts), not state.df.
# The DataFrame contract is unchanged from regex_extract; aliases enforce that
# no step silently mutates the df structure during extraction/validation.
# ---------------------------------------------------------------------------

extract_output_schema = regex_extract_output_schema
validate_output_schema = regex_extract_output_schema

# ---------------------------------------------------------------------------
# Step 7: Clean + Enrich output
# (adds enrichment columns from LLM + cleaning pipeline)
# ---------------------------------------------------------------------------

clean_enrich_output_schema = pa.DataFrameSchema(
    {
        "row_id": pa.Column(
            pa.String,
            nullable=False,
            unique=True,
        ),
        "job_family": pa.Column(pa.String, nullable=True),
        "technical_skills": pa.Column(pa.String, nullable=True),
        "benefit_categories": pa.Column(pa.String, nullable=True),
        "description_quality": pa.Column(pa.String, nullable=True),
        "salary_min": pa.Column(nullable=True),
        "salary_max": pa.Column(nullable=True),
        "validation_flags": pa.Column(pa.String, nullable=True),
    },
    coerce=False,
    strict=False,
)

# ---------------------------------------------------------------------------
# Step 8: Export output
# (final column set after reorder_columns(); unexpected columns dropped)
# ---------------------------------------------------------------------------

export_output_schema = pa.DataFrameSchema(
    {
        "row_id": pa.Column(
            pa.String,
            nullable=False,
            unique=True,
        ),
        "title": pa.Column(pa.String, nullable=True),
        "job_family": pa.Column(pa.String, nullable=True),
        "description": pa.Column(pa.String, nullable=False),
    },
    coerce=False,
    strict=False,
)

# ---------------------------------------------------------------------------
# Standalone pipeline schemas (used by extraction/exporter.py and cleaning/pipeline.py)
# ---------------------------------------------------------------------------

extraction_output_schema = pa.DataFrameSchema(
    {
        "row_id": pa.Column(pa.String, nullable=False),
        "title": pa.Column(pa.String, nullable=True),
        "company_name": pa.Column(pa.String, nullable=True),
        "job_url": pa.Column(pa.String, nullable=False),
        "job_family": pa.Column(pa.String, nullable=True),
        "technical_skills": pa.Column(pa.String, nullable=True),
    },
    coerce=False,
    strict=False,
)

cleaning_output_schema = pa.DataFrameSchema(
    {
        "row_id": pa.Column(pa.String, nullable=False),
        "title": pa.Column(pa.String, nullable=True),
        "company_name": pa.Column(pa.String, nullable=True),
        "job_url": pa.Column(pa.String, nullable=False),
        "city": pa.Column(pa.String, nullable=True),
        "job_family": pa.Column(pa.String, nullable=True),
        "technical_skills": pa.Column(pa.String, nullable=True),
        "validation_flags": pa.Column(pa.String, nullable=True),
        "description_quality": pa.Column(pa.String, nullable=True),
        "benefit_categories": pa.Column(pa.String, nullable=True),
    },
    coerce=False,
    strict=False,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def validate_step_output(
    df: pd.DataFrame,
    schema: pa.DataFrameSchema,
    step_name: str,
) -> pd.DataFrame:
    """Validate a DataFrame against a schema, logging errors clearly on failure.

    Args:
        df: DataFrame to validate.
        schema: Pandera DataFrameSchema to validate against.
        step_name: Name of the step (used in error messages).

    Returns:
        The validated DataFrame (may be a copy if pandera modifies it).

    Raises:
        pa.errors.SchemaErrors: If any constraint is violated (uses lazy validation).
    """
    try:
        return schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        logger.error(
            "Schema violation after step '%s':\n%s",
            step_name,
            exc.failure_cases.to_string() if hasattr(exc, "failure_cases") else str(exc),
        )
        raise
    except pa.errors.SchemaError as exc:
        logger.error("Schema violation after step '%s': %s", step_name, exc)
        raise
