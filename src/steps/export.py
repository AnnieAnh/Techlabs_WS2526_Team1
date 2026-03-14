"""Step 8: Export — column ordering, invariant checks, CSV output, cost report.

Serialization and output formatting are output concerns, not data transformation.
Isolating export means: re-run after changing column order without re-running cleaning;
invariant failures don't require re-running the entire pipeline.

Saves a ``.debug.csv`` before running invariant checks. On success, renamed to final.
On invariant failure, the debug CSV remains for forensic inspection.
"""

import logging
from pathlib import Path

from cleaning.output_formatter import (
    _load_valid_job_families,
    assert_invariants,
    drop_columns_and_rows,
    reorder_columns,
)
from extraction.reporting.cost import build_cost_report, read_batch_token_usage, save_cost_report
from pipeline_state import PipelineState
from shared.io import write_csv_safe
from shared.schemas import export_output_schema, validate_step_output

logger = logging.getLogger("pipeline.export")


def run_export(state: PipelineState, cfg: dict) -> None:
    """Column-order, invariant-check, and write the final cleaned CSV.

    Saves ``<output>.debug.csv`` before assertions; on success renames to
    the final output path configured in ``cfg["export"]["output_path"]`` (or
    falls back to ``data/cleaning/cleaned_jobs.csv``).

    Also writes a cost report.

    Args:
        state: Mutable pipeline state — reads ``state.df``.
        cfg: Pipeline config dict.
    """
    df = state.require_df("export")

    logger.info("=" * 70)
    logger.info("Step 8: Export (%d rows, %d columns)", len(df), len(df.columns))
    logger.info("=" * 70)

    # Resolve output path from config or use default
    output_path = Path(
        cfg.get("export", {}).get("output_path", "data/cleaning/cleaned_jobs.csv")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path = output_path.with_suffix(".debug.csv")

    # — 1. Drop unneeded columns and parse-failure rows
    valid_families = _load_valid_job_families()
    df = drop_columns_and_rows(df, valid_families)

    # — 2. Reorder to canonical column order
    df = reorder_columns(df)

    # — 3. Write debug CSV before assertions
    write_csv_safe(df, debug_path)
    logger.info("Debug CSV written to %s", debug_path)

    # — 4. Assert invariants (fail loudly, leave debug CSV for forensics)
    try:
        assert_invariants(df, valid_families)
    except AssertionError as exc:
        logger.error(
            "Invariant check FAILED — debug CSV preserved at %s\n%s",
            debug_path,
            exc,
        )
        raise

    # — 5. Validate final schema, then rename debug to final output
    validate_step_output(df, export_output_schema, "export")
    debug_path.rename(output_path)
    logger.info("Final CSV written to %s", output_path)

    # — 6. Cost report
    extracted_dir = cfg["paths"].get("extracted_dir")
    if extracted_dir:
        token_usage = read_batch_token_usage(Path(extracted_dir))
        ext_stats = state.extraction_stats or {}
        cost_report = build_cost_report({**ext_stats, **token_usage}, None, cfg)
        save_cost_report(cost_report, cfg)
        cost_usd = cost_report.get("total_cost_usd", 0.0)
        logger.info("Total pipeline cost: $%.4f", cost_usd)
    else:
        cost_usd = 0.0

    rows_exported = len(df)
    logger.info(
        "Export complete: %d rows → %s",
        rows_exported,
        output_path,
        extra={
            "event": "export_done",
            "rows_exported": rows_exported,
            "cost_usd": cost_usd,
            "output_path": str(output_path),
        },
    )

    state.df = df
