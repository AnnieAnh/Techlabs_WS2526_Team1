"""Step 9: Export Light — produce a lightweight analysis CSV without heavy columns.

Drops columns not needed for analysis (description, validation_flags,
description_quality, original title) to produce a much smaller file
suitable for notebooks and version control.
"""

import logging
from pathlib import Path

from pipeline_state import PipelineState
from shared.io import write_csv_safe

logger = logging.getLogger("pipeline.export_light")

# Columns to exclude from the light version
_DROP_COLUMNS = ["description", "description_quality", "validation_flags", "title"]


def run_export_light(state: PipelineState, cfg: dict) -> None:
    """Create a lightweight CSV by dropping heavy/debug columns.

    Reads the full cleaned CSV produced by the export step and writes a
    smaller version without description, validation_flags, description_quality,
    and the original title column.

    Args:
        state: Mutable pipeline state — reads ``state.df``.
        cfg: Pipeline config dict.
    """
    df = state.require_df("export_light")

    logger.info("=" * 70)
    logger.info("Step 9: Export Light (%d rows, %d columns)", len(df), len(df.columns))
    logger.info("=" * 70)

    # Drop heavy columns (silently skip any that don't exist)
    cols_to_drop = [c for c in _DROP_COLUMNS if c in df.columns]
    df_light = df.drop(columns=cols_to_drop)

    logger.info(
        "Dropped %d columns: %s",
        len(cols_to_drop),
        ", ".join(cols_to_drop),
    )

    # Resolve output path
    full_path = Path(
        cfg.get("export", {}).get("output_path", "data/cleaning/cleaned_jobs.csv")
    )
    light_path = full_path.with_name("cleaned_jobs_light.csv")
    light_path.parent.mkdir(parents=True, exist_ok=True)

    write_csv_safe(df_light, light_path)

    logger.info(
        "Export Light complete: %d rows, %d columns → %s",
        len(df_light),
        len(df_light.columns),
        light_path,
        extra={
            "event": "export_light_done",
            "rows": len(df_light),
            "columns": len(df_light.columns),
            "output_path": str(light_path),
        },
    )
