"""Step 1: Ingest — load source CSVs, normalize schema and dates.

Delegates to the existing ingestion pipeline (run_pipeline) for all logic,
then loads the output CSV into state.df so subsequent steps can read from
state without a disk round-trip.
"""

import hashlib
import logging

from extraction.checkpoint import Checkpoint
from ingestion.pipeline import run_pipeline
from pipeline_state import PipelineState
from shared.io import read_csv_safe
from shared.schemas import ingestion_output_schema, validate_step_output

logger = logging.getLogger("pipeline.ingest")


def _row_id_from_url(job_url: str) -> str:
    """Return a 12-char deterministic hash of the job URL."""
    return hashlib.sha256(job_url.encode()).hexdigest()[:12]


def run_ingest(state: PipelineState, cfg: dict) -> None:
    """Load source CSVs, normalize columns and dates, write combined_jobs.csv.

    Calls ``ingestion.pipeline.run_pipeline(cfg=cfg)`` which reads
    ``ingestion/config/settings.yaml`` for source column mappings and
    scrape_date, but uses ``cfg["paths"]["ingestion_output"]`` as the
    canonical output path (controlled by the orchestrator).

    On success, sets ``state.df`` to the combined DataFrame so the next
    step can consume it in-memory without re-reading from disk.

    Args:
        state: Mutable pipeline state — sets ``state.df``.
        cfg: Orchestrator pipeline config dict — controls output path.
    """
    logger.info("=" * 70)
    logger.info("Step 1: Ingest")
    logger.info("=" * 70)

    # Pass orchestrator cfg so ingestion writes to the canonical path.
    # settings.yaml is still loaded internally for source column maps and
    # scrape_date, which are ingestion-specific.
    output_path = run_pipeline(cfg=cfg)

    if not output_path.exists():
        raise FileNotFoundError(
            f"Ingestion output not found at {output_path}. "
            "run_pipeline() should have written it."
        )

    df = read_csv_safe(output_path)

    # Assign deterministic row IDs (hash of job_url) and register in checkpoint.
    df = df.copy()
    df["row_id"] = df["job_url"].apply(_row_id_from_url)

    # Apply row limit if set (for test runs with --limit N)
    if state.row_limit:
        df = df.head(state.row_limit)
        logger.info("Row limit applied: keeping first %d rows", state.row_limit)

    cp = Checkpoint(cfg["paths"]["checkpoint_db"])
    cp.register_rows(
        [{"row_id": rid, "file_path": str(output_path)} for rid in df["row_id"]]
    )

    state.df = df

    validate_step_output(state.df, ingestion_output_schema, "ingest")

    logger.info(
        "Ingest complete: %d rows loaded from %s",
        len(state.df),
        output_path,
        extra={
            "event": "ingest_complete",
            "rows_after": len(state.df),
            "output_path": str(output_path),
        },
    )
