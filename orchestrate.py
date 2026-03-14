"""End-to-end pipeline orchestrator — single entry point for all 8 steps.

Usage::

    # Run all steps
    python orchestrate.py

    # Run only up to and including a specific step
    python orchestrate.py --step deduplicate

    # Resume from a specific step (skip completed steps)
    python orchestrate.py --from validate

    # Run exactly one step
    python orchestrate.py --only clean_enrich

    # Dry run: show what would run
    python orchestrate.py --dry-run

    # List all steps with descriptions
    python orchestrate.py --list

    # Clear progress and start fresh
    python orchestrate.py --reset
"""

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / "src" / "extraction" / ".env")

# Set up shared logger before any other imports that log
from shared.logging import setup_pipeline_logger  # noqa: E402

setup_pipeline_logger()

from pipeline_state import PipelineState  # noqa: E402
from steps.clean_enrich import run_clean_enrich  # noqa: E402
from steps.deduplicate import run_deduplicate  # noqa: E402
from steps.export import run_export  # noqa: E402
from steps.extract import run_extract  # noqa: E402
from steps.ingest import run_ingest  # noqa: E402
from steps.prepare import run_prepare  # noqa: E402
from steps.regex_extract import run_regex_extract  # noqa: E402
from steps.validate import run_validate  # noqa: E402

logger = logging.getLogger("pipeline.orchestrate")

# ---------------------------------------------------------------------------
# Step registry — order is execution order
# ---------------------------------------------------------------------------

_STEPS: list[tuple[str, str, object]] = [
    ("ingest", "Load source CSVs, normalize schema and dates", run_ingest),
    ("prepare", "Validate input, parse locations, normalize titles", run_prepare),
    ("deduplicate", "Filter flagged rows, URL/composite dedup, grouping", run_deduplicate),
    ("regex_extract", "Deterministic regex field extraction (8 fields)", run_regex_extract),
    ("extract", "LLM semantic extraction via DeepSeek (grouped)", run_extract),
    ("validate", "Post-extraction validation, corrections, quality report", run_validate),
    ("clean_enrich", "Clean data + add enrichment columns", run_clean_enrich),
    ("export", "Column order, invariant checks, write final CSV", run_export),
]
_STEP_NAMES = [s[0] for s in _STEPS]

_PROGRESS_FILE = Path("data/pipeline_progress.json")


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------


def _load_progress() -> dict:
    """Load progress file or return empty dict."""
    if _PROGRESS_FILE.exists():
        with open(_PROGRESS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_progress(progress: dict) -> None:
    """Persist progress to JSON file."""
    _PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


def _mark_step(progress: dict, step_name: str, status: str, **kwargs: object) -> None:
    """Update a step's status in the progress dict."""
    entry: dict = {"status": status, **kwargs}
    if status in ("complete", "failed"):
        entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    progress[step_name] = entry
    _save_progress(progress)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_cfg() -> dict:
    """Load and validate merged extraction + ingestion config via shared.config."""
    from shared.config import load_pipeline_config, validate_config

    cfg = load_pipeline_config()
    validate_config(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Step execution loop
# ---------------------------------------------------------------------------


def _steps_to_run(
    progress: dict,
    step_filter: str | None,
    from_step: str | None,
    only_step: str | None,
) -> list[tuple[str, str, object]]:
    """Return the ordered list of (name, description, fn) to execute."""
    if only_step:
        return [(n, d, f) for n, d, f in _STEPS if n == only_step]

    stop_idx = _STEP_NAMES.index(step_filter) if step_filter else len(_STEP_NAMES) - 1
    steps = [(n, d, f) for n, d, f in _STEPS if _STEP_NAMES.index(n) <= stop_idx]

    if from_step:
        from_idx = _STEP_NAMES.index(from_step)
        steps = [(n, d, f) for n, d, f in steps if _STEP_NAMES.index(n) >= from_idx]
    else:
        # Skip steps already complete (resume behaviour)
        steps = [(n, d, f) for n, d, f in steps if progress.get(n, {}).get("status") != "complete"]

    return steps


def _rehydrate_state(state: PipelineState, cfg: dict, steps: list[tuple[str, str, object]]) -> None:
    """Restore state.df (and description_groups) from disk when earlier steps are skipped.

    Intermediate DataFrames live only in memory, so resuming from a mid-pipeline
    step requires re-reading from the most recent disk artifact.

    Priority:
    1. ``deduplicate`` skipped → load the most-recent deduped CSV + description_groups.json.
    2. ``prepare`` skipped but ``deduplicate`` will run → no prepare artifact exists; raise.
    3. Only ``ingest`` skipped → load combined_jobs.csv and re-assign row IDs.

    Args:
        state: Mutable pipeline state — sets ``state.df`` (and optionally
            ``state.description_groups``) when skipping completed steps.
        cfg: Pipeline config dict — supplies all relevant paths.
        steps: The ordered list of steps that will actually run this session.
    """
    from extraction.checkpoint import Checkpoint
    from shared.io import read_csv_safe

    step_names = {s[0] for s in steps}

    if "ingest" in step_names:
        return  # ingest will populate state.df itself

    # ------------------------------------------------------------------ #
    # Case 1: deduplicate was completed and is being skipped.             #
    # Best artifact: deduped_*.csv (post-filter, post-dedup) +           #
    #                description_groups.json                              #
    # ------------------------------------------------------------------ #
    if "deduplicate" not in step_names:
        # If clean_enrich is also being skipped, try loading its disk artifact
        # (enriched_cleaned.csv) written by the clean_enrich step. Fall back to error if missing.
        if "clean_enrich" not in step_names:
            enriched_path = Path(cfg["paths"]["extracted_dir"]) / "enriched_cleaned.csv"
            if enriched_path.exists():
                df = read_csv_safe(enriched_path)
                state.df = df
                logger.info(
                    "Rehydrated state.df from %s (%d rows)",
                    enriched_path,
                    len(df),
                    extra={
                        "event": "state_rehydrated",
                        "source": str(enriched_path),
                        "rows": len(df),
                    },
                )
                return
            raise RuntimeError(
                "Cannot resume: 'clean_enrich' is marked complete but no disk "
                "artifact found at " + str(enriched_path) + ". "
                "Re-run from clean_enrich: python orchestrate.py --from clean_enrich"
            )

        deduped_dir: Path = cfg["paths"]["deduped_dir"]
        csvs = sorted(deduped_dir.glob("deduped_*.csv"))
        if not csvs:
            raise FileNotFoundError(
                f"Cannot resume: no deduped CSV found in {deduped_dir}. "
                "Re-run from deduplicate: python orchestrate.py --from deduplicate"
            )
        latest_csv = csvs[-1]
        df = read_csv_safe(latest_csv)

        groups_path = deduped_dir / "description_groups.json"
        if groups_path.exists():
            with open(groups_path, encoding="utf-8") as f:
                state.description_groups = json.load(f)
            logger.info(
                "Rehydrated description_groups from %s (%d groups)",
                groups_path,
                len(state.description_groups or {}),
            )
        else:
            logger.warning(
                "description_groups.json not found in %s — extract will process all rows",
                deduped_dir,
            )

        # If regex_extract is also being skipped, re-run it inline (fast,
        # deterministic, no API calls). The deduped CSV doesn't have regex_*
        # columns since regex_extract now runs as a separate step after dedup.
        if "regex_extract" not in step_names and "regex_salary_min" not in df.columns:
            from extraction.preprocessing.regex_extractor import extract_regex_fields

            logger.info("Re-running regex extraction inline during rehydration (%d rows)", len(df))
            rows = df.to_dict("records")
            regex_rows = [
                extract_regex_fields(
                    str(r.get("description", "")),
                    str(r.get("title_cleaned") or r.get("title", "")),
                )
                for r in rows
            ]
            for key in (
                "contract_type", "work_modality", "salary_min", "salary_max",
                "experience_years", "seniority_from_title", "languages", "education_level",
            ):
                df[f"regex_{key}"] = [r[key] for r in regex_rows]
            logger.info("Regex rehydration complete")

        state.df = df
        logger.info(
            "Rehydrated state.df from %s (%d rows)",
            latest_csv,
            len(df),
            extra={"event": "state_rehydrated", "source": str(latest_csv), "rows": len(df)},
        )
        return

    # ------------------------------------------------------------------ #
    # Case 2: prepare was completed but deduplicate will run.             #
    # prepare does not write a disk artifact, so we cannot rehydrate.    #
    # ------------------------------------------------------------------ #
    if "prepare" not in step_names:
        raise RuntimeError(
            "Cannot resume: 'prepare' is marked complete but no disk artifact exists "
            "(prepare output lives only in memory). "
            "Re-run from prepare: python orchestrate.py --from prepare"
        )

    # ------------------------------------------------------------------ #
    # Case 3: only ingest was completed; prepare will run next.           #
    # Load combined_jobs.csv and re-assign row IDs.                       #
    # ------------------------------------------------------------------ #
    from steps.ingest import _row_id_from_url

    ingestion_path = cfg["paths"]["ingestion_output"]
    if not ingestion_path.exists():
        raise FileNotFoundError(
            f"Cannot resume: ingestion output not found at {ingestion_path}. "
            "Re-run from ingest: python orchestrate.py --from ingest"
        )

    df = read_csv_safe(ingestion_path)
    df = df.copy()
    df["row_id"] = df["job_url"].apply(_row_id_from_url)

    cp = Checkpoint(cfg["paths"]["checkpoint_db"])
    cp.register_rows([{"row_id": rid, "file_path": str(ingestion_path)} for rid in df["row_id"]])

    state.df = df
    logger.info(
        "Rehydrated state.df from %s (%d rows)",
        ingestion_path,
        len(df),
        extra={"event": "state_rehydrated", "source": str(ingestion_path), "rows": len(df)},
    )


def _run_steps(
    steps: list[tuple[str, str, object]],
    state: PipelineState,
    cfg: dict,
    progress: dict,
) -> None:
    """Execute steps in order, updating progress after each."""
    for name, _desc, fn in steps:
        logger.info(
            "=" * 70 + "\nStep: %s\n" + "=" * 70,
            name,
            extra={"event": "step_start", "step": name},
        )
        t0 = time.monotonic()
        _mark_step(progress, name, "running", started_at=datetime.now(timezone.utc).isoformat())
        try:
            fn(state, cfg)  # type: ignore[operator]
            elapsed = time.monotonic() - t0
            _mark_step(progress, name, "complete", elapsed_s=round(elapsed, 1))
            logger.info(
                "Step %s complete in %.1fs",
                name,
                elapsed,
                extra={"event": "step_complete", "step": name, "elapsed_s": elapsed},
            )
        except Exception as exc:
            elapsed = time.monotonic() - t0
            _mark_step(progress, name, "failed", elapsed_s=round(elapsed, 1), error=str(exc))
            logger.error(
                "Step %s FAILED after %.1fs: %s",
                name,
                elapsed,
                exc,
                extra={"event": "step_failed", "step": name, "error": str(exc)},
            )
            raise


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


def _cmd_clean(cfg: dict) -> None:
    """Remove all intermediate/cached pipeline data so a fresh run can start.

    Deletes:
    - data/pipeline_progress.json (orchestrator progress)
    - data/extraction/ subdirs (deduped, extracted, failed, reports, batches)
    - data/extraction/pipeline_state.db (checkpoint SQLite)
    - data/ingestion/combined_jobs.csv
    - data/cleaning/ contents
    """
    removed: list[str] = []

    if _PROGRESS_FILE.exists():
        _PROGRESS_FILE.unlink()
        removed.append(str(_PROGRESS_FILE))

    db_path: Path = cfg["paths"]["checkpoint_db"]
    if db_path.exists():
        db_path.unlink()
        removed.append(str(db_path))

    enriched_csv = Path(cfg["paths"].get("extracted_dir", "")) / "enriched_cleaned.csv"
    if enriched_csv.exists():
        enriched_csv.unlink()
        removed.append(str(enriched_csv))

    for subdir_key in ("deduped_dir", "extracted_dir", "failed_dir", "reports_dir"):
        subdir = cfg["paths"].get(subdir_key)
        if subdir and Path(subdir).exists():
            shutil.rmtree(subdir)
            removed.append(str(subdir))

    batches_dir = cfg["paths"].get("batches_dir")
    if batches_dir and Path(batches_dir).exists():
        shutil.rmtree(batches_dir)
        removed.append(str(batches_dir))

    ingestion_output: Path = cfg["paths"]["ingestion_output"]
    if ingestion_output.exists():
        ingestion_output.unlink()
        removed.append(str(ingestion_output))

    cleaning_dir = Path("data/cleaning")
    if cleaning_dir.exists():
        for f in cleaning_dir.iterdir():
            if f.is_file():
                f.unlink()
                removed.append(str(f))

    if removed:
        print(f"\nCleaned {len(removed)} item(s):")
        for p in removed:
            print(f"  - {p}")
    else:
        print("\nNothing to clean — no cached data found.")
    print()


def _cmd_list() -> None:
    """Print all steps with descriptions."""
    print("\nPipeline steps (execution order):\n")
    for i, (name, desc, _) in enumerate(_STEPS, 1):
        print(f"  {i:>2}. {name:<15}  {desc}")
    print()


def _cmd_dry_run(
    progress: dict,
    step_filter: str | None,
    from_step: str | None,
    only_step: str | None,
) -> None:
    """Print which steps would run without executing them."""
    steps_to_run_names = {s[0] for s in _steps_to_run(progress, step_filter, from_step, only_step)}
    print("\nDRY RUN -- no steps will be executed:\n")
    for name, desc, _ in _STEPS:
        status = progress.get(name, {}).get("status", "pending")
        if name not in steps_to_run_names:
            marker = "[SKIP]"
        elif status == "complete":
            marker = "[DONE]"
        elif status == "failed":
            marker = "[FAIL]"
        else:
            marker = "[TODO]"
        print(f"  {marker}  {name:<15}  {desc}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="End-to-end German IT jobs pipeline orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--step",
        choices=_STEP_NAMES,
        metavar="STEP",
        help="Run all steps up to and including STEP",
    )
    group.add_argument(
        "--from",
        dest="from_step",
        choices=_STEP_NAMES,
        metavar="STEP",
        help="Resume from STEP onwards (re-runs STEP even if previously complete)",
    )
    group.add_argument(
        "--only",
        choices=_STEP_NAMES,
        metavar="STEP",
        help="Run exactly one step",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would run")
    parser.add_argument("--list", action="store_true", dest="list_steps", help="List all steps")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear progress file and start fresh",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove all intermediate/cached data (progress, checkpoints, outputs)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help="Limit pipeline to the first N rows after ingest (for test runs)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Orchestrate the end-to-end pipeline."""
    args = _parse_args(argv)

    if args.list_steps:
        _cmd_list()
        return

    if args.clean:
        cfg = _load_cfg()
        _cmd_clean(cfg)
        return

    progress = _load_progress()

    if args.reset:
        logger.warning("--reset: clearing pipeline progress")
        progress = {}
        _save_progress(progress)

    if args.dry_run:
        _cmd_dry_run(progress, args.step, args.from_step, args.only)
        return

    cfg = _load_cfg()
    state = PipelineState()
    state.row_limit = args.limit

    steps = _steps_to_run(progress, args.step, args.from_step, args.only)

    if not steps:
        logger.info("All steps already complete. Use --reset to start fresh or --from STEP.")
        return

    _rehydrate_state(state, cfg, steps)

    # Fingerprint inputs for reproducibility / cache-invalidation
    from shared.fingerprint import fingerprint_inputs

    fingerprint = fingerprint_inputs(cfg)

    # Warn if config changed since last run (stale results risk)
    prev_fp = progress.get("_fingerprint", {})
    prev_config_hash = prev_fp.get("config_hash")
    if prev_config_hash and prev_config_hash != fingerprint["config_hash"]:
        logger.warning(
            "Config fingerprint changed since last run (%s → %s). "
            "Previously completed steps used different settings — consider --reset.",
            prev_config_hash,
            fingerprint["config_hash"],
        )

    progress["_fingerprint"] = fingerprint
    _save_progress(progress)

    pipeline_start = time.monotonic()
    logger.info(
        "Pipeline starting — %d step(s) to run: %s",
        len(steps),
        ", ".join(s[0] for s in steps),
        extra={"event": "pipeline_start", "steps": [s[0] for s in steps]},
    )

    try:
        _run_steps(steps, state, cfg, progress)
    except Exception:
        elapsed = time.monotonic() - pipeline_start
        logger.error(
            "Pipeline failed after %.1fs",
            elapsed,
            extra={"event": "pipeline_failed", "elapsed_s": elapsed},
        )
        sys.exit(1)

    elapsed = time.monotonic() - pipeline_start
    logger.info(
        "Pipeline complete in %.1fs",
        elapsed,
        extra={"event": "pipeline_complete", "elapsed_s": elapsed},
    )


if __name__ == "__main__":
    main()
