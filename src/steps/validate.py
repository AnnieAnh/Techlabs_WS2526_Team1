"""Step 6: Validate — post-extraction validation, corrections, evaluation, quality report.

Separate from extraction so validation rules can be iterated without re-running
the expensive LLM step. Reads from ``state.extraction_results`` when available,
falling back to disk on resume — avoids redundant file re-reads during normal
sequential execution.

Sub-tasks in order:
1. run_validators — skill normalization, skill verification, cross-field checks, salary sanity
2. apply_remap_categoricals — map raw LLM enum values to canonical forms
3. apply_fix_cpp_inference — correct C++ hallucination
4. apply_normalize_skill_casing — canonical skill casing (e.g. python → Python)
5. evaluate (optional) — golden set accuracy if golden_set.csv exists
6. generate_quality_report — coverage, distributions, hallucination summary
"""

import json
import logging
from pathlib import Path
from typing import Any

from extraction.post_extraction import (
    apply_fix_cpp_inference,
    apply_normalize_skill_casing,
    apply_remap_categoricals,
)
from extraction.reporting.evaluation import evaluate, save_accuracy_report
from extraction.reporting.quality import generate_quality_report
from extraction.validators.runner import run_validators
from pipeline_state import PipelineState
from shared.schemas import validate_output_schema, validate_step_output

logger = logging.getLogger("pipeline.validate")


def run_validate(state: PipelineState, cfg: dict) -> None:
    """Validate, correct, and report on extraction results.

    Reads ``state.extraction_results`` if already populated; otherwise loads
    from ``extraction_results.json`` on disk (for resume after interruption).

    Writes validated results back to ``extraction_results.json`` after corrections.

    Args:
        state: Mutable pipeline state — reads/modifies ``state.extraction_results``.
        cfg: Pipeline config dict.
    """
    logger.info("=" * 70)
    logger.info("Step 6: Validate")
    logger.info("=" * 70)

    extracted_dir: Path = cfg["paths"]["extracted_dir"]
    results_path = extracted_dir / "extraction_results.json"

    # Load results from state or fall back to disk
    if state.extraction_results is None:
        if not results_path.exists():
            logger.info("No extraction_results.json found — skipping validate step")
            return
        with open(results_path, encoding="utf-8") as f:
            state.extraction_results = json.load(f)

    df_rows: list[dict[str, Any]] = (
        state.df.to_dict("records") if not state.df.empty else []  # type: ignore[assignment]
    )

    # — 1. Validators (skill normalization, skill verification, cross-field, salary)
    state.extraction_results, val_report = run_validators(
        state.extraction_results, df_rows, cfg
    )
    rows_flagged = val_report["rows_with_flags"]
    total_validated = val_report["total_validated"]
    logger.info(
        "Validation: %d/%d rows flagged",
        rows_flagged,
        total_validated,
        extra={
            "event": "validation_done",
            "rows_flagged": rows_flagged,
            "total_validated": total_validated,
        },
    )

    # — 2–4. Post-extraction corrections
    state.extraction_results = apply_remap_categoricals(state.extraction_results)
    desc_by_id = {r["row_id"]: r.get("description", "") for r in df_rows}
    state.extraction_results = apply_fix_cpp_inference(state.extraction_results, desc_by_id)
    state.extraction_results = apply_normalize_skill_casing(state.extraction_results)

    # Write corrected results back to disk
    extracted_dir.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(state.extraction_results, f, indent=2, ensure_ascii=False)
    logger.info("Validated + corrected results saved to %s", results_path)

    # — 5. Evaluation (optional — skip if no golden set)
    golden_path = cfg["paths"]["validation_dir"] / "golden_set.csv"
    if golden_path.exists():
        import csv

        with open(golden_path, encoding="utf-8", newline="") as gf:
            golden = list(csv.DictReader(gf))
        acc_report = evaluate(state.extraction_results, golden, df=state.df)
        save_accuracy_report(acc_report, cfg)
        logger.info(
            "Evaluation: %d rows matched, accuracy report saved",
            acc_report["matched_rows"],
        )
    else:
        logger.info("No golden_set.csv at %s — skipping evaluation", golden_path)

    # — 6. Quality report
    df_for_report = state.df if not state.df.empty else None
    generate_quality_report(state.extraction_results, df_for_report, cfg)
    logger.info("Quality report generated")
    validate_step_output(state.df, validate_output_schema, "validate")
