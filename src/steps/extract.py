"""Step 5: Extract — LLM semantic extraction via DeepSeek (description-grouped).

Most expensive step (API cost, wall-clock time). Only step requiring external
API access. Isolated so retry after failure doesn't re-run prepare/dedup;
iterating on preparation logic doesn't burn API credits.

Description grouping (from step 3) is used to call the LLM only for
representative rows, then propagate results to group members.
"""

import json
import logging
from pathlib import Path

from extraction.checkpoint import Checkpoint
from extraction.llm.processor import run_extraction as _llm_run_extraction
from pipeline_state import PipelineState
from shared.schemas import extract_output_schema, validate_step_output

logger = logging.getLogger("pipeline.extract")


def _propagate_results(
    results: list[dict],
    groups: dict,
    cp: Checkpoint,
) -> list[dict]:
    """Fan out extraction results from representative rows to group members.

    Idempotent: member rows already present in results are not duplicated.

    Args:
        results: Extraction results from the LLM (representative rows only).
        groups: Description dedup groups (from description_dedup.group_rows_by_description).
        cp: Checkpoint instance — member rows are marked as 'extracted'.

    Returns:
        Expanded results list including entries for all group members.
    """
    by_id: dict[str, dict] = {r["row_id"]: r for r in results}
    for group in groups.values():
        rep_id = group["representative_row_id"]
        rep_result = by_id.get(rep_id)
        if rep_result is None:
            continue
        for member_id in group["member_row_ids"]:
            if member_id == rep_id or member_id in by_id:
                continue
            by_id[member_id] = {**rep_result, "row_id": member_id}
            cp.advance_stage(member_id, "extracted")
    propagated = len(by_id) - len(results)
    if propagated:
        logger.info("Propagated extraction results to %d group members", propagated)
    return list(by_id.values())


def run_extract(state: PipelineState, cfg: dict) -> None:
    """Run LLM extraction on representative rows, propagate results to group members.

    If ``state.description_groups`` is populated (from deduplicate step), filters
    df to representative rows only before calling the LLM. Results are then fanned
    out to all group members. This avoids redundant LLM calls for identical descriptions.

    On completion, saves ``extraction_results.json`` and sets:
    - ``state.extraction_results``: list of per-row extraction dicts.
    - ``state.extraction_stats``: summary stats (success rate, cost, etc.).

    Args:
        state: Mutable pipeline state — reads ``state.df`` and ``state.description_groups``.
        cfg: Pipeline config dict.
    """
    df = state.require_df("extract")

    logger.info("=" * 70)
    logger.info("Step 5: Extract (%d rows)", len(df))
    logger.info("=" * 70)

    cp = Checkpoint(cfg["paths"]["checkpoint_db"])
    groups = state.description_groups or {}

    # Filter to representative rows only (saves API calls)
    if groups:
        rep_ids = {g["representative_row_id"] for g in groups.values()}
        df_input = df[df["row_id"].isin(rep_ids)]
        logger.info(
            "Filtered to %d representative rows (of %d total) for LLM extraction",
            len(df_input),
            len(df),
        )
    else:
        df_input = df

    results, ext_report = _llm_run_extraction(df_input, cfg, cp)

    # Fan out results to group members
    if groups:
        results = _propagate_results(results, groups, cp)
        extracted_dir: Path = cfg["paths"]["extracted_dir"]
        extracted_dir.mkdir(parents=True, exist_ok=True)
        results_path = extracted_dir / "extraction_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("Saved propagated extraction_results.json (%d rows)", len(results))

    state.extraction_results = results
    state.extraction_stats = ext_report

    success_rate = ext_report.get("success_rate", 0)
    logger.info(
        "Extraction complete: %d rows, %.1f%% success",
        len(results),
        success_rate * 100,
        extra={
            "event": "extraction_complete",
            "success_rate": success_rate,
            "total_rows": len(results),
        },
    )
    validate_step_output(state.df, extract_output_schema, "extract")
