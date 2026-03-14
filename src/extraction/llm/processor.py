"""Async extraction processor: concurrent DeepSeek calls with checkpoint resume.

Uses asyncio.Semaphore to cap parallel requests (controlled by
llm.deepseek_max_workers in settings.yaml).
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.asyncio import tqdm as atqdm

from extraction.checkpoint import Checkpoint
from extraction.llm.client import LLMResponse, call_deepseek, get_token_usage, reset_token_usage
from extraction.llm.prompt_builder import build_message, load_extraction_prompt, prompt_version
from extraction.llm.response_parser import (
    ParseResult,
    load_output_schema,
    log_parse_summary,
    parse_response,
)

logger = logging.getLogger("pipeline.llm.processor")


async def _process_one(
    row: dict[str, Any],
    system: str,
    schema: dict,
    model: str,
    max_tokens: int,
    max_desc_tokens: int,
    max_tasks: int,
    prompt_config: dict,
    checkpoint: Checkpoint,
    sem: asyncio.Semaphore,
    successes: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    parse_results: list[ParseResult],
    pv: str = "",
    temperature: float = 0.0,
) -> None:
    """Process a single row: call DeepSeek, parse, checkpoint.

    Args:
        row: Dict with 'row_id', 'title', 'description', etc.
        system: System prompt string.
        schema: JSON Schema dict for response validation.
        model: DeepSeek model identifier.
        max_tokens: Maximum tokens for the API response.
        max_desc_tokens: Max description tokens (truncation threshold).
        max_tasks: Maximum number of tasks to keep in LLM response.
        prompt_config: Prompt template config from load_extraction_prompt().
        checkpoint: Checkpoint instance for resume support.
        sem: asyncio.Semaphore controlling concurrency.
        successes: Shared list accumulating successful extraction dicts.
        failures: Shared list accumulating failure dicts.
        parse_results: Shared list accumulating ParseResult objects for summary.
        pv: Prompt version fingerprint (8-char MD5) to record with each result.
        temperature: Sampling temperature forwarded from cfg["extraction"]["temperature"].
    """
    async with sem:
        row_id = row["row_id"]
        try:
            msg = build_message(row, prompt_config, max_desc_tokens)
            was_truncated = msg.get("was_truncated", False)
            user = msg["messages"][0]["content"]
            llm_resp: LLMResponse = await call_deepseek(
                system=system,
                user=user,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            raw = llm_resp.text
            if llm_resp.was_truncated:
                was_truncated = True
        except Exception as exc:
            logger.error("API error row %s: %s", row_id, exc)
            checkpoint.mark_failed(row_id, f"api_error: {exc}")
            failures.append({
                "row_id": row_id,
                "error": str(exc),
                "failure_type": "api_error",
            })
            return

        pr = parse_response(raw, schema, row_id=row_id, max_tasks=max_tasks)
        parse_results.append(pr)

        if pr.success:
            checkpoint.advance_stage(row_id, "extracted")
            result: dict[str, Any] = {
                "row_id": row_id,
                "data": pr.data,
                "warnings": pr.warnings,
                "parse_strategy": pr.parse_strategy,
                "prompt_version": pv,
            }
            if was_truncated:
                result["was_truncated"] = True
            if pr.list_truncations:
                result["list_truncations"] = [
                    {
                        "field": t.field,
                        "original_count": t.original_count,
                        "kept_count": t.kept_count,
                    }
                    for t in pr.list_truncations
                ]
            successes.append(result)
        else:
            checkpoint.mark_failed(row_id, f"parse_failed: {pr.error}")
            failures.append({
                "row_id": row_id,
                "error": pr.error,
                "raw_text": raw[:500],
                "failure_type": "parse_failed",
            })


async def _run_async(
    pending_rows: list[dict[str, Any]],
    prompt_config: dict,
    cfg: dict,
    checkpoint: Checkpoint,
    schema: dict,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Inner coroutine: dispatch all rows concurrently under a semaphore."""
    model = cfg["extraction"]["model"]
    max_tokens = cfg["extraction"].get("max_tokens", 2000)
    max_desc_tokens = cfg["extraction"]["max_description_tokens"]
    max_tasks = cfg["extraction"].get("max_tasks", 7)
    concurrency = cfg["llm"].get("deepseek_max_workers", 20)
    temperature = float(cfg["extraction"].get("temperature", 0.0))
    system = prompt_config["system_prompt"]
    _prompt_ver = prompt_version(system)

    sem = asyncio.Semaphore(concurrency)
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    parse_results: list[ParseResult] = []

    tasks = [
        _process_one(
            row, system, schema, model, max_tokens, max_desc_tokens,
            max_tasks, prompt_config, checkpoint, sem, successes, failures,
            parse_results, _prompt_ver, temperature,
        )
        for row in pending_rows
    ]

    await atqdm.gather(*tasks, desc="Extraction", unit="row", ncols=80)
    log_parse_summary(parse_results)
    return successes, failures


def run_extraction(
    df: pd.DataFrame,
    cfg: dict,
    checkpoint: Checkpoint,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run async extraction for all pending rows in the DataFrame.

    Handles resume automatically: rows already checkpointed as 'extracted'
    are skipped. On Ctrl+C, completed rows are saved and the pipeline can
    resume from the next unprocessed row.

    Args:
        df: DataFrame with 'row_id' column.
        cfg: Pipeline config dict.
        checkpoint: Checkpoint instance.

    Returns:
        Tuple (results, report):
          - results: list of extraction dicts with 'row_id', 'data', 'warnings',
            'parse_strategy', 'prompt_version', and optionally 'was_truncated'
            and 'list_truncations'.
          - report: summary dict with counts and success rate.
    """
    stage_start = time.monotonic()
    logger.info("=== STAGE: Async Extraction ===")

    extracted_dir: Path = cfg["paths"]["extracted_dir"]
    failed_dir: Path = cfg["paths"]["failed_dir"]

    completed_ids = set(checkpoint.get_completed("extracted"))
    all_rows: list[dict[str, Any]] = df.to_dict("records")  # type: ignore[assignment]
    pending_rows = [r for r in all_rows if r["row_id"] not in completed_ids]

    logger.info(
        "Rows needing extraction: %d (of %d total, %d already done)",
        len(pending_rows),
        len(all_rows),
        len(completed_ids),
    )

    if not pending_rows:
        logger.info("All rows already extracted — skipping stage")
        results_path = extracted_dir / "extraction_results.json"
        if results_path.exists():
            with open(results_path, encoding="utf-8") as f:
                return json.load(f), {"skipped": True}
        return [], {"skipped": True}

    schema = load_output_schema()
    prompt_config = load_extraction_prompt()

    try:
        successes, failures = asyncio.run(
            _run_async(pending_rows, prompt_config, cfg, checkpoint, schema)
        )
    except KeyboardInterrupt:
        logger.warning("Extraction interrupted by user — progress saved in checkpoint.")
        raise

    # Persist results — merge with existing to preserve data across resumed runs
    extracted_dir.mkdir(parents=True, exist_ok=True)
    results_path = extracted_dir / "extraction_results.json"

    # 1. extraction_results.json: merge new successes with previously-extracted rows
    existing: list[dict[str, Any]] = []
    if results_path.exists():
        try:
            with open(results_path, encoding="utf-8") as f:
                existing = json.load(f)
        except json.JSONDecodeError:
            logger.warning("Existing results file corrupt — starting fresh")
    merged_by_id: dict[str, dict[str, Any]] = {r["row_id"]: r for r in existing}
    merged_by_id.update({r["row_id"]: r for r in successes})
    merged = list(merged_by_id.values())
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    logger.info(
        "Extraction results saved to %s (%d new, %d total)",
        results_path, len(successes), len(merged),
    )

    if failures:
        failed_dir.mkdir(parents=True, exist_ok=True)
        failures_path = failed_dir / "parse_failures.json"
        # 2. parse_failures.json: merge, dedup by row_id
        existing_failures: list[dict[str, Any]] = []
        if failures_path.exists():
            try:
                with open(failures_path, encoding="utf-8") as f:
                    existing_failures = json.load(f)
            except json.JSONDecodeError:
                pass
        existing_fail_ids = {r["row_id"] for r in existing_failures}
        merged_failures = existing_failures + [
            flt for flt in failures if flt["row_id"] not in existing_fail_ids
        ]
        with open(failures_path, "w", encoding="utf-8") as f:
            json.dump(merged_failures, f, indent=2, ensure_ascii=False)
        logger.warning(
            "Failures (%d new, %d total) saved to %s",
            len(failures), len(merged_failures), failures_path,
        )

    # 3. token_usage.json: accumulate across resumed runs (not overwrite)
    token_usage = get_token_usage()
    usage_path = extracted_dir / "token_usage.json"
    existing_usage: dict[str, int] = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
    if usage_path.exists():
        try:
            with open(usage_path, encoding="utf-8") as f:
                existing_usage = json.load(f)
        except json.JSONDecodeError:
            pass
    merged_usage = {
        "calls": existing_usage.get("calls", 0) + len(successes) + len(failures),
        "input_tokens": existing_usage.get("input_tokens", 0) + token_usage["input_tokens"],
        "output_tokens": existing_usage.get("output_tokens", 0) + token_usage["output_tokens"],
    }
    with open(usage_path, "w", encoding="utf-8") as f:
        json.dump(merged_usage, f, indent=2)
    reset_token_usage()
    logger.info(
        "Token usage: %d input, %d output (this run) → %d input, %d output (total) → saved to %s",
        token_usage["input_tokens"],
        token_usage["output_tokens"],
        merged_usage["input_tokens"],
        merged_usage["output_tokens"],
        usage_path,
    )

    elapsed = time.monotonic() - stage_start
    n_new = len(successes) + len(failures)
    report: dict[str, Any] = {
        "total_rows": len(merged),
        "new_in_this_run": n_new,
        "successes": len(successes),
        "failures": len(failures),
        "success_rate": len(successes) / max(1, n_new),
        "elapsed_seconds": round(elapsed, 1),
    }
    logger.info(
        "Stage extraction complete: %d/%d extracted this run, %d total in %.1fs",
        len(successes),
        n_new,
        len(merged),
        elapsed,
    )
    return merged, report
