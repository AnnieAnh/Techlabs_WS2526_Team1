"""Pipeline cost report — extraction API cost calculation and reporting."""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("pipeline.reporting.cost")

# DeepSeek V3 pricing (standard, non-cached)
_DEEPSEEK_INPUT_COST_PER_M = 0.27    # USD per million input tokens
_DEEPSEEK_OUTPUT_COST_PER_M = 1.10   # USD per million output tokens


def _compute_cost(input_tokens: int, output_tokens: int) -> float:
    """Compute DeepSeek V3 API cost in USD."""
    return (
        input_tokens * _DEEPSEEK_INPUT_COST_PER_M / 1_000_000
        + output_tokens * _DEEPSEEK_OUTPUT_COST_PER_M / 1_000_000
    )


def build_cost_report(
    extraction_stats: dict[str, Any],
    title_stats: dict[str, Any] | None,
    cfg: dict,
) -> dict[str, Any]:
    """Build cost report from extraction and title classification stats.

    Args:
        extraction_stats: Dict with keys: calls (int), input_tokens (int),
            output_tokens (int).
        title_stats: Optional dict with keys: calls (int), input_tokens (int),
            output_tokens (int). Pass None if title classification was not run.
        cfg: Pipeline config (reads extraction model name for reference).

    Returns:
        Cost report dict with per-stage breakdown and total cost.
    """
    model = cfg.get("extraction", {}).get("model", "deepseek-chat")

    ext_input = int(extraction_stats.get("input_tokens", 0))
    ext_output = int(extraction_stats.get("output_tokens", 0))
    ext_cost = _compute_cost(ext_input, ext_output)

    report: dict[str, Any] = {
        "model": model,
        "pricing": {
            "input_cost_per_m_tokens": _DEEPSEEK_INPUT_COST_PER_M,
            "output_cost_per_m_tokens": _DEEPSEEK_OUTPUT_COST_PER_M,
            "note": "DeepSeek V3 standard pricing",
        },
        "extraction": {
            "calls": int(extraction_stats.get("calls", 0)),
            "input_tokens": ext_input,
            "output_tokens": ext_output,
            "cost_usd": round(ext_cost, 4),
            "retries": int(extraction_stats.get("retries", 0)),
            "failed": int(extraction_stats.get("failed", 0)),
        },
    }

    if title_stats is not None:
        t_input = int(title_stats.get("input_tokens", 0))
        t_output = int(title_stats.get("output_tokens", 0))
        t_cost = _compute_cost(t_input, t_output)
        report["title_classification"] = {
            "calls": int(title_stats.get("calls", 0)),
            "input_tokens": t_input,
            "output_tokens": t_output,
            "cost_usd": round(t_cost, 4),
        }
    else:
        report["title_classification"] = None

    title_cost = report["title_classification"]["cost_usd"] if title_stats else 0.0
    total_cost = round(ext_cost + title_cost, 4)
    report["total_cost_usd"] = total_cost

    logger.info(
        "Cost report: extraction=$%.2f, title_classification=$%.2f, total=$%.2f",
        ext_cost,
        title_cost,
        total_cost,
    )
    return report


def read_batch_token_usage(extracted_dir: Path) -> dict[str, int]:
    """Read DeepSeek token usage from the token_usage.json file saved during LLM extraction.

    Args:
        extracted_dir: Directory containing ``token_usage.json`` (written after extraction).

    Returns:
        Dict with keys: calls (int), input_tokens (int), output_tokens (int).
        All zero if the file is not found.
    """
    usage_path = Path(extracted_dir) / "token_usage.json"
    if not usage_path.exists():
        logger.warning("token_usage.json not found at %s — returning zero usage", usage_path)
        return {"calls": 0, "input_tokens": 0, "output_tokens": 0}

    try:
        with open(usage_path, encoding="utf-8") as f:
            data = json.load(f)
        calls = int(data.get("calls", 0))
        input_tokens = int(data.get("input_tokens", 0))
        output_tokens = int(data.get("output_tokens", 0))
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        logger.error("Failed to read token_usage.json: %s", exc)
        return {"calls": 0, "input_tokens": 0, "output_tokens": 0}

    logger.info(
        "DeepSeek token usage: %d calls, %d input tokens, %d output tokens",
        calls,
        input_tokens,
        output_tokens,
    )
    return {"calls": calls, "input_tokens": input_tokens, "output_tokens": output_tokens}


def save_cost_report(report: dict[str, Any], cfg: dict) -> Path:
    """Save cost report to reports_dir/cost_report.json.

    Args:
        report: Cost report dict from build_cost_report().
        cfg: Pipeline config (reads cfg["paths"]["reports_dir"]).

    Returns:
        Path to the saved file.
    """
    reports_dir = Path(cfg.get("paths", {}).get("reports_dir", "data/reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "cost_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Cost report saved to %s", path)
    return path
