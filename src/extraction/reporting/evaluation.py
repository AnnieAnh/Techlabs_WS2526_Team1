"""Automated accuracy calculator — compare pipeline output to a golden set."""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from shared.constants import MISSING_SENTINELS as _MISSING_SENTINELS

logger = logging.getLogger("pipeline.evaluation")

_CATEGORICAL_FIELDS = [
    "contract_type", "work_modality", "seniority",
]
_NUMERIC_FIELDS = [
    "salary_min", "salary_max", "experience_years",
]
_LIST_FIELDS = [
    "technical_skills", "soft_skills", "nice_to_have_skills", "benefits", "tasks",
]

# Tier-1 fields live in the DataFrame, not in LLM results["data"].
# Map evaluation field name → list of candidate DataFrame column names
# (regex-prefixed first, then plain — covers both pre- and post-merge states).
_TIER1_DF_CANDIDATES: dict[str, list[str]] = {
    "contract_type": ["regex_contract_type", "contract_type"],
    "work_modality": ["regex_work_modality", "work_modality"],
    "seniority": ["regex_seniority_from_title", "seniority_from_title"],
    "salary_min": ["regex_salary_min", "salary_min"],
    "salary_max": ["regex_salary_max", "salary_max"],
    "experience_years": ["regex_experience_years", "experience_years"],
}


def _normalise_golden_value(val: Any) -> Any:
    """Normalise golden-set values loaded from CSV.

    csv.DictReader reads empty cells as ``""`` and NA markers as literal
    strings.  Convert these to ``None`` so comparisons with pipeline
    ``None`` values work correctly.
    """
    if val is None:
        return None
    if isinstance(val, str) and val.strip() in _MISSING_SENTINELS:
        return None
    return val


def _categorical_accuracy(
    pipeline: list[dict], golden: list[dict], field: str
) -> dict[str, Any]:
    """Compute exact match accuracy + confusion matrix for a categorical field."""
    correct = 0
    total = 0
    confusion: dict[str, dict[str, int]] = {}

    for p, g in zip(pipeline, golden):
        p_val = str(p.get(field) if p.get(field) is not None else "null")
        g_val = str(g.get(field) if g.get(field) is not None else "null")
        total += 1
        if p_val == g_val:
            correct += 1
        if g_val not in confusion:
            confusion[g_val] = {}
        confusion[g_val][p_val] = confusion[g_val].get(p_val, 0) + 1

    pct = round(correct / max(1, total) * 100, 1)

    mismatches: list[dict[str, Any]] = []
    for true_label, pred_counts in confusion.items():
        for pred_label, count in pred_counts.items():
            if true_label != pred_label:
                mismatches.append({"true": true_label, "predicted": pred_label, "count": count})
    mismatches.sort(key=lambda x: -x["count"])

    return {
        "exact_match_pct": pct,
        "correct": correct,
        "total": total,
        "confusion_matrix": confusion,
        "top_mismatches": mismatches[:10],
    }


def _numeric_accuracy(
    pipeline: list[dict], golden: list[dict], field: str
) -> dict[str, Any]:
    """Compute exact match and within-10% accuracy for a numeric field."""
    exact = 0
    within_10_pct = 0
    both_null = 0
    total = 0

    for p, g in zip(pipeline, golden):
        p_val = p.get(field)
        g_val = g.get(field)
        total += 1

        p_null = p_val is None or (isinstance(p_val, float) and pd.isna(p_val))
        g_null = g_val is None or (isinstance(g_val, float) and pd.isna(g_val))

        if p_null and g_null:
            both_null += 1
            exact += 1
            within_10_pct += 1
            continue
        if p_null or g_null:
            continue

        try:
            pv = float(p_val)  # type: ignore[arg-type]
            gv = float(g_val)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue

        if pv == gv:
            exact += 1
            within_10_pct += 1
        elif gv != 0 and abs(pv - gv) / abs(gv) <= 0.1:
            within_10_pct += 1

    return {
        "exact_match_pct": round(exact / max(1, total) * 100, 1),
        "within_10pct_pct": round(within_10_pct / max(1, total) * 100, 1),
        "both_null": both_null,
        "correct": exact,
        "total": total,
    }


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union > 0 else 0.0


def _parse_list_field(val: Any) -> list:
    """Parse a list field that may be a JSON string (from CSV), a list, or None."""
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        val = val.strip()
        if not val or val in ("[]", "NA"):
            return []
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return []


def _list_accuracy(
    pipeline: list[dict], golden: list[dict], field: str
) -> dict[str, Any]:
    """Compute Jaccard similarity, precision, recall for a list field."""
    jaccard_sum = 0.0
    precision_sum = 0.0
    recall_sum = 0.0
    total = 0

    for p, g in zip(pipeline, golden):
        p_vals = {str(v).lower() for v in _parse_list_field(p.get(field))}
        g_vals = {str(v).lower() for v in _parse_list_field(g.get(field))}
        total += 1

        jaccard_sum += _jaccard(p_vals, g_vals)

        if p_vals:
            precision_sum += len(p_vals & g_vals) / len(p_vals)
        else:
            precision_sum += 1.0 if not g_vals else 0.0

        if g_vals:
            recall_sum += len(p_vals & g_vals) / len(g_vals)
        else:
            recall_sum += 1.0 if not p_vals else 0.0

    n = max(1, total)
    return {
        "avg_jaccard_pct": round(jaccard_sum / n * 100, 1),
        "avg_precision_pct": round(precision_sum / n * 100, 1),
        "avg_recall_pct": round(recall_sum / n * 100, 1),
        "total": total,
    }


def evaluate(
    results: list[dict[str, Any]],
    golden: list[dict[str, Any]],
    df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Compare pipeline output to golden set by row_id match.

    Args:
        results: Pipeline extraction results, each with 'row_id' and 'data'.
        golden: Golden set rows, each with 'row_id' and field values at top level.
        df: Optional DataFrame with Tier-1 fields (contract_type, work_modality,
            seniority_from_title, salary_min, salary_max, experience_years).
            If provided, Tier-1 fields are sourced from here instead of results.

    Returns:
        Accuracy report with per-field metrics for categorical, numeric, and list fields.
    """
    logger.info(
        "=== Accuracy Evaluation: %d pipeline rows, %d golden rows ===",
        len(results),
        len(golden),
    )

    # Build pipeline lookup: start with LLM data, then overlay Tier-1 from DataFrame
    pipeline_by_id: dict[str, dict] = {r["row_id"]: dict(r.get("data") or {}) for r in results}

    if df is not None and hasattr(df, "columns") and "row_id" in df.columns:
        for _, row in df.iterrows():
            rid = row.get("row_id")
            if rid and rid in pipeline_by_id:
                for eval_field, candidates in _TIER1_DF_CANDIDATES.items():
                    df_col = next((c for c in candidates if c in df.columns), None)
                    if df_col is not None:
                        val = row[df_col]
                        if val is not None and not (isinstance(val, float) and pd.isna(val)):
                            pipeline_by_id[rid][eval_field] = val
                        else:
                            pipeline_by_id[rid][eval_field] = None

    golden_by_id: dict[str, dict] = {
        g["row_id"]: {
            k: _normalise_golden_value(v) for k, v in g.items() if k != "row_id"
        }
        for g in golden
    }

    common_ids = sorted(set(pipeline_by_id) & set(golden_by_id))
    logger.info(
        "Matched %d rows (pipeline: %d, golden: %d)",
        len(common_ids),
        len(pipeline_by_id),
        len(golden_by_id),
    )

    if not common_ids:
        logger.warning("No matching row_ids between pipeline and golden set")
        return {"matched_rows": 0, "total_pipeline_rows": len(results),
                "total_golden_rows": len(golden), "fields": {}}

    p_list = [pipeline_by_id[rid] for rid in common_ids]
    g_list = [golden_by_id[rid] for rid in common_ids]

    fields: dict[str, Any] = {}

    for field in _CATEGORICAL_FIELDS:
        fields[field] = _categorical_accuracy(p_list, g_list, field)
        pct = fields[field]["exact_match_pct"]
        if pct < 80:
            logger.warning("Field '%s' accuracy %.1f%% below 80%% threshold", field, pct)
        else:
            logger.info("Field '%s': %.1f%% exact match", field, pct)

    for field in _NUMERIC_FIELDS:
        fields[field] = _numeric_accuracy(p_list, g_list, field)
        pct = fields[field]["exact_match_pct"]
        if pct < 80:
            logger.warning("Field '%s' accuracy %.1f%% below 80%% threshold", field, pct)
        else:
            logger.info(
                "Field '%s': %.1f%% exact match, %.1f%% within 10%%",
                field,
                pct,
                fields[field]["within_10pct_pct"],
            )

    for field in _LIST_FIELDS:
        fields[field] = _list_accuracy(p_list, g_list, field)
        jaccard = fields[field]["avg_jaccard_pct"]
        if jaccard < 80:
            logger.warning("Field '%s' Jaccard %.1f%% below 80%% threshold", field, jaccard)
        else:
            logger.info("Field '%s': %.1f%% Jaccard similarity", field, jaccard)

    return {
        "matched_rows": len(common_ids),
        "total_pipeline_rows": len(results),
        "total_golden_rows": len(golden),
        "fields": fields,
    }


def save_accuracy_report(report: dict[str, Any], cfg: dict) -> Path:
    """Save accuracy report to reports_dir/accuracy_report.json.

    Args:
        report: Report dict from evaluate().
        cfg: Pipeline config (reads cfg["paths"]["reports_dir"]).

    Returns:
        Path to the saved file.
    """
    reports_dir = Path(cfg.get("paths", {}).get("reports_dir", "data/reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "accuracy_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Accuracy report saved to %s", path)
    return path
