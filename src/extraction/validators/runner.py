"""Post-extraction validation orchestrator — runs all validators in sequence."""

import json
import logging
import math
from pathlib import Path
from typing import Any

from extraction.validators.cross_field import validate_all_cross_fields
from extraction.validators.hallucination import (
    verify_all_skills,
    verify_evidence_item,
)
from extraction.validators.salary import validate_salaries
from extraction.validators.skills import load_skill_aliases, normalize_all_skills

logger = logging.getLogger("pipeline.validators.runner")

_DEFAULT_ALIASES_PATH = Path(__file__).parent.parent / "config" / "skill_aliases.yaml"


def run_validators(
    results: list[dict[str, Any]],
    df_rows: list[dict[str, Any]],
    cfg: dict,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run all post-extraction validators in sequence and aggregate flags per row.

    Steps executed in order:
      1. Skills normalisation — alias resolution, dedup, contradiction removal
      2. Skill verification — flag skills absent from the description
      3. Hallucination removal — drop flagged skills from data in-place
      4. Evidence flattening — preserve full evidence objects, then flatten names
      5. Cross-field consistency — 7 rules across title, seniority, modality, etc.
      6. Salary validation — range and floor/ceiling checks on Tier 1 salary fields
      7. Truncation flags — mark rows with truncated descriptions
      8. List truncation flags — mark rows where task lists were truncated during parsing

    Args:
        results: Extraction result dicts, each with 'row_id' and 'data'.
        df_rows: Original DataFrame rows with 'row_id', 'description', 'title_cleaned',
                 and 'seniority_from_title' fields (used for context-aware checks).
        cfg: Pipeline config dict.

    Returns:
        Tuple (final_results, report):
          - final_results: Results with 'validation_flags' list added to each row.
          - report: Summary dict saved to reports_dir/validation_report.json.
    """
    logger.info("=== STAGE: Post-Extraction Validation ===")

    # Resolve skill_aliases path
    aliases_path_raw = cfg.get("paths", {}).get("skill_aliases", _DEFAULT_ALIASES_PATH)
    aliases_path = Path(aliases_path_raw)
    if not aliases_path.exists():
        aliases_path = _DEFAULT_ALIASES_PATH
    aliases = load_skill_aliases(aliases_path)

    def _nn(v: object) -> object:
        """Return None for float NaN; return v unchanged otherwise.

        df.to_dict('records') produces float('nan') for missing values in object
        columns (a pandas quirk: replace("NA", None) stores NaN, not Python None).
        NaN is truthy, so (nan or "").lower() crashes. Normalise here at the boundary.
        """
        return None if isinstance(v, float) and math.isnan(v) else v

    # Build lookups from original rows
    desc_by_id: dict[str, str] = {
        r["row_id"]: str(_nn(r.get("description")) or "") for r in df_rows
    }
    title_by_id: dict[str, str] = {
        r["row_id"]: str(_nn(r.get("title_cleaned")) or _nn(r.get("title")) or "")
        for r in df_rows
    }
    regex_by_id: dict[str, dict[str, Any]] = {
        r["row_id"]: {
            "regex_contract_type": _nn(r.get("regex_contract_type")),
            "regex_work_modality": _nn(r.get("regex_work_modality")),
            "regex_experience_years": _nn(r.get("regex_experience_years")),
            "regex_salary_min": _nn(r.get("regex_salary_min")),
            "regex_salary_max": _nn(r.get("regex_salary_max")),
        }
        for r in df_rows
    }

    # 1. Normalise skills (modifies 'data' inside each result dict)
    results, skill_stats = normalize_all_skills(results, aliases)

    # 2. Skill verification
    threshold = cfg.get("validation", {}).get("skill_hallucination_threshold", 0.3)
    skill_flags = verify_all_skills(results, desc_by_id, aliases, threshold)

    # 3. Remove hallucinated skills from data (evidence items not grounded in description)
    removed_skills_total = 0
    for r in results:
        rid = r.get("row_id", "")
        desc = desc_by_id.get(rid, "")
        if not desc:
            continue
        data = r.get("data") or {}
        for field in ("technical_skills", "nice_to_have_skills"):
            raw = list(data.get(field) or [])
            if not raw:
                continue
            clean = [
                item for item in raw
                if verify_evidence_item(item, desc, aliases)[0]
            ]
            removed = len(raw) - len(clean)
            if removed:
                data[field] = clean
                removed_skills_total += removed
                logger.debug(
                    "[%s] Removed %d unverified item(s) from %s: %s",
                    rid, removed, field,
                    [item["name"] for item in raw if item not in clean],
                )
    if removed_skills_total:
        logger.info("Removed %d hallucinated skill(s) across all rows.", removed_skills_total)

    # 4. Preserve full evidence objects, then flatten names for downstream compatibility
    _EVIDENCE_FIELDS = ("technical_skills", "nice_to_have_skills", "benefits", "tasks")
    for r in results:
        data = r.get("data") or {}
        for field in _EVIDENCE_FIELDS:
            items = data.get(field)
            if isinstance(items, list):
                # Save full evidence (with source quotes) to a parallel key
                data[f"{field}_evidence"] = [
                    item for item in items
                    if isinstance(item, dict) and item.get("name")
                ]
                # Flatten to plain name strings for backward compatibility
                data[field] = [
                    item["name"] for item in items
                    if isinstance(item, dict) and item.get("name")
                ]

    # 5. Cross-field validation — enrich results with metadata for context checks
    enriched: list[dict[str, Any]] = []
    for r in results:
        rid = r["row_id"]
        enriched.append({
            **r,
            "description": desc_by_id.get(rid, ""),
            "title_cleaned": title_by_id.get(rid, ""),
            **regex_by_id.get(rid, {}),
        })
    cross_flags = validate_all_cross_fields(enriched)

    # 6. Salary validation — reads regex_salary_min/max from enriched row (Tier 1 fields)
    salary_flags = validate_salaries(enriched, cfg)

    # Aggregate all flags by row
    from extraction.validators import ValidationFlag

    # 7. Truncation flags — mark rows where the description was truncated before LLM call
    truncation_flags: list[ValidationFlag] = []
    for r in results:
        if r.get("was_truncated"):
            truncation_flags.append(ValidationFlag(
                row_id=r["row_id"],
                field="description",
                rule="truncated",
                severity="warning",
                message=(
                    "Description truncated at max_description_tokens. "
                    "Salary, benefits, and other fields near the end may be incomplete."
                ),
            ))
    if truncation_flags:
        logger.info("Truncation flags: %d rows had truncated descriptions", len(truncation_flags))

    # 8. List truncation flags — mark rows where task lists were truncated during parsing
    list_truncation_flags: list[ValidationFlag] = []
    for r in results:
        for trunc in r.get("list_truncations", []):
            list_truncation_flags.append(ValidationFlag(
                row_id=r["row_id"],
                field=trunc["field"],
                rule="list_truncated",
                severity="info",
                message=(
                    f"{trunc['field']}: truncated from {trunc['original_count']} "
                    f"to {trunc['kept_count']}"
                ),
            ))
    if list_truncation_flags:
        logger.info(
            "List truncation flags: %d rows had truncated lists",
            len(list_truncation_flags),
        )

    all_flags: list[ValidationFlag] = (
        skill_flags + cross_flags + salary_flags + truncation_flags + list_truncation_flags
    )
    flags_by_row: dict[str, list[ValidationFlag]] = {}
    for flag in all_flags:
        flags_by_row.setdefault(flag.row_id, []).append(flag)

    # Attach validation_flags to each result
    final_results: list[dict[str, Any]] = []
    for r in results:
        rid = r["row_id"]
        row_flags = flags_by_row.get(rid, [])
        final_results.append({
            **r,
            "validation_flags": [
                {
                    "field": f.field,
                    "rule": f.rule,
                    "severity": f.severity,
                    "message": f.message,
                    **({"context": f.context} if f.context is not None else {}),
                }
                for f in row_flags
            ],
        })

    # Build summary report
    rule_counts: dict[str, int] = {}
    severity_counts: dict[str, int] = {}
    for flag in all_flags:
        rule_counts[flag.rule] = rule_counts.get(flag.rule, 0) + 1
        severity_counts[flag.severity] = severity_counts.get(flag.severity, 0) + 1

    rows_with_flags = sum(1 for r in final_results if r["validation_flags"])
    clean_count = len(final_results) - rows_with_flags
    clean_pct = clean_count / max(1, len(final_results)) * 100

    report: dict[str, Any] = {
        "total_validated": len(final_results),
        "rows_with_flags": rows_with_flags,
        "rows_clean": clean_count,
        "clean_pct": round(clean_pct, 1),
        "flags_by_rule": rule_counts,
        "flags_by_severity": severity_counts,
        "skill_normalisation": skill_stats,
    }

    # Persist report
    reports_dir = Path(cfg.get("paths", {}).get("reports_dir", "data/reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(
        "Validation complete: %d/%d rows flagged (%.1f%% clean). "
        "Errors: %d, Warnings: %d, Info: %d.",
        rows_with_flags,
        len(final_results),
        clean_pct,
        severity_counts.get("error", 0),
        severity_counts.get("warning", 0),
        severity_counts.get("info", 0),
    )
    logger.info("Validation report saved to %s", report_path)

    return final_results, report
