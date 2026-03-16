"""Pipeline quality report — field coverage, distributions, and data quality concerns.

Generates two artefacts:
  - data/reports/quality_report.json  — machine-readable full report
  - data/reports/quality_report.md    — human-readable summary
"""

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger("pipeline.reporting.quality")


def _field_coverage(results: list[dict[str, Any]]) -> dict[str, float]:
    """Return per-field coverage % — fraction of rows where the field is non-null/non-empty.

    None, empty string, and empty list are all treated as missing.
    Fields with 0% coverage are included so quality_concerns can flag them.
    """
    if not results:
        return {}

    # Initialise all known fields to 0 so 0%-coverage fields are visible
    all_fields: set[str] = set()
    for row in results:
        all_fields.update((row.get("data") or {}).keys())

    field_counts: dict[str, int] = {f: 0 for f in all_fields}
    for row in results:
        data = row.get("data", {}) or {}
        for field, value in data.items():
            if value is None:
                continue
            if isinstance(value, list) and len(value) == 0:
                continue
            if isinstance(value, str) and value == "":
                continue
            field_counts[field] += 1

    n = len(results)
    return {
        field: round(count / n * 100, 1)
        for field, count in sorted(field_counts.items())
    }


def _distribution(results: list[dict[str, Any]], field: str) -> dict[str, int]:
    """Count value frequency for a categorical field, sorted by frequency descending."""
    counts: dict[str, int] = {}
    for row in results:
        value = (row.get("data") or {}).get(field)
        key = "null" if value is None else str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def _distribution_from_df(df: Any, column: str) -> dict[str, int]:
    """Count value frequency for a Tier-1 field from the DataFrame.

    Tier-1 fields (e.g. regex_contract_type, regex_work_modality) are
    stored in the DataFrame, not in LLM results. Returns empty dict if df
    is None or lacks the column.
    """
    if df is None or not hasattr(df, "columns") or column not in df.columns:
        return {}
    import pandas as pd  # noqa: PLC0415 — deferred to avoid import when df is None

    counts: dict[str, int] = {}
    for value in df[column]:
        is_null = value is None or (isinstance(value, float) and pd.isna(value))
        key = "null" if is_null else str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def _top_skills(results: list[dict[str, Any]], n: int = 50) -> list[dict[str, Any]]:
    """Return the top-N most frequent technical skills across all rows."""
    counter: Counter[str] = Counter()
    for row in results:
        skills = (row.get("data") or {}).get("technical_skills") or []
        counter.update(skills)
    return [{"skill": skill, "count": count} for skill, count in counter.most_common(n)]


def _top_soft_skills(results: list[dict[str, Any]], n: int = 20) -> list[dict[str, Any]]:
    """Return the top-N most frequent soft skills across all rows."""
    counter: Counter[str] = Counter()
    for row in results:
        skills = (row.get("data") or {}).get("soft_skills") or []
        counter.update(skills)
    return [{"skill": skill, "count": count} for skill, count in counter.most_common(n)]


def _hallucination_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise skill verification flags from validation_flags across all rows."""
    skill_flag_count = 0
    high_hallucination_rows = 0
    most_flagged: Counter[str] = Counter()

    for row in results:
        flags = row.get("validation_flags") or []
        row_skill_flags = [f for f in flags if f.get("rule") == "skill_not_in_description"]
        if row_skill_flags:
            skill_flag_count += len(row_skill_flags)
        if any(f.get("rule") == "high_hallucination_rate" for f in flags):
            high_hallucination_rows += 1
        for f in row_skill_flags:
            msg = f.get("message", "")
            # Message format: "'SkillName' not grounded in description"
            m = re.search(r"'([^']+)'\s+not grounded", msg)
            if m:
                most_flagged[m.group(1)] += 1

    return {
        "total_skill_flags": skill_flag_count,
        "high_hallucination_rows": high_hallucination_rows,
        "top_flagged_skills": dict(most_flagged.most_common(10)),
    }


def _benefit_coverage(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise benefit and benefit_category coverage across all rows."""
    n = len(results)
    if n == 0:
        return {}
    with_benefits = sum(
        1 for r in results
        if (r.get("data") or {}).get("benefits")
        and len((r.get("data") or {}).get("benefits", [])) > 0
    )
    with_categories = sum(
        1 for r in results
        if (r.get("data") or {}).get("benefit_categories")
        and len((r.get("data") or {}).get("benefit_categories", [])) > 0
    )
    return {
        "rows_with_benefits_pct": round(with_benefits / n * 100, 1),
        "rows_with_benefit_categories_pct": round(with_categories / n * 100, 1),
    }


def _salary_stats(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute salary statistics from LLM ``data`` dict (Tier 2 salary, if present).

    Note: salary is Tier 1 (regex-extracted) so ``data`` typically lacks these
    fields.  This function will return count=0 unless the LLM also returned
    salary values.  Post-cleaning salary stats should be computed from the
    final DataFrame instead.
    """

    def _extract(field: str) -> list[float]:
        values = []
        for r in results:
            v = (r.get("data") or {}).get(field)
            if v is None:
                continue
            try:
                values.append(float(v))
            except (TypeError, ValueError):
                pass
        return values

    def _stats(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"count": 0}
        sv = sorted(values)
        n = len(sv)
        return {
            "count": n,
            "min": int(min(sv)),
            "max": int(max(sv)),
            "mean": int(sum(sv) / n),
            "median": int(sv[n // 2]),
        }

    mins = _extract("salary_min")
    maxs = _extract("salary_max")
    return {
        "salary_min": _stats(mins),
        "salary_max": _stats(maxs),
        "rows_with_salary": len(mins),
        "coverage_pct": round(len(mins) / max(1, len(results)) * 100, 1),
    }


def _location_stats(df: Any) -> dict[str, Any]:
    """Compute location distribution from the original DataFrame.

    Returns an empty dict if df is None or lacks location columns.
    """
    if df is None:
        return {}

    result: dict[str, Any] = {}

    if hasattr(df, "columns"):
        if "city" in df.columns:
            result["top_cities"] = df["city"].dropna().value_counts().head(20).to_dict()
            result["na_city_pct"] = round(
                df["city"].isna().sum() / max(1, len(df)) * 100, 1
            )
        if "state" in df.columns:
            result["top_states"] = df["state"].dropna().value_counts().head(20).to_dict()
        if "country" in df.columns:
            result["country_distribution"] = df["country"].value_counts().to_dict()

    return result


def _validation_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise validation flags across all rows."""
    rule_counts: dict[str, int] = {}
    severity_counts: dict[str, int] = {}
    rows_with_flags = 0

    for row in results:
        flags = row.get("validation_flags") or []
        if flags:
            rows_with_flags += 1
        for flag in flags:
            rule = flag.get("rule", "unknown")
            severity = flag.get("severity", "unknown")
            rule_counts[rule] = rule_counts.get(rule, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

    n = len(results)
    return {
        "rows_with_flags": rows_with_flags,
        "rows_clean": n - rows_with_flags,
        "clean_pct": round((n - rows_with_flags) / max(1, n) * 100, 1),
        "flags_by_rule": dict(sorted(rule_counts.items(), key=lambda x: -x[1])),
        "flags_by_severity": severity_counts,
    }


def _quality_concerns(coverage: dict[str, float]) -> list[str]:
    """List fields whose coverage falls below warning thresholds."""
    concerns = []
    for field, pct in coverage.items():
        if pct < 50:
            concerns.append(f"{field}: {pct}% coverage (below 50% threshold)")
        elif pct < 70:
            concerns.append(f"{field}: {pct}% coverage (moderate — below 70%)")
    return concerns


def _to_markdown(report: dict[str, Any]) -> str:
    """Convert a quality report dict to human-readable Markdown."""
    lines: list[str] = [
        "# Pipeline Quality Report",
        "",
        "## Summary",
        "",
        f"- Total rows: **{report['summary']['total_rows']}**",
        f"- Extraction coverage: **{report['summary']['extraction_coverage_pct']}%**",
        "",
        "## Field Coverage",
        "",
    ]

    for field, pct in report["field_coverage"].items():
        filled = int(pct / 10)
        bar = "▓" * filled + "░" * (10 - filled)
        lines.append(f"- `{field}`: {bar} {pct}%")

    lines += ["", "## Categorical Distributions", ""]
    for cat_field in ("contract_type", "work_modality", "seniority"):
        dist = report["distributions"].get(cat_field, {})
        if dist:
            lines.append(f"### {cat_field}")
            for value, count in list(dist.items())[:8]:
                lines.append(f"- {value}: {count}")
            lines.append("")

    lines += ["## Top 20 Technical Skills", ""]
    for entry in report["top_skills"][:20]:
        lines.append(f"- {entry['skill']}: {entry['count']}")

    lines += ["", "## Top 20 Soft Skills", ""]
    for entry in report.get("top_soft_skills", [])[:20]:
        lines.append(f"- {entry['skill']}: {entry['count']}")

    lines += ["", "## Salary Statistics", ""]
    sal = report["salary_stats"]
    lines.append(f"- Rows with salary: {sal['rows_with_salary']} ({sal['coverage_pct']}%)")
    if sal["salary_min"].get("count", 0) > 0:
        sm = sal["salary_min"]
        lines.append(
            f"- Min salary range: {sm['min']:,} – {sm['max']:,} (median {sm['median']:,} EUR)"
        )
    if sal["salary_max"].get("count", 0) > 0:
        sx = sal["salary_max"]
        lines.append(
            f"- Max salary range: {sx['min']:,} – {sx['max']:,} (median {sx['median']:,} EUR)"
        )

    loc = report.get("location_stats", {})
    if loc.get("top_cities"):
        lines += ["", "## Top 10 Cities", ""]
        for city, count in list(loc["top_cities"].items())[:10]:
            lines.append(f"- {city}: {count}")
    if loc.get("top_states"):
        lines += ["", "## Top 10 States", ""]
        for state, count in list(loc["top_states"].items())[:10]:
            lines.append(f"- {state}: {count}")

    vs = report.get("validation_summary", {})
    if vs:
        lines += [
            "",
            "## Validation Summary",
            "",
            f"- Clean rows: {vs.get('rows_clean', 0)} ({vs.get('clean_pct', 0)}%)",
            f"- Rows with flags: {vs.get('rows_with_flags', 0)}",
        ]
        if vs.get("flags_by_rule"):
            lines.append("")
            for rule, count in list(vs["flags_by_rule"].items())[:10]:
                lines.append(f"- `{rule}`: {count}")

    hal = report.get("hallucination_summary", {})
    if hal:
        lines += [
            "",
            "## Hallucination Summary",
            "",
            f"- Total skill-not-in-description flags: {hal.get('total_skill_flags', 0)}",
            f"- High-hallucination rows: {hal.get('high_hallucination_rows', 0)}",
        ]
        if hal.get("top_flagged_skills"):
            lines.append("")
            for skill, count in list(hal["top_flagged_skills"].items())[:10]:
                lines.append(f"- '{skill}': {count}")

    bc = report.get("benefit_coverage", {})
    if bc:
        lines += [
            "",
            "## Benefit Coverage",
            "",
            f"- Rows with benefits: {bc.get('rows_with_benefits_pct', 0)}%",
            f"- Rows with benefit categories: {bc.get('rows_with_benefit_categories_pct', 0)}%",
        ]

    if report.get("quality_concerns"):
        lines += ["", "## Quality Concerns", ""]
        for concern in report["quality_concerns"]:
            lines.append(f"- {concern}")

    return "\n".join(lines) + "\n"


def generate_quality_report(
    results: list[dict[str, Any]],
    df: Any,
    cfg: dict,
    save: bool = True,
) -> dict[str, Any]:
    """Generate a full quality report from extraction results.

    Args:
        results: Extraction result dicts, each containing 'row_id', 'data',
                 and (optionally) 'validation_flags'.
        df: Original DataFrame with location/title columns, or None if unavailable.
        cfg: Pipeline config dict (reads cfg["paths"]["reports_dir"]).
        save: If True, persist JSON and Markdown to reports_dir.

    Returns:
        Report dict with sections: summary, data_fingerprint, field_coverage,
        list_truncation_summary, distributions, top_skills, top_soft_skills,
        hallucination_summary, benefit_coverage, salary_stats, location_stats,
        validation_summary, quality_concerns.
    """
    logger.info("=== Generating Quality Report for %d rows ===", len(results))

    coverage = _field_coverage(results)

    # Include dataset fingerprint if available (best-effort; never fails the report)
    try:
        from shared.fingerprint import fingerprint_inputs
        data_fingerprint: dict[str, Any] = fingerprint_inputs(cfg)
    except Exception:
        data_fingerprint = {}

    truncated_count = sum(1 for r in results if r.get("was_truncated"))
    truncation_rate = round(truncated_count / max(1, len(results)) * 100, 1)

    # List truncation summary (tasks, etc.)
    list_trunc_counts: dict[str, int] = {}
    for r in results:
        for trunc in r.get("list_truncations", []):
            field_name = trunc["field"]
            list_trunc_counts[field_name] = list_trunc_counts.get(field_name, 0) + 1

    report: dict[str, Any] = {
        "summary": {
            "total_rows": len(results),
            "extraction_coverage_pct": round(
                sum(1 for r in results if r.get("data")) / max(1, len(results)) * 100, 1
            ),
            "truncation_rate": truncation_rate,
            "truncated_descriptions": truncated_count,
        },
        "data_fingerprint": data_fingerprint,
        "field_coverage": coverage,
        "list_truncation_summary": list_trunc_counts,
        "distributions": {
            "contract_type": _distribution_from_df(df, "regex_contract_type"),
            "work_modality": _distribution_from_df(df, "regex_work_modality"),
            "seniority": _distribution_from_df(df, "regex_seniority_from_title"),
        },
        "top_skills": _top_skills(results),
        "top_soft_skills": _top_soft_skills(results),
        "hallucination_summary": _hallucination_summary(results),
        "benefit_coverage": _benefit_coverage(results),
        "salary_stats": _salary_stats(results),
        "location_stats": _location_stats(df),
        "validation_summary": _validation_summary(results),
        "quality_concerns": _quality_concerns(coverage),
    }

    if save:
        reports_dir = Path(cfg.get("paths", {}).get("reports_dir", "data/reports"))
        reports_dir.mkdir(parents=True, exist_ok=True)

        json_path = reports_dir / "quality_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info("Quality report (JSON) saved to %s", json_path)

        md_path = reports_dir / "quality_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(_to_markdown(report))
        logger.info("Quality report (Markdown) saved to %s", md_path)

    logger.info(
        "Report complete: %d rows, %.1f%% extraction coverage, %d quality concerns",
        len(results),
        report["summary"]["extraction_coverage_pct"],
        len(report["quality_concerns"]),
    )
    return report
