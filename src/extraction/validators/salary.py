"""Salary range and consistency validation."""

import logging
from typing import Any

from extraction.validators import ValidationFlag
from shared.constants import (
    SALARY_MAX_CEILING as _SALARY_MAX_CEILING,
)
from shared.constants import (
    SALARY_MIN_FLOOR as _SALARY_MIN_FLOOR,
)
from shared.constants import (
    SALARY_MONTHLY_THRESHOLD as _MONTHLY_THRESHOLD,
)

logger = logging.getLogger("pipeline.validators.salary")


def check_salary(
    row_id: str,
    salary_min: Any,
    salary_max: Any,
    cfg: dict,
) -> list[ValidationFlag]:
    """Validate salary fields for a single row.

    Args:
        row_id: Row identifier.
        salary_min: Extracted minimum salary (int/float/None).
        salary_max: Extracted maximum salary (int/float/None).
        cfg: Pipeline config dict (reads cfg["validation"] thresholds).

    Returns:
        List of ValidationFlag instances, empty if all checks pass.
    """
    flags: list[ValidationFlag] = []

    val_cfg = cfg.get("validation", {})
    floor = val_cfg.get("salary_min_floor", _SALARY_MIN_FLOOR)
    ceiling = val_cfg.get("salary_max_ceiling", _SALARY_MAX_CEILING)
    monthly_threshold = val_cfg.get("salary_monthly_threshold", _MONTHLY_THRESHOLD)

    if salary_min is None and salary_max is None:
        return flags

    # Range checks per field
    for field_name, value in [("salary_min", salary_min), ("salary_max", salary_max)]:
        if value is None:
            continue
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue

        if v < monthly_threshold:
            flags.append(ValidationFlag(
                row_id=row_id,
                field=field_name,
                rule="possible_monthly",
                severity="warning",
                message=f"{field_name}={v:.0f} may be monthly salary (< {monthly_threshold})",
            ))
        elif v > ceiling:
            flags.append(ValidationFlag(
                row_id=row_id,
                field=field_name,
                rule="possible_revenue",
                severity="error",
                message=f"{field_name}={v:.0f} exceeds ceiling ({ceiling}) — possible revenue",
            ))
        elif v < floor:
            flags.append(ValidationFlag(
                row_id=row_id,
                field=field_name,
                rule="below_floor",
                severity="warning",
                message=f"{field_name}={v:.0f} below annual floor ({floor})",
            ))

    # min must not exceed max
    if salary_min is not None and salary_max is not None:
        try:
            if float(salary_min) > float(salary_max):
                flags.append(ValidationFlag(
                    row_id=row_id,
                    field="salary_min",
                    rule="min_greater_than_max",
                    severity="error",
                    message=f"salary_min ({salary_min}) > salary_max ({salary_max})",
                ))
        except (TypeError, ValueError):
            pass

    return flags


def validate_salaries(
    results: list[dict[str, Any]],
    cfg: dict,
) -> list[ValidationFlag]:
    """Run salary validation on all extraction results.

    Salary is Tier 1 (regex-extracted), so values are read from the top-level
    ``regex_salary_min`` and ``regex_salary_max`` keys injected by runner.py
    when building the enriched row dict — NOT from ``data["salary_min"]``.

    Args:
        results: Enriched row dicts with 'row_id', 'regex_salary_min', 'regex_salary_max'.
        cfg: Pipeline config dict.

    Returns:
        List of ValidationFlag instances across all rows.
    """
    logger.info("=== Salary Validation ===")
    all_flags: list[ValidationFlag] = []

    for row in results:
        row_id = row.get("row_id", "unknown")
        flags = check_salary(
            row_id=row_id,
            salary_min=row.get("regex_salary_min"),
            salary_max=row.get("regex_salary_max"),
            cfg=cfg,
        )
        all_flags.extend(flags)

    rule_counts: dict[str, int] = {}
    for f in all_flags:
        rule_counts[f.rule] = rule_counts.get(f.rule, 0) + 1

    logger.info(
        "Salary validation: %d rows checked, %d flagged. Rules: %s",
        len(results),
        len({f.row_id for f in all_flags}),
        rule_counts,
    )
    return all_flags
