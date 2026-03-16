"""Cross-field consistency validation rules."""

import logging
import math
from typing import Any

from extraction.validators import ValidationFlag


def _to_str(val: object) -> str:
    """Return '' for None or NaN; str(val) otherwise.

    Guards against float('nan') values that arrive when a DataFrame is converted
    to dicts via .to_dict('records') — NaN is truthy, so (nan or '') returns nan,
    and nan.lower() raises AttributeError.
    """
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return ""
    return str(val)

logger = logging.getLogger("pipeline.validators.cross_field")

# Keywords that indicate on-site presence
_ONSITE_KEYWORDS = ("vor ort", "on-site", "onsite", "im büro", "im buero", "in office")
# Keywords that indicate full remote
_REMOTE_KEYWORDS = (
    "100% remote", "fully remote", "full remote",
    "vollständig remote", "vollstaendig remote",
)
# Intern/working-student title keywords
_INTERN_TITLE_KEYWORDS = ("intern", "werkstudent", "working student", "praktikant", "trainee")
# Contract types acceptable for interns
_INTERN_CONTRACT_TYPES = ("Part-time", "Internship", None)


def validate_cross_fields(row: dict[str, Any]) -> list[ValidationFlag]:
    """Apply cross-field consistency rules to a single row.

    Active rules: title/seniority mismatch, intern contract mismatch,
    seniority source mismatch, unrealistic experience years,
    no skills on long description, remote/onsite text contradictions.

    The row dict must contain:
        - 'row_id': str
        - 'data': dict with LLM-extracted fields
        - 'description': str (original job description)
        - 'title_cleaned': str (cleaned title)
        - 'seniority_from_title': str (from title normalizer, may be absent)

    Args:
        row: Combined row dict with extraction data and original metadata.

    Returns:
        List of ValidationFlag instances, empty if no issues found.
    """
    flags: list[ValidationFlag] = []
    row_id = row.get("row_id", "unknown")
    data = row.get("data", {})

    title = (_to_str(row.get("title_cleaned")) or _to_str(row.get("title"))).lower()
    description = _to_str(row.get("description")).lower()
    seniority_from_title = _to_str(row.get("seniority_from_title")).lower()

    # NOTE: the LLM schema does not include a "seniority" field (seniority
    # is Tier 1, regex-extracted).  data.get("seniority") will return None,
    # so Rules 1 and 3 below are currently inert.  Kept for forward-compat
    # if the LLM prompt ever adds a seniority field.
    extracted_seniority = _to_str(data.get("seniority")).lower()
    contract_type = _to_str(row.get("regex_contract_type"))
    modality = _to_str(row.get("regex_work_modality")).lower()
    exp_years = row.get("regex_experience_years")
    technical_skills = list(data.get("technical_skills") or [])

    # Rule 1: Title says "Junior" but extracted seniority is "Senior"
    if "junior" in title and extracted_seniority == "senior":
        flags.append(ValidationFlag(
            row_id=row_id,
            field="seniority",
            rule="title_seniority_mismatch",
            severity="warning",
            message=f"Title contains 'Junior' but extracted seniority is '{data.get('seniority')}'",
        ))

    # Rule 2: Title suggests intern/working student but contract_type doesn't match
    if any(kw in title for kw in _INTERN_TITLE_KEYWORDS):
        if contract_type and contract_type not in _INTERN_CONTRACT_TYPES:
            flags.append(ValidationFlag(
                row_id=row_id,
                field="contract_type",
                rule="intern_contract_mismatch",
                severity="warning",
                message=(
                    f"Title suggests intern/working student but contract_type='{contract_type}'"
                ),
            ))

    # Rule 3: seniority_from_title != extracted seniority (informational)
    if (
        seniority_from_title
        and seniority_from_title not in ("unknown", "")
        and extracted_seniority
        and extracted_seniority not in ("")
        and seniority_from_title != extracted_seniority
    ):
        flags.append(ValidationFlag(
            row_id=row_id,
            field="seniority",
            rule="seniority_source_mismatch",
            severity="info",
            message=(
                f"seniority_from_title='{seniority_from_title}' "
                f"!= extracted seniority='{data.get('seniority')}'"
            ),
        ))

    # Rule 4: experience_years > 10 is unrealistic — flag and set to None
    if exp_years is not None:
        try:
            if float(exp_years) > 10:
                flags.append(ValidationFlag(
                    row_id=row_id,
                    field="experience_years",
                    rule="experience_unrealistic",
                    severity="warning",
                    message=f"experience_years={exp_years} exceeds 10 — set to None",
                ))
                row["regex_experience_years"] = None
        except (TypeError, ValueError):
            pass

    # Rule 5: No technical skills extracted but description is long (likely extraction miss)
    if not technical_skills and len(description) > 500:
        flags.append(ValidationFlag(
            row_id=row_id,
            field="technical_skills",
            rule="no_skills_long_description",
            severity="info",
            message="No technical_skills extracted but description is > 500 chars",
        ))

    # Rule 6: work_modality is "Remote" but description contains on-site keywords
    if modality == "remote":
        if any(kw in description for kw in _ONSITE_KEYWORDS):
            flags.append(ValidationFlag(
                row_id=row_id,
                field="work_modality",
                rule="remote_but_onsite_text",
                severity="warning",
                message="work_modality='Remote' but description contains on-site keywords",
            ))

    # Rule 7: work_modality is "On-site" but description says fully remote
    if modality in ("on-site", "onsite"):
        if any(kw in description for kw in _REMOTE_KEYWORDS):
            flags.append(ValidationFlag(
                row_id=row_id,
                field="work_modality",
                rule="onsite_but_remote_text",
                severity="warning",
                message="work_modality='Onsite' but description mentions fully remote work",
            ))

    return flags


def validate_all_cross_fields(
    rows: list[dict[str, Any]],
) -> list[ValidationFlag]:
    """Run cross-field validation across all rows.

    Args:
        rows: List of row dicts, each containing 'data', 'description',
              'title_cleaned', and 'seniority_from_title'.

    Returns:
        All ValidationFlags produced across all rows.
    """
    logger.info("=== Cross-Field Validation ===")
    all_flags: list[ValidationFlag] = []

    for row in rows:
        all_flags.extend(validate_cross_fields(row))

    rule_counts: dict[str, int] = {}
    for f in all_flags:
        rule_counts[f.rule] = rule_counts.get(f.rule, 0) + 1

    logger.info(
        "Cross-field validation: %d rows, %d flags. By rule: %s",
        len(rows),
        len(all_flags),
        rule_counts,
    )
    return all_flags
