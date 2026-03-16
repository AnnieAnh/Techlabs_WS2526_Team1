"""Shared constants for the cleaning pipeline.

These are the canonical column lists and thresholds used across
cleaning modules. Import from here to avoid duplication.
"""

from shared.constants import (
    SALARY_MAX_CEILING as _SALARY_MAX_CEILING,
)
from shared.constants import (
    SALARY_MIN_FLOOR as _SALARY_MIN_FLOOR,
)

COLUMN_ORDER = [
    "row_id",
    "job_url",
    "date_posted",
    "company_name",
    "city",
    "state",
    "title",
    "title_cleaned",
    "job_family",
    "job_summary",
    "seniority_from_title",
    "contract_type",
    "work_modality",
    "salary_min",
    "salary_max",
    "experience_years",
    "education_level",
    "technical_skills",
    "soft_skills",
    "nice_to_have_skills",
    "benefits",
    "tasks",
    # Enrichment columns added by the cleaning pipeline
    "languages",
    "benefit_categories",
    "soft_skill_categories",
    "description_quality",
    "site",
    "validation_flags",
    "description",
]

# Columns excluded from the analysis-ready output (not needed for aggregation/charting)
_QA_ONLY_COLUMNS = {"validation_flags", "description_quality", "description"}

ANALYSIS_COLUMN_ORDER = [c for c in COLUMN_ORDER if c not in _QA_ONLY_COLUMNS]

_LIST_COLUMNS = [
    "technical_skills",
    "soft_skills",
    "nice_to_have_skills",
    "benefits",
    "tasks",
    "languages",
]

_STRING_EXTRACTED_COLUMNS = [
    "job_family",
    "seniority_from_title",
    "contract_type",
    "work_modality",
]

_SALARY_COLUMNS = ["salary_min", "salary_max"]

__all__ = [
    "COLUMN_ORDER",
    "ANALYSIS_COLUMN_ORDER",
    "_QA_ONLY_COLUMNS",
    "_LIST_COLUMNS",
    "_STRING_EXTRACTED_COLUMNS",
    "_SALARY_COLUMNS",
    "_SALARY_MIN_FLOOR",
    "_SALARY_MAX_CEILING",
]
