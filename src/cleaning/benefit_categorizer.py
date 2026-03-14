"""Benefit categorization helpers for the cleaning pipeline."""

import json
from pathlib import Path

import yaml


def _load_config(filename: str) -> dict:
    """Load a YAML config file from cleaning/config/."""
    path = Path(__file__).parent / "config" / filename
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


BENEFIT_CATEGORIES: dict[str, list[str]] = _load_config("benefit_categories.yaml")["categories"]


def categorize_benefit(benefit: str) -> str:
    """Map a single benefit string to its category name (or 'other').

    Args:
        benefit: A single benefit string.

    Returns:
        Category name string, or 'other' if no category matched.
    """
    b_lower = benefit.lower()
    for category, keywords in BENEFIT_CATEGORIES.items():
        if any(kw in b_lower for kw in keywords):
            return category
    return "other"


def benefit_category_set(benefits_json: object) -> str:
    """Derive unique benefit categories from a JSON benefits list.

    Args:
        benefits_json: JSON string containing a list of benefit strings.

    Returns:
        JSON string with sorted unique category names.
    """
    try:
        benefits = json.loads(str(benefits_json))
        if not isinstance(benefits, list):
            return "[]"
    except (json.JSONDecodeError, TypeError):
        return "[]"
    categories = sorted({categorize_benefit(b) for b in benefits if isinstance(b, str)})
    return json.dumps(categories, ensure_ascii=False)
