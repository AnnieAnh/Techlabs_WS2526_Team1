"""Soft skill normalization and categorization helpers for the cleaning pipeline."""

import json
import re
from pathlib import Path

import pandas as pd
import yaml


def _load_config(filename: str) -> dict:
    """Load a YAML config file from cleaning/config/."""
    path = Path(__file__).parent / "config" / filename
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


_LANGUAGE_SKILL_RE = re.compile(r"(kenntnisse|sprachkenntnisse)", re.IGNORECASE)

_SOFT_SKILL_CATEGORIES: dict[str, list[str]] = (
    _load_config("soft_skill_categories.yaml")["categories"]
)


def normalize_soft_skills(df: pd.DataFrame) -> pd.DataFrame:
    """Filter language-like entries from soft_skills; add soft_skill_categories column.

    Removes skills matching language-skill patterns (e.g. 'Englischkenntnisse').
    Adds a new 'soft_skill_categories' column mapping remaining skills to 8 categories.

    Args:
        df: DataFrame with a 'soft_skills' column (JSON strings).

    Returns:
        DataFrame with cleaned 'soft_skills' and new 'soft_skill_categories' column.
    """
    if "soft_skills" not in df.columns:
        return df
    df = df.copy()

    def _clean(json_str: object) -> str:
        try:
            skills = json.loads(str(json_str))
            if not isinstance(skills, list):
                return str(json_str)
            return json.dumps(
                [s for s in skills if isinstance(s, str) and not _LANGUAGE_SKILL_RE.search(s)],
                ensure_ascii=False,
            )
        except (json.JSONDecodeError, TypeError):
            return str(json_str)

    def _categorize(json_str: object) -> str:
        try:
            skills = json.loads(str(json_str))
            if not isinstance(skills, list):
                return "[]"
            categories: set[str] = set()
            for skill in skills:
                if not isinstance(skill, str):
                    continue
                s_lower = skill.lower()
                for cat, keywords in _SOFT_SKILL_CATEGORIES.items():
                    if any(kw in s_lower for kw in keywords):
                        categories.add(cat)
            return json.dumps(sorted(categories), ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            return "[]"

    df["soft_skills"] = df["soft_skills"].apply(_clean)
    df["soft_skill_categories"] = df["soft_skills"].apply(_categorize)
    return df
