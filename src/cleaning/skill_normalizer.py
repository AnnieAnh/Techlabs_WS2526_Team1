"""Skill normalization helpers for the cleaning pipeline."""

import json
import logging
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger("pipeline.skill_normalizer")

# Shared skill variant lookup — same source as extraction/validators/hallucination.py
_SKILL_VARIANTS_PATH = (
    Path(__file__).parent.parent / "extraction" / "config" / "skill_variants.yaml"
)


def _load_skill_variants() -> dict[str, list[str]]:
    """Load the canonical-skill -> variant-list mapping from skill_variants.yaml.

    Returns:
        Dict mapping each canonical skill name to a list of known variant
        strings (lowercase substrings used for description matching).
    """
    with open(_SKILL_VARIANTS_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["variants"]


SKILL_VARIANTS: dict[str, list[str]] = _load_skill_variants()

# Bare-C regex — matches uppercase 'C' NOT preceded by alphanumeric,
# and NOT followed by alphanumeric, '+', '#', '/'
# Case-sensitive: only uppercase C indicates the C programming language.
_CPP_WORD_RE = re.compile(r"(?<![a-zA-Z0-9])C(?![a-zA-Z0-9+#/])")


def _skill_in_description(skill: str, text_lower: str) -> bool:
    """Word-boundary check: is skill present in description text?

    Args:
        skill: Skill name to search for.
        text_lower: Lowercased job description text.

    Returns:
        True if the skill (or a known variant) appears in the description.
    """
    if re.search(rf"\b{re.escape(skill.lower())}\b", text_lower):
        return True
    for variant in SKILL_VARIANTS.get(skill, []):
        if variant in text_lower:
            return True
    return False


def normalize_skill_casing(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize skill casing — most-frequent capitalisation variant wins per skill.

    Args:
        df: DataFrame with JSON list skill columns.

    Returns:
        DataFrame with skill casing normalized across all skill columns.
    """
    _all_skill_cols = ("technical_skills", "soft_skills", "nice_to_have_skills")
    skill_cols = [c for c in _all_skill_cols if c in df.columns]
    if not skill_cols:
        return df
    df = df.copy()

    # Count all skill occurrences across all skill columns
    skill_counter: Counter[str] = Counter()
    for col in skill_cols:
        for val in df[col]:
            try:
                skills = json.loads(str(val))
                if isinstance(skills, list):
                    skill_counter.update(s for s in skills if isinstance(s, str))
            except (json.JSONDecodeError, TypeError):
                pass

    # For each lowercase group, the first (most-frequent) occurrence wins
    canonical_map: dict[str, str] = {}
    for skill in (s for s, _ in skill_counter.most_common()):
        key = skill.lower()
        if key not in canonical_map:
            canonical_map[key] = skill

    n_normalized = 0

    def _normalize_list(json_str: object) -> str:
        nonlocal n_normalized
        try:
            skills = json.loads(str(json_str))
            if not isinstance(skills, list):
                return str(json_str)
            new_skills = []
            for s in skills:
                if isinstance(s, str):
                    canonical = canonical_map.get(s.lower(), s)
                    if canonical != s:
                        n_normalized += 1
                    new_skills.append(canonical)
                else:
                    new_skills.append(s)
            return json.dumps(new_skills, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            return str(json_str)

    for col in skill_cols:
        df[col] = df[col].apply(_normalize_list)

    if n_normalized:
        logger.info("Normalized %d skill casing variants", n_normalized)
    return df


def fix_cpp_inference(row: pd.Series) -> pd.Series:
    """Correct hallucinated C++ entries when description context doesn't support C++.

    Decision tree:
    - description contains 'c++' → genuine C++, return unchanged.
    - description contains bare 'C' (no C++ / C# / C/) → replace 'C++' → 'C' in skill lists.
    - neither → true hallucination, remove 'C++' from skill lists entirely.

    Args:
        row: A single DataFrame row (Series).

    Returns:
        The (possibly modified) row Series.
    """
    description_original = str(row.get("description", ""))

    if "c++" in description_original.lower():
        return row  # genuine C++ — leave unchanged

    row = row.copy()

    has_bare_c = bool(_CPP_WORD_RE.search(description_original))
    replacement: str | None = "C" if has_bare_c else None

    def _replace_cpp(json_str: object) -> str:
        try:
            skills = json.loads(str(json_str))
            if not isinstance(skills, list):
                return str(json_str)
            if replacement is None:
                skills = [s for s in skills if not (isinstance(s, str) and s.lower() == "c++")]
            else:
                skills = [
                    replacement if (isinstance(s, str) and s.lower() == "c++") else s
                    for s in skills
                ]
                # deduplicate preserving order (replacement may collide with existing entry)
                seen: set[str] = set()
                deduped = []
                for s in skills:
                    key = s.lower() if isinstance(s, str) else s
                    if key not in seen:
                        seen.add(key)
                        deduped.append(s)
                skills = deduped
            return json.dumps(skills, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            return str(json_str)

    for col in ("technical_skills", "nice_to_have_skills"):
        if col in row.index:
            row[col] = _replace_cpp(row[col])

    return row


def re_verify_skills_post_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Re-verify skills against description after C++ inference fix and skill casing normalization.

    Scans technical_skills and nice_to_have_skills for entries not present in
    the description. Replaces stale extraction-time skill flags (which may
    reference renamed or removed skills) with fresh post-clean flags. Non-skill
    flags (salary, cross_field, etc.) are preserved unchanged.

    Args:
        df: Cleaned DataFrame (after C++ fix and skill casing, before column drop).

    Returns:
        DataFrame with updated validation_flags column.
    """
    if "description" not in df.columns:
        return df
    df = df.copy()
    skill_cols = [c for c in ("technical_skills", "nice_to_have_skills") if c in df.columns]
    n_flagged = 0

    for idx, row in df.iterrows():
        text_lower = str(row.get("description", "")).lower()
        new_flags: list[dict] = []

        # Parse existing flags
        try:
            existing = json.loads(str(row.get("validation_flags", "[]")))
            if not isinstance(existing, list):
                existing = []
        except (json.JSONDecodeError, TypeError):
            existing = []

        # Keep non-skill flags; discard stale extraction-time skill flags
        kept_flags = [
            f for f in existing
            if isinstance(f, dict)
            and f.get("rule") not in ("skill_not_in_description", "high_hallucination_rate")
        ]

        for col in skill_cols:
            try:
                skills = json.loads(str(row[col]))
                if not isinstance(skills, list):
                    continue
            except (json.JSONDecodeError, TypeError):
                continue

            for skill in skills:
                if not isinstance(skill, str):
                    continue
                if not _skill_in_description(skill, text_lower):
                    new_flags.append({
                        "field": col,
                        "rule": "skill_not_in_description",
                        "severity": "warning",
                        "message": f"'{skill}' not found in description (post-clean check)",
                        "context": {"skill": skill},
                    })
                    n_flagged += 1

        all_flags = kept_flags + new_flags
        if all_flags != existing:
            df.at[idx, "validation_flags"] = json.dumps(all_flags, ensure_ascii=False)

    if n_flagged:
        logger.info("%d skill-not-in-description flags added (post-clean)", n_flagged)
    return df
