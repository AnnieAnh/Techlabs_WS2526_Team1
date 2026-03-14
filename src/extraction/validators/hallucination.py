"""Skill hallucination checker — verifies extracted items are grounded in the description.

Uses evidence-based verification: checks that the LLM-provided source quote
exists in the description. Falls back to word-boundary name matching when
the source field is empty.
"""

import logging
import re
from pathlib import Path
from typing import Any

import yaml

from extraction.validators import ValidationFlag

logger = logging.getLogger("pipeline.validators.hallucination")

_DEFAULT_THRESHOLD = 0.3  # Flag row if > 30% of its skills are unverified

# Known skill variant spellings — loaded once from shared skill_variants.yaml.
# Key = canonical name, value = lowercase plain-substring alternatives.
_VARIANTS_PATH = Path(__file__).parent.parent / "config" / "skill_variants.yaml"


def _load_skill_variants() -> dict[str, list[str]]:
    with open(_VARIANTS_PATH, encoding="utf-8") as _f:
        return yaml.safe_load(_f)["variants"]


SKILL_VARIANTS: dict[str, list[str]] = _load_skill_variants()


# ---------------------------------------------------------------------------
# Word-boundary matching (used as fallback when source is empty)
# ---------------------------------------------------------------------------

def _word_match(term: str, text_lower: str) -> bool:
    """Return True if term appears in text_lower as a whole word (not a substring)."""
    escaped = re.escape(term.lower())
    pattern = r"(?<![a-zA-Z0-9])" + escaped + r"(?![a-zA-Z0-9])"
    return bool(re.search(pattern, text_lower))


def _word_match_v2(term: str, text_lower: str, variants: dict[str, list[str]]) -> bool:
    """Return True if term or any known variant appears in text_lower.

    Search order:
    1. Word-boundary match of the original term.
    2. Each variant string as a substring in text_lower.
    3. Dehyphenated form of the term as a substring.

    Args:
        term: Canonical skill name to search for.
        text_lower: Lowercased description text.
        variants: SKILL_VARIANTS dict mapping canonical → list of lowercase alternatives.

    Returns:
        True if any form of the skill is found.
    """
    if _word_match(term, text_lower):
        return True

    term_lower = term.lower()
    for canonical, variant_list in variants.items():
        if canonical.lower() == term_lower or term_lower in variant_list:
            for variant in variant_list:
                if variant in text_lower:
                    return True
            break

    # Try dehyphenated form (e.g. "scikit-learn" → "scikit learn")
    dehyphenated = term_lower.replace("-", " ")
    if dehyphenated != term_lower and dehyphenated in text_lower:
        return True

    return False


def verify_skill_in_description(
    skill: str,
    description: str,
    aliases: dict[str, str],
) -> bool:
    """Return True if the skill (or any known alias/variant) appears in the description.

    Uses word-boundary matching to avoid false positives like 'Java' inside 'JavaScript'.
    Also checks SKILL_VARIANTS for common alternate spellings (REST APIs, scikit-learn, etc.).

    Args:
        skill: Canonical skill name to verify (e.g. 'Kubernetes').
        description: Original job description text.
        aliases: Alias→canonical mapping (lowercased keys, as loaded by load_skill_aliases).

    Returns:
        True if skill is verifiable in the description.
    """
    text_lower = description.lower()

    # Check the canonical name itself (with variants)
    if _word_match_v2(skill, text_lower, SKILL_VARIANTS):
        return True

    # Check all aliases that point to this canonical skill
    skill_lower = skill.lower()
    for alias_key, canonical in aliases.items():
        if canonical.lower() == skill_lower and alias_key != skill_lower:
            if _word_match_v2(alias_key, text_lower, SKILL_VARIANTS):
                return True

    return False


# ---------------------------------------------------------------------------
# Evidence-based source verification
# ---------------------------------------------------------------------------

def _normalize_whitespace(text: str) -> str:
    """Collapse all whitespace sequences to single spaces and strip."""
    return re.sub(r"\s+", " ", text).strip()


def verify_source_in_description(source: str, description: str) -> bool:
    """Check if the source phrase from an evidence item exists in the description.

    Uses a three-tier matching strategy:
      1. Case-insensitive substring match (handles 95%+ of cases).
      2. Whitespace-normalized substring (handles line breaks in source quotes).
      3. Token overlap — tokenize both, require >=80% of source tokens found
         within the description (handles minor LLM rephrasing of quotes).

    Args:
        source: The source phrase provided by the LLM.
        description: Original job description text.

    Returns:
        True if the source phrase is grounded in the description.
    """
    if not source or not description:
        return False

    source_lower = source.lower()
    desc_lower = description.lower()

    # Tier 1: exact substring (case-insensitive)
    if source_lower in desc_lower:
        return True

    # Tier 2: whitespace-normalized substring
    source_norm = _normalize_whitespace(source_lower)
    desc_norm = _normalize_whitespace(desc_lower)
    if source_norm in desc_norm:
        return True

    # Tier 3: token overlap (>=80% of source tokens found in description)
    source_tokens = set(re.findall(r"\w+", source_norm))
    if not source_tokens:
        return False
    desc_tokens = set(re.findall(r"\w+", desc_norm))
    matched = source_tokens & desc_tokens
    overlap = len(matched) / len(source_tokens)
    return overlap >= 0.8


def verify_evidence_item(
    item: dict[str, str],
    description: str,
    aliases: dict[str, str],
) -> tuple[bool, str]:
    """Verify a single evidence item is grounded in the description.

    If the item has a non-empty source, checks that the source phrase exists
    in the description. Otherwise falls back to word-boundary matching
    on the item name.

    Args:
        item: Evidence dict with "name" and "source" keys.
        description: Original job description text.
        aliases: Alias→canonical mapping.

    Returns:
        Tuple of (grounded: bool, method: str) where method is one of
        "source_verified", "name_verified", "unverified".
    """
    source = item.get("source", "").strip()

    # Primary: verify the source quote
    if source:
        if verify_source_in_description(source, description):
            return True, "source_verified"
        # Source provided but not found — still try name as fallback
        # (LLM might have slightly mangled the quote but the skill is real)
        if verify_skill_in_description(item["name"], description, aliases):
            return True, "name_verified"
        return False, "unverified"

    # No source provided — fall back to name matching
    if verify_skill_in_description(item["name"], description, aliases):
        return True, "name_verified"
    return False, "unverified"


def verify_row_evidence(
    row_id: str,
    items: list[dict[str, str]],
    description: str,
    aliases: dict[str, str],
    threshold: float = _DEFAULT_THRESHOLD,
    field: str = "technical_skills",
) -> list[ValidationFlag]:
    """Verify that each evidence item in a row is grounded in the description.

    Args:
        row_id: Row identifier.
        items: List of evidence dicts [{"name": ..., "source": ...}].
        description: Original job description.
        aliases: Alias→canonical mapping.
        threshold: Fraction of unverified items above which the whole row is flagged.
        field: The field name (e.g. 'technical_skills'). Used in flag metadata.

    Returns:
        List of ValidationFlags — one per unverified item, plus an extra row-level
        flag if the unverified fraction exceeds the threshold.
    """
    flags: list[ValidationFlag] = []
    if not items:
        return flags

    unverified: list[dict[str, str]] = []
    for item in items:
        grounded, method = verify_evidence_item(item, description, aliases)
        if not grounded:
            unverified.append(item)
            flags.append(ValidationFlag(
                row_id=row_id,
                field=field,
                rule="skill_not_in_description",
                severity="warning",
                message=(
                    f"'{item['name']}' not grounded in description"
                    f" (source: '{item.get('source', '')}')"
                ),
                context={"skill": item["name"], "source": item.get("source", "")},
            ))

    if unverified and len(unverified) / len(items) > threshold:
        flags.append(ValidationFlag(
            row_id=row_id,
            field=field,
            rule="high_hallucination_rate",
            severity="error",
            message=(
                f"{len(unverified)}/{len(items)} items "
                f"({len(unverified) / len(items) * 100:.0f}%) not grounded in description"
            ),
        ))

    return flags


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verify_all_skills(
    results: list[dict[str, Any]],
    desc_by_id: dict[str, str],
    aliases: dict[str, str],
    threshold: float = _DEFAULT_THRESHOLD,
) -> list[ValidationFlag]:
    """Verify skills for all extraction results.

    Expects evidence format: [{"name": ..., "source": ...}].

    Args:
        results: Extraction result dicts (each with 'row_id' and 'data').
        desc_by_id: Mapping from row_id → original description text.
        aliases: Alias→canonical mapping.
        threshold: Hallucination rate threshold (per row).

    Returns:
        All ValidationFlags produced across all rows.
    """
    logger.info("=== Skill Verification ===")
    all_flags: list[ValidationFlag] = []
    total_skills = 0
    unverified_total = 0
    high_hallucination_rows = 0

    for row in results:
        row_id = row.get("row_id", "unknown")
        data = row.get("data", {})
        description = desc_by_id.get(row_id, "")

        req = list(data.get("technical_skills") or [])
        nice = list(data.get("nice_to_have_skills") or [])

        if not (req or nice) or not description:
            continue

        flags = verify_row_evidence(
            row_id, req, description, aliases, threshold, field="technical_skills"
        ) + verify_row_evidence(
            row_id, nice, description, aliases, threshold, field="nice_to_have_skills"
        )

        all_flags.extend(flags)

        total_skills += len(req) + len(nice)
        row_unverified = sum(1 for f in flags if f.rule == "skill_not_in_description")
        unverified_total += row_unverified
        if any(f.rule == "high_hallucination_rate" for f in flags):
            high_hallucination_rows += 1

    verified_pct = (1 - unverified_total / max(1, total_skills)) * 100
    logger.info(
        "Skill verification: %.1f%% verified (%d/%d total). "
        "Unverified across %d rows. High-hallucination rows: %d.",
        verified_pct,
        total_skills - unverified_total,
        total_skills,
        len({f.row_id for f in all_flags if f.rule == "skill_not_in_description"}),
        high_hallucination_rows,
    )
    return all_flags
