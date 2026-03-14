"""Skills normalisation using alias map from config/skill_aliases.yaml.

Works with evidence format: [{"name": ..., "source": ...}].
"""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("pipeline.validators.skills")


def load_skill_aliases(path: Path) -> dict[str, str]:
    """Load alias→canonical mapping from YAML file.

    Args:
        path: Path to skill_aliases.yaml.

    Returns:
        Dict mapping lowercased alias to its canonical skill name.
    """
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    aliases: dict[str, str] = {}
    if raw:
        for alias, canonical in raw.items():
            aliases[str(alias).lower()] = str(canonical)
    return aliases


def _normalize_skill(skill: str, aliases: dict[str, str]) -> str:
    """Return canonical form of a skill, or the skill itself if not in aliases."""
    return aliases.get(skill.strip().lower(), skill.strip())


def normalize_skills_evidence(
    items: list[dict[str, str]],
    aliases: dict[str, str],
) -> list[dict[str, str]]:
    """Normalize aliases and deduplicate an evidence-format skill list.

    Applies alias resolution to the 'name' field while preserving 'source'.
    Deduplicates by canonical name (case-insensitive), keeping the first occurrence.

    Args:
        items: Evidence dicts with "name" and "source" keys.
        aliases: Alias→canonical mapping (lowercased keys).

    Returns:
        Sorted list of unique, canonicalized evidence items.
    """
    seen: set[str] = set()
    result: list[dict[str, str]] = []
    for item in items:
        normalized = _normalize_skill(item["name"], aliases)
        key = normalized.lower()
        if key not in seen:
            seen.add(key)
            result.append({"name": normalized, "source": item.get("source", "")})
    return sorted(result, key=lambda x: x["name"])


def normalize_and_reconcile_evidence(
    technical: list[dict[str, str]],
    nice_to_have: list[dict[str, str]],
    aliases: dict[str, str],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[str]]:
    """Normalize both evidence skill lists and resolve contradictions.

    Any skill that appears in both lists after normalization is removed from
    nice_to_have (since it's already in technical_skills).

    Args:
        technical: Required skills as evidence dicts.
        nice_to_have: Nice-to-have skills as evidence dicts.
        aliases: Alias→canonical mapping.

    Returns:
        Tuple of (normalized_technical, normalized_nice_to_have, contradiction_names).
    """
    norm_tech = normalize_skills_evidence(technical, aliases)
    norm_nice = normalize_skills_evidence(nice_to_have, aliases)

    tech_lower = {item["name"].lower() for item in norm_tech}
    contradictions = [item["name"] for item in norm_nice if item["name"].lower() in tech_lower]
    resolved_nice = [item for item in norm_nice if item["name"].lower() not in tech_lower]

    return norm_tech, resolved_nice, contradictions


def normalize_all_skills(
    results: list[dict[str, Any]],
    aliases: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Normalize skills for all extraction results, returning new dicts.

    Expects evidence format: [{"name": ..., "source": ...}].

    Args:
        results: List of extraction result dicts.
        aliases: Alias→canonical mapping.

    Returns:
        Tuple of (updated_results, stats) where stats has alias/dedup/contradiction counts.
    """
    logger.info("=== Skills Normalization ===")

    total_aliases = 0
    total_deduped = 0
    total_contradictions = 0
    updated: list[dict[str, Any]] = []

    for row in results:
        data = row.get("data", {})
        technical = list(data.get("technical_skills") or [])
        nice = list(data.get("nice_to_have_skills") or [])

        before_total = len(technical) + len(nice)

        for item in technical + nice:
            if _normalize_skill(item["name"], aliases) != item["name"].strip():
                total_aliases += 1

        norm_tech, norm_nice, contradiction_names = normalize_and_reconcile_evidence(
            technical, nice, aliases
        )

        new_row = dict(row)
        new_row["data"] = dict(data)
        new_row["data"]["technical_skills"] = norm_tech
        new_row["data"]["nice_to_have_skills"] = norm_nice

        after_total = len(norm_tech) + len(norm_nice)
        total_deduped += max(0, before_total - after_total - len(contradiction_names))
        total_contradictions += len(contradiction_names)

        # Normalize soft_skills (dedup only — no alias matching for interpersonal skills)
        soft = list(data.get("soft_skills") or [])
        if soft:
            new_row["data"]["soft_skills"] = sorted({s.strip() for s in soft if s.strip()})

        updated.append(new_row)

    stats = {
        "aliases_resolved": total_aliases,
        "deduped": total_deduped,
        "contradictions_resolved": total_contradictions,
    }
    logger.info(
        "Skills normalisation: %d rows. Aliases resolved: %d. Deduped: %d. Contradictions: %d.",
        len(results),
        stats["aliases_resolved"],
        stats["deduped"],
        stats["contradictions_resolved"],
    )
    return updated, stats
