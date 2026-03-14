"""Post-extraction correction functions applied to extraction results before export.

These functions correct LLM output quality issues (categorical remapping, C++
hallucination, skill casing) on the extraction results JSON format
(list of dicts with a 'data' sub-dict).

AUTHORITATIVE correction pass: corrections here are applied to the JSON results
before export and before the cleaning pipeline ever runs.  The equivalent steps
in cleaning/pipeline.py act as a safety net only — they exist so that a
pre-existing enriched CSV (exported before this code existed) can be cleaned
correctly without re-running extraction.  Under normal pipeline operation,
this module runs first and the cleaning equivalents are redundant but harmless.
"""

import logging
import re
from collections import Counter
from pathlib import Path

import yaml

from extraction.preprocessing.regex_extractor import _strip_markdown_escaping

logger = logging.getLogger("pipeline.post_extraction")

_REMAP_CONFIG_PATH = (
    Path(__file__).parent.parent / "cleaning" / "config" / "job_family_remap.yaml"
)

# Bare-C regex — matches uppercase 'C' NOT followed/preceded by alphanumeric, '+', '#', '/'
# Case-sensitive: only uppercase C indicates the C programming language.
_CPP_WORD_RE = re.compile(r"(?<![a-zA-Z0-9])C(?![a-zA-Z0-9+#/])")


def _load_remap_config() -> dict:
    """Load categorical remap config from cleaning/config/job_family_remap.yaml."""
    with open(_REMAP_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_remap_categoricals(results: list[dict]) -> list[dict]:
    """Remap LLM categorical outputs to canonical forms.

    Applies job_family, contract_type, and seniority_from_title remaps
    defined in cleaning/config/job_family_remap.yaml to extraction results.

    Args:
        results: Extraction result dicts, each with 'row_id' and 'data'.

    Returns:
        Results with remapped categorical values in 'data'.
    """
    cfg = _load_remap_config()
    job_family_remap: dict[str, str] = cfg.get("remap", {})
    contract_remap: dict[str, str] = cfg.get("contract_type_remap", {})
    seniority_remap: dict[str, str] = cfg.get("seniority_remap", {})

    n_remapped = 0
    for r in results:
        data = r.get("data") or {}

        if "job_family" in data and data["job_family"] in job_family_remap:
            data["job_family"] = job_family_remap[data["job_family"]]
            n_remapped += 1

        if "contract_type" in data and data["contract_type"] in contract_remap:
            data["contract_type"] = contract_remap[data["contract_type"]]
            n_remapped += 1

        if "seniority_from_title" in data and data["seniority_from_title"] in seniority_remap:
            data["seniority_from_title"] = seniority_remap[data["seniority_from_title"]]
            n_remapped += 1

    if n_remapped:
        logger.info("Post-extraction: remapped %d categorical values", n_remapped)
    return results


def apply_fix_cpp_inference(results: list[dict], desc_by_id: dict[str, str]) -> list[dict]:
    """Correct hallucinated C++ extraction when description context doesn't support it.

    For each result, checks the job description:
    - Contains 'c++' → genuine C++, leave unchanged.
    - Contains bare 'C' (no c++ / c# / c/) → replace 'C++' → 'C' in skill lists.
    - Neither → hallucination, remove 'C++' from skill lists.

    Args:
        results: Extraction result dicts, each with 'row_id' and 'data'.
        desc_by_id: Mapping of row_id → description text.

    Returns:
        Results with corrected skill lists in 'data'.
    """
    n_fixed = 0
    for r in results:
        rid = r.get("row_id", "")
        description_original = _strip_markdown_escaping(str(desc_by_id.get(rid, "")))
        data = r.get("data") or {}

        if "c++" in description_original.lower():
            continue  # genuine C++ — leave unchanged

        has_bare_c = bool(_CPP_WORD_RE.search(description_original))
        replacement = "C" if has_bare_c else None

        for field in ("technical_skills", "nice_to_have_skills"):
            raw: list = list(data.get(field) or [])
            if not any(isinstance(s, str) and s.lower() == "c++" for s in raw):
                continue
            if replacement is None:
                data[field] = [s for s in raw if not (isinstance(s, str) and s.lower() == "c++")]
            else:
                data[field] = [
                    replacement if (isinstance(s, str) and s.lower() == "c++") else s
                    for s in raw
                ]
            n_fixed += 1

    if n_fixed:
        logger.info("Post-extraction: fixed C++ inference in %d skill lists", n_fixed)
    return results


def apply_normalize_skill_casing(results: list[dict]) -> list[dict]:
    """Normalize skill casing across all extraction results — most-frequent wins.

    Args:
        results: Extraction result dicts, each with 'row_id' and 'data'.

    Returns:
        Results with skill casing normalized in 'data'.
    """
    skill_cols = ("technical_skills", "nice_to_have_skills")

    # Count all skill occurrences across all results and columns
    skill_counter: Counter[str] = Counter()
    for r in results:
        data = r.get("data") or {}
        for col in skill_cols:
            for skill in (data.get(col) or []):
                if isinstance(skill, str):
                    skill_counter[skill] += 1

    # Build canonical map: lowercase → most-frequent casing
    canonical_map: dict[str, str] = {}
    for skill, _ in skill_counter.most_common():
        key = skill.lower()
        if key not in canonical_map:
            canonical_map[key] = skill

    n_normalized = 0
    for r in results:
        data = r.get("data") or {}
        for col in skill_cols:
            raw = list(data.get(col) or [])
            normalized = [
                canonical_map.get(s.lower(), s) if isinstance(s, str) else s
                for s in raw
            ]
            changed = sum(1 for a, b in zip(raw, normalized) if a != b)
            if changed:
                data[col] = normalized
                n_normalized += changed

    if n_normalized:
        logger.info("Post-extraction: normalized %d skill casing variants", n_normalized)
    return results
