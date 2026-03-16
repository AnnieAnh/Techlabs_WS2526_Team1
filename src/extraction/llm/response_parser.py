"""LLM response parser: raw text → validated dict.

Tries 4 strategies in order (first success wins):
  1. Direct json.loads
  2. Strip markdown fences → json.loads
  3. Fix trailing commas → json.loads
  4. Extract between first { and last } → json.loads

Post-parse:
  - Fix known key typos (e.g. ``nice_to_h_have_skills`` → ``nice_to_have_skills``)
  - Validate against output_schema.json via jsonschema
  - Coerce fixable type errors:
      skills string → single-item array
      evidence arrays normalised to [{name, source}] dicts
      tasks > max_tasks → truncated to max_tasks (default 7)
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator

logger = logging.getLogger("pipeline.response_parser")

# Fields whose schema violations cause a hard parse failure (data=None, success=False).
# Other schema errors are demoted to warnings so minor LLM formatting issues don't
# discard otherwise-valid extractions.
# Unknown families are handled by the remap in cleaning/config/job_family_remap.yaml
# instead of causing a hard parse failure.
_CRITICAL_FIELDS: frozenset[str] = frozenset({"technical_skills"})

# Common LLM key typos → correct key. Applied before schema validation so that
# otherwise-valid data isn't rejected due to a misspelled field name.
_KEY_TYPOS: dict[str, str] = {
    "nice_to_h_have_skills": "nice_to_have_skills",
    "requred_skills": "required_skills",
    "requried_skills": "required_skills",
    "benifits": "benefits",
    "seniority_leve": "seniority_level",
    "techincal_skills": "technical_skills",
    "tehnical_skills": "technical_skills",
}


def _fix_known_typos(data: dict) -> dict:
    """Correct known LLM key typos in the parsed dict."""
    return {_KEY_TYPOS.get(k, k): v for k, v in data.items()}

STRATEGY_DIRECT = "direct"
STRATEGY_STRIP_FENCES = "strip_fences"
STRATEGY_FIX_COMMAS = "fix_commas"
STRATEGY_EXTRACT_BRACES = "extract_braces"
STRATEGY_FAILED = "failed"


@dataclass
class ListTruncation:
    """Record of a list field that was truncated during coercion."""

    field: str
    original_count: int
    kept_count: int


@dataclass
class ParseResult:
    """Outcome of parsing a single LLM response."""

    data: dict[str, Any] | None
    raw_text: str
    parse_strategy: str
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    list_truncations: list[ListTruncation] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.data is not None and self.error is None


_DEFAULT_SCHEMA_PATH = Path(__file__).parent.parent / "config" / "output_schema.json"


def load_output_schema(schema_path: Path | None = None) -> dict:
    """Load config/output_schema.json."""
    path = schema_path or _DEFAULT_SCHEMA_PATH
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _try_direct(text: str) -> dict | None:
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def _try_strip_fences(text: str) -> dict | None:
    """Strip ``` or ```json fences and try again."""
    stripped = re.sub(r"^```(?:json)?\s*\n?", "", text.strip(), flags=re.IGNORECASE)
    stripped = re.sub(r"\n?```\s*$", "", stripped)
    try:
        result = json.loads(stripped)
        return result if isinstance(result, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def _try_fix_trailing_commas(text: str) -> dict | None:
    """Remove trailing commas before ] or } and try again."""
    fixed = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        result = json.loads(fixed)
        return result if isinstance(result, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def _try_extract_braces(text: str) -> dict | None:
    """Extract substring between first { and last } and try again."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    fragment = text[start : end + 1]
    # Also fix trailing commas in the fragment
    fragment = re.sub(r",\s*([}\]])", r"\1", fragment)
    try:
        result = json.loads(fragment)
        return result if isinstance(result, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def _coerce_german_number(value: Any) -> int | None:
    """Convert German-formatted number string to int.

    50.000 → 50000 (German thousands separator)
    50,000 → 50000 (English format also handled)
    "85000" → 85000

    NOTE: This function is intentionally not called from _coerce_data() because
    salary (salary_min / salary_max) is currently Tier 1 — extracted by
    regex_extractor.py, not by the LLM.  Wire it into _coerce_data() for
    salary_min/salary_max if salary ever moves back to Tier 2 LLM extraction.
    """
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        # If LLM returned 50.0 (misread 50.000), multiply — but we can't reliably
        # distinguish 50.0 (fifty) from 50000.0 (fifty thousand). Flag as warning.
        return int(value)
    if isinstance(value, str):
        # Remove currency symbols, spaces
        cleaned = re.sub(r"[€$£\s]", "", value)
        # Remove German thousands separator (dot before 3 digits)
        cleaned = re.sub(r"\.(?=\d{3}(?:\.|$))", "", cleaned)
        # Remove English thousands separator (comma before 3 digits)
        cleaned = re.sub(r",(?=\d{3}(?:,|$))", "", cleaned)
        # Remove trailing .00 or ,00
        cleaned = re.sub(r"[.,]00$", "", cleaned)
        try:
            return int(float(cleaned))
        except (ValueError, TypeError):
            return None
    return None


def _normalize_evidence_array(
    items: list,
    field_name: str,
    warnings: list[str],
) -> list[dict[str, str]]:
    """Convert mixed string/object array to uniform evidence objects.

    Handles three input formats:
      - Plain string: "Python" → {"name": "Python", "source": ""}
      - Evidence object: {"name": "Python", "source": "Python-Entwicklung"} → kept as-is
      - Malformed items: skipped with a warning

    Args:
        items: Raw list from LLM output (may contain strings, dicts, or mixed).
        field_name: Field name for warning messages.
        warnings: Mutable list to append warning strings.

    Returns:
        List of uniform {"name": str, "source": str} dicts.
    """
    normalized: list[dict[str, str]] = []
    for item in items:
        if isinstance(item, str):
            if item.strip():
                normalized.append({"name": item.strip(), "source": ""})
        elif isinstance(item, dict) and "name" in item:
            name = str(item["name"]).strip()
            if name:
                normalized.append({
                    "name": name,
                    "source": str(item.get("source", "")).strip(),
                })
        else:
            warnings.append(f"{field_name}: skipped malformed item {item!r}")
    return normalized


def _coerce_data(
    data: dict,
    warnings: list[str],
    max_tasks: int = 7,
) -> tuple[dict, list[ListTruncation]]:
    """Apply type coercions to fix common LLM output errors (Tier 2 fields only).

    Args:
        data: Parsed LLM output dict.
        warnings: Mutable list to append warning strings.
        max_tasks: Maximum number of tasks to keep (from settings.yaml).

    Returns:
        Tuple of (coerced data dict, list of ListTruncation records).
    """
    truncations: list[ListTruncation] = []

    # Soft skills: string → single-item array (flat strings, no evidence)
    val = data.get("soft_skills")
    if isinstance(val, str):
        warnings.append("soft_skills: string value coerced to array")
        data["soft_skills"] = [val] if val else []

    # Evidence fields: normalize to uniform [{"name": ..., "source": ...}] format
    for field_name in ("technical_skills", "nice_to_have_skills", "benefits", "tasks"):
        val = data.get(field_name)
        if isinstance(val, str):
            warnings.append(f"{field_name}: string value coerced to evidence array")
            data[field_name] = [{"name": val, "source": ""}] if val else []
        elif isinstance(val, list):
            data[field_name] = _normalize_evidence_array(val, field_name, warnings)

    # Tasks: truncate to max_tasks
    tasks = data.get("tasks")
    if isinstance(tasks, list) and len(tasks) > max_tasks:
        original_count = len(tasks)
        warnings.append(f"tasks: truncated from {original_count} to {max_tasks}")
        data["tasks"] = tasks[:max_tasks]
        truncations.append(ListTruncation(
            field="tasks",
            original_count=original_count,
            kept_count=max_tasks,
        ))

    return data, truncations


def _validate_schema(data: dict, schema: dict, warnings: list[str]) -> list[str]:
    """Validate data against JSON schema. Returns ALL validation error messages.

    Uses Draft7Validator.iter_errors() to collect every error — not just the first.
    This ensures critical-field checks are not masked by an earlier non-critical error.
    """
    validator = Draft7Validator(schema)
    errors = []
    for error in validator.iter_errors(data):
        path_str = " → ".join(str(p) for p in error.absolute_path)
        errors.append(f"Schema validation: {error.message} (path: {path_str})")
    return errors


def parse_response(
    raw_text: str,
    schema: dict,
    row_id: str = "",
    max_tasks: int = 7,
) -> ParseResult:
    """Parse a raw LLM response into a validated extraction dict.

    Tries 4 strategies in order. On success, coerces types and validates schema.

    Args:
        raw_text: Raw text from the LLM response.
        schema: Loaded JSON schema dict (from load_output_schema()).
        row_id: Optional row identifier for logging.
        max_tasks: Maximum number of tasks to keep (from settings.yaml).

    Returns:
        ParseResult with data, strategy used, warnings, and any error.
    """
    text = raw_text.strip() if raw_text else ""
    prefix = f"[{row_id}] " if row_id else ""

    if not text:
        return ParseResult(
            data=None,
            raw_text=raw_text,
            parse_strategy=STRATEGY_FAILED,
            error=f"{prefix}Empty response",
        )

    # Guard: responses < 30 chars cannot be valid JSON extraction output
    if len(text) < 30:
        logger.warning("%sResponse too short (%d chars): %r", prefix, len(text), text)
        return ParseResult(
            data=None,
            raw_text=raw_text,
            parse_strategy=STRATEGY_FAILED,
            error=f"{prefix}Response too short ({len(text)} chars): {text!r}",
        )

    # Strategy 1: direct
    data = _try_direct(text)
    strategy = STRATEGY_DIRECT

    # Strategy 2: strip fences
    if data is None:
        data = _try_strip_fences(text)
        if data is not None:
            strategy = STRATEGY_STRIP_FENCES
            logger.debug("%sNon-direct parse: strip_fences", prefix)

    # Strategy 3: fix trailing commas
    if data is None:
        data = _try_fix_trailing_commas(text)
        if data is not None:
            strategy = STRATEGY_FIX_COMMAS
            logger.warning("%sNon-direct parse: fix_commas", prefix)

    # Strategy 4: extract braces
    if data is None:
        data = _try_extract_braces(text)
        if data is not None:
            strategy = STRATEGY_EXTRACT_BRACES
            logger.warning("%sNon-direct parse: extract_braces", prefix)

    # All strategies exhausted — fail
    if data is None:
        logger.error("%sParse failed — all strategies exhausted. Raw: %s...", prefix, text[:200])
        return ParseResult(
            data=None,
            raw_text=raw_text,
            parse_strategy=STRATEGY_FAILED,
            error=f"{prefix}All parse strategies failed",
        )

    logger.debug("%sParsed via %s", prefix, strategy)

    # Fix known LLM key typos before validation
    data = _fix_known_typos(data)

    # Coerce types
    warnings: list[str] = []
    data, list_truncations = _coerce_data(data, warnings, max_tasks=max_tasks)

    # Schema validation — collect ALL errors, then split critical vs non-critical
    schema_errors = _validate_schema(data, schema, warnings)
    critical_errors = [e for e in schema_errors if any(f in e for f in _CRITICAL_FIELDS)]
    non_critical = [e for e in schema_errors if e not in critical_errors]

    if critical_errors:
        logger.error("%sSchema critical failure: %s", prefix, critical_errors)
        return ParseResult(
            data=None,
            raw_text=raw_text,
            parse_strategy=STRATEGY_FAILED,
            error=f"{prefix}Schema critical error: {critical_errors[0]}",
        )

    for err in non_critical:
        logger.warning("%s%s", prefix, err)
    warnings.extend(non_critical)

    return ParseResult(
        data=data,
        raw_text=raw_text,
        parse_strategy=strategy,
        warnings=warnings,
        list_truncations=list_truncations,
    )


def log_parse_summary(results: list[ParseResult]) -> None:
    """Log strategy distribution for a batch of parse results."""
    total = len(results)
    if total == 0:
        return

    counts: dict[str, int] = {}
    for r in results:
        counts[r.parse_strategy] = counts.get(r.parse_strategy, 0) + 1

    failed = counts.get(STRATEGY_FAILED, 0)
    logger.info(
        "Parse summary: %d total, %d failed (%.1f%%)",
        total, failed, failed / total * 100,
    )
    for strategy, count in sorted(counts.items(), key=lambda x: -x[1]):
        logger.info("  %-20s : %5d  (%.1f%%)", strategy, count, count / total * 100)
