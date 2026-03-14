"""Regex-based pre-extraction for deterministic job description fields.

Extracts 8 fields using German-aware patterns before the LLM call.
This reduces token usage and adds reproducibility for high-confidence fields.

Fields extracted:
  - contract_type:        Vollzeit / Teilzeit / Freelance / Contract
  - work_modality:        Remote / Hybrid / On-site
  - salary_min:           Parsed from German number format (50.000 = 50,000)
  - salary_max:           Parsed from German number format
  - experience_years:     Integer from "X Jahre Berufserfahrung" etc.
  - seniority_from_title: Senior / Junior / Lead from title keywords
  - languages:            List of {language, level} dicts from description
  - education_level:      Highest education level found, or None
"""

import re
from typing import Any

from shared.constants import (
    SALARY_MAX_CEILING as _SALARY_MAX_CEILING,
)
from shared.constants import (
    SALARY_MIN_FLOOR as _SALARY_MIN_FLOOR,
)

# ---------------------------------------------------------------------------
# Contract type patterns
# ---------------------------------------------------------------------------

_CONTRACT_TYPE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bVollzeit\b|\bfull[- ]?time\b|\bFestanstellung\b", re.IGNORECASE), "Full-time"),
    (re.compile(r"\bTeilzeit\b|\bpart[- ]?time\b", re.IGNORECASE), "Part-time"),
    (re.compile(r"\bFreelance\b|\bfreelancer\b|\bFreiberufler\b", re.IGNORECASE), "Freelance"),
    # "unbefristet" must be checked BEFORE "befristet" to avoid false matches
    (re.compile(r"\bunbefristet(?:e[rsnm]?)?\b", re.IGNORECASE), "Permanent"),
    (re.compile(
        r"\bContractor\b|\bAuftragnehmer\b|(?<!un)\bbefristet(?:e[rsnm]?)?\b",
        re.IGNORECASE,
    ), "Contract"),
    (re.compile(r"\bWerkstudent(?:in)?\b", re.IGNORECASE), "Working Student"),
    (re.compile(r"\bPraktik(?:um|ant(?:in)?)\b", re.IGNORECASE), "Internship"),
]

# ---------------------------------------------------------------------------
# Work modality patterns
# ---------------------------------------------------------------------------

_WORK_MODALITY_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Hybrid BEFORE Remote — hybrid job ads routinely mention "Home Office" as
    # a subset ("2 Tage Homeoffice"). First-match-wins causes Remote to fire
    # before Hybrid is checked without this ordering.
    (re.compile(r"\bHybrid", re.IGNORECASE), "Hybrid"),
    (re.compile(r"\bmobile[snmr]?\s+Arbeiten\b", re.IGNORECASE), "Hybrid"),
    (re.compile(r"\bflexible[rns]?\s+Arbeitsort", re.IGNORECASE), "Hybrid"),
    (re.compile(r"\bRemote\b|\bHome\s*[Oo]ffice\b|\bHomeoffice\b", re.IGNORECASE), "Remote"),
    # "Präsenzpflicht", "vor Ort", "on-site", "onsite"
    (re.compile(r"\bvor\s+Ort\b|\bon[- ]?site\b|\bOnsite\b|\bPräsenz", re.IGNORECASE), "On-site"),
]

# ---------------------------------------------------------------------------
# Salary patterns (German number format: 50.000 = 50,000)
# ---------------------------------------------------------------------------

# Salary range: "60.000 - 80.000 EUR" or "60.000–80.000€" or "70.000 und 95.000 Euro"
_SALARY_RANGE_RE = re.compile(
    r"(\d{1,3}(?:\.\d{3})*)(?:,\d{2})?\s*(?:€|EUR|Euro)?\s*"
    r"(?:bis|to|-|–|und|bis\s+zu)\s*"
    r"(\d{1,3}(?:\.\d{3})*)(?:,\d{2})?\s*(?:€|EUR|Euro)",
    re.IGNORECASE,
)

# Single salary: "ab 60.000 EUR" or "60.000 EUR"
_SALARY_SINGLE_RE = re.compile(
    r"(?:ab|bis\s+zu|from|up\s+to)?\s*(\d{1,3}(?:\.\d{3})+)(?:,\d{2})?\s*(?:€|EUR|Euro)",
    re.IGNORECASE,
)

# €K format: "€85k" or "€80K - €110K"
_SALARY_K_RANGE_RE = re.compile(
    r"€\s*(\d{2,4})\s*[kK]\s*(?:bis|to|-|–)\s*€?\s*(\d{2,4})\s*[kK]",
)
_SALARY_K_SINGLE_RE = re.compile(
    r"€\s*(\d{2,4})\s*[kK]",
)

# Monthly salary context — reject matches near per-month indicators
_MONTHLY_CONTEXT_RE = re.compile(
    r"(?:/\s*Monat|pro\s+Monat|mtl|monthly|/\s*month)",
    re.IGNORECASE,
)

# Words that indicate a number is NOT a salary (employee/customer counts etc.)
# Checked in ±50-char context around each salary match.
_NON_SALARY_CONTEXT_RE = re.compile(
    r"\b(?:Mitarbeiter|Kunden|Standorte|Bewerber|Kollegen|Beschäftigte|employees|users|customers)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Experience patterns
# ---------------------------------------------------------------------------

_EXPERIENCE_RE = re.compile(
    r"(\d+)\+?\s*(?:Jahre?|years?)\s+(?:Berufserfahrung|Erfahrung|experience)",
    re.IGNORECASE,
)
_EXPERIENCE_ALT_RE = re.compile(
    r"(?:mindestens|at\s+least|min\.?)\s*(\d+)\s*(?:Jahre?|years?)",
    re.IGNORECASE,
)
_EXPERIENCE_QUALITATIVE_RE = re.compile(
    r"\b(Mehrjährige|Langjährige|Erste|Einschlägige)\w*\s+"
    r"(?:Berufs)?(?:erfahrung|Praxiserfahrung)\b",
    re.IGNORECASE,
)
_QUALITATIVE_YEARS: dict[str, int] = {
    "mehrjährige": 3,
    "langjährige": 5,
    "erste": 1,
    "einschlägige": 3,
}

# ---------------------------------------------------------------------------
# Language patterns
# ---------------------------------------------------------------------------

_LANG_CEFR_RE = re.compile(
    r"\b(Deutsch|Englisch|Fran[cç]ösisch|German|English|French)\s+(A1|A2|B1|B2|C1|C2)\b",
    re.IGNORECASE,
)
_LANG_CEFR_REV_RE = re.compile(
    r"\b(A1|A2|B1|B2|C1|C2)\s+(Deutsch|Englisch|Fran[cç]ösisch|German|English|French)\b",
    re.IGNORECASE,
)
_LANG_FLUENCY_RE = re.compile(
    r"\b(fließend|fluent|verhandlungssicher|muttersprachlich|native)"
    r"(?:e[rsnm]?)?"  # German adjective inflection (-e, -er, -es, -en, -em)
    r"\s+(?:in\s+)?(Deutsch|Englisch|Fran[cç]ösisch|German|English|French)"
    r"(?:kenntnisse|sprachkenntnisse)?\b",
    re.IGNORECASE,
)
_LANG_KENNTNISSE_RE = re.compile(
    r"\b(Deutsch|Englisch|Fran[cç]ösisch)(kenntnisse|sprachkenntnisse)\b",
    re.IGNORECASE,
)
# Compound: "Deutsch- und Englischkenntnisse" — captures both languages
_LANG_COMPOUND_RE = re.compile(
    r"\b(Deutsch|Englisch|Fran[cç]ösisch)-?\s+und\s+"
    r"(Deutsch|Englisch|Fran[cç]ösisch)(kenntnisse|sprachkenntnisse)\b",
    re.IGNORECASE,
)
# Proficiency qualifiers: "gute Deutschkenntnisse", "sehr gute Englischkenntnisse"
_LANG_GUTE_RE = re.compile(
    r"\b(?:sehr\s+)?(?:gute|sichere|solide)[rsnm]?\s+"
    r"(Deutsch|Englisch|Fran[cç]ösisch|German|English|French)"
    r"(?:kenntnisse|sprachkenntnisse)?\b",
    re.IGNORECASE,
)

_FLUENCY_TO_CEFR: dict[str, str] = {
    "fließend": "B2+",
    "fluent": "B2+",
    "verhandlungssicher": "C1+",
    "muttersprachlich": "C2",
    "native": "C2",
}
_LANG_NORMALIZE: dict[str, str] = {
    "english": "English",
    "englisch": "English",
    "deutsch": "German",
    "german": "German",
    "französisch": "French",
    "francösisch": "French",
    "french": "French",
}

# Boilerplate patterns that mention languages without being actual requirements.
# If a language is ONLY found inside a boilerplate section, it's excluded.
_BOILERPLATE_RE = re.compile(
    r"(?:equal\s+opportunity|diversity|inclusive|"
    r"chancengleichheit|vielfalt|diskriminierung|"
    r"unabhängig\s+von|regardless\s+of|"
    r"freuen\s+uns\s+auf\s+(?:Ihre|deine|alle)\b|"
    r"welcome\s+applications?\b)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Education patterns (highest level wins)
# ---------------------------------------------------------------------------

_EDUCATION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bPromotion\b|\bPh\.?D\.?\b|\bDoktor", re.IGNORECASE), "PhD"),
    (re.compile(r"\bMaster\b|\bM\.?\s?Sc\.?\b|\bDiplom", re.IGNORECASE), "Master"),
    (re.compile(r"\bBachelor\b|\bB\.?\s?Sc\.?\b", re.IGNORECASE), "Bachelor"),
    (re.compile(r"\bStudium\b|\bHochschulabschluss\b|\bdegree\b", re.IGNORECASE), "Degree"),
    (re.compile(r"\bAusbildung\b|\bBerufsausbildung\b", re.IGNORECASE), "Vocational"),
]

# Order defines priority (first = highest level)
_EDUCATION_PRIORITY = ["PhD", "Master", "Bachelor", "Degree", "Vocational"]

# ---------------------------------------------------------------------------
# Seniority keywords (order matters — more specific first)
# ---------------------------------------------------------------------------

_SENIORITY_KEYWORDS: list[tuple[str, str]] = [
    ("Chief", "C-Level"),
    ("CTO", "C-Level"),
    ("CIO", "C-Level"),
    ("Vice President", "Director"),
    ("Director", "Director"),
    ("VP", "Director"),
    ("Head of", "Lead"),
    ("Principal", "Lead"),
    ("Staff", "Lead"),
    ("Tech Lead", "Lead"),
    ("Team Lead", "Lead"),
    ("Teamlead", "Lead"),
    ("Teamleiter", "Lead"),
    ("Lead", "Lead"),
    ("Senior", "Senior"),
    ("Junior", "Junior"),
    ("Trainee", "Junior"),
    ("Werkstudent", "Junior"),     # also matches Werkstudentin via partial \b
    ("Werkstudentin", "Junior"),
    ("Intern", "Junior"),
    ("Praktikant", "Junior"),
    ("Praktikantin", "Junior"),
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _parse_german_number(s: str) -> int | None:
    """Parse a German-format number string (50.000 → 50000)."""
    try:
        return int(s.replace(".", "").replace(",", ""))
    except ValueError:
        return None


def _in_bounds(val: int) -> bool:
    return _SALARY_MIN_FLOOR <= val <= _SALARY_MAX_CEILING


def _strip_markdown_escaping(text: str) -> str:
    r"""Remove markdown escape backslashes from text.

    Scraped descriptions often contain escaped punctuation (e.g. ``\\.`` instead
    of ``.``).  This strips one or more leading backslashes before common
    punctuation so downstream regexes can match normally.
    """
    return re.sub(r"\\+([.\-+*(){}|^$#!@&~`\[\]])", r"\1", text)


def _is_salary_context(text: str, start: int, end: int) -> bool:
    """Return True if the match context does NOT look like employee/customer counts.

    Scans 50 characters before and after the match for non-salary indicator words
    and monthly salary indicators (per-month patterns are rejected).
    """
    context = text[max(0, start - 50): end + 50]
    if _NON_SALARY_CONTEXT_RE.search(context):
        return False
    if _MONTHLY_CONTEXT_RE.search(context):
        return False
    return True


# ---------------------------------------------------------------------------
# Field extractors
# ---------------------------------------------------------------------------

def _extract_contract_type(description: str, title: str = "") -> str | None:
    """Search description first, fall back to title if no match."""
    for text in (description, title):
        for pattern, canonical in _CONTRACT_TYPE_PATTERNS:
            if pattern.search(text):
                return canonical
    return None


def _extract_work_modality(description: str, title: str = "") -> str | None:
    """Search description first, fall back to title if no match."""
    for text in (description, title):
        for pattern, canonical in _WORK_MODALITY_PATTERNS:
            if pattern.search(text):
                return canonical
    return None


def _extract_salary(description: str) -> tuple[int | None, int | None]:
    """Return (salary_min, salary_max) as ints or None.

    Tries 4 regex pattern strategies in order: German-format range, German-format
    single, euro-K range, and euro-K single.
    """
    # German-format range: "60.000 - 80.000 EUR"
    m = _SALARY_RANGE_RE.search(description)
    if m:
        lo = _parse_german_number(m.group(1))
        hi = _parse_german_number(m.group(2))
        if (lo and hi and _in_bounds(lo) and _in_bounds(hi) and lo <= hi
                and _is_salary_context(description, m.start(), m.end())):
            return lo, hi

    # German-format single: "ab 60.000 EUR"
    m = _SALARY_SINGLE_RE.search(description)
    if m:
        val = _parse_german_number(m.group(1))
        if val and _in_bounds(val) and _is_salary_context(description, m.start(), m.end()):
            return val, None

    # €K range: "€80K - €110K"
    m = _SALARY_K_RANGE_RE.search(description)
    if m:
        lo = int(m.group(1)) * 1000
        hi = int(m.group(2)) * 1000
        if _in_bounds(lo) and _in_bounds(hi) and lo <= hi:
            return lo, hi

    # €K single: "€85k"
    m = _SALARY_K_SINGLE_RE.search(description)
    if m:
        val = int(m.group(1)) * 1000
        if _in_bounds(val):
            return val, None

    return None, None


def _extract_experience(description: str) -> int | None:
    """Extract years of experience from description, with qualitative fallback."""
    for pattern in (_EXPERIENCE_RE, _EXPERIENCE_ALT_RE):
        m = pattern.search(description)
        if m:
            return int(m.group(1))
    # Fallback: qualitative terms (Mehrjährige → 3, Langjährige → 5, etc.)
    m = _EXPERIENCE_QUALITATIVE_RE.search(description)
    if m:
        return _QUALITATIVE_YEARS.get(m.group(1).lower())
    return None


def _extract_seniority_from_title(title: str) -> str | None:
    """Return the seniority level inferred from title keywords, or None."""
    for keyword, level in _SENIORITY_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", title, re.IGNORECASE):
            return level
    return None


def _is_in_boilerplate(description: str, match_start: int, context_chars: int = 200) -> bool:
    """Return True if the match position is inside a boilerplate section."""
    start = max(0, match_start - context_chars)
    end = min(len(description), match_start + context_chars)
    context = description[start:end]
    return bool(_BOILERPLATE_RE.search(context))


# CEFR level ordering for deduplication (higher index = more specific / higher level)
_LEVEL_RANK: dict[str, int] = {
    "required": 0,
    "A1": 1, "A2": 2, "B1": 3, "B2": 4, "B2+": 5,
    "C1": 6, "C1+": 7, "C2": 8,
}


def _extract_languages(description: str) -> list[dict[str, str]]:
    """Extract language requirements and CEFR levels from description text.

    Filters out language mentions found only in boilerplate/diversity sections.
    Deduplicates by language, keeping the highest specificity level.

    Returns:
        List of dicts with 'language' and 'level' keys, sorted by language name.
    """
    if not description:
        return []

    candidates: list[tuple[str, str, int]] = []  # (lang, level, match_start)

    for m in _LANG_CEFR_RE.finditer(description):
        lang = _LANG_NORMALIZE.get(m.group(1).lower(), m.group(1).title())
        candidates.append((lang, m.group(2).upper(), m.start()))
    for m in _LANG_CEFR_REV_RE.finditer(description):
        lang = _LANG_NORMALIZE.get(m.group(2).lower(), m.group(2).title())
        candidates.append((lang, m.group(1).upper(), m.start()))
    for m in _LANG_FLUENCY_RE.finditer(description):
        level = _FLUENCY_TO_CEFR.get(m.group(1).lower(), "B2+")
        lang = _LANG_NORMALIZE.get(m.group(2).lower(), m.group(2).title())
        candidates.append((lang, level, m.start()))
    for m in _LANG_KENNTNISSE_RE.finditer(description):
        lang = _LANG_NORMALIZE.get(m.group(1).lower(), m.group(1).title())
        candidates.append((lang, "required", m.start()))
    for m in _LANG_COMPOUND_RE.finditer(description):
        lang1 = _LANG_NORMALIZE.get(m.group(1).lower(), m.group(1).title())
        lang2 = _LANG_NORMALIZE.get(m.group(2).lower(), m.group(2).title())
        candidates.append((lang1, "required", m.start()))
        candidates.append((lang2, "required", m.start()))
    for m in _LANG_GUTE_RE.finditer(description):
        lang = _LANG_NORMALIZE.get(m.group(1).lower(), m.group(1).title())
        candidates.append((lang, "required", m.start()))

    # Filter out matches that are only in boilerplate context
    filtered = [
        (lang, level) for lang, level, pos in candidates
        if not _is_in_boilerplate(description, pos)
    ]

    # Deduplicate by language, keeping the highest-specificity level
    best: dict[str, str] = {}
    for lang, level in filtered:
        existing = best.get(lang)
        if existing is None or _LEVEL_RANK.get(level, 0) > _LEVEL_RANK.get(existing, 0):
            best[lang] = level

    return [{"language": lang, "level": level} for lang, level in sorted(best.items())]


def _extract_education(description: str) -> str | None:
    """Return highest education level found in description, or None."""
    found: set[str] = set()
    for pattern, level in _EDUCATION_PATTERNS:
        if pattern.search(description):
            found.add(level)
    for level in _EDUCATION_PRIORITY:
        if level in found:
            return level
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_regex_fields(description: str, title: str) -> dict[str, Any]:
    """Extract deterministic fields from description and title before LLM call.

    Uses German-aware patterns. All fields return None (or []) when not found.

    Args:
        description: Job description text.
        title: Raw job title string.

    Returns:
        Dict with keys: contract_type, work_modality, salary_min, salary_max,
        experience_years, seniority_from_title, languages, education_level.
        Scalar fields are None when not found; languages is [] when not found.
    """
    # Strip markdown escape backslashes so downstream regexes can match
    # punctuation normally (e.g. "55\\.000" → "55.000"). Title is not escaped.
    desc = _strip_markdown_escaping(description)
    salary_min, salary_max = _extract_salary(desc)
    return {
        "contract_type": _extract_contract_type(desc, title),
        "work_modality": _extract_work_modality(desc, title),
        "salary_min": salary_min,
        "salary_max": salary_max,
        "experience_years": _extract_experience(desc),
        "seniority_from_title": _extract_seniority_from_title(title),
        "languages": _extract_languages(desc),
        "education_level": _extract_education(desc),
    }
