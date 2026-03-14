"""Rule-based title normalizer: clean gender suffixes, translate German terms, fix casing.

Normalisation steps:
1. Strip gender suffix parentheticals (80+ variants)
2. Strip :in / *in / /-in gendered endings
3. Strip trailing dashes
4. Normalize whitespace
5. ALL-CAPS → Title Case
"""

import logging
import re
import time
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger("pipeline.title_normalizer")

# A parenthetical is a gender suffix if it contains:
#   - slash/pipe-separated single chars from the gender set {m,w,f,d,x,i}
#   - "all gender(s)" variants
#   - "mwd" (compact form)
#   - German "alle Geschlechter" / "alle Geschlechter"
# We use a substring search on the paren content.
# The slash/pipe pattern requires word-boundary isolation (lookbehind/lookahead) so that
# letters like d/f inside "backend/frontend" are NOT falsely detected.
# Each gender token is 1-3 chars from {m,w,f,d,x,i,p,*}; tokens separated by /|*.
_GENDER_SLASH_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"[mwfdxip*]{1,3}"
    r"(?:[/|*][mwfdxip*]{1,3})+"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_ALL_GENDERS_RE = re.compile(r"all\s+genders?|alle\s+geschlechter|\*all\s+gender", re.IGNORECASE)
_MWD_COMPACT_RE = re.compile(r"\bmwd\b", re.IGNORECASE)
# Additional gender markers — "(gn)" and comma-separated "(m,w,d)" / "(m,f,d)"
_GN_RE = re.compile(r"^gn$", re.IGNORECASE)
_GENDER_COMMA_RE = re.compile(r"^[mwfxdi*],[mwfxdi*],[mwfd]$", re.IGNORECASE)


def _is_gender_paren(content: str) -> bool:
    """Return True if parenthetical content looks like a gender marker."""
    inner = content.strip("()").strip()
    return bool(
        _GENDER_SLASH_RE.search(content)
        or _ALL_GENDERS_RE.search(content)
        or _MWD_COMPACT_RE.search(content)
        or _GN_RE.match(inner)
        or _GENDER_COMMA_RE.match(inner)
    )


def _strip_gender_parens(title: str) -> str:
    """Remove all parentheticals that are gender suffixes."""
    return re.sub(r"\([^)]*\)", lambda m: "" if _is_gender_paren(m.group()) else m.group(), title)


# Matches: Entwickler:in, Entwickler*in, Entwickler*:in, Entwickler/-in, Entwickler/in
# The suffix must be attached to a word character.
_IN_SUFFIX_RE = re.compile(r"(?<=[A-Za-z\u00c0-\u024f])[/*:]+(?:-)?in\b", re.IGNORECASE)


def _strip_in_suffix(title: str) -> str:
    """Strip gendered :in/*in//-in endings."""
    return _IN_SUFFIX_RE.sub("", title)


_TRAILING_DASH_RE = re.compile(r"[\s\u2013\u2014\-–]+$")


def _strip_trailing_dash(title: str) -> str:
    """Remove trailing dashes (ASCII and em/en dashes) and surrounding whitespace."""
    return _TRAILING_DASH_RE.sub("", title).strip()


def _normalize_whitespace(title: str) -> str:
    """Collapse consecutive whitespace into a single space and strip edges."""
    return re.sub(r"\s+", " ", title).strip()


def _fix_allcaps_casing(title: str) -> str:
    """Convert ALL-CAPS titles to Title Case. Leaves mixed-case titles untouched."""
    alpha = "".join(c for c in title if c.isalpha())
    if len(alpha) > 3 and alpha == alpha.upper():
        return title.title()
    return title


def normalize_title(title: str) -> str:
    """Apply all rule-based cleanup steps to a single title string.

    Steps (in order):
    1. Strip gender suffix parentheticals
    2. Strip :in/*in gendered endings
    3. Strip trailing dashes
    4. Normalize whitespace
    5. ALL-CAPS → Title Case

    Args:
        title: Raw job title string.

    Returns:
        Cleaned title string. Returns empty string unchanged if input is empty.
    """
    if not title or not title.strip():
        return title

    result = _strip_gender_parens(title)
    result = _strip_in_suffix(result)
    result = _strip_trailing_dash(result)
    result = _normalize_whitespace(result)
    result = _fix_allcaps_casing(result)
    return result


def load_title_translations(config_path: Path | None = None) -> list[tuple[re.Pattern, str]]:
    """Load German→English title term mappings from YAML, sorted longest-first.

    Returns:
        List of (compiled regex pattern, replacement) tuples, ready for substitution.
    """
    path = config_path or Path(__file__).parent.parent / "config" / "title_translations.yaml"
    with open(path, encoding="utf-8") as f:
        raw: dict[str, str] = yaml.safe_load(f)

    # Sort by source length descending so "Softwareentwickler" is tried before "Entwickler"
    pairs = sorted(raw.items(), key=lambda kv: len(kv[0]), reverse=True)
    compiled = []
    for german, english in pairs:
        pattern = re.compile(r"\b" + re.escape(german) + r"\b", re.IGNORECASE)
        compiled.append((pattern, english))
    return compiled


def translate_title(title: str, translations: list[tuple[re.Pattern, str]]) -> str:
    """Apply German→English term substitutions with word-boundary awareness.

    Args:
        title: Title string (should already be normalized).
        translations: Compiled translation patterns from load_title_translations().

    Returns:
        Title with German terms replaced by English equivalents.
    """
    result = title
    for pattern, replacement in translations:
        result = pattern.sub(replacement, result)
    return result


def normalize_all_titles(
    df,  # pd.DataFrame
    translations: list[tuple[re.Pattern, str]],
    checkpoint,
    reports_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    """Apply title normalization to every row and add title columns.

    Adds:
        title_original  — copy of raw title (unchanged)
        title_cleaned   — after all rule-based cleanup + translation

    Args:
        df: DataFrame with a 'title' column.
        translations: Compiled translation patterns.
        checkpoint: Checkpoint instance (not advanced here — wait for LLM step).
        reports_dir: Directory to write title_normalization_report.json.

    Returns:
        Tuple of (annotated DataFrame, report dict).
    """

    logger.info("=== STAGE: Title Normalization ===")
    start = time.monotonic()

    originals = df["title"].tolist()
    cleaned = []
    gender_stripped = 0
    in_stripped = 0
    translated = 0

    for raw in originals:
        raw_str = str(raw) if raw is not None else ""

        # Track changes for stats
        after_gender = _strip_gender_parens(raw_str)
        if after_gender != raw_str:
            gender_stripped += 1

        after_in = _strip_in_suffix(after_gender)
        if after_in != after_gender:
            in_stripped += 1

        normalized = _strip_trailing_dash(after_in)
        normalized = _normalize_whitespace(normalized)
        normalized = _fix_allcaps_casing(normalized)

        translated_title = translate_title(normalized, translations)
        if translated_title != normalized:
            translated += 1

        cleaned.append(translated_title)

    df = df.copy()
    df["title_original"] = originals
    df["title_cleaned"] = cleaned

    unique_before = len(set(originals))
    unique_after = len(set(cleaned))

    logger.info("Title normalization: %d rows processed", len(df))
    logger.info("  Gender suffixes stripped from: %d titles", gender_stripped)
    logger.info("  :in/*in suffixes stripped from: %d titles", in_stripped)
    logger.info("  German→English translations applied: %d titles", translated)
    logger.info(
        "  Unique titles: %d → %d (%d collapsed)",
        unique_before,
        unique_after,
        unique_before - unique_after,
    )

    report = {
        "total_rows": len(df),
        "gender_suffixes_stripped": gender_stripped,
        "in_suffixes_stripped": in_stripped,
        "translations_applied": translated,
        "unique_titles_before": unique_before,
        "unique_titles_after": unique_after,
        "unique_titles_collapsed": unique_before - unique_after,
    }

    import json as _json  # noqa: PLC0415

    report_path = reports_dir / "title_normalization_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        _json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Title normalization report saved to %s", report_path)

    elapsed = time.monotonic() - start
    logger.info("Stage Title Normalization complete: %d rows, %.1fs", len(df), elapsed)
    return df, report
