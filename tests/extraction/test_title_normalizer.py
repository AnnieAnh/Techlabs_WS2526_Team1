"""Tests for extraction/preprocessing/title_normalizer.py."""

import pytest

from extraction.preprocessing.title_normalizer import (
    _fix_allcaps_casing,
    _is_gender_paren,
    _strip_gender_parens,
    _strip_in_suffix,
    _strip_trailing_dash,
    load_title_translations,
    normalize_title,
    translate_title,
)


@pytest.mark.parametrize("content, expected", [
    # --- Should be detected as gender ---
    ("(m/w/d)", True),
    ("(w/m/d)", True),
    ("(m/f/d)", True),
    ("(f/m/d)", True),
    ("(d/m/w)", True),
    ("(d/f/m)", True),
    ("(x/w/m)", True),
    ("(w/*/m)", True),
    ("(m/w/x)", True),
    ("(m/f/x)", True),
    ("(f/m/x)", True),
    ("(m/w)", True),
    ("(f/m)", True),
    ("(m/f)", True),
    ("(m/w/div.)", True),
    ("(m/f/div)", True),
    ("(m/w/divers)", True),
    ("(m/f/diverse)", True),
    ("(m|w|d)", True),
    ("(w|m|d)", True),
    ("(m|w|x)", True),
    ("(mwd)", True),
    ("(MWD)", True),
    ("(all genders)", True),
    ("(all gender)", True),
    ("(All Genders)", True),
    ("(all genders welcome)", True),
    ("(all genders!)", True),
    ("( all genders)", True),
    ("(*all gender)", True),
    ("(alle Geschlechter)", True),
    # Additional gender markers
    ("(gn)", True),
    ("(GN)", True),
    ("(m,w,d)", True),
    ("(m,f,d)", True),
    # With extra info — still gender
    ("(m/f/d - remote)", True),
    ("(m/f/d | remote in sachsen)", True),
    ("(w/m/d, befristet auf 2 jahre)", True),
    ("(full time, w/m/d)", True),
    ("(remote, f/m/d)", True),
    ("(m/w 100%)", True),
    ("(m/w, 80-100%)", True),
    # --- Should NOT be detected as gender ---
    ("(Remote)", False),
    ("(Senior)", False),
    ("(Junior)", False),
    ("(Go / TypeScript)", False),
    ("(PHP + React)", False),
    ("(Linux/C/SQL)", False),
    ("(Automotive Consultant)", False),
    ("(backend/frontend/fullstack)", False),
])
def test_is_gender_paren(content, expected):
    assert _is_gender_paren(content) == expected


@pytest.mark.parametrize("raw, expected", [
    # Standard German suffixes
    ("Software Engineer (m/w/d)", "Software Engineer "),
    ("Backend Developer (w/m/d)", "Backend Developer "),
    ("Data Analyst (m/f/d)", "Data Analyst "),
    ("Cloud Engineer (all genders)", "Cloud Engineer "),
    ("DevOps Engineer (All Genders)", "DevOps Engineer "),
    ("IT Consultant (all genders welcome)", "IT Consultant "),
    ("QA Engineer (mwd)", "QA Engineer "),
    ("Network Engineer (m|w|d)", "Network Engineer "),

    # Mixed case suffixes
    ("Product Manager (M/W/D)", "Product Manager "),
    ("SENIOR SOFTWARE QUALITY ASSURANCE ENGINEER (M/W/D)", "SENIOR SOFTWARE QUALITY ASSURANCE ENGINEER "),

    # With extra info in gender paren
    ("Java Developer (m/f/d - remote)", "Java Developer "),
    ("Engineer (w/m/d, befristet auf 2 jahre)", "Engineer "),

    # Prefix (Senior) must NOT be stripped
    ("(Senior) Backend Developer (m/w/d)", "(Senior) Backend Developer "),
    ("(Junior) System Engineer (all genders)", "(Junior) System Engineer "),

    # Multiple gender parens
    ("Developer (m/w/d) (all genders)", "Developer  "),

    # No gender suffix — unchanged
    ("Senior Software Engineer", "Senior Software Engineer"),
    ("Cloud Engineer (Remote)", "Cloud Engineer (Remote)"),
    ("Developer (Go / TypeScript)", "Developer (Go / TypeScript)"),

    # Empty string — unchanged
    ("", ""),
])
def test_strip_gender_parens(raw, expected):
    assert _strip_gender_parens(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("Entwickler:in", "Entwickler"),
    ("Entwicklerin*", "Entwicklerin*"),  # *in at end with no word break is different
    ("Software Developer:in", "Software Developer"),
    ("Systemadministrator:in", "Systemadministrator"),
    ("TYPO3-Entwickler/-in", "TYPO3-Entwickler"),
    ("Softwareentwickler*in", "Softwareentwickler"),
    ("Projektleiter*:in", "Projektleiter"),
    # No suffix — unchanged
    ("Backend Developer", "Backend Developer"),
    ("Senior Engineer", "Senior Engineer"),
])
def test_strip_in_suffix(raw, expected):
    assert _strip_in_suffix(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("Senior Developer –", "Senior Developer"),
    ("Backend Engineer -", "Backend Engineer"),
    ("Cloud Engineer   –  ", "Cloud Engineer"),
    ("IT Admin -  ", "IT Admin"),
    # Not at end — unchanged
    ("Developer – Backend", "Developer – Backend"),
    ("Engineer - Java / Angular", "Engineer - Java / Angular"),
    # Clean title — unchanged
    ("Software Engineer", "Software Engineer"),
])
def test_strip_trailing_dash(raw, expected):
    assert _strip_trailing_dash(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("BACKEND DEVELOPER", "Backend Developer"),
    ("SENIOR SOFTWARE ENGINEER", "Senior Software Engineer"),
    ("IT ADMINISTRATOR", "It Administrator"),   # acronym — acceptable
    ("FIELD APPS ENGINEER", "Field Apps Engineer"),
    # Mixed case — leave unchanged
    ("Senior Backend Developer", "Senior Backend Developer"),
    ("Backend Developer", "Backend Developer"),
    ("SAP Consultant", "SAP Consultant"),   # mixed-case — left unchanged
    # Edge: short strings
    ("AB", "AB"),  # ≤3 alpha chars — not converted
])
def test_allcaps_to_title(raw, expected):
    assert _fix_allcaps_casing(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    # Gender suffix removed, whitespace cleaned
    ("Software Engineer (m/w/d)", "Software Engineer"),
    ("Backend Developer (w/m/d) –", "Backend Developer"),
    ("(Senior) AWS Cloud Engineer (m/w/d)", "(Senior) AWS Cloud Engineer"),
    # :in stripping
    ("Systemadministrator:in", "Systemadministrator"),
    # ALL-CAPS conversion
    ("BACKEND DEVELOPER", "Backend Developer"),
    # Already clean — unchanged
    ("Senior Software Engineer", "Senior Software Engineer"),
    # (Remote) NOT stripped
    ("Cloud Engineer (Remote)", "Cloud Engineer (Remote)"),
    # (Go / TypeScript) NOT stripped
    ("Backend Developer (Go / TypeScript)", "Backend Developer (Go / TypeScript)"),
    # (gn) and (m,w,d) / (m,f,d) stripped
    ("Software Engineer (gn)", "Software Engineer"),
    ("Backend Developer (m,w,d)", "Backend Developer"),
    ("Data Analyst (m,f,d)", "Data Analyst"),
    # Empty
    ("", ""),
    # Multiple spaces cleaned
    ("Senior  Developer", "Senior Developer"),
    # Trailing whitespace
    ("Engineer   ", "Engineer"),
])
def test_normalize_title(raw, expected):
    assert normalize_title(raw) == expected


@pytest.fixture(scope="module")
def translations():
    return load_title_translations()


def test_translations_loaded(translations):
    assert len(translations) > 10


def test_entwickler_to_developer(translations):
    result = translate_title("Backend Entwickler", translations)
    assert result == "Backend Developer"


def test_softwareentwickler_before_entwickler(translations):
    """Softwareentwickler must match before Entwickler (longest first)."""
    result = translate_title("Softwareentwickler", translations)
    assert result == "Software Developer"


def test_werkstudent_to_working_student(translations):
    result = translate_title("Werkstudent Full Stack Entwickler", translations)
    assert "Working Student" in result
    assert "Developer" in result


def test_english_title_unchanged(translations):
    result = translate_title("Senior Backend Developer", translations)
    assert result == "Senior Backend Developer"


def test_projektleiter_to_project_manager(translations):
    result = translate_title("Projektleiter Software", translations)
    assert "Project Manager" in result


def test_ingenieur_to_engineer(translations):
    result = translate_title("Senior Ingenieur Automotive", translations)
    assert "Engineer" in result


def test_word_boundary_respected(translations):
    """'Entwickler' inside 'Softwareentwickler' must not cause double replacement."""
    result = translate_title("Softwareentwickler", translations)
    # Should be "Software Developer", NOT "Software Developerer"
    assert result == "Software Developer"
    assert "Developerer" not in result
