"""Tests for extraction/preprocessing/regex_extractor.py — German-aware regex pre-extraction."""

import pytest

from extraction.preprocessing.regex_extractor import (
    _extract_contract_type,
    _extract_education,
    _extract_experience,
    _extract_languages,
    _extract_salary,
    _extract_seniority_from_title,
    _extract_work_modality,
    _strip_markdown_escaping,
    extract_regex_fields,
)

# ---------------------------------------------------------------------------
# Contract type
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("desc,expected", [
    ("Wir suchen Vollzeit Mitarbeiter", "Full-time"),
    ("This is a full-time position", "Full-time"),
    ("full time role available", "Full-time"),
    ("Teilzeit möglich, 20 Stunden", "Part-time"),
    ("part-time job opening", "Part-time"),
    ("Freelance Auftrag für 6 Monate", "Freelance"),
    ("We are looking for a Freelancer", "Freelance"),
    ("Contractor position available", "Contract"),
    ("No contract info here", None),
    ("", None),
])
def test_extract_contract_type(desc: str, expected: str | None) -> None:
    assert _extract_contract_type(desc) == expected


# ---------------------------------------------------------------------------
# Work modality
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("desc,expected", [
    ("100% Remote möglich", "Remote"),
    ("Home Office Option vorhanden", "Remote"),
    ("Homeoffice nach Absprache", "Remote"),
    ("Hybrides Arbeitsmodell", "Hybrid"),
    ("Hybrid work arrangement", "Hybrid"),
    ("Arbeit vor Ort erforderlich", "On-site"),
    ("on-site position in Berlin", "On-site"),
    ("onsite work required", "On-site"),
    ("Präsenzpflicht im Büro", "On-site"),
    ("No modality info", None),
])
def test_extract_work_modality(desc: str, expected: str | None) -> None:
    assert _extract_work_modality(desc) == expected


@pytest.mark.parametrize("desc,expected", [
    # T-02 regression: Hybrid must win when both "Hybrid" and "Home Office" appear.
    # Previously Remote fired first because it was listed before Hybrid.
    ("Hybrides Arbeitsmodell, 2 Tage Home Office möglich", "Hybrid"),
    ("hybrid work environment with remote option available", "Hybrid"),
    ("Hybrid: 3 Tage Präsenz, 2 Tage remote", "Hybrid"),
    # Pure Remote still classified correctly
    ("100% Remote, Homeoffice möglich", "Remote"),
    ("Full remote position from anywhere in Germany", "Remote"),
    # Pure On-site
    ("Arbeit vor Ort in unserem Berliner Büro", "On-site"),
])
def test_work_modality_precedence(desc: str, expected: str) -> None:
    """Hybrid must be checked before Remote (T-02 fix)."""
    assert _extract_work_modality(desc) == expected


# ---------------------------------------------------------------------------
# Salary extraction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("desc,expected_min,expected_max", [
    ("Gehalt: 60.000 - 80.000 EUR jährlich", 60000, 80000),
    ("60.000 bis 80.000 EUR", 60000, 80000),
    ("60.000–80.000€", 60000, 80000),
    ("ab 60.000 EUR pro Jahr", 60000, None),
    ("Salary: 70.000 EUR", 70000, None),
    # Outliers are rejected
    ("Umsatz: 5.000.000 EUR", None, None),
    ("No salary info", None, None),
    ("", None, None),
])
def test_extract_salary(desc: str, expected_min: int | None, expected_max: int | None) -> None:
    got_min, got_max = _extract_salary(desc)
    assert got_min == expected_min
    assert got_max == expected_max


def test_salary_range_order_preserved() -> None:
    """Min must be <= max for range to be extracted."""
    # Reversed range is rejected
    min_val, max_val = _extract_salary("80.000 - 60.000 EUR")
    # If reversed, falls back to single or None
    if min_val is not None:
        assert min_val <= max_val if max_val is not None else True


def test_salary_employee_count_not_extracted() -> None:
    """Large number ranges adjacent to 'Mitarbeiter' must not be treated as salary."""
    desc = "Unser Unternehmen hat 10.000 bis 50.000 Mitarbeiter weltweit."
    min_val, max_val = _extract_salary(desc)
    assert min_val is None
    assert max_val is None


def test_salary_customer_count_not_extracted() -> None:
    """Numbers adjacent to 'Kunden' must not be treated as salary."""
    desc = "Wir betreuen über 20.000 Kunden in ganz Deutschland."
    min_val, max_val = _extract_salary(desc)
    assert min_val is None
    assert max_val is None


def test_salary_real_salary_not_blocked() -> None:
    """A genuine salary mention not near non-salary words must still be extracted."""
    desc = "Das Gehalt liegt bei 60.000 bis 80.000 EUR jährlich."
    min_val, max_val = _extract_salary(desc)
    assert min_val == 60_000
    assert max_val == 80_000


# ---------------------------------------------------------------------------
# Experience extraction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("desc,expected", [
    ("3+ Jahre Berufserfahrung", 3),
    ("5 years experience required", 5),
    ("mindestens 2 Jahre Erfahrung", 2),
    ("at least 4 years", 4),
    ("min. 3 Jahre Berufserfahrung", 3),
    ("10 Jahre Erfahrung im Bereich", 10),
    ("no experience requirement", None),
    ("", None),
])
def test_extract_experience(desc: str, expected: int | None) -> None:
    assert _extract_experience(desc) == expected


# ---------------------------------------------------------------------------
# Seniority from title
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("title,expected", [
    ("Senior Software Engineer (m/w/d)", "Senior"),
    ("Junior Developer", "Junior"),
    ("Lead Data Scientist", "Lead"),
    ("Tech Lead Backend", "Lead"),
    ("Team Lead Fullstack", "Lead"),
    ("Principal Engineer", "Lead"),
    ("Head of Engineering", "Lead"),
    ("Staff Software Engineer", "Lead"),
    ("Werkstudent IT (m/w/d)", "Junior"),
    ("Praktikant Backend", "Junior"),
    ("Software Engineer", None),
    ("Data Analyst", None),
    ("", None),
])
def test_extract_seniority_from_title(title: str, expected: str | None) -> None:
    assert _extract_seniority_from_title(title) == expected


# ---------------------------------------------------------------------------
# Integration: extract_regex_fields()
# ---------------------------------------------------------------------------

def test_extract_regex_fields_full_example() -> None:
    desc = (
        "Wir suchen einen Senior Backend Developer (m/w/d) für ein Vollzeit-Projekt in Berlin. "
        "Remote-Arbeit ist möglich. Gehalt: 70.000 - 90.000 EUR jährlich. "
        "Mindestens 4 Jahre Berufserfahrung erforderlich. "
        "Skills: Python, FastAPI, PostgreSQL."
    )
    title = "Senior Backend Developer (m/w/d)"
    result = extract_regex_fields(desc, title)

    assert result["contract_type"] == "Full-time"
    assert result["work_modality"] == "Remote"
    assert result["salary_min"] == 70000
    assert result["salary_max"] == 90000
    assert result["experience_years"] == 4
    assert result["seniority_from_title"] == "Senior"


def test_extract_regex_fields_all_none() -> None:
    result = extract_regex_fields("We have a job opening.", "Software Engineer")
    assert result["contract_type"] is None
    assert result["work_modality"] is None
    assert result["salary_min"] is None
    assert result["salary_max"] is None
    assert result["experience_years"] is None
    assert result["seniority_from_title"] is None


def test_extract_regex_fields_returns_all_keys() -> None:
    result = extract_regex_fields("", "")
    expected_keys = {
        "contract_type", "work_modality", "salary_min", "salary_max",
        "experience_years", "seniority_from_title",
        "languages", "education_level",
    }
    assert set(result.keys()) == expected_keys


@pytest.mark.parametrize("desc,title,field,expected", [
    ("Teilzeit 20h/Woche", "Junior Developer", "contract_type", "Part-time"),
    ("Arbeit vor Ort in München", "Data Engineer", "work_modality", "On-site"),
    ("ab 55.000 EUR", "Engineer", "salary_min", 55000),
    ("5 Jahre Berufserfahrung", "Engineer", "experience_years", 5),
    ("", "Head of Product", "seniority_from_title", "Lead"),
    ("", "Werkstudentin IT", "seniority_from_title", "Junior"),
])
def test_extract_regex_fields_individual(
    desc: str, title: str, field: str, expected: str
) -> None:
    result = extract_regex_fields(desc, title)
    assert result[field] == expected


# ---------------------------------------------------------------------------
# Language extraction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("desc,expected_lang,expected_level", [
    ("Deutsch C1 Kenntnisse erforderlich", "German", "C1"),
    ("English B2 required", "English", "B2"),
    ("C1 Deutsch", "German", "C1"),
    ("fließend Deutsch", "German", "B2+"),
    ("fluent English", "English", "B2+"),
    ("verhandlungssicher Deutsch", "German", "C1+"),
    ("muttersprachlich Deutsch", "German", "C2"),
    ("native English", "English", "C2"),
    ("Deutschkenntnisse erforderlich", "German", "required"),
    ("Englischkenntnisse von Vorteil", "English", "required"),
])
def test_extract_languages_single(desc: str, expected_lang: str, expected_level: str) -> None:
    result = _extract_languages(desc)
    assert len(result) >= 1
    langs = {r["language"]: r["level"] for r in result}
    assert langs[expected_lang] == expected_level


def test_extract_languages_multiple() -> None:
    desc = "fließend Deutsch und English B2"
    result = _extract_languages(desc)
    langs = {r["language"] for r in result}
    assert "German" in langs
    assert "English" in langs


def test_extract_languages_empty() -> None:
    assert _extract_languages("") == []
    assert _extract_languages("No language requirements") == []


def test_extract_languages_deduplicates() -> None:
    """Same (language, level) pair must appear only once."""
    desc = "Deutsch C1 und Deutsch C1 Kenntnisse"
    result = _extract_languages(desc)
    german_entries = [r for r in result if r["language"] == "German" and r["level"] == "C1"]
    assert len(german_entries) == 1


# ---------------------------------------------------------------------------
# Education extraction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("desc,expected", [
    ("Promotion oder Doktor bevorzugt", "PhD"),
    ("Ph.D. required", "PhD"),
    ("Master der Informatik oder Diplom", "Master"),
    ("M.Sc. in Computer Science", "Master"),
    ("Bachelor Abschluss in Informatik", "Bachelor"),
    ("B.Sc. required", "Bachelor"),
    ("Hochschulabschluss erwünscht", "Degree"),
    ("university degree required", "Degree"),
    ("Berufsausbildung als IT-Kaufmann", "Vocational"),
    ("Ausbildung als Fachinformatiker", "Vocational"),
    ("No education mentioned", None),
    ("", None),
])
def test_extract_education(desc: str, expected: str | None) -> None:
    assert _extract_education(desc) == expected


def test_extract_education_highest_wins() -> None:
    """When multiple levels found, highest priority is returned."""
    desc = "Master oder Bachelor Abschluss erforderlich"
    assert _extract_education(desc) == "Master"


def test_extract_regex_fields_languages() -> None:
    desc = "fließend Deutsch und English B2 erforderlich"
    result = extract_regex_fields(desc, "Engineer")
    assert isinstance(result["languages"], list)
    assert len(result["languages"]) >= 2


def test_extract_regex_fields_education() -> None:
    result = extract_regex_fields("Master der Informatik oder Diplom", "Data Scientist")
    assert result["education_level"] == "Master"


def test_extract_regex_fields_no_language_no_education() -> None:
    result = extract_regex_fields("Great team, exciting work", "Engineer")
    assert result["languages"] == []
    assert result["education_level"] is None


# ---------------------------------------------------------------------------
# New German contract type patterns (Item 6)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("desc,expected", [
    ("Festanstellung ab sofort", "Full-time"),
    ("unbefristeter Vertrag", "Permanent"),
    ("befristete Stelle für 12 Monate", "Contract"),
    ("Position als Werkstudent (m/w/d)", "Working Student"),
    ("Werkstudentin im Bereich IT", "Working Student"),
    ("Praktikum im Bereich Data Science", "Internship"),
    ("Praktikant/in für 6 Monate", "Internship"),
])
def test_extract_contract_type_german_patterns(desc: str, expected: str) -> None:
    assert _extract_contract_type(desc) == expected


# ---------------------------------------------------------------------------
# Title scanning fallback (Item 4)
# ---------------------------------------------------------------------------


def test_work_modality_from_title_when_description_empty():
    """Title contains 'Hybrid' but description has no modality → title match."""
    assert _extract_work_modality("No modality here.", "Hybrid Software Developer") == "Hybrid"


def test_work_modality_description_wins_over_title():
    """Description match takes priority over title match."""
    assert _extract_work_modality("Arbeit vor Ort in Berlin", "Remote Developer") == "On-site"


def test_contract_type_from_title_fallback():
    """Contract type found in title when description has none."""
    assert _extract_contract_type("No contract info.", "Part-time Developer") == "Part-time"


def test_work_modality_from_title_integration():
    """Integration test: title 'Hybrid' picked up when description has no modality."""
    result = extract_regex_fields("We have a job opening.", "Hybrid Software Engineer")
    assert result["work_modality"] == "Hybrid"


# ---------------------------------------------------------------------------
# New seniority keywords (Item 5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("title,expected", [
    ("Chief Technology Officer", "C-Level"),
    ("CTO", "C-Level"),
    ("CIO / IT Director", "C-Level"),
    ("Director of Engineering", "Director"),
    ("VP Engineering", "Director"),
    ("Vice President of Product", "Director"),
    ("Teamlead Frontend", "Lead"),
    ("Teamleiter Backend-Entwicklung", "Lead"),
])
def test_extract_seniority_new_keywords(title: str, expected: str) -> None:
    assert _extract_seniority_from_title(title) == expected


# ---------------------------------------------------------------------------
# Language extraction dedup and boilerplate filtering (Item 10)
# ---------------------------------------------------------------------------


def test_extract_languages_dedup_keeps_highest_level():
    """When same language appears at multiple levels, highest specificity wins."""
    desc = "Deutschkenntnisse erforderlich. Fließend Deutsch."
    result = _extract_languages(desc)
    german_entries = [r for r in result if r["language"] == "German"]
    assert len(german_entries) == 1
    # "B2+" (fließend) outranks "required" (Kenntnisse)
    assert german_entries[0]["level"] == "B2+"


def test_extract_languages_boilerplate_filtered():
    """Language mentioned only in diversity/boilerplate context is excluded."""
    desc = (
        "We are looking for a Python developer. "
        "We are an equal opportunity employer regardless of English proficiency."
    )
    result = _extract_languages(desc)
    # "English" is only in boilerplate → should be excluded
    assert not any(r["language"] == "English" for r in result)


def test_extract_languages_boilerplate_german():
    """German boilerplate 'freuen uns auf alle' language mention excluded."""
    desc = (
        "Wir suchen einen Entwickler. "
        "Wir freuen uns auf Ihre Bewerbung. Deutschkenntnisse und diversity."
    )
    # "Deutsch" appears near boilerplate → should be excluded
    result = _extract_languages(desc)
    assert not any(r["language"] == "German" for r in result)


def test_extract_languages_real_requirement_not_filtered():
    """Language requirements in the actual job requirements section are kept."""
    desc = (
        "Anforderungen: Fließend Deutsch und English B2 erforderlich. "
        "Mindestens 3 Jahre Erfahrung."
    )
    result = _extract_languages(desc)
    langs = {r["language"] for r in result}
    assert "German" in langs
    assert "English" in langs


# ---------------------------------------------------------------------------
# Fix 1 (BUG-3): Markdown escaping → salary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    (r"55\.000€", "55.000€"),
    (r"80\.000,00€", "80.000,00€"),
    (r"Deutsch\- und Englischkenntnisse", "Deutsch- und Englischkenntnisse"),
    ("no escaping here", "no escaping here"),
    (r"multiple\\\.dots", "multiple.dots"),
])
def test_strip_markdown_escaping(text: str, expected: str) -> None:
    assert _strip_markdown_escaping(text) == expected


@pytest.mark.parametrize("desc,expected_min,expected_max", [
    # Escaped periods — the main BUG-3 scenario
    (r"Gehalt: 60\.000 - 80\.000 EUR jährlich", 60000, 80000),
    (r"ab 55\.000€ pro Jahr", 55000, None),
    # Comma-decimal format
    ("Gehalt: 60.000,00 - 80.000,00 EUR", 60000, 80000),
    # "und" separator
    ("zwischen 70.000 und 95.000 Euro", 70000, 95000),
    # "Euro" as currency word
    ("ab 65.000 Euro brutto", 65000, None),
    # Unescaped salary still works (regression)
    ("60.000 - 80.000 EUR jährlich", 60000, 80000),
])
def test_salary_with_escaping_and_formats(
    desc: str, expected_min: int | None, expected_max: int | None,
) -> None:
    result = extract_regex_fields(desc, "Engineer")
    assert result["salary_min"] == expected_min
    assert result["salary_max"] == expected_max


def test_salary_monthly_rejected() -> None:
    """Monthly salary amounts must not be extracted as annual salary."""
    result = extract_regex_fields("2.117 €/Monat brutto", "Engineer")
    assert result["salary_min"] is None


# ---------------------------------------------------------------------------
# Fix 2 (BUG-4): Fluency regex inflection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("desc,expected_lang,expected_level", [
    ("fließende Deutschkenntnisse", "German", "B2+"),
    ("fließendes Deutsch", "German", "B2+"),
    ("verhandlungssichere Englischkenntnisse", "English", "C1+"),
    ("verhandlungssicheres Deutsch", "German", "C1+"),
    ("muttersprachliches Deutsch", "German", "C2"),
    ("muttersprachlicher Englisch", "English", "C2"),
    # Base forms still work (regression)
    ("fließend Deutsch", "German", "B2+"),
    ("verhandlungssicher Englisch", "English", "C1+"),
])
def test_fluency_inflection(desc: str, expected_lang: str, expected_level: str) -> None:
    result = _extract_languages(desc)
    langs = {r["language"]: r["level"] for r in result}
    assert langs.get(expected_lang) == expected_level


# ---------------------------------------------------------------------------
# Fix 3 (BUG-5): Compound language pattern
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("desc,expected_langs", [
    ("Deutsch- und Englischkenntnisse in Wort und Schrift", {"German", "English"}),
    ("Englisch- und Deutschkenntnisse", {"German", "English"}),
    ("Englisch- und Deutschkenntnisse in Wort und Schrift", {"German", "English"}),
])
def test_compound_language_pattern(desc: str, expected_langs: set[str]) -> None:
    result = _extract_languages(desc)
    langs = {r["language"] for r in result}
    assert expected_langs.issubset(langs)


def test_compound_with_escaping() -> None:
    """Escaped dash in compound pattern handled after strip_markdown_escaping."""
    desc = r"Deutsch\- und Englischkenntnisse"
    result = extract_regex_fields(desc, "Engineer")
    langs = {r["language"] for r in result["languages"]}
    assert "German" in langs
    assert "English" in langs


# ---------------------------------------------------------------------------
# Fix 4 (BUG-6): gute/sehr gute language pattern
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("desc,expected_lang", [
    ("Gute Deutschkenntnisse", "German"),
    ("sehr gute Englischkenntnisse", "English"),
    ("sichere Deutschkenntnisse", "German"),
    ("solide Englischkenntnisse", "English"),
    ("gute Deutsch- und Englischkenntnisse", "German"),
])
def test_gute_language_pattern(desc: str, expected_lang: str) -> None:
    result = _extract_languages(desc)
    langs = {r["language"] for r in result}
    assert expected_lang in langs


def test_gute_does_not_override_cefr() -> None:
    """A more specific CEFR level wins over gute's 'required' level."""
    desc = "Gute Deutschkenntnisse. Deutsch C1 erforderlich."
    result = _extract_languages(desc)
    german = [r for r in result if r["language"] == "German"]
    assert len(german) == 1
    assert german[0]["level"] == "C1"  # C1 outranks "required"


# ---------------------------------------------------------------------------
# Fix 5: Work modality — Hybrid indicators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("desc,expected", [
    ("Möglichkeit zum mobilen Arbeiten", "Hybrid"),
    ("mobiles Arbeiten 2 Tage pro Woche", "Hybrid"),
    ("flexibler Arbeitsort", "Hybrid"),
    ("flexible Arbeitsorte möglich", "Hybrid"),
    # Homeoffice-Möglichkeit stays Remote (per user decision)
    ("Homeoffice-Möglichkeit vorhanden", "Remote"),
    # Standalone Homeoffice stays Remote
    ("Homeoffice nach Absprache", "Remote"),
    # Pure Remote regression
    ("100% Remote", "Remote"),
])
def test_hybrid_indicators(desc: str, expected: str) -> None:
    assert _extract_work_modality(desc) == expected


# ---------------------------------------------------------------------------
# Fix 6: Qualitative experience terms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("desc,expected", [
    ("Mehrjährige Berufserfahrung im Bereich IT", 3),
    ("Langjährige Erfahrung in der Softwareentwicklung", 5),
    ("Erste Berufserfahrung in Python", 1),
    ("Einschlägige Berufserfahrung", 3),
    # Digit-based still takes priority (regression)
    ("5 Jahre Berufserfahrung", 5),
    ("mindestens 3 Jahre Erfahrung", 3),
])
def test_qualitative_experience(desc: str, expected: int) -> None:
    assert _extract_experience(desc) == expected


def test_digit_experience_wins_over_qualitative() -> None:
    """When both digit and qualitative are present, digit-based takes priority."""
    desc = "Mehrjährige Berufserfahrung, idealerweise 5 Jahre Erfahrung"
    assert _extract_experience(desc) == 5


# ---------------------------------------------------------------------------
# Fix 7: €K salary format
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("desc,expected_min,expected_max", [
    ("€80K - €110K per year", 80000, 110000),
    ("Salary: €85k", 85000, None),
    ("€70k–€90k", 70000, 90000),
    # Out-of-bounds rejected
    ("€5k", None, None),
])
def test_salary_k_format(
    desc: str, expected_min: int | None, expected_max: int | None,
) -> None:
    got_min, got_max = _extract_salary(desc)
    assert got_min == expected_min
    assert got_max == expected_max
