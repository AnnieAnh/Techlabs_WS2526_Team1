"""Tests for Epic 6: Post-extraction validators (E6-01 through E6-05)."""

import json
from pathlib import Path

import pytest

from extraction.validators import ValidationFlag
from extraction.validators.cross_field import validate_all_cross_fields, validate_cross_fields
from extraction.validators.hallucination import (
    verify_all_skills,
    verify_evidence_item,
    verify_row_evidence,
    verify_skill_in_description,
    verify_source_in_description,
)
from extraction.validators.runner import run_validators
from extraction.validators.salary import check_salary, validate_salaries
from extraction.validators.skills import (
    load_skill_aliases,
    normalize_all_skills,
    normalize_and_reconcile_evidence,
    normalize_skills_evidence,
)


def _cfg(tmp_path: Path) -> dict:
    return {
        "validation": {
            "salary_min_floor": 15_000,
            "salary_max_ceiling": 300_000,
            "salary_monthly_threshold": 5_000,
            "skill_hallucination_threshold": 0.3,
        },
        "paths": {
            "reports_dir": tmp_path / "reports",
        },
    }


def _result(row_id: str, data: dict) -> dict:
    """Build a minimal extraction result dict."""
    return {"row_id": row_id, "data": data}


def _row(
    row_id: str,
    description: str = "",
    title: str = "",
    seniority_from_title: str = "",
    **extra,
) -> dict:
    """Build a minimal original-row dict (as would come from the DataFrame)."""
    return {
        "row_id": row_id,
        "title": title,
        "title_cleaned": title,
        "description": description,
        "seniority_from_title": seniority_from_title,
        **extra,
    }


def _full_data(**overrides) -> dict:
    """Return a complete valid extraction data dict with sensible defaults."""
    base = {
        "salary_min": 60_000,
        "salary_max": 80_000,
        "technical_skills": [],
        "soft_skills": [],
        "nice_to_have_skills": [],
        "seniority": None,
        "work_modality": None,
        "experience_years": None,
        "contract_type": None,
    }
    base.update(overrides)
    return base


@pytest.fixture
def aliases_file(tmp_path: Path) -> Path:
    """Write a minimal skill_aliases.yaml for tests."""
    content = (
        "Golang: Go\n"
        "K8s: Kubernetes\n"
        "k8s: Kubernetes\n"
        "JS: JavaScript\n"
        "React.js: React\n"
        "NodeJS: Node.js\n"
    )
    p = tmp_path / "skill_aliases.yaml"
    p.write_text(content, encoding="utf-8")
    return p


def test_validation_flag_fields():
    f = ValidationFlag(
        row_id="r1", field="salary_min", rule="below_floor", severity="warning", message="oops"
    )
    assert f.row_id == "r1"
    assert f.field == "salary_min"
    assert f.rule == "below_floor"
    assert f.severity == "warning"
    assert f.message == "oops"


def test_salary_clean_passes(tmp_path):
    flags = check_salary("r1", 60_000, 90_000, _cfg(tmp_path))
    assert flags == []


def test_salary_both_null_no_flags(tmp_path):
    flags = check_salary("r1", None, None, _cfg(tmp_path))
    assert flags == []


def test_salary_one_null_no_min_max_flag(tmp_path):
    """When only one side is set, min>max check should not fire."""
    flags = check_salary("r1", None, 80_000, _cfg(tmp_path))
    rules = [f.rule for f in flags]
    assert "min_greater_than_max" not in rules


def test_salary_min_gt_max_flagged(tmp_path):
    flags = check_salary("r1", 90_000, 60_000, _cfg(tmp_path))
    rules = [f.rule for f in flags]
    assert "min_greater_than_max" in rules


def test_salary_possible_monthly_flagged(tmp_path):
    flags = check_salary("r1", 3_000, 4_500, _cfg(tmp_path))
    rules = [f.rule for f in flags]
    assert "possible_monthly" in rules


def test_salary_possible_revenue_flagged(tmp_path):
    flags = check_salary("r1", 500_000, 800_000, _cfg(tmp_path))
    rules = [f.rule for f in flags]
    assert "possible_revenue" in rules


def test_salary_currency_no_longer_validated(tmp_path):
    """salary_currency field removed — no non_eur_currency flag should be raised."""
    flags = check_salary("r1", 60_000, 90_000, _cfg(tmp_path))
    rules = [f.rule for f in flags]
    assert "non_eur_currency" not in rules


def test_salary_below_floor_flagged(tmp_path):
    flags = check_salary("r1", 10_000, 14_000, _cfg(tmp_path))
    rules = [f.rule for f in flags]
    assert "below_floor" in rules


def test_validate_salaries_batch(tmp_path):
    # salary is Tier 1 (regex-extracted) — values live in top-level regex_salary_* keys
    results = [
        {"row_id": "r1", "regex_salary_min": 60_000, "regex_salary_max": 80_000, "data": {}},
        {"row_id": "r2", "regex_salary_min": 90_000, "regex_salary_max": 60_000, "data": {}},
    ]
    flags = validate_salaries(results, _cfg(tmp_path))
    flagged_ids = {f.row_id for f in flags}
    assert "r2" in flagged_ids  # min > max
    assert "r1" not in flagged_ids  # clean


def test_load_skill_aliases(aliases_file):
    aliases = load_skill_aliases(aliases_file)
    assert aliases["golang"] == "Go"
    assert aliases["k8s"] == "Kubernetes"
    assert aliases["js"] == "JavaScript"


def test_normalize_skills_evidence_alias_resolved_basic():
    aliases = {"js": "JavaScript"}
    items = [
        {"name": "JS", "source": "JS framework"},
        {"name": "React", "source": "React"},
    ]
    result = normalize_skills_evidence(items, aliases)
    names = [r["name"] for r in result]
    assert "JavaScript" in names
    assert "JS" not in names


def test_normalize_skills_evidence_dedup_via_alias():
    aliases = {"golang": "Go"}
    items = [
        {"name": "Go", "source": "Go lang"},
        {"name": "Golang", "source": "Golang dev"},
        {"name": "Python", "source": "Python"},
    ]
    result = normalize_skills_evidence(items, aliases)
    names = [r["name"] for r in result]
    assert "Go" in names
    assert "Python" in names
    assert len(result) == 2


def test_normalize_skills_evidence_sorted():
    aliases = {}
    items = [
        {"name": "Zebra", "source": "z"},
        {"name": "Apple", "source": "a"},
        {"name": "Mango", "source": "m"},
    ]
    result = normalize_skills_evidence(items, aliases)
    assert [r["name"] for r in result] == ["Apple", "Mango", "Zebra"]


def test_normalize_skills_evidence_empty():
    assert normalize_skills_evidence([], {}) == []


def test_normalize_and_reconcile_evidence_contradiction_removed():
    aliases = {"golang": "Go"}
    tech = [
        {"name": "Go", "source": "Go"},
        {"name": "Python", "source": "Python"},
    ]
    nice = [
        {"name": "Go", "source": "Go nice"},
        {"name": "Docker", "source": "Docker"},
    ]
    req, nice_out, contradictions = normalize_and_reconcile_evidence(tech, nice, aliases)
    req_names = [i["name"] for i in req]
    nice_names = [i["name"] for i in nice_out]
    assert "Go" in req_names
    assert "Go" not in nice_names
    assert len(contradictions) == 1
    assert contradictions[0] == "Go"


def test_normalize_and_reconcile_evidence_no_contradictions():
    aliases = {}
    tech = [{"name": "Python", "source": "Python"}]
    nice = [{"name": "Docker", "source": "Docker"}]
    req, nice_out, contradictions = normalize_and_reconcile_evidence(tech, nice, aliases)
    assert [i["name"] for i in req] == ["Python"]
    assert [i["name"] for i in nice_out] == ["Docker"]
    assert contradictions == []


def test_skill_in_description_direct():
    assert verify_skill_in_description("Python", "We need Python developers.", {}) is True


def test_skill_not_in_description():
    assert verify_skill_in_description("Rust", "We need Python developers.", {}) is False


def test_skill_java_not_matches_javascript():
    """'Java' word-boundary check must NOT match inside 'JavaScript'."""
    assert verify_skill_in_description("Java", "We use JavaScript frameworks.", {}) is False


def test_skill_java_matches_when_standalone():
    assert verify_skill_in_description("Java", "We need Java and Spring.", {}) is True


def test_skill_kubernetes_via_k8s_alias():
    """K8s in description should verify Kubernetes skill via alias lookup."""
    aliases = {"k8s": "Kubernetes"}
    assert verify_skill_in_description("Kubernetes", "We use K8s in production.", aliases) is True


def test_skill_case_insensitive():
    assert verify_skill_in_description("Python", "must know python well", {}) is True


def test_verify_row_evidence_all_found_basic():
    items = [
        {"name": "Python", "source": "Python"},
        {"name": "Docker", "source": "Docker"},
    ]
    flags = verify_row_evidence("r1", items, "Python and Docker are required.", {})
    skill_flags = [f for f in flags if f.rule == "skill_not_in_description"]
    assert skill_flags == []


def test_verify_row_evidence_unverified_flagged_basic():
    items = [{"name": "Haskell", "source": "functional programming"}]
    flags = verify_row_evidence("r1", items, "Python is great.", {})
    rules = [f.rule for f in flags]
    assert "skill_not_in_description" in rules


def test_verify_row_evidence_high_hallucination_rate_basic():
    """More than 30% unverified → high_hallucination_rate flag."""
    items = [
        {"name": "Python", "source": "Python"},
        {"name": "Rust", "source": "Rust lang"},
        {"name": "Haskell", "source": "Haskell FP"},
        {"name": "Erlang", "source": "Erlang OTP"},
    ]
    flags = verify_row_evidence("r1", items, "Python is nice.", {}, threshold=0.3)
    rules = [f.rule for f in flags]
    assert "skill_not_in_description" in rules
    assert "high_hallucination_rate" in rules


def test_verify_row_evidence_below_threshold_no_row_flag():
    """Exactly 1/4 unverified (25%) is below the 30% threshold → no row-level flag."""
    items = [
        {"name": "Python", "source": "Python"},
        {"name": "Docker", "source": "Docker"},
        {"name": "SQL", "source": "SQL"},
        {"name": "Haskell", "source": "Haskell FP"},
    ]
    desc = "Python Docker SQL developer needed."
    flags = verify_row_evidence("r1", items, desc, {}, threshold=0.3)
    rules = [f.rule for f in flags]
    assert "skill_not_in_description" in rules
    assert "high_hallucination_rate" not in rules


def test_verify_row_evidence_empty():
    flags = verify_row_evidence("r1", [], "Any description.", {})
    assert flags == []


def test_verify_all_skills_batch():
    results = [
        _result("r1", {
            "technical_skills": [{"name": "Python", "source": "Python"}],
            "nice_to_have_skills": [],
        }),
        _result("r2", {
            "technical_skills": [{"name": "Cobol", "source": "COBOL mainframe"}],
            "nice_to_have_skills": [],
        }),
    ]
    desc_by_id = {
        "r1": "We need Python developers.",
        "r2": "We build cloud-native microservices.",
    }
    flags = verify_all_skills(results, desc_by_id, {})
    flagged_ids = {f.row_id for f in flags if f.rule == "skill_not_in_description"}
    assert "r2" in flagged_ids
    assert "r1" not in flagged_ids


def _cross_row(
    row_id: str,
    data: dict,
    description: str = "",
    title: str = "",
    seniority_from_title: str = "",
) -> dict:
    return {
        "row_id": row_id,
        "data": data,
        "description": description,
        "title_cleaned": title,
        "seniority_from_title": seniority_from_title,
        # Mirror Tier-1 fields at the top-level regex_* keys (as run_validators injects)
        "regex_contract_type": data.get("contract_type"),
        "regex_work_modality": data.get("work_modality"),
        "regex_experience_years": data.get("experience_years"),
    }


def test_cross_clean_row_no_flags():
    row = _cross_row(
        "r1",
        _full_data(seniority="Senior", technical_skills=["Python"], experience_years=5),
        description="Senior Python engineer with Docker experience.",
        title="Senior Python Engineer",
        seniority_from_title="senior",
    )
    flags = validate_cross_fields(row)
    assert flags == []


def test_cross_rule1_junior_title_senior_seniority():
    row = _cross_row(
        "r1",
        _full_data(seniority="Senior", technical_skills=["Python"]),
        title="Junior Software Engineer",
        seniority_from_title="junior",
        description="Junior dev role.",
    )
    rules = [f.rule for f in validate_cross_fields(row)]
    assert "title_seniority_mismatch" in rules


def test_cross_rule1_no_flag_when_consistent():
    row = _cross_row(
        "r1",
        _full_data(seniority="Junior"),
        title="Junior Developer",
        seniority_from_title="junior",
        description="Entry level position.",
    )
    rules = [f.rule for f in validate_cross_fields(row)]
    assert "title_seniority_mismatch" not in rules


def test_cross_rule2_intern_with_fulltime_contract():
    row = _cross_row(
        "r1",
        _full_data(contract_type="Full-time"),
        title="Werkstudent Backend Developer",
        description="Working student position.",
    )
    rules = [f.rule for f in validate_cross_fields(row)]
    assert "intern_contract_mismatch" in rules


def test_cross_rule2_intern_with_parttime_ok():
    row = _cross_row(
        "r1",
        _full_data(contract_type="Part-time"),
        title="Werkstudent Backend Developer",
        description="Working student position.",
    )
    rules = [f.rule for f in validate_cross_fields(row)]
    assert "intern_contract_mismatch" not in rules


def test_cross_rule4_experience_unrealistic():
    row = _cross_row(
        "r1",
        _full_data(experience_years=25),
        description="Senior developer role.",
        title="Senior Engineer",
    )
    rules = [f.rule for f in validate_cross_fields(row)]
    assert "experience_unrealistic" in rules


def test_cross_rule4_normal_experience_no_flag():
    row = _cross_row(
        "r1",
        _full_data(experience_years=5),
        description="Developer role.",
        title="Engineer",
    )
    rules = [f.rule for f in validate_cross_fields(row)]
    assert "experience_unrealistic" not in rules


def test_cross_rule5_no_skills_long_description():
    row = _cross_row(
        "r1",
        _full_data(technical_skills=[]),
        description="A" * 600,
        title="Engineer",
    )
    rules = [f.rule for f in validate_cross_fields(row)]
    assert "no_skills_long_description" in rules


def test_cross_rule5_no_skills_short_description_ok():
    row = _cross_row(
        "r1",
        _full_data(technical_skills=[]),
        description="Short job.",
        title="Engineer",
    )
    rules = [f.rule for f in validate_cross_fields(row)]
    assert "no_skills_long_description" not in rules


def test_cross_rule7_remote_but_onsite_text():
    row = _cross_row(
        "r1",
        _full_data(work_modality="Remote", technical_skills=["Python"]),
        description="You must work vor Ort at our Munich office.",
        title="Engineer",
    )
    rules = [f.rule for f in validate_cross_fields(row)]
    assert "remote_but_onsite_text" in rules


def test_cross_rule7_remote_no_onsite_text_ok():
    row = _cross_row(
        "r1",
        _full_data(work_modality="Remote", technical_skills=["Python"]),
        description="Fully flexible work from anywhere.",
        title="Engineer",
    )
    rules = [f.rule for f in validate_cross_fields(row)]
    assert "remote_but_onsite_text" not in rules


def test_cross_rule8_onsite_but_remote_text():
    row = _cross_row(
        "r1",
        _full_data(work_modality="Onsite", technical_skills=["Python"]),
        description="This is a 100% remote position with no office required.",
        title="Engineer",
    )
    rules = [f.rule for f in validate_cross_fields(row)]
    assert "onsite_but_remote_text" in rules


def test_cross_rule8_onsite_no_remote_text_ok():
    row = _cross_row(
        "r1",
        _full_data(work_modality="Onsite", technical_skills=["Python"]),
        description="You will work in our Hamburg office daily.",
        title="Engineer",
    )
    rules = [f.rule for f in validate_cross_fields(row)]
    assert "onsite_but_remote_text" not in rules


def test_validate_all_cross_fields_aggregates():
    rows = [
        _cross_row(
            "r1",
            _full_data(seniority="Senior", technical_skills=["Python"]),
            title="Junior Developer",
            seniority_from_title="junior",
            description="Junior dev.",
        ),
        _cross_row(
            "r2",
            _full_data(technical_skills=["Python"]),
            description="Python developer needed.",
            title="Developer",
        ),
    ]
    flags = validate_all_cross_fields(rows)
    flagged_ids = {f.row_id for f in flags}
    assert "r1" in flagged_ids  # title_seniority_mismatch


def test_run_validators_adds_validation_flags_key(tmp_path, aliases_file):
    cfg = _cfg(tmp_path)
    cfg["paths"]["skill_aliases"] = str(aliases_file)

    results = [
        _result("r1", _full_data()),
    ]
    # regex_experience_years=25 → experience_unrealistic cross-field flag
    df_rows = [
        _row("r1", description="Engineering role.", title="Engineer", regex_experience_years=25)
    ]

    final, report = run_validators(results, df_rows, cfg)

    assert "validation_flags" in final[0]
    assert any(f["rule"] == "experience_unrealistic" for f in final[0]["validation_flags"])


def test_run_validators_saves_report_json(tmp_path, aliases_file):
    cfg = _cfg(tmp_path)
    cfg["paths"]["skill_aliases"] = str(aliases_file)

    results = [_result("r1", _full_data(
        technical_skills=[{"name": "Python", "source": "Python"}],
    ))]
    df_rows = [_row("r1", description="Python developer needed.", title="Python Developer")]

    run_validators(results, df_rows, cfg)

    report_path = tmp_path / "reports" / "validation_report.json"
    assert report_path.exists()
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["total_validated"] == 1
    assert "flags_by_rule" in saved
    assert "skill_normalisation" in saved


def test_run_validators_clean_row_no_salary_flags(tmp_path, aliases_file):
    cfg = _cfg(tmp_path)
    cfg["paths"]["skill_aliases"] = str(aliases_file)

    results = [
        _result(
            "r1",
            _full_data(
                salary_min=60_000,
                salary_max=80_000,
                technical_skills=[
                    {"name": "Python", "source": "Python"},
                    {"name": "SQL", "source": "SQL"},
                ],
                nice_to_have_skills=[
                    {"name": "Docker", "source": "Docker is a plus"},
                ],
                seniority="Senior",
                work_modality="Hybrid",
                experience_years=3,
                contract_type="Full-time",
            ),
        )
    ]
    df_rows = [
        _row(
            "r1",
            description="Python and SQL developer needed. Docker is a plus. Senior position.",
            title="Senior Python Developer",
            seniority_from_title="senior",
        )
    ]

    final, report = run_validators(results, df_rows, cfg)

    salary_flags = [f for f in final[0]["validation_flags"] if "salary" in f["rule"]]
    exp_flags = [f for f in final[0]["validation_flags"] if "experience" in f["rule"]]
    assert salary_flags == []
    assert exp_flags == []


def test_run_validators_skill_normalisation_applied(tmp_path, aliases_file):
    """Skills are normalised before flags are computed."""
    cfg = _cfg(tmp_path)
    cfg["paths"]["skill_aliases"] = str(aliases_file)

    results = [
        _result("r1", _full_data(
            technical_skills=[
                {"name": "Golang", "source": "Go development"},
                {"name": "Golang", "source": "Go lang"},
            ],
            nice_to_have_skills=[],
        )),
    ]
    df_rows = [_row("r1", description="Go development wanted.", title="Go Developer")]

    final, _ = run_validators(results, df_rows, cfg)

    req = final[0]["data"]["technical_skills"]
    assert req == ["Go"]  # Golang → Go, deduped, then flattened


def test_run_validators_report_counts_match(tmp_path, aliases_file):
    cfg = _cfg(tmp_path)
    cfg["paths"]["skill_aliases"] = str(aliases_file)

    results = [
        _result("r1", _full_data()),  # clean
        _result("r2", _full_data()),  # flagged via experience_unrealistic
    ]
    df_rows = [
        _row("r1", description="Engineering role.", title="Engineer"),
        _row("r2", description="Engineering role.", title="Engineer", regex_experience_years=25),
    ]

    final, report = run_validators(results, df_rows, cfg)

    assert report["total_validated"] == 2
    assert report["rows_with_flags"] >= 1
    assert report["rows_clean"] == report["total_validated"] - report["rows_with_flags"]


def test_run_validators_removes_hallucinated_skills(tmp_path, aliases_file):
    """Skills not grounded in the description are stripped from the result data."""
    cfg = _cfg(tmp_path)
    cfg["paths"]["skill_aliases"] = str(aliases_file)

    results = [
        _result(
            "r1",
            _full_data(
                technical_skills=[
                    {"name": "Python", "source": "Python"},
                    {"name": "Haskell", "source": "functional paradigm"},
                ],
                nice_to_have_skills=[
                    {"name": "Erlang", "source": "Erlang OTP"},
                ],
            ),
        )
    ]
    # Only Python is mentioned — Haskell and Erlang sources not in description
    df_rows = [_row("r1", description="We use Python for our backend.", title="Python Developer")]

    final, _ = run_validators(results, df_rows, cfg)

    req = final[0]["data"]["technical_skills"]
    nice = final[0]["data"]["nice_to_have_skills"]
    assert req == ["Python"]
    assert nice == []


# Seniority is derived from title by regex_extractor.py at the regex_extract stage.


def test_run_validators_salary_flags_from_regex_fields(tmp_path, aliases_file):
    """Salary validation reads regex_salary_min/max from df_rows, not from data."""
    cfg = _cfg(tmp_path)
    cfg["paths"]["skill_aliases"] = str(aliases_file)

    results = [_result("r1", _full_data(
        technical_skills=[{"name": "Python", "source": "Python"}],
        nice_to_have_skills=[],
    ))]
    df_rows = [
        _row(
            "r1",
            description="Python developer needed.",
            title="Developer",
            regex_salary_min=90_000,  # min > max → should flag
            regex_salary_max=60_000,
        )
    ]

    final, _ = run_validators(results, df_rows, cfg)
    flag_rules = [f["rule"] for f in final[0]["validation_flags"]]
    assert "min_greater_than_max" in flag_rules


def test_run_validators_salary_none_no_flags(tmp_path, aliases_file):
    """When regex salary fields are absent, salary validator produces no flags."""
    cfg = _cfg(tmp_path)
    cfg["paths"]["skill_aliases"] = str(aliases_file)

    results = [_result("r1", _full_data(
        technical_skills=[{"name": "Python", "source": "Python"}],
        nice_to_have_skills=[],
    ))]
    df_rows = [_row("r1", description="Python developer needed.", title="Developer")]

    final, _ = run_validators(results, df_rows, cfg)
    salary_flags = [f for f in final[0]["validation_flags"] if f["rule"] in (
        "min_greater_than_max", "possible_monthly", "possible_revenue", "below_floor"
    )]
    assert salary_flags == []


# ---------------------------------------------------------------------------
# P3: Skill variant matching (SKILL_VARIANTS + _word_match_v2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("skill, description, expected", [
    ("REST APIs", "We need REST API experience with RESTful services", True),
    ("scikit-learn", "Experience with sklearn is required", True),
    ("CI/CD", "Knowledge of cicd pipelines", True),
    ("C#", "Experience with csharp or C# is preferred", True),
    ("Node.js", "nodejs experience required for backend", True),
    ("C++", "C/C++ development experience preferred", True),
    (".NET", "dotnet framework and .NET Core experience", True),
    ("Power BI", "powerbi or power-bi dashboards", True),
])
def test_verify_skill_variants(skill: str, description: str, expected: bool) -> None:
    """P3: Skill variant spellings are correctly matched in description."""
    from extraction.validators.hallucination import verify_skill_in_description

    assert verify_skill_in_description(skill, description, {}) == expected


# ---------------------------------------------------------------------------
# Evidence-based source verification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source, description, expected", [
    # Tier 1: exact substring
    ("Python-Entwicklung", "Erfahrung in Python-Entwicklung gesucht", True),
    ("Java und Spring Boot", "Kenntnisse in Java und Spring Boot", True),
    # Tier 1: case-insensitive
    ("python", "Experience with Python required", True),
    # Tier 2: whitespace-normalized
    ("Multi Sensor\nDaten Fusion", "Multi Sensor Daten Fusion system", True),
    # Tier 3: token overlap
    (
        "Entwicklung von Microservices",
        "Entwicklung und Wartung von Microservices Architektur",
        True,
    ),
    # Fabricated source — not in description
    ("Kubernetes orchestration", "We use Python for our backend services.", False),
    # Empty source
    ("", "Any description.", False),
])
def test_verify_source_in_description(source: str, description: str, expected: bool) -> None:
    assert verify_source_in_description(source, description) == expected


def test_verify_evidence_item_source_found():
    """Item with valid source is grounded."""
    item = {"name": "Systems Engineering", "source": "Systemingenieur"}
    grounded, method = verify_evidence_item(
        item, "Wir suchen einen Systemingenieur.", {}
    )
    assert grounded is True
    assert method == "source_verified"


def test_verify_evidence_item_source_not_found_name_fallback():
    """Item with bad source but name in description → grounded via name fallback."""
    item = {"name": "Python", "source": "fabricated phrase"}
    grounded, method = verify_evidence_item(
        item, "We need Python developers.", {}
    )
    assert grounded is True
    assert method == "name_verified"


def test_verify_evidence_item_unverified():
    """Item with bad source and name not in description → unverified."""
    item = {"name": "Kubernetes", "source": "container orchestration platform"}
    grounded, method = verify_evidence_item(
        item, "We use Python for our backend.", {}
    )
    assert grounded is False
    assert method == "unverified"


def test_verify_evidence_item_empty_source_name_fallback():
    """Item with empty source falls back to name matching."""
    item = {"name": "Docker", "source": ""}
    grounded, method = verify_evidence_item(
        item, "Docker containers are used.", {}
    )
    assert grounded is True
    assert method == "name_verified"


def test_verify_row_evidence_all_grounded():
    items = [
        {"name": "Python", "source": "Python"},
        {"name": "Docker", "source": "Docker"},
    ]
    flags = verify_row_evidence("r1", items, "Python and Docker are required.", {})
    skill_flags = [f for f in flags if f.rule == "skill_not_in_description"]
    assert skill_flags == []


def test_verify_row_evidence_unverified_flagged():
    items = [{"name": "Haskell", "source": "functional programming"}]
    flags = verify_row_evidence("r1", items, "Python is great.", {})
    rules = [f.rule for f in flags]
    assert "skill_not_in_description" in rules


def test_verify_row_evidence_high_hallucination_rate():
    """More than 30% unverified evidence items → high_hallucination_rate flag."""
    items = [
        {"name": "Python", "source": "Python"},
        {"name": "Rust", "source": "Rust lang"},
        {"name": "Haskell", "source": "Haskell FP"},
        {"name": "Erlang", "source": "Erlang OTP"},
    ]
    flags = verify_row_evidence("r1", items, "Python is nice.", {}, threshold=0.3)
    rules = [f.rule for f in flags]
    assert "high_hallucination_rate" in rules


def test_normalize_all_skills_evidence_format(aliases_file):
    """normalize_all_skills handles evidence format correctly."""
    aliases = load_skill_aliases(aliases_file)
    results = [
        _result("r1", {
            "technical_skills": [
                {"name": "Golang", "source": "Go development"},
                {"name": "Go", "source": "Go lang"},
            ],
            "nice_to_have_skills": [
                {"name": "K8s", "source": "Kubernetes experience"},
            ],
        }),
    ]
    updated, stats = normalize_all_skills(results, aliases)
    tech = updated[0]["data"]["technical_skills"]
    nice = updated[0]["data"]["nice_to_have_skills"]
    # Go and Golang → same canonical "Go"
    assert len(tech) == 1
    assert tech[0]["name"] == "Go"
    # K8s → Kubernetes
    assert nice[0]["name"] == "Kubernetes"


# ---------------------------------------------------------------------------
# Runner: evidence flatten + hallucination removal
# ---------------------------------------------------------------------------


def test_run_validators_evidence_flattened_to_strings(tmp_path, aliases_file):
    """After run_validators, evidence objects are flattened to plain strings."""
    cfg = _cfg(tmp_path)
    cfg["paths"]["skill_aliases"] = str(aliases_file)

    results = [
        _result("r1", _full_data(
            technical_skills=[
                {"name": "Python", "source": "Python development"},
            ],
            nice_to_have_skills=[
                {"name": "Docker", "source": "Docker containers"},
            ],
            benefits=[
                {"name": "30 vacation days", "source": "30 Urlaubstage"},
            ],
            tasks=[
                {"name": "Build APIs", "source": "API-Entwicklung"},
            ],
        )),
    ]
    df_rows = [_row(
        "r1",
        description="Python development. Docker containers."
        " 30 Urlaubstage. API-Entwicklung.",
        title="Developer",
    )]

    final, _ = run_validators(results, df_rows, cfg)

    # All evidence fields should be flattened to plain strings
    assert final[0]["data"]["technical_skills"] == ["Python"]
    assert final[0]["data"]["nice_to_have_skills"] == ["Docker"]
    assert final[0]["data"]["benefits"] == ["30 vacation days"]
    assert final[0]["data"]["tasks"] == ["Build APIs"]

    # Evidence columns preserve full objects with source quotes
    assert final[0]["data"]["technical_skills_evidence"] == [
        {"name": "Python", "source": "Python development"},
    ]
    assert final[0]["data"]["nice_to_have_skills_evidence"] == [
        {"name": "Docker", "source": "Docker containers"},
    ]
    assert final[0]["data"]["benefits_evidence"] == [
        {"name": "30 vacation days", "source": "30 Urlaubstage"},
    ]
    assert final[0]["data"]["tasks_evidence"] == [
        {"name": "Build APIs", "source": "API-Entwicklung"},
    ]


def test_run_validators_evidence_hallucination_removed(tmp_path, aliases_file):
    """Evidence items with fabricated sources are removed before flattening."""
    cfg = _cfg(tmp_path)
    cfg["paths"]["skill_aliases"] = str(aliases_file)

    results = [
        _result("r1", _full_data(
            technical_skills=[
                {"name": "Python", "source": "Python"},
                {"name": "Haskell", "source": "functional programming paradigm"},
            ],
            nice_to_have_skills=[
                {"name": "Erlang", "source": "Erlang OTP framework"},
            ],
        )),
    ]
    df_rows = [_row("r1", description="We use Python for our backend.", title="Python Developer")]

    final, _ = run_validators(results, df_rows, cfg)

    # Only Python should survive — Haskell and Erlang sources not in description
    assert final[0]["data"]["technical_skills"] == ["Python"]
    assert final[0]["data"]["nice_to_have_skills"] == []


def test_run_validators_german_source_grounded(tmp_path, aliases_file):
    """German source quotes found in German description → skills preserved."""
    cfg = _cfg(tmp_path)
    cfg["paths"]["skill_aliases"] = str(aliases_file)

    results = [
        _result("r1", _full_data(
            technical_skills=[
                {"name": "Systems Engineering", "source": "Systemingenieur"},
                {"name": "Sensor Fusion", "source": "Multi Sensor Daten Fusion"},
            ],
            nice_to_have_skills=[],
        )),
    ]
    df_rows = [_row(
        "r1",
        description="Wir suchen einen Systemingenieur für Multi Sensor Daten Fusion.",
        title="Systemingenieur",
    )]

    final, _ = run_validators(results, df_rows, cfg)

    skills = final[0]["data"]["technical_skills"]
    assert "Sensor Fusion" in skills
    assert "Systems Engineering" in skills
