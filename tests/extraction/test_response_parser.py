"""Tests for extraction/llm/response_parser.py — parsing strategies and Tier 2 coercions."""

import json

import pytest

from extraction.llm.response_parser import (
    STRATEGY_DIRECT,
    STRATEGY_EXTRACT_BRACES,
    STRATEGY_FAILED,
    STRATEGY_FIX_COMMAS,
    STRATEGY_STRIP_FENCES,
    _coerce_german_number,
    load_output_schema,
    parse_response,
)


@pytest.fixture(scope="module")
def schema():
    return load_output_schema()


def _minimal_valid() -> dict:
    """Minimal valid Tier 2 extraction output (evidence format)."""
    return {
        "technical_skills": [
            {"name": "Python", "source": "Python development"},
            {"name": "FastAPI", "source": "FastAPI"},
        ],
        "soft_skills": ["teamwork"],
        "nice_to_have_skills": [
            {"name": "Docker", "source": "Docker experience"},
        ],
        "benefits": [
            {"name": "30 Urlaubstage", "source": "30 Urlaubstage"},
        ],
        "tasks": [
            {"name": "Build APIs", "source": "Build APIs"},
            {"name": "Code reviews", "source": "Code reviews"},
        ],
        "job_family": "Backend Developer",
        "job_summary": None,
    }



def _evidence_names(items: list[dict]) -> list[str]:
    """Extract just the names from evidence objects for easy assertions."""
    return [item["name"] for item in items]


def test_strategy_direct(schema):
    raw = json.dumps(_minimal_valid())
    result = parse_response(raw, schema)
    assert result.success
    assert result.parse_strategy == STRATEGY_DIRECT


def test_strategy_strip_fences_backtick(schema):
    raw = "```json\n" + json.dumps(_minimal_valid()) + "\n```"
    result = parse_response(raw, schema)
    assert result.success
    assert result.parse_strategy == STRATEGY_STRIP_FENCES


def test_strategy_strip_fences_plain(schema):
    raw = "```\n" + json.dumps(_minimal_valid()) + "\n```"
    result = parse_response(raw, schema)
    assert result.success
    assert result.parse_strategy in (STRATEGY_STRIP_FENCES, STRATEGY_DIRECT)


def test_strategy_fix_trailing_comma(schema):
    valid = _minimal_valid()
    # Inject trailing comma inside the soft_skills array (flat strings)
    raw = json.dumps(valid).replace('["teamwork"]', '["teamwork",]')
    result = parse_response(raw, schema)
    assert result.success
    assert result.parse_strategy in (STRATEGY_FIX_COMMAS, STRATEGY_EXTRACT_BRACES)


def test_strategy_extract_braces_with_preamble(schema):
    raw = "Here is the extracted data:\n" + json.dumps(_minimal_valid())
    result = parse_response(raw, schema)
    assert result.success
    assert result.parse_strategy in (STRATEGY_EXTRACT_BRACES, STRATEGY_FIX_COMMAS)


def test_strategy_failed_empty(schema):
    result = parse_response("", schema)
    assert not result.success
    assert result.parse_strategy == STRATEGY_FAILED


def test_strategy_failed_garbage(schema):
    result = parse_response("Sorry, I cannot extract this.", schema)
    assert not result.success
    assert result.parse_strategy == STRATEGY_FAILED


def test_strategy_failed_array_not_object(schema):
    """LLM returned an array — extract_braces finds the inner {} and warns about schema."""
    raw = '[{"technical_skills": ["Python"], "job_family": "Backend Developer"}]'
    result = parse_response(raw, schema)
    assert result.parse_strategy != STRATEGY_DIRECT
    assert any("required" in w.lower() for w in result.warnings)


def test_skills_as_string_coerced_to_array(schema):
    data = _minimal_valid()
    data["technical_skills"] = "Python"
    result = parse_response(json.dumps(data), schema)
    assert result.success
    assert isinstance(result.data["technical_skills"], list)
    assert _evidence_names(result.data["technical_skills"]) == ["Python"]


def test_tasks_truncated_to_seven(schema):
    data = _minimal_valid()
    data["tasks"] = [{"name": f"Task {i}", "source": f"source {i}"} for i in range(10)]
    result = parse_response(json.dumps(data), schema)
    assert result.success
    assert len(result.data["tasks"]) == 7


def test_tasks_truncated_emits_list_truncation(schema):
    """10-item task list → truncated to 7 + list_truncations record emitted."""
    data = _minimal_valid()
    data["tasks"] = [{"name": f"Task {i}", "source": f"source {i}"} for i in range(10)]
    result = parse_response(json.dumps(data), schema)
    assert result.success
    assert len(result.list_truncations) == 1
    trunc = result.list_truncations[0]
    assert trunc.field == "tasks"
    assert trunc.original_count == 10
    assert trunc.kept_count == 7


def test_tasks_custom_max_no_truncation(schema):
    """Setting max_tasks=10 → no truncation for 10-item list."""
    data = _minimal_valid()
    data["tasks"] = [{"name": f"Task {i}", "source": f"source {i}"} for i in range(10)]
    result = parse_response(json.dumps(data), schema, max_tasks=10)
    assert result.success
    assert len(result.data["tasks"]) == 10
    assert len(result.list_truncations) == 0


def test_tasks_custom_max_truncation(schema):
    """Setting max_tasks=5 → truncates 10 tasks to 5."""
    data = _minimal_valid()
    data["tasks"] = [{"name": f"Task {i}", "source": f"source {i}"} for i in range(10)]
    result = parse_response(json.dumps(data), schema, max_tasks=5)
    assert result.success
    assert len(result.data["tasks"]) == 5
    assert result.list_truncations[0].kept_count == 5


def test_benefits_as_string_coerced_to_array(schema):
    data = _minimal_valid()
    data["benefits"] = "30 Urlaubstage"
    result = parse_response(json.dumps(data), schema)
    assert result.success
    assert isinstance(result.data["benefits"], list)
    assert _evidence_names(result.data["benefits"]) == ["30 Urlaubstage"]


def test_soft_skills_as_string_coerced_to_array(schema):
    data = _minimal_valid()
    data["soft_skills"] = "teamwork"
    result = parse_response(json.dumps(data), schema)
    assert result.success
    assert isinstance(result.data["soft_skills"], list)


def test_empty_response(schema):
    result = parse_response("", schema)
    assert not result.success
    assert result.error is not None


def test_html_contaminated(schema):
    """Response wrapped in HTML — should still find JSON."""
    raw = "<html><body>" + json.dumps(_minimal_valid()) + "</body></html>"
    result = parse_response(raw, schema)
    assert result.success


@pytest.mark.parametrize("raw, expected", [
    ("85.000", 85000),
    ("100.000", 100000),
    ("1.200.000", 1200000),
    ("80000", 80000),
    ("80,000", 80000),
    ("€ 90.000", 90000),
    (85000, 85000),
    ("85.00", 85),      # 2 digits after dot → decimal (not German thousands), .00 stripped → 85
])
def test_coerce_german_number(raw, expected):
    result = _coerce_german_number(raw)
    assert result == expected


def test_parse_result_success_property(schema):
    r = parse_response(json.dumps(_minimal_valid()), schema)
    assert r.success is True


def test_parse_result_has_warnings_list(schema):
    r = parse_response(json.dumps(_minimal_valid()), schema)
    assert isinstance(r.warnings, list)


def test_job_family_missing_is_non_critical_warning(schema):
    """Missing job_family is a required-field error but non-critical → success=True with warning."""
    data = _minimal_valid()
    del data["job_family"]
    result = parse_response(json.dumps(data), schema)
    assert result.success
    assert any("job_family" in w for w in result.warnings)


def test_job_family_unknown_value_accepted(schema):
    """Unknown job_family string passes schema (no enum constraint) → success=True."""
    data = _minimal_valid()
    data["job_family"] = "Lead Backend Developer"  # not in taxonomy — handled by remap
    result = parse_response(json.dumps(data), schema)
    assert result.success
    assert result.data["job_family"] == "Lead Backend Developer"


def test_job_family_empty_string_rejected(schema):
    """Empty job_family string fails minLength=1 → warning (non-critical)."""
    data = _minimal_valid()
    data["job_family"] = ""
    result = parse_response(json.dumps(data), schema)
    assert result.success
    assert any("job_family" in w for w in result.warnings)


def test_all_schema_errors_collected(schema):
    """Both errors on the same response are collected (not just the first)."""
    # Remove soft_skills AND nice_to_have_skills — two distinct required-field errors
    data = _minimal_valid()
    del data["soft_skills"]
    del data["nice_to_have_skills"]
    result = parse_response(json.dumps(data), schema)
    # These are non-critical (not job_family), so they become warnings
    assert result.success
    assert sum(1 for w in result.warnings if "soft_skills" in w or "nice_to_have_skills" in w) >= 2


def test_non_critical_schema_error_stays_as_warning(schema):
    """Non-critical schema violations → success=True with warning."""
    # tasks is required but not a critical field
    data = _minimal_valid()
    del data["tasks"]
    result = parse_response(json.dumps(data), schema)
    assert result.success
    assert any("tasks" in w for w in result.warnings)


def test_job_summary_null_accepted(schema):
    data = _minimal_valid()
    data["job_summary"] = None
    result = parse_response(json.dumps(data), schema)
    assert result.success
    assert result.data["job_summary"] is None


# ---------------------------------------------------------------------------
# Evidence format tests
# ---------------------------------------------------------------------------


def test_evidence_format_passes_schema(schema):
    """Evidence objects with name+source pass schema validation."""
    result = parse_response(json.dumps(_minimal_valid()), schema)
    assert result.success
    skills = result.data["technical_skills"]
    assert all(isinstance(s, dict) and "name" in s and "source" in s for s in skills)



def test_mixed_string_and_evidence_normalized(schema):
    """Mixed arrays (some strings, some objects) are unified to evidence format."""
    data = _minimal_valid()
    data["technical_skills"] = [
        "Python",
        {"name": "Docker", "source": "Docker-Container"},
    ]
    result = parse_response(json.dumps(data), schema)
    assert result.success
    skills = result.data["technical_skills"]
    assert len(skills) == 2
    assert skills[0] == {"name": "Python", "source": ""}
    assert skills[1] == {"name": "Docker", "source": "Docker-Container"}


def test_evidence_malformed_items_skipped(schema):
    """Malformed items (not string or valid dict) are skipped during coercion."""
    data = _minimal_valid()
    data["technical_skills"] = [
        {"name": "Python", "source": "Python"},
        42,  # malformed
        {"invalid": "no name key"},  # malformed
    ]
    result = parse_response(json.dumps(data), schema)
    assert result.success
    skills = result.data["technical_skills"]
    assert len(skills) == 1
    assert skills[0]["name"] == "Python"
