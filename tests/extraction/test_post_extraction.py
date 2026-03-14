"""Tests for extraction/post_extraction.py — C++ inference fix and skill casing."""

import pytest

from extraction.post_extraction import (
    apply_fix_cpp_inference,
    apply_normalize_skill_casing,
)


def _result(row_id: str, skills: list, nice: list | None = None) -> dict:
    """Build a minimal extraction result dict."""
    return {
        "row_id": row_id,
        "data": {
            "technical_skills": skills,
            "nice_to_have_skills": nice or [],
        },
    }


# ---------------------------------------------------------------------------
# C++ inference fix — case-sensitive uppercase C detection
# ---------------------------------------------------------------------------


class TestFixCppInference:
    def test_genuine_cpp_unchanged(self):
        """Description contains 'C++' → skill list unchanged."""
        results = [_result("r1", ["C++", "Python"])]
        desc = {"r1": "We use C++ for performance-critical code."}
        out = apply_fix_cpp_inference(results, desc)
        assert "C++" in out[0]["data"]["technical_skills"]

    def test_bare_uppercase_c_replaces_cpp(self):
        """Description has bare uppercase C → C++ replaced with C."""
        results = [_result("r1", ["C++", "Python"])]
        desc = {"r1": "Experience with the C programming language is required."}
        out = apply_fix_cpp_inference(results, desc)
        skills = out[0]["data"]["technical_skills"]
        assert "C" in skills
        assert "C++" not in skills
        assert "Python" in skills

    def test_no_c_at_all_removes_cpp(self):
        """No C or C++ in description → C++ removed entirely."""
        results = [_result("r1", ["C++", "Python"])]
        desc = {"r1": "Experience with Python and Java required."}
        out = apply_fix_cpp_inference(results, desc)
        skills = out[0]["data"]["technical_skills"]
        assert "C++" not in skills
        assert "C" not in skills
        assert "Python" in skills

    def test_lowercase_c_no_false_positive(self):
        """Lowercase 'c' in words must NOT trigger bare-C detection."""
        results = [_result("r1", ["C++", "Python"])]
        desc = {"r1": "Scrum ceremonies and agile practices required."}
        out = apply_fix_cpp_inference(results, desc)
        skills = out[0]["data"]["technical_skills"]
        # 'Scrum' has lowercase 'c' preceded by 'S' — should not match
        # No uppercase bare C → C++ removed
        assert "C++" not in skills
        assert "C" not in skills

    def test_c_in_slash_notation_not_bare(self):
        """'C/' notation should not match bare C (lookahead excludes '/')."""
        results = [_result("r1", ["C++", "Python"])]
        desc = {"r1": "Knowledge of C/C++ is beneficial."}
        out = apply_fix_cpp_inference(results, desc)
        # 'C/C++' contains 'c++' (case-insensitive) → genuine C++
        assert "C++" in out[0]["data"]["technical_skills"]

    def test_nice_to_have_also_fixed(self):
        """C++ in nice_to_have_skills is also corrected."""
        results = [_result("r1", ["Python"], nice=["C++"])]
        desc = {"r1": "Experience in C and embedded systems."}
        out = apply_fix_cpp_inference(results, desc)
        nice = out[0]["data"]["nice_to_have_skills"]
        assert "C" in nice
        assert "C++" not in nice

    def test_markdown_escaped_cpp_preserved(self):
        r"""Description has markdown-escaped C\+\+ → skill list unchanged."""
        results = [_result("r1", ["C++", "Python"])]
        desc = {"r1": r"Experience with C\+\+ and modern C\+\+ standards required."}
        out = apply_fix_cpp_inference(results, desc)
        assert "C++" in out[0]["data"]["technical_skills"]
        assert "Python" in out[0]["data"]["technical_skills"]

    def test_markdown_escaped_cpp_single_mention(self):
        r"""Single markdown-escaped C\+\+ mention → still recognised as genuine."""
        results = [_result("r1", ["C++"])]
        desc = {"r1": r"Wir suchen einen C\+\+ Entwickler."}
        out = apply_fix_cpp_inference(results, desc)
        assert "C++" in out[0]["data"]["technical_skills"]

    @pytest.mark.parametrize("desc_text", [
        "Programming in C is required.",
        "Kenntnisse in C sind erforderlich.",
        "Must know C, Python, and Java.",
        "C language experience.",
    ])
    def test_various_bare_c_patterns(self, desc_text):
        """Various patterns with uppercase bare C should all match."""
        results = [_result("r1", ["C++"])]
        desc = {"r1": desc_text}
        out = apply_fix_cpp_inference(results, desc)
        assert "C" in out[0]["data"]["technical_skills"]
        assert "C++" not in out[0]["data"]["technical_skills"]


# ---------------------------------------------------------------------------
# Skill casing normalization
# ---------------------------------------------------------------------------


class TestNormalizeSkillCasing:
    def test_most_frequent_wins(self):
        """Most common casing variant becomes canonical."""
        results = [
            _result("r1", ["python", "React"]),
            _result("r2", ["Python", "react"]),
            _result("r3", ["Python", "React"]),
        ]
        out = apply_normalize_skill_casing(results)
        # "Python" appears 2x, "python" 1x → canonical is "Python"
        assert out[0]["data"]["technical_skills"][0] == "Python"

    def test_no_change_when_consistent(self):
        """All same casing → no changes."""
        results = [
            _result("r1", ["Python", "React"]),
            _result("r2", ["Python", "React"]),
        ]
        out = apply_normalize_skill_casing(results)
        assert out[0]["data"]["technical_skills"] == ["Python", "React"]
