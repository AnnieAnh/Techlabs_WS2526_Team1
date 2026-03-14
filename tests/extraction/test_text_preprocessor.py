"""Tests for extraction/preprocessing/text_preprocessor.py."""

import pytest

from extraction.preprocessing.text_preprocessor import (
    clean_html,
    clean_special_chars,
    preprocess_description,
)

# ---------------------------------------------------------------------------
# clean_html
# ---------------------------------------------------------------------------

class TestCleanHtml:
    def test_removes_simple_tags(self) -> None:
        assert clean_html("<p>Hello</p>") == " Hello "

    def test_removes_nested_tags(self) -> None:
        result = clean_html("<div><b>Python</b> developer</div>")
        assert "Python" in result
        assert "<" not in result
        assert ">" not in result

    def test_decodes_html_entities(self) -> None:
        assert clean_html("&amp;") == "&"
        assert clean_html("&lt;div&gt;") == "<div>"
        assert clean_html("&uuml;") == "ü"
        assert clean_html("&auml;") == "ä"

    def test_decodes_euro_sign(self) -> None:
        assert clean_html("Salary: &euro;60.000") == "Salary: €60.000"

    def test_passthrough_plain_text(self) -> None:
        text = "Plain text without HTML"
        assert clean_html(text) == text

    def test_removes_self_closing_tags(self) -> None:
        result = clean_html("Line 1<br/>Line 2")
        assert "<" not in result
        assert "Line 1" in result
        assert "Line 2" in result


# ---------------------------------------------------------------------------
# clean_special_chars
# ---------------------------------------------------------------------------

class TestCleanSpecialChars:
    def test_removes_asterisks(self) -> None:
        assert "*" not in clean_special_chars("**Required** skills: Python")

    def test_removes_double_slash(self) -> None:
        assert "//" not in clean_special_chars("See details // apply now")

    def test_removes_triple_slash(self) -> None:
        assert "///" not in clean_special_chars("/// comment block ///")

    def test_removes_standalone_hash(self) -> None:
        result = clean_special_chars("# Requirements\nPython 3.11")
        assert "#" not in result
        assert "Requirements" in result

    @pytest.mark.parametrize("skill", ["C#", "F#"])
    def test_preserves_csharp_and_fsharp(self, skill: str) -> None:
        result = clean_special_chars(f"We use {skill} and .NET")
        assert skill in result

    def test_removes_emojis(self) -> None:
        result = clean_special_chars("We are hiring ✅ join us 🚀")
        assert "✅" not in result
        assert "🚀" not in result
        assert "We are hiring" in result

    def test_removes_pipe(self) -> None:
        assert "|" not in clean_special_chars("Skills | Tools | Experience")

    def test_removes_backtick(self) -> None:
        assert "`" not in clean_special_chars("`python` `docker`")

    def test_preserves_standard_punctuation(self) -> None:
        text = "3+ years experience. Salary: €60,000. (Remote)"
        result = clean_special_chars(text)
        assert "." in result
        assert "," in result
        assert "(" in result
        assert ")" in result
        assert "€" in result

    def test_preserves_alphanumeric(self) -> None:
        text = "Python 3.11 React TypeScript AWS"
        assert clean_special_chars(text) == text

    def test_preserves_german_umlauts(self) -> None:
        text = "Erfahrung in der Softwareentwicklung (München)"
        result = clean_special_chars(text)
        assert "ü" in result
        assert "München" in result


# ---------------------------------------------------------------------------
# preprocess_description
# ---------------------------------------------------------------------------

class TestPreprocessDescription:
    def test_empty_string_passthrough(self) -> None:
        assert preprocess_description("") == ""

    def test_none_equivalent_passthrough(self) -> None:
        # Function only called with str — test falsy empty
        assert preprocess_description("") == ""

    def test_end_to_end_html_and_emoji(self) -> None:
        raw = "<p>We need a <b>Python</b> developer ✅ with 3+ years experience *required*</p>"
        result = preprocess_description(raw)
        assert "Python" in result
        assert "<" not in result
        assert "✅" not in result
        assert "*" not in result

    def test_collapses_excessive_whitespace(self) -> None:
        raw = "Line 1\n\n\n\n\nLine 2"
        result = preprocess_description(raw)
        assert "\n\n\n" not in result

    def test_strips_leading_trailing_whitespace(self) -> None:
        raw = "   Hello world   "
        assert preprocess_description(raw) == "Hello world"

    def test_html_entities_decoded_in_full_pipeline(self) -> None:
        raw = "<p>Geh&auml;lt: &euro;70.000 j&auml;hrlich</p>"
        result = preprocess_description(raw)
        assert "ä" in result
        assert "€" in result
        assert "<" not in result

    def test_plain_text_minimally_altered(self) -> None:
        text = "Senior Python Developer - Munich - Full-time"
        result = preprocess_description(text)
        # Should be returned with whitespace normalised but otherwise intact
        assert "Senior Python Developer" in result
        assert "Munich" in result
        assert "Full-time" in result
