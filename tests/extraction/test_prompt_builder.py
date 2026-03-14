"""Tests for extraction/llm/prompt_builder.py."""

import pytest

from extraction.llm.prompt_builder import (
    _truncate_description,
    build_batch_messages,
    build_message,
    estimate_tokens,
    load_extraction_prompt,
)


@pytest.fixture(scope="module")
def prompt_config():
    return load_extraction_prompt()


@pytest.fixture
def sample_row():
    return {
        "row_id": "abc123",
        "title": "Senior Backend Developer (m/w/d)",
        "title_cleaned": "Senior Backend Developer",
        "description": "We are looking for a Senior Backend Developer. " * 50,
        "company_name": "ACME GmbH",
        "location": "Munich, Bavaria, Germany",
    }


# ---------------------------------------------------------------------------
# Prompt config loading
# ---------------------------------------------------------------------------


def test_load_extraction_prompt():
    config = load_extraction_prompt()
    assert "system_prompt" in config
    assert "user_prompt_template" in config
    assert "version" in config


def test_prompt_config_has_examples(prompt_config):
    assert "INPUT" in prompt_config["system_prompt"]
    assert "OUTPUT" in prompt_config["system_prompt"]


def test_job_families_injected_from_taxonomy(prompt_config):
    """Job families from config/job_families.yaml are injected into the system prompt."""
    system = prompt_config["system_prompt"]
    # The placeholder should have been replaced
    assert "<<JOB_FAMILIES_LIST>>" not in system
    # Core families from the taxonomy should be present
    assert '"Backend Developer"' in system
    assert '"Frontend Developer"' in system
    assert '"Other"' in system
    assert '"Data Scientist"' in system


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def test_estimate_tokens_non_zero():
    assert estimate_tokens("Hello world") > 0


def test_estimate_tokens_proportional():
    short = estimate_tokens("short")
    long = estimate_tokens("this is a much longer text that has many more words")
    assert long > short


def test_estimate_tokens_empty():
    assert estimate_tokens("") == 1  # minimum 1


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


def test_no_truncation_for_short_description():
    short = "A short description."
    result, was_truncated = _truncate_description(short, max_tokens=3000)
    assert result == short
    assert not was_truncated


def test_truncation_marks_with_tag():
    long_text = "x" * 20000
    result, was_truncated = _truncate_description(long_text, max_tokens=100)
    assert was_truncated
    assert "[TRUNCATED]" in result


def test_truncation_respects_token_limit():
    long_text = "word " * 5000  # ~25000 chars
    result, _ = _truncate_description(long_text, max_tokens=500)
    # Result should be much shorter than original
    assert len(result) < len(long_text)


def test_truncation_token_boundary():
    """Truncated text should not significantly exceed token limit."""
    long_text = "a" * 10000
    max_tokens = 200
    result, was_truncated = _truncate_description(long_text, max_tokens=max_tokens)
    assert was_truncated
    # Allow up to 50% over due to sentence boundary search
    chars_per_token = 3.5
    assert len(result) <= max_tokens * chars_per_token * 1.5


# ---------------------------------------------------------------------------
# build_message
# ---------------------------------------------------------------------------


def test_build_message_structure(prompt_config, sample_row):
    msg = build_message(sample_row, prompt_config)
    assert "system" in msg
    assert "messages" in msg
    assert len(msg["messages"]) == 1
    assert msg["messages"][0]["role"] == "user"


def test_build_message_contains_title(prompt_config, sample_row):
    msg = build_message(sample_row, prompt_config)
    user_content = msg["messages"][0]["content"]
    assert "Senior Backend Developer" in user_content


def test_build_message_contains_description(prompt_config, sample_row):
    msg = build_message(sample_row, prompt_config)
    user_content = msg["messages"][0]["content"]
    assert "Senior Backend Developer" in user_content


def test_build_message_uses_cleaned_title(prompt_config):
    row = {
        "row_id": "x",
        "title": "Entwickler (m/w/d)",
        "title_cleaned": "Developer",
        "description": "Some description text.",
    }
    msg = build_message(row, prompt_config)
    user_content = msg["messages"][0]["content"]
    assert "Developer" in user_content


def test_build_message_truncates_long_description(prompt_config):
    row = {
        "row_id": "x",
        "title_cleaned": "Engineer",
        "description": "word " * 10000,
    }
    msg = build_message(row, prompt_config, max_description_tokens=100)
    user_content = msg["messages"][0]["content"]
    assert "[TRUNCATED]" in user_content


def test_build_message_no_truncation_for_short(prompt_config):
    row = {
        "row_id": "x",
        "title_cleaned": "Engineer",
        "description": "Short description.",
    }
    msg = build_message(row, prompt_config, max_description_tokens=3000)
    user_content = msg["messages"][0]["content"]
    assert "[TRUNCATED]" not in user_content


def test_build_message_was_truncated_flag_false(prompt_config):
    """Short description → was_truncated is False."""
    row = {
        "row_id": "x",
        "title_cleaned": "Engineer",
        "description": "Short description.",
    }
    msg = build_message(row, prompt_config, max_description_tokens=8000)
    assert msg["was_truncated"] is False


def test_build_message_was_truncated_flag_true(prompt_config):
    """Long description → was_truncated is True."""
    row = {
        "row_id": "x",
        "title_cleaned": "Engineer",
        "description": "word " * 10000,  # ~50,000 chars, well above 8000 tokens
    }
    msg = build_message(row, prompt_config, max_description_tokens=100)
    assert msg["was_truncated"] is True
    assert "[TRUNCATED]" in msg["messages"][0]["content"]


def test_build_message_12k_chars_no_truncation(prompt_config):
    """12,000-char description fits within 8,000 tokens (~28,000 chars)."""
    row = {
        "row_id": "x",
        "title_cleaned": "Engineer",
        "description": "A" * 12000,
    }
    msg = build_message(row, prompt_config, max_description_tokens=8000)
    assert msg["was_truncated"] is False


# ---------------------------------------------------------------------------
# build_batch_messages
# ---------------------------------------------------------------------------


def test_build_batch_messages_count(prompt_config, sample_row):
    rows = [sample_row, sample_row, sample_row]
    messages = build_batch_messages(rows, prompt_config)
    assert len(messages) == 3


def test_build_batch_messages_same_order(prompt_config):
    rows = [
        {"row_id": "a", "title_cleaned": "Engineer A", "description": "desc A"},
        {"row_id": "b", "title_cleaned": "Engineer B", "description": "desc B"},
    ]
    messages = build_batch_messages(rows, prompt_config)
    assert "Engineer A" in messages[0]["messages"][0]["content"]
    assert "Engineer B" in messages[1]["messages"][0]["content"]
