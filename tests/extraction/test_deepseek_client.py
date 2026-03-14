"""Tests for extraction/llm/client.py.

Async tests are wrapped with asyncio.run() since pytest-asyncio is not in the
project's dev dependencies. This avoids adding another dependency.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from extraction.llm.client import (
    LLMResponse,
    _check_retryable,
    build_messages,
    call_deepseek,
    get_client,
    reset_client,
    reset_token_usage,
)


@pytest.fixture(autouse=True)
def _clean_token_state():
    """Reset token usage before/after each test to prevent cross-contamination."""
    reset_token_usage()
    yield
    reset_token_usage()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_choice(content: str = '{"a": 1}', finish_reason: str = "stop") -> MagicMock:
    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.message.content = content
    return choice


def _make_response(content: str = '{"a": 1}', finish_reason: str = "stop") -> MagicMock:
    resp = MagicMock()
    resp.choices = [_make_choice(content, finish_reason)]
    # Set usage fields to real ints — MagicMock defaults would corrupt the
    # module-level _token_usage dict (int + MagicMock = MagicMock).
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 5
    return resp


def _run(coro):
    """Convenience wrapper so each test reads clearly."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# get_client
# ---------------------------------------------------------------------------

def test_get_client_raises_without_key(monkeypatch) -> None:
    reset_client()
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        get_client()
    reset_client()


def test_get_client_caches_instance(monkeypatch) -> None:
    reset_client()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    # Remove proxy env vars that cause socksio ImportError
    for var in ("ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy"):
        monkeypatch.delenv(var, raising=False)
    c1 = get_client()
    c2 = get_client()
    assert c1 is c2
    reset_client()


# ---------------------------------------------------------------------------
# build_messages
# ---------------------------------------------------------------------------

def test_build_messages_format() -> None:
    msgs = build_messages("sys", "usr")
    assert msgs == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]


# ---------------------------------------------------------------------------
# _check_retryable
# ---------------------------------------------------------------------------

def test_check_retryable_non_api_error() -> None:
    assert not _check_retryable(ValueError("not retryable"))


# ---------------------------------------------------------------------------
# call_deepseek — happy path
# ---------------------------------------------------------------------------

def test_call_deepseek_returns_llm_response(monkeypatch) -> None:
    reset_client()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=_make_response('{"ok": true}'))
    with patch("extraction.llm.client._client", mock_client):
        result = _run(call_deepseek("sys", "usr"))
    assert isinstance(result, LLMResponse)
    assert result.text == '{"ok": true}'
    assert result.was_truncated is False
    reset_client()


def test_call_deepseek_finish_reason_length_sets_truncated(monkeypatch, caplog) -> None:
    reset_client()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_make_response("partial", "length")
    )
    with patch("extraction.llm.client._client", mock_client):
        with caplog.at_level(logging.WARNING, logger="pipeline.llm.client"):
            result = _run(call_deepseek("sys", "usr"))
    assert isinstance(result, LLMResponse)
    assert result.text == "partial"
    assert result.was_truncated is True
    assert "finish_reason=length" in caplog.text
    reset_client()


def test_call_deepseek_none_content_returns_empty(monkeypatch) -> None:
    reset_client()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    mock_client = AsyncMock()
    resp = _make_response()
    resp.choices[0].message.content = None
    mock_client.chat.completions.create = AsyncMock(return_value=resp)
    with patch("extraction.llm.client._client", mock_client):
        result = _run(call_deepseek("sys", "usr"))
    assert result.text == ""
    assert result.was_truncated is False
    reset_client()


def test_call_deepseek_raises_after_retries_exhausted(monkeypatch) -> None:
    reset_client()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("network"))
    with patch("extraction.llm.client._client", mock_client):
        with patch("extraction.llm.client._check_retryable", return_value=False):
            with pytest.raises(RuntimeError, match="network"):
                _run(call_deepseek("sys", "usr", retries=2))
    reset_client()


def test_call_deepseek_retries_on_retryable_error(monkeypatch) -> None:
    reset_client()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    mock_client = AsyncMock()
    # Fail once, succeed on second attempt
    mock_client.chat.completions.create = AsyncMock(
        side_effect=[RuntimeError("transient"), _make_response("ok")]
    )
    with patch("extraction.llm.client._client", mock_client):
        with patch("extraction.llm.client._check_retryable", return_value=True):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = _run(call_deepseek("sys", "usr", retries=3))
    assert result.text == "ok"
    assert result.was_truncated is False
    reset_client()
