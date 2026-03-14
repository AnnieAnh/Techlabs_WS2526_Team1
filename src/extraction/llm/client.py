"""DeepSeek API client — async, semaphore-limited, with exponential backoff.

Uses the OpenAI-compatible SDK with DeepSeek's base URL.
API key must be in DEEPSEEK_API_KEY environment variable.
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

logger = logging.getLogger("pipeline.llm.client")

_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503})


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Structured response from a DeepSeek API call.

    Attributes:
        text: The generated text content (empty string if no content).
        was_truncated: True if the model hit the token limit (finish_reason="length").
    """

    text: str
    was_truncated: bool = False

_client: AsyncOpenAI | None = None

# Module-level token usage accumulator (reset between runs via reset_token_usage())
_token_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}


def get_token_usage() -> dict[str, int]:
    """Return a copy of the accumulated token usage for this process."""
    return dict(_token_usage)


def reset_token_usage() -> None:
    """Reset the accumulated token usage counters to zero."""
    _token_usage["input_tokens"] = 0
    _token_usage["output_tokens"] = 0


def get_client() -> AsyncOpenAI:
    """Return (or lazily create) the shared AsyncOpenAI client for DeepSeek.

    Raises:
        ValueError: If DEEPSEEK_API_KEY is not set.
    """
    global _client
    if _client is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY not set. "
                "Add it to your .env file or export it before running the pipeline."
            )
        _client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
        logger.debug("DeepSeek AsyncOpenAI client initialised")
    return _client


async def call_deepseek(
    system: str,
    user: str,
    model: str = "deepseek-chat",
    max_tokens: int = 2000,
    temperature: float = 0.0,
    retries: int = 3,
) -> LLMResponse:
    """Single async call to DeepSeek with exponential backoff on 429/5xx errors.

    Args:
        system: System prompt text.
        user: User message text.
        model: DeepSeek model ID (default: deepseek-chat / V3).
        max_tokens: Maximum output tokens.
        temperature: Sampling temperature (0 = deterministic).
        retries: Maximum number of attempts (including the first).

    Returns:
        LLMResponse with text and truncation flag.

    Raises:
        Exception: Re-raises the last exception after all retries are exhausted.
    """
    client = get_client()
    last_exc: Exception | None = None

    for attempt in range(retries):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            choice = resp.choices[0]
            truncated = choice.finish_reason == "length"
            if truncated:
                logger.warning(
                    "Token limit hit (finish_reason=length): model=%s max_tokens=%d",
                    model,
                    max_tokens,
                )
            # Accumulate token usage for cost reporting
            if resp.usage:
                _token_usage["input_tokens"] += resp.usage.prompt_tokens
                _token_usage["output_tokens"] += resp.usage.completion_tokens
            return LLMResponse(
                text=choice.message.content or "",
                was_truncated=truncated,
            )

        except Exception as exc:
            last_exc = exc
            _is_retryable = _check_retryable(exc)

            if attempt == retries - 1 or not _is_retryable:
                raise

            wait = 2**attempt
            logger.warning(
                "Retry %d/%d after %ds: %s: %s",
                attempt + 1,
                retries,
                wait,
                type(exc).__name__,
                exc,
            )
            await asyncio.sleep(wait)

    raise last_exc  # type: ignore[misc]


def _check_retryable(exc: Exception) -> bool:
    """Return True if the exception is worth retrying."""
    try:
        from openai import APIConnectionError, APIStatusError, APITimeoutError

        if isinstance(exc, APIStatusError):
            return exc.status_code in _RETRYABLE_STATUS_CODES
        return isinstance(exc, (APIConnectionError, APITimeoutError))
    except ImportError:
        return False


def reset_client() -> None:
    """Reset the shared client (used in tests to force re-initialisation)."""
    global _client
    _client = None


def build_messages(system: str, user: str) -> list[dict[str, Any]]:
    """Format system + user content as an OpenAI messages list.

    Args:
        system: System prompt text.
        user: User message text.

    Returns:
        List of message dicts suitable for chat.completions.create().
    """
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
