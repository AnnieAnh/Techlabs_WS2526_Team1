"""Prompt builder: constructs per-row API messages from the extraction prompt template.

Token estimation uses a rough heuristic:
  - English text: ~1 token per 4 characters
  - German text: ~1 token per 3 characters (longer words, more tokens)
  - We use 3.5 chars/token as a conservative estimate for mixed DE/EN content.

Truncation adds "[TRUNCATED]" at the cut point to signal to the LLM
that the description was cut short.
"""

import hashlib
import logging
from pathlib import Path
from typing import Any

import yaml

from extraction.preprocessing.text_preprocessor import preprocess_description
from extraction.reporting.cost import _DEEPSEEK_INPUT_COST_PER_M, _DEEPSEEK_OUTPUT_COST_PER_M

logger = logging.getLogger("pipeline.prompt_builder")

# Conservative chars-per-token for mixed DE/EN job descriptions
_CHARS_PER_TOKEN = 3.5
# DeepSeek V3 pricing — single source of truth is extraction/reporting/cost.py
_COST_PER_INPUT_TOKEN = _DEEPSEEK_INPUT_COST_PER_M / 1_000_000
_COST_PER_OUTPUT_TOKEN = _DEEPSEEK_OUTPUT_COST_PER_M / 1_000_000
_ESTIMATED_OUTPUT_TOKENS = 350  # typical extraction response size


def prompt_version(system_prompt: str) -> str:
    """Return an 8-char MD5 fingerprint of the system prompt.

    Used to record which prompt version produced each extraction result,
    enabling comparisons after prompt changes.

    Args:
        system_prompt: The system prompt string.

    Returns:
        8-character hex string uniquely identifying the prompt content.
    """
    return hashlib.md5(system_prompt.encode()).hexdigest()[:8]


_DEFAULT_PROMPT_PATH = Path(__file__).parent.parent / "config" / "extraction_prompt.yaml"
_DEFAULT_JOB_FAMILIES_PATH = Path(__file__).parent.parent / "config" / "job_families.yaml"

_JOB_FAMILIES_PLACEHOLDER = "<<JOB_FAMILIES_LIST>>"


def _load_job_families(path: Path | None = None) -> list[str]:
    """Load canonical job family list from config/job_families.yaml.

    Args:
        path: Optional override path to the job families YAML.

    Returns:
        List of canonical job family strings.
    """
    jf_path = path or _DEFAULT_JOB_FAMILIES_PATH
    with open(jf_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["families"]


def load_extraction_prompt(
    config_path: Path | None = None,
    job_families_path: Path | None = None,
) -> dict:
    """Load config/extraction_prompt.yaml with job families injected from taxonomy.

    The system prompt placeholder <<JOB_FAMILIES_LIST>> is replaced with the
    full list from config/job_families.yaml, keeping prompt and taxonomy in sync.

    Args:
        config_path: Optional override path to the extraction prompt YAML.
        job_families_path: Optional override path to the job families YAML.

    Returns:
        Prompt config dict with 'system_prompt' and 'user_prompt_template'.
    """
    path = config_path or _DEFAULT_PROMPT_PATH
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    families = _load_job_families(job_families_path)
    families_str = ", ".join(f'"{f}"' for f in families)

    if _JOB_FAMILIES_PLACEHOLDER in config.get("system_prompt", ""):
        config["system_prompt"] = config["system_prompt"].replace(
            _JOB_FAMILIES_PLACEHOLDER, families_str
        )
        logger.debug("Injected %d job families into prompt from taxonomy", len(families))

    return config


def estimate_tokens(text: str) -> int:
    """Rough token count estimate for a text string."""
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def _truncate_description(description: str, max_tokens: int) -> tuple[str, bool]:
    """Truncate description to fit within max_tokens.

    Args:
        description: Full description text.
        max_tokens: Maximum number of tokens allowed.

    Returns:
        Tuple of (possibly-truncated text, was_truncated bool).
    """
    max_chars = int(max_tokens * _CHARS_PER_TOKEN)
    if len(description) <= max_chars:
        return description, False

    truncated = description[:max_chars].rstrip()
    # Try to cut at a sentence boundary within the last 200 chars
    for sep in (". ", ".\n", "! ", "? ", "\n\n"):
        last_sep = truncated.rfind(sep, max(0, len(truncated) - 200))
        if last_sep != -1:
            truncated = truncated[: last_sep + 1]
            break

    return truncated + "\n[TRUNCATED]", True


def build_message(
    row: dict[str, Any],
    prompt_config: dict,
    max_description_tokens: int = 8000,
) -> dict:
    """Build a single API message dict for one job posting row.

    Args:
        row: Dict with at minimum 'title' and 'description' keys.
        prompt_config: Loaded extraction prompt config.
        max_description_tokens: Maximum tokens for the description field.

    Returns:
        Dict with 'system', 'messages', and 'was_truncated' keys.
        When 'was_truncated' is True, callers should append a truncation
        warning to the row's validation_flags.
    """
    title = str(row.get("title_cleaned") or row.get("title") or "")
    description = str(row.get("description") or "")
    description = preprocess_description(description)

    description_truncated, was_truncated = _truncate_description(
        description, max_description_tokens
    )

    if was_truncated:
        logger.debug(
            "Description truncated for row %s: %d → %d chars",
            row.get("row_id", "?"),
            len(description),
            len(description_truncated),
        )

    user_content = prompt_config["user_prompt_template"].format(
        title=title,
        description=description_truncated,
    )

    return {
        "system": prompt_config["system_prompt"],
        "messages": [{"role": "user", "content": user_content}],
        "was_truncated": was_truncated,
    }


def build_batch_messages(
    rows: list[dict[str, Any]],
    prompt_config: dict,
    max_description_tokens: int = 8000,
) -> list[dict]:
    """Build API messages for a list of rows.

    Args:
        rows: List of row dicts.
        prompt_config: Loaded extraction prompt config.
        max_description_tokens: Token budget per description.

    Returns:
        List of message dicts (same order as rows).
    """
    messages = []
    total_input_tokens = 0
    truncated_count = 0

    system_token_est = estimate_tokens(prompt_config["system_prompt"])

    for row in rows:
        msg = build_message(row, prompt_config, max_description_tokens)
        user_text = msg["messages"][0]["content"]
        row_tokens = system_token_est + estimate_tokens(user_text)
        total_input_tokens += row_tokens

        if "[TRUNCATED]" in user_text:
            truncated_count += 1

        messages.append(msg)

    estimated_output_tokens = len(rows) * _ESTIMATED_OUTPUT_TOKENS
    estimated_cost = (
        total_input_tokens * _COST_PER_INPUT_TOKEN
        + estimated_output_tokens * _COST_PER_OUTPUT_TOKEN
    )

    logger.info(
        "Built %d prompts | ~%s input tokens | ~%s output tokens | est. $%.2f",
        len(rows),
        f"{total_input_tokens:,}",
        f"{estimated_output_tokens:,}",
        estimated_cost,
    )
    if truncated_count:
        logger.info(
            "  Descriptions truncated: %d (%.1f%%)",
            truncated_count,
            truncated_count / len(rows) * 100,
        )

    return messages
