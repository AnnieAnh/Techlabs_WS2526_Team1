"""Text preprocessing for job descriptions before LLM extraction.

Removes HTML markup, HTML entities, emojis, and noise characters
(markdown bullets, comment markers, etc.) that add no value to LLM
extraction and waste context tokens.

Called by prompt_builder.build_message() only — the original
description in the DataFrame is never modified (skill verifier uses
the original text).
"""

import html
import logging
import re

logger = logging.getLogger("pipeline.text_preprocessor")

# HTML tags
_HTML_TAGS = re.compile(r"<[^>]+>")

# Emoji: main Unicode ranges covering most common emoji
_EMOJI = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # misc symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (enclosed letters)
    "\U00002600-\U000027BF"  # misc symbols (☆, ✓, etc.)
    "\U0001F900-\U0001F9FF"  # supplemental symbols & pictographs
    "\U00002702-\U000027B0"  # dingbats subset
    "\U000025A0-\U000025FF"  # geometric shapes (▶, ►, ▸, etc.)
    "\U00002190-\U000021FF"  # arrows (→, ←, etc.)
    "]+",
    flags=re.UNICODE,
)

# Double (or more) slashes: // or /// — comment/markdown artifacts
_DOUBLE_SLASH = re.compile(r"//+")

# Hash (#) used as Markdown headings or bullet separators.
# Preserve C# and F# skill names via negative lookbehind.
_HASH_MARKDOWN = re.compile(r"(?<![CF])#")

# Asterisk (*) used as bullet points or markdown bold/italic markers
_ASTERISK = re.compile(r"\*+")

# Remaining noise characters: pipe, backslash, tilde, backtick
_NOISE_CHARS = re.compile(r"[|\\~`]")

# Whitespace normalisation — collapse multiple blank lines and horizontal space
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")


def clean_html(text: str) -> str:
    """Remove HTML tags and decode HTML entities.

    Args:
        text: Raw text possibly containing HTML markup.

    Returns:
        Text with tags replaced by spaces and entities decoded.
    """
    text = _HTML_TAGS.sub(" ", text)
    text = html.unescape(text)
    return text


def clean_special_chars(text: str) -> str:
    """Remove emojis and markdown/comment noise characters.

    Preserves:
    - Standard punctuation (. , - : ; ! ? ( ) ' ")
    - Currency symbols (€, $, £)
    - Alphanumeric characters including accented/umlauted letters
    - C# and F# (lookbehind prevents stripping skill-name hashes)

    Args:
        text: Text to clean.

    Returns:
        Text with noise characters replaced by spaces.
    """
    text = _EMOJI.sub(" ", text)
    text = _DOUBLE_SLASH.sub(" ", text)
    text = _HASH_MARKDOWN.sub(" ", text)
    text = _ASTERISK.sub(" ", text)
    text = _NOISE_CHARS.sub(" ", text)
    return text


def _normalize_whitespace(text: str) -> str:
    """Collapse excessive whitespace while preserving paragraph breaks."""
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def preprocess_description(text: str) -> str:
    """Clean a job description for LLM input.

    Pipeline: HTML removal → entity decoding → emoji/noise removal →
    whitespace normalisation.

    This function is the single entry point for preprocessing. It is
    called by prompt_builder.build_message() before truncation. The
    original description stored in the DataFrame is never modified.

    Args:
        text: Raw job description text (may contain HTML, emojis, etc.).

    Returns:
        Cleaned description text. Returns the input unchanged if falsy.
    """
    if not text:
        return text
    text = clean_html(text)
    text = clean_special_chars(text)
    text = _normalize_whitespace(text)
    return text
