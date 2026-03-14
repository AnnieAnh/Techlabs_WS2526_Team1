"""Description quality flagging and skill frequency analysis for the cleaning pipeline."""

import json
import re
from collections import Counter
from typing import Any

import pandas as pd

_CONCAT_PATTERN = re.compile(r"[a-z][A-Z]")


def flag_description_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Flag descriptions with too many camelCase transitions (HTML-strip artefacts).

    Adds a 'description_quality' column: 'concatenated' if the description
    has more than 10 lowercase→uppercase letter transitions, otherwise 'clean'.

    Args:
        df: DataFrame with a 'description' column.

    Returns:
        DataFrame with new 'description_quality' column.
    """
    if "description" not in df.columns:
        return df
    df = df.copy()
    df["description_quality"] = df["description"].apply(
        lambda d: "concatenated" if len(_CONCAT_PATTERN.findall(str(d))) > 10 else "clean"
    )
    return df


def compute_skill_frequencies(
    df: pd.DataFrame,
    col: str,
    top_n: int = 50,
) -> list[dict[str, Any]]:
    """Compute skill frequency statistics across all rows.

    Args:
        df: DataFrame with JSON list skill columns.
        col: Column name to analyse.
        top_n: Maximum number of skills to return.

    Returns:
        List of dicts with 'skill', 'count', and 'pct' keys.
    """
    if col not in df.columns:
        return []
    skill_counter: Counter[str] = Counter()
    for val in df[col]:
        try:
            skills = json.loads(str(val))
            if isinstance(skills, list):
                skill_counter.update(s for s in skills if isinstance(s, str))
        except (json.JSONDecodeError, TypeError):
            pass
    n_rows = max(1, len(df))
    return [
        {"skill": skill, "count": count, "pct": round(count / n_rows * 100, 1)}
        for skill, count in skill_counter.most_common(top_n)
    ]
