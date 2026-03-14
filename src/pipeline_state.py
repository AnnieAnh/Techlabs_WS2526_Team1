"""Typed pipeline state dataclass shared by all step modules.

Every step takes ``(state: PipelineState, cfg: dict) -> None`` and mutates
state in place. The ``require_df`` guard prevents silent downstream failures
when a step's prerequisite hasn't run yet.

Example::

    from pipeline_state import PipelineState

    state = PipelineState()
    run_ingest(state, cfg)   # sets state.df
    run_prepare(state, cfg)  # reads state.require_df("prepare")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class PipelineState:
    """Mutable container passed between pipeline steps.

    Attributes:
        df: Main working DataFrame. Starts empty; populated by ``ingest``.
        description_groups: SHA-256 description groups produced by ``deduplicate``.
            Maps group hash → ``{"representative_row_id": str, "member_row_ids": list,
            "count": int}``.
        dedup_report: Row counts before/after each dedup pass, produced by ``deduplicate``.
        extraction_results: List of per-row extraction dicts produced by ``extract``.
        extraction_stats: Summary stats from the extraction step (cost, success rate, …).
        row_limit: Optional cap on rows to process (for testing/debugging).
    """

    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    description_groups: dict[str, Any] | None = None
    dedup_report: dict[str, Any] | None = None
    extraction_results: list[dict[str, Any]] | None = None
    extraction_stats: dict[str, Any] | None = None
    row_limit: int | None = None

    def require_df(self, step_name: str) -> pd.DataFrame:
        """Return ``self.df`` or raise a clear error if it is empty.

        Args:
            step_name: Name of the calling step (used in the error message).

        Returns:
            The non-empty DataFrame.

        Raises:
            RuntimeError: If ``self.df`` is empty (no rows or not yet populated).
        """
        if self.df.empty:
            raise RuntimeError(
                f"Step '{step_name}' requires data but state.df is empty. "
                "Run prior steps first (e.g. 'python orchestrate.py --from ingest')."
            )
        return self.df
