"""Tests for extraction/llm/processor.py."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from extraction.llm.client import LLMResponse
from extraction.llm.processor import run_extraction

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_cfg(tmp_path: Path) -> dict:
    """Minimal pipeline config pointing at tmp dirs."""
    return {
        "extraction": {
            "model": "deepseek-chat",
            "max_tokens": 2000,
            "max_description_tokens": 3000,
        },
        "llm": {"deepseek_max_workers": 2},
        "paths": {
            "extracted_dir": tmp_path / "extracted",
            "failed_dir": tmp_path / "failed",
        },
    }


def _make_df(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "row_id": [f"row{i:03d}" for i in range(n)],
        "title": [f"Job {i}" for i in range(n)],
        "description": [f"Some description {i}" * 20 for i in range(n)],
    })


def _mock_checkpoint(completed: list[str] | None = None) -> MagicMock:
    cp = MagicMock()
    cp.get_completed.return_value = completed or []
    return cp


_VALID_DATA = {
    "contract_type": "Full-time",
    "work_modality": "Hybrid",
    "seniority": "Senior",
    "salary_min": 80000,
    "salary_max": 100000,
    "experience_years": 5,
    "technical_skills": ["Python"],
    "soft_skills": ["communication"],
    "nice_to_have_skills": [],
    "benefits": ["30 days PTO"],
    "tasks": ["Write code"],
    "job_family": "Backend Developer",
    "seniority_from_title": "Senior",
}


# ---------------------------------------------------------------------------
# run_extraction — all rows already done (resume skip)
# ---------------------------------------------------------------------------

def test_run_extraction_skips_when_all_done(tmp_cfg, tmp_path) -> None:
    existing = [{"row_id": "row000", "data": _VALID_DATA}]
    results_path = tmp_cfg["paths"]["extracted_dir"]
    results_path.mkdir(parents=True)
    (results_path / "extraction_results.json").write_text(
        json.dumps(existing), encoding="utf-8"
    )

    cp = _mock_checkpoint(completed=["row000"])
    df = _make_df(1)

    results, report = run_extraction(df, tmp_cfg, cp)
    assert report == {"skipped": True}
    assert results == existing


# ---------------------------------------------------------------------------
# run_extraction — happy path (2 rows succeed)
# ---------------------------------------------------------------------------

def test_run_extraction_success_path(tmp_cfg, monkeypatch) -> None:
    cp = _mock_checkpoint()
    df = _make_df(2)

    raw_json = json.dumps(_VALID_DATA)

    async def _fake_call(*args, **kwargs) -> LLMResponse:
        return LLMResponse(text=raw_json)

    with patch("extraction.llm.processor.call_deepseek", side_effect=_fake_call):
        with patch("extraction.llm.processor.load_extraction_prompt") as mock_prompt:
            mock_prompt.return_value = {
                "system_prompt": "sys",
                "user_prompt_template": "title: {title}\n\n{description}",
            }
            results, report = run_extraction(df, tmp_cfg, cp)

    assert report["successes"] == 2
    assert report["failures"] == 0
    assert report["success_rate"] == 1.0
    assert len(results) == 2
    assert cp.advance_stage.call_count == 2


# ---------------------------------------------------------------------------
# run_extraction — API error path
# ---------------------------------------------------------------------------

def test_run_extraction_api_error_path(tmp_cfg) -> None:
    cp = _mock_checkpoint()
    df = _make_df(1)

    async def _raise(*args, **kwargs) -> str:
        raise RuntimeError("timeout")

    with patch("extraction.llm.processor.call_deepseek", side_effect=_raise):
        with patch("extraction.llm.processor.load_extraction_prompt") as mock_prompt:
            mock_prompt.return_value = {
                "system_prompt": "sys",
                "user_prompt_template": "title: {title}\n\n{description}",
            }
            results, report = run_extraction(df, tmp_cfg, cp)

    assert report["failures"] == 1
    assert report["successes"] == 0
    cp.mark_failed.assert_called_once()


# ---------------------------------------------------------------------------
# run_extraction — parse failure path
# ---------------------------------------------------------------------------

def test_run_extraction_parse_failure_path(tmp_cfg) -> None:
    cp = _mock_checkpoint()
    df = _make_df(1)

    async def _bad_json(*args, **kwargs) -> LLMResponse:
        return LLMResponse(text="this is not json at all and is longer than thirty chars")

    with patch("extraction.llm.processor.call_deepseek", side_effect=_bad_json):
        with patch("extraction.llm.processor.load_extraction_prompt") as mock_prompt:
            mock_prompt.return_value = {
                "system_prompt": "sys",
                "user_prompt_template": "title: {title}\n\n{description}",
            }
            results, report = run_extraction(df, tmp_cfg, cp)

    assert report["failures"] == 1
    failed_path = tmp_cfg["paths"]["failed_dir"] / "parse_failures.json"
    assert failed_path.exists()


# ---------------------------------------------------------------------------
# run_extraction — resume merge: prior results preserved
# ---------------------------------------------------------------------------

def test_run_extraction_resume_merges_results(tmp_cfg) -> None:
    """Partial run then resume must produce merged results, not truncated ones."""
    prior = [{"row_id": "row000", "data": _VALID_DATA, "prompt_version": "abc"}]
    extracted_dir = tmp_cfg["paths"]["extracted_dir"]
    extracted_dir.mkdir(parents=True)
    (extracted_dir / "extraction_results.json").write_text(
        json.dumps(prior), encoding="utf-8"
    )

    # row000 already done; row001 is pending
    cp = _mock_checkpoint(completed=["row000"])
    df = _make_df(2)

    raw_json = json.dumps(_VALID_DATA)

    async def _fake_call(*args, **kwargs) -> LLMResponse:
        return LLMResponse(text=raw_json)

    with patch("extraction.llm.processor.call_deepseek", side_effect=_fake_call):
        with patch("extraction.llm.processor.load_extraction_prompt") as mock_prompt:
            mock_prompt.return_value = {
                "system_prompt": "sys",
                "user_prompt_template": "title: {title}\n\n{description}",
            }
            results, report = run_extraction(df, tmp_cfg, cp)

    # Both row000 (prior) and row001 (new) must be in results
    result_ids = {r["row_id"] for r in results}
    assert "row000" in result_ids
    assert "row001" in result_ids
    assert len(results) == 2
    # Report totals reflect merged state
    assert report["total_rows"] == 2
    assert report["new_in_this_run"] == 1


def test_run_extraction_resume_merges_failures(tmp_cfg) -> None:
    """Prior failures must be preserved and not duplicated on resume."""
    prior_failure = [{"row_id": "row000", "error": "prior error", "failure_type": "api_error"}]
    failed_dir = tmp_cfg["paths"]["failed_dir"]
    failed_dir.mkdir(parents=True)
    (failed_dir / "parse_failures.json").write_text(
        json.dumps(prior_failure), encoding="utf-8"
    )

    cp = _mock_checkpoint(completed=[])
    df = _make_df(1)

    async def _raise(*args, **kwargs) -> str:
        raise RuntimeError("timeout")

    with patch("extraction.llm.processor.call_deepseek", side_effect=_raise):
        with patch("extraction.llm.processor.load_extraction_prompt") as mock_prompt:
            mock_prompt.return_value = {
                "system_prompt": "sys",
                "user_prompt_template": "title: {title}\n\n{description}",
            }
            run_extraction(df, tmp_cfg, cp)

    failures_path = tmp_cfg["paths"]["failed_dir"] / "parse_failures.json"
    merged = json.loads(failures_path.read_text(encoding="utf-8"))
    # row000 (prior) + row000 new failure — but new row_id is row000 too here, so deduped
    row_ids = [f["row_id"] for f in merged]
    # Prior failure must be present; no duplicates
    assert row_ids.count("row000") == 1


def test_run_extraction_resume_accumulates_token_usage(tmp_cfg) -> None:
    """Token usage must accumulate across resumed runs, not be overwritten."""
    extracted_dir = tmp_cfg["paths"]["extracted_dir"]
    extracted_dir.mkdir(parents=True)
    prior_usage = {"calls": 100, "input_tokens": 50_000, "output_tokens": 5_000}
    (extracted_dir / "token_usage.json").write_text(
        json.dumps(prior_usage), encoding="utf-8"
    )

    cp = _mock_checkpoint(completed=[])
    df = _make_df(1)

    raw_json = json.dumps(_VALID_DATA)

    async def _fake_call(*args, **kwargs) -> LLMResponse:
        return LLMResponse(text=raw_json)

    with patch("extraction.llm.processor.call_deepseek", side_effect=_fake_call):
        with patch("extraction.llm.processor.load_extraction_prompt") as mock_prompt:
            mock_prompt.return_value = {
                "system_prompt": "sys",
                "user_prompt_template": "title: {title}\n\n{description}",
            }
            run_extraction(df, tmp_cfg, cp)

    usage = json.loads((extracted_dir / "token_usage.json").read_text(encoding="utf-8"))
    # Must be >= prior values (accumulation)
    assert usage["calls"] >= prior_usage["calls"]
    assert usage["input_tokens"] >= prior_usage["input_tokens"]
    assert usage["output_tokens"] >= prior_usage["output_tokens"]


def test_run_extraction_uses_cfg_temperature(tmp_cfg) -> None:
    """Temperature must be read from cfg, not hardcoded."""
    tmp_cfg["extraction"]["temperature"] = 0.7
    cp = _mock_checkpoint(completed=[])
    df = _make_df(1)

    captured: dict = {}

    async def _capture(*args, **kwargs) -> LLMResponse:
        captured["temperature"] = kwargs.get("temperature")
        return LLMResponse(text=json.dumps(_VALID_DATA))

    with patch("extraction.llm.processor.call_deepseek", side_effect=_capture):
        with patch("extraction.llm.processor.load_extraction_prompt") as mock_prompt:
            mock_prompt.return_value = {
                "system_prompt": "sys",
                "user_prompt_template": "title: {title}\n\n{description}",
            }
            run_extraction(df, tmp_cfg, cp)

    assert captured.get("temperature") == pytest.approx(0.7)
