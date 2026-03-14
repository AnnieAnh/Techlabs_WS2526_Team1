"""Tests for extraction/reporting/cost.py."""

import json

import pytest

from extraction.reporting.cost import (
    _DEEPSEEK_INPUT_COST_PER_M,
    _DEEPSEEK_OUTPUT_COST_PER_M,
    build_cost_report,
    read_batch_token_usage,
    save_cost_report,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ext_stats(**overrides) -> dict:
    base = {
        "calls": 10, "input_tokens": 100_000, "output_tokens": 20_000, "retries": 1, "failed": 0,
    }
    base.update(overrides)
    return base


def _title_stats(**overrides) -> dict:
    base = {"calls": 5, "input_tokens": 50_000, "output_tokens": 10_000}
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Report structure
# ---------------------------------------------------------------------------


def test_report_has_required_keys():
    report = build_cost_report(_ext_stats(), None, {})
    for key in ("model", "pricing", "extraction", "title_classification", "total_cost_usd"):
        assert key in report, f"Missing key: {key}"


def test_extraction_section_has_required_keys():
    report = build_cost_report(_ext_stats(), None, {})
    ext = report["extraction"]
    for key in ("calls", "input_tokens", "output_tokens", "cost_usd", "retries", "failed"):
        assert key in ext, f"Missing extraction key: {key}"


def test_pricing_section_present():
    report = build_cost_report(_ext_stats(), None, {})
    assert "input_cost_per_m_tokens" in report["pricing"]
    assert "output_cost_per_m_tokens" in report["pricing"]


# ---------------------------------------------------------------------------
# Cost computation
# ---------------------------------------------------------------------------


def test_extraction_cost_zero_tokens():
    report = build_cost_report(_ext_stats(input_tokens=0, output_tokens=0), None, {})
    assert report["extraction"]["cost_usd"] == 0.0
    assert report["total_cost_usd"] == 0.0


def test_extraction_cost_formula():
    # 1M input + 1M output
    stats = _ext_stats(input_tokens=1_000_000, output_tokens=1_000_000)
    report = build_cost_report(stats, None, {})
    expected = _DEEPSEEK_INPUT_COST_PER_M + _DEEPSEEK_OUTPUT_COST_PER_M
    assert report["extraction"]["cost_usd"] == pytest.approx(expected, abs=0.001)


def test_input_heavy_cost():
    # Only input tokens
    stats = _ext_stats(input_tokens=2_000_000, output_tokens=0)
    report = build_cost_report(stats, None, {})
    expected = 2 * _DEEPSEEK_INPUT_COST_PER_M
    assert report["extraction"]["cost_usd"] == pytest.approx(expected, abs=0.001)


def test_output_heavy_cost():
    stats = _ext_stats(input_tokens=0, output_tokens=1_000_000)
    report = build_cost_report(stats, None, {})
    assert report["extraction"]["cost_usd"] == pytest.approx(_DEEPSEEK_OUTPUT_COST_PER_M, abs=0.001)


# ---------------------------------------------------------------------------
# Title classification stats
# ---------------------------------------------------------------------------


def test_title_stats_none_sets_null():
    report = build_cost_report(_ext_stats(), None, {})
    assert report["title_classification"] is None


def test_title_stats_included_when_provided():
    report = build_cost_report(_ext_stats(), _title_stats(), {})
    assert report["title_classification"] is not None
    assert "cost_usd" in report["title_classification"]


def test_total_cost_includes_title_cost():
    ext = _ext_stats(input_tokens=1_000_000, output_tokens=0)
    title = _title_stats(input_tokens=1_000_000, output_tokens=0)
    report = build_cost_report(ext, title, {})
    expected_total = 2 * _DEEPSEEK_INPUT_COST_PER_M
    assert report["total_cost_usd"] == pytest.approx(expected_total, abs=0.001)


def test_total_cost_without_title():
    stats = _ext_stats(input_tokens=1_000_000, output_tokens=0)
    report = build_cost_report(stats, None, {})
    assert report["total_cost_usd"] == pytest.approx(_DEEPSEEK_INPUT_COST_PER_M, abs=0.001)


# ---------------------------------------------------------------------------
# Metadata fields
# ---------------------------------------------------------------------------


def test_model_name_from_cfg():
    cfg = {"extraction": {"model": "claude-opus-4-6"}}
    report = build_cost_report(_ext_stats(), None, cfg)
    assert report["model"] == "claude-opus-4-6"


def test_model_name_default():
    report = build_cost_report(_ext_stats(), None, {})
    assert report["model"] == "deepseek-chat"


def test_retries_and_failed_preserved():
    stats = _ext_stats(retries=5, failed=2)
    report = build_cost_report(stats, None, {})
    assert report["extraction"]["retries"] == 5
    assert report["extraction"]["failed"] == 2


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def test_save_cost_report_creates_file(tmp_path):
    cfg = {"paths": {"reports_dir": tmp_path / "reports"}}
    report = build_cost_report(_ext_stats(), None, cfg)
    path = save_cost_report(report, cfg)
    assert path.exists()


def test_save_cost_report_valid_json(tmp_path):
    cfg = {"paths": {"reports_dir": tmp_path / "reports"}}
    report = build_cost_report(_ext_stats(), _title_stats(), cfg)
    path = save_cost_report(report, cfg)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "total_cost_usd" in data
    assert data["total_cost_usd"] == report["total_cost_usd"]


def test_save_creates_directory(tmp_path):
    cfg = {"paths": {"reports_dir": tmp_path / "nested" / "reports"}}
    report = build_cost_report(_ext_stats(), None, cfg)
    path = save_cost_report(report, cfg)
    assert path.exists()


# ---------------------------------------------------------------------------
# read_batch_token_usage (reads token_usage.json from extracted_dir)
# ---------------------------------------------------------------------------


def _write_token_usage(path, calls: int, input_tokens: int, output_tokens: int) -> None:
    """Write a fake token_usage.json file."""
    with open(path / "token_usage.json", "w", encoding="utf-8") as f:
        json.dump(
            {"calls": calls, "input_tokens": input_tokens, "output_tokens": output_tokens}, f
        )


def test_read_token_usage_missing_file(tmp_path):
    """No token_usage.json → returns all zeros."""
    result = read_batch_token_usage(tmp_path)
    assert result == {"calls": 0, "input_tokens": 0, "output_tokens": 0}


def test_read_token_usage_reads_json(tmp_path):
    _write_token_usage(tmp_path, calls=7364, input_tokens=50_000_000, output_tokens=5_000_000)
    result = read_batch_token_usage(tmp_path)
    assert result["calls"] == 7364
    assert result["input_tokens"] == 50_000_000
    assert result["output_tokens"] == 5_000_000


def test_read_token_usage_zero_values(tmp_path):
    _write_token_usage(tmp_path, calls=0, input_tokens=0, output_tokens=0)
    result = read_batch_token_usage(tmp_path)
    assert result == {"calls": 0, "input_tokens": 0, "output_tokens": 0}


def test_read_token_usage_malformed_json(tmp_path):
    """Malformed JSON returns all zeros."""
    (tmp_path / "token_usage.json").write_text("not valid json", encoding="utf-8")
    result = read_batch_token_usage(tmp_path)
    assert result == {"calls": 0, "input_tokens": 0, "output_tokens": 0}


def test_read_token_usage_missing_fields(tmp_path):
    """Partial JSON (missing fields) defaults missing keys to 0."""
    (tmp_path / "token_usage.json").write_text(
        json.dumps({"input_tokens": 1000}), encoding="utf-8"
    )
    result = read_batch_token_usage(tmp_path)
    assert result["input_tokens"] == 1000
    assert result["calls"] == 0
    assert result["output_tokens"] == 0
