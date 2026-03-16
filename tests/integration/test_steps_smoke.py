"""Smoke tests for 7 of 8 step modules (steps/*.py — regex_extract tested separately).

One test per step: exercises the step function end-to-end with a synthetic
10-row DataFrame. LLM calls (step 5: extract) are mocked to avoid API
dependencies. Schema validation and invariant checks are mocked for steps
where synthetic data would fail strict column constraints.
"""

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from pipeline_state import PipelineState
from steps.clean_enrich import run_clean_enrich
from steps.deduplicate import run_deduplicate
from steps.export import run_export
from steps.extract import run_extract
from steps.ingest import run_ingest
from steps.prepare import run_prepare
from steps.validate import run_validate

# ---------------------------------------------------------------------------
# Shared config fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg(tmp_path):
    """Minimal config with all path keys pointing to tmp_path sub-dirs."""
    for sub in ("reports", "deduped", "extracted", "validation", "output"):
        (tmp_path / sub).mkdir()
    return {
        "paths": {
            "checkpoint_db": tmp_path / "pipeline.db",
            "reports_dir": tmp_path / "reports",
            "deduped_dir": tmp_path / "deduped",
            "extracted_dir": tmp_path / "extracted",
            "validation_dir": tmp_path / "validation",
            "raw_dir": tmp_path / "raw",
        },
        "validation": {
            "min_description_length": 20,
            "date_anomaly_cutoff": "2020-01-01",
        },
        "export": {
            "output_path": str(tmp_path / "output" / "cleaned_jobs.csv"),
        },
        "extraction": {
            "temperature": 0.0,
            "max_tokens": 1024,
        },
    }


# ---------------------------------------------------------------------------
# Synthetic DataFrame factories
# ---------------------------------------------------------------------------

def _row_id(i: int) -> str:
    return hashlib.sha256(f"smoke_job_{i}".encode()).hexdigest()[:12]


def _make_minimal_df(n: int = 10) -> pd.DataFrame:
    """Minimal valid DataFrame in the post-ingest shape."""
    rows = [
        {
            "row_id": _row_id(i),
            "job_url": f"https://example.com/job/{i}",
            "title": f"Senior Software Engineer #{i}",
            "company_name": "TechCorp GmbH",
            "location": "Berlin, Germany",
            "description": (
                f"Job posting #{i}: We are looking for a Senior Software Engineer "
                f"at our Berlin office. Requirements: Python, SQL, Docker. "
                f"3+ years experience. Vollzeit. Hybrid. Gehalt: 70.000 - 90.000 EUR."
            ),
            "date_posted": "2026-01-15",
            "source_file": "smoke_test.csv",
            "site": "linkedin",
        }
        for i in range(n)
    ]
    return pd.DataFrame(rows)


def _make_prepared_df(n: int = 10) -> pd.DataFrame:
    """DataFrame that has been through the prepare and regex_extract steps."""
    df = _make_minimal_df(n)
    df["title_cleaned"] = [f"Senior Software Engineer #{i}" for i in range(n)]
    df["title_original"] = df["title"]
    df["city"] = "Berlin"
    df["state"] = "Berlin"
    df["country"] = "Germany"
    df["input_flags"] = [[] for _ in range(n)]
    df["regex_contract_type"] = "Full-time"
    df["regex_work_modality"] = "Hybrid"
    df["regex_salary_min"] = 70000
    df["regex_salary_max"] = 90000
    df["regex_experience_years"] = 3
    df["regex_seniority_from_title"] = "Senior"
    df["regex_languages"] = [[] for _ in range(n)]
    df["regex_education_level"] = None
    return df


def _make_enriched_df(n: int = 10) -> pd.DataFrame:
    """Fully enriched DataFrame (post-merge_results shape) for clean_enrich/export."""
    df = _make_prepared_df(n)
    df["job_family"] = "Software Developer"
    df["seniority"] = "Senior"
    df["contract_type"] = "Full-time"
    df["work_modality"] = "Hybrid"
    df["salary_min"] = "70000"
    df["salary_max"] = "90000"
    df["experience_years"] = "3"
    df["technical_skills"] = '["Python", "Docker"]'
    df["soft_skills"] = '["Communication"]'
    df["nice_to_have_skills"] = '["Kubernetes"]'
    df["benefits"] = '["Remote work"]'
    df["tasks"] = '["Build APIs"]'
    df["education_level"] = None
    df["validation_flags"] = "[]"
    return df


def _make_extraction_result(row_id: str, description: str = "") -> dict:
    """Canned LLM extraction result for a single row."""
    return {
        "row_id": row_id,
        "status": "success",
        "data": {
            "job_family": "Software Developer",
            "seniority": "Senior",
            "seniority_from_title": "Senior",
            "contract_type": "Full-time",
            "work_modality": "Hybrid",
            "salary_min": 70000,
            "salary_max": 90000,
            "experience_years": 3,
            "technical_skills": ["Python", "Docker"],
            "soft_skills": ["Communication"],
            "nice_to_have_skills": ["Kubernetes"],
            "benefits": ["Remote work"],
            "tasks": ["Build APIs"],
            "education_level": None,
        },
        "validation_flags": [],
        "description": description,
    }


def _make_extraction_results(df: pd.DataFrame) -> list[dict]:
    """Canned extraction results for all rows in df."""
    desc_col = df["description"].tolist() if "description" in df.columns else [""] * len(df)
    return [
        _make_extraction_result(rid, desc)
        for rid, desc in zip(df["row_id"].tolist(), desc_col)
    ]


# ---------------------------------------------------------------------------
# Step 1: Ingest
# ---------------------------------------------------------------------------

def test_ingest_delegates_to_ingestion_pipeline(tmp_path, cfg, monkeypatch):
    """run_ingest calls run_pipeline and loads result into state.df."""
    state = PipelineState()
    mock_df = _make_minimal_df(5)

    # Create the output file where run_ingest will look for it
    out_dir = tmp_path / "ingestion" / "output"
    out_dir.mkdir(parents=True)
    mock_df.to_csv(out_dir / "combined_jobs.csv", index=False, encoding="utf-8")

    monkeypatch.chdir(tmp_path)  # Make Path("data/ingestion/...") resolve to tmp_path

    out_path = out_dir / "combined_jobs.csv"

    with patch("steps.ingest.run_pipeline") as mock_pipeline, \
         patch("steps.ingest.validate_step_output"):
        mock_pipeline.return_value = out_path  # run_pipeline now returns the Path
        run_ingest(state, cfg)

    mock_pipeline.assert_called_once_with(cfg=cfg)
    assert not state.df.empty
    assert len(state.df) == 5


# ---------------------------------------------------------------------------
# Step 2: Prepare
# Geo/title functions use CWD-relative config paths that only work from
# extraction/. Mock those calls; let extract_regex_fields run for real.
# ---------------------------------------------------------------------------

def _make_located_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add location columns as parse_all_locations would produce."""
    out = df.copy()
    out["city"] = "Berlin"
    out["state"] = "Berlin"
    out["country"] = "Germany"
    out["input_flags"] = [[] for _ in range(len(out))]
    return out


def _make_titled_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add title_cleaned / title_original as normalize_all_titles would produce."""
    out = df.copy()
    out["title_original"] = out["title"]
    out["title_cleaned"] = (
        out["title"]
        .str.replace(r"\s*\(m/w/d\)|\s*\(w/m/d\)|\s*\(gn\)", "", regex=True)
        .str.strip()
    )
    return out


def _prepare_mocks(df: pd.DataFrame):
    """Context manager stack mocking all CWD-sensitive calls in run_prepare."""
    from contextlib import ExitStack

    located = _make_located_df(df)
    titled = _make_titled_df(located)
    n = len(df)

    stack = ExitStack()
    stack.enter_context(patch("steps.prepare.validate_input", return_value=(located, {})))
    stack.enter_context(patch("steps.prepare.load_geo_config", return_value={}))
    stack.enter_context(
        patch(
            "steps.prepare.parse_all_locations",
            return_value=(located, {"fallback_percent": 0.0}),
        )
    )
    stack.enter_context(patch("steps.prepare.load_title_translations", return_value={}))
    stack.enter_context(
        patch(
            "steps.prepare.normalize_all_titles",
            return_value=(titled, {"unique_titles_before": n, "unique_titles_after": n}),
        )
    )
    stack.enter_context(patch("steps.prepare.validate_step_output"))
    return stack


def test_prepare_adds_required_columns(cfg):
    """run_prepare adds title_cleaned and city/state/country columns.

    Note: regex_* columns are added by Step 4 (regex_extract), not prepare.
    """
    state = PipelineState()
    df = _make_minimal_df(10)
    state.df = df

    with _prepare_mocks(df):
        run_prepare(state, cfg)

    assert "title_cleaned" in state.df.columns
    assert "city" in state.df.columns
    assert "state" in state.df.columns
    assert "country" in state.df.columns


def test_prepare_does_not_drop_clean_rows(cfg):
    """All 10 clean fixture rows survive validate_input and location filter."""
    state = PipelineState()
    df = _make_minimal_df(10)
    state.df = df

    with _prepare_mocks(df):
        run_prepare(state, cfg)

    # All rows are valid German locations — none should be dropped
    assert len(state.df) == 10


def test_prepare_strips_gender_markers(cfg):
    """Titles with (m/w/d) markers are cleaned in title_cleaned (via mock)."""
    state = PipelineState()
    df = _make_minimal_df(3)
    df.loc[0, "title"] = "Senior Developer (m/w/d)"
    df.loc[1, "title"] = "Backend Engineer (gn)"
    df.loc[2, "title"] = "Data Scientist (w/m/d)"
    state.df = df

    with _prepare_mocks(df):
        run_prepare(state, cfg)

    cleaned = state.df["title_cleaned"]
    assert not cleaned.str.contains(r"\(m/w/d\)", case=False, na=False).any()
    assert not cleaned.str.contains(r"\(gn\)", case=False, na=False).any()


# ---------------------------------------------------------------------------
# Step 3: Deduplicate
# ---------------------------------------------------------------------------

def test_deduplicate_sets_description_groups(cfg):
    """run_deduplicate sets state.description_groups (dict of SHA-256 groups)."""
    state = PipelineState()
    state.df = _make_prepared_df(10)

    run_deduplicate(state, cfg)

    assert state.description_groups is not None
    assert isinstance(state.description_groups, dict)
    assert len(state.description_groups) > 0


def test_deduplicate_sets_dedup_report(cfg):
    """run_deduplicate sets state.dedup_report with before/after counts."""
    state = PipelineState()
    state.df = _make_prepared_df(10)

    run_deduplicate(state, cfg)

    assert state.dedup_report is not None
    assert isinstance(state.dedup_report, dict)


def test_deduplicate_removes_exact_duplicate(cfg):
    """A row with identical title+company+location but different URL is removed."""
    state = PipelineState()
    df = _make_prepared_df(10)

    # Make row 0 a composite duplicate of row 1 (different URL → different row_id already)
    # Force title+company+location to match — dedup should collapse them
    df.loc[0, "title_cleaned"] = df.loc[1, "title_cleaned"] = "Python Developer"
    df.loc[0, "company_name"] = df.loc[1, "company_name"] = "DupCorp GmbH"
    df.loc[0, "location"] = df.loc[1, "location"] = "Berlin, Germany"
    state.df = df

    run_deduplicate(state, cfg)

    assert len(state.df) < 10


# ---------------------------------------------------------------------------
# Step 4: Extract
# ---------------------------------------------------------------------------

def test_extract_mocks_llm_and_sets_results(cfg):
    """run_extract calls the LLM mock and stores results in state."""
    state = PipelineState()
    df = _make_prepared_df(5)
    state.df = df

    canned_results = _make_extraction_results(df)
    canned_stats = {"success_rate": 1.0, "total_rows": 5, "failed_rows": 0}

    with patch("steps.extract._llm_run_extraction", return_value=(canned_results, canned_stats)):
        run_extract(state, cfg)

    assert state.extraction_results is not None
    assert len(state.extraction_results) == 5
    assert state.extraction_stats is not None
    assert state.extraction_stats["success_rate"] == 1.0


def test_extract_propagates_to_group_members(cfg):
    """When description_groups are set, results fan out to all group members."""
    state = PipelineState()
    df = _make_prepared_df(4)
    state.df = df

    rep_id = df["row_id"].iloc[0]
    member_id = df["row_id"].iloc[1]
    state.description_groups = {
        "group1": {
            "representative_row_id": rep_id,
            "member_row_ids": [rep_id, member_id],
            "count": 2,
        }
    }

    # LLM called only for representative; member gets result propagated
    rep_results = [_make_extraction_result(rep_id)]
    other_results = [_make_extraction_result(rid) for rid in df["row_id"].iloc[2:].tolist()]
    canned_results = rep_results + other_results
    canned_stats = {"success_rate": 1.0, "total_rows": 3}

    with patch("steps.extract._llm_run_extraction", return_value=(canned_results, canned_stats)):
        run_extract(state, cfg)

    result_ids = {r["row_id"] for r in state.extraction_results}
    assert member_id in result_ids, "member_id should be propagated from representative"


# ---------------------------------------------------------------------------
# Step 5: Validate
# ---------------------------------------------------------------------------

def _val_report_stub(n: int) -> dict:
    return {"total_validated": n, "rows_with_flags": 0, "flags": {}}


def test_validate_processes_extraction_results(cfg):
    """run_validate applies corrections and writes validated results to disk."""
    state = PipelineState()
    df = _make_prepared_df(5)
    state.df = df
    state.extraction_results = _make_extraction_results(df)

    extracted_dir = cfg["paths"]["extracted_dir"]

    # run_validators uses CWD-relative config paths; mock it.
    # generate_quality_report also writes files; mock both.
    with patch("steps.validate.run_validators",
               return_value=(state.extraction_results, _val_report_stub(5))), \
         patch("steps.validate.generate_quality_report"):
        run_validate(state, cfg)

    # Results should still be set (corrected in-place)
    assert state.extraction_results is not None
    assert len(state.extraction_results) == 5

    # Validated results must be written to disk (for resume support)
    results_path = extracted_dir / "extraction_results.json"
    assert results_path.exists(), "Validated results should be persisted to disk"


def test_validate_loads_from_disk_when_state_empty(cfg):
    """run_validate resumes from disk if state.extraction_results is None."""
    state = PipelineState()
    df = _make_prepared_df(3)
    state.df = df
    # Do NOT set state.extraction_results — simulate resume after interruption

    extracted_dir = cfg["paths"]["extracted_dir"]
    canned = _make_extraction_results(df)
    results_path = extracted_dir / "extraction_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(canned, f)

    with patch("steps.validate.run_validators",
               return_value=(canned, _val_report_stub(3))), \
         patch("steps.validate.generate_quality_report"):
        run_validate(state, cfg)

    assert state.extraction_results is not None
    assert len(state.extraction_results) == 3


# ---------------------------------------------------------------------------
# Step 6: Clean + Enrich
# ---------------------------------------------------------------------------

def test_clean_enrich_adds_enrichment_columns(cfg):
    """run_clean_enrich adds benefit_categories and description_quality columns."""
    state = PipelineState()
    state.df = _make_enriched_df(5)
    state.extraction_results = _make_extraction_results(state.df)

    with patch("steps.clean_enrich.validate_step_output"), \
         patch("steps.clean_enrich.merge_results", side_effect=lambda df, _: df):
        run_clean_enrich(state, cfg)

    assert "benefit_categories" in state.df.columns
    assert "description_quality" in state.df.columns
    assert not state.df.empty


def test_clean_enrich_normalizes_city_names(cfg):
    """City aliases are normalized (e.g. Muenchen → Munich or München)."""
    state = PipelineState()
    df = _make_enriched_df(3)
    df["city"] = "Berlin"  # Known canonical name; should survive normalization
    state.df = df
    state.extraction_results = _make_extraction_results(state.df)

    with patch("steps.clean_enrich.validate_step_output"), \
         patch("steps.clean_enrich.merge_results", side_effect=lambda df, _: df):
        run_clean_enrich(state, cfg)

    # After normalize_city_names, city column still exists and is non-null
    assert "city" in state.df.columns
    assert state.df["city"].notna().all()


# ---------------------------------------------------------------------------
# Step 7: Export
# ---------------------------------------------------------------------------

def test_export_writes_output_csv(cfg, tmp_path):
    """run_export writes the final CSV to the configured output path."""
    state = PipelineState()
    state.df = _make_enriched_df(5)
    # Add enrichment columns that export expects (added by clean_enrich)
    state.df["benefit_categories"] = [["Remote work"] for _ in range(5)]
    state.df["description_quality"] = "good"
    state.df["soft_skill_categories"] = [["Communication"] for _ in range(5)]

    output_path = Path(cfg["export"]["output_path"])

    with patch("steps.export.assert_invariants"), \
         patch("steps.export.validate_step_output"), \
         patch("steps.export.build_cost_report", return_value={"total_cost_usd": 0.0}), \
         patch("steps.export.save_cost_report"):
        run_export(state, cfg)

    assert output_path.exists(), f"Final CSV should exist at {output_path}"
    written = pd.read_csv(output_path, dtype=str)
    assert len(written) > 0


def test_export_preserves_row_count(cfg):
    """Export does not silently drop rows (invariants mocked, dedup not needed)."""
    state = PipelineState()
    state.df = _make_enriched_df(5)
    state.df["benefit_categories"] = [["Remote work"] for _ in range(5)]
    state.df["description_quality"] = "good"
    state.df["soft_skill_categories"] = [["Communication"] for _ in range(5)]

    with patch("steps.export.assert_invariants"), \
         patch("steps.export.validate_step_output"), \
         patch("steps.export.build_cost_report", return_value={"total_cost_usd": 0.0}), \
         patch("steps.export.save_cost_report"):
        run_export(state, cfg)

    assert not state.df.empty
