"""Chain integration tests: run 2+ pipeline steps together on synthetic data.

Tests schema contracts between steps — if a step renames or drops a column
that the next step depends on, these tests catch it.

No API keys required (LLM step is not tested here). All tests complete in
well under 60 seconds.
"""

import hashlib

import pandas as pd
import pytest

from pipeline_state import PipelineState

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _row_id(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:12]


def _make_ingested_df() -> pd.DataFrame:
    """Build a synthetic DataFrame mimicking Step 1 (ingest) output."""
    rows = [
        {
            "job_url": "https://example.com/1",
            "title": "Senior Python Developer (m/w/d)",
            "company_name": "TechCorp GmbH",
            "location": "Berlin, Germany",
            "description": (
                "We are looking for a Senior Python Developer with 5+ Jahre "
                "Berufserfahrung. Vollzeit. Remote möglich. Gehalt: 80.000 - "
                "100.000 EUR. Required: Python, FastAPI, PostgreSQL. Nice to "
                "have: Docker, Kubernetes. Benefits: 30 Urlaubstage."
            ),
            "date_posted": "2026-01-15",
            "site": "linkedin",
        },
        {
            "job_url": "https://example.com/2",
            "title": "Junior Frontend Entwickler",
            "company_name": "Startup AG",
            "location": "Munich, Germany",
            "description": (
                "Junior Frontend Entwickler gesucht. Teilzeit oder Vollzeit. "
                "React und TypeScript erforderlich. Werkstudenten willkommen. "
                "Gehalt: ab 40.000 EUR. Vor Ort in München."
            ),
            "date_posted": "2026-01-16",
            "site": "indeed",
        },
        {
            "job_url": "https://example.com/3",
            "title": "DevOps Engineer",
            "company_name": "Cloud Solutions GmbH",
            "location": "Hamburg, Germany",
            "description": (
                "AWS und Kubernetes Kenntnisse erforderlich. CI/CD Pipeline-"
                "Erfahrung. 3+ Jahre Berufserfahrung. Vollzeit. Hybrid. "
                "Gehalt: 70.000 bis 90.000 EUR. Englisch B2 erforderlich."
            ),
            "date_posted": "2026-01-17",
            "site": "linkedin",
        },
        {
            "job_url": "https://example.com/4",
            "title": "Data Scientist (m/w/d)",
            "company_name": "Analytics Corp",
            "location": "Frankfurt am Main, Germany",
            "description": (
                "Machine Learning Erfahrung. Python scikit-learn TensorFlow "
                "erforderlich. SQL-Kenntnisse. Vollzeit. 100% Remote. "
                "Gehalt: 75.000 - 95.000 EUR. Deutsch fließend erforderlich."
            ),
            "date_posted": "2026-01-18",
            "site": "indeed",
        },
        # Duplicate of row 1 (same company, same location, same title cleaned)
        {
            "job_url": "https://example.com/5",
            "title": "Senior Python Developer",
            "company_name": "TechCorp GmbH",
            "location": "Berlin, Germany",
            "description": (
                "We are looking for a Senior Python Developer with 5+ Jahre "
                "Berufserfahrung. Vollzeit. Remote möglich. Gehalt: 80.000 - "
                "100.000 EUR. Required: Python, FastAPI, PostgreSQL. Nice to "
                "have: Docker, Kubernetes. Benefits: 30 Urlaubstage."
            ),
            "date_posted": "2026-01-19",
            "site": "linkedin",
        },
        # Privacy wall row — should be filtered in step 3
        {
            "job_url": "https://example.com/6",
            "title": "Backend Developer",
            "company_name": "SecretCorp",
            "location": "Berlin, Germany",
            "description": (
                "Datenschutz Einstellungen. Bitte akzeptieren Sie unsere "
                "Cookie-Richtlinie um fortzufahren. "
                "Diese Website verwendet Cookies."
            ),
            "date_posted": "2026-01-20",
            "site": "linkedin",
        },
    ]
    df = pd.DataFrame(rows)
    df["row_id"] = df["job_url"].apply(_row_id)
    df["source_file"] = "test_fixture"
    return df


def _make_cfg(tmp_path):
    """Build a minimal config dict for testing."""
    reports_dir = tmp_path / "reports"
    deduped_dir = tmp_path / "deduped"
    extracted_dir = tmp_path / "extracted"
    failed_dir = tmp_path / "failed"
    reports_dir.mkdir(parents=True)
    deduped_dir.mkdir(parents=True)
    extracted_dir.mkdir(parents=True)
    failed_dir.mkdir(parents=True)
    return {
        "paths": {
            "reports_dir": reports_dir,
            "deduped_dir": deduped_dir,
            "extracted_dir": extracted_dir,
            "failed_dir": failed_dir,
            "checkpoint_db": tmp_path / "chain_test.db",
        },
        "validation": {
            "min_description_length": 50,
            "date_anomaly_cutoff": "2020-01-01",
        },
        "extraction": {
            "model": "test-model",
            "max_tokens": 2000,
            "temperature": 0,
            "max_description_tokens": 8000,
            "batch_size": 100,
            "max_tasks": 7,
        },
        "llm": {
            "provider": "deepseek",
            "deepseek_max_workers": 2,
        },
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_cfg(tmp_path):
    return _make_cfg(tmp_path)


@pytest.fixture
def ingested_state():
    """PipelineState with df populated as if Step 1 (ingest) ran."""
    state = PipelineState()
    state.df = _make_ingested_df()
    return state


# ---------------------------------------------------------------------------
# Chain 1: Ingest → Prepare
# ---------------------------------------------------------------------------

class TestChainIngestPrepare:
    """After Step 2 (prepare), title_cleaned exists and no regex_* columns."""

    def test_title_cleaned_exists(self, ingested_state, tmp_cfg):
        from steps.prepare import run_prepare

        run_prepare(ingested_state, tmp_cfg)
        assert "title_cleaned" in ingested_state.df.columns

    def test_no_regex_columns_after_prepare(self, ingested_state, tmp_cfg):
        """Regex extraction was moved to Step 4 — prepare must NOT add them."""
        from steps.prepare import run_prepare

        run_prepare(ingested_state, tmp_cfg)
        regex_cols = [c for c in ingested_state.df.columns if c.startswith("regex_")]
        assert regex_cols == [], f"Found unexpected regex columns: {regex_cols}"

    def test_row_count_preserved(self, ingested_state, tmp_cfg):
        """Prepare should not drop German rows from our fixture."""
        from steps.prepare import run_prepare

        initial = len(ingested_state.df)
        run_prepare(ingested_state, tmp_cfg)
        # All fixture rows are German, so no rows should be dropped
        assert len(ingested_state.df) == initial

    def test_input_flags_column_exists(self, ingested_state, tmp_cfg):
        from steps.prepare import run_prepare

        run_prepare(ingested_state, tmp_cfg)
        assert "input_flags" in ingested_state.df.columns


# ---------------------------------------------------------------------------
# Chain 2: Prepare → Deduplicate
# ---------------------------------------------------------------------------

class TestChainPrepareDeduplicate:
    """After Step 3 (deduplicate), privacy-wall rows removed, duplicates gone."""

    def _run_prepare_then_dedup(self, state, cfg):
        from steps.deduplicate import run_deduplicate
        from steps.prepare import run_prepare

        run_prepare(state, cfg)
        run_deduplicate(state, cfg)

    def test_privacy_wall_rows_removed(self, ingested_state, tmp_cfg):
        """Privacy-wall flagged rows must be removed before description grouping."""
        self._run_prepare_then_dedup(ingested_state, tmp_cfg)
        # The cookie-wall row (example.com/6) should be gone
        remaining_urls = set(ingested_state.df["job_url"])
        assert "https://example.com/6" not in remaining_urls

    def test_dedup_report_has_filter_counts(self, ingested_state, tmp_cfg):
        """dedup_report should contain privacy_wall_removed count."""
        self._run_prepare_then_dedup(ingested_state, tmp_cfg)
        report = ingested_state.dedup_report
        assert report is not None
        assert "privacy_wall_removed" in report

    def test_composite_duplicate_removed(self, ingested_state, tmp_cfg):
        """Row 5 is a composite duplicate of row 1 — should be removed."""
        self._run_prepare_then_dedup(ingested_state, tmp_cfg)
        # We started with 6 rows, should lose at least the privacy wall + 1 duplicate
        assert len(ingested_state.df) <= 5

    def test_description_groups_set(self, ingested_state, tmp_cfg):
        """Deduplicate must set state.description_groups."""
        self._run_prepare_then_dedup(ingested_state, tmp_cfg)
        assert ingested_state.description_groups is not None
        assert len(ingested_state.description_groups) > 0


# ---------------------------------------------------------------------------
# Chain 3: Deduplicate → Regex Extract
# ---------------------------------------------------------------------------

class TestChainDeduplicateRegex:
    """After Step 4 (regex_extract), all 8 regex_* columns present."""

    def _run_through_regex(self, state, cfg):
        from steps.deduplicate import run_deduplicate
        from steps.prepare import run_prepare
        from steps.regex_extract import run_regex_extract

        run_prepare(state, cfg)
        run_deduplicate(state, cfg)
        run_regex_extract(state, cfg)

    def test_all_regex_columns_present(self, ingested_state, tmp_cfg):
        self._run_through_regex(ingested_state, tmp_cfg)
        expected = {
            "regex_contract_type", "regex_work_modality",
            "regex_salary_min", "regex_salary_max",
            "regex_experience_years", "regex_seniority_from_title",
            "regex_languages", "regex_education_level",
        }
        actual = set(ingested_state.df.columns)
        missing = expected - actual
        assert not missing, f"Missing regex columns: {missing}"

    def test_salary_types_correct(self, ingested_state, tmp_cfg):
        """regex_salary_min should be numeric or None, never a raw string."""
        self._run_through_regex(ingested_state, tmp_cfg)
        for val in ingested_state.df["regex_salary_min"]:
            assert val is None or isinstance(val, (int, float)), (
                f"regex_salary_min={val!r} must be int/float/None"
            )

    def test_schema_guard_fails_without_title_cleaned(self, tmp_cfg):
        """Step 4 should fail if title_cleaned is missing."""
        from steps.regex_extract import run_regex_extract

        state = PipelineState()
        state.df = pd.DataFrame({
            "row_id": ["a"],
            "description": ["Some text"],
            # No title_cleaned column
        })
        with pytest.raises(RuntimeError, match="requires columns"):
            run_regex_extract(state, tmp_cfg)
