# Shared Utilities

> **Location:** `src/shared/`
> **Purpose:** Cross-component utilities used by ingestion, extraction, cleaning, and analysis

---

## Purpose

The `src/shared/` folder provides **single-source-of-truth utilities** that all pipeline components depend on. This prevents scattered magic numbers, inconsistent CSV handling, and divergent validation logic. Every component imports from `shared/` instead of defining its own I/O, constants, or schema validation.

---

## Folder Structure

```
src/shared/
├── __init__.py
├── io.py                  # CSV read/write with NA boundary handling
├── constants.py           # Centralized numeric thresholds + sentinel values
├── config.py              # Multi-component config loader + validator
├── schemas.py             # Pandera DataFrame schemas for step boundaries
├── json_utils.py          # JSON list parsing (single source of truth)
├── fingerprint.py         # SHA-256 hashing for reproducibility
├── logging.py             # Three-output pipeline logger
└── location_filter.py     # Coarse non-German location detection (unused)
```

---

## Module Details

### `io.py` — CSV I/O Boundary Handler

**The most critical shared module.** Enforces the project-wide NA convention at the CSV boundary.

| Function | Purpose |
|----------|---------|
| `read_csv_safe(path, fallback_encoding='latin-1', **kwargs)` | Read CSV with `keep_default_na=False`; converts `"NA"` strings → Python `None` |
| `write_csv_safe(df, path, **kwargs)` | Converts all `None`/`NaN` → `"NA"` strings; writes UTF-8 CSV |

**NA Convention:**
```
              read_csv_safe()                   write_csv_safe()
CSV on disk ──────────────────> DataFrame ──────────────────────> CSV on disk
  "NA" string                   Python None                       "NA" string
                                (use .isna())
```

**Why this matters:** pandas normally auto-converts `"NA"` to `NaN` on read, which would corrupt legitimate `"NA"` strings in job postings (e.g., a company named "NA" or a location containing "NA"). `read_csv_safe()` prevents this.

**Used by:** 9 files across all 4 pipeline components.

---

### `constants.py` — Centralized Thresholds

| Constant | Value | Purpose |
|----------|-------|---------|
| `SALARY_MIN_FLOOR` | `10,000` | Minimum annual EUR salary (legitimates part-time roles) |
| `SALARY_MAX_CEILING` | `300,000` | Maximum annual EUR salary |
| `SALARY_MONTHLY_THRESHOLD` | `5,000` | Cutoff for monthly vs annual salary detection |
| `MISSING_SENTINELS` | `frozenset({"", "nan", "NaN", "None", "none", "NA", "na", "N/A", "n/a", "null"})` | All null-like strings for membership tests |

**Used by:** 12 files — salary validators, regex extractors, date parsers, cleaning modules.

---

### `config.py` — Multi-Component Config Loader

| Function | Purpose |
|----------|---------|
| `load_pipeline_config()` | Merges `src/extraction/config/settings.yaml` + `src/ingestion/config/settings.yaml` into one unified dict. Handles path resolution internally. |
| `validate_config(cfg)` | Checks 13 required nested keys. Raises `ValueError` with clear message on missing key. Called at pipeline startup before burning LLM credits. |

**Config structure returned:**
```python
{
    "paths": {
        "raw_dir": Path(...),
        "checkpoint_db": Path(...),
        "ingestion_output": Path(...),
        # ... 10 entries total
    },
    "extraction": { "model": "deepseek-chat", "temperature": 0, ... },
    "validation": { "salary_min_floor": 15000, ... },
    "ingestion": { "scrape_date": "2026-02-15", ... },
    "extraction_config": ExtractionConfig(...)  # validated dataclass
}
```

---

### `schemas.py` — Pandera DataFrame Schemas

Defines **contract validation** at every step boundary. When a step produces output that violates the contract, the next step fails immediately with a clear error.

| Schema | Produced By | Key Constraints |
|--------|-------------|-----------------|
| `ingestion_output_schema` | Step 1 (Ingest) | `job_url` non-null + starts with "http"; `description` non-null; `site` must be "indeed"/"linkedin" |
| `prepare_output_schema` | Step 2 (Prepare) | Adds `row_id` (unique), `title_cleaned`, `city`, `state`, `country` |
| `dedup_output_schema` | Step 3 (Dedup) | Same contract, fewer rows |
| `extract_output_schema` | Step 5 (Extract) | Same (LLM results in state, not df) |
| `validate_output_schema` | Step 6 (Validate) | Same |
| `clean_enrich_output_schema` | Step 7 (Clean+Enrich) | Adds `job_family`, `technical_skills`, `benefit_categories`, etc. |
| `export_output_schema` | Step 8 (Export) | Final column set |

**Helper:**
- `validate_step_output(df, schema, step_name)` — Validates with `lazy=True` (catches all errors); logs failure cases; raises on violation.

---

### `json_utils.py` — JSON List Parsing

| Function | Purpose |
|----------|---------|
| `parse_json_list(value)` | Parse JSON array string → Python list. Falls back to `ast.literal_eval()`. Returns `[]` on failure. |
| `parse_json_column(df, col)` | Apply `parse_json_list` across entire column → Series of lists |
| `explode_json_column(df, col)` | Parse + explode JSON array column → one row per element |

**Used by:** analysis (skill/benefit exploration), cleaning (soft skill normalization).

---

### `fingerprint.py` — Reproducibility Hashing

| Function | Purpose |
|----------|---------|
| `fingerprint_inputs(cfg)` | Compute SHA-256 hashes of all input CSVs + serialized config. Returns dict with file hashes, config hash, timestamp. Stored in `pipeline_progress.json` for audit trail. |

---

### `logging.py` — Three-Output Logger

| Function | Purpose |
|----------|---------|
| `setup_pipeline_logger(log_dir=None)` | Creates logger with 3 handlers: console (INFO+), rotating file (DEBUG+, 10MB), JSONL (DEBUG+, structured). |

**Structured logging pattern:**
```python
logger.info("Step %s complete", name, extra={"event": "step_complete", "step": name, "elapsed_s": 12.5})
```

**Output files:** `logs/pipeline.log` (human-readable) + `logs/pipeline.jsonl` (machine-readable).

---

### `location_filter.py` — Non-German Detection (Unused)

| Function | Purpose |
|----------|---------|
| `is_likely_non_german(location)` | Quick regex check for clearly non-German locations (USA, Prague, US state codes). Returns boolean. |

**Status:** Defined but currently has no callers. Superseded by the more detailed location parsing in `src/extraction/preprocessing/location_parser.py`.

---

## Cross-Component Usage Map

```
                    ┌─────────────────┐
                    │  shared/io.py    │
                    │  read_csv_safe   │
                    │  write_csv_safe  │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
   src/ingestion/       src/extraction/      src/cleaning/       src/analysis/
   pipeline.py          exporter.py          pipeline.py         utils.py
   date_parser.py       dedup/row_dedup.py   missing_values.py
                        steps/ingest.py
                        steps/export.py

                    ┌─────────────────────┐
                    │ shared/constants.py  │
                    │  SALARY_MIN_FLOOR    │
                    │  MISSING_SENTINELS   │
                    └────────┬────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
   src/ingestion/       src/extraction/              src/cleaning/
   date_parser.py       validators/salary.py         constants.py
                        preprocessing/               missing_values.py
                          regex_extractor.py          validation_fixer.py
                          location_parser.py
                          validate_input.py

                    ┌─────────────────────┐
                    │  shared/schemas.py   │
                    │  7 Pandera schemas   │
                    └────────┬────────────┘
                             │
        All 8 step modules in src/steps/
        + src/ingestion/pipeline.py, src/cleaning/pipeline.py, src/extraction/exporter.py
```

---

## Root-Level Files

### `pyproject.toml`
- **Python:** >=3.11, <3.14
- **Key dependencies:** pandas, openai, pyyaml, tqdm, jsonschema, datasketch, matplotlib, seaborn, jupyter
- **Dev dependencies:** pytest, ruff, mypy, pandera, pandas-stubs
- **Test paths:** `tests/`
- **Python path:** `["src"]` (enables `from extraction.X` imports)

### `Makefile`
| Target | Command |
|--------|---------|
| `make pipeline` | Run all 8 steps |
| `make ingest` | Run only ingest step |
| `make extract` | Run only extract step |
| `make test` | Run all tests across all components |
| `make lint` | `ruff check . --fix` |
| `make type-check` | `mypy src/extraction/ src/shared/` |
| `make dry-run` | Show what would run without executing |
| `make list-steps` | Print all pipeline steps |

### `orchestrate.py`
- **The unified entry point** for the entire pipeline
- Runs 8 sequential steps with resume capability
- CLI: `--only`, `--from`, `--step`, `--dry-run`, `--reset`, `--list`
- Progress tracked in `data/pipeline_progress.json`

See [Extraction Documentation](../extraction/README.md) for full orchestrator details.
