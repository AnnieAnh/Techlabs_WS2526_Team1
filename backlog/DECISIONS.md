# DECISIONS.md — Architecture Decision Log

> **Purpose:** Deep-dive reference explaining *why* every component was designed as it is,
> what edge cases it defends against, and what the developer's intent was.
>  by explaining *why* things work the way they do.

---

## Table of Contents

- [1. System-Level Architecture](#1-system-level-architecture)
- [2. Orchestration Layer (`orchestrate.py`)](#2-orchestration-layer)
- [3. Pipeline State Container](#3-pipeline-state-container)
- [4. Ingestion Component](#4-ingestion-component)
- [5. Shared Utilities](#5-shared-utilities)
- [6. Steps Layer](#6-steps-layer)
- [7. Extraction — Preprocessing](#7-extraction--preprocessing)
- [8. Extraction — Deduplication](#8-extraction--deduplication)
- [9. Extraction — LLM Layer](#9-extraction--llm-layer)
- [10. Extraction — Validators](#10-extraction--validators)
- [11. Extraction — Checkpoint](#11-extraction--checkpoint)
- [12. Extraction — Exporter & Post-Processing](#12-extraction--exporter--post-processing)
- [13. Extraction — Reporting](#13-extraction--reporting)
- [14. Cleaning Component](#14-cleaning-component)
- [15. Analysis Component](#15-analysis-component)
- [16. Configuration Files](#16-configuration-files)
- [17. Cross-Cutting Patterns](#17-cross-cutting-patterns)
- [18. Known Trade-offs and Technical Debt](#18-known-trade-offs-and-technical-debt)

---

## 1. System-Level Architecture

### Decision: Two-Tier Extraction Strategy

The pipeline splits field extraction into two tiers:

- **Tier 1 (Regex):** Salary, contract type, work modality, experience years, seniority from title, language requirements, education level. These fields are expressed in standardized German HR patterns that deterministic regexes handle reliably. Extracted in `src/extraction/preprocessing/regex_extractor.py` *before* any LLM call.
- **Tier 2 (LLM):** Technical skills, nice-to-have skills, soft skills, benefits, tasks, job summary, job family. These require semantic understanding that regex cannot provide.

**Why this split?** LLM API calls cost money and introduce non-determinism. By extracting ~7 fields deterministically first, the LLM only handles what it genuinely adds value for. The `regex_*` prefix on Tier 1 columns throughout the pipeline signals their origin clearly and is only stripped at the final export boundary.

---

### Decision: Config-Driven Taxonomy Injection

All taxonomies (job families, skill aliases, benefit categories, soft skill categories, city aliases) live in YAML files loaded at runtime. The LLM prompt receives the job family taxonomy via a `<<JOB_FAMILIES_LIST>>` placeholder substituted at prompt-build time.

**Why?** A single YAML file is the source of truth. Changing `job_families.yaml` automatically updates both the prompt constraint and the validation/cleaning logic — no code change required. This prevents prompt and code from silently diverging.

---

### Decision: Evidence-Based LLM Extraction (Prompt v2.0)

Every extracted skill, benefit, and task must include a `"source"` field — a verbatim quote from the job description proving the item was mentioned. The system prompt states: *"You MUST include the exact German/English phrase that led you to extract each item."*

**Why?** This is the primary anti-hallucination mechanism. A model that quotes its source text is demonstrably grounded. A fabricated skill will have a fabricated source quote, which `hallucination.py` will fail to locate in the actual description. Evidence adds ~10% to token usage but significantly reduces hallucination rates, paying for itself in reduced post-extraction correction work.

---

### Decision: Progressive Data Quality — Extract-Then-Clean

The pipeline deliberately does NOT reject imperfect LLM output at extraction time. Out-of-taxonomy job families are accepted, low-confidence extractions are accepted, and cross-field inconsistencies are flagged but not rejected. `src/cleaning/` is a separate stage that normalizes and repairs.

**Why?** Strict extraction-time rejection discards expensive API calls and loses recoverable data. A `job_family` of `"Senior Software Engineer"` can be remapped to `"Software Developer"` by `job_family_remap.yaml`. Rejecting at parse time throws away real signal. The `_CRITICAL_FIELDS = frozenset()` in `response_parser.py` is the visible effect of this decision.

---

## 2. Orchestration Layer

### File: `orchestrate.py`

#### Decision: Step Registry as an Ordered List of Tuples

```python
_STEPS = [
    ("ingest",     "Load and normalize raw CSVs",      run_ingest),
    ("prepare",    "Pre-process rows for extraction",  run_prepare),
    ...
]
```

A `list[tuple[str, str, callable]]` — name, description, callable — is used instead of a dict. This preserves step order without relying on Python 3.7+ dict-ordering guarantees, enables enumeration by index, and makes the CLI `--list` display trivial.

#### Decision: JSON Progress File for Human-Readable Resume State

`data/pipeline_progress.json` is updated **after every step** with `status`, `elapsed_s`, and a UTC timestamp. JSON was chosen over SQLite for this purpose because it is human-readable and hand-editable — an operator can manually mark a step as `"complete"` to skip re-running it without touching code.

#### Decision: `_rehydrate_state()` Handles Four Resume Scenarios

Because DataFrames only live in memory, resuming mid-pipeline requires reading the most recent disk artifact. The function handles four cases:

| Case | Condition | Resolution |
|------|-----------|------------|
| A | `ingest` is in the run plan | No-op — ingest populates `state.df` |
| B | Both `deduplicate` and `clean_enrich` were already run | Load `enriched_cleaned.csv` directly |
| C | `deduplicate` done, `clean_enrich` will run | Load deduped CSV + description groups; re-run regex extraction inline if needed |
| D | Only `ingest` was skipped | Load `combined_jobs.csv`, re-assign row IDs, re-register in checkpoint |

Case C includes an inline regex re-run: if the deduped CSV lacks `regex_*` columns, `extract_regex_fields` is executed row-by-row during rehydration. This makes `--from validate` seamless without requiring the user to also run `--from regex_extract`.

#### Decision: `--clean` Deletes Specific Paths, Never `data/raw/`

`_cmd_clean()` enumerates specific subdirectories from the config. `data/raw/` (read-only scraped data) is never touched. Deleted paths are logged one-by-one for auditability.

#### Decision: Config Fingerprinting as Soft Cache Invalidation

After loading config, `fingerprint_inputs()` computes a hash of source CSV files and settings. If the hash changed since the last run, a warning is emitted suggesting `--reset`. This does not force a reset — the operator decides. Intentional design: a config change might be deliberate (you want to continue with new settings on already-processed steps).

#### Decision: `time.monotonic()` for Elapsed Timing

All step timing uses the monotonic clock, not `time.time()`. The monotonic clock cannot jump backwards or be affected by system clock adjustments, giving accurate wall-clock durations for pipeline steps that can run for hours.

---

## 3. Pipeline State Container

### File: `src/pipeline_state.py`

#### Decision: `@dataclass` as a Typed Mutable Shared State

A single `PipelineState` object is passed through every step. All inter-step communication flows through this object rather than through return values. Steps have a uniform signature `(state: PipelineState, cfg: dict) -> None`.

**Trade-off vs pure functions:** Pure functions would be cleaner in isolation but would require complex argument threading across 8 steps. The shared state container is a pragmatic choice that keeps step signatures uniform and the orchestrator simple.

#### Decision: `field(default_factory=pd.DataFrame)` for the DataFrame Field

Mutable default values in dataclasses require `default_factory`. Using a bare `= pd.DataFrame()` would share one DataFrame object across all instances — the classic mutable default pitfall.

#### Decision: `require_df()` Guard

```python
def require_df(self) -> pd.DataFrame:
    if self.df.empty:
        raise RuntimeError(
            "state.df is empty — did you run the ingest step?\n"
            "Run: python orchestrate.py --from ingest"
        )
    return self.df
```

Every step that consumes `state.df` calls this first. Failure is explicit and actionable rather than producing a confusing "operation on empty DataFrame" error five steps downstream.

#### Decision: `row_limit` as a Test-Mode Hook

`state.row_limit` is set by `--limit N` in the CLI. Only the `ingest` step applies it via `df.head(state.row_limit)`. All downstream steps are completely unaware of it. This cleanly separates the test harness from production logic.

#### Decision: `dict[str, Any] | None = None` for Optional Fields

The `None` sentinel explicitly distinguishes "not yet computed" from "empty result". Code that checks `if state.description_groups:` correctly distinguishes a run that completed dedup (non-None, possibly empty dict) from one that hasn't reached dedup yet (None).

---

## 4. Ingestion Component

### Files: `src/ingestion/pipeline.py`, `loader.py`, `date_parser.py`, `config/settings.yaml`

#### Decision: Three Source Files With Different Column Schemas, One Unified Loader

LinkedIn and Indeed use different column names for the same data fields. `SourceConfig.column_mapping` maps *target* column names to *source* column names — the ergonomic direction: "what source column gives me `job_url`?". The `site` column is not in any source file; it is injected from `SourceConfig.site_label` after normalization.

Examples of schema divergence across sources:
- `date_posted`: `"date_posted"` (Indeed) vs `"posted_date"` (LinkedIn #1) vs `"time"` (LinkedIn #2)
- `job_url`: `"job_url"` vs `"link"`
- `location`: `"location"` vs `"location_job"`

#### Decision: `keep_default_na=False, na_values=[]` on All Reads

Pandas' default NA parsing would silently convert country codes, abbreviations, and empty strings to `NaN` at load time. Disabling this ensures raw string values are preserved until the project-standard `read_csv_safe()` boundary performs the controlled conversion.

#### Decision: `scrape_date` Pinned in Config, Not `datetime.now()`

Every relative date ("2 days ago", "yesterday") resolves against `scrape_date: "2026-02-15"` from `settings.yaml`. This makes date parsing **deterministic across re-runs** regardless of when the pipeline is executed. Without pinning, the same relative date resolves differently on different dates, destroying reproducibility.

#### Decision: Far-Future Sentinel Date `"2027-01-01"` for Missing `date_posted`

After normalization, remaining `None` values in `date_posted` are filled with 2027-01-01. This choice makes synthetic rows easily identifiable — any posting dated after the scrape date is not real data and can be filtered in analysis.

#### Decision: Multi-Format Date Parser — Priority Order Matters

`parse_date_to_exact()` handles five input categories **in priority order**:
1. `None`/sentinel strings → `None`
2. ISO date or ISO datetime → extract date part
3. Unix timestamp (seconds or milliseconds) — heuristic: `ts > 1e12` means milliseconds
4. Relative English strings ("today", "yesterday", "X hours ago") → resolve against reference date
5. "X days/weeks/months ago" offsets → compute from reference date

The millisecond-vs-second heuristic (`ts > 1e12`) is pragmatic: Unix seconds are currently ~1.7 billion; milliseconds ~1.7 trillion. "X months ago" uses `n * 30` days — a deliberate approximation acceptable for job posting date analysis.

#### Decision: No Geographic Filtering in Ingestion

The ingestion pipeline is a faithful merge of all source data. Geographic filtering happens in the extraction pipeline's `location_parser` stage where structured city/state/country fields are available. Ingestion errs toward maximum recall — it is always better to have extra rows that get filtered later than to miss real data at the first stage.

#### Decision: Schema Validation Before Write

`ingestion_output_schema.validate(final_df)` runs before `write_csv_safe()`. If validation fails, no file is written. This prevents a broken CSV from being silently consumed by downstream steps.

---

## 5. Shared Utilities

### Files: `src/shared/io.py`, `config.py`, `schemas.py`, `constants.py`, `fingerprint.py`, `json_utils.py`, `location_filter.py`, `logging.py`

#### Decision: `read_csv_safe()` / `write_csv_safe()` as the NA Boundary

All CSV I/O flows through these two functions. They enforce a strict contract:
- **On read:** Disable pandas NA inference (`keep_default_na=False`), then apply project-specific sentinel set → `None`
- **On write:** `df.fillna("NA")` converts all `None`/`NaN` → `"NA"` string

**Why per-column apply instead of `df.replace()`:** `df.replace("NA", None)` produces `float NaN` in numeric columns because pandas coerces `None` to `NaN` for numeric dtypes. The per-column `apply` guarantees object columns store Python `None`, not `float('nan')`. This distinction matters because `float('nan')` is truthy while `None` is falsy — patterns like `(val or "")` behave differently.

**Why encoding fallback:** `read_csv_safe` tries UTF-8 first, then `latin-1` on `UnicodeDecodeError`. Real-world scraped data may contain legacy-encoded bytes. The `fallback_encoding=None` option lets callers disable the fallback and receive the exception if they need strict enforcement.

#### Decision: `_REPO_ROOT` Anchored to `__file__`

```python
_REPO_ROOT = Path(__file__).parent.parent.parent  # src/shared/ -> src/ -> repo-root
```

Path anchoring happens at module import time using the physical location of `config.py`. Every caller is CWD-agnostic — `python orchestrate.py` works from any directory.

#### Decision: `ExtractionConfig` Validates API Parameters at Construction

LLM API parameters (`batch_size`, `max_retries`, `temperature`) are validated in `__post_init__` with explicit range checks. Misconfiguration is caught before any API call is made, not discovered after thousands of requests have been sent.

#### Decision: `_create_directories()` as a Side Effect of Loading Config

Directories are created when config is loaded — not at startup, not on first use. The config is the authoritative source of all data paths, so loading it is sufficient to ensure all paths exist. The `_file_paths` set prevents calling `mkdir` on paths that are files (e.g., `checkpoint_db`).

#### Decision: `MISSING_SENTINELS` as `frozenset`

`frozenset` is immutable (cannot be accidentally mutated), hashable, and `O(1)` for membership tests. The set covers 10 variants (`""`, `"nan"`, `"NaN"`, `"None"`, `"none"`, `"NA"`, `"na"`, `"N/A"`, `"n/a"`, `"null"`) — every real-world representation of missing values seen in scraped data.

#### Decision: Named Salary Constants Instead of Magic Numbers

```python
SALARY_MIN_FLOOR       = 10_000   # legitimate part-time IT role minimum
SALARY_MAX_CEILING     = 300_000  # upper sanity bound for German IT
SALARY_MONTHLY_THRESHOLD = 5_000  # values below this are likely monthly, not annual
```

Defined once in `shared/constants.py`, imported by every module that validates salary. Changing the floor/ceiling requires editing exactly one file.

#### Decision: Two-Stage JSON Parser — `json.loads` then `ast.literal_eval`

`parse_json_list` first tries standard JSON parsing. If that fails, it tries `ast.literal_eval`. The fallback handles single-quoted Python list strings (`"['Python', 'SQL']"`) — valid Python but invalid JSON, a known LLM output pattern. Any non-list result returns `[]`, guaranteeing the caller always receives an iterable.

**Why `except Exception` in the ast branch:** `ast.literal_eval` can raise `ValueError`, `SyntaxError`, or other exceptions on malformed input. The function's contract is "always return a list," so any failure produces `[]` rather than propagating.

#### Decision: Three-Sink Logging Architecture

| Sink | Level | Format | Purpose |
|------|-------|--------|---------|
| stdout | INFO+ | Short timestamp | Human monitoring |
| `pipeline.log` | DEBUG+ | Full timestamp, wide module column | grep-friendly diagnostics |
| `pipeline.jsonl` | DEBUG+ | JSON object per line | Structured post-hoc queries |

The JSONL sink enables: `jq 'select(.event == "step_complete") | .elapsed_s'` to extract timing for every step. `RotatingFileHandler` (10 MB × 5 backups) prevents unbounded log growth.

**Why idempotent setup:** The guard `if root_logger.handlers: return root_logger` ensures that calling `setup_pipeline_logger()` twice (common in test environments that import multiple modules) does not add duplicate handlers and duplicate log lines.

#### Decision: `location_filter.py` is Intentionally Coarse

`is_likely_non_german()` errs toward **inclusion** (low exclusion threshold). Its sole purpose is to reduce load on the extraction pipeline. False positives (keeping non-German rows) are acceptable — they are caught by the fine-grained `location_parser` later. False negatives (excluding German rows) are not acceptable — they permanently lose data.

The trailing state-code pattern `,\s*[A-Z]{2}\s*$` catches US-style "City, CA" formats. It also matches "Berlin, DE" — intentional, because strict parsing happens downstream.

#### Decision: Streaming File Hash (64 KiB Chunks, SHA-256 Truncated to 16 Chars)

```python
iter(lambda: f.read(65536), b"")  # canonical streaming hash pattern
```

Files are hashed in chunks rather than loaded entirely — memory-safe for large CSVs. SHA-256 truncated to 16 hex characters (64 bits) is sufficient for change detection. Full 64-char hashes would be unreadable in log output.

---

## 6. Steps Layer

### Files: `src/steps/*.py`

#### Decision: Uniform Step Signature `(state, cfg) -> None`

Every step takes the same two arguments and returns nothing. This uniformity enables the step registry pattern in `orchestrate.py` — all 8 steps are interchangeable in the dispatch loop.

#### Decision: Two-Pass NA Standardization

`standardize_missing_values()` is called **twice** across the pipeline:
1. **End of `prepare` (step 2):** Covers pre-LLM columns (title, location, date, input flags)
2. **Start of `clean_enrich` (step 7):** Covers LLM-extracted columns that don't exist until after `merge_results()`

The split is a direct consequence of two-stage data population — you cannot standardize columns that do not yet exist.

#### Decision: `ingest.py` — Deterministic 12-Char Row ID from URL Hash

```python
hashlib.sha256(job_url.encode()).hexdigest()[:12]
```

12 hex characters = 48 bits of entropy. Collision probability over 22,000 rows: ~2.6 × 10⁻⁸. The function is **module-level** (not inside `run_ingest`) so `orchestrate._rehydrate_state` can import it for Case D rehydration without triggering a full ingest.

**Why post-write read-back:** After `run_pipeline()` writes the CSV, the step reads it back with `read_csv_safe()`. This intentional round-trip ensures the in-memory DataFrame matches exactly what is on disk, including NA conversion. Downstream steps see the same representation whether they read from memory or disk.

#### Decision: `prepare.py` — Step Ordering Rationale

The module docstring explicitly documents why operations are ordered as they are:
- Title normalization runs BEFORE dedup (step 3) so the composite dedup key uses normalized titles
- Location parsing runs HERE (step 2) so non-German rows are excluded before expensive work begins
- Regex extraction deliberately runs AFTER dedup (step 4) — fewer rows = less work

"Cheap operations run together, separately from extraction (minutes vs API cost)" — the step boundary exists precisely because the operations on one side are free and the other side costs money.

#### Decision: `deduplicate.py` — 4-Pass Deduplication Pipeline

| Pass | Method | Purpose |
|------|--------|---------|
| 0 | Privacy-wall filter | Remove cookie consent / GDPR wall pages before MinHash |
| 1 | Exact URL dedup | Same posting scraped from multiple sources |
| 2 | Composite key dedup | Same job, slightly different URL (title + company + location) |
| 3 | SHA-256 description dedup + MinHash LSH at 95% Jaccard | Group identical/near-identical descriptions for LLM result fan-out |

**Why location is in the composite key:** The same job posted in Frankfurt and Munich is a legitimate distinct row for location analysis. Removing location from the key would merge them.

**Why 95% Jaccard threshold:** Very conservative — only near-identical descriptions are grouped. Minimizes false positives (different jobs merged) at the cost of missing some genuine near-duplicates.

**Why Pass 0 (privacy-wall filter) comes first:** Cookie consent screens and error pages would form their own near-duplicate groups in MinHash, then propagate garbage text as representative descriptions. Removing them first prevents this contamination.

#### Decision: `regex_extract.py` — `to_dict("records")` Over `df.apply`

Row-by-row extraction uses `df.to_dict("records")` + list comprehension rather than `df.apply(axis=1)`. Each `extract_regex_fields` call receives a plain Python dict rather than a pandas `Series`, avoiding `Series.__getitem__` overhead and making the function independently testable.

#### Decision: `extract.py` — LLM Fan-Out for Description Groups

The LLM is called only on *representative* rows (one per description group). Results are propagated to all group members via `_propagate_results()`. The guard `if member_id in by_id: continue` makes propagation **idempotent** — re-running extract after a failure does not duplicate entries. Each propagated result is a shallow copy with `"row_id"` overwritten.

**Why `groups = state.description_groups or {}`:** The empty-dict default means `run_extract` works correctly even when called without a preceding dedup step (e.g., `--only extract`). In that case all rows are sent to the LLM.

#### Decision: `validate.py` — Separate from Extract

Validation rules are isolated from extraction so they can be iterated **without re-running the expensive LLM step**. Changing a validation rule, running `--only validate`, seeing updated results: this cycle takes seconds, not hours.

**Disk fallback:** If `state.extraction_results` is `None` (step 5 was skipped), the step reads from `extraction_results.json`. This enables `--from validate` after a completed `extract` step.

#### Decision: `clean_enrich.py` — Debug CSV Before Assertions

```python
df.to_csv(debug_path)           # always written first
assert_invariants(df, ...)      # may raise
debug_path.rename(output_path)  # atomic rename, only on success
```

If an invariant assertion fails, the debug CSV is preserved on disk for forensic inspection. On success, `rename()` atomically produces the final output — no partial file at the output path.

#### Decision: `export.py` — Two-Level Quality Checks

1. `assert_invariants(df, valid_families)` — domain business rules (9 invariants: duplicate row IDs, valid JSON arrays, salary sanity, etc.)
2. `validate_step_output(df, export_output_schema, "export")` — structural Pandera schema

The ordering is deliberate: business invariants produce more actionable error messages than schema errors.

---

## 7. Extraction — Preprocessing

### Files: `src/extraction/preprocessing/`

#### Decision: `validate_input.py` — Flags as Lists, Not Boolean Columns

Each row gets a `List[str]` of flag strings rather than one boolean column per flag type. Adding a new flag type requires no schema change — the existing column absorbs it. The flag types (`privacy_wall`, `short_description`, `invalid_date`, etc.) are documented in the quality report, not in the column schema.

**Action vs observation:** The privacy-wall flag has a hard action (row marked as `skipped` in checkpoint, never retried). Other flags are purely observational — downstream steps decide whether to act on them.

**Date anomaly action:** Rows with anomalous dates (before 2024-01-01) have `date_posted` set to `None` rather than being dropped — the job description is still valid for extraction; only the date metadata is suspect.

#### Decision: `text_preprocessor.py` — Non-Destructive Design

The original description stored in the DataFrame is **never modified**. `preprocess_description()` returns a cleaned version for the LLM only. The skill hallucination verifier uses the raw text to confirm skills actually appear in the original description — using the pre-processed version would hide evidence.

**`C#` / `F#` preservation via lookbehind:**
```python
(?<![CF])#   # strip # only when NOT preceded by C or F
```
Without this, `C#` and `F#` would become bare `C` and `F` after markdown stripping, corrupting skill names silently.

**Broad emoji regex:** The emoji regex intentionally covers 8 Unicode blocks to maximize coverage of decorative bullets common in LinkedIn/Indeed job postings. False positives (stripping a genuine `→` arrow) are accepted as a trade-off for completeness.

#### Decision: `title_normalizer.py` — Gender Detection by Content Inspection

`_is_gender_paren()` inspects the *contents* of parentheses rather than matching a fixed string. This handles ~80+ German gender notation variants: `(m/w/d)`, `(m/f/d)`, `(all genders)`, `(mwd)`, `(gn)`, `(m,w,d)`.

**Longest-match-first for translations:**
```python
sorted(translations.items(), key=lambda kv: len(kv[0]), reverse=True)
```
Compound terms are translated before single terms. Without this, `"Softwareentwickler"` would be partially matched by `"Entwickler"` → `"Developer"` first, producing `"Softwareentwickler Developer"` instead of `"Software Developer"`.

**ALL-CAPS detection:** Title casing conversion is only applied when `len(alpha) > 3` and all-uppercase — prevents "correcting" legitimate short acronyms like `"SRE"` or `"CTO"`.

#### Decision: `location_parser.py` — Named Strategy Constants for Fallback Monitoring

Ten strategy names (`standard_3part`, `coded_3part`, `region_match`, `unknown_fallback`, etc.) are string constants. When `parse_all_locations` runs, the chosen strategy per row is recorded and counted. High fallback rates trigger a `WARNING` log and motivate adding new region patterns to the YAML.

**Why comma-splitting dispatch:** German/English location strings follow a predictable comma-delimited structure. The number of comma-separated parts determines the sub-parser: 3 parts → `"City, State, Country"`, 2 parts → `"City, Country"` or `"City, State"`, 1 part → city-state or region lookup. This matches how location strings are actually formatted on LinkedIn/Indeed.

**City-state set for Stadtstaaten:** Berlin, Hamburg, and Bremen are city-states — their city name equals their state name. The parser sets both `city` and `state` to the same value, matching how German administrative divisions work.

**Non-German regions in the German region map:** Zürich, Prague, Philadelphia appear in `german_states.yaml` with explicit `country` fields. This is pragmatic — the region section handles all single-part location strings. Having non-German entries here centralizes all location knowledge in one file.

#### Decision: `regex_extractor.py` — Order-Sensitive Pattern Lists

Pattern lists are ordered by specificity, most-specific first:
- `unbefristet` before `befristet` — the negative prefix `un-` must be matched by the unbefristet pattern before the bare `befristet` pattern fires
- Alternatively, `(?<!un)befristet` negative lookbehind — cleaner for the simple case
- `Hybrid` before `Remote` — hybrid job postings mention home office as a subset; `Remote` would match first without priority

**Salary context validation (`_is_salary_context`):** Scans ±50 characters around each salary number for employee counts and monthly indicators. Without this, `"500 Mitarbeiter"` (500 employees) would frequently be misidentified as an annual salary of €500.

**German number format stripping before salary parse:**
The regex extractor handles German thousands format (e.g., `"50.000"` → `50000`) via `_parse_german_number()` using `.replace(".", "").replace(",", "")`. The cleaning pipeline (`missing_values.py`) separately handles this with a regex `_GERMAN_FMT = re.compile(r"^\d{1,3}(\.\d{3})+$")` for post-extraction numeric fixing.

**Boilerplate filtering for languages:** Language mentions inside diversity/equal-opportunity boilerplate sections (e.g., "We welcome applications regardless of language background") are excluded from language requirement extraction.

**Qualitative experience fallback:** If no numeric experience is found, qualitative words map to estimated years: `"Mehrjährige"` (several years) → 3, `"Langjährige"` (long years) → 5. These mappings were calibrated against real German HR vocabulary.

---

## 8. Extraction — Deduplication

### Files: `src/extraction/dedup/`

#### Decision: `row_dedup.py` — Casing-Score Sort Before Duplicate Drop

Before `pandas.duplicated(keep="first")`, rows are sorted by `_company_casing_score` descending. This ensures the best-cased variant of a company name is kept regardless of which was scraped first.

Scoring heuristic:
- Uppercase-starting words: `+2` (e.g., `SAP`, `Deutsche`)
- Mixed-case words: `+1` (e.g., `GitHub`)
- All-lowercase words: `-1` (e.g., `sap`)

`kind="stable"` on `sort_values` preserves original row order within equal casing scores — deterministic output.

#### Decision: `description_dedup.py` — `datasketch` as Optional Dependency

The MinHash LSH import is wrapped in `try/except ImportError`. If `datasketch` is not installed, `find_near_duplicates()` returns an empty dict silently. The pipeline is fully functional without the near-dedup pass; it just processes a few more redundant descriptions.

This design choice reflects that near-dedup is a cost-optimization feature, not a correctness requirement.

#### Decision: SHA-256 for Exact Description Dedup

SHA-256 is used for exact description matching. SHA-256 is overkill for collision resistance here (22,000 descriptions), but it is standard, has a near-zero false-positive rate, and uses the same hash infrastructure as the row ID generation.

Descriptions are whitespace-normalized before hashing:
```python
re.sub(r"\s+", " ", text.strip()).lower()
```
This ensures descriptions differing only by trailing newlines or extra spaces are treated as identical.

#### Decision: Transitive Chain Resolution in Near-Dedup

```python
changed = True
while changed:
    changed = False
    for dup_hash, rep_hash in merge_ops:
        # ...resolve A->B, B->C chains to A->C, B->C
```

A fixpoint loop resolves transitive near-duplicate chains. Without it, a chain `A ~ B ~ C` would leave `B` pointing to itself rather than to `A` (the ultimate representative), causing `propagate_group_results` to fan out different results to different parts of the chain.

#### Decision: Fan-Out Pattern (`propagate_group_results`)

Rather than only extracting the representative and discarding member rows, extraction results are propagated to all members. Each propagated result is a shallow copy of the representative's dict with `"row_id"` overwritten. This preserves full row count for downstream location/company analysis.

The guard `if member_id in by_id: continue` makes propagation **idempotent** — safe to call multiple times.

---

## 9. Extraction — LLM Layer

### Files: `src/extraction/llm/`

#### Decision: `client.py` — Lazy Singleton Client

```python
_client: AsyncOpenAI | None = None

def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=os.environ["DEEPSEEK_API_KEY"])
    return _client
```

Instantiated only on first use. Dry runs, test imports, and code paths that never call the API pay no initialization cost. `reset_client()` exists specifically for test isolation — tests can clear the singleton between test cases.

#### Decision: Module-Level Token Accumulators

Token counts accumulate across the entire process lifetime rather than per-call. `reset_token_usage()` enables cross-run independence. This ensures cumulative cost is trackable even across interrupted runs where different `run_extraction` calls might occur.

#### Decision: `processor.py` — Two-Layer Async Architecture

```python
def run_extraction(rows, cfg, checkpoint):   # synchronous entry point
    asyncio.run(_run_async(rows, cfg, checkpoint))

async def _run_async(...):
    sem = asyncio.Semaphore(max_workers)     # gates concurrent API calls
    tasks = [_process_one(row, sem) for row in rows]
    await atqdm.gather(*tasks)               # live progress bar
```

The public API is synchronous for pipeline compatibility. The inner work is fully async for concurrent API calls. `asyncio` is single-threaded — shared mutable lists (`successes`, `failures`) are safe without locks.

`KeyboardInterrupt` is caught, the partial results are saved, then the exception is re-raised. Work done before the interrupt is preserved.

#### Decision: Prompt Version Fingerprinting

An 8-char MD5 hash of the system prompt is stored with each extraction result. MD5 is used for brevity and speed, not security. This enables auditing which prompt version produced which result after a prompt change, enabling targeted re-extraction of specific rows.

#### Decision: `processor.py` — Merge-On-Save, Additive Token Counts

When resuming, existing `extraction_results.json` is loaded and keyed by `row_id`. New results overwrite by key — re-extracting the same row (e.g., after a prompt change) replaces the old result. Token usage is **additive** — cumulative cost is preserved across interrupted runs.

#### Decision: `response_parser.py` — 5-Strategy Parse Cascade

```
1. Direct json.loads()                    → fast path for well-formatted output
2. Strip markdown fences (```json...```)  → common LLM formatting habit
3. Fix trailing commas before } or ]      → common LLM JSON error
4. Extract braces (handle preamble text)  → LLM added commentary before/after JSON
5. Failure                                → recorded with strategy name
```

Named strategy constants allow monitoring which strategies fire in production. High use of strategies 3 or 4 indicates a prompt quality issue — the LLM is consistently producing malformed JSON.

**Why `iter_errors` not `validate`:** JSON Schema validation collects ALL errors before raising. An early non-critical error (e.g., extra field) would otherwise mask a more important one (e.g., missing required field).

#### Decision: `prompt_builder.py` — Sentence-Boundary Truncation

Rather than a hard character cut, descriptions are truncated at the nearest sentence boundary (`. `, `.\n`, `! `, `? `, `\n\n`) within the last 200 characters. A `[TRUNCATED]` marker is appended so the LLM knows content was cut — preventing it from inferring a normal description ending and fabricating skills that would have appeared later.

**3.5 chars/token heuristic:** Mixed German/English content is estimated at 3.5 chars/token (between English ~4.0 and German ~3.0). Deliberately over-estimated (conservative) to avoid silently exceeding the context window.

**`_ESTIMATED_OUTPUT_TOKENS = 350`:** Used for pre-flight cost estimation. Documented as a heuristic, not an exact value.

---

## 10. Extraction — Validators

### Files: `src/extraction/validators/`

#### Decision: `runner.py` — Validation Order Is Semantically Required

The step order in `run_validators()` is not arbitrary:

```
Step 1: Skill alias normalisation   ("js" → "JavaScript")
Step 2: Hallucination flagging       (operates on canonical names)
Step 3: Hallucination removal        (removes unverified items from data)
Step 3.5: Save evidence objects      (preserve source quotes in parallel keys)
Step 4: Cross-field checks
Step 5: Salary checks
```

Normalisation must run first so that hallucination checking operates on `"JavaScript"`, not `"js"` or `"JS"`. Word-boundary matching against the description text is more reliable with canonical names. Running checks in the wrong order would produce systematically inaccurate results.

**Why evidence objects are preserved in parallel keys:** The rest of the pipeline expects plain string skill lists (`["Python", "Docker"]`). The evidence format `[{"name": "Python", "source": "..."}]` is needed for auditability. After hallucination removal, evidence objects are saved in `{field}_evidence` keys and the originals are flattened to plain name lists. The "3.5" numbering in comments acknowledges this was added after the original design without renumbering.

#### Decision: `hallucination.py` — Three-Tier Source Verification

When the LLM provides a source quote, verification proceeds through three tiers in order of cost:

| Tier | Method | Handles |
|------|--------|---------|
| 1 | Case-insensitive exact substring | ~95% of cases (direct copy of source text) |
| 2 | Whitespace-normalised substring | Copy-pasted quotes with line breaks or extra spaces |
| 3 | Token overlap ≥ 80% | LLM paraphrased the quoted text slightly |

The 80% token overlap threshold is deliberately lenient — a skill is real even if the LLM slightly mangled its own source quote.

**Name-matching fallback:** When source verification fails entirely, the verifier falls back to word-boundary matching on the skill name itself plus all known aliases and variants from `skill_variants.yaml`. A skill passing name matching is accepted with lower confidence rather than outright rejected. The intent: "trust the LLM when it shows its work, but verify even then."

**`Java` vs `JavaScript` ambiguity:** Word-boundary regex `(?<![a-zA-Z0-9])term(?![a-zA-Z0-9])` prevents `Java` from matching inside `JavaScript`. This is one of the most common skill hallucination false-positive scenarios.

#### Decision: `skills.py` — Evidence Format Throughout

Normalisation and deduplication work on the evidence format `[{"name": ..., "source": ...}]`, not plain string lists. Alias resolution is applied to `item["name"]` while `item["source"]` is preserved unchanged — the source quote remains the original LLM text for downstream verification.

**Soft skills get no alias matching:**
```python
# soft skills: simple dedup by stripped whitespace, no alias resolution
sorted({s.strip() for s in soft if s.strip()})
```
Interpersonal skills like "communication" or "teamwork" don't have technical aliases in the same way as `"js"` → `"JavaScript"`. Applying alias resolution here would create false normalizations.

#### Decision: `cross_field.py` — Rule 3 Uses `severity="info"` Intentionally

The seniority source mismatch rule (LLM seniority vs title-derived seniority disagree) fires frequently for legitimate reasons. A `"Lead"` title might produce `seniority_from_title="Lead"` while the LLM correctly maps the role to `"Senior"`. Setting this to `"info"` prevents alert fatigue from a rule that fires as expected behavior.

**Rules 6 and 7 are asymmetric by design:** "Remote but onsite text" and "Onsite but remote text" check against exact modality values (`"remote"` and `"onsite"`) — `"Hybrid"` is intentionally excluded to prevent false positives on hybrid roles.

#### Decision: `salary.py` — Tier 1 Architecture Documented at Call Site

The docstring explicitly states salary is a Tier 1 (regex-extracted) field and values are read from `regex_salary_min` / `regex_salary_max` keys. This is the only file in the validator suite that documents the Tier 1/Tier 2 distinction at the call site — making it the reference point for understanding the salary extraction architecture.

**`possible_revenue` elevated to `error` severity:** Values above the ceiling (`300,000`) could be company revenue figures that would silently corrupt salary distribution analysis. This severity distinction (error vs warning) allows reports to highlight the most dangerous issues first.

---

## 11. Extraction — Checkpoint

### File: `src/extraction/checkpoint.py`

#### Decision: SQLite Over Flat File — ACID Guarantees

SQLite provides atomic commits — safe against mid-write crashes. A flat JSON file of completed row IDs would be corrupted by a crash during the write operation. The module docstring states: *"All writes use SQLite's atomic commit semantics — safe against mid-write crashes."*

`check_same_thread=False` allows the connection to be used across threads if the pipeline ever parallelises, without needing a connection pool.

#### Decision: `INSERT OR IGNORE` for Rows, Upsert for Files — Deliberate Asymmetry

- **Row registrations** use `INSERT OR IGNORE` — never overwrite an existing row's status. A row marked `completed` stays `completed`. This is the core invariant: completed work is never discarded.
- **File registrations** use `INSERT ... ON CONFLICT DO UPDATE` — refresh metadata on re-run. A file's row count or last-seen timestamp may change between runs.

#### Decision: Commit After Every Single Row

```python
conn.execute("UPDATE rows SET status = ? WHERE row_id = ?", ...)
conn.commit()  # one commit per row
```

One commit per row is slower than batching 100 rows per commit. This is a deliberate trade-off: each row represents a ~$0.002 API call. Losing even a few hundred rows to a crash is unacceptable. The extra I/O overhead (~0.1ms per commit on modern SSDs) is negligible compared to the API latency (~500ms per call).

#### Decision: Context Manager Protocol for Connection Lifecycle

```python
with Checkpoint(path) as cp:
    cp.advance_stage(row_id, "completed")
```

`__enter__`/`__exit__` guarantee the SQLite connection is closed even if the calling code raises an exception. Without this, a process crash could leave the WAL (write-ahead log) unsynced.

#### Decision: Log at Both `logger` and `print()` in `print_progress()`

Progress is visible in both log files and the terminal simultaneously. This is important for long-running extraction (hours) — an operator watching the terminal can monitor progress without tailing log files.

---

## 12. Extraction — Exporter & Post-Processing

### Files: `src/extraction/exporter.py`, `src/extraction/post_extraction.py`

#### Decision: `exporter.py` — Late-Binding Tier 1 Column Promotion

The `regex_*` prefix is stripped at export time only:
```python
df.rename(columns={"regex_contract_type": "contract_type"}, inplace=True)
```

During the pipeline, `regex_*` names clearly signal their origin (important for cross-validation and debugging). Exported files use clean canonical names for downstream consumers (notebooks, cleaning pipeline) who should not need to know the extraction mechanism.

**Handling both cases:** If the canonical column already exists (from LLM extraction), the `regex_*` duplicate is dropped rather than renaming. If only the `regex_*` version exists, it is renamed to canonical. This handles the transition cleanly for both Tier 1-only and mixed fields.

#### Decision: `json.dumps` Not `repr()` for List Columns

```python
json.dumps(val, ensure_ascii=False)  # produces ["Python", "SQL"]
str(val)                             # would produce ['Python', 'SQL'] — invalid JSON
```

Python's `repr()` uses single quotes — not valid JSON. The cleaning pipeline calls `json.loads()` on these fields. Using `repr()` would silently break all downstream list parsing without raising an error at write time.

`ensure_ascii=False` is critical — German characters in skill names (`"Bürokommunikation"`) must be preserved, not escaped to `\u00fc`.

#### Decision: Per-Source-File CSV Exports

Each `source_file` (LinkedIn vs Indeed variants) gets its own CSV via `groupby("source_file")`. This enables source-specific analysis and makes scraper-specific quality issues easy to diagnose in isolation.

#### Decision: Schema Validation Before Final Write — Fail Fast

`extraction_output_schema.validate(df_ordered)` runs on the combined DataFrame before writing. On failure, the exception is re-raised — the pipeline stops rather than writing a schema-invalid file that would corrupt downstream analysis.

#### Decision: `post_extraction.py` — Same Corrections as Cleaning, Different Input Format

The module applies three corrections to the JSON extraction results *before* export:
1. Categorical remapping (job family, contract type, seniority)
2. C++ hallucination correction
3. Skill casing normalisation

The same corrections also exist in `cleaning/pipeline.py` as safety nets for pre-existing CSVs. The module docstring documents this hierarchy explicitly: `post_extraction.py` runs first on JSON format; `cleaning/` is the safety net for CSV format. The duplication is intentional — both execution paths (orchestrated pipeline and standalone cleaning) produce equivalent output.

#### Decision: `fix_cpp_inference` Three-Way Decision Tree

The LLM frequently infers `C++` from bare `C` mentions in job descriptions:

```
Description contains "c++" anywhere?    YES → leave unchanged (genuine C++)
Description contains bare "C"?          YES → replace C++ with C (LLM confusion)
Neither?                                     → remove C++ entirely (hallucination)
```

The regex `(?<![a-zA-Z0-9])C(?![a-zA-Z0-9+#/])` is case-sensitive (uppercase only) and excludes `C++`, `C#`, `C/C++`, and substrings in words like `CISCO`.

The same rule appears in both the LLM extraction prompt ("C and C++ are distinct languages") and the code correction logic. Both channels address the same known model failure mode from different angles — defense in depth.

---

## 13. Extraction — Reporting

### Files: `src/extraction/reporting/`

#### Decision: `cost.py` — Hardcoded Pricing Constants

```python
_COST_PER_INPUT_TOKEN  = 0.27 / 1_000_000  # USD per input token
_COST_PER_OUTPUT_TOKEN = 1.10 / 1_000_000  # USD per output token
```

DeepSeek V3 pricing is hardcoded with inline unit comments. Making pricing configurable would add complexity for little benefit — it changes infrequently and the values serve as documentation of what each run cost.

`read_batch_token_usage()` returns a zero-usage dict (not `None`, not an exception) when the token usage file is absent. This prevents the cost report from crashing if extraction was interrupted before any tokens were logged.

#### Decision: `quality.py` — Dual JSON + Markdown Output

Both `quality_report.json` and `quality_report.md` are always written together. The Markdown uses Unicode block characters (`▓`/`░`) for ASCII coverage bars — optimised for quick visual scanning in a text editor without opening the JSON.

**Graduated quality concerns:**
```python
if coverage < 0.50: "CRITICAL — below 50% coverage"
elif coverage < 0.70: "MODERATE — below 70% coverage"
```
Two thresholds produce graduated alerts. Fields barely above 50% are notable concerns without being critical. Analysts can prioritize based on severity.

**Zero-coverage fields explicitly included:** Fields with 0% coverage appear in the report as `0.0` rather than being omitted. Omitting them would hide the problem.

#### Decision: `evaluation.py` — Three Metric Types for Three Data Types

| Data type | Metric | Rationale |
|-----------|--------|-----------|
| Categorical (job_family, seniority) | Exact match | Partial correctness is meaningless for enums |
| Numeric (salary_min, salary_max) | Exact match + ±10% tolerance | Accommodates rounding differences |
| List (skills, benefits, tasks) | Jaccard + precision + recall | Order-irrelevant; partial overlap has value |

**`both_null` counted as correct:** When both pipeline output and golden annotation are null, it counts as an exact match. `both_null` is explicitly reported — preventing it from silently inflating accuracy on high-null-rate fields.

**`_TIER1_DF_CANDIDATES` — pre/post-merge state handling:**
```python
"contract_type": ["contract_type", "regex_contract_type"]
```
Tier 1 fields have `regex_` prefixes before `merge_results()` runs and canonical names after. The evaluator tries candidates in order, working correctly in both pipeline states.

---

## 14. Cleaning Component

### Files: `src/cleaning/`

#### Decision: `pipeline.py` — `standalone` Flag for Dual Execution Context

```python
def clean(df, standalone=True):
    ...
    if standalone:
        df = fix_cpp_inference(df)        # skip: already ran in validate step
        df = normalize_skill_casing(df)   # skip: already ran in validate step
```

When called from the orchestrator (`standalone=False`), these steps already ran in the `validate` step. Skipping them avoids redundant work. When called directly (`standalone=True`, e.g., standalone cleaning run), they run as safety nets. The flag is the only coupling point between the two execution contexts.

**Debug-first save pattern (same as export step):**
```python
df.to_csv(debug_path)           # always written first
assert_invariants(df, ...)      # may raise AssertionError
debug_path.rename(output_path)  # atomic rename on success only
```

#### Decision: `missing_values.py` — List Columns Normalize to `"[]"`, Not `None`

List columns (`technical_skills`, `benefits`, etc.) normalize to `"[]"` (empty JSON array string) rather than `None`. Downstream `json.loads("[]")` produces `[]` without error. Downstream `json.loads(None)` raises `TypeError`. Every consumer of list columns can safely parse without null-guarding.

**Deduplication inside `_safe_list`:** While normalizing list columns, case-insensitive deduplication removes `["Python", "python"]` — the LLM sometimes produces the same skill in two casings.

**German number format detection:**
```python
_GERMAN_FMT = re.compile(r"^\d{1,3}(\.\d{3})+$")
# "50.000" is German-formatted 50,000 — NOT the float 50.0
```
`float("50.000")` produces `50.0`. Without this pattern match, a salary of €50,000 would be stored as €50.

**`lambda v, c=col: _to_int_or_none(v, c)` — capturing loop variable:** The `c=col` default-argument capture is correct and necessary to avoid Python's late-binding closure bug inside a loop. This is a well-known Python pitfall handled properly.

#### Decision: `validation_fixer.py` — Repair Historical Serialization Bug

An older exporter wrote `str(flags_list)` instead of `json.dumps(flags_list)`, producing Python `repr` format:
```python
"['skill_hallucination', 'possible_monthly']"  # single quotes, invalid JSON
```
Rather than re-extracting all data, this fixer repairs it in the cleaning pass. The fast path (`json.loads` succeeds on already-valid JSON) has negligible overhead. Only malformed data hits the regex conversion path.

#### Decision: `location_cleaner.py` — State-Aware Frankfurt Disambiguation

`"Frankfurt"` is genuinely ambiguous in Germany:
- Frankfurt am Main (Hesse) — the financial center (~800k population)
- Frankfurt (Oder) (Brandenburg) — a small city (~60k population)

```python
if state == "Brandenburg":
    city = "Frankfurt (Oder)"
else:
    city = "Frankfurt am Main"
```

A simple alias map cannot disambiguate without context. The state field from `location_parser` is used as the signal.

**Lazy loading with module-level sentinel:** `_CITY_ALIASES: dict | None = None` — the YAML file is read on the first call to `normalize_city_names()`, not at import time. Avoids file I/O during module import (important for test environments).

#### Decision: `categorical_remapper.py` — Eager Config Load

Unlike `location_cleaner.py`, this module loads `job_family_remap.yaml` at import time. Categorical remaps are small and static — loading eagerly catches YAML parse errors at startup rather than at runtime during a production pipeline run.

**`Counter`-based most-frequent casing:**
```python
counter = Counter(df["company_name"].dropna())
canonical_map = {name.lower(): counter.most_common()[0][0] for name in ...}
```
The canonical form of each company name is determined by frequency. `"SAP SE"` (appearing 500 times) beats `"sap se"` (appearing 3 times). One definition — the most-seen variant wins.

#### Decision: `skill_normalizer.py` — Cross-Module `skill_variants.yaml` Reuse

`_SKILL_VARIANTS_PATH` deliberately points into `extraction/config/` — the cleaning module reuses the extraction-time variant list rather than duplicating it. One source of truth for how skills map to description text, used by both the extraction-time hallucination verifier and the cleaning-time re-verifier.

**Corpus-wide frequency-based casing:** The canonical casing for each skill is determined by frequency across all rows in the dataset. `"Python"` (5,000 occurrences) beats `"python"` (50 occurrences). The YAML file `skill_canonical_case.yaml` exists as an auto-generated artifact of this computation.

**`re_verify_skills_post_clean` — Replace Stale Flags, Not Append:**
After skill normalization changes skill names, original hallucination flags reference pre-normalization names and become stale. The function **replaces** `skill_not_in_description` and `high_hallucination_rate` flags with fresh ones computed against the normalized names. Non-skill flags (salary, cross-field) are preserved unchanged.

#### Decision: `soft_skill_normalizer.py` — Language Skill Filter by Regex

```python
_LANGUAGE_SKILL_RE = re.compile(r"(kenntnisse|sprachkenntnisse)", re.IGNORECASE)
```

The LLM sometimes places language requirements like `"Deutschkenntnisse (C1)"` into `soft_skills` instead of the dedicated `languages` field. Using a regex rather than a fixed list (`["Englischkenntnisse", "Deutschkenntnisse"]`) is more robust to novel LLM output forms like `"Sprachkenntnisse: Englisch"` or `"Gute Englischkenntnisse"`.

#### Decision: `output_formatter.py` — 9 Invariant Assertions as a Contract

Before the final CSV is written, 9 invariants are asserted:

| # | Invariant | What it catches |
|---|-----------|-----------------|
| 1 | No empty strings in key categoricals | LLM returning `""` instead of `None` |
| 2 | All `job_family` in canonical enum or `None` | Unmapped LLM output that slipped through remap |
| 3 | All list columns are valid JSON arrays | Serialization bugs |
| 4 | `salary_min <= salary_max` | Extraction inversions |
| 5 | Unique `row_id` | Merge/dedup bugs |
| 6 | No sentinel strings in `company_name` | NA convention violations |
| 7 | No gender markers in `title_cleaned` | Incomplete stripping |
| 8 | No duplicate skills within a row | Normalisation bugs |
| 9 | No empty-string elements in skill JSON arrays | Whitespace-only skills |

These invariants are fail-loud (`AssertionError`) by design. Silent data quality degradation in the analysis layer is worse than a clearly-failed pipeline run that demands investigation.

---

## 15. Analysis Component

### Files: `src/analysis/`

#### Decision: `charts.py` — `FIGURES_DIR` as Module-Level Global

```python
FIGURES_DIR: Path | None = None  # set once by notebooks via charts.FIGURES_DIR = path
```

Notebooks set this once at startup in `notebook_init()`. All chart functions check this before saving. This is dependency injection without requiring `figures_dir` to be passed to every function call — trading global state (acceptable for single-threaded notebook use) for convenience.

**Return-the-Figure pattern:** All functions return the `matplotlib.Figure`. Notebooks can further annotate or modify figures after calling the chart function without needing to know the internal matplotlib structure.

#### Decision: `charts.py` — `horizontal_bar` vs `value_bar` API Split

Two separate horizontal bar functions prevent a common notebook mistake:
- `horizontal_bar(series)` — accepts raw categorical Series, calls `.value_counts()` internally
- `value_bar(index, values)` — accepts pre-computed counts

Without this split, callers would either repeatedly call `.value_counts()` before passing in, or forget to call it and get incorrect charts. The explicit API split makes the intended usage unambiguous.

#### Decision: `charts.py` — Avoid seaborn for Heatmap

The heatmap is implemented with raw matplotlib `imshow` rather than `seaborn.heatmap`. This trades seaborn's convenience for fine-grained control over colorscale, annotations, and formatting — important for publication-ready charts. `sns.set_theme` is still used for the general figure style.

#### Decision: `filters.py` — Case-Insensitive String Matching Throughout

All string filters use `.str.lower()` comparisons. String values in the DataFrame are normalized but not guaranteed to be a specific case after cleaning. Case-insensitive matching prevents silent filter failures when case conventions vary.

#### Decision: `utils.py` — `notebook_init()` as Single-Call Setup

```python
df = notebook_init()  # all 9 notebooks start with this one call
```

This single call: applies matplotlib style, loads cleaned CSV, creates figures directory, sets `charts.FIGURES_DIR`, and prints a dataset summary. Future initialization changes are made in one place.

**`parse_flags()` returns a typed empty DataFrame:** Rather than an empty `pd.DataFrame()`, the empty case returns a DataFrame with explicit column names. This prevents `KeyError` in downstream code that tries `df.groupby("rule")` on an empty result.

---

## 16. Configuration Files

### `extraction/config/extraction_prompt.yaml`

#### Decision: Versioned Prompt with Embedded Changelog

The YAML includes a complete changelog (v1.0 through v2.0):
- **v1.0:** Monolithic extraction — LLM extracts everything including salary
- **v1.5:** Tier 1/2 split — salary, seniority, modality removed from LLM prompt (now regex-extracted)
- **v2.0:** Evidence-based extraction — all technical items require a `"source"` field

The history is embedded in the config file, not in a separate document. This ensures the prompt change history is inseparable from the prompt itself.

#### Decision: Two Bilingual Few-Shot Examples

One German example, one English example — both complete and consistent with all described rules. German-only examples underperform on English postings; English-only underperform on German. The bilingual set reflects that real job postings mix both languages.

#### Decision: Explicit `C/C++` Disambiguation Rule in Prompt

*"C and C++ are distinct languages. Only extract C++ if `c++` (with the plus signs) appears explicitly."*

Both the prompt and `fix_cpp_inference` in code address the same known model failure mode. Defense in depth — the prompt tries to prevent the error; the code corrects it if it occurs anyway.

---

### `extraction/config/output_schema.json`

#### Decision: `job_family` Without Enum Constraint

`job_family` is `type: string` rather than `type: string, enum: [...]`. The comment explains: *"Unknown families are handled by `job_family_remap.yaml` in cleaning, not by schema rejection."* This is the visible effect of the "clean-time normalization" design decision — strict schema rejection at parse time would discard recoverable API calls.

#### Decision: `additionalProperties: false` Everywhere

Both at root level and on the `evidence_item` definition. The LLM cannot add unrequested fields like `"confidence"` or `"reasoning"`. This prevents schema drift — the output format stays exactly what the pipeline expects.

#### Decision: `$ref` for `evidence_item`

```json
"$defs": {
  "evidence_item": {
    "type": "object",
    "properties": {"name": {...}, "source": {...}},
    "required": ["name", "source"]
  }
}
```

The `$ref` pattern avoids repeating the `{name, source}` structure for each of the four evidence fields (`technical_skills`, `nice_to_have_skills`, `benefits`, `tasks`). One definition, four references — standard JSON Schema practice.

---

### `extraction/config/skill_aliases.yaml`

#### Decision: Keys Loaded as Lowercase, File Contains Mixed Case

The loader explicitly lowercases all keys. The file has both `Golang: Go` and `golang: Go` for readability — it self-documents that the map is case-insensitive without requiring readers to know the loading implementation.

**Opinionated collapsing decisions:**
- `Docker Compose: Docker` — collapses into parent skill
- `React.js: React` — dominant framework, alias is safe
- `Next.js: Next.js` — kept distinct (not just a React alias)

These choices reflect the developer's judgment about which distinctions matter for analysis.

---

### `extraction/config/german_states.yaml`

#### Decision: Three-Layer State Lookup

```yaml
states:           [Bavaria, Berlin, ...]        # English canonical names
german_to_english: {Bayern: Bavaria, ...}       # German → English mapping
state_codes:       {BY: Bavaria, BE: Berlin, ...} # 2-letter code → canonical
```

LinkedIn uses inconsistent state formatting. Having all three formats pre-built means the parser never needs to apply multiple transformations — it directly looks up the input format.

#### Decision: Duplicate `Hannover-Braunschweig-Göttingen-Wolfsburg Region`

Appears once with the `ö` character and once as `\u00f6`. Comment: *"keep both for safety"* — defends against YAML parsers that may or may not decode Unicode escapes consistently across environments. Idempotent: two entries for the same city cause no harm.

#### Decision: Non-German Regions in the German Region Map

Zürich, Prague, Philadelphia appear with explicit `country` fields. Pragmatic centralization — the region section handles all single-part location strings. Having non-German regions here means one YAML file answers "what city/state/country is this location string?" for all cases.

---

### `cleaning/config/job_family_remap.yaml`

#### Decision: Empirically Built From Observed LLM Output

The 34 remap entries were added as specific bad LLM outputs were discovered during data inspection (e.g., `"Senior Software Engineer"` → `"Software Developer"`, `"Web Developer"` → `"Fullstack Developer"`). This reflects the practical approach: fix what actually appears rather than enumerate all possible variants upfront.

Three sub-dicts (`remap`, `contract_type_remap`, `seniority_remap`) map cleanly to three separate `replace()` calls in `categorical_remapper.py`.

---

## 17. Cross-Cutting Patterns

### Pattern: Defense-in-Depth Validation

Three independent validation layers at different granularities:

| Layer | Where | What it catches |
|-------|-------|-----------------|
| `require_df()` | Before any step consuming `state.df` | Calling a step before ingest ran |
| `validate_step_output()` (Pandera) | At every step boundary | Structural DataFrame schema violations |
| `assert_invariants()` (business rules) | In the export step | Domain-level correctness failures |

Each layer produces different error messages. A missing column caught by Pandera produces a precise schema error. The same issue caught by a business invariant would produce a vague assertion message. Defense in depth means the first layer that catches an issue produces the clearest possible diagnosis.

---

### Pattern: `df.copy()` Discipline — All Functions Are Pure from Callers' Perspective

Every public function that modifies a DataFrame calls `df = df.copy()` first. All functions in the cleaning and extraction pipelines are pure from the caller's perspective — they do not mutate the input DataFrame.

This prevents:
- `SettingWithCopyWarning` pandas warnings
- Unexpected aliasing bugs (caller's DataFrame unexpectedly modified)
- Test failures from shared mutable state between test cases

---

### Pattern: `(x or {})` / `(x or [])` Null Guards

```python
data = r.get("data") or {}
skills = list(data.get("technical_skills") or [])
```

Used pervasively to handle `None` data fields without crashing. Handles three cases simultaneously: key missing from dict, key present with `None` value, key present with empty container. Combined with `list(...)` wrapping to prevent mutation of the original.

---

### Pattern: `max(1, n)` Denominators

```python
pct = count / max(1, total_rows)
```

Every percentage calculation uses this guard. Never crashes on empty DataFrames or empty results lists. Produces 0% for empty input — the correct value.

---

### Pattern: `Path(__file__).parent` Path Resolution — CWD-Agnostic

No module uses `os.getcwd()` or assumes a working directory:
```python
_CONFIG_PATH = Path(__file__).parent / "config" / "settings.yaml"
_REPO_ROOT   = Path(__file__).parent.parent.parent
```

All paths are resolved relative to the source file's physical location. `python orchestrate.py` works from any directory. The rule came from a real bug encountered during development.

---

### Pattern: `cfg.get("key", _DEFAULT_CONSTANT)` Config Access

Config values are read with fallbacks to named constants:
```python
threshold = cfg.get("skill_hallucination_threshold", _DEFAULT_THRESHOLD)
max_workers = cfg.get("deepseek_max_workers", 10)
```

The constants serve as documented defaults when config is absent. Tests can override without modifying constants. Config files can omit settings without crashing the pipeline.

---

### Pattern: Structured Log Events

```python
logger.info("Step complete", extra={
    "event": "step_complete",
    "step": "extract",
    "elapsed_s": 3847.2,
    "rows_processed": 17185
})
```

Key pipeline events are logged with structured `extra` dicts. The JSONL log sink serializes these for post-hoc analysis:
```bash
jq 'select(.event == "step_complete")' logs/pipeline.jsonl
jq 'select(.event == "step_complete") | .elapsed_s' logs/pipeline.jsonl
```

---

### Pattern: `if "col" not in df.columns: return df` Guards

Every public function in the cleaning module guards against missing columns with an early return. This makes the pipeline robust to schema variations between runs — functions are safe to call on DataFrames that don't have the expected column yet.

---

### Pattern: `ensure_ascii=False` on All JSON Dumps

```python
json.dumps(val, ensure_ascii=False)
```

Applied universally for list column serialization. German characters in skill names, benefit descriptions, and company names (`"Bürokommunikation"`, `"Größe"`) are preserved as UTF-8, not escaped to `\u00fc`. The entire pipeline uses `encoding="utf-8"` throughout.

---

## 18. Known Trade-offs and Technical Debt

### Minor Documentation Inaccuracies (Non-Breaking)

| Location | Issue | Impact | Status |
|----------|-------|--------|--------|
| `settings.yaml` | `max_tasks: 10` but Python default is `7`; settings value wins at runtime | Confusing inconsistency | Open |
| `german_states.yaml` | Uses `"NA"` string for non-German region states, contradicting Rule 3 | NA convention violation in config only | Open |

> **Resolved (2026-03-13 audit):** `client.py` KeyError→ValueError docstring fixed; YAML headers referencing `post_clean.py` updated; `skill_canonical_case.yaml` header updated; `title_classification` removed from `_REQUIRED_KEYS`.

---

### Resolved Bugs (2026-03-13 audit)

| Location | Issue | Fix |
|----------|-------|-----|
| `filters.py` `filter_by_seniority` | Referenced `"seniority"` column; actual is `"seniority_from_title"` | Column name corrected |
| `filters.py` `filter_salary_known` | Used `str(x).isdigit()` rejecting float strings | Replaced with `.notna()` check |
| `quality.py` `_hallucination_summary` | Regex `Skill '...'` didn't match actual message format `'...' not grounded` | Regex updated to `'...' not grounded` |
| `cross_field.py` Rule 7 | `modality == "onsite"` never matched canonical `"on-site"` | Now checks both `"on-site"` and `"onsite"` |
| `shared/logging.py` | `_JsonlFormatter` leaked standard LogRecord attrs into JSONL | Uses baseline LogRecord instance keys |
| `date_parser.py` | `datetime.utcfromtimestamp()` deprecated in Python 3.12 | Replaced with `datetime.fromtimestamp(tz=UTC)` |
| `cost.py` default model | Fallback was `"claude-sonnet-4-6"` instead of `"deepseek-chat"` | Default corrected |
| `test_cost_report.py` | Test asserted wrong default model | Test updated |

---

### Intentional Architectural Trade-offs

| Trade-off | Decision Made | Rationale |
|-----------|--------------|-----------|
| Reject vs normalize job families | Normalize in cleaning, never reject at parse time | API calls cost money; recoverable data should be recovered |
| Flag vs remove hallucinated skills | Remove from data (not just flag) | Analysis should not include fabricated skills |
| Checkpoint per-row commit vs batch commit | Per-row commit (slower but safer) | Each row is a real API call; data loss is unacceptable |
| MinHash threshold 0.95 vs lower | Conservative 0.95 | Minimize false-positive groupings at cost of missing some near-duplicates |
| `iterrows()` in `re_verify_skills_post_clean` | Accepted for current dataset size | O(n·m) but fine for 17k rows; vectorization would significantly complicate logic |
| DACH region exclusion in `location_filter.py` | DACH rows excluded | Germany-only analysis; DACH-labelled remote jobs may not be Germany-specific |
| `job_families.yaml` notes not injected into prompt | Notes maintained manually in sync with prompt | Adding notes to prompt injection would add ~300 tokens per call |
| `_load_valid_job_families` (private fn) imported in `export.py` | Accepted coupling | Avoids duplicating the logic; minor architectural smell |
| `_nn()` / `_to_str()` NaN guard defined in both `runner.py` and `cross_field.py` | Not extracted to shared | Both work correctly; DRY violation but not a bug |

---

### Performance Notes

| Location | Current approach | Scalability note |
|----------|-----------------|-----------------|
| `re_verify_skills_post_clean` | `iterrows()` Python loop | O(n·m); fine for 17k rows; vectorize if dataset grows 10× |
| `assert_invariants` invariants 8 and 9 | Python `for` loop over rows | O(n·m); fine for current dataset size |
| `evaluate()` in `evaluation.py` | `df.iterrows()` for Tier 1 overlay | `df.set_index("row_id")` would be faster for large golden sets |
| `_build_lookups` in `parse_location` | Called per-row in single-row API | Called once per batch in `parse_all_locations`; inconsistent but single-row API is rarely on a hot path |

---

*This document was produced from a full static code analysis of all 68 Python source files, 14 YAML configuration files, and 1 JSON schema in this repository. Update this document whenever a design decision changes, a new edge case is discovered, or a technical debt item is resolved.*
