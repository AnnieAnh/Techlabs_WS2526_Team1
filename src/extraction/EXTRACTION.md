# Extraction Component (LLM Pipeline)

> **Status:** Complete and tested (760+ tests)
> **Location:** `src/extraction/`
> **Input:** `data/ingestion/combined_jobs.csv` (~22,526 rows)
> **Output:** `data/cleaning/cleaned_jobs.csv` (~18,500 rows, analysis-ready)
> **LLM Provider:** DeepSeek V3 (OpenAI-compatible API)

---

## Purpose

The extraction component is the **core of the pipeline**. It takes normalized job postings and enriches them with structured data extracted via regex patterns (Tier 1) and LLM semantic analysis (Tier 2). The pipeline runs in 8 sequential steps: ingest, prepare, deduplicate, regex_extract, extract, validate, clean+enrich, and export.

---

## Folder Structure

```
src/
├── pipeline_state.py                    # Typed state container (PipelineState dataclass)
│
├── steps/                               # One module per pipeline step
│   ├── ingest.py                        #   Step 1: Load source CSVs
│   ├── prepare.py                       #   Step 2: Validate, parse, normalize
│   ├── deduplicate.py                   #   Step 3: URL + composite + description dedup
│   ├── regex_extract.py                 #   Step 4: Regex field extraction
│   ├── extract.py                       #   Step 5: LLM semantic extraction
│   ├── validate.py                      #   Step 6: Post-extraction corrections
│   ├── clean_enrich.py                  #   Step 7: Merge, clean, enrich
│   └── export.py                        #   Step 8: Column ordering, invariants, CSV output
│
└── extraction/                          # Pipeline modules
    ├── preprocessing/                   # Input processing
    │   ├── validate_input.py            #   Input quality flags (short/privacy/invalid)
    │   ├── location_parser.py           #   German city/state/country extraction
    │   ├── title_normalizer.py          #   Gender suffix removal, DE→EN translation
    │   ├── text_preprocessor.py         #   HTML/emoji/markdown cleanup
    │   └── regex_extractor.py           #   Tier 1: salary, contract, modality, experience
    │
    ├── dedup/                           # Deduplication
    │   ├── row_dedup.py                 #   URL + composite key dedup
    │   └── description_dedup.py         #   SHA-256 grouping + MinHash LSH near-dedup
    │
    ├── llm/                             # LLM infrastructure
    │   ├── client.py                    #   Async DeepSeek API wrapper + retry
    │   ├── processor.py                 #   Semaphore-limited concurrent extraction
    │   ├── prompt_builder.py            #   Per-row message construction + truncation
    │   └── response_parser.py           #   JSON parsing + schema validation
    │
    ├── validators/                      # Post-extraction validation
    │   ├── __init__.py                  #   ValidationFlag dataclass
    │   ├── runner.py                    #   Orchestrate all validation passes
    │   ├── hallucination.py             #   Hallucination detection (skill vs description)
    │   ├── skills.py                    #   Alias resolution, dedup, contradiction removal
    │   ├── cross_field.py               #   7 consistency rules across fields
    │   └── salary.py                    #   Salary bounds checking
    │
    ├── reporting/                       # Reports
    │   ├── cost.py                      #   Token usage + USD cost calculation
    │   ├── quality.py                   #   Field coverage, distributions, hallucinations
    │   └── evaluation.py               #   Golden set accuracy comparison (optional)
    │
    ├── checkpoint.py                    #   SQLite state tracking (crash-safe resume)
    ├── post_extraction.py               #   Categorical remap, C++ fix, skill casing
    ├── exporter.py                      #   Merge regex + LLM results
    │
    └── config/                          # Configuration files
        ├── settings.yaml                #   Paths, model config, thresholds
        ├── extraction_prompt.yaml       #   LLM system prompt + few-shot examples (v2.0)
        ├── output_schema.json           #   JSON Schema for LLM output validation
        ├── job_families.yaml            #   42 canonical job family categories
        ├── skill_aliases.yaml           #   Skill name normalization map
        ├── skill_variants.yaml          #   Variant spellings for fuzzy matching
        ├── title_translations.yaml      #   German job title → English translations
        └── german_states.yaml           #   German state codes + names

tests/extraction/                        # 760+ test cases
├── test_*.py                            #   Unit + integration tests for every module
└── test_smoke.py                        #   14 smoke tests covering 5 pre-LLM stages

data/extraction/                         # Pipeline outputs (not in git)
├── pipeline_state.db                    #   SQLite checkpoint database
├── deduped/                             #   Deduped CSVs with timestamps
├── extracted/                           #   extraction_results.json, token_usage.json
├── failed/                              #   parse_failures.json
└── reports/                             #   quality_report.json, cost_report.json, etc.
```

---

## Pipeline Flow (8 Steps)

```
                    ┌─────────────────────────────────────────┐
                    │           orchestrate.py                 │
                    │  Entry point: poetry run python orchestrate.py      │
                    │  Resume-capable via progress.json        │
                    └──────────────┬──────────────────────────┘
                                   │
     ┌─────────────────────────────┼─────────────────────────────┐
     │                             │                             │
     ▼                             ▼                             ▼

 STEP 1: INGEST            STEP 2: PREPARE            STEP 3: DEDUPLICATE
 ┌────────────────┐        ┌────────────────┐         ┌────────────────────┐
 │ Load 3 source  │        │ Input validation│         │ Filter: privacy_   │
 │ CSVs, normalize│───────>│ Location parsing│────────>│   wall + invalid   │
 │ schema, parse  │        │ Title normalize │         │   URL rows removed │
 │ dates          │        │ NA standardize  │         │ Pass 1: URL exact  │
 │ (22,526 rows)  │        └────────────────┘         │ Pass 2: Title+Co+  │
 └────────────────┘                                    │   Location key     │
                                                       │ Pass 3: SHA-256    │
                                                       │ Pass 4: MinHash LSH│
                                                       └─────────┬──────────┘
                                                                 │
     ┌───────────────────────────────────────────────────────────┘
     │
     ▼

 STEP 4: REGEX_EXTRACT         STEP 5: EXTRACT
 ┌─────────────────────┐       ┌─────────────────────┐
 │ 8 regex fields on   │       │ Filter to represen- │
 │ clean, deduped rows │──────>│ tative rows only    │
 │ salary, contract,   │       │ (~10k unique descs) │
 │ modality, seniority,│       │ Call DeepSeek V3    │
 │ experience, lang,   │       │ Async + semaphore   │
 │ education           │       │ Fan out to groups   │
 └─────────────────────┘       └──────────┬──────────┘
                                          │
     ┌────────────────────────────────────┘
     │
     ▼

 STEP 6: VALIDATE            STEP 7: CLEAN+ENRICH         STEP 8: EXPORT
 ┌─────────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
 │ Skill normalization  │      │ Merge regex + LLM   │      │ Drop internal cols   │
 │ Hallucination detect │      │ NA standardization   │      │ Reorder → COLUMN_   │
 │ Cross-field checks   │─────>│ City name normalize  │─────>│   ORDER             │
 │ Salary sanity        │      │ Categorical remap    │      │ Save .debug.csv      │
 │ C++ fix, skill case  │      │ Benefit categories   │      │ Assert 9 invariants  │
                                                        │ Rename → final CSV   │
                                                        │ Generate cost report │
                                                        └─────────────────────┘
                                                                    │
                                                                    ▼
                                                        data/cleaning/cleaned_jobs.csv
```

---

## Step-by-Step Detail

### Step 1: Ingest (`src/steps/ingest.py`)

**Goal:** Load source CSVs, normalize schema, parse dates.

**What it does:**
1. Delegates to `src/ingestion/pipeline.py` → `run_pipeline(cfg=cfg)`
2. Reads 3 raw CSVs, maps columns to unified schema
3. Parses dates to ISO format (relative dates → pinned reference `2026-02-15`)
4. Validates output with Pandera schema
5. Writes `data/ingestion/combined_jobs.csv`

**Input:** 3 raw CSVs from `data/raw/`
**Output:** `state.df` with ~22,526 rows × 7 columns

---

### Step 2: Prepare (`src/steps/prepare.py`)

**Goal:** Validate input quality, parse locations, normalize titles, extract regex fields.

**Sub-tasks (in order):**

#### 2a. Input Validation (`src/extraction/preprocessing/validate_input.py`)
Flags per-row quality issues without removing rows:
- `missing_title` — empty title field
- `missing_company` — empty company name
- `invalid_url` — URL doesn't start with `http`
- `short_description` — description < 250 characters
- `privacy_wall` — cookie/privacy wall detected (removed in Step 3 before description grouping)
- `invalid_date` — unparseable date
- `date_anomaly` — date before 2024-01-01

#### 2b. Location Parsing (`src/extraction/preprocessing/location_parser.py`)
Extracts structured location from raw strings:
- Parses `"Stuttgart, BW, DE"` → `city="Stuttgart"`, `state="Baden-Württemberg"`, `country="Germany"`
- Handles German city names, state abbreviations (BY, BW, NRW, etc.)
- **Non-German filter:** Removes rows with US state codes, non-German countries, EMEA/APAC regions
- Result: ~19,000 rows after country filter

#### 2c. Title Normalization (`src/extraction/preprocessing/title_normalizer.py`)
- Strips 80+ gender suffix variants: `(m/w/d)`, `(gn)`, `:in`, `*in`, `-in`
- Translates German titles to English via `title_translations.yaml`
- Fixes ALL-CAPS → Title Case
- Output: `title_cleaned` column

---

### Step 3: Deduplicate (`src/steps/deduplicate.py`)

**Goal:** Filter garbage rows, remove duplicates, and group identical descriptions for LLM cost optimization.

**Pre-filter:** Rows flagged `privacy_wall` or `invalid_url` by Step 2 are removed before description grouping. This prevents cookie-consent / error-page text from poisoning MinHash LSH groups sent to the LLM.

**Four dedup passes (after filter):**

| Pass | Method | Key | Rows Removed | Cumulative |
|------|--------|-----|-------------|------------|
| 1 | URL exact | `job_url` | 0 | 22,526 |
| 2 | Composite key | `lower(title_cleaned + company_name + location)` | ~1,149 | ~21,377 |
| 3 | Description hash | SHA-256 of normalized description | Groups formed | ~21,377 |
| 4 | MinHash LSH | Jaccard similarity ≥ 95% | Groups merged | ~21,377 |

**Description grouping (Pass 3-4):** Identical or near-identical descriptions are grouped. Only one **representative row** per group is sent to the LLM. Results are fanned out to all group members. This saves ~19.6% of LLM API calls (~4,192 duplicate descriptions).

**Output:** `state.df` (deduplicated), `state.description_groups` (group metadata), `state.dedup_report` (includes `privacy_wall_removed` and `invalid_url_removed` counts)

---

### Step 4: Regex Extract (`src/steps/regex_extract.py`)

**Goal:** Deterministic regex field extraction on clean, filtered, deduped rows.

Runs after dedup so regex processes only filtered, deduped rows — fewer rows to process and no wasted work on rows that will be removed.

**Tier 1 fields** — fast, deterministic, no LLM needed:

| Field | Pattern Examples | Output |
|-------|-----------------|--------|
| `regex_contract_type` | Vollzeit, Teilzeit, Freelance | `"Full-time"`, `"Part-time"`, etc. |
| `regex_work_modality` | Home Office, Remote, Hybrid | `"Remote"`, `"Hybrid"`, `"On-site"` |
| `regex_salary_min` | `60.000 EUR`, `ab 60k` | `60000` (float) |
| `regex_salary_max` | `80.000 EUR` | `80000` (float) |
| `regex_experience_years` | `3 Jahre Berufserfahrung` | `3` (int) |
| `regex_seniority_from_title` | Senior, Junior, Lead keywords | `"Senior"`, `"Junior"`, etc. |
| `regex_languages` | Deutsch C1, English fluent | JSON array of `{language, level}` |
| `regex_education_level` | Master, Bachelor, Abitur | Highest degree found |

**German salary format:** `50.000` = 50,000 EUR (dot = thousands separator). The regex handles this.

**Work modality order matters:** Hybrid is checked BEFORE Remote because hybrid job ads mention "Home Office" which would falsely match Remote.

**Schema guard:** Requires `title_cleaned`, `description`, `row_id` columns from Step 2.

**Output:** `state.df` with 8 additional `regex_*` columns

---

### Step 5: Extract (`src/steps/extract.py`)

**Goal:** Extract semantic fields from job descriptions via DeepSeek V3 LLM.

**Architecture:**

```
state.df (18.5k rows)
    │
    ▼
Filter to representative rows only (~10k unique descriptions)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  src/extraction/llm/prompt_builder.py                       │
│  • Loads extraction_prompt.yaml (v2.0)                      │
│  • Builds per-row API message: system prompt + user data    │
│  • Truncates descriptions to 8,000 tokens (~28,000 chars)   │
│  • Adds [TRUNCATED] marker for truncated descriptions       │
│  • Emits validation_flag with rule: "truncated" on cutoff   │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  src/extraction/llm/processor.py                            │
│  • asyncio.Semaphore(concurrency=10) limits parallel calls  │
│  • tqdm.asyncio for progress bars                           │
│  • Resumes from checkpoint (skips already-extracted rows)   │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  src/extraction/llm/client.py                               │
│  • AsyncOpenAI client (OpenAI-compatible SDK)               │
│  • Base URL: https://api.deepseek.com                       │
│  • Model: deepseek-chat (V3)                                │
│  • Temperature: 0 (deterministic)                           │
│  • Exponential backoff on 429/5xx (max 3 retries)           │
│  • Tracks input/output tokens per call                      │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  src/extraction/llm/response_parser.py                      │
│  • 4-strategy JSON extraction:                              │
│    1. Direct json.loads                                     │
│    2. Strip markdown fences (```json...```)                  │
│    3. Fix trailing commas                                   │
│    4. Extract between first { and last }                    │
│  • Schema validation: critical (technical_skills) vs non-critical │
│  • Coercions: skills string→array, tasks truncated to max   │
│  • Emits list_truncated validation_flag on truncation       │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
Fan out results to all description group members
    │
    ▼
Save: data/extraction/extracted/extraction_results.json
```

**Tier 2 fields** (LLM-extracted):

| Field | Type | Description |
|-------|------|-------------|
| `technical_skills` | string[] | Named technologies (max 25) |
| `soft_skills` | string[] | Interpersonal skills (max 10) |
| `nice_to_have_skills` | string[] | Optional technologies |
| `benefits` | string[] | Tangible perks offered |
| `tasks` | string[] | Main responsibilities (max configurable, default 7 via `settings.yaml`) |
| `job_family` | enum | One of 42 canonical role categories |
| `job_summary` | string | One-sentence role summary |

**Critical extraction rule:** *"ONLY extract a skill if its name appears literally in the description"* — prevents hallucination.

---

### Step 6: Validate (`src/steps/validate.py`)

**Goal:** Post-extraction corrections, hallucination detection, quality reporting.

**Validation passes (in order):**

1. **Skill normalization** (`src/extraction/validators/skills.py`)
   - Resolve aliases: `typescript` → `TypeScript`
   - Deduplicate case variants within same row
   - Remove contradictions

2. **Skill hallucination detection** (`src/extraction/validators/hallucination.py`)
   - For each extracted skill, verify it appears in the description text
   - 3-tier matching: word-boundary → variant substring → dehyphenated form
   - If >30% of skills are unverified → flag + **remove** unverified skills

3. **Cross-field validation** (`src/extraction/validators/cross_field.py`)
   - 7 consistency rules: seniority match, contract type consistency, experience reasonableness, salary range validity, skill count limits, job family compliance

4. **Salary sanity** (`src/extraction/validators/salary.py`)
   - Floor: EUR 10,000/year | Ceiling: EUR 300,000/year
   - Flags: `below_floor`, `above_ceiling`, `inverted_range`

5. **Categorical remap** (`src/extraction/post_extraction.py`)
   - Maps LLM output variants → canonical forms (e.g., "Software Engineer" → "Software Developer")

6. **C++ hallucination fix** (`src/extraction/post_extraction.py`)
   - If description has `C++` → keep; if only bare `C` → replace `C++` with `C`; if neither → remove `C++`

7. **Quality report** (`src/extraction/reporting/quality.py`)
   - Field coverage %, top values, hallucination summary
   - Saves `data/extraction/reports/quality_report.json`

---

### Step 7: Clean + Enrich (`src/steps/clean_enrich.py`)

**Goal:** Merge extraction results into DataFrame, fix data quality, add enrichment columns.

**Operations:**

1. **Merge results** — Promote `regex_*` columns to canonical names; merge LLM Tier 2 columns via left join on `row_id`
2. **NA standardization** (2nd pass) — Now handles LLM columns that didn't exist before merge
3. **Numeric column fixing** — Cast salary to float, experience to int; German format handling
4. **City name normalization** — `München` → canonical form via `city_name_map.yaml`
5. **Categorical remapping** — Safety-net reapplication of step 5 remaps
6. **Company name fixes** — Remove placeholders, strip residual gender markers
7. **Benefit categorization** — Map benefits to 11 categories (PTO, Flexibility, Health, etc.)
8. **Soft skill normalization** — Map to 8 canonical categories (Communication, Teamwork, etc.)
9. **Description quality flags** — Detect concatenated/truncated/boilerplate descriptions
10. **Re-verify skills** — Idempotent re-check after casing normalization

---

### Step 8: Export (`src/steps/export.py`)

**Goal:** Final column ordering, invariant checks, CSV output.

**Operations:**

1. **Drop internal columns** (`country`, `location`, `source_file`, `title_original`)
2. **Drop rows** with empty `job_family` (parse failures)
3. **Reorder columns** to `COLUMN_ORDER` (29 canonical columns)
4. **Save `.debug.csv`** (preserved on failure for forensics)
5. **Assert 9 invariants:**
   - No empty strings in categoricals
   - Valid `job_family` enum values
   - Valid JSON arrays in list columns
   - Salary min ≤ max
   - Unique `row_id`s
   - No `"NA"` strings in company_name
   - No gender markers in `title_cleaned`
   - No duplicate skills (case-insensitive)
   - No empty-string skill elements
6. **Rename `.debug.csv` → final output** on success
7. **Generate cost report** — Token usage + USD cost

---

## Two-Tier Extraction Architecture

The pipeline uses a **two-tier extraction strategy** to minimize LLM costs:

```
┌──────────────────────────────────────────────────────────────────────┐
│  TIER 1 — Regex Extraction (Step 4)                                  │
│  Fast, deterministic, zero cost                                      │
│                                                                      │
│  Fields: contract_type, work_modality, salary_min, salary_max,       │
│          experience_years, seniority_from_title, languages,          │
│          education_level                                             │
│                                                                      │
│  8 fields × 22,526 rows = instant                                    │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  TIER 2 — LLM Extraction (Step 5)                                    │
│  Semantic understanding, API cost                                    │
│                                                                      │
│  Fields: technical_skills, soft_skills, nice_to_have_skills,         │
│          benefits, tasks, job_family, job_summary                    │
│                                                                      │
│  7 fields × ~10,000 representative rows = ~$6–10 total              │
└──────────────────────────────────────────────────────────────────────┘
```

**Cost optimization:** Description grouping reduces LLM calls by ~19.6% (4,192 duplicate descriptions skipped).

---

## Checkpointing & Resume

The pipeline is **crash-safe** with two layers of state tracking:

1. **Step-level progress** (`data/pipeline_progress.json`)
   - Tracks which steps completed, with timestamps
   - Orchestrator skips completed steps on restart

2. **Row-level checkpoint** (`data/extraction/pipeline_state.db`, SQLite)
   - Tracks per-row stage progress (atomic commits)
   - LLM extraction resumes from last successful row

---

## Configuration

### `src/extraction/config/settings.yaml`
```yaml
paths:
  raw_dir: data/ingestion
  extracted_dir: data/extraction/extracted
  checkpoint_db: data/extraction/pipeline_state.db
extraction:
  model: deepseek-chat             # V3 (not reasoner)
  max_tokens: 2000                 # Typical response: ~350 tokens
  temperature: 0                   # Deterministic
  max_description_tokens: 8000
  batch_size: 1000
llm:
  provider: deepseek
  deepseek_max_workers: 10         # Semaphore concurrency limit
validation:
  salary_min_floor: 15000
  salary_max_ceiling: 300000
  skill_hallucination_threshold: 0.3
```

### `src/extraction/config/extraction_prompt.yaml` (v2.0)
- ~3KB system prompt + 2 few-shot examples (German + English)
- Critical constraint: *"Extract skill ONLY if name appears literally"*
- Max items: 25 technical, 10 soft, 7 tasks

### `src/extraction/config/output_schema.json`
- JSON Schema Draft 7 for LLM output validation
- `job_family` is an enum of 42 values (critical — hard fail if invalid)
- Other fields demoted to warnings on validation failure

---

## How to Run

### Full Pipeline
```bash
poetry run python orchestrate.py
# or
make pipeline
```

### Single Step
```bash
poetry run python orchestrate.py --only extract
# or
make extract
```

### Resume from a Step
```bash
poetry run python orchestrate.py --from validate
```

### Dry Run (show what would run)
```bash
poetry run python orchestrate.py --dry-run
# or
make dry-run
```

### List All Steps
```bash
poetry run python orchestrate.py --list
```

### Reset Progress
```bash
poetry run python orchestrate.py --reset
```

### Run Without LLM (Regex Only)
```bash
poetry run python orchestrate.py --no-llm
# Skips extract + validate steps; LLM columns will be empty
# Useful for CI, testing, or inspecting regex-only data without API spend
```

### Tests
```bash
poetry run pytest tests/extraction/ -v          # 760+ tests
poetry run ruff check src/extraction/           # Linting
make test                                       # All tests across all components
```

### Prerequisites
```bash
poetry install                        # Install dependencies (from repo root)
export DEEPSEEK_API_KEY=sk-...        # Set API key (never hardcode)
```

---

## Key Design Decisions

1. **Hallucination prevention** — Extraction prompt forbids skill inference; verifier cross-checks 100% of skills against description text
2. **German-specific handling** — Salary format (`50.000`), gender suffixes (80+ patterns), title translations, location parsing
3. **Cost optimization** — Description grouping saves ~20% of LLM calls; two-tier extraction avoids LLM for deterministic fields
4. **Dual correction pipeline** — Post-extraction corrections are authoritative; cleaning step corrections are safety nets for robustness
5. **Crash safety** — SQLite checkpoint + JSON progress tracking enable seamless resume after interruption

---

## Data Flow Summary

```
data/ingestion/combined_jobs.csv       22,526 rows  ·  7 columns
        │
        ▼  [Step 1: Ingest]
   Load + validate schema
        │
        ▼  [Step 2: Prepare]
   + city, state, country, title_cleaned
   – non-German rows filtered out
        │
        ▼  [Step 3: Deduplicate]
   – URL duplicates, composite key duplicates
   + description groups for cost optimization
   ~18,500 rows remaining
        │
        ▼  [Step 4: Regex Extract]
   8 deterministic regex fields on deduped rows
        │
        ▼  [Step 5: Extract]
   + technical_skills, soft_skills, nice_to_have_skills
   + benefits, tasks, job_family, job_summary
   ~10k LLM calls (grouped descriptions)
        │
        ▼  [Step 6: Validate]
   Corrections: skill removal, categorical remap, C++ fix
   + validation_flags per row
        │
        ▼  [Step 7: Clean + Enrich]
   Merge Tier 1 + Tier 2 results
   + benefit_categories, soft_skill_categories
   + description_quality flags
        │
        ▼  [Step 8: Export]
   29 columns, 9 invariant checks
   data/cleaning/cleaned_jobs.csv    ~18,500 rows · 29 columns
```

See [Cleaning Documentation](../cleaning/CLEANING.md) for the cleaning step details, or [Analysis Documentation](../analysis/ANALYSIS.md) for downstream analysis.
