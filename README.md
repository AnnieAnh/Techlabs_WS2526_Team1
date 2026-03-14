# German IT Job Market Analysis Pipeline

An end-to-end data pipeline for analysing **~22,500 German IT job postings** scraped from LinkedIn and Indeed. Raw postings flow through five components — ingestion, extraction, cleaning, and analysis — producing structured, analysis-ready data with 24 enriched fields per job.

---

## Pipeline Flowchart

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         GERMAN IT JOB MARKET PIPELINE                            │
└──────────────────────────────────────────────────────────────────────────────────┘

   ┌─────────────┐      ┌─────────────┐      ┌──────────────────────────────────┐
   │  RAW DATA    │      │  INGESTION   │      │         EXTRACTION               │
   │             │      │             │      │  (8 sequential steps)            │
   │ Indeed CSV   │─────>│ Normalize    │─────>│                                  │
   │ LinkedIn CSV │      │ schema,      │      │  Step 1: Ingest                  │
   │ LinkedIn CSV │      │ parse dates, │      │  Step 2: Prepare (validate,      │
   │             │      │ validate     │      │    parse locations, normalize)    │
   │ 22,500 raw  │      │             │      │  Step 3: Deduplicate (filter      │
   │ postings    │      │ 22,526 rows  │      │    flagged, URL, composite,      │
   │ (3 CSVs)    │      │ (7 columns)  │      │    grouping, MinHash LSH)        │
   └─────────────┘      └─────────────┘      │  Step 4: Regex Extract (8 fields)│
                                              │  Step 5: Extract (DeepSeek V3    │
                                              │    LLM, async, grouped)          │
                                              │  Step 6: Validate (hallucination │
                                              │    detection, corrections)       │
                                              │  Step 7: Clean + Enrich (merge,  │
                                              │    categorize, quality flags)    │
                                              │  Step 8: Export (invariants,     │
                                              │    column ordering, cost report) │
                                              │                                  │
                                              │  ~18,500 rows · 24 columns       │
                                              └────────────────┬─────────────────┘
                                                               │
                                              ┌────────────────▼─────────────────┐
                                              │          CLEANING                 │
                                              │                                  │
                                              │  11 cleaning steps               │
                                              │  5 enrichment steps              │
                                              │  9 invariant checks              │
                                              │                                  │
                                              │  Output: cleaned_jobs.csv        │
                                              └────────────────┬─────────────────┘
                                                               │
                                              ┌────────────────▼─────────────────┐
                                              │          ANALYSIS                 │
                                              │                                  │
                                              │  9 Jupyter notebooks             │
                                              │  35 charts + visualizations      │
                                              │  Market overview, skills,        │
                                              │  salary, location, remote,       │
                                              │  seniority, benefits,            │
                                              │  languages, job seeker guide     │
                                              └──────────────────────────────────┘
```

---

## Workflow Flowchart

```
Developer / Analyst
        │
        ▼
┌─────────────────────────────────────────┐
│         python orchestrate.py            │
│   (single entry point for everything)    │
└───────────────────┬─────────────────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
   --dry-run    --only X    (default)
   Show plan    Run one     Run all
   No changes   step only   pending steps
                            (auto-resume)
        │           │           │
        └───────────┼───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  Progress Tracking     │
        │  pipeline_progress.json│
        │  (crash-safe resume)   │
        └───────────┬───────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
   Steps 1-4               Step 5
   (Regex-based)           (LLM API)
   No API cost             DeepSeek V3
   Deterministic           ~$6-10 total
   Instant                 ~10k calls
        │                       │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  Steps 6-8             │
        │  Validate + Clean +    │
        │  Export final CSV       │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  cd notebooks          │
        │  jupyter notebook      │
        │  (9 interactive        │
        │   analysis notebooks)  │
        └───────────────────────┘
```

---

## Quick Start

### Prerequisites

- **Python 3.11+** installed
- **Poetry** for dependency management
- **DeepSeek API key** for LLM extraction (Step 5 only)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd extractors

# Install dependencies via Poetry (from repo root)
poetry install
```

### Set API Key

```bash
# Linux/Mac
export DEEPSEEK_API_KEY=sk-...

# Windows PowerShell
$env:DEEPSEEK_API_KEY = "sk-..."
```

> **Never hardcode the API key.** Use environment variables or a `.env` file.

### Run the Pipeline

```bash
# Full pipeline (resumes automatically from last completed step)
python orchestrate.py

# Or use Make
make pipeline
```

### Run Analysis Notebooks

```bash
cd notebooks/
jupyter notebook
# Open any notebook (01–09) and run all cells
```

---

## How to Run — Command Reference

### Pipeline Commands

| Command | Description |
|---------|-------------|
| `python orchestrate.py` | Run all pending steps (auto-resume) |
| `python orchestrate.py --only ingest` | Run only the ingest step |
| `python orchestrate.py --only extract` | Run only the extract step (LLM) |
| `python orchestrate.py --from validate` | Resume from validate onwards |
| `python orchestrate.py --dry-run` | Show what would run (no changes) |
| `python orchestrate.py --list` | List all 8 steps with descriptions |
| `python orchestrate.py --reset` | Clear progress and start fresh |

### Make Targets

| Target | Description |
|--------|-------------|
| `make pipeline` | Run all 8 steps |
| `make ingest` | Run only ingest |
| `make extract` | Run only extract |
| `make clean-enrich` | Run only clean+enrich |
| `make export` | Run only export |
| `make dry-run` | Show plan without executing |
| `make test` | Run all tests (920+ across all components) |
| `make lint` | Auto-fix lint issues with ruff |
| `make type-check` | Run mypy type checking |

### Development Commands

```bash
# Run all tests (from repo root)
poetry run pytest tests/ -v

# Run tests for a specific domain
poetry run pytest tests/extraction/ -v          # Extraction (760+ tests)
poetry run pytest tests/cleaning/ -v            # Cleaning (75+ tests)
poetry run pytest tests/ingestion/ -v           # Ingestion (6 tests)
poetry run pytest tests/analysis/ -v            # Analysis
poetry run pytest tests/integration/ -v         # Integration / smoke tests

# Lint
poetry run ruff check . --fix

# Type check
poetry run mypy src/extraction/ src/shared/ --ignore-missing-imports

# Check pipeline progress
python orchestrate.py --dry-run
```

---

## Project Structure

```
extractors/                              ← git root
│
├── orchestrate.py                       ← MAIN ENTRY POINT (8-step pipeline)
├── pyproject.toml                       ← Single Poetry config + all dependencies
├── Makefile                             ← Make targets for all operations
│
├── src/                                 ← ALL source packages
│   ├── pipeline_state.py                ← Typed state container
│   ├── ingestion/                       ← Schema normalization + date parsing
│   │   ├── pipeline.py, loader.py, date_parser.py
│   │   └── config/                      ← settings.yaml (source definitions)
│   ├── extraction/                      ← LLM extraction pipeline
│   │   ├── preprocessing/               ← Input validation, text/title/location normalization
│   │   ├── dedup/                       ← URL dedup + description near-duplicate grouping
│   │   ├── llm/                         ← DeepSeek client, async processor, parser
│   │   ├── validators/                  ← Skill hallucination, cross-field, salary checks
│   │   ├── reporting/                   ← Cost, quality, evaluation reports
│   │   ├── checkpoint.py, exporter.py, post_extraction.py
│   │   └── config/                      ← YAML prompts, JSON schemas, settings
│   ├── cleaning/                        ← Post-extraction data quality + enrichment
│   │   ├── pipeline.py, skill_normalizer.py, benefit_categorizer.py, ...
│   │   └── config/                      ← Category mappings (benefits, skills, cities)
│   ├── analysis/                        ← Chart helpers + notebook utilities
│   │   └── utils.py, style.py, charts.py, filters.py
│   ├── steps/                           ← Thin orchestrator wrappers (8 step modules)
│   │   └── ingest.py, prepare.py, ..., export.py
│   └── shared/                          ← Cross-component utilities
│       └── io.py, config.py, schemas.py, constants.py, logging.py, ...
│
├── tests/                               ← ALL tests, structured by domain
│   ├── extraction/                      ← 760+ tests
│   ├── cleaning/                        ← 75+ tests
│   ├── ingestion/                       ← 6 tests
│   ├── analysis/                        ← Analysis utility tests
│   └── integration/                     ← End-to-end + step smoke tests
│
├── data/                                ← ALL pipeline I/O, scoped by domain
│   ├── raw/                             ← 3 source CSVs (read-only)
│   ├── ingestion/                       ← combined_jobs.csv
│   ├── extraction/                      ← batches/, deduped/, extracted/, reports/
│   ├── cleaning/                        ← cleaned_jobs.csv (final output)
│   └── analysis/                        ← figures/
│
├── notebooks/                           ← 9 Jupyter analysis notebooks (01–09)
└── backlog/                             ← Planning docs + archive

Each component has its own README.md under src/ with full documentation.
```

---

## Data Flow

```
data/raw/*.csv                               3 CSVs, 22,500 raw postings
        │
        │  [Step 1: Ingest]
        │  Schema normalization, date parsing
        ▼
data/ingestion/combined_jobs.csv             22,526 rows · 7 columns
        │
        │  [Step 2: Prepare]
        │  Input validation, location parsing, title normalization
        │  Non-German rows filtered out
        ▼
                                             ~19,000 rows · 7+ columns
        │
        │  [Step 3: Deduplicate]
        │  Filter privacy-wall + invalid-URL rows
        │  URL exact, composite key, description grouping, MinHash LSH
        ▼
                                             ~18,500 rows (description groups formed)
        │
        │  [Step 4: Regex Extract]
        │  8 regex fields: salary, contract, modality, experience,
        │  seniority, languages, education on clean deduped rows
        ▼
                                             + 8 regex_* columns
        │
        │  [Step 5: Extract — LLM]
        │  DeepSeek V3 API (async, semaphore-limited)
        │  ~10,000 unique descriptions (grouped to save ~20% cost)
        │  Extracts: skills, benefits, tasks, job_family, summary
        ▼
                                             + 7 LLM-extracted fields per row
        │
        │  [Step 6: Validate]
        │  Skill hallucination detection + removal
        │  Cross-field consistency, salary sanity, categorical remapping
        ▼
                                             Corrections applied, flags added
        │
        │  [Step 7: Clean + Enrich]
        │  Merge Tier 1 (regex) + Tier 2 (LLM) results
        │  City normalization, benefit/soft skill categorization
        │  Description quality flagging
        ▼
                                             ~18,500 rows · 24+ columns
        │
        │  [Step 8: Export]
        │  9 invariant checks, column ordering, cost report
        ▼
data/cleaning/cleaned_jobs.csv               ~18,500 rows · 24 columns (final)
        │
        │  [Analysis Notebooks]
        ▼
data/analysis/figures/*.png                  35 charts across 9 notebooks
```

---

## Component Documentation

| Component | Documentation | Description |
|-----------|--------------|-------------|
| Ingestion | [src/ingestion/README.md](src/ingestion/README.md) | Schema normalization, date parsing, configuration |
| Extraction | [src/extraction/README.md](src/extraction/README.md) | Full LLM pipeline (8 steps), two-tier extraction, DeepSeek API |
| Cleaning | [src/cleaning/README.md](src/cleaning/README.md) | 11 cleaning + 5 enrichment steps, invariant checks |
| Analysis | [src/analysis/README.md](src/analysis/README.md) | 9 notebooks, chart types, visualization gallery |
| Shared | [src/shared/README.md](src/shared/README.md) | IO utilities, constants, schemas, logging |
| Raw Data | [data/raw/README.md](data/raw/README.md) | Raw data sources, CSV schemas, column mapping |

---

## Technologies

### Core

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.11+ | Runtime |
| **Poetry** | latest | Dependency management, virtual environment |
| **pandas** | ^3.0 | DataFrame processing, CSV handling |
| **DeepSeek V3** | API | LLM for semantic field extraction (OpenAI-compatible) |
| **OpenAI SDK** | ^2.21 | Client library for DeepSeek API (compatible protocol) |
| **SQLite** | built-in | Checkpoint database for crash-safe pipeline resume |

### Data Processing

| Technology | Purpose |
|------------|---------|
| **PyYAML** | Configuration files (prompts, schemas, mappings) |
| **jsonschema** | LLM response validation against JSON Schema |
| **datasketch** | MinHash LSH for near-duplicate description detection |
| **pandera** | DataFrame schema validation at step boundaries |

### Analysis & Visualization

| Technology | Purpose |
|------------|---------|
| **Jupyter** | Interactive analysis notebooks |
| **matplotlib** | Chart rendering (35 charts across 9 notebooks) |
| **seaborn** | Statistical visualization theming |

### Development

| Technology | Purpose |
|------------|---------|
| **pytest** | Test framework (920+ tests across all components) |
| **ruff** | Fast Python linter (line-length 100, E/F/W/I rules) |
| **mypy** | Static type checking (strict for key modules) |
| **tqdm** | Progress bars for LLM extraction |

---

## Key Design Decisions

### Two-Tier Extraction
- **Tier 1 (Regex):** 8 fields extracted deterministically via regex patterns — zero API cost, instant, reproducible
- **Tier 2 (LLM):** 7 fields requiring semantic understanding — extracted via DeepSeek V3 API
- **Why:** Avoids wasting LLM calls on fields that regex handles reliably (salary, contract type, experience years)

### Description Grouping
- Identical/near-identical descriptions are grouped (SHA-256 hash + MinHash LSH at 95% Jaccard similarity)
- Only one representative per group is sent to the LLM; results are fanned out to all members
- **Saves ~20% of LLM API calls** (~4,192 duplicate descriptions across 22,500 postings)

### Hallucination Prevention
- The extraction prompt explicitly forbids skill inference: *"ONLY extract a skill if its name appears literally in the description"*
- A post-extraction skill verifier cross-checks 100% of extracted skills against the description text
- Skills not found in the description are **removed** (not just flagged)

### NA Convention
- **In-memory:** Python `None` (never the string `"NA"`)
- **On disk (CSV):** `"NA"` string
- **Boundary:** `read_csv_safe()` converts `"NA"` → `None` on read; `write_csv_safe()` converts `None` → `"NA"` on write
- **Why:** Prevents pandas from silently converting legitimate data to NaN

### Crash Safety
- SQLite checkpoint tracks per-row stage progress (atomic commits)
- JSON progress file tracks per-step completion with timestamps
- Pipeline resumes automatically from last completed step/row on restart

### German-Specific Handling
- **Salary:** `50.000` = 50,000 EUR (German thousands separator)
- **Titles:** 80+ gender suffix patterns (`(m/w/d)`, `(gn)`, `:in`, `*in`) stripped
- **Locations:** German city/state parsing, state abbreviations (BY, BW, NRW)
- **Languages:** Bilingual prompt handles German + English job descriptions
- **Benefits:** German-specific keywords (Urlaub, Gleitzeit, Bahncard, etc.)

---

## Output Schema (24 Columns)

The final `data/cleaning/cleaned_jobs.csv` contains:

| Column | Type | Description |
|--------|------|-------------|
| `row_id` | string | Unique identifier (SHA-256 of job_url) |
| `job_url` | string | URL to original posting |
| `date_posted` | string | ISO date (YYYY-MM-DD) |
| `company_name` | string | Hiring company |
| `city` | string | Parsed German city |
| `state` | string | German federal state |
| `title` | string | Original job title |
| `title_cleaned` | string | Normalized title (gender suffixes removed, EN translation) |
| `job_family` | enum | One of 52 canonical IT role categories |
| `seniority_from_title` | string | Junior / Mid / Senior / Lead |
| `contract_type` | string | Full-time / Part-time / Contract / Freelance |
| `work_modality` | string | Remote / Hybrid / On-site |
| `salary_min` | float | Annual salary floor (EUR) |
| `salary_max` | float | Annual salary ceiling (EUR) |
| `experience_years` | int | Required years of experience |
| `education_level` | string | Highest degree required |
| `technical_skills` | JSON[] | Required technologies (e.g., ["Python", "Docker"]) |
| `soft_skills` | JSON[] | Interpersonal skills |
| `nice_to_have_skills` | JSON[] | Optional technologies |
| `benefits` | JSON[] | Tangible perks offered |
| `tasks` | JSON[] | Main responsibilities (max configurable, default 7 via `settings.yaml`) |
| `languages` | JSON[] | Language requirements with CEFR levels |
| `benefit_categories` | JSON[] | Categorized benefits (11 categories) |
| `soft_skill_categories` | JSON[] | Categorized soft skills (8 categories) |
| `description_quality` | string | "clean" or "concatenated" |
| `site` | string | "indeed" or "linkedin" |
| `validation_flags` | JSON[] | Quality flags from validation |
| `description` | string | Full job description text |

---

## Analysis Notebooks

| # | Notebook | Focus | Charts |
|---|----------|-------|--------|
| 01 | Market Overview | Dataset shape, job families, top companies, posting trends | 4 |
| 02 | Skills Demand | Top skills, skill×job family heatmap, co-occurrence, req vs nice-to-have | 4 |
| 03 | Salary Analysis | Distribution, by job family, by seniority, by city, experience correlation | 5 |
| 04 | Location Insights | Jobs by state/city, city×role heatmap, remote by state | 4 |
| 05 | Remote Work | Modality split, remote by role, by seniority, language correlation | 4 |
| 06 | Seniority & Experience | Seniority distribution, experience by role, junior cities, salary premium | 4 |
| 07 | Benefits Landscape | Top categories, benefits×role heatmap, vacation prevalence, size correlation | 4 |
| 08 | Language Requirements | German % by role, CEFR levels, English-only roles, language×modality | 4 |
| 09 | Job Seeker Guide | 4 personas (Junior/Senior/Remote/Career-Changer), skills roadmap, cities, salary | 2 + tables |

**Total:** 35 charts across 9 notebooks.

---

## License

This project was developed as part of TechLabs Data Science coursework.
