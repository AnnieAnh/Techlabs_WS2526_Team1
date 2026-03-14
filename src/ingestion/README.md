# Ingestion Component

> **Status:** Complete and tested
> **Location:** `src/ingestion/`
> **Input:** `data/raw/*.csv` (3 raw CSV files)
> **Output:** `data/ingestion/combined_jobs.csv` (~22,526 rows)

---

## Purpose

The ingestion component is a **lightweight CSV normalization pipeline**. It reads three heterogeneous source files (Indeed + 2x LinkedIn), maps their different column names to a unified schema, normalizes dates to ISO format, and writes a single combined CSV. No deduplication, no LLM processing — just schema unification and date parsing.

---

## Folder Structure

```
src/ingestion/
├── __init__.py
├── config/
│   └── settings.yaml             # Source definitions, column mappings, reference date
├── pipeline.py               # Main orchestrator: run_pipeline()
├── loader.py                 # Per-source CSV loading + normalization
└── date_parser.py            # Date format conversion (relative → ISO)

tests/ingestion/
└── test_ingestion.py         # 6 test cases

data/ingestion/
└── combined_jobs.csv         # Pipeline output
```

---

## Pipeline Flow

```
Raw CSVs (3 files, different schemas)
│
│  Raw_Jobs_INDEED.csv        10,851 rows  ·  34 columns
│  Raw_Jobs_LINKEDIN_1.csv     7,075 rows  ·  22 columns
│  Raw_Jobs_LINKEDIN_2.csv     4,600 rows  ·  13 columns
│
▼
┌─────────────────────────────────────────────────┐
│  load_and_normalize_source() [per source file]  │
│  • Read CSV (keep_default_na=False)             │
│  • Map columns → unified TARGET_COLUMNS         │
│  • Normalize whitespace                         │
│  • Add site label ("indeed" / "linkedin")       │
└─────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────┐
│  pd.concat() — merge all sources                │
│  Result: 22,526 rows × 7 columns                │
└─────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────┐
│  normalize_date_posted()                        │
│  • ISO dates → pass through                     │
│  • Relative dates → calculate from pinned ref   │
│  • Unix timestamps → convert                    │
│  • Missing → None                               │
└─────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────┐
│  fill_missing_values()                          │
│  • Empty company_name → None                    │
└─────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────┐
│  Pandera schema validation                      │
│  • job_url: non-null, starts with "http"        │
│  • description: non-null, length ≥ 1            │
│  • site: must be "linkedin" or "indeed"         │
└─────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────┐
│  write_csv_safe()                               │
│  • None → "NA" at CSV boundary                  │
│  Output: data/ingestion/combined_jobs.csv       │
│  + processing_log.json                          │
└─────────────────────────────────────────────────┘
```

---

## How Each Step Works

### Step 1: Load and Normalize Sources (`loader.py`)

Each raw CSV has different column names. The `SourceConfig` dataclass (built from `settings.yaml`) defines the mapping per source:

| Unified Column | Indeed source | LinkedIn #1 source | LinkedIn #2 source |
|----------------|--------------|--------------------|--------------------|
| `title` | `title` | `title` | `title` |
| `job_url` | `job_url` | `job_url` | `link` |
| `company_name` | `company` | `company` | `company` |
| `location` | `location` | `location` | `location_job` |
| `date_posted` | `date_posted` | `posted_date` | `time` |
| `description` | `description` | `description` | `description` |

After column mapping, whitespace is stripped from all string columns and a `site` label is added.

### Step 2: Date Parsing (`date_parser.py`)

The `parse_date_to_exact()` function handles multiple date formats:

| Input Format | Example | Output |
|-------------|---------|--------|
| ISO date | `"2026-02-15"` | `"2026-02-15"` |
| ISO datetime | `"2026-02-15 10:30:00"` | `"2026-02-15"` |
| Relative string | `"2 weeks ago"` | Calculated from pinned date |
| Relative string | `"yesterday"` | Reference date − 1 day |
| Relative string | `"today"` / `"just posted"` | Reference date |
| Unix timestamp | `1708041600` | Converted to ISO |
| Missing/invalid | `""`, `None`, `"NA"` | `None` |

**Determinism:** All relative dates are calculated from a **pinned reference date** (`2026-02-15`, from `settings.yaml`), not the current system time. Re-running ingestion always produces identical ISO dates.

### Step 3: Missing Value Handling (`loader.py`)

- Empty/null `company_name` values are set to Python `None`
- All in-memory code uses `None`, never the string `"NA"`
- The `write_csv_safe()` function converts `None` → `"NA"` when writing to disk

### Step 4: Schema Validation

Pandera validates the output DataFrame:
- `job_url` must be non-null and start with `"http"`
- `description` must be non-null with length ≥ 1
- `site` must be `"linkedin"` or `"indeed"`
- Other fields (`title`, `company_name`, `location`, `date_posted`) are nullable

---

## Output Schema

The output CSV (`combined_jobs.csv`) has exactly 7 columns:

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `title` | string | yes | Job title as scraped |
| `site` | string | no | `"indeed"` or `"linkedin"` |
| `job_url` | string | no | URL to original posting |
| `company_name` | string | yes | Hiring company name |
| `location` | string | yes | Raw location string |
| `date_posted` | string | yes | ISO date (`YYYY-MM-DD`) |
| `description` | string | no | Full job description text |

---

## Configuration (`config/settings.yaml`)

```yaml
scrape_date: "2026-02-15"          # Pinned reference date for relative date parsing

paths:
  raw_data_dir: "data/raw"
  output_dir: "data/ingestion"

sources:
  - name: "Indeed"
    file: "Raw_Jobs_INDEED.csv"
    site_label: "indeed"
    column_mapping:
      title: "title"
      job_url: "job_url"
      company_name: "company"
      location: "location"
      date_posted: "date_posted"
      description: "description"

  - name: "LinkedIn #1"
    file: "Raw_Jobs_LINKEDIN_1.csv"
    site_label: "linkedin"
    column_mapping: { ... }           # Maps posted_date → date_posted, etc.

  - name: "LinkedIn #2"
    file: "Raw_Jobs_LINKEDIN_2.csv"
    site_label: "linkedin"
    column_mapping: { ... }           # Maps link → job_url, location_job → location, etc.
```

---

## How to Run

### Via Orchestrator (Recommended)
```bash
python orchestrate.py --only ingest
# or as part of full pipeline:
python orchestrate.py
```

### Via Make
```bash
make ingest
```

### Programmatically
```python
from ingestion.pipeline import run_pipeline
output_path = run_pipeline()  # Returns Path to combined_jobs.csv
```

### Tests
```bash
poetry run pytest tests/ingestion/ -v
# 6 test cases covering: company NA handling, date determinism, whitespace normalization
```

---

## Key Business Rules

1. **No deduplication** — Handled downstream in extraction Step 3
2. **No location filtering** — Handled downstream in extraction Step 2
3. **No LLM processing** — Pure schema normalization and CSV merging
4. **Deterministic dates** — Same input always produces same output (pinned reference date)
5. **NA boundary** — In-memory: `None` | On disk: `"NA"` string

---

## Data Flow (Next Step)

```
data/ingestion/combined_jobs.csv  (22,526 rows)
        |
        v
extraction pipeline  (8 steps: prepare → dedup → regex_extract → extract → validate → clean_enrich → export)
        |
        v
data/cleaning/cleaned_jobs.csv  (analysis-ready)
```

See [Extraction Documentation](../extraction/README.md) for the next pipeline stage.
