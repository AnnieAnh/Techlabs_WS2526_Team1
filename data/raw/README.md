# Raw Data

> **Status:** Data collection complete (read-only)
> **Location:** `data/raw/`

---

## Purpose

This directory holds the **raw job postings** collected from Indeed and LinkedIn. These CSV files are the starting point for the entire pipeline. Scraping was performed externally (via JobSpy and manual collection), and the results were placed directly here.

---

## Files

```
data/raw/
├── Raw_Jobs_INDEED.csv       10,851 rows  |  49.1 MB  |  34 columns
├── Raw_Jobs_LINKEDIN_1.csv    7,075 rows  |  25.2 MB  |  22 columns
└── Raw_Jobs_LINKEDIN_2.csv    4,600 rows  |  18.6 MB  |  13 columns
```

**Total:** ~22,500 raw job postings across 3 source files.

---

## Data Sources

### Indeed (`Raw_Jobs_INDEED.csv`)

- **Rows:** 10,851
- **Date format:** ISO (`YYYY-MM-DD`)
- **Key columns (34 total):** `id`, `job_url`, `title`, `company`, `location`, `description`, `date_posted`, `job_type`, `is_remote`, `min_amount`, `max_amount`, `currency`
- **Data quality:** Core fields 96-100% populated; extended metadata (salary, company info) mostly sparse

### LinkedIn #1 (`Raw_Jobs_LINKEDIN_1.csv`)

- **Rows:** 7,075
- **Date format:** Relative (`"2 weeks ago"`, `"3 days ago"`) + ISO timestamp (`scraped_at`)
- **Key columns (22 total):** `title`, `company`, `location`, `description`, `job_url`, `posted_date`, `job_type`, `seniority_level`, `required_skills`
- **Data quality:** Core fields 100% populated; job_type 100% populated (86% Full-time)

### LinkedIn #2 (`Raw_Jobs_LINKEDIN_2.csv`)

- **Rows:** 4,600
- **Date format:** Relative (`"3 weeks ago"`, `"1 week ago"`)
- **Key columns (13 total):** `title`, `company`, `location_job`, `description`, `link`, `time`, `search_keyword`, `search_location`
- **Data quality:** Core fields 100% populated; metadata columns (`level`, `industry`, `type`) mostly empty

---

## Column Mapping Across Sources

The ingestion component normalizes these different column names to a unified schema:

| Concept | Indeed | LinkedIn #1 | LinkedIn #2 |
|---------|--------|-------------|-------------|
| Job URL | `job_url` | `job_url` | `link` |
| Title | `title` | `title` | `title` |
| Company | `company` | `company` | `company` |
| Location | `location` | `location` | `location_job` |
| Description | `description` | `description` | `description` |
| Date Posted | `date_posted` (ISO) | `posted_date` (relative) | `time` (relative) |

---

## Data Characteristics

- **Location format:** `"City, State, Country"` (e.g., `"Stuttgart, BW, DE"` or `"Berlin, Berlin, Germany"`)
- **Job type values vary:** Indeed uses `"fulltime"`, LinkedIn uses `"Full-time"`
- **Descriptions:** Long-form text, mix of German and English, some with HTML remnants
- **Duplicates:** Raw files may contain cross-source duplicates (handled downstream by dedup stage)

---

## Rules

1. **Read-only** — No pipeline step ever modifies files in `data/raw/`
2. **UTF-8 encoding** — All files are UTF-8 encoded
3. **No deduplication here** — Dedup happens in the extraction pipeline (Step 3)

---

## Data Flow (Next Step)

```
data/raw/*.csv
        |
        v
src/ingestion/pipeline.py  (schema normalization + date parsing)
        |
        v
data/ingestion/combined_jobs.csv  (~22,526 rows)
```

See [Ingestion Documentation](../../src/ingestion/README.md) for the next pipeline stage.
