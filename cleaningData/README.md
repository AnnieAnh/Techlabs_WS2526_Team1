# Job Listings Data Processing Pipeline

A Python script that combines and cleans raw job listings from Indeed and LinkedIn into a unified dataset with a consistent schema.

## What It Does

The pipeline:

1. **Loads** raw CSV files from Indeed and LinkedIn (multiple sources)
2. **Normalizes** column names and values to a common schema
3. **Deduplicates** records based on `job_url`, `title`, and `company_name`
4. **Parses dates** from various formats (relative strings like "2 days ago", Unix timestamps, ISO dates) into `YYYY-MM-DD`
5. **Fills missing values**: `company_name` → `"na"`, `date_posted` → oldest date in the dataset
6. **Merges** all sources into a single output file
7. **Logs** processing statistics to a JSON file for auditability

## Directory Structure

```
cleaningData/
├── process_job_files.py    # Main script
├── README.md               # This file
├── RawData/                # Input files (required)
│   ├── Raw_Jobs_INDEED.csv
│   ├── Raw_Jobs_LINKEDIN_1.csv
│   └── Raw_Jobs_LINKEDIN_2.csv
└── Processed/              # Output (created automatically)
    ├── combined_jobs.csv   # Final merged dataset
    └── processing_log.json # Processing statistics
```

## Requirements

- **Python** 3.10 or higher
- **pandas** – data manipulation
- **numpy** – numerical operations

### Install Dependencies

```bash
pip install pandas numpy
```

Or with a requirements file:

```bash
pip install -r requirements.txt
```

## How to Run

From the project root:

```bash
python cleaningData/process_job_files.py
```

Or from inside the `cleaningData` directory:

```bash
cd cleaningData
python process_job_files.py
```

The script expects the raw CSV files to exist in `cleaningData/RawData/`. If any file is missing, it will raise a `FileNotFoundError`.

## Output Schema

The output file `combined_jobs.csv` has these columns:

| Column        | Description                                                       |
|---------------|-------------------------------------------------------------------|
| `title`       | Job title                                                         |
| `site`        | Source: `indeed` or `linkedin`                                    |
| `job_url`     | URL to the job posting                                            |
| `company_name`| Company name (missing values filled with `"na"`)                  |
| `location`    | Job location                                                      |
| `date_posted` | Posted date (YYYY-MM-DD; missing values filled with oldest date) |
| `description` | Job description text                                             |

## Processing Log

`processing_log.json` contains:

- **run_timestamp** – When the pipeline ran
- **processing_log** – Per-source stats (rows before/after, duplicates removed)
- **summary** – Total deduplication steps, final row count, output filename

Example:

```json
{
  "run_timestamp": "2026-02-15T15:01:32.123456",
  "processing_log": [
    {
      "source": "Indeed",
      "rows_before": 10851,
      "rows_after": 10851,
      "duplicates_removed": 0,
      "removal_percentage": 0,
      "file": "Raw_Jobs_INDEED.csv"
    }
  ],
  "summary": {
    "total_deduplication_steps": 4,
    "total_duplicates_removed_across_all_steps": 0,
    "final_output_rows": 22526,
    "output_file": "combined_jobs.csv"
  }
}
```

## Input File Expectations

| Source        | File                  | Key column mappings                          |
|---------------|-----------------------|----------------------------------------------|
| Indeed        | Raw_Jobs_INDEED.csv   | `company` → `company_name`, `date_posted`    |
| LinkedIn #1   | Raw_Jobs_LINKEDIN_1.csv | `company` → `company_name`, `posted_date` → `date_posted` |
| LinkedIn #2   | Raw_Jobs_LINKEDIN_2.csv | `link` → `job_url`, `location_job` → `location`, `time` → `date_posted` |

## Date Parsing

The script normalizes various date formats:

- **ISO**: `2026-01-28`, `2026-01-28 10:30:00`
- **Unix**: seconds or milliseconds
- **Relative**: `Today`, `Yesterday`, `2 days ago`, `1 week ago`, `3 months ago`

## Missing Value Handling

Before saving, the pipeline fills missing values:

| Column        | Fill value                          |
|---------------|-------------------------------------|
| `company_name`| `"na"`                              |
| `date_posted` | Oldest date in the dataset (min)    |
