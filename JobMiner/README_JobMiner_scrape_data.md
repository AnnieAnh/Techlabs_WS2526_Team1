# LinkedIn Job Data Cleaning for JobMiner

This module contains two Python scripts used in the **JobMiner** project to scrape LinkedIn job postings and clean outdated results.

## Purpose

The goal of these scripts is to:

- scrape LinkedIn job postings for multiple IT-related roles across Germany
- collect structured job data in **JSON** and **CSV**
- remove outdated job postings older than **120 days**
- keep the dataset cleaner and more reliable for later analysis

The scraper collects job postings directly from LinkedIn and saves job details such as title, company, location, description, salary, work mode, seniority level, skills, and more. :contentReference[oaicite:0]{index=0}  
The cleaning script then filters out old jobs based on the `posted_date` field and updates the metadata accordingly. :contentReference[oaicite:1]{index=1}

---

## Files

### `JobMiner_Linkedin.py`
This script is responsible for scraping LinkedIn job postings.

Main features:
- searches multiple **job titles** in multiple **German cities**
- supports both **English and German keywords**
- extracts extended job information such as:
  - work mode
  - company industry
  - company size
  - seniority level
  - required skills
  - experience range
- filters out jobs older than the configured time range during scraping
- saves results into:
  - `.json`
  - `.csv`
  - `log file`
  - `metadata file` 

### `clean_old_jobs.py`
This script is used after scraping to clean existing output files.

Main features:
- converts LinkedIn date strings like `2 days ago`, `3 weeks ago`, `1 year ago` into numeric day values
- removes jobs older than **120 days**
- removes entries containing `"year"` in `posted_date`
- keeps CSV and JSON files consistent
- updates metadata with cleaning statistics 

---

## Why this was implemented

LinkedIn job postings may still include old listings such as jobs posted **1 year ago**.  
To make the dataset more relevant for job market analysis, the workflow was split into 2 parts:

1. **Scrape job data from LinkedIn**
2. **Clean outdated postings from the exported files**

This improves data quality and ensures that later analysis is based on more recent job postings only. 

---

## Requirements

Install Python 3.10+ and the required libraries before running the scripts.

### Python packages
```bash
pip install pandas beautifulsoup4 requests
```

Depending on the full project structure, you may also need additional internal project files such as:
- `base_scraper.py`
- `JobListing` class definition

`JobMiner_Linkedin.py` imports these components from the main `JobMiner` project structure, so the folder setup must remain correct.

## Project structure
Example structure:
```
Project-Group-1/
│
├── Techlabs_WS2526_Team1/
│   └── JobMiner/
│       ├── JobMiner_Linkedin.py
│       ├── clean_old_jobs.py
│       └── ...
```

## How to run
1. Run the LinkedIn scraper
```
python JobMiner_Linkedin.py
```
This will:
- scrape LinkedIn jobs
- filter jobs within the configured time range
- remove duplicates
- generate output files such as:
    - Raw_Jobs_LINKEDIN_<timestamp>.csv
    - Raw_Jobs_LINKEDIN_<timestamp>.json
    - Log_LinkedIn_<timestamp>.log
    - Metadata_<timestamp>.json

2. Run the cleaning script
```
python clean_old_jobs.py
```
This will:
- load the scraped CSV and JSON files
- remove jobs older than 120 days
- save cleaned versions of the files
- update metadata with cleaning results

## Output 
After running the scripts, you will obtain:
- a structured LinkedIn job dataset in CSV
- the same dataset in JSON
- a log file for tracking the scraping process
- a metadata file summarizing the run
- cleaned files containing only more recent job postings

## Notes

- The scraper is configured to work in a safer, more human-like way using delays and sequential requests. 
- The cleaning step is useful as an extra safety layer in case old jobs were not fully filtered during scraping.  
- File paths in the scripts may need to be adjusted depending on your local machine and project location.

## Summary

These scripts support the JobMiner pipeline by first collecting LinkedIn job postings and then cleaning outdated data.

This makes the final dataset more suitable for job market analysis and further data processing.