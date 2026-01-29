# German IT Job Scraper (Bilingual)

Scrapes IT jobs from **Indeed** and **LinkedIn** across German cities, using both **English** and **German** search terms (e.g. "software engineer" and "Softwareentwickler"). Results are deduplicated and saved as CSV and JSON.

---

## Requirements

- **Python 3.12** it must have to be the exact version, because we use some specific libraries that are not compatible with other versions.
- Dependencies: `numpy==1.26.4`, `python-jobspy`, `pandas`

Install (first cell in the notebook):

```bash
pip uninstall numpy pandas jobspy -y
pip install numpy==1.26.4
pip install python-jobspy
```

---

## Input (Configuration)

Configured via `ScraperConfig` in the notebook. Main options:

| Parameter | Description | Default |
|-----------|-------------|--------|
| `job_boards` | Sources to scrape | `["indeed", "linkedin"]` |
| `cities` | German cities | 33 cities (Berlin, Munich, Hamburg, …) |
| `keywords` | Job search terms (EN + DE) | IT roles (engineer, developer, DevOps, etc.) |
| `results_per_search` | Max results per search | 150 |
| `hours_old` | Job age limit (hours) | 720×4 (~120 days) |
| `remove_duplicates` | Deduplicate by `job_url` | `True` |
| `output_dir` | Output folder | `job_data` |

---

## How it works

1. **Setup**  
   A `ScraperConfig` defines job boards, cities, keywords, and limits. The scraper creates the output directory and a timestamped log file (console + file, with colored/emoji prefixes for level).

2. **Board loop**  
   For each job board (e.g. Indeed, then LinkedIn), the scraper runs all city × keyword combinations. Total number of searches = `len(job_boards) × len(cities) × len(keywords)`.

3. **Single search**  
   Each combination (e.g. "software engineer" in Berlin on Indeed) is one search. The code calls the **python-jobspy** `scrape_jobs()` function with:
   - `site_name`, `search_term`, `location`, `results_wanted`, `job_type`, `distance`, `hours_old`
   - Board-specific options (e.g. `country_indeed` for Indeed, `linkedin_fetch_description` for LinkedIn).

4. **Retries and rate limiting**  
   If a search fails, it is retried up to `max_retries` times with `retry_delay` between attempts. After each search, the script sleeps (e.g. 3s for Indeed, 6s for LinkedIn) to avoid being blocked.

5. **Accumulation and deduplication**  
   Results are collected per board as a list of DataFrames. After all searches for a board finish, the DataFrames are concatenated. If `remove_duplicates` is `True`, rows with the same `job_url` are dropped (first occurrence kept), so the same listing found under different keywords/cities is stored once.

6. **Saving**  
   For each board, the deduplicated table is written as:
   - **CSV** (e.g. `Raw_Jobs_INDEED_2026-01-29_12-00-00.csv`) and  
   - **JSON** (same data, `orient='records'`).  
   A **Metadata** JSON file is written once at the end with the config (boards, cities, keywords), run stats (success/fail counts, job counts per board), and timestamps. In Colab, `run(download_in_colab=True)` also triggers a browser download of the output files.

7. **Summary**  
   The log prints a short summary: duration, successful/failed searches per board, raw vs unique job counts, and paths to all saved files.

---

## Output

All files are written to **`job_data/`** (or your `output_dir`).

| File pattern | Description |
|--------------|-------------|
| `Raw_Jobs_{BOARD}_{timestamp}.csv` | Jobs per board (e.g. INDEED, LINKEDIN) |
| `Raw_Jobs_{BOARD}_{timestamp}.json` | Same data in JSON (records) |
| `Log_{timestamp}.log` | Run log (searches, counts, errors) |
| `Metadata_{timestamp}.json` | Run config + stats (boards, cities, keywords, totals) |

**CSV/JSON columns (examples):** `id`, `site`, `job_url`, `title`, `company`, `location`, `date_posted`, `job_type`, `description`, `company_*`, etc.

---

## How to Use

1. **Open** `JobScraper.ipynb` in Jupyter or VS Code.
2. **Run Cell 0** to install/upgrade dependencies (Python 3.12 recommended).
3. **Run Cell 1** to load the scraper and execute the default run:
   - Scrapes Indeed and LinkedIn for all configured cities × keywords (EN + DE).
   - Saves CSV + JSON per board, plus log and metadata.
   - In Google Colab, triggers download of the output files.

**Quick test (fewer cities/keywords):** In Cell 1, use the commented “EXAMPLE 1” block: set `job_boards=["indeed"]`, 2 cities, 4 keywords, then run `BilingualJobScraper(test_config).run(download_in_colab=True)`.

**Full run:** Use the default “EXAMPLE 2” at the bottom of Cell 1: `BilingualJobScraper(ScraperConfig()).run(download_in_colab=True)`.

---

## Notes

- **Rate limiting:** Delays between requests (e.g. 3s Indeed, 6s LinkedIn) to reduce blocking.
- **Retries:** Failed searches are retried up to `max_retries` (default 3).
- **Colab:** `download_in_colab=True` uses `google.colab.files.download` when available; safe to keep `True` when not in Colab.
