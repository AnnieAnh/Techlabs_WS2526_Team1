# Analysis Component

> **Status:** Complete (11 notebooks + 5 helper modules)
> **Location:** `src/analysis/`
> **Input:** `data/cleaning/cleaned_jobs.csv` (analysis-ready)
> **Output:** `data/analysis/figures/*.png` (charts and visualizations)

---

## Purpose

The analysis component provides **11 Jupyter notebooks** that explore the cleaned German IT job market dataset from different angles — market overview, skills demand, salary distribution, geographic insights, remote work trends, seniority patterns, benefits landscape, language requirements, and a job seeker guide. All core logic lives in reusable helper modules under `src/analysis/`.

---

## Folder Structure

```
src/analysis/
├── __init__.py
├── utils.py              # Data loading + notebook_init() entry point
├── style.py              # Matplotlib/seaborn theming
├── charts.py             # 6 reusable chart functions
├── filters.py            # 14 data filtering helpers
└── compute.py            # 11 analytical computation functions

tests/analysis/
├── __init__.py
├── test_utils.py
└── test_filters.py

notebooks/                # At repo root
├── 01_market_overview.ipynb
├── 02_skills_demand.ipynb
├── 03_salary_analysis.ipynb
├── 04_location_insights.ipynb
├── 05_remote_work.ipynb
├── 06_seniority_experience.ipynb
├── 07_benefits_landscape.ipynb
├── 08_language_requirements.ipynb
├── 09_job_seeker_guide.ipynb
├── 10_role_deep_dives.ipynb
└── 11_education_soft_skills.ipynb

data/analysis/
└── figures/              # Auto-generated PNG exports
```

---

## Helper Modules

### `utils.py` — Data Loading

| Function | Purpose |
|----------|---------|
| `notebook_init()` | **Standard notebook entry point.** Sets up style, loads data via `read_csv_safe()`, creates figures directory, returns DataFrame. Call once at top of every notebook. |
| `load_enriched(path)` | Loads `data/cleaning/cleaned_jobs.csv` with NA→None conversion |
| `parse_json_col(df, col)` | Parses JSON list column strings into Python lists |
| `parse_flags(df)` | Expands `validation_flags` JSON into one-row-per-flag DataFrame |

### `style.py` — Theming

| Function | Purpose |
|----------|---------|
| `set_style()` | Applies seaborn "whitegrid" theme, sets figure size (12×6), DPI 150 |

### `charts.py` — 6 Reusable Chart Functions

| Function | Chart Type | Use Case |
|----------|-----------|----------|
| `horizontal_bar(series, title, top_n=20)` | Horizontal bar | Value counts (e.g., top skills, top cities) |
| `heatmap(data, title)` | Annotated heatmap | 2D pivot tables (e.g., skills × job families) |
| `time_series(series, title)` | Line chart | Date-indexed trends (e.g., postings over time) |
| `box_plot(df, x, y, title)` | Box plot | Numeric distributions by group (e.g., salary by role) |
| `stacked_bar(df, title)` | Stacked bar | Cross-tabulation (e.g., remote % by state) |
| `value_bar(index, values, title)` | Horizontal bar (pre-computed) | Pre-aggregated data (e.g., median salary by city) |

All functions accept `save_as` parameter for auto-saving to `data/analysis/figures/`.

### `filters.py` — 14 Data Filtering Helpers

| Function | Filter |
|----------|--------|
| `exclude_future_dates(df)` | Drops sentinel future-dated rows (2027-01-01) |
| `exclude_other_family(df)` | Drops rows where job_family is 'Other' |
| `filter_by_job_family(df, family)` | Rows matching job family (case-insensitive) |
| `filter_by_seniority(df, seniority)` | Rows matching seniority level |
| `filter_remote(df)` | Rows where `work_modality == "Remote"` |
| `filter_salary_known(df)` | Rows with both salary_min and salary_max populated |
| `salary_df(df)` | Salary subset with int columns + `salary_mid` derived field |
| `explode_json_col(df, col)` | Explodes JSON-list column into one row per element |
| `filter_by_modality(df, modality)` | Rows matching work modality |
| `filter_by_city(df, city)` | Rows matching city |
| `filter_by_state(df, state)` | Rows matching German federal state |
| `filter_description_quality(df, quality)` | Rows matching description quality label |
| `rows_with_skill(df, skill, col)` | Rows mentioning a specific skill |
| `parse_list_col(df, col)` | Parse JSON-list column to Python lists |

---

## Notebooks — Flow and Goals

### Notebook Initialization Pattern

Every notebook starts with:
```python
from analysis.utils import notebook_init
df = notebook_init()
```
This loads the data, sets the style, and creates the output directory.

---

### Notebook 01 — Market Overview

**Goal:** High-level dataset shape and hiring landscape.

```
cleaned_jobs.csv
│
├── Job Family Distribution    → horizontal_bar (top 20)    → 01_job_family.png
├── Postings Over Time         → time_series (date counts)  → 01_postings_time.png
├── Top Companies              → horizontal_bar (top 15)    → 01_companies.png
└── Source Split (Indeed/LI)   → pie chart                  → 01_source_split.png
```

**Key insights:** Which roles dominate the market, posting volume trends, which companies hire most, data source balance.

---

### Notebook 02 — Skills Demand

**Goal:** Technical skill requirements and co-occurrence patterns.

```
cleaned_jobs.csv
│
├── Top 20 Technical Skills    → horizontal_bar             → 02_top_skills.png
├── Skills × Job Family        → heatmap (10 families×15)   → 02_skill_family.png
├── Skill Co-occurrence        → heatmap (pair counts)      → 02_cooccurrence.png
└── Required vs Nice-to-Have   → bar chart (unique counts)  → 02_req_vs_nice.png
```

**Key insights:** Most-demanded technologies, which skills cluster together, which skills are required vs optional.

---

### Notebook 03 — Salary Analysis

**Goal:** Salary distributions by role, seniority, location, experience.

```
cleaned_jobs.csv  →  salary_df() (~10% have salary data)
│
├── Salary Distribution        → histogram (min/max)        → 03_salary_dist.png
├── Salary by Job Family       → box_plot                   → 03_salary_family.png
├── Salary by Seniority        → box_plot                   → 03_salary_seniority.png
├── Salary by City             → value_bar (median, EUR)    → 03_salary_city.png
└── Experience vs Salary       → scatter plot               → 03_exp_salary_scatter.png
```

**Key insights:** Salary ranges by role and level, highest-paying cities, experience-salary correlation.

---

### Notebook 04 — Location Insights

**Goal:** Geographic distribution across German states and cities.

```
cleaned_jobs.csv
│
├── Jobs by State              → horizontal_bar             → 04_state.png
├── Top 20 Cities              → horizontal_bar             → 04_cities.png
├── City × Job Family          → heatmap (10×10)            → 04_city_family.png
└── Remote vs On-site by State → stacked_bar (normalized)   → 04_remote_state.png
```

**Key insights:** IT hubs (Berlin, Munich, Hamburg), which states have most remote options, regional role specialization.

---

### Notebook 05 — Remote Work Trends

**Goal:** Work modality adoption across roles and seniority levels.

```
cleaned_jobs.csv
│
├── Work Modality Split        → horizontal_bar             → 05_modality.png
├── Remote by Job Family       → stacked_bar (normalized)   → 05_remote_family.png
├── Remote by Seniority        → stacked_bar                → 05_remote_seniority.png
└── Language × Remote          → stacked_bar                → 05_lang_modality.png
```

**Key insights:** Remote vs hybrid vs on-site share, which roles are most remote-friendly, whether German language requirements correlate with on-site work.

---

### Notebook 06 — Seniority & Experience

**Goal:** Career-level distribution and experience requirements.

```
cleaned_jobs.csv
│
├── Seniority Distribution     → horizontal_bar             → 06_seniority.png
├── Experience by Job Family   → box_plot                   → 06_exp_family.png
├── Junior Roles by City       → horizontal_bar (top 15)   → 06_junior_cities.png
└── Seniority vs Salary        → box_plot                   → 06_seniority_salary.png
```

**Key insights:** Junior/Mid/Senior balance, which roles require most experience, best cities for entry-level, seniority salary premium.

---

### Notebook 07 — Benefits Landscape

**Goal:** What benefits employers offer and which roles have the best packages.

```
cleaned_jobs.csv
│
├── Top Benefit Categories     → horizontal_bar             → 07_benefits.png
├── Benefits × Job Family      → heatmap (8×11)             → 07_benefit_family.png
├── Vacation Benefit by Role   → horizontal_bar             → 07_urlaub_family.png
└── Company Size × Benefits    → stacked_bar                → 07_size_benefits.png
```

**Key insights:** Most common benefit types, which roles get best perks, vacation prevalence by role, size-benefit correlation.

---

### Notebook 08 — Language Requirements

**Goal:** German/English proficiency expectations and CEFR level distribution.

```
cleaned_jobs.csv
│
├── German Required % by Role  → horizontal_bar (sorted)    → 08_german_family.png
├── CEFR Level Distribution    → horizontal_bar             → 08_cefr_levels.png
├── English-Only Roles         → horizontal_bar (top 15)    → 08_english_only.png
└── Language × Work Modality   → stacked_bar                → 08_lang_modality.png
```

**Key insights:** Which roles require German, CEFR level expectations, English-only opportunities, language-remote correlation.

---

### Notebook 09 — Job Seeker Guide

**Goal:** Actionable career guidance via 4 persona profiles.

**Personas:**
| Persona | Filter |
|---------|--------|
| Junior | `seniority == "Junior"` |
| Senior | `seniority in ["Senior", "Lead"]` |
| Remote-First | `work_modality == "Remote"` |
| Career-Changer | `job_family == "Other"` |

```
cleaned_jobs.csv  →  4 persona subsets
│
├── Persona Summary Table      → Count, market %, top role, median salary, remote %
├── Skills Roadmap by Persona  → 2×2 multi-panel horizontal bars  → 09_skills_by_persona.png
├── Best Cities per Persona    → 2×2 multi-panel horizontal bars  → 09_cities_by_persona.png
└── Salary Expectations Table  → Median, P25, P75 per persona
```

**Key insights:** Tailored skill recommendations, best cities, and salary expectations for each career profile.

---

## Visualization Summary

| Notebook | Charts | Chart Types Used |
|----------|--------|-----------------|
| 01 Market Overview | 4 | horizontal_bar, time_series, pie |
| 02 Skills Demand | 4 | horizontal_bar, heatmap (×2), bar |
| 03 Salary Analysis | 5 | histogram, box_plot (×2), value_bar, scatter |
| 04 Location Insights | 4 | horizontal_bar (×2), heatmap, stacked_bar |
| 05 Remote Work | 4 | horizontal_bar, stacked_bar (×3) |
| 06 Seniority | 4 | horizontal_bar (×2), box_plot (×2) |
| 07 Benefits | 4 | horizontal_bar (×2), heatmap, stacked_bar |
| 08 Language | 4 | horizontal_bar (×3), stacked_bar |
| 09 Job Seeker Guide | 2 (+ 2 tables) | Multi-panel horizontal bars |
| **Total** | **35 charts** | |

---

## How to Run

### Launch Notebooks
```bash
cd notebooks/
poetry run jupyter notebook
```
**Important:** Notebooks MUST be run from `notebooks/` directory so that `Path().resolve()` in `notebook_init()` resolves correctly.

### Cell Execution Order
1. Always run cell 1 (`notebook_init()`) first — loads data and sets up style
2. Then run any analysis cells in any order (they are independent)

### Prerequisites
- Pipeline must have completed through Step 8 (export) so that `data/cleaning/cleaned_jobs.csv` exists
- Required packages: `matplotlib`, `seaborn`, `jupyter`, `notebook` (all in Poetry dev deps)

### Tests
```bash
poetry run pytest tests/analysis/ -v
```

---

## Key Design Rules

1. **No core logic in notebooks** — All reusable functions live in `src/analysis/`. Notebooks only call helpers and configure chart parameters.
2. **Always use `notebook_init()`** — Ensures consistent data loading, style, and figure directory setup.
3. **Explode JSON columns early** — Most analyses use `explode_json_col()` to get one row per skill/benefit/language before aggregation.
4. **Pre-aggregate before charting** — Use `value_bar()` for already-computed statistics (medians, percentages), not raw data.
5. **Auto-save charts** — Pass `save_as="filename.png"` to chart functions for automatic PNG export.

---

## Data Flow

```
data/cleaning/cleaned_jobs.csv  (~18,500 rows · 29 columns)
        |
        v
notebooks/01–11  (Jupyter notebooks)
        |
        v
data/analysis/figures/*.png  (35+ charts across 11 notebooks)
```

This is the final stage of the pipeline. See [Cleaning Documentation](../cleaning/CLEANING.md) for the upstream component.
