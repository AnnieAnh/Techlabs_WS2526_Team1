# Cleaning Component

> **Status:** Complete and tested (75+ tests)
> **Location:** `src/cleaning/`
> **Input:** `data/extraction/extracted/enriched_combined.csv` (LLM-enriched)
> **Output:** `data/cleaning/cleaned_jobs.csv` (analysis-ready)

---

## Purpose

The cleaning component transforms LLM-enriched job postings into a **final, analysis-ready dataset**. It runs 11 cleaning steps (fix data quality without adding columns) followed by 5 enrichment steps (derive new columns), then validates 9 invariants before writing the final CSV. It can run standalone or as Step 7-8 of the orchestrated pipeline. When run via `orchestrate.py`, skill normalization steps (C++ fix, casing, re-verification) are skipped — they were already done in the validate step. The `standalone` gate ensures they only run when cleaning is invoked directly.

---

## Folder Structure

```
src/cleaning/
├── __init__.py
├── pipeline.py               # Main orchestrator: clean(), enrich(), clean_enriched()
├── constants.py              # COLUMN_ORDER, list/string column definitions
├── missing_values.py         # NA standardization + numeric column fixing
├── location_cleaner.py       # City name normalization (München, Köln, Frankfurt)
├── categorical_remapper.py   # Job family, contract type, seniority remapping
├── skill_normalizer.py       # C++ fix, skill casing, post-clean re-verification
├── soft_skill_normalizer.py  # Soft skill category mapping
├── benefit_categorizer.py    # Benefit keyword → category mapping
├── quality_flagger.py        # Description quality flags + skill frequency stats
├── output_formatter.py       # Column dropping, reordering, invariant assertions
├── validation_fixer.py       # Validation flags format conversion (repr→JSON)
│
└── config/
    ├── city_name_map.yaml        # City alias → canonical German name
    ├── job_family_remap.yaml     # LLM output → canonical job family + contract/seniority
    ├── benefit_categories.yaml   # Benefit keyword → 11 categories
    └── soft_skill_categories.yaml # Soft skill keyword → 8 categories

tests/cleaning/
└── test_post_clean.py        # 75+ test cases

data/cleaning/
└── cleaned_jobs.csv          # Final pipeline output
```

---

## Pipeline Flow

```
Input: enriched CSV from extraction pipeline
│
▼
┌──────────────────────────────────────────────────────────────────────┐
│                     CLEAN STAGE (11 steps)                           │
│  Fix data quality — no new columns created                           │
│                                                                      │
│  1. Safety-net Germany filter (drop non-German rows)                 │
│  2. Standardize missing values (None/NaN/"" → consistent None)       │
│  3. Fix numeric columns (salary, experience → proper types)          │
│  4. Normalize city names (Munich→München, Cologne→Köln)              │
│  5. Remap categoricals (job_family, contract_type, seniority)        │
│  6. Fix validation flags format (Python repr → JSON)                 │
│  7. Fix company name NA ("na" string → None)                         │
│  8. Fix residual gender markers in titles                            │
│  9. Fix C++ inference (hallucination correction)                     │
│ 10. Normalize skill casing (most-frequent variant wins)              │
│ 11. Normalize company casing (most-frequent variant wins)            │
└──────────────────────────────────────────────────────────────────────┘
│
▼
┌──────────────────────────────────────────────────────────────────────┐
│                    ENRICH STAGE (5 steps)                             │
│  Derive new columns from existing data                               │
│                                                                      │
│  1. Compute benefit categories (→ benefit_categories column)         │
│  2. Normalize soft skills (→ soft_skill_categories column)           │
│  3. Flag description quality (→ description_quality column)          │
│  4. Re-verify skills post-clean (update validation_flags)            │
│  5. Compute skill frequencies (→ frequency report)                   │
└──────────────────────────────────────────────────────────────────────┘
│
▼
┌──────────────────────────────────────────────────────────────────────┐
│                   FINALIZE & VALIDATE                                 │
│                                                                      │
│  1. Drop internal columns (country, location, source_file)           │
│  2. Drop rows with empty job_family (parse failures)                 │
│  3. Reorder columns → COLUMN_ORDER (29 columns)                     │
│  4. Save .debug.csv                                                  │
│  5. Assert 9 invariants                                              │
│  6. On success: rename .debug.csv → final output                     │
└──────────────────────────────────────────────────────────────────────┘
│
▼
Output: data/cleaning/cleaned_jobs.csv
```

---

## Clean Stage — Step-by-Step Detail

### Step 1: Safety-Net Germany Filter
**Module:** `pipeline.py`
**Logic:** Drop any row where `country != "Germany"`. This is a safety net — the extraction pipeline already filters non-German rows, but this catches edge cases when feeding older CSVs directly to cleaning.

### Step 2: Standardize Missing Values
**Module:** `missing_values.py` → `standardize_missing_values()`
**Logic:**
- **List columns** (`technical_skills`, `soft_skills`, `nice_to_have_skills`, `benefits`, `tasks`): Empty/None/NaN → `"[]"` (valid JSON array string)
- **String extracted columns** (`job_family`, `seniority_from_title`, `contract_type`, `work_modality`): Empty/None/NaN → Python `None`
- Validates that list columns parse as valid JSON arrays

### Step 3: Fix Numeric Columns
**Module:** `missing_values.py` → `fix_numeric_columns()`
**Logic:**
- `salary_min`, `salary_max`: Parse to float, handle German format (`50.000` → 50000), remove outliers outside [10,000 – 300,000] EUR
- `experience_years`: Parse to int
- Non-numeric values → `None`

### Step 4: Normalize City Names
**Module:** `location_cleaner.py` → `normalize_city_names()`
**Config:** `city_name_map.yaml` (13 aliases)
**Logic:**
- Maps English/variant names to canonical German forms: `Munich` → `München`, `Cologne` → `Köln`
- **Frankfurt special case:** `Frankfurt` in Hesse → `Frankfurt am Main`, in Brandenburg → `Frankfurt (Oder)`

### Step 5: Remap Categoricals
**Module:** `categorical_remapper.py` → `remap_categoricals()`
**Config:** `job_family_remap.yaml`
**Logic:**
- `job_family`: 38 LLM variants → canonical forms (e.g., `Web Developer` → `Fullstack Developer`)
- `contract_type`: `Permanent` → `Full-time`
- `seniority_from_title`: `Entry Level` → `Junior`

### Step 6: Fix Validation Flags Format
**Module:** `validation_fixer.py` → `fix_validation_flags()`
**Logic:** Converts `validation_flags` from Python repr strings to valid JSON. Tries `json.loads()` first, falls back to `ast.literal_eval()`, defaults to `"[]"`.

### Step 7: Fix Company Name NA
**Module:** `categorical_remapper.py` → `fix_company_name_na()`
**Logic:** Replaces `company_name` values where `.lower().strip() == "na"` with Python `None`.

### Step 8: Fix Residual Gender Markers
**Module:** `categorical_remapper.py` → `fix_residual_gender_markers()`
**Logic:** Regex removes `(gn)`, `(m,w,d)`, `(m,f,d)`, `(all genders)` patterns from `title_cleaned` that weren't caught by the extractors' title normalizer.

### Step 9: Fix C++ Inference
**Module:** `skill_normalizer.py` → `fix_cpp_inference()`
**Logic (decision tree):**
```
Description contains "c++"?
├── YES → Keep C++ in skills (genuine)
└── NO
    ├── Description contains bare "C" (word boundary)?
    │   └── YES → Replace "C++" with "C" in skills
    └── NO → Remove "C++" from skills (hallucination)
```
Applied to `technical_skills` and `nice_to_have_skills`.

### Step 10: Normalize Skill Casing
**Module:** `skill_normalizer.py` → `normalize_skill_casing()`
**Logic:** Groups all skills by lowercase key. The most-frequent casing variant wins across the entire dataset. Example: if 80% of rows have `"GIT"` and 20% have `"git"`, all become `"GIT"`.

### Step 11: Normalize Company Casing
**Module:** `categorical_remapper.py` → `normalize_company_casing()`
**Logic:** Same as skill casing — most-frequent variant of each company name wins.

---

## Enrich Stage — Step-by-Step Detail

### Step 1: Compute Benefit Categories
**Module:** `benefit_categorizer.py` → `benefit_category_set()`
**Config:** `benefit_categories.yaml`
**Logic:** Maps extracted benefit strings to 11 categories via case-insensitive keyword matching:

| Category | Keywords (examples) |
|----------|-------------------|
| `time_off` | urlaub, vacation, holiday, pto, days off |
| `flexible_hours` | flexible, gleitzeit, flexitime |
| `remote_work` | remote, homeoffice, telearbeit, mobiles arbeiten |
| `retirement` | rente, pension, altersvorsorge, bav |
| `health` | health, kranken, wellness, gym, gesundheit |
| `mobility` | bahncard, jobticket, dienstwagen, company car |
| `education` | weiterbildung, schulung, training, certification |
| `compensation` | bonus, prämie, provision, aktien, stock, esop |
| `food` | kantine, canteen, lunch, essenszuschuss |
| `social` | team event, offsite, firmenfeier, team building |
| `perks` | laptop, equipment, handy, parkplatz |

Multiple categories can match per benefit. Unmatched benefits → `other`.

### Step 2: Normalize Soft Skills
**Module:** `soft_skill_normalizer.py` → `normalize_soft_skills()`
**Config:** `soft_skill_categories.yaml`
**Logic:**
1. Filter out language-skill entries (e.g., "Englischkenntnisse")
2. Map remaining skills to 8 categories: Communication, Teamwork, Problem Solving, Initiative, Structured Work, Leadership, Adaptability, Customer Focus

### Step 3: Flag Description Quality
**Module:** `quality_flagger.py` → `flag_description_quality()`
**Logic:** Counts lowercase→uppercase letter transitions (camelCase patterns). >10 transitions → `"concatenated"` (likely HTML-strip artifacts); otherwise → `"clean"`.

### Step 4: Re-Verify Skills Post-Clean
**Module:** `skill_normalizer.py` → `re_verify_skills_post_clean()`
**Logic:** After skill casing normalization, re-checks all skills against description text. Replaces stale `skill_not_in_description` flags with fresh results. Idempotent — skips already-flagged skills.

### Step 5: Compute Skill Frequencies
**Module:** `quality_flagger.py` → `compute_skill_frequencies()`
**Logic:** Counts skill occurrences across all rows, returns top-N with percentages. Used for frequency reports, not added as a column.

---

## Invariant Assertions (9 Checks)

Before writing the final CSV, the pipeline asserts:

| # | Check | Fails If |
|---|-------|----------|
| 1 | No empty strings in categoricals | `job_family`, `contract_type`, `work_modality`, `seniority` contains `""` |
| 2 | Valid job_family enum | Value not in `job_families.yaml` (None is OK) |
| 3 | Valid JSON arrays | List columns don't parse as JSON arrays |
| 4 | Salary sanity | `salary_min > salary_max` |
| 5 | Unique row_ids | Duplicate `row_id` values |
| 6 | No "NA" strings in company | `company_name` is literal string `"na"/"NA"` |
| 7 | No gender markers in title | `(m/w/d)`, `(gn)`, etc. found in `title_cleaned` |
| 8 | No duplicate skills | Same skill appears twice (case-insensitive) in one row |
| 9 | No empty skill elements | `""` found inside a skill list |

**Failure handling:** `.debug.csv` is saved before assertions run. On failure, the debug file remains for investigation. On success, it's renamed to the final output path.

---

## Configuration Files

### `city_name_map.yaml`
```yaml
aliases:
  Munich: München
  Cologne: Köln
  Nuremberg: Nürnberg
  # ... 13 entries total
```

### `job_family_remap.yaml`
```yaml
remap:
  Web Developer: Fullstack Developer
  Machine Learning Engineer: ML Engineer
  Software Engineer: Software Developer
  # ... 38 entries

contract_type_remap:
  Permanent: Full-time

seniority_remap:
  Entry Level: Junior
```

### `benefit_categories.yaml`
```yaml
categories:
  time_off: [urlaub, vacation, holiday, pto, ...]
  flexible_hours: [flexible, gleitzeit, ...]
  # ... 11 categories with keyword lists
```

### `soft_skill_categories.yaml`
```yaml
categories:
  Communication: [kommunikation, presentation, ...]
  Teamwork: [team, zusammenarbeit, ...]
  # ... 8 categories with keyword lists
```

---

## How to Run

### Via Orchestrator (Recommended)
```bash
poetry run python orchestrate.py --only clean_enrich
# Runs Step 7 (clean+enrich) only
# Or run Step 7 + Step 8 together:
poetry run python orchestrate.py --from clean_enrich
```

### Standalone
```python
from cleaning.pipeline import clean_enriched
clean_enriched(
    "data/extraction/extracted/enriched_combined.csv",
    "data/cleaning/cleaned_jobs.csv"
)
```

### Tests
```bash
poetry run pytest tests/cleaning/ -v
# 75+ tests covering all cleaning steps, enrichment, and invariants
```

---

## Dual-Correction Design

Several cleaning steps duplicate corrections already applied in the extraction pipeline (`src/extraction/post_extraction.py`):
- Categorical remapping
- C++ inference fix
- Skill casing normalization

**This is intentional.** The cleaning steps are **safety nets** — they ensure correctness even when an older enriched CSV is fed directly to cleaning without going through the full extraction pipeline.

**`standalone` gate:** When run via `orchestrate.py`, the `standalone=False` default causes C++ inference fix, skill casing normalization, and post-clean skill re-verification to be skipped (already done in the validate step). When run standalone via `clean_enriched()`, `standalone=True` enables all three as safety nets.

---

## Output Schema (29 columns)

| Column | Type | Description |
|--------|------|-------------|
| `row_id` | string | Unique row identifier (SHA-256 of job_url) |
| `job_url` | string | URL to original posting |
| `date_posted` | string | ISO date |
| `company_name` | string | Hiring company |
| `city` | string | Parsed city name |
| `state` | string | German state |
| `title` | string | Original job title |
| `title_cleaned` | string | Normalized title |
| `job_family` | enum | One of 42 canonical roles |
| `seniority_from_title` | string | Junior/Mid/Senior/Lead |
| `contract_type` | string | Full-time/Part-time/Contract/Freelance |
| `work_modality` | string | Remote/Hybrid/On-site |
| `salary_min` | float | Annual salary floor (EUR) |
| `salary_max` | float | Annual salary ceiling (EUR) |
| `experience_years` | int | Required experience |
| `education_level` | string | Highest degree required |
| `technical_skills` | JSON[] | Required technologies |
| `soft_skills` | JSON[] | Interpersonal skills |
| `nice_to_have_skills` | JSON[] | Optional technologies |
| `benefits` | JSON[] | Tangible perks |
| `tasks` | JSON[] | Main responsibilities |
| `languages` | JSON[] | Language requirements with CEFR levels |
| `benefit_categories` | JSON[] | Categorized benefits |
| `soft_skill_categories` | JSON[] | Categorized soft skills |
| `description_quality` | string | "clean" or "concatenated" |
| `site` | string | "indeed" or "linkedin" |
| `validation_flags` | JSON[] | Quality flags from validation |
| `description` | string | Full job description text |

---

## Data Flow (Next Step)

```
data/cleaning/cleaned_jobs.csv  (~18,500 rows · 29 columns)
        |
        v
notebooks/01–11  (Jupyter notebooks for charts and insights)
```

See [Analysis Documentation](../analysis/README.md) for the next stage.
