# Strategy A: Rules + Keywords ("No ML" Approach)

## Roadmap & Backlog

**Estimated Duration:** ~1 week
**Cost:** $0
**Compute:** CPU only, <10 min for 100k rows
**Accuracy Target:** ~70-75% overall

---

## Requirements

### Python Dependencies

```txt
# requirements.txt
pandas>=2.0
pandera>=0.18
openpyxl>=3.1          # Excel export for review sheets
pyyaml>=6.0            # Config files
tqdm>=4.65             # Progress bars
pytest>=7.4            # Testing
```

### Hardware

- Any machine with 4GB+ RAM
- No GPU needed
- ~2GB disk for data + outputs

### Input Data

- JSON file with Indeed job listings (JobSpy format)
- Fields required: `id`, `title`, `description`, `location`, `job_type`, `is_remote`

### Knowledge Assets (You Build These)

- Curated IT skills keyword list (~200-300 terms)
- German/English section header mappings
- Benefit category taxonomy

---

## Project Folder Structure

```
job-extraction/
│
├── config/
│   ├── settings.yaml              # Paths, thresholds, toggles
│   ├── skills_keywords.yaml       # Curated skills list (DE + EN)
│   ├── section_headers.yaml       # Header → canonical section mapping
│   └── benefit_categories.yaml    # Benefit keyword → category mapping
│
├── src/
│   ├── __init__.py
│   ├── dedup.py                   # Deduplication logic
│   ├── validate_input.py          # Input data quality checks (Pandera)
│   ├── section_splitter.py        # Description → sections
│   ├── extractors/
│   │   ├── __init__.py
│   │   ├── contract_type.py       # Regex-based
│   │   ├── work_modality.py       # Regex-based
│   │   ├── seniority.py           # Title + keyword parsing
│   │   ├── salary.py              # Regex-based
│   │   ├── experience.py          # Regex-based
│   │   ├── skills.py              # Keyword matching
│   │   ├── benefits.py            # Bullet parsing + categorization
│   │   └── languages.py           # Regex-based
│   ├── cross_validation.py        # Cross-field consistency checks
│   ├── quality_report.py          # Distribution analysis
│   └── review_sampler.py          # Stratified sampling for manual review
│
├── tests/
│   ├── test_section_splitter.py
│   ├── test_extractors.py         # Parametrized tests per extractor
│   └── test_cross_validation.py
│
├── data/                          # .gitignore this, track with DVC if needed
│   ├── raw/                       # Original JSON from JobSpy
│   ├── deduped/                   # After dedup
│   ├── extracted/                 # Final enriched CSV
│   └── review/                    # Review spreadsheets
│
├── notebooks/
│   ├── 01_data_exploration.ipynb  # Understand your data first
│   ├── 02_extraction_dev.ipynb    # Develop & test extractors
│   └── 03_quality_analysis.ipynb  # Analyze extraction results
│
├── run_pipeline.py                # Main orchestrator script
├── requirements.txt
└── README.md
```

---

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        STRATEGY A PIPELINE                          │
└─────────────────────────────────────────────────────────────────────┘

  ┌──────────┐     ┌──────────────┐     ┌──────────────────┐
  │ Raw JSON │────▶│ Deduplication │────▶│ Input Validation │
  │  ~100k   │     │  URL + Title  │     │   Pandera Schema │
  └──────────┘     │  + Fingerprint│     │   Quality Report │
                   └──────┬───────┘     └────────┬─────────┘
                          │  ~70-80k              │
                          ▼                       ▼
                   ┌──────────────────────────────────────┐
                   │          Section Splitting            │
                   │   Description → tasks | requirements  │
                   │                  benefits | about     │
                   └──────────────────┬───────────────────┘
                                      │
                   ┌──────────────────┼───────────────────┐
                   │                  │                    │
                   ▼                  ▼                    ▼
          ┌────────────────┐ ┌───────────────┐ ┌──────────────────┐
          │  Regex-Based   │ │ Keyword-Based │ │  Bullet Parsing  │
          │  Extractors    │ │   Extractor   │ │    Extractor     │
          │                │ │               │ │                  │
          │ • contract     │ │ • skills      │ │ • benefits       │
          │ • salary       │ │   (from       │ │   (from benefits │
          │ • experience   │ │   requirements│ │    section)      │
          │ • work_modal   │ │   section)    │ │                  │
          │ • languages    │ │               │ │                  │
          │ • seniority    │ │               │ │                  │
          └───────┬────────┘ └───────┬───────┘ └────────┬─────────┘
                  │                  │                   │
                  └──────────────────┼───────────────────┘
                                     │
                                     ▼
                   ┌──────────────────────────────────────┐
                   │       Cross-Field Validation          │
                   │  • Seniority vs Experience            │
                   │  • Salary range sanity                │
                   │  • Contract vs Seniority logic        │
                   │  • Skills count sanity                │
                   └──────────────────┬───────────────────┘
                                      │
                          ┌───────────┼───────────┐
                          ▼           ▼           ▼
                   ┌───────────┐ ┌────────┐ ┌──────────┐
                   │ Quality   │ │ Review │ │ Enriched │
                   │ Report    │ │ Sample │ │ CSV      │
                   │ (stdout)  │ │ (.xlsx)│ │ (final)  │
                   └───────────┘ └────────┘ └──────────┘
```

---

## Roadmap

```
Week 1
├── Day 1-2: Foundation
│   ├── ███████████ Project setup + data exploration
│   └── ███████████ Deduplication + input validation
│
├── Day 3-4: Core Extraction
│   ├── ███████████ Section splitter
│   ├── ███████████ Regex extractors (contract, salary, exp, modality, lang)
│   └── ███████████ Seniority extractor (title-based)
│
├── Day 5: Skills & Benefits
│   ├── ███████████ Skills keyword list curation
│   ├── ███████████ Keyword matcher
│   └── ███████████ Benefits bullet parser
│
├── Day 6: Validation & QA
│   ├── ███████████ Cross-field validation
│   ├── ███████████ Quality report
│   └── ███████████ Review sample generation
│
└── Day 7: Review & Polish
    ├── ███████████ Manual review of 200 rows
    ├── ███████████ Fix extraction bugs found in review
    └── ███████████ Final run + export
```

---

## Backlog

### Sprint 1: Foundation (Day 1-2)

| ID    | Task                              | Priority | Est   | Details |
|-------|-----------------------------------|----------|-------|---------|
| A-001 | Project scaffolding               | P0       | 1h    | Create folder structure, install deps, create `settings.yaml` |
| A-002 | Data exploration notebook          | P0       | 2h    | Load JSON, check field completeness, sample 30 descriptions manually, identify section header patterns, count German vs English |
| A-003 | Deduplication module               | P0       | 2h    | URL dedup → title+company+location dedup → description fingerprint dedup |
| A-004 | Input validation (Pandera)         | P1       | 2h    | Schema definition, null checks, description length distribution, quality report |
| A-005 | Config loader                      | P1       | 1h    | YAML config for paths, thresholds, keyword lists |

**Exit criteria:** Deduplicated dataset loaded, quality report printed, you know your actual row count and language split.

---

### Sprint 2: Section Splitting (Day 3)

| ID    | Task                              | Priority | Est   | Details |
|-------|-----------------------------------|----------|-------|---------|
| A-006 | Section header inventory           | P0       | 1h    | Sample 50 descriptions, catalog all header patterns (DE + EN), add to `section_headers.yaml` |
| A-007 | Section splitter module            | P0       | 2h    | Markdown header detection, bold-text detection, map to canonical sections (tasks, requirements, benefits, about, other) |
| A-008 | Section splitter tests             | P0       | 1h    | Test on 5+ real descriptions: vCluster (EN), STACKIT (DE), SAP (mixed), engelhardt (DE), edge cases |
| A-009 | Coverage check                     | P1       | 0.5h  | What % of descriptions successfully split into 2+ sections? Target: >70% |

**Exit criteria:** >70% of descriptions split into at least `tasks` + `requirements` sections. Remaining 30% fall back to full-text processing.

---

### Sprint 3: Regex Extractors (Day 3-4)

| ID    | Task                              | Priority | Est   | Details |
|-------|-----------------------------------|----------|-------|---------|
| A-010 | Contract type extractor            | P0       | 1h    | Structured header match → JobSpy fallback → free-text match. Return value + confidence + source |
| A-011 | Work modality extractor            | P0       | 1.5h  | Hybrid detection (mobiles Arbeiten + office), remote-first, LI-Hybrid tag. Priority: hybrid > remote > onsite |
| A-012 | Salary extractor                   | P0       | 1.5h  | Handle €80K-€110K, €50.000-€70.000, EUR formats. German decimal separators (dots for thousands). Fall back to JobSpy `min_amount`/`max_amount` |
| A-013 | Experience years extractor         | P1       | 1h    | Numeric patterns + German terms (mehrjährige=3+, einige Jahre=3-5, ein paar Jahre=2-3). Return raw match string |
| A-014 | Seniority extractor                | P0       | 1h    | Title parsing first (most reliable). Then description keywords: Berufserfahrene, Berufseinsteiger, etc. |
| A-015 | Language requirements extractor    | P1       | 1h    | German/English with level detection (fließend, gut, sicher, C1, B2). Distinguish required vs nice-to-have where possible |
| A-016 | Unit tests for all regex extractors| P0       | 2h    | Parametrized pytest: 5+ test cases per extractor covering DE, EN, mixed, edge cases, nulls |

**Exit criteria:** All 6 regex extractors pass tests. Run on 50-row sample, manually verify >80% accuracy per field.

---

### Sprint 4: Skills & Benefits (Day 5)

| ID    | Task                              | Priority | Est   | Details |
|-------|-----------------------------------|----------|-------|---------|
| A-017 | Curate skills keyword list         | P0       | 2h    | Build `skills_keywords.yaml`. Categories: languages, frameworks, cloud, tools, databases, concepts, methodologies. Target: 200-300 terms. Source: sample your own data + Stack Overflow survey + common job board taxonomies |
| A-018 | Skills keyword matcher             | P0       | 1.5h  | Word-boundary matching. Handle multi-word terms ("Spring Boot", "CI/CD"). Case-insensitive but preserve original casing in output. Run on requirements section first, full text as fallback |
| A-019 | Benefits bullet parser             | P1       | 1.5h  | Split benefits section on bullet points/newlines. Clean markdown formatting. Categorize each benefit using `benefit_categories.yaml` |
| A-020 | Benefits category taxonomy         | P1       | 1h    | Build `benefit_categories.yaml`: compensation, health, flexibility, retirement, mobility, food, development, equipment, events, other |
| A-021 | Tasks section storage              | P2       | 0.5h  | Store raw tasks section text. No extraction logic — just preserve the section for potential future use |

**Exit criteria:** Skills extractor returns results for >80% of descriptions. Benefits parser produces categorized lists for descriptions that have a benefits section.

---

### Sprint 5: Validation & QA (Day 6)

| ID    | Task                              | Priority | Est   | Details |
|-------|-----------------------------------|----------|-------|---------|
| A-022 | Cross-field validation module      | P0       | 2h    | Seniority vs experience, salary range sanity (15k-300k EUR), skills count (0 is suspicious, >30 is noise), contract vs seniority, title vs seniority |
| A-023 | Quality report module              | P0       | 1.5h  | Field coverage %, value distributions with bar charts, top 20 skills frequency, rare skills count, flagged row count |
| A-024 | Stratified review sampler          | P0       | 1.5h  | 200 rows: high-confidence, flagged, no-skills, short-desc, German-only, English-only, has-salary, random. Export to .xlsx with blank columns for reviewer |
| A-025 | Pipeline orchestrator              | P1       | 1h    | `run_pipeline.py`: load config → dedup → validate → split → extract → validate → report → sample → export |

**Exit criteria:** Full pipeline runs end-to-end on deduped dataset. Quality report shows reasonable distributions. Review spreadsheet exported.

---

### Sprint 6: Review & Finalize (Day 7)

| ID    | Task                              | Priority | Est   | Details |
|-------|-----------------------------------|----------|-------|---------|
| A-026 | Manual review                      | P0       | 3h    | Review 200 sampled rows. Mark each field as TRUE/FALSE/PARTIAL. Note correction where FALSE |
| A-027 | Calculate per-field accuracy       | P0       | 0.5h  | Run accuracy calculation on reviewed rows. Identify which extractors need fixes |
| A-028 | Fix extraction bugs                | P0       | 2h    | Address systematic errors found in review (e.g., regex missing a common pattern, keyword list gaps) |
| A-029 | Final extraction run               | P0       | 0.5h  | Re-run full pipeline with fixes applied |
| A-030 | Export final CSV                   | P0       | 0.5h  | `jobs_enriched.csv` with all extracted fields + validation warnings column |

**Exit criteria:** Per-field accuracy >80% on contract, modality, seniority, salary. Skills accuracy documented (expected ~60-65%). Final CSV exported.

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Section splitting fails on >30% of descriptions | Skills/benefits accuracy drops | Fall back to full-text extraction; analyze failing formats and add patterns |
| Skills keyword list has low coverage | Miss niche or German-phrased skills | Sample extraction failures, expand list iteratively. Accept ~60-65% ceiling |
| German phrasing for experience defeats regex | experience_years field mostly null | Add more German patterns; accept as low-priority field if <70% |
| Salary regex matches non-salary numbers | False positives in salary field | Add context requirement (nearby keywords: Gehalt, salary, Vergütung, Compensation) |
