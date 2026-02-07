# Section Split + Targeted ML Models ("Right Tool Per Field")

## Roadmap & Backlog

**Estimated Duration:** ~2 weeks
**Cost:** $0 (local models)
**Compute:** GPU recommended (free Colab works), CPU fallback (slow)
**Accuracy Target:** ~78-82% overall

---

## Requirements

### Python Dependencies

```txt
# requirements.txt
pandas>=2.0
pandera>=0.18
openpyxl>=3.1
pyyaml>=6.0
tqdm>=4.65
pytest>=7.4

# NLP / ML
transformers>=4.40
torch>=2.2                   # or tensorflow
spacy>=3.7
sentencepiece>=0.2           # required by some HF models
protobuf>=4.25               # required by some HF models

# Models to download
# jjzha/jobbert_skill_extraction     (skill NER)
# facebook/bart-large-mnli           (zero-shot classification)
# deepset/gelectra-base-germanquad   (German QA)
```

### Hardware Options

| Setup               | Skill NER (80k rows) | Zero-shot (80k) | Total     |
|---------------------|---------------------|-----------------|-----------|
| Local GPU (RTX 3060+) | ~1-2 hours        | ~3-4 hours      | ~5-6h     |
| Google Colab (free T4) | ~2-3 hours        | ~5-6 hours      | ~8-10h    |
| CPU only (no GPU)    | ~10-14 hours        | ~20+ hours      | ~30h+     |

**Recommendation:** Use Google Colab free tier. Upload deduped data, run models on GPU, download results.

### Models to Evaluate

| Task                | Model                                  | Language | Size   |
|---------------------|----------------------------------------|----------|--------|
| Skill NER           | `jjzha/jobbert_skill_extraction`       | EN       | ~440MB |
| Skill NER (backup)  | `Nucha/Nucha_ITSkillNER_BERT`          | EN       | ~440MB |
| German QA           | `deepset/gelectra-base-germanquad`     | DE       | ~440MB |
| English QA          | `deepset/roberta-base-squad2`          | EN       | ~480MB |
| Zero-shot classify  | `facebook/bart-large-mnli`             | EN+DE    | ~1.6GB |

### Input Data

- JSON file with Indeed job listings (JobSpy format)
- Fields required: `id`, `title`, `description`, `location`, `job_type`, `is_remote`

---

## Project Folder Structure

```
job-extraction/
│
├── config/
│   ├── settings.yaml              # Paths, thresholds, model names
│   ├── skills_keywords.yaml       # Supplement keyword list
│   ├── section_headers.yaml       # Header → section mapping
│   ├── benefit_categories.yaml    # Benefit category taxonomy
│   └── skill_aliases.yaml         # Go→Golang, K8s→Kubernetes, etc.
│
├── src/
│   ├── __init__.py
│   ├── dedup.py
│   ├── validate_input.py
│   ├── section_splitter.py        # Description → sections
│   ├── extractors/
│   │   ├── __init__.py
│   │   ├── contract_type.py       # Regex (same as Strategy A)
│   │   ├── work_modality.py       # Regex (same as Strategy A)
│   │   ├── salary.py              # Regex (same as Strategy A)
│   │   ├── experience.py          # Regex (same as Strategy A)
│   │   ├── languages.py           # Regex (same as Strategy A)
│   │   ├── seniority.py           # Title-based + zero-shot fallback
│   │   ├── skills_ner.py          # JobBERT NER + keyword supplement
│   │   ├── skills_normalize.py    # Alias resolution + dedup
│   │   ├── benefits.py            # Bullet parsing + QA extraction
│   │   └── tasks_qa.py            # QA-based task extraction
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model_loader.py        # Lazy model loading + caching
│   │   └── batch_inference.py     # Batched inference with progress
│   ├── cross_validation.py
│   ├── quality_report.py
│   └── review_sampler.py
│
├── tests/
│   ├── test_section_splitter.py
│   ├── test_regex_extractors.py
│   ├── test_skills_ner.py         # Test NER on known examples
│   └── test_cross_validation.py
│
├── data/
│   ├── raw/
│   ├── deduped/
│   ├── sections/                  # Intermediate: split sections
│   ├── extracted/                 # Final enriched CSV
│   └── review/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_section_splitter_dev.ipynb
│   ├── 03_model_evaluation.ipynb    # Test each model on 50 samples
│   ├── 04_skills_ner_tuning.ipynb   # Confidence threshold tuning
│   └── 05_quality_analysis.ipynb
│
├── colab/
│   └── gpu_inference.ipynb        # Run on Colab with GPU
│
├── run_pipeline.py
├── requirements.txt
└── README.md
```

---

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                       STRATEGY C PIPELINE                           │
└─────────────────────────────────────────────────────────────────────┘

  ┌──────────┐     ┌──────────────┐     ┌──────────────────┐
  │ Raw JSON │────▶│ Deduplication │────▶│ Input Validation │
  │  ~100k   │     │  URL + Title  │     │   Quality Report │
  └──────────┘     │  + Fingerprint│     └────────┬─────────┘
                   └──────┬───────┘              │
                          │  ~70-80k              │
                          ▼                       ▼
                   ┌──────────────────────────────────────┐
                   │          Section Splitting            │
                   │   Description → tasks | requirements  │
                   │                  benefits | about     │
                   └──────────────────┬───────────────────┘
                                      │
                                      │ Each section gets
                                      │ a different extractor
                                      │
     ┌────────────────────────────────┼──────────────────────────────┐
     │                                │                              │
     │        REGEX LAYER             │      ML MODEL LAYER          │
     │     (fast, CPU, all rows)      │   (GPU, requirements +      │
     │                                │    benefits sections)        │
     │  ┌──────────────────────┐      │                              │
     │  │ Full Description     │      │  ┌─────────────────────┐    │
     │  │ • contract_type      │      │  │ Requirements Section │    │
     │  │ • salary             │      │  │                     │    │
     │  │ • work_modality      │      │  │  ┌───────────────┐  │    │
     │  │ • experience_years   │      │  │  │ JobBERT NER   │  │    │
     │  │ • languages          │      │  │  │ (skill spans) │  │    │
     │  └──────────┬───────────┘      │  │  └───────┬───────┘  │    │
     │             │                  │  │          │          │    │
     │  ┌──────────────────────┐      │  │  ┌───────▼───────┐  │    │
     │  │ Title                │      │  │  │ Keyword       │  │    │
     │  │ • seniority (primary)│      │  │  │ Supplement    │  │    │
     │  └──────────┬───────────┘      │  │  │ (catch misses)│  │    │
     │             │                  │  │  └───────┬───────┘  │    │
     │             │                  │  │          │          │    │
     │             │                  │  │  ┌───────▼───────┐  │    │
     │             │                  │  │  │ Alias Resolve │  │    │
     │             │                  │  │  │ + Deduplicate │  │    │
     │             │                  │  │  └───────┬───────┘  │    │
     │             │                  │  └──────────┼──────────┘    │
     │             │                  │             │               │
     │             │                  │  ┌──────────▼──────────┐    │
     │             │                  │  │ Benefits Section    │    │
     │             │                  │  │                     │    │
     │             │                  │  │  Bullet parsing     │    │
     │             │                  │  │  + Category classify │    │
     │             │                  │  └──────────┬──────────┘    │
     │             │                  │             │               │
     └─────────────┼──────────────────┼─────────────┼───────────────┘
                   │                  │             │
                   └──────────────────┼─────────────┘
                                      │
                                      ▼
                   ┌──────────────────────────────────────┐
                   │       Merge All Extracted Fields      │
                   │  row_id → {regex fields + ML fields}  │
                   └──────────────────┬───────────────────┘
                                      │
                                      ▼
                   ┌──────────────────────────────────────┐
                   │       Cross-Field Validation          │
                   │  + Confidence scoring per field       │
                   └──────────────────┬───────────────────┘
                                      │
                          ┌───────────┼───────────┐
                          ▼           ▼           ▼
                   ┌───────────┐ ┌────────┐ ┌──────────┐
                   │ Quality   │ │ Review │ │ Enriched │
                   │ Report    │ │ Sample │ │ CSV      │
                   └───────────┘ └────────┘ └──────────┘
```

---

### Model Selection Decision Tree

```
  Which model for which field?
  ════════════════════════════

  contract_type ──────────────────────── Regex (finite vocabulary)
  salary ─────────────────────────────── Regex (numeric patterns)
  work_modality ──────────────────────── Regex (finite vocabulary)
  experience_years ───────────────────── Regex (numeric + "mehrjährige")
  languages ──────────────────────────── Regex (finite vocabulary)

  seniority ──┬── Title contains clear term? ──── Rule-based
              │        (Senior, Junior, etc.)
              │
              └── Ambiguous? ─────────────────── Zero-shot classifier
                   (e.g. "Berufserfahrene"       (bart-large-mnli)
                    in description body)          Labels: [Junior, Mid,
                                                  Senior, Lead, Principal]

  skills ─────┬── Requirements section ────────── JobBERT NER
              │    available?                      (primary extractor)
              │         │
              │         ▼
              │    Keyword supplement ──────────── Curated keyword list
              │         │                          (catch NER misses)
              │         ▼
              │    Alias resolution ────────────── skill_aliases.yaml
              │                                    (Go→Golang dedup)
              │
              └── No requirements section? ─────── Keyword-only on full text

  benefits ───┬── Benefits section found? ──────── Bullet parsing
              │                                     + category classifier
              │
              └── No section header? ────────────── Skip (don't guess)

  tasks ──────┬── Tasks section found? ─────────── Store raw text
              │                                     (no ML extraction)
              │
              └── No section header? ────────────── Skip
```

---

### NER Inference Pipeline Detail

```
  Skills NER Processing (per row)
  ═══════════════════════════════

  Input: requirements section text (~200-800 tokens)
         "Du beherrschst Golang und/oder Rust.
          Du hast Kenntnisse der WebAssembly-Technologie
          und Kubernetes..."

       ┌──────────────────────────────────┐
       │  Chunk text (512 token windows,  │
       │  50 token overlap)               │
       │                                  │
       │  chunk_1: tokens 0-511           │
       │  chunk_2: tokens 462-973         │
       │  (overlap prevents split-word    │
       │   entity loss)                   │
       └────────────────┬─────────────────┘
                        │
                        ▼
       ┌──────────────────────────────────┐
       │  JobBERT NER inference           │
       │  model: jjzha/jobbert_skill_     │
       │         extraction               │
       │  aggregation: "simple"           │
       │                                  │
       │  → [                             │
       │      {word:"Golang",             │
       │       entity:"SKILL",            │
       │       score:0.94},               │
       │      {word:"Rust",               │
       │       entity:"SKILL",            │
       │       score:0.89},               │
       │      {word:"Kubernetes",         │
       │       entity:"SKILL",            │
       │       score:0.92}                │
       │    ]                             │
       └────────────────┬─────────────────┘
                        │
                        ▼
       ┌──────────────────────────────────┐
       │  Confidence filter               │
       │  threshold: 0.4 (tunable)        │
       │                                  │
       │  Keep: score >= 0.4              │
       │  Drop: score < 0.4              │
       └────────────────┬─────────────────┘
                        │
                        ▼
       ┌──────────────────────────────────┐
       │  Keyword supplement              │
       │  (skills_keywords.yaml)          │
       │                                  │
       │  Scan full description for       │
       │  known terms NER missed:         │
       │  e.g. "WebAssembly" (niche)      │
       │  e.g. "FaaS" (abbreviation)      │
       └────────────────┬─────────────────┘
                        │
                        ▼
       ┌──────────────────────────────────┐
       │  Alias resolution + dedup        │
       │  (skill_aliases.yaml)            │
       │                                  │
       │  "Go" + "Golang" → "Go"         │
       │  "K8s" + "Kubernetes" → "K8s"   │
       │  Deduplicate set                 │
       └────────────────┬─────────────────┘
                        │
                        ▼
  Output: ["Go", "Rust", "WebAssembly", "Kubernetes", "FaaS"]
```

---

## Roadmap

```
Week 1: Foundation + Regex + Section Splitting
├── Day 1-2: Foundation
│   ├── ██████████ Project setup, dedup, input validation
│   └── ██████████ Section splitter (build + test)
│
├── Day 3-4: Regex Extractors
│   ├── ██████████ All regex extractors (contract, salary, exp, etc.)
│   ├── ██████████ Unit tests
│   └── ██████████ Run regex layer on full dataset
│
└── Day 5: Model Evaluation
    ├── ██████████ Download + test JobBERT on 50 samples
    ├── ██████████ Download + test zero-shot on 50 samples
    └── ██████████ Tune confidence thresholds

Week 2: ML Extraction + Validation
├── Day 6-7: Skills NER Pipeline
│   ├── ██████████ Skills NER module (chunking + inference)
│   ├── ██████████ Keyword supplement
│   ├── ██████████ Alias resolution + dedup
│   └── ██████████ Run on full dataset (GPU / Colab)
│
├── Day 8: Benefits + Seniority
│   ├── ██████████ Benefits bullet parser + categorizer
│   ├── ██████████ Seniority zero-shot fallback
│   └── ██████████ Merge all fields
│
├── Day 9: Validation & QA
│   ├── ██████████ Cross-field validation
│   ├── ██████████ Quality report
│   └── ██████████ Review sample generation
│
└── Day 10: Review & Finalize
    ├── ██████████ Manual review (200 rows)
    ├── ██████████ Fix bugs + re-run
    └── ██████████ Export final CSV
```

---

## Backlog

### Sprint 1: Foundation (Day 1-2)

| ID    | Task                              | Priority | Est   | Details |
|-------|-----------------------------------|----------|-------|---------|
| C-001 | Project scaffolding               | P0       | 1h    | Folder structure, deps, settings.yaml |
| C-002 | Data exploration notebook          | P0       | 2h    | Understand data, sample descriptions, catalog section headers |
| C-003 | Deduplication module               | P0       | 1.5h  | Same as Strategy A |
| C-004 | Input validation                   | P1       | 1h    | Pandera schema, quality report |
| C-005 | Section splitter module            | P0       | 2h    | Header detection, canonical mapping, fallback for headerless descriptions |
| C-006 | Section splitter tests             | P0       | 1h    | 5+ real descriptions, edge cases |
| C-007 | Section coverage analysis          | P0       | 0.5h  | What % split successfully? Which sections found? |

**Exit criteria:** Deduped dataset ready. Section splitter covers >70% of descriptions. Clear picture of data quality.

---

### Sprint 2: Regex Extractors (Day 3-4)

| ID    | Task                              | Priority | Est   | Details |
|-------|-----------------------------------|----------|-------|---------|
| C-008 | Contract type extractor            | P0       | 1h    | Same as Strategy A |
| C-009 | Work modality extractor            | P0       | 1h    | Same as Strategy A |
| C-010 | Salary extractor                   | P0       | 1h    | Same as Strategy A |
| C-011 | Experience years extractor         | P1       | 1h    | Same as Strategy A |
| C-012 | Seniority extractor (title-based)  | P0       | 1h    | Title parsing as primary. Leave zero-shot for Sprint 4 |
| C-013 | Language requirements extractor    | P1       | 1h    | Same as Strategy A |
| C-014 | Unit tests for regex extractors    | P0       | 1.5h  | Parametrized: DE, EN, mixed, edge cases |
| C-015 | Run regex layer on full dataset    | P0       | 0.5h  | Apply all regex extractors, save intermediate results |
| C-016 | Regex accuracy spot check          | P0       | 1h    | Manually verify 30 rows for regex fields |

**Exit criteria:** All regex fields extracted. >80% accuracy on spot check for contract, salary, modality.

---

### Sprint 3: Model Evaluation (Day 5)

| ID    | Task                              | Priority | Est   | Details |
|-------|-----------------------------------|----------|-------|---------|
| C-017 | Environment setup (Colab or local) | P0       | 1h    | Install transformers, torch, verify GPU access |
| C-018 | Download + test JobBERT            | P0       | 1.5h  | Run on 50 sample descriptions. Check: does it find skills? False positives? German handling? |
| C-019 | Download + test Nucha BERT         | P1       | 1h    | Compare to JobBERT on same 50 samples. Pick better model |
| C-020 | Test zero-shot classifier          | P1       | 1h    | bart-large-mnli for seniority on 30 ambiguous cases. Is it worth the compute? |
| C-021 | Confidence threshold tuning        | P0       | 1.5h  | For chosen NER model: test thresholds 0.3, 0.4, 0.5, 0.6. Find precision/recall sweet spot on your 50 samples |
| C-022 | Model selection decision           | P0       | 0.5h  | Decide: which NER model? Use zero-shot for seniority or skip? Document reasoning |

**Exit criteria:** NER model chosen. Confidence threshold set. Decision made on zero-shot seniority.

**Threshold tuning guide:**

```
  Confidence Threshold vs Accuracy Tradeoff
  ══════════════════════════════════════════

  Threshold   Precision   Recall   F1      Notes
  ─────────   ─────────   ──────   ──────  ─────
  0.3         ~70%        ~90%     ~79%    Catches everything but noisy
  0.4         ~78%        ~85%     ~81%    ◄── Usually the sweet spot
  0.5         ~85%        ~75%     ~80%    Good precision, misses some
  0.6         ~90%        ~60%     ~72%    Too conservative, many misses
  0.7         ~93%        ~45%     ~61%    Basically keyword-level recall

  Start at 0.4, adjust based on your 50-sample evaluation.
  If you see too many garbage skills → raise to 0.5
  If you see too many missed skills → lower to 0.3
```

---

### Sprint 4: Skills NER Pipeline (Day 6-7)

| ID    | Task                              | Priority | Est   | Details |
|-------|-----------------------------------|----------|-------|---------|
| C-023 | Skills NER module                  | P0       | 2h    | Text chunking (512 tokens, 50 overlap), batched inference, confidence filtering. Process requirements section first, full text as fallback |
| C-024 | Keyword supplement module          | P0       | 1.5h  | Build `skills_keywords.yaml` (~200 terms). Word-boundary matching. Run after NER to fill gaps |
| C-025 | Skill alias resolution             | P1       | 1h    | Build `skill_aliases.yaml`. Map variants to canonical form. Deduplicate final skill list per row |
| C-026 | GPU batch inference script         | P0       | 1.5h  | Colab notebook or local script. Batch size tuning (32-64 on T4). Progress bar. Checkpoint every 5000 rows |
| C-027 | Run NER on full dataset            | P0       | 2-3h  | Execute on GPU. Monitor memory. Save results |
| C-028 | NER output quality check           | P0       | 1h    | Skills frequency analysis. Top 50 skills make sense? Any garbage? Rows with 0 skills vs >20 skills? |

**Exit criteria:** Skills extracted for all rows. Top 50 skills list looks reasonable. <5% of rows have suspicious skill counts.

---

### Sprint 5: Benefits + Seniority + Merge (Day 8)

| ID    | Task                              | Priority | Est   | Details |
|-------|-----------------------------------|----------|-------|---------|
| C-029 | Benefits bullet parser             | P1       | 1.5h  | Split benefits section on bullets. Clean markdown. Categorize |
| C-030 | Benefit category taxonomy          | P1       | 0.5h  | `benefit_categories.yaml`: compensation, health, flexibility, retirement, mobility, food, development, equipment, events |
| C-031 | Seniority zero-shot fallback       | P2       | 1.5h  | For rows where title-based seniority is null. Run zero-shot on first 200 chars. Only if C-022 decision was "yes" |
| C-032 | Tasks section storage              | P2       | 0.5h  | Store raw tasks section. No ML extraction |
| C-033 | Merge all extracted fields         | P0       | 1.5h  | Combine regex results + NER skills + benefits + seniority into single DataFrame. Handle column naming, list serialization |

**Exit criteria:** All fields merged into single DataFrame. No missing row IDs. No duplicate columns.

---

### Sprint 6: Validation & QA (Day 9)

| ID    | Task                              | Priority | Est   | Details |
|-------|-----------------------------------|----------|-------|---------|
| C-034 | Cross-field validation             | P0       | 1.5h  | Same checks as Strategy A, plus: NER confidence stats per row, skills count sanity |
| C-035 | Quality report                     | P0       | 1.5h  | Coverage per field, value distributions, top skills, confidence distribution histograms |
| C-036 | Confidence analysis                | P1       | 1h    | What % of NER extractions are >0.8 confidence? What % are in the 0.4-0.5 range (borderline)? Does low-confidence correlate with German descriptions? |
| C-037 | Review sample generation           | P0       | 1h    | Stratified 200 rows. Extra strata for this strategy: low-NER-confidence rows, high-skill-count rows |
| C-038 | Export review spreadsheet          | P0       | 0.5h  | Include NER confidence per skill for reviewer context |

**Exit criteria:** Quality report shows reasonable distributions. Review spreadsheet exported.

---

### Sprint 7: Review & Finalize (Day 10)

| ID    | Task                              | Priority | Est   | Details |
|-------|-----------------------------------|----------|-------|---------|
| C-039 | Manual review                      | P0       | 3h    | 200 rows. Special focus on: NER skill accuracy vs keywords, German description performance, false positives |
| C-040 | Per-field accuracy calculation     | P0       | 0.5h  | Compare to Strategy A baseline expectations. Skills should be >70% |
| C-041 | Fix extraction bugs                | P0       | 2h    | Adjust NER threshold if needed. Add missing keywords. Fix section splitter for failing patterns |
| C-042 | Re-run affected components         | P1       | 1-2h  | Only re-run what changed (skills only if threshold changed, not the whole pipeline) |
| C-043 | Final merge + export               | P0       | 0.5h  | `jobs_enriched.csv` with all fields + confidence scores + validation warnings |

**Exit criteria:** Per-field accuracy documented. Final CSV exported. Model selection and threshold decisions documented.

---

## Comparison: What Strategy C Adds Over Strategy A

```
  Accuracy Improvement by Field
  ═════════════════════════════

                        Strategy A     Strategy C     Delta
                        (rules only)   (rules + ML)
  ────────────────────  ───────────    ───────────    ─────
  contract_type         ~90%           ~90%           +0%   (same regex)
  work_modality         ~80%           ~80%           +0%   (same regex)
  seniority             ~85%           ~87%           +2%   (zero-shot helps edge cases)
  salary                ~95%           ~95%           +0%   (same regex)
  experience_years      ~75%           ~75%           +0%   (same regex)
  skills ◄──────────    ~60-65%        ~75-80%        +15%  ◄── MAIN GAIN
  benefits              ~70%           ~75%           +5%   (minor improvement)
  languages             ~85%           ~85%           +0%   (same regex)
  tasks                 ~50%           ~50%           +0%   (raw text both ways)

  Summary:
  ┌──────────────────────────────────────────────────────────┐
  │ Strategy C costs ~2 weeks instead of ~1 week              │
  │ Primary improvement: +15% on skills extraction            │
  │ All other fields: identical to Strategy A                 │
  │                                                           │
  │ Question to ask yourself:                                 │
  │ Is +15% skills accuracy worth an extra week of work?      │
  │ And is it worth the complexity of transformer deps?       │
  └──────────────────────────────────────────────────────────┘
```

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| JobBERT performs poorly on German text | Skills accuracy stays at ~65% (no improvement over keywords) | Test on 50 German samples FIRST (Sprint 3). If <5% improvement, skip NER and use keyword-only |
| Colab GPU session times out during inference | Partial results, need to restart | Checkpoint every 5000 rows. Save intermediate results to Drive. Resume from last checkpoint |
| Transformer dependency conflicts | Hours lost debugging pip/torch versions | Pin exact versions in requirements.txt. Test on clean Colab environment first |
| Zero-shot classification is too slow for 80k rows | 20+ hours on CPU, Colab session limits | Run zero-shot ONLY on rows where title-based seniority is null (~20-30% of rows). Skip if not worth it |
| NER confidence threshold poorly calibrated | Either too many false positives (low threshold) or too many misses (high threshold) | Systematic tuning in Sprint 3 on 50 samples. Document precision/recall at each threshold. Re-tune after full run if distributions look wrong |
| Over-engineering for marginal gains | 2 weeks of work for same quality as Strategy A | Honest checkpoint at end of Sprint 3: if models don't beat keywords meaningfully on your data, pivot to Strategy A |

