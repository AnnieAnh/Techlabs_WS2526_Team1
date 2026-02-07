# Strategy B: LLM Batch Extraction ("Throw Intelligence At It")

## Roadmap & Backlog

**Estimated Duration:** ~3-4 days
**Cost:** $5-40 (API dependent)
**Compute:** API-based, no local GPU needed
**Accuracy Target:** ~88-92% overall

---

## Requirements

### Python Dependencies

```txt
# requirements.txt
pandas>=2.0
pandera>=0.18
openpyxl>=3.1            # Excel export for review
pyyaml>=6.0              # Config
tqdm>=4.65               # Progress bars
anthropic>=0.40          # Claude API (if using Claude)
openai>=1.50             # OpenAI API (if using GPT-4o-mini)
tenacity>=8.2            # Retry logic for API calls
tiktoken>=0.7            # Token counting (OpenAI)
jsonschema>=4.20         # Validate LLM JSON output
pytest>=7.4
```

### API Access (Choose One)

| Provider        | Model            | Batch Cost / 1M input | Notes                         |
|-----------------|------------------|-----------------------|-------------------------------|
| Anthropic       | Claude Sonnet    | ~$1.50                | Best for German, batch API    |
| OpenAI          | GPT-4o-mini      | ~$0.075               | Cheapest, batch API           |
| OpenAI          | GPT-4o           | ~$1.25                | Better quality, batch API     |

**Recommendation:** Start with GPT-4o-mini ($2-5 total). If accuracy on German descriptions is weak, switch to Claude Sonnet.

### Hardware

- Any machine with internet access
- No GPU needed
- API key stored in environment variable (never in code)

### Input Data

- JSON file with Indeed job listings (JobSpy format)
- Fields required: `id`, `title`, `description`, `location`, `job_type`, `is_remote`

---

## Project Folder Structure

```
job-extraction/
│
├── config/
│   ├── settings.yaml              # API keys ref, paths, batch sizes
│   ├── extraction_prompt.yaml     # The extraction prompt (versioned!)
│   └── output_schema.json         # JSON Schema for LLM output validation
│
├── src/
│   ├── __init__.py
│   ├── dedup.py                   # Deduplication logic
│   ├── validate_input.py          # Input data quality checks
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py              # API client wrapper (Anthropic or OpenAI)
│   │   ├── prompt_builder.py      # Constructs prompt per description
│   │   ├── batch_processor.py     # Batch submission + polling + collection
│   │   ├── response_parser.py     # JSON extraction + schema validation
│   │   └── retry_handler.py       # Failed row reprocessing
│   ├── rule_validators/
│   │   ├── __init__.py
│   │   ├── salary_sanity.py       # Post-LLM salary range checks
│   │   ├── cross_field.py         # Cross-field consistency
│   │   └── skills_dedup.py        # Normalize duplicate skills (Go/Golang)
│   ├── quality_report.py          # Distribution analysis
│   └── review_sampler.py          # Stratified sampling for manual review
│
├── tests/
│   ├── test_prompt_builder.py
│   ├── test_response_parser.py    # Test JSON parsing + edge cases
│   └── test_rule_validators.py
│
├── data/
│   ├── raw/                       # Original JSON
│   ├── deduped/                   # After dedup
│   ├── batches/                   # Batch request/response files
│   │   ├── batch_001_request.jsonl
│   │   ├── batch_001_response.jsonl
│   │   └── ...
│   ├── extracted/                 # Parsed extraction results
│   ├── failed/                    # Rows that failed parsing
│   └── review/                    # Review spreadsheets
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_prompt_development.ipynb  # Test prompt on 20 descriptions
│   └── 03_quality_analysis.ipynb
│
├── run_pipeline.py
├── requirements.txt
└── README.md
```

---

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                       STRATEGY B PIPELINE                           │
└─────────────────────────────────────────────────────────────────────┘

  ┌──────────┐     ┌──────────────┐     ┌──────────────────┐
  │ Raw JSON │────▶│ Deduplication │────▶│ Input Validation │
  │  ~100k   │     │  URL + Title  │     │   Quality Report │
  └──────────┘     │  + Fingerprint│     └────────┬─────────┘
                   └──────┬───────┘              │
                          │  ~70-80k              │
                          ▼                       ▼
             ┌─────────────────────────────────────────┐
             │         Prompt Construction              │
             │  • System prompt (extraction rules)      │
             │  • Per-row: title + description + meta   │
             │  • Output: JSON with all fields          │
             └──────────────────┬──────────────────────┘
                                │
                   ┌────────────┼────────────┐
                   │   BATCH PROCESSING LOOP  │
                   │                          │
                   ▼                          │
          ┌────────────────┐                  │
          │  Submit Batch  │                  │
          │  (~1000 rows   │                  │
          │   per batch)   │                  │
          └───────┬────────┘                  │
                  │                           │
                  ▼                           │
          ┌────────────────┐                  │
          │  Poll Status   │──── pending ─────┘
          │  (wait ~30min  │
          │   per batch)   │
          └───────┬────────┘
                  │ complete
                  ▼
          ┌────────────────┐     ┌───────────────────┐
          │ Collect Results│────▶│  Parse JSON Output │
          │                │     │  + Schema Validate │
          └────────────────┘     └─────────┬─────────┘
                                           │
                              ┌────────────┼────────────┐
                              │            │            │
                              ▼            ▼            ▼
                      ┌──────────┐  ┌───────────┐  ┌──────────┐
                      │ Success  │  │  Partial  │  │  Failed  │
                      │ (valid   │  │  (some    │  │  (bad    │
                      │  JSON)   │  │  fields   │  │  JSON)   │
                      └────┬─────┘  │  missing) │  └────┬─────┘
                           │        └─────┬─────┘       │
                           │              │        ┌────▼─────┐
                           │              │        │  Retry   │
                           │              │        │  Queue   │
                           │              │        │ (re-send │
                           │              │        │  w/ fix) │
                           │              │        └────┬─────┘
                           └──────────────┼─────────────┘
                                          │
                                          ▼
                   ┌──────────────────────────────────────┐
                   │       Rule-Based Validation           │
                   │  (Post-LLM sanity checks)             │
                   │                                       │
                   │  • Salary: 15k-300k EUR range?        │
                   │  • Skills: deduplicate Go/Golang      │
                   │  • Seniority vs title consistency     │
                   │  • Cross-field contradiction flags    │
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

### Prompt Engineering Detail

```
┌─────────────────────────────────────────────────────────────────┐
│                    PROMPT ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ SYSTEM PROMPT (same for all rows)                          │  │
│  │                                                            │  │
│  │  You are a structured data extractor for German IT job     │  │
│  │  descriptions. Extract fields into JSON. Rules:            │  │
│  │                                                            │  │
│  │  • Return ONLY valid JSON, no markdown, no explanation     │  │
│  │  • Use null for fields not found (don't guess)             │  │
│  │  • Skills: only specific, named technologies/tools         │  │
│  │  • Distinguish required_skills vs nice_to_have_skills      │  │
│  │  • Salary: annual EUR, integers only                       │  │
│  │  • seniority: Intern|Junior|Mid|Senior|Lead|Principal      │  │
│  │  • languages: include level when stated                    │  │
│  │  • tasks: max 5, concise phrases                           │  │
│  │                                                            │  │
│  │  Output schema: { ... }                                    │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ USER PROMPT (per row)                                      │  │
│  │                                                            │  │
│  │  Title: {title}                                            │  │
│  │  Location: {location}                                      │  │
│  │  JobSpy job_type: {job_type}                               │  │
│  │  JobSpy is_remote: {is_remote}                             │  │
│  │                                                            │  │
│  │  Description:                                              │  │
│  │  {description}                                             │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ EXPECTED OUTPUT                                            │  │
│  │                                                            │  │
│  │  {                                                         │  │
│  │    "contract_type": "Full-time",                           │  │
│  │    "work_modality": "Remote",                              │  │
│  │    "seniority": "Senior",                                  │  │
│  │    "experience_years": "5+",                               │  │
│  │    "salary_min": 80000,                                    │  │
│  │    "salary_max": 110000,                                   │  │
│  │    "required_skills": ["Kubernetes", "Terraform", ...],    │  │
│  │    "nice_to_have_skills": ["Python", "Go", ...],           │  │
│  │    "benefits": [                                           │  │
│  │      {"name": "Competitive Salary", "category": "comp"},   │  │
│  │      {"name": "Flexible Schedule", "category": "flex"}     │  │
│  │    ],                                                      │  │
│  │    "tasks": ["Manage multi-cloud infra", ...],             │  │
│  │    "languages": [                                          │  │
│  │      {"language": "English", "level": "fluent",            │  │
│  │       "required": true}                                    │  │
│  │    ]                                                       │  │
│  │  }                                                         │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Roadmap

```
Day 1: Foundation + Prompt Dev
├── ██████ Project setup, dedup, input validation
├── ██████ Prompt engineering (iterate on 20 descriptions)
└── ██████ JSON schema + response parser

Day 2: Batch Processing
├── ██████ Batch processor (submit, poll, collect)
├── ██████ Test batch: 100 rows
├── ██████ Validate test batch accuracy
└── ██████ Prompt refinement based on test results

Day 3: Full Run + Validation
├── ██████ Process all ~70-80k rows in batches
├── ██████ Parse all responses, handle failures
├── ██████ Rule-based post-validation
├── ██████ Quality report + distributions
└── ██████ Retry failed rows

Day 4: Review + Finalize
├── ██████ Stratified review sample (200 rows)
├── ██████ Manual review
├── ██████ Calculate accuracy
└── ██████ Export final CSV
```

---

## Backlog

### Sprint 1: Foundation + Prompt Development (Day 1)

| ID    | Task                              | Priority | Est   | Details |
|-------|-----------------------------------|----------|-------|---------|
| B-001 | Project scaffolding               | P0       | 0.5h  | Folder structure, deps, config files |
| B-002 | Deduplication module               | P0       | 1.5h  | Same as Strategy A: URL + title + fingerprint |
| B-003 | Input validation                   | P1       | 1h    | Pandera schema, quality report, filter non-extractable rows |
| B-004 | Define output JSON schema          | P0       | 1h    | `output_schema.json` with all fields, types, allowed values. Use jsonschema for validation |
| B-005 | Prompt engineering - v1            | P0       | 2h    | Write system prompt + user prompt template. Test manually on 5 descriptions in the API playground. Iterate on wording |
| B-006 | Prompt engineering - v2            | P0       | 1h    | Run on 20 diverse descriptions (DE, EN, mixed, short, long). Check: does it return valid JSON? Are fields accurate? Does it hallucinate skills? |
| B-007 | Response parser module             | P0       | 1.5h  | Extract JSON from LLM response. Handle: valid JSON, JSON in markdown fences, partial JSON, non-JSON responses. Validate against schema |
| B-008 | API client wrapper                 | P0       | 1h    | Abstract Anthropic/OpenAI behind unified interface. Environment variable for API key. Rate limiting |

**Exit criteria:** Prompt produces valid, accurate JSON for 18/20 test descriptions. Parser handles all response formats.

**Anti-patterns to avoid:**
- Don't put the full JSON schema in the prompt — it wastes tokens. Describe the schema in natural language and validate the output separately
- Don't ask the LLM to explain its reasoning — you only want the JSON, explanation costs tokens and money
- Don't use temperature > 0.1 for extraction — you want consistency, not creativity

---

### Sprint 2: Batch Processing (Day 2)

| ID    | Task                              | Priority | Est   | Details |
|-------|-----------------------------------|----------|-------|---------|
| B-009 | Batch request builder              | P0       | 1.5h  | Convert DataFrame rows to JSONL batch format. Handle: token counting per request, splitting into batches of ~1000 rows, truncating extremely long descriptions |
| B-010 | Batch submission logic             | P0       | 1.5h  | Submit batch → get batch_id → save to tracking file. Handle API errors, rate limits |
| B-011 | Batch polling + collection         | P0       | 1h    | Poll batch status. On completion: download results, match to original rows by custom_id |
| B-012 | Test batch: 100 rows               | P0       | 1h    | Submit 100 diverse rows. Wait for results. Parse and manually check 20 |
| B-013 | Accuracy check on test batch       | P0       | 1h    | Compare LLM output to what you'd extract manually. Per-field accuracy. Identify systematic errors |
| B-014 | Prompt refinement                  | P1       | 1h    | Fix issues found in B-013. Common fixes: clarify German terms, add examples of edge cases to prompt, tighten skill extraction rules |

**Exit criteria:** 100-row test batch processed end-to-end. Per-field accuracy >85% on spot check. Prompt finalized.

**Batch processing detail:**

```
  Batch Lifecycle
  ═══════════════

  ┌─────────────────────────────────────────────────┐
  │           80,000 rows to process                 │
  └────────────────────┬────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          │            │            │
          ▼            ▼            ▼
     Batch 1       Batch 2      ...  Batch 80
     rows 1-1000   rows 1001-2000    rows 79001-80000
          │            │                  │
          ▼            ▼                  ▼
     ┌─────────────────────────────────────────┐
     │  Submit all batches to API               │
     │  Save batch_ids to tracking.json         │
     └─────────────────┬───────────────────────┘
                       │
     ┌─────────────────▼───────────────────────┐
     │  Poll loop (every 60 seconds)            │
     │  Check status of each batch_id           │
     │                                          │
     │  Status: pending → in_progress →         │
     │          completed / failed / expired     │
     └─────────────────┬───────────────────────┘
                       │
     ┌─────────────────▼───────────────────────┐
     │  Collect completed batches               │
     │  Re-submit failed batches (max 2 retries)│
     │  Parse JSONL responses                   │
     └─────────────────────────────────────────┘
```

---

### Sprint 3: Full Run + Post-Validation (Day 3)

| ID    | Task                              | Priority | Est   | Details |
|-------|-----------------------------------|----------|-------|---------|
| B-015 | Submit all batches                 | P0       | 1h    | Submit ~80 batches of 1000. Track progress. Estimated API time: 2-4 hours |
| B-016 | Collect + parse all results        | P0       | 1.5h  | Download all completed batches. Parse JSON. Log failures |
| B-017 | Handle failed rows                 | P0       | 1h    | Identify rows with: invalid JSON, missing required fields, API errors. Re-submit with retry. Accept remaining failures as null |
| B-018 | Skills normalization               | P1       | 1h    | Deduplicate: Go/Golang, K8s/Kubernetes, JS/JavaScript, etc. Build alias map. Lowercase comparison |
| B-019 | Salary sanity validator             | P1       | 0.5h  | Post-LLM check: range 15k-300k EUR, min < max, flag outliers. LLMs sometimes extract revenue or funding amounts as salary |
| B-020 | Cross-field validation              | P0       | 1h    | Same logic as Strategy A but applied to LLM output. Flag contradictions |
| B-021 | Quality report                     | P0       | 1h    | Field coverage, value distributions, skills frequency, flagged rows |

**Exit criteria:** All rows processed or explicitly failed. <2% failure rate. Quality report shows reasonable distributions.

**Critical: LLM output gotchas to handle:**

```
  Common LLM Output Failures
  ══════════════════════════

  1. Markdown wrapping        →  Strip ```json ... ``` fences
  2. Trailing comma in JSON   →  Regex fix before parsing
  3. Skills hallucination     →  "Python" inferred from "scripting"
                                 → Post-validate against description
  4. Salary confusion         →  Extracts company revenue as salary
                                 → Sanity check: 15k-300k EUR
  5. German number format     →  50.000 parsed as 50.0 not 50000
                                 → Prompt should specify "integers only"
  6. Empty benefits           →  LLM returns [] when benefits section
                                 exists but is just a header
                                 → Flag for review, not an error
  7. Overenthusiastic tasks   →  Returns 15 tasks when you asked for 5
                                 → Truncate to first 5 in parser
```

---

### Sprint 4: Review + Finalize (Day 4)

| ID    | Task                              | Priority | Est   | Details |
|-------|-----------------------------------|----------|-------|---------|
| B-022 | Stratified review sampler          | P0       | 1h    | 200 rows: high-confidence, flagged, LLM-nulls, German-only, English-only, has-salary, long-description, random |
| B-023 | Export review spreadsheet          | P0       | 0.5h  | Include: extracted fields, original description, blank columns for reviewer |
| B-024 | Manual review                      | P0       | 3h    | Review 200 rows. Special attention to: skills (hallucination?), benefits (complete?), German experience phrases |
| B-025 | Calculate per-field accuracy       | P0       | 0.5h  | Target: >85% on skills, >90% on contract/modality/seniority |
| B-026 | Export final enriched CSV          | P0       | 0.5h  | All extracted fields + validation_warnings + extraction_confidence |
| B-027 | Cost reconciliation                | P2       | 0.5h  | Total API cost, cost per row, tokens used. Document for future reference |

**Exit criteria:** Per-field accuracy measured. Final CSV exported with all fields. Total cost documented.

---

## Cost Estimation Breakdown

```
  Cost Model (GPT-4o-mini batch, 80k rows)
  ═════════════════════════════════════════

  Input tokens per row:
    System prompt:       ~500 tokens  (one-time, cached)
    User prompt + desc:  ~1200 tokens (avg German IT job desc)
    ─────────────────────────────────
    Total input:         ~1700 tokens/row

  Output tokens per row: ~300 tokens  (JSON response)

  Total:
    Input:  80,000 × 1,700  = 136M tokens × $0.075/1M = $10.20
    Output: 80,000 × 300    =  24M tokens × $0.30/1M  =  $7.20
    ─────────────────────────────────────────────────────
    Total (batch 50% off):                              ~$8.70

  With retries (~5%):                                   ~$9.50


  Cost Model (Claude Sonnet batch, 80k rows)
  ═════════════════════════════════════════

  Total:
    Input:  136M tokens × $1.50/1M  = $204
    Output:  24M tokens × $7.50/1M  =  $180
    ─────────────────────────────────────────
    Total (batch 50% off):           ~$192
    
  ⚠️ Claude Sonnet is significantly more expensive
     Use GPT-4o-mini first, switch only if German 
     accuracy is unacceptable
```

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| LLM hallucinates skills not in description | Inflated skill counts, false data | Post-validate: check every extracted skill actually appears in description text (fuzzy match) |
| LLM returns invalid JSON | Rows lost, need retry | Robust parser with fallback strategies. Budget for 5% retry rate |
| German descriptions get worse accuracy | Lower quality on majority of data | Test on 20 German descriptions first. If <80% accuracy, switch model or add German examples to prompt |
| Batch API takes very long | Delays full pipeline | Submit all batches early, poll async. Budget 4-6 hours wait time |
| API cost exceeds estimate | Budget overrun | Token-count before submitting. Truncate descriptions >3000 tokens. Start with GPT-4o-mini |
| Rate limits or API outages | Partial results | Checkpoint: save results per batch. Resume from last completed batch |
| Non-deterministic output | Same row gives different results on retry | Temperature=0, fixed seed if supported. Accept minor variance between runs |
