# Golden Set for Evaluation

## Purpose

The `golden_set.csv` contains 50 randomly sampled rows for manual annotation.
When filled in, the pipeline's `validate` step automatically computes
precision/recall/Jaccard metrics by comparing extraction results against this
ground truth.

## How to fill it in

1. Open `golden_set.csv` in a spreadsheet editor.
2. For each `row_id`, look up the original job description in the pipeline output
   or in `data/extraction/extracted/extraction_results.json`.
3. Manually fill in the correct values for each field:
   - **Categorical fields** (`contract_type`, `work_modality`, `seniority`): use canonical values
   - **Numeric fields** (`salary_min`, `salary_max`, `experience_years`): use numbers or leave blank if not mentioned
   - **List fields** (`technical_skills`, `soft_skills`, `nice_to_have_skills`, `benefits`, `tasks`): use JSON arrays, e.g. `["Python", "Docker", "AWS"]`
4. Leave fields blank (empty string) if the information is genuinely not present in the description.
5. Save as CSV (UTF-8 encoding).

## Running evaluation

The pipeline automatically picks up `golden_set.csv` during the `validate` step:

```bash
python orchestrate.py --only validate
```

Results are saved to `data/extraction/reports/accuracy_report.json`.
