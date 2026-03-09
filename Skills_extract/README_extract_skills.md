
# Multilingual Skill Extraction Pipeline with SQLite Checkpointing

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![NLP](https://img.shields.io/badge/NLP-Skill%20Extraction-green)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Status](https://img.shields.io/badge/Status-Project%20Ready-brightgreen)

A hybrid NLP pipeline for extracting **technical skills**, **soft skills**, and **language requirements** from multilingual job descriptions, with built-in **SQLite checkpointing** for scalable and fault-tolerant processing.

---

## Table of Contents

- [Overview](#overview)
- [Why This Project Uses a Hybrid Approach](#why-this-project-uses-a-hybrid-approach)
- [Key Features](#key-features)
- [Input Requirements](#input-requirements)
- [Installation](#installation)
- [Models Used](#models-used)
- [How to Configure the Script](#how-to-configure-the-script)
- [How to Run](#how-to-run)
- [Checkpointing and Resume Logic](#checkpointing-and-resume-logic)
- [Output Files](#output-files)
- [Output Columns](#output-columns)
- [Example Use Cases](#example-use-cases)
- [Performance Notes](#performance-notes)
- [Common Issues](#common-issues)
- [Project Structure](#project-structure)
- [Why This Project Is Useful in a Portfolio](#why-this-project-is-useful-in-a-portfolio)

---

## Overview

This project was designed to process large job posting datasets and extract structured information from unstructured job descriptions.

The script focuses on extracting:

- **technical skills**
- **soft skills**
- **language requirements**
- **detected language of the text**

It uses a **hybrid extraction strategy**, combining rule-based methods with transformer-based token classification. This makes the pipeline more practical for real-world job market data, where text is noisy, inconsistent, and multilingual.

The script is especially useful for:

- job market analysis
- skill demand analysis
- NLP feature engineering
- dashboard preparation
- research or portfolio projects in data science / applied NLP

---

## Why This Project Uses a Hybrid Approach

Using only one extraction method is usually not enough for real job descriptions.

### Regex only is too limited
Regex is fast and effective for known patterns, but it struggles with wording variations and implicit skill mentions.

### Transformers only are expensive
Transformer models are more flexible, but they are slower and more resource-intensive, especially when processing thousands of rows.

### The hybrid approach solves both problems
This script combines the strengths of several techniques:

- **Dictionary matching** for known soft skills
- **Regex patterns** for structured phrases and technical keywords
- **Transformer-based skill span extraction** for more flexible recognition
- **Normalization / canonical mapping** to merge equivalent forms into a cleaner final output

### Why SQLite checkpointing is included
Long-running NLP jobs may fail due to:

- runtime errors
- manual interruption
- memory limits
- machine restart

SQLite checkpointing allows the script to resume from the last processed batch instead of restarting from the beginning.

---

## Key Features

- Multilingual job description processing
- English and German support
- Hybrid extraction pipeline:
  - dictionary matching
  - regex matching
  - transformer token classification
- Technical skill extraction
- Soft skill extraction
- Language requirement detection
- Automatic language detection
- SQLite checkpoint database
- Resume-after-interruption support
- Batch processing for large datasets
- CSV and JSON export
- End-of-run summary statistics

---

## Input Requirements

The script expects a **CSV file** as input.

At minimum, the CSV should contain a column named:

```csv
description
```
Additional columns such as `title`, `company_name`, and `location` can also be included. These will remain in the final output.

## Example input
```
title,company_name,location,description
Data Analyst,ABC GmbH,Bochum,"We are looking for a Data Analyst with SQL, Python, and strong communication skills..."
```

## Installation
Recommended Python version:
```
Python 3.10 or higher
```
Install the required packages with:
```
pip install pandas torch transformers tqdm langdetect
```
The script also uses built-in Python libraries such as:
- `json`
- `re`
- `sqlite3`
- `os`
- `logging`
- `datetime`
- `threading`
- `warnings`
- `multiprocessing`
- `collections`

No separate installation is needed for these.

## Models Used

The script loads the following Hugging Face models:
- `jjzha/jobbert_skill_extraction`
- `jjzha/escoxlmr_skill_extraction`

These models are downloaded automatically the first time the script runs.

## Note
The first execution may take longer because the models need to be downloaded and cached locally.

## How to Configure the Script
At the bottom of the script, the main processing function is called with parameters similar to this:
```
if __name__ == "__main__":
    df_result = analyze_dataset_with_checkpoint(
        csv_file="combined_jobs.csv",
        output_csv="jobs_with_skills_extracted_full.csv",
        output_json="jobs_with_skills_extracted_full.json",
        db_path="skill_extraction_checkpoint.db",
        device=-1,
        batch_size=100,
        sample_size=None,
        num_workers=1,
    )
```
Before running the script, update these values to match your environment.
### Parameter explanation
`csv_file`
Path to the input CSV file.

`output_csv`
File path for the final CSV output.

`output_json`
File path for the final JSON output.

`db_path`
Path to the SQLite checkpoint database.

`device`
Controls where inference runs:
- `-1` = CPU
- `0` = GPU (CUDA), if available

`batch_size`
Number of rows processed before saving a checkpoint.

`sample_size`
Useful for testing:
- `None` = full dataset
- `100` = process first 100 rows only

`num_workers`
Number of worker processes:
- `1` = safest and most stable
- `2+` = potentially faster, but uses more RAM

## Quick Start
1. Place your files in one folder

Example:
```
project/
│
├── A_test-skills-3-with-checkpoint.py
└── combined_jobs.csv
```
2. Install dependencies
```
pip install pandas torch transformers tqdm langdetect
```
3. Update the script parameters

Example:
```
if __name__ == "__main__":
    df_result = analyze_dataset_with_checkpoint(
        csv_file="combined_jobs.csv",
        output_csv="jobs_with_skills_extracted_full.csv",
        output_json="jobs_with_skills_extracted_full.json",
        db_path="skill_extraction_checkpoint.db",
        device=-1,
        batch_size=100,
        sample_size=None,
        num_workers=1,
    )
```
4. Run the script
```
python A_test-skills-3-with-checkpoint.py
```

## Recommended First Test Run

Before processing the full dataset, test the pipeline on a smaller sample:
```
if __name__ == "__main__":
    df_result = analyze_dataset_with_checkpoint(
        csv_file="combined_jobs.csv",
        output_csv="jobs_with_skills_extracted_sample.csv",
        output_json="jobs_with_skills_extracted_sample.json",
        db_path="skill_extraction_checkpoint_sample.db",
        device=-1,
        batch_size=50,
        sample_size=100,
        num_workers=1,
    )
```
This helps verify that:

- the file path is correct
- the required columns exist
- dependencies are installed correctly
- models load successfully
- the output format is as expected

## How to Run

Open a terminal in the project directory and run:
```
python A_test-skills-3-with-checkpoint.py
```

If you renamed the script, replace the filename accordingly.

## Checkpointing and Resume Logic

During execution, processed results are stored in a SQLite checkpoint database such as:
```
skill_extraction_checkpoint.db
```
If the script is interrupted, rerunning it allows processing to continue from the last saved checkpoint.

This is especially helpful for large datasets where full processing may take a long time.

The script also creates backup copies of the checkpoint database periodically to reduce the risk of losing progress.

## Output Files

After successful execution, the script typically generates:

### Final CSV output
```
jobs_with_skills_extracted_full.csv
```
### Final JSON output
```
jobs_with_skills_extracted_full.json
```
### SQLite checkpoint database
```
skill_extraction_checkpoint.db
```

## Output Columns

The script preserves the original dataset columns and appends skill-related outputs.

Typical added columns include:
- `lang_detected` — detected language of the job description
- `languages` — required languages found in the text
- `soft_skills_final` — normalized soft skills
- `technical_skills_final` — normalized technical skills
- `soft_skills_dict` — dictionary-matched soft skills
- `soft_skills_categories` — soft skills inferred from regex/context patterns
- `tech_keywords_regex` — technical skills found with regex
- `skill_spans` — extracted skill spans from transformer models
- `skill_spans_soft` — soft-related spans

Count columns may also be added, such as:
-`soft_skills_count_dict`
- `soft_skills_count_final`
-`span_soft_count`
- `span_tech_count`

## What Results This Script Produces

After processing, the dataset can be used to answer questions such as:
- Which technical skills are most frequently required?
- Which soft skills appear most often in job descriptions?
- How often do employers require German and/or English?
- How do skill requirements vary across roles or companies?
- What patterns appear in the IT job market?

At the end of the run, the script also prints summary statistics such as:
- top soft skills
- top technical skills
- top required languages

## Example Use Cases

This script can be used for:
- analyzing skill demand in IT job postings
- comparing technical vs soft skill requirements
- preparing data for dashboards or BI reports
- creating structured features for machine learning or NLP tasks
- supporting academic research on labor market trends

## Performance Notes
### CPU mode
The default setting is:
```
device=-1
```
This runs the pipeline on CPU and is the most stable option for most environments.

### GPU mode
If you have a CUDA-compatible GPU, you can try:
```
device=0
```
This may speed up transformer inference.

### Multiprocessing
Increasing `num_workers` may improve speed, but it also increases memory usage because each worker may load its own model instance.

Recommended:
- use `num_workers`=1 for stability
- increase only if your machine has enough RAM

## Common Issues
`ModuleNotFoundError`
Install missing packages:
```
pip install pandas torch transformers tqdm langdetect
```

## Model download fails

Check your internet connection and verify that Hugging Face downloads are allowed in your environment.

## File not found

Make sure the `csv_file` path is correct.

## Missing `description` column

Your CSV must contain a `description` column, or the script must be adjusted to use a different text column.

## The run is interrupted

Simply rerun the script. The checkpointing system should allow it to continue from the last saved batch.

## Project Structure
Example project layout:
```
project/
│
├── A_test-skills-3-with-checkpoint.py
├── combined_jobs.csv
├── jobs_with_skills_extracted_full.csv
├── jobs_with_skills_extracted_full.json
├── skill_extraction_checkpoint.db
└── checkpoint_backups/
```

## Minimal Example
```
if __name__ == "__main__":
    df_result = analyze_dataset_with_checkpoint(
        csv_file="combined_jobs.csv",
        output_csv="jobs_with_skills_extracted_full.csv",
        output_json="jobs_with_skills_extracted_full.json",
        db_path="skill_extraction_checkpoint.db",
        device=-1,
        batch_size=100,
        sample_size=None,
        num_workers=1,
    )
```
Run with:
```
python A_test-skills-3-with-checkpoint.py
```

## Why This Project Is Useful in a Portfolio
This project demonstrates practical skills in:
- multilingual NLP
- hybrid information extraction
- regex and dictionary-based text processing
- transformer-based token classification
- skill normalization
- batch processing for large datasets
- checkpointing and recovery design
- structured data export for downstream analytics

As a result, it works well not only as a functional extraction tool, but also as a portfolio project that shows applied NLP and data engineering skills on real-world text data.