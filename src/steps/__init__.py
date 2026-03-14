"""Step modules for the 8-step end-to-end pipeline.

Step interface contract
-----------------------
Every step is a function with this signature::

    def run_<name>(state: PipelineState, cfg: dict) -> None:
        ...

Steps mutate ``state`` in place and return nothing. They log their progress
using ``logging.getLogger("pipeline.<step_name>")``.

Step registry (execution order)
---------------------------------
1. ingest         — load source CSVs, normalize schema, parse dates
2. prepare        — validate input, parse locations, normalize titles
3. deduplicate    — URL dedup, composite dedup, description grouping, MinHash near-dup
4. regex_extract  — deterministic regex field extraction (salary, experience, etc.)
5. extract        — LLM semantic extraction via DeepSeek (description-grouped)
6. validate       — post-extraction validators, corrections, quality report
7. clean_enrich   — data cleaning, city normalization, skill/soft-skill enrichment
8. export         — column ordering, invariant checks, CSV output, cost report
"""
