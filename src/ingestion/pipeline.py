"""Ingestion pipeline orchestrator.

Reads settings from ingestion/config/settings.yaml, loads all source CSVs,
normalizes dates, fills missing values, and writes combined_jobs.csv.

Deduplication and location filtering are handled in the extraction pipeline.

Entry point: run_pipeline(settings_path=None)
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from ingestion.date_parser import normalize_date_posted
from ingestion.loader import (
    TARGET_COLUMNS,
    SourceConfig,
    ensure_string_type,
    fill_missing_values,
    load_and_normalize_source,
)
from shared.io import write_csv_safe
from shared.schemas import ingestion_output_schema

logger = logging.getLogger("pipeline.ingestion")

_INGESTION_ROOT = Path(__file__).parent
_REPO_ROOT = _INGESTION_ROOT.parent.parent   # src/ingestion/ → src/ → repo root
_DEFAULT_SETTINGS = _INGESTION_ROOT / "config" / "settings.yaml"


def _load_settings(settings_path: Path | None = None) -> dict:
    """Load ingestion settings from YAML.

    Args:
        settings_path: Path to settings.yaml. Defaults to ingestion/config/settings.yaml.

    Returns:
        Settings dict.
    """
    path = settings_path or _DEFAULT_SETTINGS
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_source_configs(settings: dict, raw_data_dir: Path) -> list[SourceConfig]:
    """Construct SourceConfig objects from settings dict.

    Args:
        settings: Loaded settings dict.
        raw_data_dir: Directory containing raw CSV files.

    Returns:
        List of SourceConfig objects.
    """
    configs = []
    for src in settings.get("sources", []):
        configs.append(SourceConfig(
            name=src["name"],
            file_path=raw_data_dir / src["file"],
            site_label=src["site_label"],
            column_mapping=src["column_mapping"],
        ))
    return configs


def run_pipeline(
    settings_path: Path | None = None,
    cfg: dict | None = None,
) -> Path:
    """Execute the full ingestion pipeline.

    Reads settings.yaml, loads source CSVs, normalizes dates, fills missing
    values, and writes combined_jobs.csv + processing_log.json.

    When ``cfg`` is provided (orchestrator mode), uses:
      - ``cfg["paths"]["ingestion_output"]`` for the output CSV path

    Falls back to ingestion/config/settings.yaml paths when ``cfg`` is None
    (standalone mode). Either way, settings.yaml is still loaded for
    ``scrape_date`` and source column mappings, which are ingestion-specific
    and not present in the orchestrator config.

    Args:
        settings_path: Optional path to settings.yaml (defaults to config/settings.yaml).
        cfg: Orchestrator pipeline config dict. When provided, controls output path.

    Returns:
        Path to the written combined_jobs.csv.
    """
    settings = _load_settings(settings_path)
    run_timestamp = datetime.now().isoformat()

    raw_data_dir = (_REPO_ROOT / settings["paths"]["raw_data_dir"]).resolve()
    scrape_date = datetime.fromisoformat(settings["scrape_date"])

    if cfg is not None:
        output_path = Path(cfg["paths"]["ingestion_output"]).resolve()
        output_dir = output_path.parent
    else:
        output_dir = (_REPO_ROOT / settings["paths"]["output_dir"]).resolve()
        output_path = output_dir / "combined_jobs.csv"

    source_configs = _build_source_configs(settings, raw_data_dir)

    # Step 1: Load and normalize each source
    source_dfs = []
    for src_cfg in source_configs:
        df = load_and_normalize_source(src_cfg)
        source_dfs.append(df)

    # Step 2: Merge all sources
    logger.info("=" * 70)
    logger.info("Merging %d sources", len(source_dfs))
    logger.info("=" * 70)

    string_cols = TARGET_COLUMNS
    normalized_dfs = [ensure_string_type(df, string_cols) for df in source_dfs]

    final_df = pd.concat(normalized_dfs, ignore_index=True)
    logger.info("Combined records: %d", len(final_df))

    # Step 3: Normalize dates
    final_df = normalize_date_posted(final_df, reference_date=scrape_date)
    logger.info("Normalized date_posted (reference: %s).", scrape_date.date())

    # Step 4: Fill missing values
    final_df = fill_missing_values(final_df)

    # Fill missing date_posted with a fixed sentinel date
    if "date_posted" in final_df.columns:
        final_df["date_posted"] = final_df["date_posted"].fillna("2027-01-01")
        logger.info("Filled missing date_posted with sentinel date: 2027-01-01.")

    logger.info("Final row count: %d", len(final_df))

    # Step 5: Schema validation — catch column/type drift before writing
    try:
        ingestion_output_schema.validate(final_df)
        logger.info("Schema validation passed (%d rows)", len(final_df))
    except Exception as exc:
        logger.warning("Schema validation failed: %s", exc)
        raise

    # Step 6: Save output
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_safe(final_df, output_path)
    logger.info("Saved combined jobs to: %s", output_path)

    report = {
        "run_timestamp": run_timestamp,
        "summary": {
            "final_output_rows": len(final_df),
            "reference_date": scrape_date.isoformat(),
            "output_file": output_path.name,
        },
    }
    log_path = output_dir / "processing_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Saved processing log to: %s", log_path)

    return output_path
