"""Consolidated pipeline configuration loader.

Single entry point that merges extraction and ingestion settings into one dict.

Usage::

    from shared.config import load_pipeline_config
    cfg = load_pipeline_config()
    cfg["paths"]["extracted_dir"]   # Path object
    cfg["ingestion"]["scrape_date"] # "2026-02-15"

All paths are resolved relative to the repo root via ``Path(__file__)``,
so callers are CWD-agnostic.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger("pipeline.config")

_SRC_ROOT = Path(__file__).parent.parent    # src/shared/ → src/
_REPO_ROOT = _SRC_ROOT.parent               # src/ → repo root
_DEFAULT_CONFIG_PATH = _SRC_ROOT / "extraction" / "config" / "settings.yaml"
_REQUIRED_KEYS = ["paths", "input_files", "extraction", "validation"]


@dataclass
class ExtractionConfig:
    """Validated extraction settings parsed from the 'extraction' config section.

    Attributes:
        model: DeepSeek model identifier.
        batch_size: Number of rows processed per async batch (1–10000).
        max_retries: Maximum retries per row on transient failures (1–20).
        temperature: Sampling temperature passed to the API (0.0–2.0).
    """

    model: str
    batch_size: int
    max_retries: int = 3
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if not 1 <= self.batch_size <= 10000:
            raise ValueError(f"batch_size must be 1–10000, got {self.batch_size}")
        if not 1 <= self.max_retries <= 20:
            raise ValueError(f"max_retries must be 1–20, got {self.max_retries}")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be 0.0–2.0, got {self.temperature}")


def _create_directories(cfg: dict) -> None:
    """Create all data directories defined in config paths (except checkpoint_db)."""
    _file_paths = {"checkpoint_db", "ingestion_output"}
    for name, path in cfg["paths"].items():
        if name in _file_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)


def _load_extraction_config(config_path: Path = _DEFAULT_CONFIG_PATH) -> dict:
    """Load settings.yaml, convert path strings to Path objects, and create directories.

    Args:
        config_path: Path to the YAML settings file. Defaults to config/settings.yaml.

    Returns:
        Config dict with all path strings under 'paths' converted to pathlib.Path objects.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If a required top-level key is missing from the config.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for key in _REQUIRED_KEYS:
        if key not in cfg:
            raise ValueError(
                f"Missing required config key: '{key}'. "
                f"Check {config_path} for all required sections: {_REQUIRED_KEYS}"
            )

    for name, raw_path in cfg["paths"].items():
        p = Path(raw_path)
        cfg["paths"][name] = (_REPO_ROOT / p).resolve() if not p.is_absolute() else p

    _create_directories(cfg)

    ext = cfg["extraction"]
    cfg["extraction_config"] = ExtractionConfig(
        model=ext.get("model", ""),
        batch_size=int(ext.get("batch_size", 1)),
        max_retries=int(ext.get("max_retries_per_row", 3)),
        temperature=float(ext.get("temperature", 0.0)),
    )

    n_files = len(cfg.get("input_files", []))
    logger.debug(
        "Config loaded: %d input files, model=%s, batch_size=%s",
        n_files,
        ext.get("model", "unknown"),
        ext.get("batch_size", "unknown"),
    )

    return cfg


def load_pipeline_config() -> dict:
    """Load and merge extraction + ingestion configuration.

    1. Loads the extraction config from ``src/extraction/config/settings.yaml``
       (paths resolved relative to the repo root).
    2. Loads ``src/ingestion/config/settings.yaml`` (path-independent).
    3. Returns a merged config dict with ``cfg["ingestion"]`` key added.

    Returns:
        Merged config dict. ``cfg["paths"]`` contains Path objects.
        ``cfg["ingestion"]`` contains raw ingestion settings (not converted).
    """
    cfg = _load_extraction_config()

    # Overlay ingestion settings
    ingestion_settings_path = _SRC_ROOT / "ingestion" / "config" / "settings.yaml"
    if ingestion_settings_path.exists():
        with open(ingestion_settings_path, encoding="utf-8") as f:
            cfg["ingestion"] = yaml.safe_load(f)
    else:
        logger.warning("Ingestion settings not found at %s", ingestion_settings_path)
        cfg["ingestion"] = {}

    logger.debug("Pipeline config loaded (ingestion + extraction merged)")
    return cfg


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

_REQUIRED_NESTED: list[tuple[str, str]] = [
    ("paths", "raw_dir"),
    ("paths", "checkpoint_db"),
    ("paths", "reports_dir"),
    ("paths", "deduped_dir"),
    ("paths", "extracted_dir"),
    ("paths", "ingestion_output"),
    ("validation", "min_description_length"),
    # export.output_path is optional — steps/export.py defaults to
    # data/cleaning/cleaned_jobs.csv if not set.
]
_REQUIRED_TOP: list[str] = ["input_files"]


def validate_config(cfg: dict) -> None:
    """Validate that all required orchestrator config keys are present.

    Called at startup before any step runs. Raises ValueError with a clear
    message naming the missing key and its location, so misconfiguration is
    caught immediately rather than failing mid-run after burning LLM credits.

    Args:
        cfg: Merged pipeline config dict from load_pipeline_config().

    Raises:
        ValueError: If any required key is missing.
    """
    for key in _REQUIRED_TOP:
        if key not in cfg:
            raise ValueError(
                f"Missing required config key: '{key}'. "
                "Check src/extraction/config/settings.yaml."
            )
    for section, key in _REQUIRED_NESTED:
        if section not in cfg or key not in cfg[section]:
            raise ValueError(
                f"Missing required config key: cfg['{section}']['{key}']. "
                f"Check src/extraction/config/settings.yaml under '{section}:'."
            )
