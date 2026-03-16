"""Dataset fingerprinting for reproducibility and cache invalidation.

Computes SHA-256 hashes of input CSV files and the merged pipeline config
so that any change in raw data or settings is detectable across runs.

Usage::

    from shared.fingerprint import fingerprint_inputs
    fp = fingerprint_inputs(cfg)
    # {"input_files": {"Raw_Jobs_INDEED.csv": "a1b2c3d4..."},
    #  "config_hash": "...", "timestamp": "..."}
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("pipeline.fingerprint")

_HASH_LENGTH = 16  # characters of hex digest to keep


def _file_hash(path: Path) -> str:
    """Compute SHA-256[:16] of a file, reading in 64 KiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:_HASH_LENGTH]


def _config_hash(cfg: dict) -> str:
    """Compute SHA-256[:16] of the serialized config dict.

    Excludes "paths" and "extraction_config" keys, as well as any
    remaining values that are Path objects (not JSON-serializable).
    """
    # Exclude Path objects (not JSON-serializable) and runtime-only keys
    serializable = {
        k: v for k, v in cfg.items()
        if k not in ("paths", "extraction_config") and not isinstance(v, Path)
    }
    blob = json.dumps(serializable, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()[:_HASH_LENGTH]


def fingerprint_inputs(cfg: dict) -> dict[str, Any]:
    """Compute fingerprints of all input CSV files and the pipeline config.

    Args:
        cfg: Pipeline config dict from ``shared.config.load_pipeline_config()``.
             Reads ``cfg["paths"]["raw_dir"]`` for input CSV discovery.

    Returns:
        Dict with keys:
        - ``input_files``: mapping of filename → SHA-256[:16] hash
        - ``config_hash``: SHA-256[:16] of the merged pipeline config
        - ``timestamp``: ISO-8601 UTC timestamp of when fingerprinting ran
    """
    raw_dir: Path | None = cfg.get("paths", {}).get("raw_dir")
    file_hashes: dict[str, str] = {}

    if raw_dir and Path(raw_dir).is_dir():
        csv_files = sorted(Path(raw_dir).glob("*.csv"))
        for csv_path in csv_files:
            try:
                file_hashes[csv_path.name] = _file_hash(csv_path)
                logger.debug("Fingerprinted %s → %s", csv_path.name, file_hashes[csv_path.name])
            except OSError as exc:
                logger.warning("Could not fingerprint %s: %s", csv_path.name, exc)
    else:
        logger.debug("raw_dir not found in cfg; skipping input file fingerprinting")

    result = {
        "input_files": file_hashes,
        "config_hash": _config_hash(cfg),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    logger.info(
        "Fingerprint: %d input files, config_hash=%s",
        len(file_hashes),
        result["config_hash"],
    )
    return result
