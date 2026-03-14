"""Unified pipeline logger — three outputs: terminal, text file, JSONL.

Standard event names for structured logging (use in extra={"event": ...}):
    pipeline_start, pipeline_complete, pipeline_failed
    step_start, step_complete, step_failed
    ingest_complete, dedup_complete, location_parse_done
    title_normalize_done, extraction_complete, validation_done
    clean_enrich_done, export_done

Usage::

    # Once at startup (orchestrate.py):
    from shared.logging import setup_pipeline_logger
    setup_pipeline_logger()

    # In every module:
    import logging
    logger = logging.getLogger("pipeline.<module>")

    # Plain log:
    logger.info("Dedup complete: %d → %d", before, after)

    # Structured log (captured as JSON in pipeline.jsonl):
    logger.info(
        "Dedup complete: %d → %d (%d removed)",
        before, after, removed,
        extra={"event": "dedup_complete", "rows_before": before,
               "rows_after": after, "rows_removed": removed},
    )
"""

import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_DIR = Path("logs")
_LOG_PATH = _LOG_DIR / "pipeline.log"
_JSONL_PATH = _LOG_DIR / "pipeline.jsonl"

_TERMINAL_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_FILE_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-35s | %(message)s"
_TERMINAL_DATEFMT = "%H:%M:%S"
_FILE_DATEFMT = "%Y-%m-%d %H:%M:%S"

_MAX_BYTES = 10_000_000  # 10 MB
_BACKUP_COUNT = 5


class _JsonlFormatter(logging.Formatter):
    """Format log records as JSON Lines (one JSON object per line)."""

    # Standard LogRecord instance attributes — everything not in this set is
    # caller-provided extra context that we forward into the JSONL entry.
    _STANDARD_KEYS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)

    def format(self, record: logging.LogRecord) -> str:  # noqa: A002
        entry: dict = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Forward any extra fields the caller passed
        for key, val in record.__dict__.items():
            if key not in self._STANDARD_KEYS and not key.startswith("_"):
                entry[key] = val
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False, default=str)


def setup_pipeline_logger(log_dir: Path | None = None) -> logging.Logger:
    """Configure three-output logger: terminal (INFO+), file (DEBUG+), JSONL (DEBUG+).

    Call once at startup before any imports that log. All modules acquire a
    child logger via ``logging.getLogger("pipeline.<module>")``.

    Args:
        log_dir: Directory for log files. Defaults to ``logs/`` relative to CWD.

    Returns:
        The root ``pipeline`` logger.
    """
    root_logger = logging.getLogger("pipeline")
    if root_logger.handlers:
        return root_logger

    root_logger.setLevel(logging.DEBUG)

    resolved_dir = log_dir or _LOG_DIR
    resolved_dir.mkdir(parents=True, exist_ok=True)

    # — Terminal handler (INFO+, human-readable)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(_TERMINAL_FORMAT, datefmt=_TERMINAL_DATEFMT))

    # — Rotating text file (DEBUG+)
    file_handler = RotatingFileHandler(
        resolved_dir / "pipeline.log",
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_FILE_FORMAT, datefmt=_FILE_DATEFMT))

    # — Rotating JSONL file (DEBUG+)
    jsonl_handler = RotatingFileHandler(
        resolved_dir / "pipeline.jsonl",
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    jsonl_handler.setLevel(logging.DEBUG)
    jsonl_handler.setFormatter(_JsonlFormatter())

    root_logger.addHandler(console)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(jsonl_handler)

    return root_logger
