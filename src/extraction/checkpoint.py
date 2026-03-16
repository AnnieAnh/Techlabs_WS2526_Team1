"""SQLite-backed pipeline state tracker. Makes every stage interruptible and resumable."""

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("pipeline.checkpoint")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    file_path   TEXT PRIMARY KEY,
    status      TEXT DEFAULT 'pending',
    row_count   INTEGER,
    loaded_at   TIMESTAMP
);

CREATE TABLE IF NOT EXISTS rows (
    row_id        TEXT PRIMARY KEY,
    file_path     TEXT,
    stage         TEXT DEFAULT 'loaded',
    status        TEXT DEFAULT 'pending',
    error_message TEXT,
    updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class Checkpoint:
    """SQLite-backed state tracker for the extraction pipeline.

    Tracks per-file and per-row state across interrupted runs.
    All writes use SQLite's atomic commit semantics — safe against mid-write crashes.
    """

    def __init__(self, db_path: str | Path) -> None:
        """Open (or create) the checkpoint database and ensure all tables exist.

        Args:
            db_path: Path to the SQLite database file.
        """
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        existed = self._db_path.exists()
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

        if existed:
            row_count = self._conn.execute("SELECT COUNT(*) FROM rows").fetchone()[0]
            last_updated = self._conn.execute(
                "SELECT MAX(updated_at) FROM rows"
            ).fetchone()[0]
            logger.info(
                "Checkpoint database opened: %d rows tracked, last updated %s",
                row_count,
                last_updated or "never",
            )
        else:
            logger.info("Checkpoint database created at %s", self._db_path)

    def register_file(self, file_path: str, row_count: int) -> None:
        """Register a CSV file in the files table (upsert).

        Args:
            file_path: Path to the source CSV file.
            row_count: Number of rows loaded from the file.
        """
        self._conn.execute(
            """
            INSERT INTO files (file_path, status, row_count, loaded_at)
            VALUES (?, 'loaded', ?, ?)
            ON CONFLICT(file_path) DO UPDATE SET
                status = 'loaded',
                row_count = excluded.row_count,
                loaded_at = excluded.loaded_at
            """,
            (str(file_path), row_count, _now()),
        )
        self._conn.commit()

    def register_rows(self, rows: list[dict]) -> None:
        """Bulk-register rows, skipping any that already exist.

        Args:
            rows: List of dicts with keys 'row_id' and 'file_path'.
        """
        self._conn.executemany(
            "INSERT OR IGNORE INTO rows (row_id, file_path) VALUES (:row_id, :file_path)",
            rows,
        )
        self._conn.commit()

    def advance_stage(self, row_id: str, stage: str) -> None:
        """Mark a row as having completed a pipeline stage.

        Args:
            row_id: The row identifier.
            stage: The stage name (e.g. 'validated', 'extracted').
        """
        self._conn.execute(
            "UPDATE rows SET stage = ?, status = 'completed', updated_at = ? WHERE row_id = ?",
            (stage, _now(), row_id),
        )
        self._conn.commit()
        logger.debug("Checkpoint: row %s -> %s", row_id, stage)

    def mark_failed(self, row_id: str, error_message: str) -> None:
        """Mark a row as permanently failed.

        Args:
            row_id: The row identifier.
            error_message: Description of the failure.
        """
        self._conn.execute(
            "UPDATE rows SET status = 'failed', error_message = ?, updated_at = ? WHERE row_id = ?",
            (error_message, _now(), row_id),
        )
        self._conn.commit()
        logger.warning("Row %s failed: %s", row_id, error_message)

    def mark_skipped(self, row_id: str) -> None:
        """Mark a row as skipped (e.g. privacy wall, duplicate).

        Args:
            row_id: The row identifier.
        """
        self._conn.execute(
            "UPDATE rows SET status = 'skipped', updated_at = ? WHERE row_id = ?",
            (_now(), row_id),
        )
        self._conn.commit()

    def get_pending(self, stage: str) -> list[str]:
        """Return row_ids that are pending for the given stage.

        Args:
            stage: The pipeline stage to query.

        Returns:
            List of row_id strings with status 'pending' at the given stage.
        """
        rows = self._conn.execute(
            "SELECT row_id FROM rows WHERE stage = ? AND status = 'pending'",
            (stage,),
        ).fetchall()
        return [r["row_id"] for r in rows]

    def get_completed(self, stage: str) -> list[str]:
        """Return row_ids that have completed the given stage.

        Args:
            stage: The pipeline stage to query.

        Returns:
            List of row_id strings with status 'completed' at the given stage.
        """
        rows = self._conn.execute(
            "SELECT row_id FROM rows WHERE stage = ? AND status = 'completed'",
            (stage,),
        ).fetchall()
        return [r["row_id"] for r in rows]

    def get_all_failed(self) -> list[str]:
        """Return all row_ids with status='failed', regardless of stage.

        Returns:
            List of all row_id strings that have permanently failed.
        """
        rows = self._conn.execute(
            "SELECT row_id FROM rows WHERE status = 'failed'"
        ).fetchall()
        return [r["row_id"] for r in rows]

    def get_failed(self, stage: str) -> list[str]:
        """Return row_ids that failed at the given stage.

        Args:
            stage: The pipeline stage to query.

        Returns:
            List of row_id strings with status 'failed' at the given stage.
        """
        rows = self._conn.execute(
            "SELECT row_id FROM rows WHERE stage = ? AND status = 'failed'",
            (stage,),
        ).fetchall()
        return [r["row_id"] for r in rows]

    def get_progress(self) -> dict[str, int]:
        """Return counts of completed rows per stage.

        Returns:
            Dict mapping stage name to count of completed rows at that stage.
        """
        rows = self._conn.execute(
            "SELECT stage, COUNT(*) as count FROM rows WHERE status = 'completed' GROUP BY stage"
        ).fetchall()
        progress = {r["stage"]: r["count"] for r in rows}
        logger.info("Pipeline progress: %s", progress)
        return progress

    def print_progress(self) -> None:
        """Pretty-print pipeline progress to the terminal and log file."""
        progress = self.get_progress()

        total = self._conn.execute("SELECT COUNT(*) FROM rows").fetchone()[0]
        skipped = self._conn.execute(
            "SELECT COUNT(*) FROM rows WHERE status = 'skipped'"
        ).fetchone()[0]
        failed = self._conn.execute(
            "SELECT COUNT(*) FROM rows WHERE status = 'failed'"
        ).fetchone()[0]

        lines = [
            "",
            "=" * 50,
            "  Pipeline Progress",
            "=" * 50,
            f"  Total rows registered : {total:>8,}",
            f"  Skipped               : {skipped:>8,}",
            f"  Failed                : {failed:>8,}",
            "-" * 50,
        ]
        for stage, count in sorted(progress.items()):
            lines.append(f"  {stage:<30} : {count:>8,}")
        lines.append("=" * 50)

        output = "\n".join(lines)
        logger.info(output)
        print(output)

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> "Checkpoint":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
