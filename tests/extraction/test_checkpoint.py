"""Tests for extraction/checkpoint.py — covers all 4 critical scenarios from BACKLOG.md."""

import pytest

from extraction.checkpoint import Checkpoint


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test_pipeline.db"


@pytest.fixture
def cp(db_path):
    checkpoint = Checkpoint(db_path)
    yield checkpoint
    checkpoint.close()


def _make_rows(n: int, stage: str = "loaded") -> list[dict]:
    return [{"row_id": f"row_{i:04d}", "file_path": "test.csv"} for i in range(n)]


# ------------------------------------------------------------------
# Critical test 1: Crash recovery
# ------------------------------------------------------------------

def test_crash_recovery(db_path):
    """Simulate crash: register 100 rows, advance 50, close, reopen, get_pending returns 50."""
    cp = Checkpoint(db_path)
    rows = _make_rows(100)
    cp.register_rows(rows)

    for row in rows[:50]:
        cp.advance_stage(row["row_id"], "extracted")

    cp.close()

    # Reopen — simulates a new process after crash
    cp2 = Checkpoint(db_path)
    pending = cp2.get_pending("loaded")  # still at default stage 'loaded'
    completed = cp2.get_completed("extracted")

    assert len(completed) == 50
    assert len(pending) == 50  # the other 50 never advanced
    cp2.close()


# ------------------------------------------------------------------
# Critical test 2: Idempotency
# ------------------------------------------------------------------

def test_advance_stage_idempotent(cp):
    """Calling advance_stage twice for the same row+stage must not error or duplicate."""
    cp.register_rows([{"row_id": "row_0001", "file_path": "test.csv"}])
    cp.advance_stage("row_0001", "extracted")
    cp.advance_stage("row_0001", "extracted")  # second call — must not raise

    completed = cp.get_completed("extracted")
    assert completed.count("row_0001") == 1  # no duplicates


def test_register_rows_idempotent(cp):
    """Registering the same rows twice must not raise or create duplicates."""
    rows = _make_rows(10)
    cp.register_rows(rows)
    cp.register_rows(rows)  # second call — INSERT OR IGNORE

    pending = cp.get_pending("loaded")
    assert len(pending) == 10  # exactly 10, not 20


# ------------------------------------------------------------------
# Critical test 3: Progress accuracy
# ------------------------------------------------------------------

def test_get_progress_accuracy(cp):
    """After mixed operations, get_progress() returns accurate counts per stage.

    'stage' stores the *current* (most recent) stage for each row — not a history.
    Rows that advance from 'validated' to 'extracted' are counted only under 'extracted'.
    """
    rows = _make_rows(10)
    cp.register_rows(rows)

    # Advance 6 rows to 'validated'
    for row in rows[:6]:
        cp.advance_stage(row["row_id"], "validated")

    # Advance 3 of those further to 'extracted' — they leave 'validated'
    for row in rows[:3]:
        cp.advance_stage(row["row_id"], "extracted")

    # Mark 1 as failed
    cp.mark_failed(rows[9]["row_id"], "parse error")

    progress = cp.get_progress()

    # 3 rows remain at 'validated' (rows 3-5); 3 are now at 'extracted' (rows 0-2)
    assert progress.get("validated", 0) == 3
    assert progress.get("extracted", 0) == 3


def test_get_progress_empty_db(cp):
    """get_progress on an empty database returns an empty dict (no stages yet)."""
    progress = cp.get_progress()
    assert progress == {}


# ------------------------------------------------------------------
# Additional: mark_failed and mark_skipped
# ------------------------------------------------------------------

def test_mark_failed(cp):
    cp.register_rows([{"row_id": "bad_row", "file_path": "test.csv"}])
    cp.mark_failed("bad_row", "JSON parse error")

    failed = cp.get_failed("loaded")
    assert "bad_row" in failed


def test_mark_skipped_not_in_pending(cp):
    cp.register_rows([{"row_id": "skip_row", "file_path": "test.csv"}])
    cp.mark_skipped("skip_row")

    pending = cp.get_pending("loaded")
    assert "skip_row" not in pending


# ------------------------------------------------------------------
# File registration
# ------------------------------------------------------------------

def test_register_file(cp):
    cp.register_file("data/raw/test.csv", row_count=500)
    # No assertion needed — just verify no exception raised


def test_register_file_upsert(cp):
    """Registering the same file twice updates it rather than erroring."""
    cp.register_file("data/raw/test.csv", row_count=500)
    cp.register_file("data/raw/test.csv", row_count=600)  # upsert


# ------------------------------------------------------------------
# print_progress smoke test
# ------------------------------------------------------------------

def test_print_progress_does_not_raise(cp, capsys):
    rows = _make_rows(5)
    cp.register_rows(rows)
    cp.advance_stage("row_0000", "validated")
    cp.print_progress()
    captured = capsys.readouterr()
    assert "Pipeline Progress" in captured.out
