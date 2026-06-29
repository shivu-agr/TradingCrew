"""Regression tests for the ``clear_runs`` helper and DELETE endpoint.

The runs history is the user-facing recent-runs panel in the UI, so the
deletion path needs to:

* Move per-ticker records and the global index into a timestamped trash
  folder (so an accidental click is recoverable).
* Drop only the requested ticker when ``ticker`` is passed, leaving
  records for other tickers — and the index lines that reference them —
  intact.
* Re-create an empty ``runs/`` directory on a full clear so subsequent
  kickoffs don't crash trying to write into a missing tree.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trading_crew.agentic.runs import (
    RunRecord,
    clear_runs,
    list_recent_runs,
    write_run,
)


@pytest.fixture
def isolated_cache(monkeypatch, tmp_path):
    """Route ``TRADINGCREW_CACHE_DIR`` into a tmpdir for the test."""
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    return tmp_path


def _make_record(ticker: str, run_id: str) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        ticker=ticker.upper(),
        started_at=f"2026-01-{int(run_id[-2:]):02d}T10:00:00+00:00",
        completed_at=f"2026-01-{int(run_id[-2:]):02d}T10:05:00+00:00",
        status="completed",
        final_decision={"action": "NEUTRAL", "size_pct_of_book": 0.0},
    )


def test_clear_all_moves_runs_aside_and_resets_index(isolated_cache):
    write_run(_make_record("AAPL", "r01"))
    write_run(_make_record("MSFT", "r02"))
    write_run(_make_record("AAPL", "r03"))

    assert len(list_recent_runs(limit=10)) == 3

    result = clear_runs(ticker=None)

    assert result["removed"] == 3
    assert result["trash_path"], "full-clear must report a trash path"

    trash_dir = Path(result["trash_path"])
    assert trash_dir.exists(), "the runs tree should be preserved in trash, not deleted"
    assert (trash_dir / "AAPL").exists()
    assert (trash_dir / "MSFT").exists()
    assert (trash_dir / "index.jsonl").exists()

    runs_dir = isolated_cache / "runs"
    assert runs_dir.exists(), "runs/ should be re-created so kickoffs can keep writing"
    assert list_recent_runs(limit=10) == []

    write_run(_make_record("AAPL", "r04"))
    assert len(list_recent_runs(limit=10)) == 1


def test_clear_by_ticker_drops_only_that_ticker(isolated_cache):
    write_run(_make_record("AAPL", "r01"))
    write_run(_make_record("MSFT", "r02"))
    write_run(_make_record("AAPL", "r03"))
    write_run(_make_record("GOOG", "r04"))

    result = clear_runs(ticker="AAPL")

    assert result["removed"] == 2
    assert result["trash_path"], "per-ticker clear must report a trash path"
    trash = Path(result["trash_path"])
    assert trash.exists()
    assert ".trash" in trash.parts

    remaining = list_recent_runs(limit=10)
    tickers = {entry["ticker"] for entry in remaining}
    assert tickers == {"MSFT", "GOOG"}, f"unexpected survivors: {tickers}"

    runs_dir = isolated_cache / "runs"
    assert not (runs_dir / "AAPL").exists(), "AAPL directory should be gone"
    assert (runs_dir / "MSFT").exists()
    assert (runs_dir / "GOOG").exists()

    index_path = runs_dir / "index.jsonl"
    with open(index_path) as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert all(e["ticker"] != "AAPL" for e in lines)


def test_clear_runs_no_history_is_safe(isolated_cache):
    """Calling clear when no runs were ever written must not raise."""
    result = clear_runs(ticker=None)
    assert result == {"removed": 0, "trash_path": None}


def test_clear_runs_per_ticker_with_no_match_is_noop(isolated_cache):
    """Clearing a ticker that has no saved runs should report zero removed."""
    write_run(_make_record("AAPL", "r01"))
    result = clear_runs(ticker="TSLA")
    assert result["removed"] == 0
    assert result["trash_path"] is None
    assert len(list_recent_runs(limit=10)) == 1
