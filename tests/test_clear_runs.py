"""Regression tests for the ``clear_runs`` helper (now DB-backed).

Run history lives in the ``runs`` table of ``prediction.db``.  The deletion
path needs to:

* Drop only the requested ticker when ``ticker`` is passed, leaving records
  for other tickers intact.
* Clear everything when ``ticker`` is ``None``.
* Report the number of removed records so the UI can show a toast.
* Be safe to call when no runs were ever written.
"""

from __future__ import annotations

import pytest

from trading_crew.agentic.runs import (
    RunRecord,
    clear_runs,
    list_recent_runs,
    load_run,
    write_run,
)


@pytest.fixture
def isolated_cache(monkeypatch, tmp_path):
    """Route ``TRADINGCREW_CACHE_DIR`` into a tmpdir for the test (the DB and
    all state land there)."""
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


def test_clear_all_removes_every_run(isolated_cache):
    write_run(_make_record("AAPL", "r01"))
    write_run(_make_record("MSFT", "r02"))
    write_run(_make_record("AAPL", "r03"))

    assert len(list_recent_runs(limit=10)) == 3

    result = clear_runs(ticker=None)

    assert result["removed"] == 3
    assert result["trash_path"] is None  # DB delete — no filesystem trash
    assert list_recent_runs(limit=10) == []

    # subsequent kickoffs still write cleanly
    write_run(_make_record("AAPL", "r04"))
    assert len(list_recent_runs(limit=10)) == 1


def test_clear_by_ticker_drops_only_that_ticker(isolated_cache):
    write_run(_make_record("AAPL", "r01"))
    write_run(_make_record("MSFT", "r02"))
    write_run(_make_record("AAPL", "r03"))
    write_run(_make_record("GOOG", "r04"))

    result = clear_runs(ticker="AAPL")

    assert result["removed"] == 2
    remaining = list_recent_runs(limit=10)
    tickers = {entry["ticker"] for entry in remaining}
    assert tickers == {"MSFT", "GOOG"}, f"unexpected survivors: {tickers}"


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


def _record_with_reports(ticker: str = "AAPL", run_id: str = "r10") -> RunRecord:
    rec = _make_record(ticker, run_id)
    rec.reports = {
        "News Analyst": "news body",
        "Bullish Researcher": "bull r2",
    }
    rec.reports_by_round = {
        "News Analyst": [
            {"round": 1, "total_rounds": 1, "output": "news body", "degraded": False},
        ],
        "Bullish Researcher": [
            {"round": 1, "total_rounds": 2, "output": "bull r1", "degraded": False},
            {"round": 2, "total_rounds": 2, "output": "bull r2", "degraded": False},
        ],
    }
    return rec


def test_run_report_bodies_stay_out_of_lite_load(isolated_cache):
    from web.backend import prediction_db as db

    write_run(_record_with_reports())
    listed = list_recent_runs(limit=10)
    assert listed[0]["run_id"] == "r10"
    assert "reports" not in listed[0]

    lite = load_run("r10", "AAPL")
    assert lite["reports"]["News Analyst"] == ""
    news = lite["reports_by_round"]["News Analyst"][0]
    assert news["has_body"] is True
    assert news["output"] == ""
    bull = lite["reports_by_round"]["Bullish Researcher"]
    assert len(bull) == 2
    assert all(entry["output"] == "" and entry["has_body"] for entry in bull)

    full = load_run("r10", "AAPL", full=True)
    assert full["reports"]["News Analyst"] == "news body"
    assert full["reports_by_round"]["Bullish Researcher"][0]["output"] == "bull r1"
    assert full["reports_by_round"]["Bullish Researcher"][1]["output"] == "bull r2"

    one = db.get_run_reports("r10", "AAPL", role="News Analyst")
    assert one["reports"]["News Analyst"] == "news body"
    assert one["reports_by_round"]["News Analyst"][0]["output"] == "news body"
    assert "Bullish Researcher" not in one["reports"]


def test_legacy_fat_data_json_explodes_on_migrate(isolated_cache):
    import json

    from web.backend import prediction_db as db

    db.init_db()
    fat = {
        "run_id": "legacy-1",
        "ticker": "AAPL",
        "status": "completed",
        "reports": {"News Analyst": "legacy body"},
        "reports_by_round": {
            "News Analyst": [
                {"round": 1, "total_rounds": 1, "output": "legacy body", "degraded": False},
            ],
        },
    }
    conn, lock = db.shared_connection()
    with lock:
        conn.execute(
            """INSERT INTO runs
                 (run_id, ticker, started_at, completed_at, status,
                  final_action, final_size_pct, cascade_route, data_json)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            ("legacy-1", "AAPL", "2026-01-01T00:00:00+00:00", None, "completed",
             None, None, None, json.dumps(fat)),
        )
        conn.commit()
        db._explode_existing_run_bodies(conn)
        conn.commit()
    lite = db.get_run("legacy-1", "AAPL")
    assert lite["reports"]["News Analyst"] == ""
    assert lite["reports_by_round"]["News Analyst"][0]["has_body"] is True
    full = db.get_run("legacy-1", "AAPL", full=True)
    assert full["reports"]["News Analyst"] == "legacy body"
