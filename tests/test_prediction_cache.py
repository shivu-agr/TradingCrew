"""Same-day Prediction-index cache (single + bulk skip + force rerun)."""

from __future__ import annotations

from typing import List

import pytest
from fastapi.testclient import TestClient

from trading_crew.agentic.runs import RunRecord, write_run
from web.backend import app as backend_app
from web.backend import session_manager as sm_module


@pytest.fixture
def isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    return tmp_path


def _completed_record(ticker: str = "BHARATFORG.NS", run_id: str = "r-cache-1") -> RunRecord:
    return RunRecord(
        run_id=run_id,
        ticker=ticker,
        started_at="2026-08-23T10:00:00+00:00",
        completed_at="2026-08-23T10:12:00+00:00",
        status="completed",
        config={"asset_class": "stock", "run_mode": "autonomous"},
        final_decision={"action": "BUY", "confidence": 0.72, "size_pct_of_book": 0.04},
        action_proposal={
            "side": "BUY",
            "conviction_score": 0.72,
            "target_weight": 0.04,
            "expected_return_pct": 0.08,
            "rationale": "test cache",
        },
        action_proposal_markdown="# BHARATFORG.NS — forward prediction",
    )


def test_write_run_indexes_today_as_done(isolated_cache):
    from web.backend import prediction_db as db

    write_run(_completed_record())
    today = db.prediction_day()
    pred = db.get_prediction("BHARATFORG.NS", today)
    assert pred is not None
    assert pred["status"] == "done"
    assert pred["action"] == "BUY"
    assert pred["run_id"] == "r-cache-1"
    assert "BHARATFORG.NS" in db.done_tickers(today)
    assert db.done_prediction("BHARATFORG.NS", today)["run_id"] == "r-cache-1"
    cell = db.load_collect_state(today).get("BHARATFORG.NS")
    assert cell and cell["status"] == "done"
    days = {d["run_date"] for d in db.list_run_days()}
    assert today in days


def test_failed_row_is_not_a_cache_hit(isolated_cache):
    from web.backend import prediction_db as db

    rec = _completed_record("AXISBANK.NS", "r-fail")
    rec.status = "error"
    rec.error = "llm timeout"
    rec.action_proposal = None
    rec.final_decision = None
    write_run(rec)
    today = db.prediction_day()
    row = db.get_prediction("AXISBANK.NS", today)
    assert row is not None
    assert row["status"] == "failed"
    assert db.done_prediction("AXISBANK.NS", today) is None
    assert db.cached_session_payload("AXISBANK.NS") is None


def test_five_tier_action_strips_enum_prefix():
    from web.backend.prediction_db import _five_tier_action

    assert _five_tier_action("ActionSide.BUY", 0.4) == "BUY"
    assert _five_tier_action("BUY", 0.4) == "BUY"


def test_day_snapshot_includes_run_id(isolated_cache):
    from web.backend import prediction_db as db

    write_run(_completed_record("INFY.NS", "r-infy"))
    today = db.prediction_day()
    snap = db.get_day_snapshot(today)
    names = [c for s in snap["sectors"] for c in s["companies"]]
    assert snap["done"] == 1
    assert names[0]["ticker"] == "INFY.NS"
    assert names[0]["run_id"] == "r-infy"


def test_commodity_run_is_not_indexed(isolated_cache):
    from web.backend import prediction_db as db

    rec = _completed_record("CL=F", "r-cmdty")
    rec.config = {"asset_class": "commodity"}
    write_run(rec)
    assert db.get_prediction("CL=F", db.prediction_day()) is None
    assert db.cached_session_payload("CL=F") is None


def test_force_rerun_disables_bulk_skip(isolated_cache):
    from web.backend import prediction_db as db
    from web.backend.scale_runner import _should_skip_name

    write_run(_completed_record("INFY.NS", "r-infy"))
    today = db.prediction_day()
    assert _should_skip_name(today, "INFY.NS", force_rerun=False) is True
    assert _should_skip_name(today, "INFY.NS", force_rerun=True) is False
    assert _should_skip_name(today, "TCS.NS", force_rerun=False) is False


class FakeRunner:
    def __init__(self, loop):
        self._loop = loop
        self.events = __import__("asyncio").Queue()
        self._cancelled = __import__("asyncio").Event()

    def cancel(self):
        self._cancelled.set()

    async def run(self, ticker, cfg):
        FakeRunner.started.append(ticker)
        await self.events.put({"type": "run_completed"})


FakeRunner.started: List[str] = []


@pytest.fixture
def session_client(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    sm_module.reset_session_manager_for_tests()
    FakeRunner.started = []

    original_init = sm_module.init_session_manager

    def patched_init(loop):
        sm = original_init(loop)
        sm._runner_factory = lambda l: FakeRunner(l)  # noqa: SLF001
        return sm

    monkeypatch.setattr(sm_module, "init_session_manager", patched_init)
    monkeypatch.setattr(backend_app, "init_session_manager", patched_init)
    with TestClient(backend_app.app) as c:
        yield c
    sm_module.reset_session_manager_for_tests()


def test_session_returns_cached_prediction(session_client, isolated_cache):
    write_run(_completed_record("RELIANCE.NS", "r-rel"))
    r = session_client.post("/api/sessions", json={"ticker": "RELIANCE.NS"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["cached"] is True
    assert body["ticker"] == "RELIANCE.NS"
    assert body["status"] == "completed"
    assert body["run_id"] == "r-rel"
    assert body["prediction"]["status"] == "done"
    assert FakeRunner.started == []


def test_session_force_rerun_starts_new_run(session_client, isolated_cache):
    write_run(_completed_record("RELIANCE.NS", "r-rel"))
    r = session_client.post(
        "/api/sessions",
        json={"ticker": "RELIANCE.NS", "force_rerun": True},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("cached") is not True
    assert "run_id" in body
    assert body["run_id"] != "r-rel"


def test_session_cache_skips_commodity(session_client, isolated_cache):
    write_run(_completed_record("CL=F", "r-cmdty"))
    r = session_client.post(
        "/api/sessions",
        json={"ticker": "CL=F", "asset_class": "commodity"},
    )
    assert r.status_code == 200, r.text
    assert r.json().get("cached") is not True


def test_run_api_is_lite_until_reports_fetched(session_client):
    rec = _completed_record("INFY.NS", "r-lite")
    rec.reports = {"News Analyst": "full news"}
    rec.reports_by_round = {
        "News Analyst": [
            {"round": 1, "total_rounds": 1, "output": "full news", "degraded": False},
        ],
    }
    write_run(rec)
    lite = session_client.get("/api/runs/INFY.NS/r-lite")
    assert lite.status_code == 200, lite.text
    body = lite.json()
    assert body["reports"]["News Analyst"] == ""
    assert body["reports_by_round"]["News Analyst"][0]["has_body"] is True
    assert body["reports_by_round"]["News Analyst"][0]["output"] == ""
    full = session_client.get("/api/runs/INFY.NS/r-lite", params={"full": "true"})
    assert full.json()["reports"]["News Analyst"] == "full news"
    reports = session_client.get(
        "/api/runs/INFY.NS/r-lite/reports", params={"role": "News Analyst"},
    )
    assert reports.status_code == 200
    assert reports.json()["reports"]["News Analyst"] == "full news"
