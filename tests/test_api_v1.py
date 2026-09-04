"""Programmatic /api/v1 Prediction API."""

from __future__ import annotations

from typing import List

import pytest
from fastapi.testclient import TestClient

from trading_crew.agentic.runs import RunRecord, write_run
from trading_crew.api_run import ApiRunConfigError, build_session_cfg, load_api_run_config
from web.backend import app as backend_app
from web.backend import session_manager as sm_module


@pytest.fixture
def isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    return tmp_path


def test_packaged_api_run_config_loads():
    cfg = load_api_run_config()
    assert cfg["asset_class"] == "stock"
    assert cfg["run_mode"] in {"autonomous", "deterministic", "harness"}
    assert isinstance(cfg["tools_enabled"], dict)
    assert cfg["llm_preset"]


def test_api_run_llm_preset_comes_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRADINGCREW_LLM_PRESET", "hosted-vllm")
    assert load_api_run_config()["llm_preset"] == "hosted-vllm"
    monkeypatch.setenv("TRADINGCREW_LLM_PRESET", "local-vllm")
    assert load_api_run_config()["llm_preset"] == "local-vllm"


def test_build_session_cfg_requires_ticker_and_rounds():
    with pytest.raises(ApiRunConfigError, match="ticker"):
        build_session_cfg(ticker="  ", debate_rounds=2, risk_rounds=1)
    cfg = load_api_run_config()
    if cfg["run_mode"] == "harness":
        payload = build_session_cfg(ticker="ntnx", debate_rounds=2, risk_rounds=1)
        assert payload["ticker"] == "NTNX"
        assert "debate_rounds" not in payload
        assert payload["tools_enabled"]["market_analyst"]
        return
    with pytest.raises(ApiRunConfigError, match="debate_rounds"):
        build_session_cfg(ticker="NTNX", debate_rounds=0, risk_rounds=1)
    payload = build_session_cfg(ticker="ntnx", debate_rounds=2, risk_rounds=1)
    assert payload["ticker"] == "NTNX"
    assert payload["debate_rounds"] == 2
    assert payload["risk_rounds"] == 1
    assert payload["tools_enabled"]["market_analyst"]


def test_build_session_cfg_harness_omits_rounds(monkeypatch):
    from trading_crew import api_run

    base = load_api_run_config()
    monkeypatch.setattr(api_run, "load_api_run_config", lambda: {**base, "run_mode": "harness"})
    payload = build_session_cfg(ticker="NTNX", debate_rounds=None, risk_rounds=None)
    assert payload["run_mode"] == "harness"
    assert "debate_rounds" not in payload
    assert "risk_rounds" not in payload
    ignored = build_session_cfg(ticker="NTNX", debate_rounds=8, risk_rounds=8)
    assert "debate_rounds" not in ignored
    assert "risk_rounds" not in ignored


def test_missing_config_key_is_an_error(monkeypatch, tmp_path):
    path = tmp_path / "api_run.yaml"
    path.write_text("run_mode: autonomous\n", encoding="utf-8")
    monkeypatch.setenv("TRADINGCREW_API_CONFIG", str(path))
    with pytest.raises(ApiRunConfigError, match="missing keys"):
        load_api_run_config()


class FakeRunner:
    def __init__(self, loop):
        self._loop = loop
        self.events = __import__("asyncio").Queue()
        self._cancelled = __import__("asyncio").Event()

    def cancel(self):
        self._cancelled.set()

    async def run(self, ticker, cfg):
        FakeRunner.started.append({"ticker": ticker, "cfg": cfg})
        await self.events.put({"type": "run_completed"})


FakeRunner.started: List[dict] = []


@pytest.fixture
def api_client(monkeypatch, tmp_path):
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


def test_v1_health_and_config(api_client):
    health = api_client.get("/api/v1/health")
    assert health.status_code == 200
    body = health.json()
    assert body["ok"] is True
    assert body["api"] == "v1"
    assert body["config_present"] is True
    cfg = api_client.get("/api/v1/config")
    assert cfg.status_code == 200
    assert "agents.yaml" in cfg.json()["prompts"]["agents"]
    assert "tasks.yaml" in cfg.json()["prompts"]["tasks"]
    assert cfg.json()["config"]["run_mode"] in {"autonomous", "deterministic", "harness"}
    assert cfg.json()["config"]["llm_preset"] == load_api_run_config()["llm_preset"]


def test_v1_start_status_stop(api_client):
    missing = api_client.post("/api/v1/runs", json={"debate_rounds": 2, "risk_rounds": 1})
    assert missing.status_code == 422
    started = api_client.post(
        "/api/v1/runs",
        json={"ticker": "INFY.NS", "debate_rounds": 2, "risk_rounds": 1, "force_rerun": True},
    )
    assert started.status_code == 200, started.text
    body = started.json()
    assert body["cached"] is False
    assert body["ticker"] == "INFY.NS"
    run_id = body["run_id"]
    started_cfg = FakeRunner.started[0]["cfg"]
    if started_cfg["run_mode"] == "harness":
        assert "debate_rounds" not in body
        assert "debate_rounds" not in started_cfg
    else:
        assert body["debate_rounds"] == 2
        assert body["risk_rounds"] == 1
        assert started_cfg["debate_rounds"] == 2
    assert started_cfg["llm_preset"]

    listed = api_client.get("/api/v1/runs")
    assert listed.status_code == 200
    ids = {row["run_id"] for row in listed.json()["runs"]}
    assert run_id in ids

    status = api_client.get(f"/api/v1/runs/{run_id}")
    assert status.status_code == 200
    assert status.json()["source"] == "live"
    assert status.json()["ticker"] == "INFY.NS"

    stopped = api_client.post(f"/api/v1/runs/{run_id}/stop")
    assert stopped.status_code in {200, 404}


def test_v1_result_from_persisted_run(api_client):
    rec = RunRecord(
        run_id="r-api-1",
        ticker="NTNX",
        started_at="2026-08-23T10:00:00+00:00",
        completed_at="2026-08-23T10:12:00+00:00",
        status="completed",
        config={"asset_class": "stock"},
        final_decision={"action": "BUY", "size_pct_of_book": 0.04},
        reports={"News Analyst": "news body"},
        reports_by_round={
            "News Analyst": [
                {"round": 1, "total_rounds": 1, "output": "news body", "degraded": False},
            ],
        },
    )
    write_run(rec)
    missing = api_client.get("/api/v1/runs/no-such-run")
    assert missing.status_code == 404
    status = api_client.get("/api/v1/runs/r-api-1")
    assert status.status_code == 200
    assert status.json()["source"] == "persisted"
    assert status.json()["ticker"] == "NTNX"
    result = api_client.get("/api/v1/runs/r-api-1/result")
    assert result.status_code == 200
    assert result.json()["final_decision"]["action"] == "BUY"
    assert result.json()["reports"]["News Analyst"] == ""
    reports = api_client.get("/api/v1/runs/r-api-1/reports", params={"role": "News Analyst"})
    assert reports.status_code == 200
    assert reports.json()["reports"]["News Analyst"] == "news body"
