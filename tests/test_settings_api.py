"""Settings page + /api/settings. Existing /api/strategy* stay."""

from __future__ import annotations

from fastapi.testclient import TestClient

from web.backend.app import app


def _bulk() -> dict:
    return {
        "run_mode": "autonomous",
        "quality_review": False,
        "compliance": False,
        "critic_enabled": True,
        "fusion_enabled": False,
        "fusion_include_crowd": True,
        "tools_enabled": {"market": ["get_YFin_data"]},
        "book": "paper",
        "llm_preset": "local",
    }


def test_settings_page_serves_spa():
    client = TestClient(app)
    page = client.get("/settings")
    assert page.status_code == 200
    assert "text/html" in page.headers.get("content-type", "")


def test_settings_get_put_bulk_strategy_prompts(tmp_path, monkeypatch):
    monkeypatch.delenv("TRADINGCREW_STRATEGY_PATH", raising=False)
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    client = TestClient(app)

    got = client.get("/api/settings?book=equity")
    assert got.status_code == 200, got.text
    body = got.json()
    assert body["book"] == "equity"
    assert body["bulk"] is None
    assert "mandate" in body["strategy"]
    assert any(p["key"] == "trader" for p in body["prompts"])
    assert body["strategy_source"] in ("packaged", "live")

    incomplete = {k: _bulk()[k] for k in _bulk() if k != "llm_preset"}
    bad_bulk = client.put("/api/settings?book=equity", json={"bulk": incomplete})
    assert bad_bulk.status_code == 400
    assert client.get("/api/settings?book=equity").json()["bulk"] is None

    saved = client.put(
        "/api/settings?book=equity",
        json={
            "strategy": {**body["strategy"], "mandate": "Settings-edited mandate."},
            "prompts": [{
                "key": "trader",
                "goal": "Size from evidence.",
                "backstory": "Operator overlay.",
            }],
            "bulk": _bulk(),
        },
    )
    assert saved.status_code == 200, saved.text
    out = saved.json()
    assert out["strategy"]["mandate"] == "Settings-edited mandate."
    assert out["strategy_source"] == "live"
    assert out["bulk"]["run_mode"] == "autonomous"
    assert "debate_rounds" not in out["bulk"]
    assert "embedding_preset" not in out["bulk"]
    assert "memory" not in out["bulk"]
    trader = next(p for p in out["prompts"] if p["key"] == "trader")
    assert trader["overlay"] is True
    assert trader["goal"] == "Size from evidence."

    empty = client.put("/api/settings?book=equity", json={})
    assert empty.status_code == 200
    assert empty.json()["strategy"]["mandate"] == "Settings-edited mandate."

    unset = client.put(
        "/api/settings?book=equity",
        json={"unset_prompts": ["trader"]},
    )
    assert unset.status_code == 200
    trader = next(p for p in unset.json()["prompts"] if p["key"] == "trader")
    assert trader["overlay"] is False

    reverted = client.put("/api/settings?book=equity", json={"revert_strategy": True})
    assert reverted.status_code == 200, reverted.text
    assert reverted.json()["strategy_source"] == "packaged"
    assert "Settings-edited" not in reverted.json()["strategy"]["mandate"]

    commodity = client.get("/api/settings?book=commodity")
    assert commodity.status_code == 200
    assert commodity.json()["book"] == "commodity"
    assert commodity.json()["bulk"] is None
    assert any("role" in p for p in commodity.json()["prompts"])


def test_settings_save_blocked_when_env_path_set(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    client = TestClient(app)
    got = client.get("/api/settings?book=equity")
    assert got.status_code == 200
    blocked = client.put(
        "/api/settings?book=equity",
        json={"strategy": got.json()["strategy"]},
    )
    assert blocked.status_code == 400
    assert "TRADINGCREW_STRATEGY_PATH" in blocked.json()["detail"]


def test_strategy_endpoints_still_work(tmp_path, monkeypatch):
    monkeypatch.delenv("TRADINGCREW_STRATEGY_PATH", raising=False)
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    client = TestClient(app)

    got = client.get("/api/strategy?book=equity")
    assert got.status_code == 200
    edit = got.json()["strategy_edit"]
    saved = client.put("/api/strategy?book=equity", json=edit)
    assert saved.status_code == 200
    opened = client.post(
        "/api/strategy/experiment?book=equity",
        json={"action": "open", "knob": "debate_rounds_1"},
    )
    assert opened.status_code == 200
    assert opened.json()["experiment"]["knob"] == "debate_rounds_1"
