"""Hermes Training APIs: strategy save, prompt overlay, experiment open."""

from __future__ import annotations

from fastapi.testclient import TestClient

from web.backend.app import app


def test_strategy_prompt_and_experiment_api(tmp_path, monkeypatch):
    monkeypatch.delenv("TRADINGCREW_STRATEGY_PATH", raising=False)
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    client = TestClient(app)

    got = client.get("/api/strategy?book=equity")
    assert got.status_code == 200
    body = got.json()
    assert body["strategy_edit"]["scientific_method"]["min_reviews_to_change"] >= 1
    assert any(p["key"] == "trader" for p in body["prompts"])

    edit = body["strategy_edit"]
    edit["mandate"] = "API-edited mandate."
    saved = client.put("/api/strategy?book=equity", json=edit)
    assert saved.status_code == 200, saved.text
    assert saved.json()["strategy_source"] == "live"
    assert saved.json()["mandate"] == "API-edited mandate."

    prompt = client.put(
        "/api/strategy/prompts?book=equity",
        json={"key": "trader", "goal": "Size from evidence.", "backstory": "Operator overlay."},
    )
    assert prompt.status_code == 200, prompt.text
    assert prompt.json()["overlay"] is True

    bad = client.put(
        "/api/strategy/prompts?book=equity",
        json={"key": "not_a_desk", "goal": "x", "backstory": "y"},
    )
    assert bad.status_code == 400

    opened = client.post(
        "/api/strategy/experiment?book=equity",
        json={"action": "open", "knob": "debate_rounds_1"},
    )
    assert opened.status_code == 200, opened.text
    assert opened.json()["experiment"]["knob"] == "debate_rounds_1"

    closed = client.post(
        "/api/strategy/experiment?book=equity",
        json={"action": "close"},
    )
    assert closed.status_code == 200
    assert closed.json()["experiment"] is None
