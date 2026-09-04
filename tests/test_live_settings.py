"""live_settings: no invented bulk; missing keys fail."""

from __future__ import annotations

import pytest

from trading_crew.agentic.live_settings import (
    SettingsError,
    load_bulk,
    save_bulk,
    settings_book,
    settings_payload,
)


def test_settings_book_rejects_unknown():
    with pytest.raises(SettingsError, match="equity or commodity"):
        settings_book("fx")


def test_load_bulk_missing_is_none(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    assert load_bulk("equity") is None


def _autonomous_bulk() -> dict:
    return {
        "run_mode": "autonomous",
        "quality_review": False,
        "compliance": False,
        "critic_enabled": True,
        "fusion_enabled": False,
        "fusion_include_crowd": True,
        "tools_enabled": {},
        "book": "paper",
        "llm_preset": "local",
    }


def test_save_bulk_requires_every_key(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    with pytest.raises(SettingsError, match="missing required keys"):
        save_bulk("equity", {"debate_rounds": 2})
    assert load_bulk("equity") is None


def test_save_bulk_autonomous_drops_rounds_and_embedding(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    saved = save_bulk(
        "equity",
        {
            **_autonomous_bulk(),
            "debate_rounds": 5,
            "risk_rounds": 4,
            "embedding_preset": "vllm-embedding",
            "memory": True,
        },
    )
    assert "debate_rounds" not in saved
    assert "risk_rounds" not in saved
    assert "embedding_preset" not in saved
    assert "memory" not in saved
    assert saved["run_mode"] == "autonomous"
    assert "embedding_preset" not in load_bulk("equity")


def test_save_bulk_deterministic_requires_dialogue(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    with pytest.raises(SettingsError, match="missing required keys"):
        save_bulk("equity", {**_autonomous_bulk(), "run_mode": "deterministic"})
    saved = save_bulk(
        "equity",
        {**_autonomous_bulk(), "run_mode": "deterministic", "debate_rounds": 3, "risk_rounds": 2},
    )
    assert saved["debate_rounds"] == 3
    assert saved["risk_rounds"] == 2


def test_save_bulk_harness_drops_rounds(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    saved = save_bulk(
        "equity",
        {**_autonomous_bulk(), "run_mode": "harness", "debate_rounds": 5, "risk_rounds": 4},
    )
    assert saved["run_mode"] == "harness"
    assert "debate_rounds" not in saved
    assert "risk_rounds" not in saved


def test_load_bulk_strips_embedding_preset_from_disk(tmp_path, monkeypatch):
    import json

    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    from trading_crew.agentic.live_settings import bulk_path

    path = bulk_path()
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({
            "equity": {
                **_autonomous_bulk(),
                "embedding_preset": "vllm-embedding",
                "memory": True,
                "debate_rounds": 2,
            }
        }),
        encoding="utf-8",
    )
    loaded = load_bulk("equity")
    assert loaded is not None
    assert "embedding_preset" not in loaded
    assert "memory" not in loaded
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert "embedding_preset" not in on_disk["equity"]
    assert "memory" not in on_disk["equity"]


def test_settings_payload_has_no_bulk_until_saved(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    payload = settings_payload("equity")
    assert payload["bulk"] is None
    assert payload["strategy"]["mandate"]
    assert payload["prompts"]
