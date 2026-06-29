"""Phase 2E — workflow improvements: analyst cache, per-agent LLM, retry."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Analyst output cache
# ---------------------------------------------------------------------------


def test_cache_key_is_stable(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    from web.backend.analyst_cache import make_cache_key

    k1 = make_cache_key(
        ticker="nvda",
        trade_date="2026-06-12",
        tools_enabled={"market_analyst": ["get_stock_data", "get_indicators"]},
        task_id="market_task",
    )
    k2 = make_cache_key(
        ticker="NVDA",
        trade_date="2026-06-12",
        tools_enabled={"market_analyst": ["get_indicators", "get_stock_data"]},  # reordered
        task_id="market_task",
    )
    assert k1.digest() == k2.digest()


def test_cache_key_changes_with_ticker(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    from web.backend.analyst_cache import make_cache_key

    k1 = make_cache_key(ticker="NVDA", trade_date="2026-06-12", tools_enabled={}, task_id="market_task")
    k2 = make_cache_key(ticker="AMD", trade_date="2026-06-12", tools_enabled={}, task_id="market_task")
    assert k1.digest() != k2.digest()


def test_save_and_load_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    from web.backend.analyst_cache import load_entry, make_cache_key, save_entry

    key = make_cache_key(
        ticker="NVDA", trade_date="2026-06-12",
        tools_enabled={"market_analyst": ["get_stock_data"]},
        task_id="market_task",
    )
    assert load_entry(key) is None
    save_entry(key, agent_role="Market Analyst", raw="**Hello world.**")
    entry = load_entry(key)
    assert entry is not None
    assert entry["raw"] == "**Hello world.**"
    assert entry["agent_role"] == "Market Analyst"


def test_clear_removes_all_entries(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    from web.backend.analyst_cache import clear, make_cache_key, save_entry

    for i in range(3):
        save_entry(
            make_cache_key(ticker=f"T{i}", trade_date="2026-06-12", tools_enabled={}, task_id="market_task"),
            agent_role="Market Analyst",
            raw=f"content {i}",
        )
    n = clear()
    assert n == 3


# ---------------------------------------------------------------------------
# Per-agent LLM overrides
# ---------------------------------------------------------------------------


def test_per_agent_overrides_invalid_json_falls_back_to_empty(monkeypatch):
    from trading_crew._common import _load_per_agent_overrides

    monkeypatch.setenv("LLM_PER_AGENT", "{not json")
    assert _load_per_agent_overrides() == {}


def test_per_agent_overrides_parses_valid_json(monkeypatch):
    from trading_crew._common import _load_per_agent_overrides

    monkeypatch.setenv("LLM_PER_AGENT", json.dumps({
        "social_analyst": {"model": "small-model", "temperature": 0.1},
        "research_manager": {"model": "long-context-model"},
    }))
    overrides = _load_per_agent_overrides()
    assert overrides["social_analyst"]["model"] == "small-model"
    assert overrides["research_manager"]["model"] == "long-context-model"


def test_per_agent_overrides_no_env_returns_empty(monkeypatch):
    from trading_crew._common import _load_per_agent_overrides

    monkeypatch.delenv("LLM_PER_AGENT", raising=False)
    assert _load_per_agent_overrides() == {}


# ---------------------------------------------------------------------------
# Tool retry with exponential backoff
# ---------------------------------------------------------------------------


def test_with_retry_returns_value_on_first_success():
    from trading_crew.tools import _with_retry

    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return 42

    assert _with_retry(fn, attempts=3, base_delay=0.0) == 42
    assert calls["n"] == 1


def test_with_retry_retries_then_succeeds():
    from trading_crew.tools import _with_retry

    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return "ok"

    out = _with_retry(fn, attempts=3, base_delay=0.0)
    assert out == "ok"
    assert calls["n"] == 3


def test_with_retry_raises_after_exhaustion():
    from trading_crew.tools import _with_retry

    def fn():
        raise RuntimeError("always fails")

    with pytest.raises(RuntimeError):
        _with_retry(fn, attempts=2, base_delay=0.0)
