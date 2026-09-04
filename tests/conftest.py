"""Test fixtures shared across the suite."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _packaged_strategy_unless_overridden(monkeypatch):
    """Keep unit tests on the checked-in yaml unless a test sets its own path."""
    monkeypatch.setenv(
        "TRADINGCREW_STRATEGY_PATH",
        str(
            Path(__file__).resolve().parents[1]
            / "trading_crew"
            / "config"
            / "desk_strategy.yaml"
        ),
    )


@pytest.fixture(autouse=True)
def _clear_tool_memoize_cache():
    """Reset the per-process tool-result cache between every test.

    ``trading_crew.tools`` ships a tiny TTL cache (default 10 min) that
    dedups repeated calls to the same tool with the same arguments
    inside a single workflow.  In production that's purely a latency
    optimisation, but in the test suite it causes spooky-action: test
    A stubs ``get_corporate_actions`` with one set of fake Tavily hits,
    test B stubs it with a *different* set, and test B sees test A's
    cached payload through.  Clearing the cache at the boundary makes
    every test independent again.
    """
    from trading_crew import tools as tools_mod

    with tools_mod._tool_cache_lock:
        tools_mod._tool_cache.clear()
    yield
    with tools_mod._tool_cache_lock:
        tools_mod._tool_cache.clear()
