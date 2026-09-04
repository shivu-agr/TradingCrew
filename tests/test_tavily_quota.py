"""Tavily quota / 429 must fail once — no backoff retries, then skip."""

from __future__ import annotations

import pytest


def test_is_hard_quota_detects_usage_limit():
    from trading_crew.tools import _is_hard_quota

    assert _is_hard_quota(RuntimeError(
        "This request exceeds this API key's set usage limit."
    ))
    assert _is_hard_quota(RuntimeError("quota exceeded"))
    assert _is_hard_quota(RuntimeError("rate limit exceeded (429)"))

    class _HTTP(Exception):
        status_code = 429

    assert _is_hard_quota(_HTTP("nope"))
    assert not _is_hard_quota(RuntimeError("connection reset by peer"))


def test_with_retry_does_not_retry_quota():
    from trading_crew.tools import _with_retry

    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise RuntimeError("This request exceeds this API key's set usage limit.")

    with pytest.raises(RuntimeError, match="usage limit"):
        _with_retry(fn, attempts=3, base_delay=0.0)
    assert calls["n"] == 1


def test_tavily_news_rows_trips_process_flag(monkeypatch):
    from trading_crew import tools

    tools.reset_tavily_quota_flag()
    monkeypatch.setattr(tools, "_tavily", lambda: type("C", (), {
        "search": staticmethod(lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("This request exceeds this API key's set usage limit.")
        )),
    })())
    monkeypatch.setattr(
        "trading_crew.data_sources.news_providers.read_keys",
        lambda name: ["only-key"],
    )

    assert tools._tavily_news_rows("NTNX news") == []
    assert tools._tavily_quota_tripped is True
    # Second call must not touch the client again.
    calls = {"n": 0}

    def _boom(*a, **k):
        calls["n"] += 1
        raise AssertionError("should not be called after quota trip")

    monkeypatch.setattr(tools, "_tavily", lambda: type("C", (), {"search": _boom})())
    assert tools._tavily_news_rows("NTNX again") == []
    assert calls["n"] == 0
    tools.reset_tavily_quota_flag()
