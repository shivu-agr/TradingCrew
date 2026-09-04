"""Phase 1 — news-provider key rotation + Firecrawl + PIT gating in _web_news."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from trading_crew.data_sources import news_providers, pit


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        "TAVILY_API_KEY", "TAVILY_API_KEYS",
        "SERPAPI_API_KEY", "SERPAPI_API_KEYS",
        "FIRECRAWL_API_KEY", "FIRECRAWL_API_KEYS",
    ):
        monkeypatch.delenv(var, raising=False)
    news_providers.reset_rotation()
    yield
    news_providers.reset_rotation()


def test_read_keys_plural_wins_and_dedupes(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEYS", "a, b ,a, c")
    assert news_providers.read_keys("tavily") == ["a", "b", "c"]


def test_read_keys_falls_back_to_singular(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "solo")
    assert news_providers.read_keys("serpapi") == ["solo"]


def test_next_key_round_robins(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEYS", "k1,k2,k3")
    got = [news_providers.next_key("firecrawl") for _ in range(6)]
    assert got == ["k1", "k2", "k3", "k1", "k2", "k3"]


def test_next_key_none_when_unset():
    assert news_providers.next_key("tavily") is None
    assert news_providers.has_keys("tavily") is False


def test_web_news_sequence_falls_through_to_firecrawl(monkeypatch):
    """Sequence Tavily -> SerpAPI -> Firecrawl: empty earlier providers fall
    through to the first non-empty one, which alone supplies the rows."""
    from trading_crew import tools

    monkeypatch.setattr(tools, "_tavily_news_rows", lambda q, max_results=8: [])
    monkeypatch.setattr(tools, "_serpapi_news", lambda q, max_results=8: [])
    monkeypatch.setattr(tools, "_firecrawl_news", lambda q, max_results=8: [
        {"title": "Fire C", "url": "https://ft.com/c", "snippet": "s", "provider": "firecrawl"},
    ])
    monkeypatch.setattr(tools, "_web_news_via_research", lambda q: "", raising=False)
    out = tools._web_news("q", max_results=6)
    assert "Fire C" in out
    assert "firecrawl" in out


def test_web_news_sequence_stops_at_first_nonempty(monkeypatch):
    """Firecrawl must NOT be consulted once Tavily has already produced rows."""
    from trading_crew import tools

    fire_called = {"n": 0}

    def _fire(q, max_results=8):
        fire_called["n"] += 1
        return [{"title": "Fire C", "url": "https://ft.com/c", "snippet": "s", "provider": "firecrawl"}]

    # Stub the free custom-crawl tier empty so the paid API sequence (the
    # subject of this test) is exercised regardless of default ordering.
    monkeypatch.setattr(tools, "_web_news_via_research", lambda q: "", raising=False)
    monkeypatch.setattr(tools, "_tavily_news_rows", lambda q, max_results=8: [
        {"title": "Tav A", "url": "https://reuters.com/a", "snippet": "s", "provider": "tavily"},
    ])
    monkeypatch.setattr(tools, "_serpapi_news", lambda q, max_results=8: [])
    monkeypatch.setattr(tools, "_firecrawl_news", _fire)
    out = tools._web_news("q", max_results=6)
    assert "Tav A" in out and "Fire C" not in out
    assert fire_called["n"] == 0


def test_web_news_pit_filters_future_rows(monkeypatch):
    from trading_crew import tools

    # Stub the free custom-crawl tier empty so the dated API rows (which
    # carry the publish dates this PIT test checks) are the ones filtered.
    monkeypatch.setattr(tools, "_web_news_via_research", lambda q: "", raising=False)
    monkeypatch.setattr(tools, "_tavily_news_rows", lambda q, max_results=8: [
        {"title": "Old news", "url": "https://reuters.com/old", "snippet": "s",
         "provider": "tavily", "published": "2024-06-01"},
        {"title": "Future news", "url": "https://reuters.com/fut", "snippet": "s",
         "provider": "tavily", "published": "2025-03-01"},
    ])
    monkeypatch.setattr(tools, "_serpapi_news", lambda q, max_results=8: [])
    monkeypatch.setattr(tools, "_firecrawl_news", lambda q, max_results=8: [])

    as_of = datetime(2024, 9, 1, tzinfo=timezone.utc)
    with pit.as_of_scope(as_of, mode="backfill", strict=True):
        out = tools._web_news("q", max_results=6)
    assert "Old news" in out
    assert "Future news" not in out


def test_firecrawl_rows_empty_without_key():
    assert news_providers.firecrawl_rows("q") == []
