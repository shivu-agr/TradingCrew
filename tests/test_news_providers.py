"""Tests for the merged Tavily + SerpAPI news layer and the
anti-fabrication placeholder-URL filter.

Background
----------
The user reported two coupled problems:

* News context was too thin (single Tavily provider), so the model
  fabricated ``example.com`` placeholder URLs to satisfy the strict
  provenance rule, which downstream compliance / critic then flagged as
  "unverified sources" and used to cap conviction into HOLD.
* Breadth gaps (e.g. NVDA's hyperscaler-capex demand cycle) were missed.

The fix wires SerpAPI (Google News) as a PARALLEL provider alongside
Tavily (a merge, not a fallback), de-duplicates the union, and rejects
any placeholder / fabricated URL before it can leave the tool layer.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Placeholder-URL rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/news/foo",
        "http://example.org/x",
        "https://www.example.net/story",
        "https://sub.example.com/a/b",
        "http://test.com/headline",
        "https://placeholder.com/article",
        "",
        "not-a-url",
        "/relative/path",
        "http://localhost/x",
        "https://foo",  # bare host, no dot
    ],
)
def test_is_placeholder_url_rejects_fabricated_and_relative(url):
    from trading_crew.tools import _is_placeholder_url

    assert _is_placeholder_url(url) is True, f"{url!r} should be rejected"


@pytest.mark.parametrize(
    "url",
    [
        "https://economictimes.indiatimes.com/markets/foo",
        "https://www.reuters.com/business/x",
        "http://livemint.com/companies/y",
        "https://finance.yahoo.com/news/z",
    ],
)
def test_is_placeholder_url_accepts_real_urls(url):
    from trading_crew.tools import _is_placeholder_url

    assert _is_placeholder_url(url) is False, f"{url!r} should be accepted"


def test_reject_placeholder_urls_drops_only_fakes():
    from trading_crew.tools import _reject_placeholder_urls

    rows = [
        {"title": "Real", "url": "https://reuters.com/a", "provider": "tavily"},
        {"title": "Fake", "url": "https://example.com/b", "provider": "serpapi"},
        {"title": "Empty", "url": "", "provider": "tavily"},
    ]
    out = _reject_placeholder_urls(rows)
    assert [r["title"] for r in out] == ["Real"]


# ---------------------------------------------------------------------------
# _web_news merge + de-duplicate across both providers
# ---------------------------------------------------------------------------


def test_web_news_merges_both_providers(monkeypatch):
    """Both providers' rows must appear in the merged output (parallel
    merge, not a fallback where SerpAPI only fires if Tavily is empty)."""
    from trading_crew import tools

    monkeypatch.setattr(
        tools, "_tavily_news_rows",
        lambda q, max_results=8: [
            {"title": "Tavily story A", "url": "https://reuters.com/a",
             "snippet": "s", "provider": "tavily"},
        ],
    )
    monkeypatch.setattr(
        tools, "_serpapi_news",
        lambda q, max_results=8: [
            {"title": "SerpAPI story B", "url": "https://livemint.com/b",
             "snippet": "s", "provider": "serpapi"},
        ],
    )

    out = tools._web_news("HAL order book", max_results=6)
    assert "Tavily story A" in out
    assert "SerpAPI story B" in out
    # The source line names BOTH providers.
    assert "serpapi" in out and "tavily" in out


def test_web_news_dedupes_by_url_and_title(monkeypatch):
    """The same article surfaced by both providers (or trivially
    different URLs) must appear only once."""
    from trading_crew import tools

    monkeypatch.setattr(
        tools, "_tavily_news_rows",
        lambda q, max_results=8: [
            {"title": "HAL wins ₹26,000 cr order", "url": "https://reuters.com/hal",
             "snippet": "s", "provider": "tavily"},
        ],
    )
    monkeypatch.setattr(
        tools, "_serpapi_news",
        lambda q, max_results=8: [
            # Same story: different query params + www + trailing slash.
            {"title": "HAL wins ₹26,000 cr order",
             "url": "https://www.reuters.com/hal/?utm=x", "snippet": "s2",
             "provider": "serpapi"},
        ],
    )

    out = tools._web_news("HAL order", max_results=6)
    assert out.count("HAL wins ₹26,000 cr order") == 1


def test_web_news_rejects_fabricated_provider_rows(monkeypatch):
    """Even if a provider returns a placeholder URL, it must never reach
    the tool output."""
    from trading_crew import tools

    monkeypatch.setattr(
        tools, "_tavily_news_rows",
        lambda q, max_results=8: [
            {"title": "Fake", "url": "https://example.com/x", "snippet": "s",
             "provider": "tavily"},
        ],
    )
    monkeypatch.setattr(tools, "_serpapi_news", lambda q, max_results=8: [])

    out = tools._web_news("q", max_results=6)
    assert "example.com" not in out
    assert "No news found." in out


def test_web_news_survives_one_provider_error(monkeypatch):
    """A provider raising must NOT crash the merge — the other
    provider's rows still come through."""
    from trading_crew import tools

    def _boom(q, max_results=8):
        raise RuntimeError("provider down")

    monkeypatch.setattr(tools, "_tavily_news_rows", _boom)
    monkeypatch.setattr(
        tools, "_serpapi_news",
        lambda q, max_results=8: [
            {"title": "Still here", "url": "https://reuters.com/z",
             "snippet": "s", "provider": "serpapi"},
        ],
    )

    out = tools._web_news("q", max_results=6)
    assert "Still here" in out


def test_serpapi_news_returns_empty_without_key(monkeypatch):
    """No key configured → no rows (and no network call)."""
    from trading_crew import tools

    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    assert tools._serpapi_news("q") == []


def test_serpapi_news_parses_google_news_payload(monkeypatch):
    """The Google-News JSON shape (news_results with link/title, plus
    grouped 'stories') must be flattened into normalized rows."""
    from trading_crew import tools

    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "news_results": [
                    {"title": "Top story", "link": "https://reuters.com/top",
                     "snippet": "snip", "source": {"name": "Reuters"}},
                    {"stories": [
                        {"title": "Grouped 1", "link": "https://livemint.com/g1",
                         "snippet": "s1"},
                        {"title": "Grouped 2", "link": "https://livemint.com/g2"},
                    ]},
                ]
            }

    monkeypatch.setattr(
        tools, "_with_retry", lambda fn, *a, **k: _Resp(),
    )

    rows = tools._serpapi_news("q", max_results=8)
    titles = [r["title"] for r in rows]
    assert "Top story" in titles
    assert "Grouped 1" in titles and "Grouped 2" in titles
    assert all(r["provider"] == "serpapi" for r in rows)
