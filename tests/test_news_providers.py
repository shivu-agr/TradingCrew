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

News is fetched via a FALLBACK SEQUENCE — Tavily -> SerpAPI -> Firecrawl ->
custom websearch (search+crawl+summarize) — where the first provider that
yields usable, PIT-valid rows wins (each provider rotates its own key list).
Rows are de-duplicated within a provider and any placeholder / fabricated URL
is rejected before it can leave the tool layer.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _stub_web_fallbacks(monkeypatch):
    """Stub the two later stages of the news fallback SEQUENCE so these
    tests stay deterministic and never make a live call: Firecrawl (3rd in
    the chain) and the custom-websearch summarizer (final fallback)."""
    from trading_crew import tools

    monkeypatch.setattr(tools, "_firecrawl_news", lambda q, max_results=8: [], raising=False)
    monkeypatch.setattr(tools, "_web_news_via_research", lambda q: "", raising=False)


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
        # scheme-less absolute hosts — search backends emit these
        "www.reuters.com/business/x",
        "economictimes.indiatimes.com/markets/foo",
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
        {"title": "Scheme-less", "url": "www.moneycontrol.com/news/rbi", "provider": "tavily"},
    ]
    out = _reject_placeholder_urls(rows)
    assert [r["title"] for r in out] == ["Real", "Scheme-less"]
    assert out[1]["url"] == "https://www.moneycontrol.com/news/rbi"


# ---------------------------------------------------------------------------
# _web_news fallback SEQUENCE: Tavily -> SerpAPI -> Firecrawl -> websearch
# ---------------------------------------------------------------------------


def test_web_news_uses_first_provider_in_sequence(monkeypatch):
    """When Tavily returns rows, the later providers are NOT consulted
    (this is a fallback chain, not a parallel merge)."""
    from trading_crew import tools

    serp_called = {"n": 0}

    monkeypatch.setattr(
        tools, "_tavily_news_rows",
        lambda q, max_results=8: [
            {"title": "Tavily story A", "url": "https://reuters.com/a",
             "snippet": "s", "provider": "tavily"},
        ],
    )

    def _serp(q, max_results=8):
        serp_called["n"] += 1
        return [{"title": "SerpAPI story B", "url": "https://livemint.com/b",
                 "snippet": "s", "provider": "serpapi"}]

    monkeypatch.setattr(tools, "_serpapi_news", _serp)

    out = tools._web_news("HAL order book", max_results=6)
    assert "Tavily story A" in out
    assert "SerpAPI story B" not in out       # SerpAPI not reached
    assert serp_called["n"] == 0
    assert "tavily" in out


def test_web_news_falls_back_to_serpapi_when_tavily_empty(monkeypatch):
    """SerpAPI is consulted only when Tavily yields nothing usable."""
    from trading_crew import tools

    monkeypatch.setattr(tools, "_tavily_news_rows", lambda q, max_results=8: [])
    monkeypatch.setattr(
        tools, "_serpapi_news",
        lambda q, max_results=8: [
            {"title": "SerpAPI story B", "url": "https://livemint.com/b",
             "snippet": "s", "provider": "serpapi"},
        ],
    )

    out = tools._web_news("HAL order book", max_results=6)
    assert "SerpAPI story B" in out
    assert "serpapi" in out


def test_web_news_falls_through_to_apis_when_custom_crawl_degraded(monkeypatch):
    """A custom-crawl summary that self-reports its sources are irrelevant
    ('do not contain', 'no factual briefing') must NOT be shipped as-is —
    the paid news APIs get consulted instead (the custom tier's keyless
    crawler is 403-blocked by bot-protected news sites, so a degraded digest
    is exactly the NTNX "no relevant info" failure mode)."""
    from trading_crew import tools

    monkeypatch.setattr(
        tools, "_web_news_via_research",
        lambda q: "The supplied web pages do not contain any information "
                  "about the Federal Reserve's recent interest-rate decision. "
                  "No factual briefing can be compiled from the provided "
                  "material.",
    )
    monkeypatch.setattr(
        tools, "_tavily_news_rows",
        lambda q, max_results=8: [
            {"title": "Fed holds rates steady", "url": "https://reuters.com/fed",
             "snippet": "s", "provider": "tavily"},
        ],
    )
    monkeypatch.setattr(tools, "_serpapi_news", lambda q, max_results=8: [])

    out = tools._web_news("Federal Reserve", max_results=6)
    assert "Fed holds rates steady" in out
    assert "do not contain" not in out
    assert "custom crawl" not in out
    assert "tavily" in out


def test_useful_digest_mentioning_irrelevant_pages_is_not_degraded():
    """The summarizer is told to ignore irrelevant pages, so a real
    digest often contains that word. Bare 'irrelevant' used to discard
    the whole crawl (NTNX geo → No news found)."""
    from trading_crew.tools import _is_degraded_news_summary

    useful = (
        "Ignored irrelevant Wikipedia pages.\n"
        "- Nutanix revenue delayed by supply strains "
        "[source: https://stocktitan.net/news/NTNX]"
    )
    assert _is_degraded_news_summary(useful) is False
    empty = (
        "The supplied web pages do not contain any information about "
        "export controls. No factual briefing can be compiled."
    )
    assert _is_degraded_news_summary(empty) is True


def test_web_news_uses_custom_crawl_tier_when_it_returns_content(monkeypatch):
    """With custom-crawl preferred (default), a non-empty summary from the
    free tier is used directly and the paid APIs are not consulted."""
    from trading_crew import tools

    monkeypatch.setattr(tools, "_tavily_news_rows", lambda q, max_results=8: [])
    monkeypatch.setattr(tools, "_serpapi_news", lambda q, max_results=8: [])
    monkeypatch.setattr(tools, "_firecrawl_news", lambda q, max_results=8: [])
    monkeypatch.setattr(
        tools, "_web_news_via_research",
        lambda q: "- TCS beat estimates [source: https://reuters.com/tcs]",
    )

    out = tools._web_news("TCS earnings", max_results=6)
    assert "TCS beat estimates" in out
    assert "custom crawl" in out


def test_web_news_dedupes_within_provider(monkeypatch):
    """The same article returned twice by a provider (trivially different
    URLs) must appear only once."""
    from trading_crew import tools

    monkeypatch.setattr(
        tools, "_tavily_news_rows",
        lambda q, max_results=8: [
            {"title": "HAL wins ₹26,000 cr order", "url": "https://reuters.com/hal",
             "snippet": "s", "provider": "tavily"},
            # Same story: different query params + www + trailing slash.
            {"title": "HAL wins ₹26,000 cr order",
             "url": "https://www.reuters.com/hal/?utm=x", "snippet": "s2",
             "provider": "tavily"},
        ],
    )

    out = tools._web_news("HAL order", max_results=6)
    assert out.count("HAL wins ₹26,000 cr order") == 1


def test_web_news_rejects_fabricated_provider_rows(monkeypatch):
    """Even if a provider returns a placeholder URL, it must never reach
    the tool output (and an all-empty chain says 'No news found.')."""
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
    """A provider raising must NOT crash the chain — the next provider in
    the sequence still comes through."""
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
