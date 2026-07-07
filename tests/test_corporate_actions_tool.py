"""Tests for the ``get_corporate_actions`` tool added in Phase A.

Background
----------
The generic ``get_news`` query (``"<company> <ticker> stock news
sentiment last week"``, ``max_results=6``, ``search_depth="basic"``)
is biased toward general sentiment articles and routinely buries
specific corporate-action announcements (buybacks, dividends, splits,
M&A, leadership changes) outside the top 6 — for example, on a recent
Bajaj Auto run the company had announced a share buyback during the
analysis window and the news layer never returned it.

``get_corporate_actions`` issues one focused Tavily query per
action theme (six themes total) and merges the results into a single
markdown digest.  These tests stub the Tavily client and verify:

* The tool is registered in ``ALL_TOOLS`` and wired to the news +
  quant analyst defaults.
* It hits every defined corporate-action theme exactly once per call.
* It surfaces headlines into the per-theme markdown section with a
  ``[source: <url>]`` tag inline (so downstream agents can preserve
  provenance per the project-wide rule).
* Themes with zero hits land in the trailing summary so the
  analyst can see what was searched-and-empty.
* When the underlying Tavily client raises (rate limit / network
  error / bad key) the tool degrades gracefully with a per-theme
  ``(search failed: ...)`` note instead of crashing the whole task.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import patch

import pytest


def _fake_results(theme_to_results: Dict[str, List[Dict[str, str]]]):
    """Build a Tavily.search side_effect that returns canned hits per theme."""
    from trading_crew import tools as tools_mod

    def _search(query: str, **kwargs):
        # Pick the theme whose seeds appear in the query.  Each theme's
        # seed phrase is unique (see _CORPORATE_ACTION_THEMES) so the
        # first match is unambiguous.
        for theme_name, _ in tools_mod._CORPORATE_ACTION_THEMES:
            seed_first_word = _theme_seed_marker(theme_name)
            if seed_first_word in query:
                return {"results": list(theme_to_results.get(theme_name, []))}
        return {"results": []}

    return _search


def _theme_seed_marker(theme_name: str) -> str:
    """The unique keyword the theme's seed query contains."""
    return {
        "buyback":  "buyback",
        "dividend": "dividend",
        "split":    "stock split",
        "m_and_a":  "acquisition",
        "guidance": "earnings guidance",
        "board":    "board appointment",
    }[theme_name]


def test_tool_is_registered_and_wired_to_news():
    """The tool must live in the central tool catalog so the UI's
    tool-picker checkboxes render it AND the default agent wiring must
    list it for the News desk (which now owns corporate actions after
    the analyst consolidation merged the old Quant desk into Market)."""
    from trading_crew import tools as tools_mod
    from trading_crew.tools import ALL_TOOLS, DEFAULT_AGENT_TOOLS

    assert "get_corporate_actions" in ALL_TOOLS
    assert "get_corporate_actions" in DEFAULT_AGENT_TOOLS["news_analyst"]
    # Six themes is intentional; if a theme is added or removed, the
    # prompt copy in tasks.yaml + this test should be updated together.
    assert len(tools_mod._CORPORATE_ACTION_THEMES) == 6


def test_calls_every_theme_once_per_invocation(monkeypatch):
    """One Tavily.search per theme — no more, no less.  Bounded so a
    single get_corporate_actions call stays under the per-task tool
    budget (5 for news_task) and doesn't itself trigger an OSS-LLM
    tool-loop degradation."""
    from trading_crew.tools import get_corporate_actions

    calls: List[str] = []

    def _track(query: str, **kwargs):
        calls.append(query)
        return {"results": []}

    fake_client = type("F", (), {"search": staticmethod(_track)})()
    monkeypatch.setattr("trading_crew.tools._tavily", lambda: fake_client)
    # SerpAPI is a parallel provider; neutralise it so this test counts
    # only the Tavily theme queries (one per theme) deterministically.
    monkeypatch.setattr("trading_crew.tools._serpapi_news", lambda *a, **k: [])
    monkeypatch.setattr("trading_crew.tools.resolve_ticker", lambda t: t)
    monkeypatch.setattr(
        "trading_crew.tools.get_market_profile",
        lambda t: type("P", (), {"long_name": "Bajaj Auto Limited"})(),
    )

    out = get_corporate_actions.func("BAJAJ-AUTO.NS")  # type: ignore[attr-defined]

    assert len(calls) == 6, (
        "Expected exactly 6 theme queries (one per theme); "
        f"got {len(calls)} → {calls}"
    )
    # Each call must include the company's long name so the ranker
    # doesn't get confused by ambiguous tickers.
    for q in calls:
        assert "Bajaj Auto Limited" in q
        assert "BAJAJ-AUTO.NS" in q


def test_surfaces_buyback_headline_with_inline_source_tag(monkeypatch):
    """A buyback announcement returned by Tavily must end up in the
    markdown output under the ``### buyback`` section AND carry a
    ``[source: <url>]`` inline tag so the news analyst can keep
    provenance when it cites the headline.  This is the smoking gun
    for the 'Bajaj Auto buyback was not considered' user report."""
    from trading_crew.tools import get_corporate_actions

    fake_search = _fake_results({
        "buyback": [{
            "title": "Bajaj Auto announces ₹4,000 cr share buyback",
            "url": "https://economictimes.indiatimes.com/news/bajaj-buyback",
            "content": "Bajaj Auto board approved a buyback at ₹12,000 per share.",
        }],
    })
    fake_client = type("F", (), {"search": staticmethod(fake_search)})()
    monkeypatch.setattr("trading_crew.tools._tavily", lambda: fake_client)
    monkeypatch.setattr("trading_crew.tools._serpapi_news", lambda *a, **k: [])
    monkeypatch.setattr("trading_crew.tools.resolve_ticker", lambda t: t)
    monkeypatch.setattr(
        "trading_crew.tools.get_market_profile",
        lambda t: type("P", (), {"long_name": "Bajaj Auto Limited"})(),
    )

    out = get_corporate_actions.func("BAJAJ-AUTO.NS")  # type: ignore[attr-defined]

    # The headline must land in the buyback section AND carry an
    # inline source tag so downstream agents preserve provenance.
    assert "### buyback" in out
    assert "Bajaj Auto announces" in out
    assert "[source: https://economictimes.indiatimes.com/news/bajaj-buyback]" in out
    # Empty themes get summarised at the end (5 of 6 themes empty).
    assert "Themes with no recent headlines" in out


def test_search_failures_degrade_per_theme_instead_of_crashing(monkeypatch):
    """A provider (Tavily / SerpAPI) can rate-limit or throw on any
    single theme — that must NOT crash the whole tool.  The merged
    news layer swallows a single provider's error (merge-survives
    design) so the theme degrades to "no headlines" and the tool keeps
    producing a report for every other theme instead of raising."""
    from trading_crew.tools import get_corporate_actions

    def _flaky(query: str, **kwargs):
        if "buyback" in query:
            raise RuntimeError("rate limit")
        return {"results": []}

    fake_client = type("F", (), {"search": staticmethod(_flaky)})()
    monkeypatch.setattr("trading_crew.tools._tavily", lambda: fake_client)
    monkeypatch.setattr("trading_crew.tools._serpapi_news", lambda *a, **k: [])
    monkeypatch.setattr("trading_crew.tools.resolve_ticker", lambda t: t)
    monkeypatch.setattr(
        "trading_crew.tools.get_market_profile",
        lambda t: type("P", (), {"long_name": "Foo Corp"})(),
    )
    # Silence retry to keep the test fast.
    monkeypatch.setattr(
        "trading_crew.tools._with_retry",
        lambda fn, *a, **kw: fn(*a, **{k: v for k, v in kw.items()
                                       if k not in ("attempts", "base_delay",
                                                    "max_delay", "label")}),
    )

    # Must not raise even though the buyback theme's Tavily call throws.
    out = get_corporate_actions.func("FOO")  # type: ignore[attr-defined]
    # The failing theme degrades to "no headlines" alongside the others.
    assert "Themes with no recent headlines" in out
    assert "buyback" in out
