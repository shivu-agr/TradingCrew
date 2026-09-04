"""Unit tests for the ticker resolver + market-profile + dynamic query layer.

These cover ticker resolution, facts-only market profiles, analyst-written
search queries, and the live (not dictionary) peer universe.

We stub yfinance so the tests are deterministic and run offline.
"""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _clear_caches():
    """Each test starts with empty resolver + profile caches."""
    from trading_crew import market_context as mc
    mc.clear_resolve_cache()
    mc.clear_profile_cache()
    yield
    mc.clear_resolve_cache()
    mc.clear_profile_cache()


class _FakeTicker:
    """Minimal yfinance.Ticker stand-in used by the resolver probes."""

    def __init__(self, symbol, history_map, info_map):
        self.symbol = symbol
        self._history = history_map
        self._info = info_map

    def history(self, period="1mo"):
        return self._history.get(self.symbol, pd.DataFrame())

    @property
    def info(self):
        return self._info.get(self.symbol, {})


def _patch_yf(monkeypatch, history_map, info_map):
    """Patch yfinance.Ticker for both the resolver and the profile builder."""
    import yfinance as yf
    from trading_crew import market_context as mc

    def _factory(symbol):
        return _FakeTicker(symbol, history_map, info_map)

    monkeypatch.setattr(yf, "Ticker", _factory)
    monkeypatch.setattr(mc, "yf", yf)


# ---------------------------------------------------------------------------
# 1. Ticker resolution
# ---------------------------------------------------------------------------


def test_resolver_passes_through_already_suffixed_symbol(monkeypatch):
    """Symbols with a dot/equals/caret should not be probed further."""
    from trading_crew.market_context import resolve_ticker
    _patch_yf(monkeypatch, history_map={}, info_map={})
    assert resolve_ticker("MAZDOCK.NS") == "MAZDOCK.NS"
    assert resolve_ticker("CL=F") == "CL=F"
    assert resolve_ticker("^GSPC") == "^GSPC"


def test_resolver_returns_bare_us_symbol_when_it_has_data(monkeypatch):
    """A bare US ticker that yfinance already knows must NOT get .NS appended."""
    from trading_crew.market_context import resolve_ticker
    sample = pd.DataFrame({"Close": [100.0]})
    _patch_yf(monkeypatch, history_map={"AAPL": sample}, info_map={})
    assert resolve_ticker("AAPL") == "AAPL"


def test_resolver_appends_ns_for_indian_ticker(monkeypatch):
    """The MAZDOCK bug — bare input + only ``.NS`` has data → resolve to .NS."""
    from trading_crew.market_context import resolve_ticker
    sample = pd.DataFrame({"Close": [4500.0]})
    # Only the .NS variant returns history; bare + .BO + others are empty.
    _patch_yf(
        monkeypatch,
        history_map={"MAZDOCK.NS": sample},
        info_map={},
    )
    assert resolve_ticker("MAZDOCK") == "MAZDOCK.NS"


def test_resolver_returns_raw_on_total_miss(monkeypatch):
    """No suffix worked → return input untouched so the tool surfaces the gap."""
    from trading_crew.market_context import resolve_ticker
    _patch_yf(monkeypatch, history_map={}, info_map={})
    assert resolve_ticker("ZZZUNKNOWN") == "ZZZUNKNOWN"


def test_resolver_uses_cache_for_repeat_calls(monkeypatch):
    """Second call should not re-probe — verified by counting history hits."""
    from trading_crew import market_context as mc
    calls: list[str] = []

    def _counting_factory(symbol):
        calls.append(symbol)
        df = pd.DataFrame({"Close": [1.0]}) if symbol == "INFY.NS" else pd.DataFrame()
        return _FakeTicker(symbol, {"INFY.NS": df}, {})

    monkeypatch.setattr(mc.yf, "Ticker", _counting_factory)
    assert mc.resolve_ticker("INFY") == "INFY.NS"
    n_after_first = len(calls)
    assert mc.resolve_ticker("INFY") == "INFY.NS"
    # Cache hit → no additional yf.Ticker constructions.
    assert len(calls) == n_after_first


def test_resolver_short_circuits_company_name(monkeypatch, caplog):
    """An LLM mistakenly passing a company *name* in must NOT trigger
    the exchange-suffix probe storm (~11 network calls + 11 yfinance
    ERROR log lines).  We fail fast and log once at WARNING.
    """
    from trading_crew import market_context as mc

    calls: list[str] = []

    def _counting_factory(symbol):
        calls.append(symbol)
        return _FakeTicker(symbol, {}, {})

    monkeypatch.setattr(mc.yf, "Ticker", _counting_factory)
    with caplog.at_level("WARNING", logger="trading_crew.market_context"):
        out = mc.resolve_ticker("LARSEN & TOUBRO LIMITED")
    assert out == "LARSEN & TOUBRO LIMITED"
    assert calls == [], (
        "Non-ticker input must not trigger any yf.Ticker construction; "
        f"got {calls}"
    )
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert warnings, "Expected one WARNING line on non-ticker input"
    assert "LARSEN & TOUBRO LIMITED" in warnings[0].getMessage()


@pytest.mark.parametrize("garbage", [
    "Reliance Industries Ltd",   # spaces
    "BHARTI AIRTEL",             # space
    "abc 123",                   # space + lower-case mix
    "AAAAAAAAAAAAAA",            # length > 12
    "TICKER?",                   # ? not allowed
    "SOME/THING",                # / not allowed
    "X.X.X.X",                   # already-dotted; pass-through, no probe
])
def test_resolver_rejects_a_range_of_non_tickers(monkeypatch, garbage):
    """Belt-and-braces — every shape of "obviously not a ticker" input
    we've seen in real logs must short-circuit the probe."""
    from trading_crew import market_context as mc

    calls: list[str] = []

    def _counting_factory(symbol):
        calls.append(symbol)
        return _FakeTicker(symbol, {}, {})

    monkeypatch.setattr(mc.yf, "Ticker", _counting_factory)
    out = mc.resolve_ticker(garbage)
    # Whitespace / punctuation / over-long bare symbols fail the
    # structural check. Dotted junk (``X.X.X.X``) is treated as already
    # suffixed and passed through. In all cases we must NOT probe.
    assert calls == [], (
        f"non-ticker {garbage!r} should not have triggered yf calls; got {calls}"
    )
    assert out == garbage.strip().upper()


def test_resolver_silences_yfinance_error_chatter(monkeypatch, caplog):
    """yfinance logs every empty-history response at ERROR level — but a
    probe miss is *expected*, not a real error.  The resolver must
    temporarily mute yfinance's logger during the probe so the project
    log stays signal-rich.
    """
    import logging
    from trading_crew import market_context as mc

    def _emitting_factory(symbol):
        # Simulate yfinance's own ERROR log for an empty response,
        # then return an empty DataFrame so the probe fails.
        logging.getLogger("yfinance").error(
            "$%s: possibly delisted; no price data found", symbol,
        )
        return _FakeTicker(symbol, {}, {})

    monkeypatch.setattr(mc.yf, "Ticker", _emitting_factory)
    with caplog.at_level("ERROR", logger="yfinance"):
        out = mc.resolve_ticker("NOMATCH")
    assert out == "NOMATCH"
    yf_errors = [r for r in caplog.records if r.name == "yfinance"]
    assert yf_errors == [], (
        "yfinance ERROR lines should be suppressed during probing; "
        f"leaked: {[r.getMessage() for r in yf_errors]}"
    )


# ---------------------------------------------------------------------------
# 2. MarketProfile — country / macro basket / themes
# ---------------------------------------------------------------------------


def test_profile_for_us_name_uses_us_macro_basket(monkeypatch):
    from trading_crew.market_context import get_market_profile
    _patch_yf(
        monkeypatch,
        history_map={},
        info_map={
            "NVDA": {
                "country": "United States", "exchange": "NMS",
                "longName": "NVIDIA Corporation",
                "sector": "Technology", "industry": "Semiconductors",
            }
        },
    )
    prof = get_market_profile("NVDA")
    assert prof.country == "United States"
    assert prof.currency == "USD"
    assert prof.peer_index == "^GSPC"
    assert "us_10y_treasury_yield" in prof.macro_tickers
    assert prof.macro_tickers["us_10y_treasury_yield"] == "^TNX"
    # Facts only — no canned TSMC / export-control phrasebook.
    assert prof.industry.lower().startswith("semiconductor")
    q = prof.geopolitical_query().lower()
    assert "nvidia" in q and "tariff" in q


def test_profile_for_indian_shipbuilder_uses_india_basket_and_ship_seeds(monkeypatch):
    """MAZDOCK regression — profile must surface India macro + ship themes,
    NOT the US Fed/UST/DXY defaults."""
    from trading_crew.market_context import get_market_profile
    _patch_yf(
        monkeypatch,
        history_map={},
        info_map={
            "MAZDOCK.NS": {
                "country": "India", "exchange": "NSI",
                "longName": "Mazagon Dock Shipbuilders Limited",
                "sector": "Industrials",
                "industry": "Aerospace & Defense",
            }
        },
    )
    prof = get_market_profile("MAZDOCK.NS")
    assert prof.country == "India"
    assert prof.currency == "INR"
    assert prof.peer_index == "^NSEI"
    # India macro basket must dominate, not the US one.
    assert "nifty_50" in prof.macro_tickers
    assert "usd_inr" in prof.macro_tickers
    assert prof.macro_tickers["usd_inr"] == "INR=X"
    assert "us_10y_treasury_yield" not in prof.macro_tickers
    # Brief is facts (India + aerospace), not a Hormuz/RBI phrasebook.
    brief = prof.to_brief().lower()
    assert "india" in brief
    assert "aerospace" in brief or "defense" in brief
    assert "hormuz" not in brief
    assert "federal reserve" not in brief
    assert "no canned theme list" in brief or "query contract" in brief


def test_profile_falls_back_to_exchange_when_country_missing(monkeypatch):
    """yfinance sometimes omits ``country`` — exchange code must still
    route an NSE-listed name to the India basket."""
    from trading_crew.market_context import get_market_profile
    _patch_yf(
        monkeypatch,
        history_map={},
        info_map={"INFY.NS": {"exchange": "NSI", "sector": "Technology", "industry": "IT Services"}},
    )
    prof = get_market_profile("INFY.NS")
    assert prof.country == "India"


# ---------------------------------------------------------------------------
# 3. Dynamic query builders
# ---------------------------------------------------------------------------


def test_country_news_query_for_indian_name_contains_india_themes(monkeypatch):
    from trading_crew.market_context import get_market_profile
    _patch_yf(
        monkeypatch,
        history_map={},
        info_map={
            "MAZDOCK.NS": {
                "country": "India", "exchange": "NSI",
                "longName": "Mazagon Dock Shipbuilders Limited",
                "sector": "Industrials", "industry": "Aerospace & Defense",
            }
        },
    )
    q = get_market_profile("MAZDOCK.NS").country_news_query().lower()
    assert "mazagon" in q
    assert "federal reserve" not in q
    assert "treasury yield" not in q


def test_macro_news_query_is_country_macro_not_company_news(monkeypatch):
    """``macro_news_query`` is for get_global_news — country macro, framed as
    recent news, and NOT prefixed with the company name (that prefix diluted
    the search into generic/navigational home pages for broad macro terms)."""
    from trading_crew.market_context import get_market_profile
    _patch_yf(
        monkeypatch,
        history_map={},
        info_map={
            "MAZDOCK.NS": {
                "country": "India", "exchange": "NSI",
                "longName": "Mazagon Dock Shipbuilders Limited",
                "sector": "Industrials", "industry": "Aerospace & Defense",
            }
        },
    )
    q = get_market_profile("MAZDOCK.NS").macro_news_query().lower()
    assert "mazagon" not in q                       # not company news
    assert "india" in q
    assert "central bank" in q or "inflation" in q


def test_geopolitical_query_for_shipbuilder_uses_facts_not_china_taiwan(monkeypatch):
    """Default geo query is name + country + industry + lens verbs.
    Hormuz / China-Taiwan are the *agent's* job to add via query=."""
    from trading_crew.market_context import get_market_profile
    _patch_yf(
        monkeypatch,
        history_map={},
        info_map={
            "MAZDOCK.NS": {
                "country": "India", "exchange": "NSI",
                "longName": "Mazagon Dock Shipbuilders Limited",
                "sector": "Industrials", "industry": "Aerospace & Defense",
            }
        },
    )
    prof = get_market_profile("MAZDOCK.NS")
    q = prof.geopolitical_query().lower()
    assert q.startswith("mazagon dock shipbuilders limited")
    assert "india" in q
    assert "aerospace" in q or "defense" in q
    assert "tariff" in q or "sanction" in q
    assert "taiwan" not in q
    written = prof.geopolitical_query("Mazagon Dock Hormuz Red Sea naval sanctions").lower()
    assert "hormuz" in written and "mazagon" in written


def test_ntnx_default_query_is_facts_only_agent_query_wins(monkeypatch):
    """No industry phrasebook. Default = name/country/industry/lens verbs.
    An agent-written query is used verbatim."""
    from trading_crew.market_context import compose_search_query, get_market_profile
    _patch_yf(
        monkeypatch,
        history_map={},
        info_map={
            "NTNX": {
                "country": "United States", "exchange": "NMS",
                "longName": "Nutanix, Inc.",
                "sector": "Technology", "industry": "Software - Infrastructure",
            }
        },
    )
    prof = get_market_profile("NTNX")
    q = prof.geopolitical_query().lower()
    assert q.startswith("nutanix")
    assert "united states" in q
    assert "software" in q
    assert "tariff" in q or "sanction" in q
    assert "enterprise it spending" not in q
    assert "capital expenditure cycle" not in q
    assert len(q.split()) <= 16
    written = "Nutanix US China AI hardware export controls"
    assert compose_search_query(prof, lens="geo", agent_query=written) == written
    sq = prof.supply_chain_query().lower()
    assert "nutanix" in sq and "supply chain" in sq
    assert "enterprise it spending" not in sq


def test_supply_chain_query_uses_industry_words(monkeypatch):
    from trading_crew.market_context import get_market_profile
    _patch_yf(
        monkeypatch,
        history_map={},
        info_map={
            "NVDA": {
                "country": "United States", "exchange": "NMS",
                "longName": "NVIDIA Corporation",
                "sector": "Technology", "industry": "Semiconductors",
            }
        },
    )
    prof = get_market_profile("NVDA")
    q = prof.supply_chain_query().lower()
    assert "nvidia" in q
    assert "supply chain" in q
    assert "semiconductor" in q
    written = prof.supply_chain_query("NVIDIA TSMC ASML fab capacity")
    assert "TSMC" in written


# ---------------------------------------------------------------------------
# 4. Live peer universe (no per-ticker dictionary)
# ---------------------------------------------------------------------------


def _profile(**overrides):
    from trading_crew.market_context import MarketProfile

    base = dict(
        symbol="MAZDOCK.NS",
        raw_input="MAZDOCK.NS",
        long_name="Mazagon Dock Shipbuilders Limited",
        country="India",
        currency="INR",
        exchange="NSI",
        sector="Industrials",
        industry="Aerospace & Defense",
        industry_key="aerospace-defense",
        market_cap=1.1e11,
        peer_index="^NSEI",
        macro_tickers={"usd_inr": "INR=X"},
    )
    base.update(overrides)
    return MarketProfile(**base)


def test_no_hardcoded_sector_peer_dictionary():
    import trading_crew.market_context as mc

    assert not hasattr(mc, "_SECTOR_PEERS")


def test_long_nse_listings_are_valid_tickers():
    from trading_crew.market_context import _looks_like_ticker

    assert _looks_like_ticker("COCHINSHIP.NS")
    assert _looks_like_ticker("TATAMOTORS.NS")
    assert _looks_like_ticker("BHARTIARTL.NS")
    assert not _looks_like_ticker("LARSEN & TOUBRO LIMITED")


def test_yahoo_region_from_exchange_and_suffix():
    from trading_crew.market_context import yahoo_region_for

    assert yahoo_region_for(exchange="NSI") == "in"
    assert yahoo_region_for(exchange="NMS") == "us"
    assert yahoo_region_for(symbol="MAZDOCK.NS") == "in"
    assert yahoo_region_for(country="Japan") == "jp"


def test_screener_industry_matches_yahoo_enum_not_a_phrasebook():
    from trading_crew.market_context import _screener_industry_name

    name = _screener_industry_name(
        "Software - Infrastructure", "software-infrastructure", "Technology",
    )
    assert name == "Software—Infrastructure"


def test_agent_peers_win_and_do_not_call_the_live_screen(monkeypatch):
    from trading_crew import market_context as mc

    monkeypatch.setattr(mc, "resolve_ticker", lambda s: s.strip().upper())

    def _boom(*_a, **_k):
        raise AssertionError("live screen must not run when peers= is set")

    monkeypatch.setattr(mc, "_run_equity_screen", _boom)
    uni = mc.discover_peers(
        "NTNX",
        extra_peers="PSTG, NTAP",
        profile=_profile(
            symbol="NTNX", raw_input="NTNX", long_name="Nutanix, Inc.",
            country="United States", currency="USD", exchange="NMS",
            sector="Technology", industry="Software - Infrastructure",
            industry_key="software-infrastructure", market_cap=1.8e10,
            peer_index="^GSPC",
        ),
    )
    assert uni.method == "agent"
    assert uni.symbols == ["NTNX", "PSTG", "NTAP"]


def test_live_screen_ranks_same_listing_ahead_of_foreign_mega_cap(monkeypatch):
    """MAZDOCK must pick Indian aerospace names from the screen, not a
    curated COCHINSHIP list and not a US semi dictionary."""
    from trading_crew import market_context as mc

    def _screen(query, **_kwargs):
        return {
            "quotes": [
                {"symbol": "MAZDOCK.NS", "marketCap": 1.1e11},
                {"symbol": "COCHINSHIP.NS", "marketCap": 9.0e10},
                {"symbol": "GRSE.NS", "marketCap": 5.0e10},
                {"symbol": "LMT", "marketCap": 1.2e12},
            ]
        }

    monkeypatch.setattr(mc, "_run_equity_screen", _screen)
    uni = mc.discover_peers("MAZDOCK.NS", profile=_profile())
    assert uni.method.startswith("industry+")
    assert "COCHINSHIP.NS" in uni.symbols
    assert "GRSE.NS" in uni.symbols
    assert "NVDA" not in uni.symbols
    # Same listing (.NS) ranks ahead of Lockheed.
    others = [s for s in uni.symbols if s != "MAZDOCK.NS"]
    assert others[0].endswith(".NS")


def test_ntnx_live_screen_is_not_the_old_dell_hpe_dictionary(monkeypatch):
    from trading_crew import market_context as mc

    def _screen(query, **_kwargs):
        return {
            "quotes": [
                {"symbol": "NTNX", "marketCap": 1.8e10},
                {"symbol": "PSTG", "marketCap": 2.0e10},
                {"symbol": "NTAP", "marketCap": 2.2e10},
                {"symbol": "CRWD", "marketCap": 8.0e10},
            ]
        }

    monkeypatch.setattr(mc, "_run_equity_screen", _screen)
    uni = mc.discover_peers(
        "NTNX",
        profile=_profile(
            symbol="NTNX", raw_input="NTNX", long_name="Nutanix, Inc.",
            country="United States", currency="USD", exchange="NMS",
            sector="Technology", industry="Software - Infrastructure",
            industry_key="software-infrastructure", market_cap=1.8e10,
            peer_index="^GSPC",
        ),
    )
    assert "PSTG" in uni.symbols
    assert "DELL" not in uni.symbols
    assert "HPE" not in uni.symbols
    assert "IBM" not in uni.symbols


def test_peer_basket_focus_only_when_screen_empty(monkeypatch):
    from trading_crew import market_context as mc

    monkeypatch.setattr(mc, "_info_safe", lambda _s: {})
    monkeypatch.setattr(mc, "_run_equity_screen", lambda *_a, **_k: {"quotes": []})
    uni = mc.discover_peers("ZZZUNKNOWN")
    assert uni.symbols == ["ZZZUNKNOWN"]
    assert uni.method == "focus_only"


def test_format_peer_comparison_computes_alpha_and_flags_mismatch():
    from trading_crew.market_context import (
        PeerRow,
        PeerUniverse,
        format_peer_comparison,
    )

    uni = PeerUniverse(
        symbols=["NTNX", "PSTG", "DELL"],
        method="industry+region",
        industry="Software—Infrastructure",
        region="us",
        notes="test",
    )
    text = format_peer_comparison("NTNX", uni, [
        PeerRow("NTNX", 10.0, 5.0, 1.8e10, 40.0,
                "Software - Infrastructure", "United States"),
        PeerRow("PSTG", 20.0, 8.0, 2.0e10, 30.0,
                "Software - Infrastructure", "United States"),
        PeerRow("DELL", 200.0, 50.0, 8.0e10, 15.0,
                "Computer Hardware", "United States"),
    ])
    assert "INDUSTRY_MISMATCH" in text
    assert "Focus alpha vs median YTD: -10.00 pp" in text
    assert "Median peer YTD (ex-focus, matched industry): +20.00%" in text
    # Hardware outlier must not enter the median the PM will quote.
    assert "Focus alpha vs median YTD: + " not in text
    assert "do not recompute" in text.lower() or "Quote the alpha" in text or "re-call with peers=" in text


# ---------------------------------------------------------------------------
# 5. Brief rendering
# ---------------------------------------------------------------------------


def test_brief_renders_country_currency_exchange_macro_lines(monkeypatch):
    from trading_crew.market_context import get_market_profile
    _patch_yf(
        monkeypatch,
        history_map={},
        info_map={
            "MAZDOCK.NS": {
                "country": "India", "exchange": "NSI",
                "longName": "Mazagon Dock Shipbuilders Limited",
                "sector": "Industrials", "industry": "Aerospace & Defense",
            }
        },
    )
    brief = get_market_profile("MAZDOCK.NS").to_brief()
    # Single brief should be self-contained — country, currency, exchange,
    # peer index, live-peer contract, and macro tickers are all visible.
    assert "India" in brief
    assert "INR" in brief
    assert "^NSEI" in brief
    assert "INR=X" in brief
    assert "Mazagon Dock" in brief
    assert "get_sector_peers" in brief
    assert "not a ticker dictionary" in brief
    assert "COCHINSHIP" not in brief
    assert "DELL" not in brief
