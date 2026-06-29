"""Unit tests for the ticker resolver + market-profile + dynamic query layer.

These cover the two user-reported bugs:

1. Indian (NSE) tickers must resolve from bare input — typing
   ``MAZDOCK`` should turn into ``MAZDOCK.NS`` end-to-end, with the
   downstream tools using the canonical symbol.
2. News + macro queries must be ticker-aware — a US name still gets
   Fed/UST/DXY context, but a non-US name gets its own
   country-specific macro themes (RBI, INR, Union Budget, …) and the
   industry-specific seeds (shipbuilder → Strait of Hormuz / Make-in-
   India naval orders).

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
    "X.X.X.X",                   # length OK but multiple dots is suspicious
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
    # "X.X.X.X" is rejected for length (> 12 chars).  Everything else
    # is rejected for whitespace or disallowed punctuation.  In all
    # cases we must NOT probe.
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
    # Industry seeds for semis must be present (TSMC / Taiwan / chip export).
    seeds = " ".join(prof.industry_themes).lower()
    assert "tsmc" in seeds or "chip export" in seeds


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
    # Macro themes must include RBI / INR / Union Budget — not Fed terms.
    themes_joined = " ".join(prof.macro_themes).lower()
    assert "rbi" in themes_joined or "reserve bank" in themes_joined
    assert "inr" in themes_joined or "rupee" in themes_joined.replace("inr", "")
    assert "fed" not in themes_joined
    # Industry seeds must include shipbuilding-specific phrases.
    industry_joined = " ".join(prof.industry_themes).lower()
    assert "shipbuilding" in industry_joined or "naval" in industry_joined
    assert "hormuz" in industry_joined or "shipping" in industry_joined


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
    assert "rbi" in q or "reserve bank" in q or "rupee" in q or "inr" in q
    assert "federal reserve" not in q
    assert "treasury yield" not in q


def test_geopolitical_query_for_shipbuilder_does_not_default_to_china_taiwan(monkeypatch):
    """The old hard-coded ``"China Taiwan export controls"`` is gone —
    a shipbuilder gets shipping / naval / Hormuz instead."""
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
    q = get_market_profile("MAZDOCK.NS").geopolitical_query().lower()
    assert "shipbuilding" in q or "naval" in q
    assert "hormuz" in q or "shipping" in q
    # We don't insist China/Taiwan are absent (defence sector globally is
    # entangled with both), but they must NOT be the *anchor* of the
    # query — the company name + industry words come first.
    assert q.startswith("mazagon dock shipbuilders limited")


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
    q = get_market_profile("NVDA").supply_chain_query().lower()
    assert "nvidia" in q
    # Semis profile keeps TSMC / ASML / Taiwan seeds.
    assert "tsmc" in q or "asml" in q or "taiwan" in q


# ---------------------------------------------------------------------------
# 4. Peer basket
# ---------------------------------------------------------------------------


def test_peer_basket_for_indian_shipbuilder_returns_indian_peers():
    from trading_crew.market_context import peer_basket
    peers = peer_basket("MAZDOCK.NS")
    assert "MAZDOCK.NS" in peers
    # Must include Indian comparables — NOT US semis.
    assert any(p.endswith(".NS") for p in peers if p != "MAZDOCK.NS")
    assert all("NVDA" not in p for p in peers)


def test_peer_basket_falls_back_to_self_when_unknown():
    from trading_crew.market_context import peer_basket
    assert peer_basket("ZZZUNKNOWN") == ["ZZZUNKNOWN"]


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
    # peer index, peer basket, and macro tickers are all visible to the
    # analyst so they can phrase searches grounded in the brief.
    assert "India" in brief
    assert "INR" in brief
    assert "^NSEI" in brief
    assert "INR=X" in brief
    assert "Mazagon Dock" in brief
