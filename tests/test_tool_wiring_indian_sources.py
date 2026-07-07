"""End-to-end wiring tests: tools.py -> screener_source / upstox_source.

These tests do NOT hit the network — every source function is patched
to return a deterministic payload.  What we're proving is that the
tool layer:

* Routes ``.NS`` / ``.BO`` tickers to screener.in when it's enabled,
  and to yfinance otherwise (screener disabled OR non-Indian symbol).
* Routes Indian OHLCV to upstox when configured, and to yfinance for
  non-Indian tickers.
* Handles screener.in exceptions gracefully — an ``UpstoxError`` /
  ``ScreenerRateLimitError`` returns a clean "returned an error" line
  instead of leaking a stack trace into the LLM prompt.
* The new ``get_shareholding_pattern`` tool exists, routes only for
  Indian tickers, and emits an "N/A" line for others.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from trading_crew import tools
from trading_crew.data_sources import routing, screener_source as ss
from trading_crew.data_sources import upstox_source as us


# ---------------------------------------------------------------------------
# Shared fixtures — reset every tool's memoize cache + routing config
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """Prime env vars + wipe every cache these tests touch.

    The shared tool-memoize cache is already cleared by
    ``tests/conftest.py``; we only need to reset the routing snapshot
    and the two data-source caches here.
    """
    monkeypatch.setenv("TRADINGCREW_SCREENER_ENABLED", "1")
    monkeypatch.setenv("TRADINGCREW_UPSTOX_ENABLED", "1")
    monkeypatch.setenv("UPSTOX_ANALYTICS_TOKEN", "fake-token")
    routing.reset_config_cache()
    ss.clear_cache()
    us.clear_catalog_cache()
    yield
    routing.reset_config_cache()
    ss.clear_cache()
    us.clear_catalog_cache()


def _fake_company_page() -> ss.CompanyPage:
    """A minimal but valid CompanyPage the renderers accept.

    The parser tests already prove the parsers work — here we just need
    an object shaped correctly so we can watch which render path fires.
    """
    return ss.CompanyPage(
        symbol="RELIANCE",
        source_url="https://www.screener.in/company/RELIANCE/consolidated/",
        mode="consolidated",
        long_name="Reliance Industries Ltd",
        about="Test issuer",
        top=ss.TopRatios(raw={"Market Cap": "₹ 1,000 Cr."}, numeric={"Market Cap": 1000.0}),
        quarters=ss.FinancialTable(kind="quarters", columns=["Q1", "Q2"], rows=[("Sales", ["100", "200"])]),
        profit_loss=ss.FinancialTable(kind="profit-loss", columns=["Y1"], rows=[("Sales", ["100"])]),
        balance_sheet=ss.FinancialTable(kind="balance-sheet", columns=["Y1"], rows=[("Equity", ["100"])]),
        cash_flow=ss.FinancialTable(kind="cash-flow", columns=["Y1"], rows=[("CFO", ["10"])]),
        ratios=ss.FinancialTable(kind="ratios", columns=["Y1"], rows=[("ROCE", ["12"])]),
        shareholding=ss.ShareholdingPattern(
            quarterly=ss.FinancialTable(
                kind="shareholding-quarterly",
                columns=["Q1", "Q2"],
                rows=[("Promoters", ["50%", "50%"])],
            ),
        ),
        documents=[ss.DocumentLink(title="Buyback Approval 20 Jun", url="https://example.com/buyback")],
        pros_cons=ss.ProsCons(pros=["debt reduction"], cons=["low RoE"]),
    )


# ---------------------------------------------------------------------------
# get_fundamentals routing
# ---------------------------------------------------------------------------


def test_get_fundamentals_uses_screener_for_indian_ticker():
    with patch.object(ss, "fetch_company", return_value=_fake_company_page()) as mock:
        out = tools.get_fundamentals.run("RELIANCE.NS")
    mock.assert_called_once_with("RELIANCE.NS")
    assert "screener.in" in out
    assert "Reliance Industries Ltd" in out
    assert "Market Cap" in out


def test_get_fundamentals_uses_yfinance_for_us_ticker():
    """Non-Indian tickers must never touch screener.in."""
    fake_info = {"shortName": "Apple Inc.", "marketCap": 3e12, "trailingPE": 32.0}
    ticker_obj = MagicMock()
    ticker_obj.info = fake_info
    with patch.object(ss, "fetch_company") as scr_mock, \
         patch.object(tools.yf, "Ticker", return_value=ticker_obj):
        out = tools.get_fundamentals.run("AAPL")
    scr_mock.assert_not_called()
    assert "Apple Inc." in out
    assert "yfinance" in out


def test_get_fundamentals_falls_back_to_yfinance_when_screener_disabled(monkeypatch):
    """Flipping the env kill-switch during a screener.in outage must
    take Indian tickers back to the yfinance path.
    """
    monkeypatch.setenv("TRADINGCREW_SCREENER_ENABLED", "0")
    routing.reset_config_cache()
    ticker_obj = MagicMock()
    ticker_obj.info = {"shortName": "Reliance"}
    with patch.object(ss, "fetch_company") as scr_mock, \
         patch.object(tools.yf, "Ticker", return_value=ticker_obj):
        out = tools.get_fundamentals.run("RELIANCE.NS")
    scr_mock.assert_not_called()
    assert "yfinance" in out


def test_get_fundamentals_surfaces_screener_rate_limit_gracefully():
    """A 429 from screener.in must NOT crash the tool call — the LLM
    should see a clean warning line so the analyst can proceed with
    the rest of its tools.
    """
    err = ss.ScreenerRateLimitError("rate limited")
    with patch.object(ss, "fetch_company", side_effect=err):
        out = tools.get_fundamentals.run("RELIANCE.NS")
    assert "screener.in returned an error" in out
    assert "ScreenerRateLimitError" in out


# ---------------------------------------------------------------------------
# Statement tools — BS / CF / P&L
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool_fn,statement_id",
    [
        (tools.get_balance_sheet, "balance-sheet"),
        (tools.get_cashflow, "cash-flow"),
        (tools.get_income_statement, "profit-loss"),
    ],
)
def test_statement_tools_route_indian_tickers_to_screener(tool_fn, statement_id):
    with patch.object(ss, "fetch_company", return_value=_fake_company_page()) as mock:
        out = tool_fn.run("RELIANCE.NS")
    mock.assert_called_once()
    # Section anchor of the source URL is captured in the Source line
    assert f"#{statement_id}" in out
    assert "Reliance Industries Ltd" in out


def test_get_balance_sheet_uses_yfinance_for_non_indian():
    ticker_obj = MagicMock()
    ticker_obj.balance_sheet = pd.DataFrame(
        {"latest": [1000, 500, 500]}, index=["Assets", "Liabilities", "Equity"]
    )
    with patch.object(ss, "fetch_company") as scr, \
         patch.object(tools.yf, "Ticker", return_value=ticker_obj):
        out = tools.get_balance_sheet.run("AAPL")
    scr.assert_not_called()
    assert "yfinance balance_sheet" in out


# ---------------------------------------------------------------------------
# get_shareholding_pattern (new tool)
# ---------------------------------------------------------------------------


def test_get_shareholding_pattern_runs_for_indian_ticker():
    with patch.object(ss, "fetch_company", return_value=_fake_company_page()) as mock:
        out = tools.get_shareholding_pattern.run("RELIANCE.NS")
    mock.assert_called_once()
    assert "Shareholding" in out or "Promoters" in out


def test_get_shareholding_pattern_returns_na_for_non_indian():
    """The shareholding tool must not attempt to scrape screener.in
    for non-Indian tickers — it should emit a clear "not applicable"
    line so the fundamentals analyst does not stall.
    """
    with patch.object(ss, "fetch_company") as mock:
        out = tools.get_shareholding_pattern.run("AAPL")
    mock.assert_not_called()
    lower = out.lower()
    assert (
        "not-applicable" in lower
        or "not applicable" in lower
        or "only tracked for indian" in lower
    ), f"expected an N/A phrasing, got: {out!r}"


# ---------------------------------------------------------------------------
# _load_price_history — OHLCV router
# ---------------------------------------------------------------------------


def test_load_price_history_uses_upstox_for_indian_ticker():
    fake_df = pd.DataFrame(
        {"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [100]},
        index=pd.to_datetime(["2026-07-01"], utc=True),
    )
    cf = us.CandleFrame(
        df=fake_df,
        instrument_key="NSE_EQ|INE002A01018",
        unit="days",
        interval="1",
        from_date="2021-07-01",
        to_date="2026-07-01",
    )
    with patch.object(us, "fetch_daily_candles", return_value=cf) as mock:
        df, source_id = tools._load_price_history("RELIANCE.NS", period="5y")
    mock.assert_called_once_with("RELIANCE.NS", lookback_days=5 * 365)
    assert df is fake_df
    assert "upstox v3 historical-candle" in source_id
    assert "NSE_EQ|INE002A01018" in source_id


def test_load_price_history_uses_yfinance_for_non_indian():
    fake_df = pd.DataFrame({"Close": [100]})
    ticker_obj = MagicMock()
    ticker_obj.history.return_value = fake_df
    with patch.object(us, "fetch_daily_candles") as upstox_mock, \
         patch.object(tools.yf, "Ticker", return_value=ticker_obj):
        df, source_id = tools._load_price_history("AAPL", period="5y")
    upstox_mock.assert_not_called()
    assert df is fake_df
    assert "yfinance OHLCV" in source_id


def test_load_price_history_falls_back_to_yfinance_when_upstox_disabled(monkeypatch):
    """Without an analytics token the router must degrade to yfinance
    for Indian tickers too — this is the tokenless-deploy default.
    """
    monkeypatch.delenv("UPSTOX_ANALYTICS_TOKEN", raising=False)
    routing.reset_config_cache()
    ticker_obj = MagicMock()
    ticker_obj.history.return_value = pd.DataFrame({"Close": [100]})
    with patch.object(us, "fetch_daily_candles") as upstox_mock, \
         patch.object(tools.yf, "Ticker", return_value=ticker_obj):
        df, source_id = tools._load_price_history("RELIANCE.NS", period="5y")
    upstox_mock.assert_not_called()
    assert "yfinance" in source_id


# ---------------------------------------------------------------------------
# get_options_summary routing
# ---------------------------------------------------------------------------


def test_get_options_summary_uses_upstox_for_indian_fno_ticker():
    chain = us.OptionChain(
        underlying="RELIANCE",
        expiry="2026-07-31",
        spot=1305.0,
        rows=[
            us.OptionChainRow(strike=1300, call_ltp=45.0, call_oi=3000, put_ltp=12.5, put_oi=4500),
            us.OptionChainRow(strike=1350, call_ltp=15.0, call_oi=5000, put_ltp=42.0, put_oi=1500),
        ],
    )
    with patch.object(us, "fetch_option_chain", return_value=chain) as mock:
        out = tools.get_options_summary.run("RELIANCE.NS")
    assert mock.called
    assert "upstox v3" in out
    assert "Put/Call OI ratio" in out


def test_get_options_summary_yfinance_for_non_indian():
    ticker_obj = MagicMock()
    ticker_obj.options = ()  # no expiries
    with patch.object(us, "fetch_option_chain") as upstox_mock, \
         patch.object(tools.yf, "Ticker", return_value=ticker_obj):
        out = tools.get_options_summary.run("AAPL")
    upstox_mock.assert_not_called()
    assert "No listed options" in out


# ---------------------------------------------------------------------------
# get_corporate_actions augmentation
# ---------------------------------------------------------------------------


def test_get_corporate_actions_augments_with_screener_docs_for_indian_ticker(monkeypatch):
    """For an Indian ticker the corporate-actions tool must:

    1. keep the existing Tavily news search unchanged, AND
    2. append the screener.in ``#documents`` announcements block.

    The buyback announcement is the exact class of signal that the
    Tavily news search sometimes misses on Indian tickers (per the
    BAJAJ-AUTO buyback-miss failure the user reported).
    """
    # Stub the merged news layer so we don't hit the live news APIs
    # (Tavily + SerpAPI).  The screener augmentation is what this test
    # exercises, so returning no web-news rows keeps it focused.
    monkeypatch.setattr(tools, "_web_news_rows", lambda q, max_results=6: [])
    with patch.object(ss, "fetch_company", return_value=_fake_company_page()):
        out = tools.get_corporate_actions.run("RELIANCE.NS")
    assert "Buyback Approval" in out
    assert "screener.in #documents" in out
