"""All 17 tools used across the analyst team and the trader.

Tool ownership (also enforced when wiring agents in ``crew.py``):

* Market Analyst:        ``get_stock_data``, ``get_indicators``
* Social Analyst:        ``get_news``
* News Analyst:          ``get_news``, ``get_global_news``,
                         ``get_insider_transactions``
* Fundamentals Analyst:  ``get_fundamentals``, ``get_balance_sheet``,
                         ``get_cashflow``, ``get_income_statement``
* Macro Analyst:         ``get_macro_data``, ``get_global_news``
* Geopolitical Analyst:  ``get_geopolitical_news``, ``get_global_news``,
                         ``get_supply_chain_risk``
* Sector / Peer:         ``get_sector_peers``, ``get_indicators``
* Quant / Options:       ``get_options_summary``,
                         ``get_analyst_recommendations``
* Trader:                ``get_event_proximity``, ``backtest_setup``

Market / fundamentals / macro / options data is sourced from ``yfinance``;
news comes from Tavily web search.

Provenance
----------
Every tool's response ends with a ``Source: <identifier> · retrieved
<UTC ISO timestamp>`` line so the analyst that cites the data can copy
that identifier into a ``[source: <identifier>]`` tag next to any
quantitative claim it lifts.  The Reflective Critic uses these tags
(and the deduplicated ``sources`` list the Portfolio Manager emits) to
distinguish real fetches from hallucinated numbers.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import random
import time

import pandas as pd
import yfinance as yf
from crewai.tools import tool
from tavily import TavilyClient

logger = logging.getLogger(__name__)


def _with_retry(
    fn,
    *args,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 4.0,
    label: str = "",
    **kwargs,
):
    """Phase 2E — retry a network call with jittered exponential backoff.

    The Tavily + yfinance wrappers occasionally raise on transient
    timeouts / 502s; one retry usually clears it, three is plenty.  We
    keep the implementation dependency-free (no ``tenacity``) — plain
    ``time.sleep`` + ``random.uniform`` so the failure mode stays
    deterministic enough to debug.
    """
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if i == attempts - 1:
                break
            delay = min(max_delay, base_delay * (2 ** i)) + random.uniform(0, base_delay)
            logger.warning(
                "Tool retry %s/%s after %.2fs failed: %s (%s)",
                i + 1, attempts, delay, label or fn.__name__, exc,
            )
            time.sleep(delay)
    # Re-raise the last exception once attempts are exhausted.
    raise last_exc  # type: ignore[misc]

from trading_crew.market_context import (
    get_market_profile,
    peer_basket,
    resolve_ticker,
)


# ---------------------------------------------------------------------------
# Provenance helpers — every tool MUST end its output with one of these so
# downstream agents can quote the identifier in an inline [source: …] tag.
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    """UTC ISO-8601 minute-resolution timestamp (e.g. ``2026-06-12T14:32Z``).

    Minute resolution is enough for an audit trail — second precision
    would bloat the rationale tags without helping the critic.
    """
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ")


def _source_line(identifier: str) -> str:
    """Render the standard ``Source: …`` footer the analyst can copy
    verbatim into a ``[source: <identifier>]`` inline tag.

    Keep ``identifier`` short and human-readable — long URLs are fine, but
    avoid newlines or square brackets so the tag parser stays simple.
    """
    safe = identifier.replace("\n", " ").replace("[", "(").replace("]", ")")
    return f"\nSource: {safe} · retrieved {_utc_now_iso()}"

# ---------------------------------------------------------------------------
# Tavily client (constructed lazily so import works even without the key set)
# ---------------------------------------------------------------------------

_tavily_client: TavilyClient | None = None


def _tavily() -> TavilyClient:
    global _tavily_client
    if _tavily_client is None:
        api_key = os.environ["TAVILY_API_KEY"]
        _tavily_client = TavilyClient(api_key=api_key)
    return _tavily_client


def _tavily_news(query: str, max_results: int = 5) -> str:
    res = _with_retry(
        _tavily().search,
        query,
        max_results=max_results,
        topic="news",
        search_depth="basic",
        label=f"tavily.search({query!r})",
    )
    rows = []
    for r in res.get("results", []):
        snippet = (r.get("content") or "").strip().replace("\n", " ")[:240]
        rows.append(f"- {r['title']}\n  {r['url']}\n  {snippet}")
    body = "\n".join(rows) if rows else "No news found."
    # The Source line names the query so analysts can cite "Tavily News
    # · q='NTNX earnings'" — and each row already carries the article URL
    # which the analyst can use as a more granular [source: <url>] tag.
    return body + _source_line(f"Tavily News Search · q='{query}'")


# ---------------------------------------------------------------------------
# Market Analyst tools
# ---------------------------------------------------------------------------


@tool("get_stock_data")
def get_stock_data(ticker: str) -> str:
    """Retrieve 5-year OHLCV history for a given ticker symbol.

    Returns a layered summary so the analyst gets BOTH the long-horizon
    context (multi-year returns, 52-week range, drawdown, monthly
    aggregates) AND short-term detail (recent daily sessions) without
    flooding the LLM with 1200+ raw daily rows.  The output sections are:

      1. ``5-YEAR OVERVIEW`` — current price, 52-week high/low (with
         dates), returns at 1m/3m/6m/1y/3y/5y, trailing annualized
         volatility, and max drawdown.
      2. ``MONTHLY OHLCV`` — resampled to month-end (~60 rows for 5y)
         so the analyst can see the long-term trend at a glance.
      3. ``DAILY OHLCV`` — the last 30 daily sessions for short-term
         price action and key recent levels.
    """
    df = _with_retry(
        yf.Ticker(ticker).history,
        period="5y",
        label=f"yfinance.history({ticker}, 5y)",
    )
    if df.empty:
        return f"No price data for {ticker}." + _source_line(
            f"yfinance OHLCV {ticker} (empty)"
        )

    df.index = pd.to_datetime(df.index)
    close = df["Close"].dropna()
    if close.empty:
        return f"No close data for {ticker}." + _source_line(
            f"yfinance OHLCV {ticker} (5y, no close)"
        )

    last_px = float(close.iloc[-1])
    last_date = close.index[-1].strftime("%Y-%m-%d")

    def _ret_n_days(n: int) -> str:
        if len(close) <= n:
            return "n/a"
        pct = (close.iloc[-1] / close.iloc[-n - 1] - 1) * 100
        return f"{pct:+.1f}%"

    ret_5y_pct = (close.iloc[-1] / close.iloc[0] - 1) * 100

    # 52-week high/low with dates.  Use the trailing 252 sessions when
    # available; otherwise fall back to the full window so the readout
    # still works for IPOs / shorter histories.
    window = close.iloc[-252:] if len(close) >= 252 else close
    hi_val = float(window.max())
    lo_val = float(window.min())
    hi_date = window.idxmax().strftime("%Y-%m-%d")
    lo_date = window.idxmin().strftime("%Y-%m-%d")

    daily_ret = close.pct_change().dropna()
    if len(daily_ret) >= 252:
        vol_1y_ann = float(daily_ret.iloc[-252:].std() * (252 ** 0.5) * 100)
        vol_str = f"{vol_1y_ann:.1f}%"
    else:
        vol_str = "n/a"

    # Max drawdown over the whole 5y window (close-to-close basis).
    roll_max = close.cummax()
    drawdown = (close / roll_max - 1.0) * 100
    max_dd = float(drawdown.min())
    max_dd_date = drawdown.idxmin().strftime("%Y-%m-%d")

    # Monthly aggregates (~60 rows for 5y) — long-term trend at a glance.
    monthly = (
        df[["Open", "High", "Low", "Close", "Volume"]]
        .resample("ME")
        .agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        })
        .dropna(how="all")
    )
    monthly.index = monthly.index.strftime("%Y-%m")

    summary_lines = [
        f"5-YEAR OVERVIEW for {ticker}",
        f"  Current price: {last_px:.2f}  (as of {last_date})",
        f"  52-week high : {hi_val:.2f}  on {hi_date}",
        f"  52-week low  : {lo_val:.2f}  on {lo_date}",
        (
            f"  Returns: 1m={_ret_n_days(21)}  3m={_ret_n_days(63)}  "
            f"6m={_ret_n_days(126)}  1y={_ret_n_days(252)}  "
            f"3y={_ret_n_days(252*3)}  5y={ret_5y_pct:+.1f}%"
        ),
        f"  Annualised volatility (1y trailing): {vol_str}",
        f"  Max drawdown (5y): {max_dd:.1f}%  (trough {max_dd_date})",
    ]

    daily_tail = df[["Open", "High", "Low", "Close", "Volume"]].tail(30).round(2)

    return (
        "\n".join(summary_lines)
        + f"\n\nMONTHLY OHLCV ({len(monthly)} months, 5y window):\n"
        + monthly.round(2).to_string()
        + f"\n\nDAILY OHLCV (last {len(daily_tail)} sessions):\n"
        + daily_tail.to_string()
        + _source_line(f"yfinance OHLCV {ticker} (5y, monthly + recent daily)")
    )


@tool("get_indicators")
def get_indicators(ticker: str) -> str:
    """Retrieve a technical-indicator snapshot computed over 5 years of
    daily history.

    Returns price, SMA20 / SMA50 / SMA200, RSI14, 60-day return, and
    1-year return.  The 5-year window is needed so SMA200 and the 1-year
    return are well-defined for the Market Analyst's regime call —
    short windows force them to ``n/a`` for any IPO younger than the
    window."""
    df = _with_retry(
        yf.Ticker(ticker).history,
        period="5y",
        label=f"yfinance.history({ticker}, 5y)",
    )
    if df.empty:
        return f"No price data for {ticker}." + _source_line(
            f"yfinance OHLCV {ticker} (empty)"
        )
    close = df["Close"].dropna()
    if close.empty:
        return f"No close data for {ticker}." + _source_line(
            f"yfinance OHLCV {ticker} (5y, no close)"
        )

    def _fmt(val: float) -> str:
        # NaN-safe formatter — short histories (IPOs, etc.) produce NaN
        # on rolling windows longer than the data; surface that as "n/a"
        # rather than the literal ``nan`` string the LLM might mis-cite.
        return f"{val:.2f}" if pd.notna(val) else "n/a"

    sma20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else float("nan")
    sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else float("nan")
    sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else float("nan")

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = (100 - 100 / (1 + gain / loss)).iloc[-1] if len(close) >= 15 else float("nan")

    ret60 = (
        (close.iloc[-1] / close.iloc[-61] - 1) * 100
        if len(close) >= 61
        else float("nan")
    )
    ret252 = (
        (close.iloc[-1] / close.iloc[-253] - 1) * 100
        if len(close) >= 253
        else float("nan")
    )

    def _fmt_pct(val: float) -> str:
        return f"{val:.2f}%" if pd.notna(val) else "n/a"

    return (
        f"{ticker} indicators: price={close.iloc[-1]:.2f}, "
        f"SMA20={_fmt(sma20)}, SMA50={_fmt(sma50)}, SMA200={_fmt(sma200)}, "
        f"RSI14={_fmt(rsi)}, "
        f"60d_return={_fmt_pct(ret60)}, 1y_return={_fmt_pct(ret252)}"
        + _source_line(f"yfinance OHLCV {ticker} (5y, indicators)")
    )


# ---------------------------------------------------------------------------
# Social / News / Insider tools
# ---------------------------------------------------------------------------


@tool("get_news")
def get_news(ticker: str) -> str:
    """Retrieve recent news data for a given ticker symbol (company-specific
    headlines plus social-style sentiment cues).

    The search query is built from the ticker's *long name* (e.g.
    "Mazagon Dock Shipbuilders Ltd") rather than just the symbol, so we
    don't lose hits to ambiguous tickers (e.g. ``MAZDOCK`` returning
    German-investor-confidence headlines).
    """
    canonical = resolve_ticker(ticker)
    profile = get_market_profile(canonical)
    name = profile.long_name or canonical
    return _tavily_news(f"{name} {canonical} stock news sentiment last week", max_results=6)


@tool("get_global_news")
def get_global_news(ticker: str = "", query: str = "") -> str:
    """Retrieve global / macro news in the **ticker's own country**.

    Prefer calling with the ticker symbol so the search is grounded in
    the right macro regime — for a US name we surface Fed / UST / DXY
    headlines; for an Indian name we surface RBI / INR / Union Budget /
    crude-import headlines.

    ``query`` is an optional override; when set it takes precedence
    over the ticker-derived themes. Use it only when the analyst
    layer has a specific catalyst in mind ("OPEC meeting", "FOMC dots").
    """
    if query:
        return _tavily_news(query, max_results=6)
    if not ticker:
        # No ticker, no override → return a *neutral* global query rather
        # than silently defaulting to US Fed terms.
        return _tavily_news(
            "global markets macro overview central bank policy currencies",
            max_results=6,
        )
    canonical = resolve_ticker(ticker)
    profile = get_market_profile(canonical)
    return _tavily_news(profile.country_news_query(), max_results=6)


@tool("get_insider_transactions")
def get_insider_transactions(ticker: str) -> str:
    """Retrieve insider transaction information about a company (executives
    or directors buying/selling their own stock)."""
    t = yf.Ticker(ticker)
    df = getattr(t, "insider_transactions", None)
    src = _source_line(f"yfinance insider_transactions {ticker}")
    if df is None or len(df) == 0:
        return f"No insider transactions for {ticker}." + src
    return f"Insider transactions for {ticker}:\n" + df.head(10).to_string() + src


# ---------------------------------------------------------------------------
# Fundamentals tools
# ---------------------------------------------------------------------------


@tool("get_fundamentals")
def get_fundamentals(ticker: str) -> str:
    """Retrieve comprehensive fundamental data for a given ticker symbol
    (P/E, EPS, market cap, profitability, growth, etc.)."""
    info = yf.Ticker(ticker).info
    keys = [
        "shortName", "sector", "industry", "marketCap", "currentPrice",
        "trailingPE", "forwardPE", "trailingEps", "priceToBook",
        "profitMargins", "operatingMargins", "returnOnEquity", "returnOnAssets",
        "debtToEquity", "freeCashflow", "totalRevenue",
        "earningsGrowth", "revenueGrowth", "dividendYield",
    ]
    return json.dumps({k: info.get(k) for k in keys}, default=str, indent=2) + _source_line(
        f"yfinance .info {ticker} (fundamentals)"
    )


def _latest_column(df: pd.DataFrame, n: int = 20) -> str:
    if df is None or df.empty:
        return "(no data)"
    return df.iloc[:, 0].head(n).to_string()


@tool("get_balance_sheet")
def get_balance_sheet(ticker: str) -> str:
    """Retrieve balance-sheet data (assets, liabilities, equity) for a given
    ticker symbol — most-recent fiscal period."""
    return (
        f"Balance sheet for {ticker} (latest period):\n"
        + _latest_column(yf.Ticker(ticker).balance_sheet)
        + _source_line(f"yfinance balance_sheet {ticker} (latest period)")
    )


@tool("get_cashflow")
def get_cashflow(ticker: str) -> str:
    """Retrieve cash-flow-statement data for a given ticker symbol —
    most-recent fiscal period."""
    return (
        f"Cash flow for {ticker} (latest period):\n"
        + _latest_column(yf.Ticker(ticker).cashflow)
        + _source_line(f"yfinance cashflow {ticker} (latest period)")
    )


@tool("get_income_statement")
def get_income_statement(ticker: str) -> str:
    """Retrieve income-statement data (revenue, costs, net income) for a
    given ticker symbol — most-recent fiscal period."""
    return (
        f"Income statement for {ticker} (latest period):\n"
        + _latest_column(yf.Ticker(ticker).income_stmt)
        + _source_line(f"yfinance income_stmt {ticker} (latest period)")
    )


# ---------------------------------------------------------------------------
# Macro Analyst tools
# ---------------------------------------------------------------------------


@tool("get_macro_data")
def get_macro_data(ticker: str = "") -> str:
    """Retrieve a country-aware macro basket (rates / FX / vol / commodities).

    Pass the analysis ticker so the basket is chosen for the right
    country: US → 10y UST, DXY, VIX, WTI, gold;
    India → Nifty, India VIX, INR/USD, Sensex, Brent;
    UK → 10y gilt proxy, GBP/USD, FTSE 100, Brent, gold;
    Hong Kong → HSI, USD/HKD, CSI 300, Brent;
    Japan → Nikkei, USD/JPY, TOPIX, Brent.

    When ``ticker`` is omitted we fall back to the US basket — but
    that's a degradation, not the intended path.
    """
    if ticker:
        canonical = resolve_ticker(ticker)
        profile = get_market_profile(canonical)
        macro = profile.macro_tickers
        country = profile.country
    else:
        from trading_crew.market_context import _COUNTRY_PROFILE  # internal default basket
        country = "United States"
        macro = dict(_COUNTRY_PROFILE[country]["macro_tickers"])  # type: ignore[arg-type]

    rows = []
    for label, sym in macro.items():
        df = yf.Ticker(sym).history(period="2mo")
        if df is None or df.empty:
            rows.append(f"- {label} ({sym}): no data")
            continue
        last = df["Close"].iloc[-1]
        ago = df["Close"].iloc[0]
        chg = (last / ago - 1) * 100
        rows.append(
            f"- {label} ({sym}): {last:.2f} ({chg:+.2f}% over ~30 trading days)"
        )
    syms = ", ".join(macro.values())
    return (
        f"Macro snapshot ({country}):\n"
        + "\n".join(rows)
        + _source_line(f"yfinance OHLCV macro basket ({country}) [{syms}] (2mo)")
    )


# ---------------------------------------------------------------------------
# Sector / Peer Analyst tools
# ---------------------------------------------------------------------------


@tool("get_sector_peers")
def get_sector_peers(ticker: str) -> str:
    """Compare a ticker to its sector peers across YTD price return,
    market cap, and trailing P/E.

    Peer baskets come from :mod:`trading_crew.market_context` and now
    include Indian (NSE) and other non-US clusters (shipbuilders,
    Indian banks, Indian IT, oil & gas) on top of the original US
    sets.  If no curated basket exists for the symbol we degrade
    gracefully to a single-line "no peers configured" entry rather
    than hide the gap.
    """
    canonical = resolve_ticker(ticker)
    peers = peer_basket(canonical)
    rows = ["Sector peer comparison:"]
    for sym in peers:
        t = yf.Ticker(sym)
        hist = t.history(period="ytd")
        if hist is None or hist.empty:
            rows.append(f"- {sym}: no data")
            continue
        ytd = (hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100
        try:
            info = t.info or {}
        except Exception:
            info = {}
        mcap = info.get("marketCap")
        pe = info.get("trailingPE")
        marker = "  <-- focus" if sym == canonical else ""
        rows.append(f"- {sym}: YTD={ytd:+.2f}%, mcap={mcap}, PE={pe}{marker}")
    return "\n".join(rows) + _source_line(
        f"yfinance OHLCV + .info peer basket [{', '.join(peers)}] (YTD)"
    )


# ---------------------------------------------------------------------------
# Geopolitical Analyst tools
# ---------------------------------------------------------------------------


@tool("get_geopolitical_news")
def get_geopolitical_news(ticker: str) -> str:
    """Retrieve geopolitical news (tariffs, export controls, sanctions,
    regulatory action, regional conflict) relevant to the ticker.

    The query is built dynamically from the ticker's country + industry
    seeds — an Indian shipbuilder gets "Strait of Hormuz Red Sea naval
    procurement" instead of the previous hard-coded "China Taiwan
    export controls".
    """
    canonical = resolve_ticker(ticker)
    profile = get_market_profile(canonical)
    return _tavily_news(profile.geopolitical_query(), max_results=6)


@tool("get_supply_chain_risk")
def get_supply_chain_risk(ticker: str) -> str:
    """Retrieve supply-chain and customer-concentration risk for a given
    ticker — top customers, supplier dependencies, geographic revenue
    concentration, and any disclosed concentration risks.

    Query terms are derived from the ticker's industry profile rather
    than a single hard-coded "TSMC China dependency" string.  A semis
    name still picks up TSMC + ASML; a shipbuilder picks up naval
    contract pipeline + ship-insurance war-risk premia instead.
    """
    canonical = resolve_ticker(ticker)
    profile = get_market_profile(canonical)
    return _tavily_news(profile.supply_chain_query(), max_results=6)


# ---------------------------------------------------------------------------
# Cross-cutting research brief
# ---------------------------------------------------------------------------


@tool("get_market_context")
def get_market_context(ticker: str) -> str:
    """One-shot research brief for ``ticker`` — country, sector, peer
    basket, macro basket, and country/industry news seeds.

    Call this FIRST so downstream analyst queries are grounded in the
    right country and industry.  The brief deliberately includes the
    yfinance symbols of the macro basket and peers so subsequent tool
    calls (``get_macro_data``, ``get_sector_peers``) hit the same
    sources you cite in the analyst report.

    For a non-US name the brief makes the country-specific themes
    explicit (e.g. RBI policy, INR/USD, Strait of Hormuz, Make-in-India
    defence orders for an Indian shipbuilder) so the news / macro
    analysts don't fall back to US-centric defaults.
    """
    canonical = resolve_ticker(ticker)
    profile = get_market_profile(canonical)
    return profile.to_brief() + _source_line(
        f"yfinance .info {canonical} + curated profile"
    )


# ---------------------------------------------------------------------------
# Quant / Options Analyst tools
# ---------------------------------------------------------------------------


@tool("get_options_summary")
def get_options_summary(ticker: str) -> str:
    """Retrieve nearest-expiry options summary (ATM call/put implied
    volatility, total open interest, put/call ratio) for a given ticker."""
    t = yf.Ticker(ticker)
    expiries = t.options
    if not expiries:
        return f"No listed options for {ticker}." + _source_line(
            f"yfinance .options {ticker} (no expiries)"
        )
    expiry = expiries[0]
    chain = t.option_chain(expiry)
    spot = t.history(period="5d")["Close"].iloc[-1]
    calls, puts = chain.calls, chain.puts
    atm_call = calls.iloc[(calls["strike"] - spot).abs().argsort()[:1]]
    atm_put = puts.iloc[(puts["strike"] - spot).abs().argsort()[:1]]
    call_oi = int(calls["openInterest"].fillna(0).sum())
    put_oi = int(puts["openInterest"].fillna(0).sum())
    pcr = put_oi / max(call_oi, 1)
    return (
        f"Options summary for {ticker} (expiry {expiry}, spot {spot:.2f}):\n"
        f"- ATM call IV={float(atm_call['impliedVolatility'].iloc[0]):.2%}, "
        f"strike={float(atm_call['strike'].iloc[0])}\n"
        f"- ATM put  IV={float(atm_put['impliedVolatility'].iloc[0]):.2%}, "
        f"strike={float(atm_put['strike'].iloc[0])}\n"
        f"- Total call OI={call_oi}, put OI={put_oi}, put/call ratio={pcr:.2f}"
        + _source_line(f"yfinance option_chain {ticker} expiry={expiry}")
    )


@tool("get_analyst_recommendations")
def get_analyst_recommendations(ticker: str) -> str:
    """Retrieve sell-side analyst recommendations and consensus price
    targets for a given ticker."""
    t = yf.Ticker(ticker)
    rec = getattr(t, "recommendations_summary", None)
    if rec is None:
        rec = t.recommendations
    info = t.info
    spot = info.get("currentPrice")
    target_mean = info.get("targetMeanPrice")
    target_high = info.get("targetHighPrice")
    target_low = info.get("targetLowPrice")
    rec_text = "(no recommendation history)"
    if rec is not None and len(rec) > 0:
        rec_text = rec.head(8).to_string()
    return (
        f"Analyst consensus for {ticker}:\n"
        f"- Spot: {spot}, target mean: {target_mean}, "
        f"high: {target_high}, low: {target_low}\n"
        f"- Recent rating distribution:\n{rec_text}"
        + _source_line(f"yfinance recommendations_summary {ticker}")
    )


# ---------------------------------------------------------------------------
# Trader tools (catalyst proximity + historical base-rate backtest)
# ---------------------------------------------------------------------------


@tool("get_event_proximity")
def get_event_proximity(ticker: str) -> str:
    """Retrieve the next earnings date and recent earnings-surprise history
    so the trader can size around catalyst dates. Returns next-earnings
    date, days-to-earnings, EPS/revenue estimates and the last 4 surprise
    %-values."""
    t = yf.Ticker(ticker)
    cal = t.calendar or {}
    ed = t.earnings_dates

    next_dates = cal.get("Earnings Date") or []
    next_d = next_dates[0] if isinstance(next_dates, list) and next_dates else None
    days = "(unknown)"
    if next_d is not None:
        d = next_d.date() if hasattr(next_d, "date") else next_d
        days = (d - _dt.date.today()).days

    eps_est = (cal.get("Earnings Average"), cal.get("Earnings Low"), cal.get("Earnings High"))
    rev_est = (cal.get("Revenue Average"), cal.get("Revenue Low"), cal.get("Revenue High"))

    surp_text = "(no surprise history)"
    if ed is not None and len(ed) > 0 and "Surprise(%)" in ed.columns:
        surp_text = ed["Surprise(%)"].dropna().head(4).to_string()

    return (
        f"Catalyst proximity for {ticker}:\n"
        f"- Next earnings: {next_d}, days_to_event: {days}\n"
        f"- EPS estimate (avg/low/high): {eps_est}\n"
        f"- Revenue estimate (avg/low/high): {rev_est}\n"
        f"- Last 4 surprises (%):\n{surp_text}"
        + _source_line(f"yfinance .calendar + .earnings_dates {ticker}")
    )


def _simulate_horizon(closes, horizon: int, tgt: float, stp: float):
    """Walk one (target, stop, horizon) combo and return per-trade stats.

    Returns ``(hit_rate, expectancy, payoff_ratio, avg, median, total,
    wins, losses, timeouts, avg_win, avg_loss)`` so the caller can render
    the table and compute a composite confidence-cap.

    Expectancy (per-trade expected pct return) is the *risk-adjusted*
    base rate the PM should care about — a 25% hit-rate with a 4:1
    payoff ratio still has positive expectancy and should NOT collapse
    confidence to a hard floor.
    """
    n = len(closes)
    wins = losses = timeouts = 0
    win_returns: list[float] = []
    loss_returns: list[float] = []
    timeout_returns: list[float] = []
    all_returns: list[float] = []
    for i in range(n - horizon):
        entry = closes[i]
        outcome = None
        for j in range(1, horizon + 1):
            r = (closes[i + j] / entry) - 1
            if r >= tgt:
                wins += 1
                win_returns.append(r * 100)
                all_returns.append(r * 100)
                outcome = "win"
                break
            if r <= -stp:
                losses += 1
                loss_returns.append(r * 100)
                all_returns.append(r * 100)
                outcome = "loss"
                break
        if outcome is None:
            timeouts += 1
            tor = (closes[i + horizon] / entry - 1) * 100
            timeout_returns.append(tor)
            all_returns.append(tor)

    total = wins + losses + timeouts
    if total == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0.0, 0.0)
    hit_rate = wins / total
    avg = sum(all_returns) / len(all_returns)
    rs = sorted(all_returns)
    median = rs[len(rs) // 2]
    avg_win = (sum(win_returns) / len(win_returns)) if win_returns else 0.0
    avg_loss = (sum(loss_returns) / len(loss_returns)) if loss_returns else 0.0
    payoff = (avg_win / abs(avg_loss)) if avg_loss < 0 else float("inf") if avg_win > 0 else 0.0
    # Per-trade expectancy in % (loss is negative).  Adds the timeout
    # bucket as a partial outcome so trades that drift sideways for the
    # full horizon are not silently dropped from the base rate.
    timeout_mean = (sum(timeout_returns) / len(timeout_returns)) if timeout_returns else 0.0
    expectancy = (
        hit_rate * avg_win
        + (losses / total) * avg_loss
        + (timeouts / total) * timeout_mean
    )
    return (hit_rate, expectancy, payoff, avg, median, total, wins, losses, timeouts, avg_win, avg_loss)


@tool("backtest_setup")
def backtest_setup(
    ticker: str,
    horizon_days: int,
    target_pct: float,
    stop_pct: float,
) -> str:
    """Score a proposed trade plan against historical analogous setups.

    Walks 5 years of daily price history. For each starting day, simulates
    entry at close, exit at +target_pct, exit at -stop_pct, or timeout
    after horizon_days — whichever comes first.

    Returns the per-horizon table AND a multi-horizon panel covering
    short-term (the trader's chosen horizon), 60d, 120d, and 252d so the
    PM can weigh both short-term technicals and the long-term thesis.
    For each horizon we surface:

    * hit-rate                 — base rate the target gets hit
    * payoff ratio             — avg_win / |avg_loss|
    * expectancy (per trade)   — risk-adjusted base rate; can be positive
                                 even with a sub-40% hit-rate when the
                                 payoff ratio is favourable
    * avg / median realized %

    This richer view is what lets the PM avoid the "hit-rate < 40% =>
    NEUTRAL by default" failure mode.  Whenever expectancy > 0 with
    payoff >= 1.5 on any horizon, the PM is allowed to express full
    conviction even if the short-term hit-rate is low.
    """
    df = yf.Ticker(ticker).history(period="5y")
    if df.empty or len(df) < horizon_days + 30:
        return f"Insufficient history for {ticker} backtest." + _source_line(
            f"yfinance OHLCV {ticker} (5y, insufficient for backtest)"
        )
    closes = df["Close"].values
    n = len(closes)
    tgt = abs(float(target_pct)) / 100.0
    stp = abs(float(stop_pct)) / 100.0

    short_h = max(5, int(horizon_days))
    horizons = sorted({short_h, 60, 120, 252})
    # Don't try to simulate a horizon longer than what the history can support.
    horizons = [h for h in horizons if h + 30 <= n]
    if not horizons:
        horizons = [short_h]

    rows = []
    for h in horizons:
        hit_rate, expectancy, payoff, avg, median, total, wins, losses, timeouts, avg_win, avg_loss = _simulate_horizon(
            closes, h, tgt, stp
        )
        label = f"{h}d" + (" (trader)" if h == short_h else "")
        rows.append({
            "h": h,
            "label": label,
            "hit_rate": hit_rate,
            "expectancy": expectancy,
            "payoff": payoff,
            "avg": avg,
            "median": median,
            "total": total,
            "wins": wins,
            "losses": losses,
            "timeouts": timeouts,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
        })

    # Build the legacy "Backtest for ..." block on the trader's chosen horizon
    # so existing prompt regexes / log parsers keep working.
    primary = next((r for r in rows if r["h"] == short_h), rows[0])
    primary_total = primary["total"]
    legacy = (
        f"Backtest for {ticker} (horizon={primary['h']}d, "
        f"target=+{tgt*100:.1f}%, stop=-{stp*100:.1f}%):\n"
        f"- N samples: {primary_total}\n"
        f"- Wins (target hit): {primary['wins']} ({primary['hit_rate']*100:.1f}%)\n"
        f"- Stops hit: {primary['losses']} ({(primary['losses']/primary_total*100) if primary_total else 0:.1f}%)\n"
        f"- Timeouts: {primary['timeouts']} ({(primary['timeouts']/primary_total*100) if primary_total else 0:.1f}%)\n"
        f"- Avg realized return: {primary['avg']:+.2f}%, median {primary['median']:+.2f}%\n"
        f"- Avg win: {primary['avg_win']:+.2f}%, avg loss: {primary['avg_loss']:+.2f}%\n"
        f"- Payoff ratio: {primary['payoff']:.2f}, per-trade expectancy: {primary['expectancy']:+.2f}%"
    )

    # Multi-horizon panel — PM uses this for the short-vs-long view.
    multi_lines = ["", "Multi-horizon panel (short-term vs long-term, same target/stop):"]
    multi_lines.append(
        "  horizon  | hit-rate | payoff | expectancy | avg     | median"
    )
    for r in rows:
        multi_lines.append(
            f"  {r['label']:<9}| {r['hit_rate']*100:6.1f}%  | {r['payoff']:>5.2f}x | "
            f"{r['expectancy']:+6.2f}%   | {r['avg']:+6.2f}% | {r['median']:+6.2f}%"
        )

    # Headline: best expectancy + best hit-rate across the panel so the
    # PM can quote them directly without re-doing the math.
    best_expectancy = max(rows, key=lambda r: r["expectancy"])
    best_hit = max(rows, key=lambda r: r["hit_rate"])
    headline = [
        "",
        f"Best expectancy: {best_expectancy['expectancy']:+.2f}% on {best_expectancy['label']} "
        f"(hit-rate {best_expectancy['hit_rate']*100:.1f}%, payoff {best_expectancy['payoff']:.2f}x).",
        f"Best hit-rate:   {best_hit['hit_rate']*100:.1f}% on {best_hit['label']} "
        f"(expectancy {best_hit['expectancy']:+.2f}%).",
    ]

    return (
        legacy
        + "\n"
        + "\n".join(multi_lines)
        + "\n"
        + "\n".join(headline)
        + _source_line(
            f"backtest_setup({ticker}, "
            f"h={'+'.join(f'{h}d' for h in horizons)}, "
            f"tgt={tgt*100:.1f}%, stop={stp*100:.1f}%) on yfinance 5y OHLCV"
        )
    )


# ---------------------------------------------------------------------------
# M3 — Episodic memory retrieval (embargo-aware)
# ---------------------------------------------------------------------------


def _rl_policy_recommendation_impl(ticker: str, as_of: str = "") -> str:
    """Plain-Python implementation shared by the equity + commodity tools."""
    from datetime import datetime
    import pandas as pd

    ticker = (ticker or "").upper().strip()
    if not ticker:
        return "rl_policy_recommendation: ticker is required."

    try:
        from trading_crew.agentic.rl import load_policy
    except Exception as exc:
        return f"RL policy module unavailable: {exc}"

    client = load_policy(ticker)
    if client is None:
        return (
            f"No RL policy has been promoted for {ticker} yet. "
            "Open the **RL Training** tab in the UI to train one and "
            "promote it once eval metrics look reasonable."
        )

    # Fetch a window of OHLCV up to ``as_of`` so the policy sees the
    # same data the analyst sees.  We deliberately pull from the same
    # ``_fetch_ohlcv`` the rest of the UI uses, so the audit trail is
    # consistent across tools.
    as_of_dt = datetime.utcnow()
    if as_of:
        try:
            as_of_dt = datetime.strptime(as_of[:10], "%Y-%m-%d")
        except ValueError:
            pass
    try:
        from web.backend.charts import _fetch_ohlcv  # type: ignore
        df = _fetch_ohlcv(ticker, as_of_dt, lookback_days=180)
    except Exception as exc:
        return f"OHLCV fetch failed for {ticker}: {exc}"

    if df is None or df.empty:
        return f"No OHLCV available for {ticker} as of {as_of_dt:%Y-%m-%d}."

    df = df.rename(columns={c: c.lower() for c in df.columns})
    if "date" in df.columns:
        df = df[df["date"] <= pd.Timestamp(as_of_dt)].set_index("date").sort_index()

    try:
        rec = client.recommend(df)
    except Exception as exc:
        return f"Policy inference failed: {exc}"

    direction = "LONG" if rec.best_target_weight > 0 else (
        "SHORT" if rec.best_target_weight < 0 else "FLAT"
    )
    dist_lines = []
    for w, p in zip(rec.action_weights, rec.action_distribution):
        bar = "█" * int(round(p * 20))
        dist_lines.append(f"  {w:+.0%}  {p:.0%}  {bar}")
    return (
        f"## RL policy recommendation for {ticker} (as of {rec.as_of[:10]})\n"
        f"- **Direction**: {direction} {abs(rec.best_target_weight):.0%}\n"
        f"- **Confidence**: {rec.confidence:.0%}  ·  "
        f"**Value estimate**: {rec.value_estimate:+.3f}\n"
        f"- **Run id**: `{rec.run_id}`\n"
        f"- **Action distribution** (target weight → probability):\n"
        + "\n".join(dist_lines)
        + "\n\nThis is advisory.  Cross-check against the fundamental + "
        "technical analyst reports before committing to a Trade Plan."
        + _source_line(f"RL policy run_id={rec.run_id} on yfinance OHLCV {ticker}")
    )


@tool("rl_policy_recommendation")
def rl_policy_recommendation(ticker: str, as_of: str = "") -> str:
    """Return the trained RL policy's recommendation for ``ticker``.

    This is an **advisory** signal — the crew is free to follow it,
    discount it, or override it.  The recommendation is the output of
    an L4 PPO policy trained on past OHLCV through the same M2
    simulator the deterministic pipeline uses for sizing + execution,
    so its action distribution reflects realised after-cost PnL.

    Use ONCE per run, AFTER reading recent prices but BEFORE writing
    the Trade Plan, so you can compare the policy's prior with your
    own thesis.  If no policy has been *promoted* for this ticker the
    tool returns a note saying so — that's the signal that you need
    to fall back to pure fundamental + technical reasoning.

    Args:
        ticker: ticker symbol (uppercase preferred).
        as_of: optional decision date in YYYY-MM-DD; defaults to today.

    Returns:
        Markdown block with the policy's recommended direction +
        confidence + full action distribution, or a "no policy" note.
    """
    return _rl_policy_recommendation_impl(ticker, as_of)


@tool("retrieve_past_episodes")
def retrieve_past_episodes(ticker: str, as_of: str = "", k: int = 3) -> str:
    """Return up to ``k`` past episodes for ``ticker`` that ended *before* ``as_of``.

    Embargo-aware: any episode whose outcome timestamp is on or after the
    current decision date is excluded so the analyst can never lean on
    information that didn't exist at the time of the trade.  Each
    returned episode includes the action taken, the regime tag, and the
    realised return (if any) so the analyst can update on prior outcomes.

    ``as_of`` defaults to today.  Use this tool ONCE per run, at the
    start, to ground your analysis in the trader's actual track record
    on this name.  Don't call it repeatedly — the corpus is small and
    the same episodes will come back.
    """
    import os
    from datetime import datetime
    from pathlib import Path
    from trading_crew.agentic.memory.embedding import make_memory

    cache_dir = os.environ.get("TRADINGCREW_CACHE_DIR") or os.path.expanduser("~/.trading_crew")
    store_path = Path(cache_dir) / "memory" / "episodes.jsonl"
    if not store_path.exists():
        return "No past episodes recorded for any ticker yet."

    as_of = as_of or datetime.utcnow().date().isoformat()
    try:
        mem = make_memory(store_path)
        results = mem.retrieve(query=ticker, as_of=as_of, k=k, symbol=ticker)
    except Exception as exc:
        return f"Episodic memory unavailable: {exc}"

    if not results:
        return (
            f"No prior episodes for {ticker} have cleared their outcome embargo "
            f"as of {as_of} — this is the first traceable analysis or the "
            "outcomes are still pending."
        )

    lines = [f"## Past episodes for {ticker} (embargoed as-of {as_of})"]
    for r in results:
        ep = r.episode
        ap = ep.action_proposal or {}
        action = ap.get("side") or ap.get("action") or "?"
        weight = ap.get("target_weight")
        size_str = f"{(weight * 100):+.1f}%" if isinstance(weight, (int, float)) else "?"
        realised = ep.realised_return
        realised_str = f"{realised * 100:+.2f}%" if isinstance(realised, (int, float)) else "pending"
        # Reflections from L2 outcome-resolution carry a [tag | conf X.XX] prefix.
        # We keep that prefix because it gives the agent a self-quantified
        # confidence weight on the lesson (low conf ⇒ market noise, ignore).
        reflection = (ep.reflection or "(no reflection yet)").strip()[:480]
        alpha = ep.alpha_return
        alpha_str = f" · α={alpha * 100:+.2f}%" if isinstance(alpha, (int, float)) else ""
        mdd = ep.max_drawdown
        mdd_str = f" · mdd={mdd * 100:+.2f}%" if isinstance(mdd, (int, float)) and mdd != 0 else ""
        lines.append(
            f"- **{ep.decision_ts[:10]}** · regime={ep.regime.value} · action={action} "
            f"size={size_str} · realised={realised_str}{alpha_str}{mdd_str} · score={r.score:.2f}\n"
            f"  lesson: {reflection}"
        )
    return "\n".join(lines) + _source_line(
        f"Episodic memory (M3) {store_path.name} for {ticker} as-of {as_of}"
    )


# ---------------------------------------------------------------------------
# Tool catalog (used by the web layer to render checkboxes / docstrings)
# ---------------------------------------------------------------------------

ALL_TOOLS = {
    "get_market_context": get_market_context,
    "get_stock_data": get_stock_data,
    "get_indicators": get_indicators,
    "get_news": get_news,
    "get_global_news": get_global_news,
    "get_insider_transactions": get_insider_transactions,
    "get_fundamentals": get_fundamentals,
    "get_balance_sheet": get_balance_sheet,
    "get_cashflow": get_cashflow,
    "get_income_statement": get_income_statement,
    "get_macro_data": get_macro_data,
    "get_sector_peers": get_sector_peers,
    "get_geopolitical_news": get_geopolitical_news,
    "get_supply_chain_risk": get_supply_chain_risk,
    "get_options_summary": get_options_summary,
    "get_analyst_recommendations": get_analyst_recommendations,
    "get_event_proximity": get_event_proximity,
    "backtest_setup": backtest_setup,
    "retrieve_past_episodes": retrieve_past_episodes,
    "rl_policy_recommendation": rl_policy_recommendation,
}

# Default tool ownership per agent (mirrors crew.py wiring).
# The M3 retrieve_past_episodes tool is wired to the market analyst (first
# in the chain — grounds the rest of the debate) and the trader (whose
# job is to translate the thesis into a position; past hits/misses help
# calibrate the sizing recommendation).
#
# ``get_market_context`` is wired to the analysts whose queries depend on
# country/industry context (news, macro, geopolitical, sector) so they
# pull the brief BEFORE phrasing their searches — otherwise we end up
# with US-centric questions on Indian / UK / HK / JP names.
DEFAULT_AGENT_TOOLS: dict[str, list[str]] = {
    "market_analyst": ["get_stock_data", "get_indicators", "retrieve_past_episodes", "rl_policy_recommendation"],
    "social_analyst": ["get_market_context", "get_news"],
    "news_analyst": ["get_market_context", "get_news", "get_global_news", "get_insider_transactions"],
    "fundamentals_analyst": [
        "get_fundamentals", "get_balance_sheet",
        "get_cashflow", "get_income_statement",
    ],
    "macro_analyst": ["get_market_context", "get_macro_data", "get_global_news"],
    "geopolitical_analyst": [
        "get_market_context", "get_geopolitical_news", "get_global_news", "get_supply_chain_risk",
    ],
    "sector_analyst": ["get_market_context", "get_sector_peers", "get_indicators"],
    "quant_analyst": ["get_options_summary", "get_analyst_recommendations"],
    "trader": ["get_event_proximity", "backtest_setup", "retrieve_past_episodes", "rl_policy_recommendation"],
}
