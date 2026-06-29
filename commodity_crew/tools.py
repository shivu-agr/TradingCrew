"""Futures-aware tools used by the commodity crew.

We deliberately stay narrow: each tool is single-purpose and emits text
the agent can paste into its rationale.  No tool blocks on missing API
keys — if a data source isn't available the tool returns an explicit
"data unavailable" string so the agent can pivot rather than crash.

Data sources:

- ``yfinance``     — OHLCV for continuous futures (``CL=F`` etc.) and the
                     individual delivery months (``CLN26.NYM``).
- ``stockstats``   — technical indicators on the OHLCV frame.
- CFTC public CSV  — weekly Commitments of Traders report
                     (https://www.cftc.gov/dea/futures/deacmesf.htm).  No
                     auth required, but the report is heavy so we cache
                     it for 24h.
- ``Tavily``       — open-web news / geopolitical search (shared key with
                     trading_crew).

These tools are intentionally kept independent of trading_crew's tool
implementations so the commodity crew can evolve its data surface
without breaking equity analysis.
"""

from __future__ import annotations

import csv
import datetime as _dt
import io
import json
import logging
import math
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf
from crewai.tools import tool
from tavily import TavilyClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provenance helpers — mirror trading_crew.tools so every commodity tool
# ends its output with a ``Source: …`` footer the analyst can copy verbatim
# into an inline ``[source: <identifier>]`` tag. The Reflective Critic
# uses those tags + the PM's ``sources`` list to verify claims aren't
# hallucinated.
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ")


def _source_line(identifier: str) -> str:
    safe = identifier.replace("\n", " ").replace("[", "(").replace("]", ")")
    return f"\nSource: {safe} · retrieved {_utc_now_iso()}"


# ---------------------------------------------------------------------------
# Tavily (shared with trading_crew)
# ---------------------------------------------------------------------------


_tavily_client: TavilyClient | None = None


def _tavily() -> TavilyClient:
    global _tavily_client
    if _tavily_client is None:
        api_key = os.environ["TAVILY_API_KEY"]
        _tavily_client = TavilyClient(api_key=api_key)
    return _tavily_client


def _tavily_news(query: str, max_results: int = 5) -> str:
    try:
        res = _tavily().search(
            query, max_results=max_results, topic="news", search_depth="basic"
        )
    except KeyError:
        return "TAVILY_API_KEY not set — news search unavailable." + _source_line(
            "Tavily News Search (key missing)"
        )
    except Exception as exc:
        return f"News search failed: {exc}" + _source_line(
            "Tavily News Search (request failed)"
        )
    rows = []
    for r in res.get("results", []):
        snippet = (r.get("content") or "").strip().replace("\n", " ")[:240]
        rows.append(f"- {r['title']}\n  {r['url']}\n  {snippet}")
    body = "\n".join(rows) if rows else "No news found."
    return body + _source_line(f"Tavily News Search · q='{query}'")


# ---------------------------------------------------------------------------
# Commodity metadata — root symbol -> (display name, contract size,
# search keywords, CFTC commodity name)
# ---------------------------------------------------------------------------


# cftc_name values are substrings to search inside "Market and Exchange Names"
# in CFTC's legacy futures-only annual report (deacot{year}.zip).  We use
# substrings (not full names) because the report's exact spacing/hyphenation
# changes slightly year to year and we want the tool to keep working
# across those revisions.
#
# Note: a few NYMEX-listed contracts are not in the legacy report (e.g.
# the standard "CRUDE OIL, LIGHT SWEET-NYMEX" is only in the disaggregated
# report).  For those we fall back to the ICE-listed equivalent, which
# carries equivalent positioning signal for our purposes.
COMMODITY_META: Dict[str, Dict[str, Any]] = {
    "CL": {"name": "WTI Crude Oil", "contract_size": 1000.0, "unit": "bbl",
           "keywords": "WTI crude oil OPEC inventories EIA",
           "cftc_name": "WTI FINANCIAL CRUDE OIL"},
    "BZ": {"name": "Brent Crude", "contract_size": 1000.0, "unit": "bbl",
           "keywords": "Brent crude oil OPEC",
           "cftc_name": "BRENT LAST DAY"},
    "NG": {"name": "Natural Gas", "contract_size": 10000.0, "unit": "MMBtu",
           "keywords": "natural gas storage EIA weather Henry Hub",
           "cftc_name": "NAT GAS NYME"},
    "HO": {"name": "Heating Oil", "contract_size": 42000.0, "unit": "gal",
           "keywords": "heating oil distillates diesel",
           "cftc_name": "NY HARBOR ULSD"},
    "RB": {"name": "RBOB Gasoline", "contract_size": 42000.0, "unit": "gal",
           "keywords": "gasoline RBOB driving season",
           "cftc_name": "GASOLINE RBOB"},
    "GC": {"name": "Gold", "contract_size": 100.0, "unit": "oz",
           "keywords": "gold safe haven inflation Fed real yields",
           "cftc_name": "GOLD - COMMODITY EXCHANGE INC."},
    "SI": {"name": "Silver", "contract_size": 5000.0, "unit": "oz",
           "keywords": "silver industrial demand",
           "cftc_name": "SILVER - COMMODITY EXCHANGE INC."},
    "HG": {"name": "Copper", "contract_size": 25000.0, "unit": "lb",
           "keywords": "copper Chinese demand industrial",
           "cftc_name": "COPPER- #1 - COMMODITY EXCHANGE INC."},
    "PL": {"name": "Platinum", "contract_size": 50.0, "unit": "oz",
           "keywords": "platinum autocatalyst",
           "cftc_name": "PLATINUM - NEW YORK MERCANTILE EXCHANGE"},
    "PA": {"name": "Palladium", "contract_size": 100.0, "unit": "oz",
           "keywords": "palladium autocatalyst",
           "cftc_name": "PALLADIUM - NEW YORK MERCANTILE EXCHANGE"},
    "ZC": {"name": "Corn", "contract_size": 5000.0, "unit": "bu",
           "keywords": "corn USDA WASDE planting harvest ethanol",
           "cftc_name": "CORN - CHICAGO BOARD OF TRADE"},
    "ZS": {"name": "Soybeans", "contract_size": 5000.0, "unit": "bu",
           "keywords": "soybeans USDA WASDE China crush",
           "cftc_name": "SOYBEANS - CHICAGO BOARD OF TRADE"},
    "ZW": {"name": "Wheat", "contract_size": 5000.0, "unit": "bu",
           "keywords": "wheat USDA WASDE Black Sea",
           "cftc_name": "WHEAT-SRW - CHICAGO BOARD OF TRADE"},
    "ZM": {"name": "Soybean Meal", "contract_size": 100.0, "unit": "ton",
           "keywords": "soybean meal crush margin",
           "cftc_name": "SOYBEAN MEAL - CHICAGO BOARD OF TRADE"},
    "ZL": {"name": "Soybean Oil", "contract_size": 60000.0, "unit": "lb",
           "keywords": "soybean oil biodiesel",
           "cftc_name": "SOYBEAN OIL - CHICAGO BOARD OF TRADE"},
    "KC": {"name": "Coffee", "contract_size": 37500.0, "unit": "lb",
           "keywords": "arabica coffee Brazil ICO",
           "cftc_name": "COFFEE C - ICE FUTURES U.S."},
    "CC": {"name": "Cocoa", "contract_size": 10.0, "unit": "MT",
           "keywords": "cocoa West Africa harvest",
           "cftc_name": "COCOA - ICE FUTURES U.S."},
    "SB": {"name": "Sugar", "contract_size": 112000.0, "unit": "lb",
           "keywords": "sugar #11 Brazil India",
           "cftc_name": "SUGAR NO. 11 - ICE FUTURES U.S."},
    "CT": {"name": "Cotton", "contract_size": 50000.0, "unit": "lb",
           "keywords": "cotton USDA mill demand",
           "cftc_name": "COTTON NO. 2 - ICE FUTURES U.S."},
    "LE": {"name": "Live Cattle", "contract_size": 40000.0, "unit": "lb",
           "keywords": "live cattle USDA cattle on feed",
           "cftc_name": "LIVE CATTLE - CHICAGO MERCANTILE EXCHANGE"},
    "HE": {"name": "Lean Hogs", "contract_size": 40000.0, "unit": "lb",
           "keywords": "lean hogs pork",
           "cftc_name": "LEAN HOGS - CHICAGO MERCANTILE EXCHANGE"},
}


def _root_of(symbol: str) -> str:
    """Strip yfinance suffix to recover the root code (CL=F -> CL).

    Handles three input shapes:
      - Continuous: ``CL=F`` or ``GC=F`` -> just split on ``=``.
      - Specific delivery: ``CLN26.NYM`` -> strip exchange, then strip
        the year-digits + the single month-code letter immediately before
        them (because in a real contract code the month letter is always
        followed by year digits, e.g. N26 = July 2026).
      - Already-root: ``CL``, ``HG``, ``ZC`` -> passes through unchanged.

    Critical: we only strip the trailing month-code letter when we
    actually stripped year digits first.  Without this guard, the root
    ``HG`` (copper) would have its trailing ``G`` mis-stripped to ``H``
    because ``G`` is also a CME month code.
    """
    s = (symbol or "").upper().strip()
    if "=" in s:
        s = s.split("=", 1)[0]
    if "." in s:
        s = s.split(".", 1)[0]
    # Strip trailing year digits; only if any were stripped do we then
    # peel off the preceding single month-code letter.
    stripped_digits = False
    while s and s[-1].isdigit():
        s = s[:-1]
        stripped_digits = True
    if stripped_digits and s and s[-1] in "FGHJKMNQUVXZ":
        s = s[:-1]
    return s or symbol.upper()


def _meta_for(symbol: str) -> Dict[str, Any]:
    root = _root_of(symbol)
    return COMMODITY_META.get(root, {
        "name": symbol, "contract_size": 1.0, "unit": "?",
        "keywords": symbol, "cftc_name": None,
    })


# ---------------------------------------------------------------------------
# 1. Price + indicators (futures-aware OHLCV)
# ---------------------------------------------------------------------------


@tool("get_commodity_ohlcv")
def get_commodity_ohlcv(ticker: str) -> str:
    """Retrieve 5-year futures OHLCV for a given symbol (yfinance ``=F`` syntax).

    Returns a layered summary with 5-year context (multi-horizon returns,
    52-week high/low, drawdown), monthly aggregates for the full window,
    and the last 30 daily sessions for short-term detail.  Pass the
    continuous front-month symbol (e.g. ``CL=F`` for crude, ``GC=F`` for
    gold).  For a specific delivery month use the full yfinance code
    (e.g. ``CLN26.NYM`` for July 2026 WTI).
    """
    df = yf.Ticker(ticker).history(period="5y")
    if df.empty:
        return f"No futures OHLCV data for {ticker}." + _source_line(
            f"yfinance futures OHLCV {ticker} (empty)"
        )
    df.index = pd.to_datetime(df.index)
    close = df["Close"].dropna()
    if close.empty:
        return f"No close data for {ticker}." + _source_line(
            f"yfinance futures OHLCV {ticker} (5y, no close)"
        )

    last_px = float(close.iloc[-1])
    last_date = close.index[-1].strftime("%Y-%m-%d")

    def _ret_n(n: int) -> str:
        if len(close) <= n:
            return "n/a"
        return f"{(close.iloc[-1] / close.iloc[-n - 1] - 1) * 100:+.1f}%"

    ret_5y_pct = (close.iloc[-1] / close.iloc[0] - 1) * 100
    window = close.iloc[-252:] if len(close) >= 252 else close
    hi_val = float(window.max())
    lo_val = float(window.min())
    hi_date = window.idxmax().strftime("%Y-%m-%d")
    lo_date = window.idxmin().strftime("%Y-%m-%d")

    daily_ret = close.pct_change().dropna()
    if len(daily_ret) >= 252:
        vol_str = f"{float(daily_ret.iloc[-252:].std() * (252 ** 0.5) * 100):.1f}%"
    else:
        vol_str = "n/a"

    roll_max = close.cummax()
    drawdown = (close / roll_max - 1.0) * 100
    max_dd = float(drawdown.min())
    max_dd_date = drawdown.idxmin().strftime("%Y-%m-%d")

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
    daily_tail = df[["Open", "High", "Low", "Close", "Volume"]].tail(30).round(2)

    meta = _meta_for(ticker)
    header = (
        f"Futures OHLCV for {ticker} ({meta['name']}, "
        f"contract_size={meta['contract_size']} {meta['unit']})"
    )
    summary_lines = [
        f"5-YEAR OVERVIEW for {ticker}",
        f"  Current price: {last_px:.2f}  (as of {last_date})",
        f"  52-week high : {hi_val:.2f}  on {hi_date}",
        f"  52-week low  : {lo_val:.2f}  on {lo_date}",
        (
            f"  Returns: 1m={_ret_n(21)}  3m={_ret_n(63)}  "
            f"6m={_ret_n(126)}  1y={_ret_n(252)}  "
            f"3y={_ret_n(252*3)}  5y={ret_5y_pct:+.1f}%"
        ),
        f"  Annualised volatility (1y trailing): {vol_str}",
        f"  Max drawdown (5y): {max_dd:.1f}%  (trough {max_dd_date})",
    ]

    return (
        header + "\n"
        + "\n".join(summary_lines)
        + f"\n\nMONTHLY OHLCV ({len(monthly)} months, 5y window):\n"
        + monthly.round(2).to_string()
        + f"\n\nDAILY OHLCV (last {len(daily_tail)} sessions):\n"
        + daily_tail.to_string()
        + _source_line(f"yfinance futures OHLCV {ticker} (5y, monthly + recent daily)")
    )


@tool("get_commodity_indicators")
def get_commodity_indicators(ticker: str) -> str:
    """Technical-indicator readout for a futures ticker, computed over
    5 years of daily history.

    Returns price, 20-SMA, 50-SMA, 200-SMA, RSI14, 60-day return,
    1-year return and a roll-yield proxy (front vs 6-month-out close)
    when both contracts are quoted.  The 5-year window is needed so
    SMA200 and the 1-year return are well-defined for the regime call.
    """
    df = yf.Ticker(ticker).history(period="5y")
    if df.empty:
        return f"No data for {ticker}." + _source_line(
            f"yfinance futures OHLCV {ticker} (5y, empty)"
        )
    close = df["Close"].dropna()
    if close.empty:
        return f"No close data for {ticker}." + _source_line(
            f"yfinance futures OHLCV {ticker} (5y, no close)"
        )

    def _fmt(val: float) -> str:
        return f"{val:.2f}" if pd.notna(val) else "n/a"

    def _fmt_pct(val: float) -> str:
        return f"{val:.2f}%" if pd.notna(val) else "n/a"

    sma20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else math.nan
    sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else math.nan
    sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else math.nan
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, math.nan)
    rsi14 = (100 - (100 / (1 + rs))).iloc[-1] if len(close) >= 15 else math.nan
    ret60 = (
        (close.iloc[-1] / close.iloc[-61] - 1.0) * 100
        if len(close) >= 61
        else math.nan
    )
    ret252 = (
        (close.iloc[-1] / close.iloc[-253] - 1.0) * 100
        if len(close) >= 253
        else math.nan
    )
    return (
        f"{ticker} technicals: price={close.iloc[-1]:.2f} "
        f"SMA20={_fmt(sma20)} SMA50={_fmt(sma50)} SMA200={_fmt(sma200)} "
        f"RSI14={_fmt(rsi14)} 60d_ret={_fmt_pct(ret60)} 1y_ret={_fmt_pct(ret252)}"
        + _source_line(f"yfinance futures OHLCV {ticker} (5y, indicators)")
    )


# ---------------------------------------------------------------------------
# 2. Futures curve (term structure)
# ---------------------------------------------------------------------------


_CME_MONTH_CODES = ["F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"]


def _futures_chain_symbols(root: str, n_months: int = 12, start_year: Optional[int] = None) -> List[str]:
    """Build yfinance symbols for the next ``n_months`` delivery contracts
    of a root code.  Format follows yfinance convention: ``CLU26.NYM``
    for September 2026 WTI.  Energy/metals trade on NYMEX/COMEX; ag on
    CBOT — we encode the exchange suffix in COMMODITY_META below.
    """
    year = start_year if start_year is not None else _dt.date.today().year
    today = _dt.date.today()
    out: List[str] = []
    # CBOT for grains, NYMEX/COMEX for energy/metals — pick the exchange.
    exchange = {
        "ZC": "CBT", "ZS": "CBT", "ZW": "CBT", "ZM": "CBT", "ZL": "CBT",
        "KC": "ICE", "CC": "ICE", "SB": "ICE", "CT": "ICE",
        "LE": "CME", "HE": "CME",
        "GC": "CMX", "SI": "CMX", "HG": "CMX",
    }.get(root, "NYM")
    month_offset = today.month - 1
    for i in range(n_months):
        m = (month_offset + i) % 12
        y_off = (month_offset + i) // 12
        sym = f"{root}{_CME_MONTH_CODES[m]}{(year + y_off) % 100:02d}.{exchange}"
        out.append(sym)
    return out


@tool("get_futures_curve")
def get_futures_curve(ticker: str, n_months: int = 12) -> str:
    """Pull the next ``n_months`` delivery contracts of the underlying
    commodity and return their last closes.  Identifies the curve
    structure (CONTANGO / BACKWARDATION / FLAT).

    ``ticker`` may be a continuous symbol (``CL=F``) or a specific
    delivery contract (``CLN26.NYM``); the root is extracted either way.
    """
    root = _root_of(ticker)
    if root not in COMMODITY_META:
        return f"No curve metadata for {ticker} (root={root})." + _source_line(
            f"yfinance futures curve {root} (no metadata)"
        )

    symbols = _futures_chain_symbols(root, n_months=n_months)
    rows: List[Tuple[str, float]] = []
    for sym in symbols:
        try:
            hist = yf.Ticker(sym).history(period="5d")
            if hist.empty:
                continue
            close = float(hist["Close"].dropna().iloc[-1])
            rows.append((sym, close))
        except Exception:
            continue

    if len(rows) < 2:
        return (
            f"Could not fetch a usable curve for {root} — yfinance has "
            f"data for {len(rows)} of the next {n_months} contracts. "
            "This is common for less-liquid commodities."
            + _source_line(f"yfinance futures curve {root} ({n_months}mo, sparse)")
        )

    prices = [p for _, p in rows]
    front, back = prices[0], prices[-1]
    slope_pct = (back / front - 1.0) * 100.0
    if slope_pct > 0.3:
        structure = "CONTANGO"
    elif slope_pct < -0.3:
        structure = "BACKWARDATION"
    else:
        structure = "FLAT"

    # Approximate annualised roll-yield as the front-back slope scaled
    # to the months covered.  A long position pays this drag.
    months = max(1, n_months)
    ann_roll = (-slope_pct / months) * 12.0  # negative for contango (drag)

    lines = [
        f"Futures curve for {root} ({COMMODITY_META[root]['name']}):",
        f"  structure: {structure} (front->back {slope_pct:+.2f}% over ~{n_months} mo)",
        f"  annualised roll-yield approx: {ann_roll:+.2f}%",
        "  prices:",
    ]
    for sym, p in rows:
        lines.append(f"    {sym}  {p:.2f}")
    return "\n".join(lines) + _source_line(
        f"yfinance futures curve {root} (next {n_months} contracts)"
    )


# ---------------------------------------------------------------------------
# 3. Seasonality
# ---------------------------------------------------------------------------


@tool("get_seasonality")
def get_seasonality(ticker: str, years: int = 5) -> str:
    """Compute average monthly returns over the last ``years`` years.

    Useful for commodities with strong seasonal patterns — natural gas
    winters, heating-oil winters, grains around USDA-cycle, etc.  The
    output is a small table ``{month -> avg_return, n_years_observed}``.
    """
    years = max(2, min(years, 20))
    df = yf.Ticker(ticker).history(period=f"{years}y")
    if df.empty:
        return f"No data for {ticker}." + _source_line(
            f"yfinance OHLCV {ticker} ({years}y, empty)"
        )
    df = df[["Close"]].dropna().copy()
    df.index = pd.to_datetime(df.index)
    df["year"] = df.index.year
    df["month"] = df.index.month
    # Monthly returns = last close of month / last close of prior month - 1
    monthly_last = df.groupby(["year", "month"]).last().reset_index()
    monthly_last["ret"] = monthly_last["Close"].pct_change()
    agg = monthly_last.groupby("month")["ret"].agg(["mean", "count"])
    months_in_words = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    lines = [f"Monthly seasonality for {ticker} over last {years}y:"]
    for m in range(1, 13):
        if m not in agg.index:
            continue
        row = agg.loc[m]
        lines.append(
            f"  {months_in_words[m]:>3}  avg={row['mean'] * 100:+.2f}%  (n={int(row['count'])})"
        )
    return "\n".join(lines) + _source_line(
        f"yfinance OHLCV {ticker} ({years}y, monthly aggregation)"
    )


# ---------------------------------------------------------------------------
# 4. CFTC Commitments of Traders (positioning)
# ---------------------------------------------------------------------------


def _cftc_cache_path() -> Path:
    cache_dir = Path(os.environ.get("TRADINGCREW_CACHE_DIR") or os.path.expanduser("~/.trading_crew"))
    p = cache_dir / "cftc"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cftc_url(year: Optional[int] = None) -> str:
    """CFTC publishes annual ZIPs at deacot{year}.zip; the current year
    is the live file.  We always pull the live one for the most recent
    week (it's small — ~150KB).
    """
    y = year or _dt.date.today().year
    return f"https://www.cftc.gov/files/dea/history/deacot{y}.zip"


def _download_cftc(year: int, max_age_hours: int = 24) -> Optional[Path]:
    """Download (or reuse cached) annual COT ZIP. Returns the path to the
    extracted CSV, or None on failure.
    """
    out_dir = _cftc_cache_path() / f"deacot{year}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "annual.txt"
    if csv_path.exists():
        age = _dt.datetime.now().timestamp() - csv_path.stat().st_mtime
        if age < max_age_hours * 3600:
            return csv_path
    try:
        req = urllib.request.Request(_cftc_url(year), headers={"User-Agent": "Mozilla/5.0 (commodity_crew/0.1)"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
        zf = zipfile.ZipFile(io.BytesIO(data))
        # The annual file's name varies — pick the largest text file in the zip.
        text_members = [m for m in zf.infolist() if m.filename.lower().endswith(".txt")]
        if not text_members:
            logger.warning("CFTC zip had no .txt members")
            return None
        biggest = max(text_members, key=lambda m: m.file_size)
        with zf.open(biggest) as src, csv_path.open("wb") as dst:
            dst.write(src.read())
        return csv_path
    except Exception as exc:
        logger.warning("CFTC download failed: %s", exc)
        return None


@tool("get_cot_report")
def get_cot_report(ticker: str, weeks: int = 6) -> str:
    """Return the last ``weeks`` weeks of CFTC Commitments-of-Traders
    positioning for the underlying commodity.

    Reports Long / Short / Net positioning for Commercial (hedgers) vs
    Non-commercial (managed money) traders.  Extreme positioning
    (managed money very long with prices stalling) is a classic
    counter-trend signal.
    """
    meta = _meta_for(ticker)
    cftc_name = meta.get("cftc_name")
    cftc_year = _dt.date.today().year
    if not cftc_name:
        return f"No CFTC mapping for {ticker} (root={_root_of(ticker)})." + _source_line(
            f"CFTC COT (no mapping for root={_root_of(ticker)})"
        )

    csv_path = _download_cftc(cftc_year)
    if csv_path is None or not csv_path.exists():
        # Try previous year if January and current year file isn't out yet
        cftc_year -= 1
        csv_path = _download_cftc(cftc_year)
        if csv_path is None:
            return "CFTC COT data temporarily unavailable." + _source_line(
                f"CFTC COT deacot{cftc_year}.zip (download failed)"
            )

    rows: List[Dict[str, str]] = []
    try:
        with csv_path.open("r", encoding="latin-1") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                # The annual legacy report uses spaces ("Market and Exchange Names");
                # the disaggregated report uses underscores. Try both so the tool
                # works regardless of which file CFTC published this year.
                market = (
                    row.get("Market and Exchange Names")
                    or row.get("Market_and_Exchange_Names")
                    or ""
                )
                if cftc_name in market:
                    rows.append(row)
    except Exception as exc:
        return f"CFTC parse failed: {exc}" + _source_line(
            f"CFTC COT deacot{cftc_year}.zip (parse error)"
        )

    if not rows:
        return f"No CFTC rows matched '{cftc_name}'." + _source_line(
            f"CFTC COT deacot{cftc_year}.zip (no rows matched {cftc_name})"
        )

    # Newest reports first; CFTC writes them in chronological order.
    rows.sort(
        key=lambda r: r.get("As of Date in Form YYYY-MM-DD") or r.get("As_of_Date_In_Form_YYMMDD") or "",
        reverse=True,
    )
    rows = rows[:weeks]

    out = [f"CFTC COT for {meta['name']} ({cftc_name}) last {len(rows)} weeks:"]
    for r in rows:
        date_s = (
            r.get("As of Date in Form YYYY-MM-DD")
            or r.get("Report_Date_as_YYYY-MM-DD")
            or r.get("Report_Date_as_MM_DD_YYYY")
            or "?"
        )
        try:
            # Annual report column names (with spaces); fall back to underscore variants.
            comm_long = int(float(
                r.get("Commercial Positions-Long (All)")
                or r.get("Comm_Positions_Long_All", "0") or "0"
            ))
            comm_short = int(float(
                r.get("Commercial Positions-Short (All)")
                or r.get("Comm_Positions_Short_All", "0") or "0"
            ))
            ncomm_long = int(float(
                r.get("Noncommercial Positions-Long (All)")
                or r.get("NonComm_Positions_Long_All", "0") or "0"
            ))
            ncomm_short = int(float(
                r.get("Noncommercial Positions-Short (All)")
                or r.get("NonComm_Positions_Short_All", "0") or "0"
            ))
        except Exception:
            continue
        net_ncomm = ncomm_long - ncomm_short
        net_comm = comm_long - comm_short
        out.append(
            f"  {date_s}: ManagedMoney net={net_ncomm:+,d} "
            f"(L={ncomm_long:,d} S={ncomm_short:,d})  "
            f"Commercials net={net_comm:+,d}"
        )
    return "\n".join(out) + _source_line(
        f"CFTC COT {_cftc_url(cftc_year)} ({cftc_name})"
    )


# ---------------------------------------------------------------------------
# 5. Commodity news + geopolitical
# ---------------------------------------------------------------------------


@tool("get_commodity_news")
def get_commodity_news(ticker: str) -> str:
    """Latest news for the underlying commodity.  Uses keyword expansion
    from COMMODITY_META so a search for ``ZC=F`` automatically queries
    'corn USDA WASDE planting harvest ethanol' rather than 'ZC=F'."""
    meta = _meta_for(ticker)
    return _tavily_news(f"{meta['name']} {meta['keywords']} commodity market")


@tool("get_commodity_geopolitical")
def get_commodity_geopolitical(ticker: str) -> str:
    """Geopolitical / supply-disruption news affecting the commodity."""
    meta = _meta_for(ticker)
    return _tavily_news(
        f"{meta['name']} supply disruption sanctions OPEC strike geopolitical risk"
    )


# ---------------------------------------------------------------------------
# 6. Past episodes (shared with stock crew — embargo-aware retrieval)
# ---------------------------------------------------------------------------


@tool("retrieve_past_episodes_commodity")
def retrieve_past_episodes_commodity(ticker: str, as_of: str = "", k: int = 3) -> str:
    """Same outcome-embargo-aware retrieval as the stock crew; the
    underlying episodic memory store is shared so cross-asset lessons
    propagate (e.g. an inflation-driven gold trade can inform a
    later silver trade).
    """
    import os as _os
    from datetime import datetime as _dt2
    from trading_crew.agentic.memory import EpisodicMemory

    cache_dir = _os.environ.get("TRADINGCREW_CACHE_DIR") or _os.path.expanduser("~/.trading_crew")
    store_path = Path(cache_dir) / "memory" / "episodes.jsonl"
    if not store_path.exists():
        return "No past episodes recorded yet." + _source_line(
            "Episodic memory (M3) episodes.jsonl (not created yet)"
        )

    as_of = as_of or _dt2.utcnow().date().isoformat()
    try:
        from trading_crew.agentic.memory.embedding import make_memory
        mem = make_memory(store_path)
        results = mem.retrieve(query=ticker, as_of=as_of, k=k, symbol=ticker)
    except Exception as exc:
        return f"Episodic memory unavailable: {exc}" + _source_line(
            "Episodic memory (M3) episodes.jsonl (load failed)"
        )
    if not results:
        return f"No prior episodes for {ticker} cleared the embargo as-of {as_of}." + _source_line(
            f"Episodic memory (M3) episodes.jsonl for {ticker} as-of {as_of}"
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
        reflection = (ep.reflection or "(no reflection yet)").strip()[:480]
        lines.append(
            f"- **{ep.decision_ts[:10]}** · regime={ep.regime.value} · action={action} "
            f"size={size_str} · realised={realised_str} · score={r.score:.2f}\n"
            f"  lesson: {reflection}"
        )
    return "\n".join(lines) + _source_line(
        f"Episodic memory (M3) episodes.jsonl for {ticker} as-of {as_of}"
    )


# ---------------------------------------------------------------------------
# RL policy advisor — same shape as the equity tool but lives under the
# commodity_crew namespace so the asset-class tool registries stay clean.
# Implementation just delegates to the equity tool's underlying logic,
# because the L4 policy module is itself asset-class-agnostic.
# ---------------------------------------------------------------------------


@tool("rl_policy_recommendation_commodity")
def rl_policy_recommendation_commodity(ticker: str, as_of: str = "") -> str:
    """Return the trained RL policy's recommendation for a futures ticker.

    Same semantics as the equity ``rl_policy_recommendation`` tool but
    namespaced for the commodity crew so the M2 cost preset selected
    by the runner (``futures_standard``) lines up with what the policy
    was trained against.  Returns a "no policy" note if nothing has
    been promoted for this contract yet.
    """
    # Delegate to the equity-side implementation; the underlying
    # ``load_policy`` lookup keys on (ticker, run_id) which is the same
    # for both asset classes.  The asset_class is preserved in the
    # promoted-policy pointer so callers can still tell them apart.
    from trading_crew.tools import _rl_policy_recommendation_impl
    return _rl_policy_recommendation_impl(ticker, as_of)


# ---------------------------------------------------------------------------
# Tool catalog
# ---------------------------------------------------------------------------


ALL_TOOLS = {
    "get_commodity_ohlcv": get_commodity_ohlcv,
    "get_commodity_indicators": get_commodity_indicators,
    "get_futures_curve": get_futures_curve,
    "get_seasonality": get_seasonality,
    "get_cot_report": get_cot_report,
    "get_commodity_news": get_commodity_news,
    "get_commodity_geopolitical": get_commodity_geopolitical,
    "retrieve_past_episodes_commodity": retrieve_past_episodes_commodity,
    "rl_policy_recommendation_commodity": rl_policy_recommendation_commodity,
}

# Default tool ownership per commodity agent.
#
# Researchers (bull/bear/research_manager), quality reviewer, risk team,
# and portfolio manager intentionally have NO tools — they reason from
# the analyst reports + debate transcript handed to them via ``context=``
# rather than re-querying data sources. The compliance officer optionally
# uses geopolitical news to verify sanctions / route concerns.
DEFAULT_AGENT_TOOLS: Dict[str, List[str]] = {
    "market_analyst": ["get_commodity_ohlcv", "get_commodity_indicators",
                       "retrieve_past_episodes_commodity",
                       "rl_policy_recommendation_commodity"],
    "curve_analyst": ["get_futures_curve"],
    "inventories_analyst": ["get_commodity_news"],
    "supply_demand_analyst": ["get_commodity_news"],
    "macro_analyst": ["get_commodity_news"],
    "geopolitical_analyst": ["get_commodity_geopolitical", "get_commodity_news"],
    "quant_analyst": ["get_cot_report", "get_seasonality"],
    "trader": ["get_futures_curve", "get_seasonality",
               "retrieve_past_episodes_commodity",
               "rl_policy_recommendation_commodity"],
    "compliance_officer": ["get_commodity_geopolitical"],
}


def get_tool_catalog() -> Dict[str, Dict[str, str]]:
    """Tool name -> {name, description} for the UI sidebar checkboxes."""
    out: Dict[str, Dict[str, str]] = {}
    for name, fn in ALL_TOOLS.items():
        doc = (getattr(fn, "description", None) or fn.__doc__ or "").strip().splitlines()
        short = next((ln.strip() for ln in doc if ln.strip()), name)
        out[name] = {"name": name, "description": short}
    return out
