"""Upstox v3 client — exchange-clean OHLCV + F&O option chain.

Upstox is a SEBI-registered Indian broker whose developer API exposes
the same market-data feeds their trading platform uses.  For our
purposes the appeal is threefold:

* **Corporate-action-adjusted history** at the exchange level — the
  primary source of Yahoo's Indian data quality problems
  (RAYMOND / HDFCBANK / BAJAJ-AUTO demerger + buyback mis-adjustments,
  documented on the yfinance issue tracker and the Zerodha Q&A forum).
* **F&O option chain + Greeks** for NSE index / stock options —
  something yfinance simply does not expose for Indian tickers.
* **1-year analytics token** — read-only access without the daily
  OAuth refresh a trading-authorized token would require.

Auth model
----------
The developer console produces two flavours of token:

* ``access_token``  — daily OAuth, can place orders, expires 3:30 AM IST.
* ``analytics_token`` — 1-year read-only, market data only, no login
  round-trip required.

We use the analytics token.  It is passed as a Bearer header via the
env var ``UPSTOX_ANALYTICS_TOKEN``; when unset the routing config
disables the source and every Indian-ticker OHLCV call falls back to
yfinance (which is the current behaviour anyway).

Design choices
--------------
* **Direct requests, no SDK.**  ``upstox-python-sdk`` is installed for
  optional future use (option-chain Greeks etc.), but the historical
  candles + option-chain endpoints are small enough that a direct
  ``requests`` call is easier to mock in tests and easier to audit
  during a live incident.
* **Instrument-key resolution.**  Upstox identifies equities by
  ``NSE_EQ|<ISIN>`` / ``BSE_EQ|<ISIN>``.  We download the exchange's
  instrument catalog (~1.5 MB gzipped for NSE, ~1 MB for BSE), cache
  it in memory keyed by tradingsymbol, and refresh once per process
  lifetime.  The catalog is stable enough that a stale cache is safe
  for the duration of a single analysis run.
* **No hidden fallback.**  If an upstox call fails, the wrapper raises
  a typed ``UpstoxError`` and the tool layer either logs a "upstox
  returned <status>" line or lets the analyst-task degrade path
  handle it — we do not silently swap to yfinance, per the project's
  routing rule.
"""

from __future__ import annotations

import csv
import datetime as _dt
import gzip
import io
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
import requests

from .routing import get_config, is_indian_ticker, strip_exchange_suffix

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class UpstoxError(Exception):
    """Base class for upstox client errors."""


class UpstoxAuthError(UpstoxError):
    """The analytics token is missing, malformed, or rejected."""


class UpstoxNotFoundError(UpstoxError):
    """The requested tradingsymbol has no matching instrument_key."""


# ---------------------------------------------------------------------------
# Instrument-key cache
# ---------------------------------------------------------------------------


_NSE_INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.csv.gz"
_BSE_INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/BSE.csv.gz"
_INSTRUMENT_CATALOG_TTL_SECONDS = 24 * 60 * 60  # daily refresh is plenty

_CATALOG_LOCK = threading.Lock()
# key: (exchange, tradingsymbol) -> instrument_key
_CATALOG: Dict[tuple, str] = {}
_CATALOG_FETCHED_AT: float = 0.0


def _load_exchange_catalog(exchange: str) -> Dict[str, str]:
    """Fetch + parse the upstox instrument CSV for one exchange.

    ``exchange`` is either ``"NSE"`` or ``"BSE"``.  Returns a mapping
    ``{tradingsymbol -> instrument_key}`` filtered to equities +
    indices (we drop the F&O rows here because they inflate the map
    ~100x; option-chain lookups use their own separate flow).
    """
    url = _NSE_INSTRUMENTS_URL if exchange == "NSE" else _BSE_INSTRUMENTS_URL
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        raise UpstoxError(
            f"upstox instrument catalog GET {url} returned HTTP {resp.status_code}"
        )
    raw = gzip.decompress(resp.content).decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(raw))
    mapping: Dict[str, str] = {}
    for row in reader:
        itype = (row.get("instrument_type") or "").upper()
        if itype not in ("EQUITY", "INDEX"):
            continue
        tsym = (row.get("tradingsymbol") or "").upper()
        ikey = row.get("instrument_key") or ""
        if tsym and ikey:
            mapping[tsym] = ikey
    return mapping


def _ensure_catalog() -> None:
    """Populate the global instrument catalog if stale or empty.

    Thread-safe: parallel analyst tasks that both trigger a lookup on
    a cold cache will serialise on the lock, but only the first one
    pays the CSV download cost.
    """
    global _CATALOG, _CATALOG_FETCHED_AT
    now = time.time()
    with _CATALOG_LOCK:
        if _CATALOG and (now - _CATALOG_FETCHED_AT) < _INSTRUMENT_CATALOG_TTL_SECONDS:
            return
        merged: Dict[tuple, str] = {}
        for exchange in ("NSE", "BSE"):
            try:
                m = _load_exchange_catalog(exchange)
            except Exception as exc:
                # A catalog fetch failure is not fatal — the other
                # exchange may still populate.  We surface a warning
                # and continue; if BOTH fail, subsequent lookups will
                # raise ``UpstoxNotFoundError`` per-ticker.
                logger.warning("upstox: %s catalog fetch failed: %s", exchange, exc)
                continue
            for tsym, ikey in m.items():
                merged[(exchange, tsym)] = ikey
        _CATALOG = merged
        _CATALOG_FETCHED_AT = now
        logger.info(
            "upstox: instrument catalog loaded (%d equities/indices across NSE+BSE)",
            len(merged),
        )


def resolve_instrument_key(symbol: str) -> str:
    """Return the ``NSE_EQ|<ISIN>`` / ``BSE_EQ|<ISIN>`` key for ``symbol``.

    ``symbol`` accepts a suffixed yfinance ticker (``RELIANCE.NS`` /
    ``500325.BO``) or the raw tradingsymbol (``RELIANCE``).  The
    exchange is inferred from the suffix; unsuffixed inputs default
    to NSE (screener.in's convention and the more liquid Indian
    exchange for most equities).

    Raises ``UpstoxNotFoundError`` if the tradingsymbol has no match
    in the catalog after a fresh fetch.
    """
    _ensure_catalog()
    up = symbol.strip().upper()
    if up.endswith(".BO"):
        exchange = "BSE"
        tsym = up[:-3]
    elif up.endswith(".NS"):
        exchange = "NSE"
        tsym = up[:-3]
    else:
        exchange = "NSE"
        tsym = up
    key = _CATALOG.get((exchange, tsym))
    if key is None:
        # Some Indian tickers use hyphens ("BAJAJ-AUTO") while yfinance
        # sometimes normalises them.  Try both variants before giving
        # up.
        alt = tsym.replace("-", "")
        key = _CATALOG.get((exchange, alt))
    if key is None:
        raise UpstoxNotFoundError(
            f"upstox has no {exchange} instrument for tradingsymbol {tsym!r} "
            f"(searched {len(_CATALOG)} entries)."
        )
    return key


def clear_catalog_cache() -> None:
    """Test hook — force the next lookup to re-download."""
    global _CATALOG, _CATALOG_FETCHED_AT
    with _CATALOG_LOCK:
        _CATALOG = {}
        _CATALOG_FETCHED_AT = 0.0


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------


_API_BASE = "https://api.upstox.com"


def _auth_headers() -> Dict[str, str]:
    """Build the Authorization header from the routing config.

    Raises ``UpstoxAuthError`` if no analytics token is configured —
    the caller should have gated on ``get_config().upstox_enabled``
    first, but we double-check to give a clean error on misconfigured
    deployments.
    """
    cfg = get_config()
    if not cfg.upstox_analytics_token:
        raise UpstoxAuthError(
            "UPSTOX_ANALYTICS_TOKEN is not set — cannot call upstox APIs. "
            "Generate an analytics token at https://account.upstox.com/developer/apps"
        )
    return {
        "Authorization": f"Bearer {cfg.upstox_analytics_token}",
        "Accept": "application/json",
    }


# ---------------------------------------------------------------------------
# Historical candles
# ---------------------------------------------------------------------------


@dataclass
class CandleFrame:
    """A tidy candle dataframe + provenance metadata."""

    df: "pd.DataFrame"
    instrument_key: str
    unit: str
    interval: str
    from_date: str
    to_date: str


def fetch_daily_candles(
    symbol: str,
    *,
    lookback_days: int = 5 * 365,
    to_date: Optional[str] = None,
) -> CandleFrame:
    """Fetch daily OHLCV candles for an Indian equity via upstox v3.

    Returns a pandas DataFrame with columns ``Open`` / ``High`` /
    ``Low`` / ``Close`` / ``Volume`` and a UTC ``DatetimeIndex`` — the
    same shape ``yf.Ticker().history()`` returns, so downstream
    indicator + backtest code can consume either source unchanged.

    ``lookback_days`` defaults to 5 years to mirror the yfinance
    ``period="5y"`` calls in ``get_stock_data`` and
    ``backtest_setup``.  Upstox's daily-candle endpoint has no
    per-request row cap so we pull the full window in one call.

    Raises ``UpstoxError`` on non-200 responses or empty payloads.
    """
    ikey = resolve_instrument_key(symbol)
    to_dt = (
        _dt.date.fromisoformat(to_date)
        if to_date
        else _dt.date.today()
    )
    from_dt = to_dt - _dt.timedelta(days=lookback_days)
    url = (
        f"{_API_BASE}/v3/historical-candle/"
        f"{requests.utils.quote(ikey, safe='')}/"
        f"days/1/{to_dt.isoformat()}/{from_dt.isoformat()}"
    )
    r = requests.get(url, headers=_auth_headers(), timeout=30)
    if r.status_code == 401:
        raise UpstoxAuthError(
            f"upstox rejected the analytics token (HTTP 401) — regenerate at "
            f"https://account.upstox.com/developer/apps"
        )
    if r.status_code != 200:
        raise UpstoxError(
            f"upstox historical-candle {symbol} ({ikey}) returned HTTP {r.status_code}: "
            f"{r.text[:200]}"
        )
    payload = r.json()
    candles = (payload.get("data") or {}).get("candles") or []
    if not candles:
        raise UpstoxError(
            f"upstox historical-candle {symbol} ({ikey}) returned an empty candle list."
        )
    # v3 candle shape: [timestamp, open, high, low, close, volume, oi]
    # We keep the standard 5 OHLCV columns to match yfinance conventions.
    df = pd.DataFrame(
        candles,
        columns=["timestamp", "Open", "High", "Low", "Close", "Volume", "OI"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    return CandleFrame(
        df=df,
        instrument_key=ikey,
        unit="days",
        interval="1",
        from_date=from_dt.isoformat(),
        to_date=to_dt.isoformat(),
    )


# ---------------------------------------------------------------------------
# Option chain
# ---------------------------------------------------------------------------


@dataclass
class OptionChainRow:
    """One strike's put + call snapshot at a specific expiry."""

    strike: float
    call_ltp: Optional[float] = None
    call_oi: Optional[float] = None
    call_volume: Optional[float] = None
    call_iv: Optional[float] = None
    call_delta: Optional[float] = None
    put_ltp: Optional[float] = None
    put_oi: Optional[float] = None
    put_volume: Optional[float] = None
    put_iv: Optional[float] = None
    put_delta: Optional[float] = None


@dataclass
class OptionChain:
    """The full put-call chain for one underlying + expiry."""

    underlying: str  # e.g. "RELIANCE" / "NIFTY"
    expiry: str      # ISO YYYY-MM-DD
    rows: List[OptionChainRow] = field(default_factory=list)
    spot: Optional[float] = None


def fetch_option_chain(symbol: str, expiry_date: str) -> OptionChain:
    """Fetch the put-call option chain for ``symbol`` at ``expiry_date``.

    Uses the v2 put-call chain endpoint.  ``expiry_date`` is
    ``YYYY-MM-DD`` (the endpoint accepts that format directly).

    Note this only works for F&O-enabled tickers.  Non-F&O names will
    return a 400 which we surface as an ``UpstoxError``.
    """
    ikey = resolve_instrument_key(symbol)
    url = f"{_API_BASE}/v2/option/chain"
    r = requests.get(
        url,
        headers=_auth_headers(),
        params={"instrument_key": ikey, "expiry_date": expiry_date},
        timeout=30,
    )
    if r.status_code == 401:
        raise UpstoxAuthError("upstox rejected the analytics token (HTTP 401).")
    if r.status_code != 200:
        raise UpstoxError(
            f"upstox option/chain {symbol} @ {expiry_date} returned HTTP "
            f"{r.status_code}: {r.text[:200]}"
        )
    payload = r.json()
    data = payload.get("data") or []
    rows: List[OptionChainRow] = []
    spot: Optional[float] = None
    for entry in data:
        strike = float(entry.get("strike_price") or 0.0)
        c = entry.get("call_options") or {}
        p = entry.get("put_options") or {}
        c_md = c.get("market_data") or {}
        c_greeks = c.get("option_greeks") or {}
        p_md = p.get("market_data") or {}
        p_greeks = p.get("option_greeks") or {}
        spot = entry.get("underlying_spot_price") or spot
        rows.append(OptionChainRow(
            strike=strike,
            call_ltp=c_md.get("ltp"),
            call_oi=c_md.get("oi"),
            call_volume=c_md.get("volume"),
            call_iv=c_greeks.get("iv"),
            call_delta=c_greeks.get("delta"),
            put_ltp=p_md.get("ltp"),
            put_oi=p_md.get("oi"),
            put_volume=p_md.get("volume"),
            put_iv=p_greeks.get("iv"),
            put_delta=p_greeks.get("delta"),
        ))
    rows.sort(key=lambda r: r.strike)
    return OptionChain(
        underlying=strip_exchange_suffix(symbol),
        expiry=expiry_date,
        rows=rows,
        spot=float(spot) if spot is not None else None,
    )


# ---------------------------------------------------------------------------
# Markdown formatters
# ---------------------------------------------------------------------------


def render_candle_summary(cf: CandleFrame, name: str, ticker: str) -> str:
    """Render an ``get_stock_data``-style block from an upstox candle frame.

    Mirrors the layout of ``get_stock_data``'s yfinance path so the
    downstream Market Analyst can consume either source identically.
    """
    df = cf.df
    if df.empty:
        return f"# {name} ({ticker})\n\n_(upstox returned an empty candle window)_\n"
    last = df.iloc[-1]
    first = df.iloc[0]
    high = df["High"].max()
    low = df["Low"].min()
    ret_pct = ((last["Close"] / first["Close"]) - 1.0) * 100.0
    vol_annualised = (df["Close"].pct_change().std() * (252 ** 0.5)) * 100.0
    return (
        f"# {name} ({ticker}) — Price snapshot (upstox v3, exchange-clean)\n\n"
        f"- Instrument: `{cf.instrument_key}`\n"
        f"- Window: {cf.from_date} → {cf.to_date} ({len(df)} daily bars)\n"
        f"- Last close: ₹ {last['Close']:.2f} | Range: ₹ {low:.2f} – ₹ {high:.2f}\n"
        f"- Cumulative return over window: {ret_pct:+.2f} %\n"
        f"- Annualised realised vol: {vol_annualised:.2f} %\n"
        f"- Last-bar volume: {int(last['Volume']):,}\n\n"
        f"| Date | Open | High | Low | Close | Volume |\n"
        f"|---|---|---|---|---|---|\n"
        + "\n".join(
            f"| {idx.date().isoformat()} | {r['Open']:.2f} | {r['High']:.2f} | "
            f"{r['Low']:.2f} | {r['Close']:.2f} | {int(r['Volume']):,} |"
            for idx, r in df.tail(10).iterrows()
        )
        + "\n"
    )


def render_option_chain_summary(chain: OptionChain, name: str) -> str:
    """Render an ``get_options_summary``-style block from an upstox chain."""
    if not chain.rows:
        return (
            f"# {name} option chain @ {chain.expiry} (upstox v3)\n\n"
            "_(no strikes returned — is the ticker F&O enabled?)_\n"
        )
    # Pick the 10 strikes closest to spot for a compact digest.
    if chain.spot is not None:
        chain.rows.sort(key=lambda r: abs(r.strike - chain.spot))
        near_atm = sorted(chain.rows[:15], key=lambda r: r.strike)
    else:
        near_atm = chain.rows[:15]
    total_call_oi = sum((r.call_oi or 0) for r in chain.rows)
    total_put_oi = sum((r.put_oi or 0) for r in chain.rows)
    pcr = (total_put_oi / total_call_oi) if total_call_oi else float("nan")
    header = (
        f"# {name} option chain @ {chain.expiry} (upstox v3)\n\n"
        + (f"- Spot: ₹ {chain.spot:.2f}\n" if chain.spot else "")
        + f"- Total call OI: {int(total_call_oi):,} | Total put OI: {int(total_put_oi):,}\n"
        + f"- Put/Call OI ratio: {pcr:.2f}\n\n"
    )
    rows = "\n".join(
        f"| {r.strike:.0f} | {r.call_ltp if r.call_ltp is not None else '—'} | "
        f"{int(r.call_oi) if r.call_oi else '—'} | {r.call_iv if r.call_iv is not None else '—'} | "
        f"{r.put_ltp if r.put_ltp is not None else '—'} | "
        f"{int(r.put_oi) if r.put_oi else '—'} | {r.put_iv if r.put_iv is not None else '—'} |"
        for r in near_atm
    )
    return (
        header
        + "| Strike | Call LTP | Call OI | Call IV | Put LTP | Put OI | Put IV |\n"
        + "|---|---|---|---|---|---|---|\n"
        + rows
        + "\n"
    )


# ---------------------------------------------------------------------------
# Convenience: is this ticker F&O-enabled?
# ---------------------------------------------------------------------------


def is_indian_and_upstox_available(symbol: str) -> bool:
    """Return True iff ``symbol`` is an Indian ticker AND the upstox
    source is enabled + configured.  Used by the tool layer to gate
    the "prefer upstox for OHLCV" branch.
    """
    return bool(is_indian_ticker(symbol) and get_config().upstox_enabled)
