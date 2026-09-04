"""Virtual growth-weighted paper book.

A simple, deterministic virtual portfolio layered on top of the stored
predictions: for a given run date we invest a fixed notional of
``DEFAULT_CAPITAL_INR`` (₹10,00,000) across every BUY / STRONG_BUY name of that
run, **dividing the capital in proportion to each name's predicted growth**
(``expected_return_pct``).  Each holding is then marked-to-market using the
*latest* price we have for that ticker — i.e. every subsequent run re-fetches
the current price, so opening the tab after a newer run automatically updates
the value of an older run's book.

Design notes
------------
* **Prices are deterministic, never predicted.**  A portfolio grades realised
  P&L, so BOTH the entry (the real close the user could have bought at on the
  run date) and the current mark (the real close as-of the latest run date)
  come straight from OHLCV — *not* from the model's ``limit_price`` /
  ``entry_price``, which is an LLM output and can be garbage (it once emitted
  $1.04 for a ~$105 name).  A name we genuinely can't price (e.g. delisted) is
  excluded rather than valued off a fabricated level.
* **No new table.**  Only the *selection* (which BUY names, and the predicted
  growth used to weight the split) comes from the ``predictions`` table; the
  valuation is layered on live prices.  Re-running a day re-derives its book,
  and every newer run refreshes the current marks.
* **FX overlay.**  US holdings are bought with the ₹ notional, so their INR
  value rides both the stock's move and USD→INR; we convert at the run-date
  rate on entry and the mark-date rate now (``INR=X``).  IN names use fx==1.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from . import prediction_db as db
from .charts import _fetch_ohlcv  # same deterministic OHLCV source the charts use

logger = logging.getLogger(__name__)

# ₹10,00,000 (10 lakh) virtual capital deployed per successful run.
DEFAULT_CAPITAL_INR = 1_000_000.0

# ---------------------------------------------------------------------------
# Realistic growth signal for the capital split
# ---------------------------------------------------------------------------
# Capital is divided across BUY names in proportion to their predicted growth,
# but the model's raw ``expected_return_pct`` is unusable as-is for two reasons:
#   1. It's frequently inflated — dozens of names pegged near a 150-180% cap.
#   2. It's quoted over wildly different horizons (120-730 days), so a 2-year
#      150% and a 4-month 20% aren't comparable and a flat clamp just re-ties
#      every inflated name at the ceiling.
# So we HARD-CLAMP each name's total expected return to what a realistic equity
# expectation of ``GROWTH_ANNUAL_CAP`` per year would compound to over THAT
# name's horizon.  This caps the fantasy numbers while keeping the split
# meaningful: a longer, higher-conviction thesis can still earn more of the
# book than a short one, but nobody is weighted off a 150% claim.
GROWTH_ANNUAL_CAP = 0.30  # ~30%/yr — generous but realistic ceiling


def _capped_growth(expected_return_pct: Any, horizon_days: Any) -> float:
    """Non-negative total expected return, hard-clamped to a horizon-scaled
    realistic ceiling.  This is the weight basis for the capital split."""
    er = float(expected_return_pct) if isinstance(expected_return_pct, (int, float)) else 0.0
    er = max(er, 0.0)
    h = float(horizon_days) if isinstance(horizon_days, (int, float)) and horizon_days > 0 else 252.0
    cap = (1.0 + GROWTH_ANNUAL_CAP) ** (h / 365.0) - 1.0
    return min(er, cap)


def _annualized_growth(total_return: float, horizon_days: Any) -> float:
    """Compound-annualise a (capped) total return to a comparable 1-year rate —
    the realistic per-year growth we surface in the UI."""
    h = float(horizon_days) if isinstance(horizon_days, (int, float)) and horizon_days > 0 else 252.0
    tr = max(float(total_return or 0.0), 0.0)
    return (1.0 + tr) ** (365.0 / h) - 1.0

# ---------------------------------------------------------------------------
# USD→INR overlay
# ---------------------------------------------------------------------------
# US holdings are bought with the ₹ notional, so their INR value must ride BOTH
# the stock's own move AND the rupee's move against the dollar.  We convert at
# the run-date rate on entry and the latest rate now:
#   value_inr = invested_inr · (cur_price/entry_price) · (fx_now/fx_entry)
# For IN names fx==1 on both sides, so they're untouched.  The daily USDINR
# series is fetched once from yfinance (``INR=X`` == USD per INR quote, i.e. ₹
# per $) and cached in-process; if it can't be fetched at all we raise rather
# than silently mis-price US names.
_FX_LOCK = threading.Lock()
_FX_CACHE: Dict[str, Any] = {"series": None, "ts": 0.0}
_FX_TTL_S = 3600.0  # refresh the series at most hourly


def _usdinr_series() -> List[Tuple[str, float]]:
    """Sorted ``[(YYYY-MM-DD, rate), …]`` of daily USD→INR closes (₹ per $)."""
    now = time.time()
    cached = _FX_CACHE.get("series")
    if cached is not None and (now - _FX_CACHE.get("ts", 0.0)) < _FX_TTL_S:
        return cached
    import yfinance as yf  # local import — heavy, and only needed for US names

    hist = yf.Ticker("INR=X").history(period="2y", auto_adjust=False)
    series = [
        (idx.date().isoformat(), float(c))
        for idx, c in hist["Close"].items()
        if c == c and c > 0  # drop NaN / non-positive
    ]
    series.sort()
    if not series:
        raise RuntimeError("USDINR (INR=X) series is empty — cannot apply FX overlay")
    with _FX_LOCK:
        _FX_CACHE["series"] = series
        _FX_CACHE["ts"] = now
    return series


def _fx_asof(series: List[Tuple[str, float]], date_str: str) -> Tuple[str, float]:
    """Latest USDINR close on/before ``date_str`` (clamped to the series start)."""
    chosen = series[0]
    for d, rate in series:
        if d <= date_str:
            chosen = (d, rate)
        else:
            break
    return chosen


def _decision(row: Any) -> Dict[str, Any]:
    raw = row["decision_json"] if "decision_json" in row.keys() else None
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return {}


def _latest_run_date() -> Optional[str]:
    """Global most-recent run date — the 'as of' date for every current mark
    (both native price and FX come from this run, so they stay in lock-step)."""
    conn = db._conn()
    with db._LOCK:
        row = conn.execute("SELECT MAX(run_date) AS m FROM predictions").fetchone()
    return row["m"] if row else None


# ---------------------------------------------------------------------------
# Deterministic price series (the ONLY source of entry / current marks)
# ---------------------------------------------------------------------------
# A portfolio tracks realised P&L, so both the entry (the price the user could
# actually buy at on the run date) and the current mark MUST be real traded
# prices — never the LLM's predicted ``limit_price``/``entry_price`` (which the
# model can and did hallucinate, e.g. $1.04 for a ~$105 name).  We read the
# actual daily closes from the same OHLCV source the charts use, cache the
# per-ticker series in-process (TTL), and pick the close as-of a given date.
_PRICE_LOCK = threading.Lock()
_PRICE_CACHE: Dict[str, Any] = {}  # ticker -> (series, ts)
_PRICE_TTL_S = 3600.0              # refresh a name's series at most hourly
_PRICE_LOOKBACK_DAYS = 400         # covers every plausible run date


def _price_series(ticker: str) -> List[Tuple[str, float]]:
    """Sorted ``[(YYYY-MM-DD, close), …]`` of a name's daily closes.

    Cached per ticker with a TTL.  Empty list when the symbol can't be priced
    (e.g. delisted) — callers must treat that as "unpriceable", never fabricate
    a level."""
    from datetime import datetime as _dt

    now = time.time()
    with _PRICE_LOCK:
        cached = _PRICE_CACHE.get(ticker)
        if cached is not None and (now - cached[1]) < _PRICE_TTL_S:
            return cached[0]
    try:
        df = _fetch_ohlcv(ticker, _dt.utcnow(), _PRICE_LOOKBACK_DAYS)
    except Exception:  # noqa: BLE001 — a single bad fetch must not break the book
        logger.warning("price series fetch failed for %s", ticker, exc_info=True)
        df = None
    series: List[Tuple[str, float]] = []
    if df is not None and not getattr(df, "empty", True) and "Close" in getattr(df, "columns", []):
        for d, c in zip(df["Date"], df["Close"]):
            try:
                c = float(c)
            except (TypeError, ValueError):
                continue
            if c == c and c > 0:  # drop NaN / non-positive
                ds = d.date().isoformat() if hasattr(d, "date") else str(d)[:10]
                series.append((ds, c))
        series.sort()
    with _PRICE_LOCK:
        _PRICE_CACHE[ticker] = (series, now)
    return series


def _price_asof(series: List[Tuple[str, float]], date_str: str) -> Optional[float]:
    """Latest close on/before ``date_str``; ``None`` if the series is empty or
    starts after the date (never clamps forward — we don't want to price an
    entry off a bar that didn't exist yet)."""
    chosen: Optional[float] = None
    for d, c in series:
        if d <= date_str:
            chosen = c
        else:
            break
    return chosen


def compute_portfolio(
    run_date: str,
    capital: float = DEFAULT_CAPITAL_INR,
    market: str = "all",
) -> Dict[str, Any]:
    """Build + value the virtual book for ``run_date``.

    ``market`` filters the investable universe to ``"IN"`` / ``"US"`` (or
    ``"all"`` for both).  The full ``capital`` is always deployed *within* the
    selected market — weights re-normalise across the filtered names — so an
    IN-only or US-only view is a self-contained ₹10L book, not a slice of the
    combined one.

    Returns ``{run_date, market, capital, currency, count, fx, holdings[],
    totals{}}`` where each holding carries ``invested`` / ``current_value`` /
    ``pct_change``.
    """
    market_sel = (market or "all").strip().upper()
    if market_sel not in ("IN", "US"):
        market_sel = "ALL"
    conn = db._conn()
    with db._LOCK:
        rows = conn.execute(
            """SELECT p.ticker, p.decision_json, c.name, c.market, c.homepage
                 FROM predictions p
                 LEFT JOIN companies c ON c.ticker = p.ticker
                WHERE p.run_date = ? AND p.status = 'done'""",
            (run_date,),
        ).fetchall()

    buys: List[Dict[str, Any]] = []
    for r in rows:
        d = _decision(r)
        if (d.get("side") or "").upper() != "BUY":
            continue
        er = d.get("expected_return_pct")
        conv = d.get("conviction_score")
        hz = d.get("horizon_days")
        buys.append({
            "ticker": r["ticker"],
            "name": r["name"] or r["ticker"],
            "market": r["market"],
            "homepage": r["homepage"] if "homepage" in r.keys() else None,
            "expected_return_pct": float(er) if isinstance(er, (int, float)) else 0.0,
            "horizon_days": int(hz) if isinstance(hz, (int, float)) and hz > 0 else None,
            "conviction": float(conv) if isinstance(conv, (int, float)) else None,
            # STRONG == HIGH conviction, same rule the 5-tier display uses.
            "action": db._five_tier_action("BUY", conv),
        })

    # Restrict to the selected market — the full capital is then re-deployed
    # across just these names (weights re-normalise below).
    if market_sel in ("IN", "US"):
        buys = [b for b in buys if (b["market"] or "").upper() == market_sel]

    # 'As of' date for every current mark — the latest run date, so an older
    # book revalues the moment a newer run lands (and the native price + FX
    # stay in lock-step, see below).
    mark_date = _latest_run_date() or run_date

    # DETERMINISTIC pricing.  Entry = the real close the user could have bought
    # at on the run date; current = the real close as-of the mark date.  Both
    # come from OHLCV, never the model's predicted price — a portfolio grades
    # realised P&L, so a hallucinated level can't be allowed to leak in.  We
    # prefetch every name's series in parallel (each is cached, so repeat loads
    # are instant) then price + drop anything we genuinely can't price (e.g. a
    # delisted symbol) rather than fabricate a level for it.
    if buys:
        with ThreadPoolExecutor(max_workers=min(8, len(buys))) as ex:
            list(ex.map(_price_series, {b["ticker"] for b in buys}))
    priced_buys: List[Dict[str, Any]] = []
    for b in buys:
        series = _price_series(b["ticker"])
        entry = _price_asof(series, run_date)
        cur = _price_asof(series, mark_date)
        if entry is None or cur is None or entry <= 0:
            logger.warning(
                "vportfolio: no deterministic price for %s (entry/current unavailable) — excluded",
                b["ticker"],
            )
            continue
        b["entry_price"] = entry
        b["current_price"] = cur
        priced_buys.append(b)
    buys = priced_buys

    # Growth-weighted allocation: capital split in proportion to each name's
    # REALISTIC (horizon-scaled, hard-clamped) predicted growth — never the raw,
    # frequently-inflated expected_return_pct.  The equal-weight branch only
    # guards a degenerate all-zero case.
    weights_raw = [_capped_growth(b["expected_return_pct"], b["horizon_days"]) for b in buys]
    total_w = sum(weights_raw)
    n = len(buys)

    # USD→INR overlay — only fetched if the book actually holds a US name.
    # The current mark's FX is taken as-of the same mark_date as the price so
    # the two stay in lock-step (pairing a mark-date price with today's live FX
    # would inject a phantom currency move).
    need_fx = any((b["market"] or "").upper() != "IN" for b in buys)
    fx_meta: Optional[Dict[str, Any]] = None
    fx_entry_usd = fx_current_usd = 1.0
    if need_fx:
        series = _usdinr_series()
        entry_date, fx_entry_usd = _fx_asof(series, run_date)
        current_date, fx_current_usd = _fx_asof(series, mark_date)
        fx_meta = {
            "pair": "USDINR",
            "entry_rate": fx_entry_usd,
            "current_rate": fx_current_usd,
            "entry_date": entry_date,
            "current_date": current_date,
        }

    holdings: List[Dict[str, Any]] = []
    tot_invested = 0.0
    tot_current = 0.0
    for b, wr in zip(buys, weights_raw):
        w = (wr / total_w) if total_w > 0 else (1.0 / n if n else 0.0)
        invested = capital * w
        entry = b["entry_price"]
        cur_price = b["current_price"]
        is_in = (b["market"] or "").upper() == "IN"
        currency = "INR" if is_in else "USD"
        fx_e = 1.0 if is_in else fx_entry_usd
        fx_c = 1.0 if is_in else fx_current_usd
        # ₹ notional → native at entry FX → shares → native value → ₹ at now FX.
        native_cost = invested / fx_e if fx_e > 0 else 0.0
        qty = native_cost / entry if entry > 0 else 0.0
        current_value = qty * cur_price * fx_c
        pct = (current_value / invested - 1.0) if invested > 0 else 0.0
        tot_invested += invested
        tot_current += current_value
        holdings.append({
            **b,
            "currency": currency,
            "weight": w,
            "invested": invested,
            "current_price": cur_price,
            "qty": qty,
            "current_value": current_value,
            "pct_change": pct,
            "priced": True,
            # Realistic growth surfaced to the UI: the clamped total return used
            # for the split, and its comparable per-year rate (capped at
            # GROWTH_ANNUAL_CAP).  ``expected_return_pct`` stays raw for audit.
            "growth_capped": wr,
            "growth_annual": _annualized_growth(wr, b["horizon_days"]),
        })

    holdings.sort(key=lambda h: h["current_value"], reverse=True)
    return {
        "run_date": run_date,
        "market": market_sel.lower(),
        "capital": capital,
        "currency": "INR",
        "count": len(holdings),
        "fx": fx_meta,
        "holdings": holdings,
        "totals": {
            "invested": tot_invested,
            "current": tot_current,
            "pct_change": (tot_current / tot_invested - 1.0) if tot_invested else 0.0,
        },
    }
