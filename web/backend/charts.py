"""Build OHLCV + technical-indicator series for the frontend charts.

The Market Analyst calls ``get_stock_data`` / ``get_indicators`` from
``trading_crew/tools.py`` (which are yfinance-backed). This module fetches
the same yfinance data and additionally computes the indicators the chart
UI exposes (EMA / SMA / Bollinger / RSI / MACD / ATR) using ``stockstats``.

We deliberately do NOT cache to disk — the dataset is small (180 daily
rows) and a fresh fetch keeps the chart consistent with whatever the
Market Analyst is seeing on the same kickoff.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf
from stockstats import wrap

logger = logging.getLogger(__name__)


# Indicators we expose to the chart UI. Each maps to a stockstats column.
SUPPORTED_INDICATORS: Dict[str, str] = {
    "close_10_ema": "10 EMA",
    "close_50_sma": "50 SMA",
    "close_200_sma": "200 SMA",
    "boll": "Bollinger Mid",
    "boll_ub": "Bollinger Upper",
    "boll_lb": "Bollinger Lower",
    "rsi": "RSI",
    "macd": "MACD",
    "macds": "MACD Signal",
    "macdh": "MACD Histogram",
    "atr": "ATR",
}


def _safe_float(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if pd.isna(f):
        return None
    return f


def _fetch_ohlcv(ticker: str, end_date: datetime, lookback_days: int) -> pd.DataFrame:
    """Pull a daily OHLCV history using yfinance.

    We always pull a buffered window (lookback_days + 360) so indicators
    that need long warm-up (200-SMA) seed correctly even when the user
    asks for a short window.
    """
    buffer_days = max(lookback_days + 360, 540)
    start = end_date - timedelta(days=buffer_days)
    df = yf.Ticker(ticker).history(start=start.date(), end=(end_date + timedelta(days=1)).date())
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    if "Date" not in df.columns and "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df


def build_chart_payload(
    ticker: str,
    trade_date: Optional[str] = None,
    lookback_days: int = 180,
    indicators: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Return a JSON-ready dict with candles + indicator series.

    Output schema::

        {
          "ticker": "NTNX",
          "trade_date": "2026-06-03",
          "candles": [{"date","open","high","low","close","volume"}, ...],
          "indicators": {"close_50_sma": [{"date","value"}, ...], ...},
          "indicator_labels": {"close_10_ema": "10 EMA", ...}
        }
    """
    symbol = ticker.upper().strip()
    if not symbol:
        return {"ticker": symbol, "trade_date": trade_date, "candles": [], "indicators": {},
                "indicator_labels": SUPPORTED_INDICATORS, "error": "empty ticker"}

    # Resolve unsuffixed Indian / non-US tickers (``MAZDOCK`` -> ``MAZDOCK.NS``)
    # so the chart matches what the analyst crew is reading.  Commodity
    # symbols (``CL=F``) and already-suffixed inputs are no-ops.
    try:
        from trading_crew.market_context import resolve_ticker
        symbol = resolve_ticker(symbol)
    except Exception:
        pass

    end_dt = datetime.now()
    if trade_date:
        try:
            end_dt = datetime.strptime(trade_date, "%Y-%m-%d")
        except ValueError:
            pass

    try:
        full = _fetch_ohlcv(symbol, end_dt, lookback_days)
    except Exception as exc:
        logger.warning("yfinance fetch failed for %s: %s", symbol, exc)
        return {
            "ticker": symbol, "trade_date": trade_date, "candles": [], "indicators": {},
            "indicator_labels": SUPPORTED_INDICATORS, "error": str(exc),
        }

    if full.empty:
        return {
            "ticker": symbol, "trade_date": trade_date, "candles": [], "indicators": {},
            "indicator_labels": SUPPORTED_INDICATORS,
            "error": f"No price history for {symbol}",
        }

    # Trim to user-visible window (the rest is just used to seed indicators)
    window_start = end_dt - timedelta(days=lookback_days)
    visible = full[full["Date"] >= pd.Timestamp(window_start)].copy()
    if visible.empty:
        visible = full.tail(min(lookback_days, len(full))).copy()

    visible["DateStr"] = visible["Date"].dt.strftime("%Y-%m-%d")

    candles: List[Dict[str, Any]] = []
    for _, row in visible.iterrows():
        candles.append({
            "date": row["DateStr"],
            "open": _safe_float(row.get("Open")),
            "high": _safe_float(row.get("High")),
            "low": _safe_float(row.get("Low")),
            "close": _safe_float(row.get("Close")),
            "volume": _safe_float(row.get("Volume")),
        })

    requested = indicators or list(SUPPORTED_INDICATORS.keys())
    indicator_series: Dict[str, List[Dict[str, Any]]] = {}
    try:
        df_ind = wrap(full[["Date", "Open", "High", "Low", "Close", "Volume"]].copy())
        df_ind["__date_str"] = df_ind["Date"].dt.strftime("%Y-%m-%d")
        window_dates = set(visible["DateStr"])
        for col in requested:
            if col not in SUPPORTED_INDICATORS:
                continue
            try:
                values = df_ind[col]
            except Exception as exc:
                logger.warning("indicator %s failed for %s: %s", col, symbol, exc)
                continue
            series: List[Dict[str, Any]] = []
            for date_str, val in zip(df_ind["__date_str"], values):
                if date_str not in window_dates:
                    continue
                fv = _safe_float(val)
                if fv is not None:
                    series.append({"date": date_str, "value": fv})
            indicator_series[col] = series
    except Exception as exc:
        logger.warning("indicator computation failed for %s: %s", symbol, exc)

    return {
        "ticker": symbol,
        "trade_date": end_dt.strftime("%Y-%m-%d"),
        "candles": candles,
        "indicators": indicator_series,
        "indicator_labels": SUPPORTED_INDICATORS,
    }
