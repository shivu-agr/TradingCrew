"""Alternate data sources for Indian equities.

The default ``yfinance``-backed tool layer works well for US tickers but
has documented gaps on Indian ones:

* Corporate actions (buyback / demerger / dividend) — Yahoo's adjusted
  close is frequently mis-applied on Indian names (RAYMOND demerger,
  HDFCBANK, BAJAJ-AUTO buyback in a prior BAJAJ-AUTO.NS live run).
* Shareholding pattern — promoter / FII / DII / public splits are not
  exposed at all in yfinance's payload.
* Ind-AS fundamentals — Yahoo's ``.info`` maps GAAP fields imperfectly
  onto the Indian accounting standard, producing stale or missing
  cash-flow / balance-sheet snapshots for smaller-cap NSE names.

This package layers two India-first sources on top of yfinance without
displacing it:

* ``screener_source`` — scrapes screener.in company pages (Ind-AS
  fundamentals, quarterly + annual statements, shareholding pattern,
  SEBI/exchange announcements, concalls).  No auth required; optional
  session login for higher rate limits.
* ``upstox_source`` — official upstox v3 REST SDK for NSE/BSE
  historical candles (exchange-clean corporate-action adjustments) and
  F&O option chain.  Uses a 1-year *analytics token* (env var
  ``UPSTOX_ANALYTICS_TOKEN``) so callers don't have to run the OAuth
  dance every day.

``routing`` centralises "is this ticker Indian?" and the "which source
handles this?" decision so the tool layer never has to string-match
suffixes inline.
"""

from .routing import (
    is_indian_ticker,
    strip_exchange_suffix,
    IndianDataSourcesConfig,
    get_config,
    reset_config_cache,
)
from . import screener_source, upstox_source

__all__ = [
    "is_indian_ticker",
    "strip_exchange_suffix",
    "IndianDataSourcesConfig",
    "get_config",
    "reset_config_cache",
    "screener_source",
    "upstox_source",
]
