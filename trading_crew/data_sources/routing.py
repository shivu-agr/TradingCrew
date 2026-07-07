"""Ticker routing + env-driven enablement for Indian data sources.

Every downstream tool that has an India-first path calls into here so
the "is this ticker Indian? / do we have a token? / is it enabled?"
decision lives in exactly one place.

Design principles
-----------------
* **No hidden fallbacks.**  Per the user's operating rule, we do not
  silently retry with yfinance when a preferred source fails —
  ``IndianDataSourcesConfig.screener_enabled`` and
  ``upstox_enabled`` are simple booleans the tool inspects, and the
  tool itself decides how to fail (typically by raising / emitting a
  clear "screener returned <status>" line).  Routing here is purely
  *selection*, not error-handling.
* **Suffix-based classification.**  yfinance conventions:
  ``.NS`` → NSE (India), ``.BO`` → BSE (India).  A ticker without a
  suffix (e.g. ``RELIANCE``) is *ambiguous* — the ``resolve_ticker``
  helper in ``market_context`` resolves it to ``RELIANCE.NS`` and
  passes the resolved symbol here, so the router only sees fully
  qualified strings.
* **Env-var toggles + presence-of-credential toggles.**  screener.in
  needs no auth so it's on by default whenever
  ``TRADINGCREW_SCREENER_ENABLED != "0"``.  upstox needs an analytics
  token so it's on only when ``UPSTOX_ANALYTICS_TOKEN`` is non-empty
  AND ``TRADINGCREW_UPSTOX_ENABLED != "0"``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# yfinance uses ``.NS`` for NSE and ``.BO`` for BSE.  Both are Indian
# exchanges; we treat them identically for routing purposes.
_INDIAN_SUFFIXES = (".NS", ".BO")


def is_indian_ticker(symbol: str) -> bool:
    """Return True iff ``symbol`` is a fully-qualified Indian equity.

    The check is a pure suffix match on the resolved yfinance symbol.
    Callers that receive raw user input (e.g. ``RELIANCE``) should run
    it through ``trading_crew.market_context.resolve_ticker`` first so
    the suffix is populated.
    """
    if not symbol:
        return False
    up = symbol.strip().upper()
    return any(up.endswith(sfx) for sfx in _INDIAN_SUFFIXES)


def strip_exchange_suffix(symbol: str) -> str:
    """Return the exchange-free root of an Indian ticker.

    ``RELIANCE.NS`` -> ``RELIANCE``; ``500325.BO`` -> ``500325``.
    screener.in URLs and upstox instrument-key lookups both use the
    unsuffixed root, so this is the canonical adapter.
    """
    if not symbol:
        return symbol
    up = symbol.strip().upper()
    for sfx in _INDIAN_SUFFIXES:
        if up.endswith(sfx):
            return up[: -len(sfx)]
    return up


@dataclass(frozen=True)
class IndianDataSourcesConfig:
    """Snapshot of the routing knobs at construction time.

    Reading env vars once at snapshot construction (rather than on
    every tool call) means a mid-run env-var flip won't cause a tool
    to swap sources between two concurrent analyst tasks.  Call
    ``get_config.cache_clear()`` in tests when you need to refresh.
    """

    screener_enabled: bool
    """Route Indian-ticker fundamentals / corporate actions through
    screener.in.  Defaults to ON — the source needs no auth, and the
    scraper library ships with rate-limit handling."""

    screener_email: Optional[str]
    """Optional login for higher screener.in rate limits.  Passed
    through to the client verbatim; ``None`` runs anonymously."""

    screener_password: Optional[str]
    """Password for the optional screener.in login."""

    upstox_enabled: bool
    """Route Indian-ticker OHLCV / option-chain through the upstox
    v3 API.  Defaults to OFF unless ``UPSTOX_ANALYTICS_TOKEN`` is set
    (and ``TRADINGCREW_UPSTOX_ENABLED != "0"``)."""

    upstox_analytics_token: Optional[str]
    """1-year analytics token from the Upstox developer console.
    Preferred over the daily-refresh access token because the tool
    layer runs headless / in CI where an interactive OAuth loop is
    impractical."""


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean env var with a sensible default.

    Accepts ``0`` / ``false`` / ``no`` (case-insensitive) as "off";
    anything else non-empty is "on".  Missing env falls back to
    ``default``.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def _load_config() -> IndianDataSourcesConfig:
    upstox_token = os.environ.get("UPSTOX_ANALYTICS_TOKEN", "").strip() or None
    return IndianDataSourcesConfig(
        screener_enabled=_env_bool("TRADINGCREW_SCREENER_ENABLED", default=True),
        screener_email=os.environ.get("SCREENER_EMAIL", "").strip() or None,
        screener_password=os.environ.get("SCREENER_PASSWORD", "").strip() or None,
        upstox_enabled=(
            _env_bool("TRADINGCREW_UPSTOX_ENABLED", default=True)
            and upstox_token is not None
        ),
        upstox_analytics_token=upstox_token,
    )


# In-process cache so downstream tools don't re-read env every call.
# Tests reset this via ``reset_config_cache()``.
_CONFIG_CACHE: Optional[IndianDataSourcesConfig] = None


def get_config() -> IndianDataSourcesConfig:
    """Return the process-scoped ``IndianDataSourcesConfig`` snapshot."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        _CONFIG_CACHE = _load_config()
        logger.info(
            "Indian data-source routing: screener=%s upstox=%s",
            "on" if _CONFIG_CACHE.screener_enabled else "off",
            "on" if _CONFIG_CACHE.upstox_enabled else "off",
        )
    return _CONFIG_CACHE


def reset_config_cache() -> None:
    """Drop the cached snapshot so the next ``get_config()`` re-reads env.

    Test hook only.  In production the config is intentionally frozen
    for the lifetime of the process — see the ``IndianDataSourcesConfig``
    docstring for why.
    """
    global _CONFIG_CACHE
    _CONFIG_CACHE = None
