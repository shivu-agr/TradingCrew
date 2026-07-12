"""Tools used across the (4 merged) analyst desks and the trader.

Tool ownership (also enforced when wiring agents in ``crew.py``).  The
analyst team was consolidated from eight narrow desks to four wider ones,
so each desk now owns the union of the tools its merged predecessors had:

* Market Analyst (price + positioning):
    ``get_stock_data``, ``get_indicators``, ``get_options_summary``,
    ``get_analyst_recommendations``
* News Analyst (catalysts + sentiment):
    ``get_news``, ``get_global_news``, ``get_corporate_actions``,
    ``get_insider_transactions``
* Fundamentals Analyst (financials + long-term + sector):
    ``get_fundamentals``, ``get_balance_sheet``, ``get_cashflow``,
    ``get_income_statement``, ``get_shareholding_pattern``,
    ``get_long_term_thesis``, ``get_sector_peers``
* Macro Analyst (macro + geopolitical + supply-chain):
    ``get_macro_data``, ``get_global_news``, ``get_geopolitical_news``,
    ``get_supply_chain_risk``
* Trader:                ``get_event_proximity``

Market / fundamentals / macro / options data is sourced from ``yfinance``;
news comes from Tavily + SerpAPI web search (merged, de-duplicated).

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

import concurrent.futures as _cf
import datetime as _dt
import functools
import json
import logging
import os
import random
import re
import threading
import time
from typing import Optional

import pandas as pd
import requests
import yfinance as yf
from crewai.tools import tool
from tavily import TavilyClient

from trading_crew.data_sources import (
    get_config as get_data_sources_config,
    is_indian_ticker,
)
from trading_crew.data_sources import screener_source, upstox_source
from trading_crew.data_sources.screener_source import (
    ScreenerError,
    ScreenerRateLimitError,
    ScreenerNotFoundError,
)
from trading_crew.data_sources.upstox_source import (
    UpstoxError,
    UpstoxAuthError,
    UpstoxNotFoundError,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-process TTL cache for tool results
# ---------------------------------------------------------------------------
#
# Several analysts call the same tool with the same ticker during a single
# run — e.g. ``get_market_context`` was called 5x (Social, Sector, News,
# Macro, Geopolitical analysts) and ``get_indicators`` / ``get_corporate_
# actions`` were each called 2x in a verified BAJAJ-AUTO.NS run.  These
# tools return the same answer every time within a single workflow (the
# underlying data — yfinance ``.info``, Tavily news, fundamentals — does
# not change in the ~5-15 min window of a run), so the duplicate calls
# burn external API quota AND force the agent loop to re-process the
# same payload through another LLM iteration.
#
# We cache by ``(tool_name, args)`` with a configurable TTL (default 10
# minutes — comfortably longer than any single run, short enough that a
# subsequent run picks up fresh data).  Cache is thread-safe so the
# async analyst phase's 8 parallel agents can hit it concurrently
# without races.
#
# IMPORTANT: this is a *read-through* cache, not a substitute for the
# tool itself.  Quality is unaffected because the cached value is the
# exact bytes the underlying tool would have produced.  Only the wall
# clock, the external API quota, and the duplicate agent-loop LLM call
# are saved — the analyst still sees the data, still cites it inline,
# and still owns the [source: ...] tag.  Operators can set
# ``TRADINGCREW_TOOL_CACHE_TTL_SECONDS=0`` to disable caching entirely
# if they want to verify a regression isn't cache-induced.
# ---------------------------------------------------------------------------

_TOOL_CACHE_TTL_DEFAULT = 600.0
_tool_cache_lock = threading.Lock()
_tool_cache: dict[str, tuple[float, object]] = {}


def _tool_cache_ttl() -> float:
    try:
        return float(os.environ.get("TRADINGCREW_TOOL_CACHE_TTL_SECONDS", _TOOL_CACHE_TTL_DEFAULT))
    except ValueError:
        return _TOOL_CACHE_TTL_DEFAULT


def _memoize_tool(fn):
    """Memoize a tool function on ``(qualname, args, kwargs)``.

    Wraps the inner function (NOT the CrewAI ``@tool`` decorator) so the
    Tool the agent sees is unchanged — only the body short-circuits on a
    cache hit.  Use as::

        @tool("get_market_context")
        @_memoize_tool
        def get_market_context(ticker: str) -> str:
            ...

    The cache key includes the function name so two tools with the same
    signature don't collide.  Disable via ``TRADINGCREW_TOOL_CACHE_TTL_
    SECONDS=0`` for an A/B regression test.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        ttl = _tool_cache_ttl()
        if ttl <= 0:
            return fn(*args, **kwargs)
        # Build a stable JSON key from args + kwargs.  We avoid hashing
        # bare positional args directly because tool inputs may include
        # dicts / lists; ``json.dumps(..., sort_keys=True)`` gives us a
        # deterministic key regardless of dict ordering.
        try:
            key_payload = json.dumps([args, kwargs], sort_keys=True, default=str)
        except (TypeError, ValueError):
            # Un-cacheable input shape — fall back to bypassing the cache
            # so we never lose a real call to a serialisation quirk.
            return fn(*args, **kwargs)
        key = f"{fn.__qualname__}::{key_payload}"
        now = time.monotonic()
        with _tool_cache_lock:
            entry = _tool_cache.get(key)
            if entry is not None and (now - entry[0]) < ttl:
                logger.debug("tool cache hit: %s", fn.__qualname__)
                return entry[1]
        # Cache miss / expired — compute outside the lock so concurrent
        # readers don't serialise behind a slow yfinance / Tavily call.
        result = fn(*args, **kwargs)
        with _tool_cache_lock:
            _tool_cache[key] = (now, result)
        return result

    return wrapper


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
# Data-source router: upstox for Indian tickers (if enabled), else yfinance
# ---------------------------------------------------------------------------


def _load_price_history(ticker: str, *, period: str = "5y") -> tuple[pd.DataFrame, str]:
    """Return (OHLCV DataFrame, source identifier) for ``ticker``.

    Indian tickers (``.NS`` / ``.BO``) are served from the upstox v3
    historical-candle API when ``UPSTOX_ANALYTICS_TOKEN`` is set — this
    gives exchange-clean corporate-action adjustments and cleaner
    backtest numbers than yfinance's frequently mis-adjusted "Adj
    Close" (RAYMOND demerger, HDFCBANK, BAJAJ-AUTO buyback cases).
    All other tickers go through the existing yfinance path.

    ``period`` accepts the yfinance short form (``"5y"``, ``"10y"``,
    ``"max"``); the upstox path maps those to a lookback-days count.
    Longer windows are only asked for by the multi-year backtest panel;
    day-count math and the LLM prompts don't care about the raw string.

    Per the project's no-hidden-fallback rule, when the upstox call
    fails we let the exception propagate — the tool layer surfaces it
    as a "upstox returned <status>" line and the analyst continues
    with a degraded report.  The alternative (silent yfinance fallback)
    would hide the very corporate-action-adjustment problem we're
    trying to fix.

    Returns ``(df, identifier)`` where ``identifier`` is the string the
    tool should embed in its ``Source:`` footer.
    """
    if upstox_source.is_indian_and_upstox_available(ticker):
        # Map yfinance-style period strings to a lookback-days number.
        # ``"max"`` on upstox means "as much history as the exchange
        # has for this ticker" — we send 20 years which is more than
        # any NSE cash-equity listing.
        lookback = {
            "max": 20 * 365,
            "15y": 15 * 365,
            "10y": 10 * 365,
            "5y": 5 * 365,
            "2y": 2 * 365,
            "1y": 365,
            "2mo": 60,
        }.get(period, 5 * 365)
        cf = upstox_source.fetch_daily_candles(ticker, lookback_days=lookback)
        return cf.df, f"upstox v3 historical-candle {ticker} ({cf.instrument_key}, {cf.from_date}→{cf.to_date})"
    df = _with_retry(
        yf.Ticker(ticker).history,
        period=period,
        label=f"yfinance.history({ticker}, {period})",
    )
    return df, f"yfinance OHLCV {ticker} ({period})"

# ---------------------------------------------------------------------------
# News providers — Tavily + SerpAPI (Google News), merged and de-duplicated.
#
# Both providers are queried IN PARALLEL whenever their API key is set; the
# results are merged and de-duplicated by normalized URL / headline.  This
# is a breadth-widening MERGE, not a fallback chain — neither provider
# "backs up" the other (per the project's no-hidden-fallback rule).  If a
# provider errors or its key is missing, its contribution is simply empty
# and the merge proceeds with whatever the other returned.
# ---------------------------------------------------------------------------

_tavily_client: TavilyClient | None = None

# Hosts that only ever appear in fabricated / placeholder citations.  A
# real news article never lives on one of these, so any row (or any
# LLM-authored [source: …] tag) pointing here is an invented source.
_PLACEHOLDER_URL_HOSTS: tuple[str, ...] = (
    "example.com",
    "example.org",
    "example.net",
    "example.edu",
    "test.com",
    "test.org",
    "domain.com",
    "url.com",
    "placeholder.com",
    "yourwebsite.com",
)


def _tavily() -> TavilyClient:
    global _tavily_client
    if _tavily_client is None:
        api_key = os.environ["TAVILY_API_KEY"]
        _tavily_client = TavilyClient(api_key=api_key)
    return _tavily_client


def _is_placeholder_url(url: str) -> bool:
    """True when ``url`` is empty, relative, or points at a known
    placeholder host — i.e. a fabricated citation rather than a real one."""
    if not url or not isinstance(url, str):
        return True
    u = url.strip().lower()
    if not (u.startswith("http://") or u.startswith("https://")):
        # Relative or scheme-less "URLs" are never real article links.
        return True
    host = u.split("://", 1)[1].split("/", 1)[0]
    host = host.split("?", 1)[0].split("#", 1)[0].split(":", 1)[0]
    if not host or "." not in host:
        # A bare host with no dot (e.g. http://localhost, http://foo) is
        # not a real news article.
        return True
    for bad in _PLACEHOLDER_URL_HOSTS:
        if host == bad or host.endswith("." + bad):
            return True
    return False


def _reject_placeholder_urls(rows: list[dict]) -> list[dict]:
    """Drop any row whose ``url`` is a placeholder / fabricated link, so a
    fake citation can never leave the tool layer."""
    return [r for r in rows if not _is_placeholder_url(r.get("url", ""))]


# Matches an inline provenance tag ``[source: <identifier>]`` (the tag the
# whole workflow uses to attribute a claim to a tool output).
_SOURCE_TAG_RE = re.compile(r"\[source:\s*([^\]]+?)\s*\]", re.IGNORECASE)


def _is_fabricated_identifier(identifier: str) -> bool:
    """True when a ``[source: <identifier>]`` payload is a fabricated /
    placeholder citation rather than a real URL or a real tool identifier.

    We deliberately DO NOT treat every non-URL identifier as fabricated:
    legitimate tool identifiers (``yfinance .info ...``, ``screener.in
    ... #ratios``, ``long_term_thesis ...``) are not URLs and must be
    preserved.  Only two shapes are fabricated:

      1. an ``http(s)://`` URL that resolves to a placeholder host, or
      2. any string containing a known placeholder host token
         (``example.com`` etc.) even without a scheme.
    """
    if not identifier or not isinstance(identifier, str):
        return False
    ident = identifier.strip().lower()
    if ident.startswith("http://") or ident.startswith("https://"):
        return _is_placeholder_url(ident)
    for bad in _PLACEHOLDER_URL_HOSTS:
        if bad in ident:
            return True
    return False


def sanitize_citations(text: str) -> str:
    """Rewrite fabricated ``[source: <placeholder-url>]`` tags in free
    prose to an honest ``[analyst judgement]`` label.

    LLMs sometimes invent an ``example.com`` citation in their own
    reasoning to satisfy the provenance rule.  A downstream reviewer /
    critic then (correctly) flags it as unsupported and downgrades an
    otherwise sound thesis.  This deterministic pass fixes the root
    cause: a model-invented URL is relabelled as the honest qualitative
    judgement it actually is, so a genuine claim keeps its weight while
    the fake link is removed.  Real URLs and real tool identifiers are
    left untouched.
    """
    if not text or not isinstance(text, str):
        return text

    def _repl(m: "re.Match[str]") -> str:
        identifier = m.group(1)
        if _is_fabricated_identifier(identifier):
            return "[analyst judgement]"
        return m.group(0)

    return _SOURCE_TAG_RE.sub(_repl, text)


def _normalize_url(url: str) -> str:
    u = url.strip().lower()
    if "://" in u:
        u = u.split("://", 1)[1]
    if u.startswith("www."):
        u = u[4:]
    u = u.split("?", 1)[0].split("#", 1)[0]
    return u.rstrip("/")


def _normalize_title(title: str) -> str:
    return " ".join((title or "").lower().split())


def _tavily_news_rows(query: str, max_results: int = 8) -> list[dict]:
    """Fetch news rows from Tavily using ADVANCED search depth for richer
    coverage.  Returns ``[]`` (never raises) so the merge always proceeds."""
    try:
        res = _with_retry(
            _tavily().search,
            query,
            max_results=max_results,
            topic="news",
            search_depth="advanced",
            label=f"tavily.search({query!r})",
        )
    except Exception as exc:  # noqa: BLE001 — merge must survive one provider
        logger.warning("tavily news failed for %r: %s", query, exc)
        return []
    rows: list[dict] = []
    for r in res.get("results", []):
        rows.append(
            {
                "title": (r.get("title") or "").strip(),
                "url": (r.get("url") or "").strip(),
                "snippet": (r.get("content") or "").strip().replace("\n", " ")[:240],
                "provider": "tavily",
            }
        )
    return rows


def _serpapi_news(query: str, max_results: int = 8) -> list[dict]:
    """Fetch news rows from SerpAPI's Google News engine.  Returns ``[]``
    when the key is unset or the call fails (merge must survive)."""
    api_key = os.environ.get("SERPAPI_API_KEY")
    if not api_key:
        return []
    try:
        resp = _with_retry(
            requests.get,
            "https://serpapi.com/search.json",
            params={
                "engine": "google_news",
                "q": query,
                "api_key": api_key,
                "hl": "en",
                "gl": "us",
            },
            timeout=20,
            label=f"serpapi.google_news({query!r})",
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001 — merge must survive one provider
        logger.warning("serpapi news failed for %r: %s", query, exc)
        return []

    def _row(r: dict) -> dict | None:
        url = (r.get("link") or "").strip()
        title = (r.get("title") or "").strip()
        if not url or not title:
            return None
        src = r.get("source") or {}
        src_name = src.get("name") if isinstance(src, dict) else str(src)
        snippet = (r.get("snippet") or src_name or "").strip().replace("\n", " ")[:240]
        return {"title": title, "url": url, "snippet": snippet, "provider": "serpapi"}

    rows: list[dict] = []
    for r in data.get("news_results", []):
        # Google News groups related coverage under a "stories" list; flatten it.
        stories = r.get("stories")
        if isinstance(stories, list) and stories:
            for s in stories:
                row = _row(s)
                if row:
                    rows.append(row)
        else:
            row = _row(r)
            if row:
                rows.append(row)
    return rows[:max_results]


def _web_news_rows(query: str, max_results: int = 6) -> list[dict]:
    """Query Tavily AND SerpAPI in parallel and return the merged,
    de-duplicated, placeholder-free rows (no formatting)."""
    rows: list[dict] = []
    with _cf.ThreadPoolExecutor(max_workers=2) as pool:
        fut_tav = pool.submit(_tavily_news_rows, query, max_results + 4)
        fut_serp = pool.submit(_serpapi_news, query, max_results + 4)
        for fut in (fut_tav, fut_serp):
            try:
                rows.extend(fut.result())
            except Exception as exc:  # noqa: BLE001 — defensive; providers already guard
                logger.warning("news provider future failed for %r: %s", query, exc)

    rows = _reject_placeholder_urls(rows)

    seen_url: set[str] = set()
    seen_title: set[str] = set()
    merged: list[dict] = []
    for row in rows:
        nu = _normalize_url(row["url"])
        nt = _normalize_title(row["title"])
        if nu in seen_url or (nt and nt in seen_title):
            continue
        seen_url.add(nu)
        if nt:
            seen_title.add(nt)
        merged.append(row)

    # Two providers deliberately widen breadth, so keep a bit more than a
    # single-provider search would.
    return merged[: max_results * 2]


def _web_news(query: str, max_results: int = 6) -> str:
    """Query Tavily AND SerpAPI in parallel, merge + de-duplicate, and
    render a markdown digest with real article URLs.

    Each row carries an inline ``[source: <url>]`` tag so the analyst can
    cite the concrete article instead of inventing a placeholder link.
    """
    merged = _web_news_rows(query, max_results)

    if merged:
        body = "\n".join(
            f"- {r['title']}\n  {r['url']}\n  {r['snippet']}\n  [source: {r['url']}]"
            for r in merged
        )
        providers = sorted({r["provider"] for r in merged})
    else:
        body = "No news found."
        providers = []

    provider_tag = "+".join(providers) if providers else "none"
    return body + _source_line(
        f"Web News ({provider_tag}) · q='{query}'"
    )


# ---------------------------------------------------------------------------
# Market Analyst tools
# ---------------------------------------------------------------------------


@tool("get_stock_data")
@_memoize_tool
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
    try:
        df, source_id = _load_price_history(ticker, period="5y")
    except (UpstoxError, UpstoxAuthError, UpstoxNotFoundError) as exc:
        logger.warning("get_stock_data: upstox failed for %s: %s", ticker, exc)
        return (
            f"upstox v3 returned an error for {ticker}: {exc}. "
            "No price panel produced for this analyst turn."
        ) + _source_line(f"upstox v3 historical-candle {ticker} (failed: {type(exc).__name__})")
    if df.empty:
        return f"No price data for {ticker}." + _source_line(
            f"{source_id} (empty)"
        )

    df.index = pd.to_datetime(df.index)
    close = df["Close"].dropna()
    if close.empty:
        return f"No close data for {ticker}." + _source_line(
            f"{source_id} (no close)"
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
        + _source_line(f"{source_id} · monthly + recent daily")
    )


@tool("get_indicators")
@_memoize_tool
def get_indicators(ticker: str) -> str:
    """Retrieve a technical-indicator snapshot computed over 5 years of
    daily history.

    Returns price, SMA20 / SMA50 / SMA200, RSI14, 60-day return, and
    1-year return.  The 5-year window is needed so SMA200 and the 1-year
    return are well-defined for the Market Analyst's regime call —
    short windows force them to ``n/a`` for any IPO younger than the
    window."""
    try:
        df, source_id = _load_price_history(ticker, period="5y")
    except (UpstoxError, UpstoxAuthError, UpstoxNotFoundError) as exc:
        logger.warning("get_indicators: upstox failed for %s: %s", ticker, exc)
        return (
            f"upstox v3 returned an error for {ticker}: {exc}. "
            "No indicator panel produced for this analyst turn."
        ) + _source_line(f"upstox v3 historical-candle {ticker} (failed: {type(exc).__name__})")
    if df.empty:
        return f"No price data for {ticker}." + _source_line(
            f"{source_id} (empty)"
        )
    close = df["Close"].dropna()
    if close.empty:
        return f"No close data for {ticker}." + _source_line(
            f"{source_id} (no close)"
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
        + _source_line(f"{source_id} · indicators")
    )


# ---------------------------------------------------------------------------
# Social / News / Insider tools
# ---------------------------------------------------------------------------


@tool("get_news")
@_memoize_tool
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
    return _web_news(f"{name} {canonical} stock news sentiment last week", max_results=6)


@tool("get_global_news")
@_memoize_tool
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
        return _web_news(query, max_results=6)
    if not ticker:
        # No ticker, no override → return a *neutral* global query rather
        # than silently defaulting to US Fed terms.
        return _web_news(
            "global markets macro overview central bank policy currencies",
            max_results=6,
        )
    canonical = resolve_ticker(ticker)
    profile = get_market_profile(canonical)
    return _web_news(profile.country_news_query(), max_results=6)


@tool("get_insider_transactions")
@_memoize_tool
def get_insider_transactions(ticker: str) -> str:
    """Retrieve insider transaction information about a company (executives
    or directors buying/selling their own stock)."""
    t = yf.Ticker(ticker)
    df = getattr(t, "insider_transactions", None)
    src = _source_line(f"yfinance insider_transactions {ticker}")
    if df is None or len(df) == 0:
        return f"No insider transactions for {ticker}." + src
    return f"Insider transactions for {ticker}:\n" + df.head(10).to_string() + src


# Corporate-action search themes for ``get_corporate_actions``.  Targeted
# enough that a generic "stock news sentiment last week" query
# (``get_news`` above) would routinely miss them — buybacks, dividends,
# splits, M&A and board changes have specific vocabulary the model needs
# to be steered toward.  Each theme is a SHORT phrase the Tavily ranker
# can match without burning the per-query budget.
_CORPORATE_ACTION_THEMES: tuple[tuple[str, str], ...] = (
    ("buyback",  "share buyback OR share repurchase OR tender offer"),
    ("dividend", "dividend special dividend interim dividend record date"),
    ("split",    "stock split bonus issue rights issue"),
    ("m_and_a",  "acquisition merger takeover stake purchase divestiture spin-off"),
    ("guidance", "earnings guidance management commentary outlook update"),
    ("board",    "board appointment management change CEO CFO resignation"),
)


@tool("get_corporate_actions")
@_memoize_tool
def get_corporate_actions(ticker: str) -> str:
    """Surface specific *corporate-action* catalysts for a ticker —
    buybacks, dividends, splits, M&A, guidance updates, board /
    management changes — that the generic ``get_news`` sentiment query
    routinely misses.

    Why this exists
    ---------------
    ``get_news`` issues a single Tavily search with the phrase ``"stock
    news sentiment last week"`` and ``max_results=6``.  That biases the
    ranker toward generic sentiment / market-roundup articles and
    silently buries specific corporate-action announcements outside the
    top 6 — for example, on a recent Bajaj Auto run the company had
    announced a share buyback during the analysis window and the news
    layer never returned it.

    This tool runs **one focused Tavily query per action theme** (six
    themes total: buyback, dividend, split, M&A, guidance, board
    changes) and merges the results into a single markdown digest.
    Each theme's query uses the company's long name from
    ``get_market_context`` so the ranker doesn't get confused by
    ambiguous tickers.

    Output
    ------
    A markdown report with one section per theme that returned hits.
    Themes with no hits are listed in a single ``(no recent
    headlines)`` summary line at the end so the analyst can see what
    was searched-and-empty vs not-searched.  Every headline carries
    the article URL as ``[source: <url>]`` so downstream agents can
    preserve provenance.
    """
    canonical = resolve_ticker(ticker)
    profile = get_market_profile(canonical)
    name = profile.long_name or canonical

    sections: list[str] = []
    empty_themes: list[str] = []
    for theme, query_seeds in _CORPORATE_ACTION_THEMES:
        query = f"{name} {canonical} {query_seeds}"
        try:
            results = _web_news_rows(query, max_results=4)
        except Exception as exc:
            logger.warning(
                "get_corporate_actions: %s theme %s failed: %s",
                canonical, theme, exc,
            )
            sections.append(f"### {theme}\n  (search failed: {exc})")
            continue
        rows = [
            f"- {r['title']}\n  {r['url']}\n  {(r['snippet'] or '')[:200]}\n  [source: {r['url']}]"
            for r in results
        ]
        if rows:
            sections.append(f"### {theme}\n" + "\n".join(rows))
        else:
            empty_themes.append(theme)

    body = (
        f"# Corporate-action scan · {name} ({canonical})\n\n"
        + ("\n\n".join(sections) if sections else "(no corporate-action hits across any theme)")
    )
    if empty_themes:
        body += (
            "\n\n"
            f"_Themes with no recent headlines:_ {', '.join(empty_themes)}."
        )

    # India-first augmentation: append screener.in's primary SEBI/BSE
    # announcements feed so specific filings (buyback resolutions,
    # dividend record dates, board-meeting outcomes) that miss the
    # Tavily ranker's top-4 still land here.  This is the direct fix
    # for the BAJAJ-AUTO buyback-miss failure mode observed in a prior
    # live run — the buyback announcement was public on BSE the whole
    # time; only Tavily's ranker dropped it.
    if _screener_available_for(canonical):
        try:
            page = screener_source.fetch_company(canonical)
        except (ScreenerRateLimitError, ScreenerNotFoundError, ScreenerError) as exc:
            logger.warning(
                "get_corporate_actions: screener.in augmentation failed for %s: %s",
                canonical, exc,
            )
        else:
            docs_block = screener_source.render_announcements_report(page, limit=12)
            body += "\n\n" + docs_block

    return body + _source_line(
        f"Web News (Tavily+SerpAPI) · corporate-action scan ({canonical}, "
        f"themes={','.join(t for t, _ in _CORPORATE_ACTION_THEMES)})"
        + (" + screener.in #documents" if _screener_available_for(canonical) else "")
    )


# ---------------------------------------------------------------------------
# Fundamentals tools
# ---------------------------------------------------------------------------


def _screener_available_for(ticker: str) -> bool:
    """Return True iff the ticker is Indian and screener.in is enabled.

    Kept as a tiny helper so every fundamentals tool routes through the
    same predicate — one wire to flip if we ever want to disable the
    India-first path (e.g. during a screener.in outage without waiting
    for the env-var propagation).
    """
    return bool(is_indian_ticker(ticker) and get_data_sources_config().screener_enabled)


@tool("get_fundamentals")
@_memoize_tool
def get_fundamentals(ticker: str) -> str:
    """Retrieve comprehensive fundamental data for a given ticker symbol
    (P/E, EPS, market cap, profitability, growth, etc.).

    For Indian tickers (``.NS`` / ``.BO``) we prefer the Ind-AS
    fundamentals from screener.in — Yahoo's ``.info`` payload is
    frequently stale or empty on NSE names outside the top-100 float.
    The screener path emits the header ratios (Market Cap / P/E /
    Book Value / ROCE / ROE / etc.) plus screener.in's rule-generated
    pros/cons, all wrapped with an inline ``[source: <screener URL>]``
    tag.  Non-Indian tickers fall through to the yfinance path
    unchanged.
    """
    if _screener_available_for(ticker):
        try:
            page = screener_source.fetch_company(ticker)
        except (ScreenerRateLimitError, ScreenerNotFoundError, ScreenerError) as exc:
            logger.warning("get_fundamentals: screener.in failed for %s: %s", ticker, exc)
            return (
                f"screener.in returned an error for {ticker}: {exc}. "
                "No fundamentals produced for this analyst turn."
            ) + _source_line(f"screener.in company {ticker} (failed: {type(exc).__name__})")
        return screener_source.render_fundamentals_report(page) + _source_line(
            f"screener.in {page.source_url}"
        )
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


def _render_indian_statement(ticker: str, statement: str) -> str:
    """Route an Indian-ticker financial-statement lookup to screener.in.

    ``statement`` is the screener.in section id (``"balance-sheet"``,
    ``"cash-flow"``, ``"profit-loss"``).  Returns a full markdown block
    (10+ years of Ind-AS history) with the source URL captured for
    provenance.
    """
    try:
        page = screener_source.fetch_company(ticker)
    except (ScreenerRateLimitError, ScreenerNotFoundError, ScreenerError) as exc:
        logger.warning(
            "get_%s: screener.in failed for %s: %s",
            statement.replace("-", "_"), ticker, exc,
        )
        return (
            f"screener.in returned an error for {ticker}: {exc}. "
            f"No {statement} produced for this analyst turn."
        ) + _source_line(f"screener.in company {ticker} (failed: {type(exc).__name__})")
    return screener_source.render_financial_statement_report(page, statement) + _source_line(
        f"screener.in {page.source_url} #{statement}"
    )


@tool("get_balance_sheet")
@_memoize_tool
def get_balance_sheet(ticker: str) -> str:
    """Retrieve balance-sheet data (assets, liabilities, equity) for a given
    ticker symbol.  Indian tickers use screener.in's Ind-AS 10-year table;
    others use yfinance's single-latest-period snapshot."""
    if _screener_available_for(ticker):
        return _render_indian_statement(ticker, "balance-sheet")
    return (
        f"Balance sheet for {ticker} (latest period):\n"
        + _latest_column(yf.Ticker(ticker).balance_sheet)
        + _source_line(f"yfinance balance_sheet {ticker} (latest period)")
    )


@tool("get_cashflow")
@_memoize_tool
def get_cashflow(ticker: str) -> str:
    """Retrieve cash-flow-statement data for a given ticker symbol.
    Indian tickers use screener.in's Ind-AS 10-year table; others use
    yfinance's single-latest-period snapshot."""
    if _screener_available_for(ticker):
        return _render_indian_statement(ticker, "cash-flow")
    return (
        f"Cash flow for {ticker} (latest period):\n"
        + _latest_column(yf.Ticker(ticker).cashflow)
        + _source_line(f"yfinance cashflow {ticker} (latest period)")
    )


@tool("get_income_statement")
@_memoize_tool
def get_income_statement(ticker: str) -> str:
    """Retrieve income-statement data (revenue, costs, net income) for a
    given ticker symbol.  Indian tickers use screener.in's Ind-AS 10-year
    P&L table (annual + compounded growth); others use yfinance's
    single-latest-period snapshot."""
    if _screener_available_for(ticker):
        return _render_indian_statement(ticker, "profit-loss")
    return (
        f"Income statement for {ticker} (latest period):\n"
        + _latest_column(yf.Ticker(ticker).income_stmt)
        + _source_line(f"yfinance income_stmt {ticker} (latest period)")
    )


@tool("get_shareholding_pattern")
@_memoize_tool
def get_shareholding_pattern(ticker: str) -> str:
    """Return the quarterly shareholding pattern for an Indian ticker.

    Surfaces the promoter / FII / DII / public / government split
    across the last ~8 quarters, sourced from screener.in
    (``#shareholding`` section on the company page).  yfinance does
    not expose this data at all, and it's one of the most
    trader-relevant signals on Indian names — promoter pledges,
    FII outflows, and DII accumulation regularly precede large price
    moves on NSE.

    Only meaningful for Indian tickers.  A US / European ticker
    receives a clear "not applicable" note so the analyst doesn't
    waste an LLM turn trying to reconcile the absent data.
    """
    if not is_indian_ticker(ticker):
        return (
            f"Shareholding pattern is only tracked for Indian tickers via "
            f"screener.in.  {ticker} is not an NSE/BSE symbol; the "
            "analyst should treat this signal as not-applicable and "
            "focus on 13F / 13D filings instead."
        ) + _source_line("shareholding_pattern (not-applicable, non-Indian ticker)")
    if not get_data_sources_config().screener_enabled:
        return (
            "screener.in source is disabled "
            "(``TRADINGCREW_SCREENER_ENABLED=0``); no shareholding "
            "pattern available for this run."
        ) + _source_line("shareholding_pattern (screener.in disabled)")
    try:
        page = screener_source.fetch_company(ticker)
    except (ScreenerRateLimitError, ScreenerNotFoundError, ScreenerError) as exc:
        logger.warning("get_shareholding_pattern: screener.in failed for %s: %s", ticker, exc)
        return (
            f"screener.in returned an error for {ticker}: {exc}. "
            "No shareholding pattern produced for this analyst turn."
        ) + _source_line(f"screener.in company {ticker} (failed: {type(exc).__name__})")
    return screener_source.render_shareholding_report(page) + _source_line(
        f"screener.in {page.source_url} #shareholding"
    )


# ---------------------------------------------------------------------------
# Long-term thesis (multi-year fundamentals aggregation for LONG horizon)
# ---------------------------------------------------------------------------


def _cagr(values: list[float], years: float) -> float | None:
    """Compute a compound annual growth rate.

    Returns ``None`` if either endpoint is non-positive (a CAGR from
    a loss-making year to a profitable one is undefined; forcing a
    number there would hallucinate a growth rate).  ``years`` is the
    number of *periods* between the endpoints, not the count of
    values — e.g. 4 annual samples span 3 years.
    """
    if len(values) < 2 or years <= 0:
        return None
    start, end = values[0], values[-1]
    if start <= 0 or end <= 0:
        return None
    return ((end / start) ** (1.0 / years) - 1.0) * 100.0


def _pl_row(table, row_name: str) -> list[float]:
    """Best-effort numeric extraction of one row from a screener.in
    ``FinancialTable``.  Non-numeric cells (``"n/a"``, empty) are
    dropped; TTM columns are kept because they usefully bookend a
    growth-rate calculation.
    """
    if table is None:
        return []
    import re as _re
    _NUM = _re.compile(r"-?\d[\d,]*(?:\.\d+)?")
    for metric, values in table.rows:
        if metric.strip().lower() == row_name.strip().lower():
            out: list[float] = []
            for v in values:
                m = _NUM.search(str(v).replace(",", ""))
                if m:
                    try:
                        out.append(float(m.group(0).replace(",", "")))
                    except ValueError:
                        pass
            return out
    return []


def _long_term_thesis_from_screener(ticker: str) -> str:
    """Render a LONG-horizon thesis using screener.in's 10-year Ind-AS
    tables.  Aggregates revenue CAGR, EPS CAGR, ROCE / ROE trend,
    debt trajectory, and shareholding stability into one markdown
    block the LLM can quote verbatim.

    Every numeric claim carries a ``[source: screener.in <URL>]`` tag
    so the Reflective Critic can verify provenance.
    """
    page = screener_source.fetch_company(ticker)
    source = f"screener.in {page.source_url}"

    pl = page.profit_loss
    bs = page.balance_sheet

    sales = _pl_row(pl, "Sales") or _pl_row(pl, "Revenue")
    profit = _pl_row(pl, "Net Profit")
    eps = _pl_row(pl, "EPS in Rs") or _pl_row(pl, "EPS")

    borrowings = _pl_row(bs, "Borrowings")
    equity = _pl_row(bs, "Reserves") + _pl_row(bs, "Equity Capital")

    lines: list[str] = [f"# {page.long_name} — Long-term thesis ({page.mode})"]
    lines.append("")
    lines.append(
        "This block aggregates screener.in's 10-year Ind-AS tables into a "
        "single LONG-horizon (5–10 year) view.  Use it to decide whether "
        "the compounding thesis is intact BEFORE picking a directional tier."
    )
    lines.append("")

    # Growth block: revenue / earnings / EPS CAGR over the *full* window
    # (typically 10 years of annuals + TTM = 11 columns).
    if sales:
        span_years = max(1.0, len(sales) - 1.0)
        rev_cagr = _cagr(sales, span_years)
        lines.append(
            f"- **Revenue CAGR** over {len(sales)-1}y "
            f"({sales[0]:,.0f} → {sales[-1]:,.0f} Cr): "
            f"{('%+.1f%%' % rev_cagr) if rev_cagr is not None else 'undefined'} "
            f"[source: {source} #profit-loss]"
        )
    if profit:
        span_years = max(1.0, len(profit) - 1.0)
        prof_cagr = _cagr(profit, span_years)
        lines.append(
            f"- **Net Profit CAGR** over {len(profit)-1}y "
            f"({profit[0]:,.0f} → {profit[-1]:,.0f} Cr): "
            f"{('%+.1f%%' % prof_cagr) if prof_cagr is not None else 'undefined'} "
            f"[source: {source} #profit-loss]"
        )
    if eps:
        span_years = max(1.0, len(eps) - 1.0)
        eps_cagr = _cagr(eps, span_years)
        lines.append(
            f"- **EPS CAGR** over {len(eps)-1}y "
            f"({eps[0]:.1f} → {eps[-1]:.1f}): "
            f"{('%+.1f%%' % eps_cagr) if eps_cagr is not None else 'undefined'} "
            f"[source: {source} #profit-loss]"
        )

    # Balance-sheet trajectory — debt and equity direction over the last
    # ~5 years.  For a long-term compounder we want equity ↑ and
    # borrowings flat/↓; for a value trap the opposite.
    if borrowings and len(borrowings) >= 2:
        chg = borrowings[-1] - borrowings[0]
        direction = "reduced" if chg < 0 else ("increased" if chg > 0 else "flat")
        lines.append(
            f"- **Borrowings**: {borrowings[0]:,.0f} → {borrowings[-1]:,.0f} Cr "
            f"({direction} by {abs(chg):,.0f} Cr over the window) "
            f"[source: {source} #balance-sheet]"
        )

    # Return-on-capital trend — the true compounder tell.
    roce_row = _pl_row(page.ratios, "ROCE %") if page.ratios else []
    roe_row = _pl_row(page.ratios, "ROE %") if page.ratios else []
    if roce_row:
        avg_roce = sum(roce_row) / len(roce_row)
        lines.append(
            f"- **ROCE (avg over {len(roce_row)}y)**: {avg_roce:.1f}% "
            f"(latest {roce_row[-1]:.1f}%) — "
            f"{'high-quality compounder' if avg_roce >= 15 else ('durable' if avg_roce >= 10 else 'below cost of capital — value-trap risk')} "
            f"[source: {source} #ratios]"
        )
    if roe_row:
        avg_roe = sum(roe_row) / len(roe_row)
        lines.append(
            f"- **ROE (avg over {len(roe_row)}y)**: {avg_roe:.1f}% "
            f"(latest {roe_row[-1]:.1f}%) [source: {source} #ratios]"
        )

    # Shareholding stability — a promoter / FII / DII pattern shift is
    # the earliest signal of a broken long-term thesis on Indian names.
    sh = page.shareholding
    if sh and sh.quarterly and sh.quarterly.rows:
        prom_row = next((r for r in sh.quarterly.rows if r[0].lower().startswith("promoter")), None)
        fii_row = next((r for r in sh.quarterly.rows if r[0].lower().startswith("fii") or "foreign" in r[0].lower()), None)
        if prom_row and len(prom_row[1]) >= 2:
            lines.append(
                f"- **Promoter holding trajectory** ({sh.quarterly.columns[0]} → {sh.quarterly.columns[-1]}): "
                f"{prom_row[1][0]} → {prom_row[1][-1]} "
                f"[source: {source} #shareholding]"
            )
        if fii_row and len(fii_row[1]) >= 2:
            lines.append(
                f"- **FII holding trajectory** ({sh.quarterly.columns[0]} → {sh.quarterly.columns[-1]}): "
                f"{fii_row[1][0]} → {fii_row[1][-1]} "
                f"[source: {source} #shareholding]"
            )

    # Screener's own pros/cons distillation — cheap way to include a
    # human-quality summary without asking the LLM to re-write it.
    if page.pros_cons.pros:
        lines.append("")
        lines.append("## Structural pros (from screener.in analysis)")
        lines.extend(f"- {p} [source: {source} #analysis]" for p in page.pros_cons.pros[:5])
    if page.pros_cons.cons:
        lines.append("")
        lines.append("## Structural cons (from screener.in analysis)")
        lines.extend(f"- {c} [source: {source} #analysis]" for c in page.pros_cons.cons[:5])

    return "\n".join(lines)


def _long_term_thesis_from_yfinance(ticker: str) -> str:
    """Fallback long-term thesis for non-Indian tickers.

    yfinance's ``.income_stmt`` / ``.balance_sheet`` typically hand
    back 4 annual periods (~3 years of change) — not the 10y screener
    delivers but still enough to compute a directional CAGR and a
    debt-trajectory line.  Deliberately does NOT try to fabricate a
    10-year story from 3 years of data — it says so explicitly so the
    LLM does not over-weight the row.
    """
    t = yf.Ticker(ticker)
    info = t.info or {}
    name = info.get("shortName") or info.get("longName") or ticker

    lines: list[str] = [f"# {name} ({ticker}) — Long-term thesis (yfinance path)"]
    lines.append("")
    lines.append(
        "yfinance only exposes ~3–4 years of annual statements — this "
        "block is a *lower-bound* long-term view.  For a genuine 10y "
        "read on an Indian ticker, route via screener.in; for a US name, "
        "consider 10-K comparisons manually."
    )
    lines.append("")

    inc = t.income_stmt
    bs = t.balance_sheet

    def _vals_and_years(series):
        """Return (values, fiscal-years) oldest→newest for a statement row.

        yfinance indexes statement columns by period-end Timestamps; we keep
        the actual fiscal years so the report can label the *real* window
        (e.g. ``FY2022→FY2025``) instead of a bare period count that the LLM
        was previously re-dressing as an invented ``2019–2023`` range.
        """
        s = series.dropna()
        try:
            s = s.sort_index()  # oldest first
        except Exception:
            pass
        vals = [float(v) for v in s.values]
        years: list[int | None] = []
        for idx in s.index:
            yr = getattr(idx, "year", None)
            if yr is None:
                try:
                    yr = int(str(idx)[:4])
                except Exception:
                    yr = None
            years.append(yr)
        return vals, years

    def _span_label(years: list[int | None], n: int) -> tuple[str, float]:
        """Human window label + the true year-span for CAGR annualisation."""
        if years and years[0] and years[-1] and years[-1] > years[0]:
            return f"FY{years[0]}→FY{years[-1]}", float(years[-1] - years[0])
        return f"{n - 1} annual periods", max(1.0, n - 1.0)

    window_note = ""
    if inc is not None and not inc.empty:
        for row_name in ("Total Revenue", "Net Income"):
            if row_name in inc.index:
                vals, years = _vals_and_years(inc.loc[row_name])
                if len(vals) >= 2:
                    label, span_years = _span_label(years, len(vals))
                    window_note = label
                    cagr = _cagr(vals, span_years)
                    lines.append(
                        f"- **{row_name} CAGR** over {label} ({len(vals)} annual periods) "
                        f"({vals[0]/1e9:.2f}B → {vals[-1]/1e9:.2f}B): "
                        f"{('%+.1f%%' % cagr) if cagr is not None else 'undefined'} "
                        f"[source: yfinance income_stmt {ticker}]"
                    )
    if bs is not None and not bs.empty:
        if "Total Debt" in bs.index:
            vals, years = _vals_and_years(bs.loc["Total Debt"])
            if len(vals) >= 2:
                label, _ = _span_label(years, len(vals))
                chg = vals[-1] - vals[0]
                direction = "reduced" if chg < 0 else ("increased" if chg > 0 else "flat")
                lines.append(
                    f"- **Total Debt** over {label}: {vals[0]/1e9:.2f}B → {vals[-1]/1e9:.2f}B "
                    f"({direction} by {abs(chg)/1e9:.2f}B) "
                    f"[source: yfinance balance_sheet {ticker}]"
                )

    for label, key in (
        ("Return on Equity", "returnOnEquity"),
        ("Return on Assets", "returnOnAssets"),
        ("Debt/Equity", "debtToEquity"),
    ):
        v = info.get(key)
        if v is None:
            continue
        # yfinance uses fractional units for ratios (0.15 = 15%) but
        # occasionally reports raw percentages — we render both without
        # guessing.  Labelled a LATEST SNAPSHOT, never a multi-year trend,
        # so the LLM cannot re-present it as "ROE fell from X to Y over N years".
        if isinstance(v, (int, float)):
            lines.append(
                f"- **{label}** (LATEST snapshot — NOT a multi-year trend): {v} "
                f"[source: yfinance .info {ticker}]"
            )

    # Explicit data-availability contract so the analyst / PM cannot launder
    # fabricated precision through this tool's source tag.  This is the direct
    # fix for reports that cited e.g. "ROCE fell 18%→12% (2019–2023)
    # [source: yfinance get_long_term_thesis]" — a metric and calendar range
    # this path never produced.
    lines.append("")
    lines.append("## Data availability (yfinance path — READ BEFORE CITING)")
    lines.append(
        f"- AVAILABLE here: multi-year Revenue / Net-Income CAGR over "
        f"{window_note or 'the fiscal years listed above'}, the total-debt "
        f"trajectory, and a LATEST-snapshot ROE / ROA / Debt-Equity."
    )
    lines.append(
        "- NOT AVAILABLE here (do NOT cite these with a "
        "`[source: ... get_long_term_thesis]` tag): a **ROCE** figure or trend, "
        "a multi-year **ROE/EPS trend**, or any number for a fiscal year outside "
        "the window above. If your thesis needs one of these, either omit it or "
        "label the reasoning `[analyst judgement]` — never attribute an invented "
        "figure to this tool."
    )

    return "\n".join(lines)


@tool("get_long_term_thesis")
@_memoize_tool
def get_long_term_thesis(ticker: str) -> str:
    """Multi-year fundamentals thesis for a **LONG-horizon** (5–10 year)
    decision.

    Aggregates the evidence that matters for a compounder / value
    thesis into a single markdown block the LLM can quote verbatim:

    - Revenue / Net Profit / EPS **CAGR** over the longest available
      window (10y from screener.in on Indian tickers; 3–4y from
      yfinance elsewhere).
    - **Balance-sheet trajectory** — direction and magnitude of debt
      over the window.
    - **ROCE / ROE trend** — the true "quality of compounding" tell.
    - **Shareholding stability** (Indian names only) — promoter /
      FII / DII drift is the earliest signal of a broken thesis.
    - Screener.in's own structural pros/cons — a curated summary the
      LLM can cite alongside the raw numbers.

    Use this tool BEFORE picking ``horizon = LONG`` — a LONG decision
    without a supporting revenue/EPS CAGR line, or with degrading
    ROCE, is exactly the "value-trap" failure mode the
    Reflective Critic will REVISE.
    """
    canonical = resolve_ticker(ticker)
    if _screener_available_for(canonical):
        try:
            return _long_term_thesis_from_screener(canonical) + _source_line(
                f"long_term_thesis via screener.in {canonical}"
            )
        except (ScreenerRateLimitError, ScreenerNotFoundError, ScreenerError) as exc:
            logger.warning("get_long_term_thesis: screener.in failed for %s: %s", canonical, exc)
            return (
                f"screener.in returned an error for {canonical}: {exc}. "
                "No long-term thesis produced — the PM should NOT pick "
                "horizon=LONG for this run."
            ) + _source_line(
                f"screener.in company {canonical} (failed: {type(exc).__name__})"
            )
    return _long_term_thesis_from_yfinance(canonical) + _source_line(
        f"long_term_thesis via yfinance {canonical}"
    )


# ---------------------------------------------------------------------------
# Macro Analyst tools
# ---------------------------------------------------------------------------


@tool("get_macro_data")
@_memoize_tool
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
@_memoize_tool
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
@_memoize_tool
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
    return _web_news(profile.geopolitical_query(), max_results=6)


@tool("get_supply_chain_risk")
@_memoize_tool
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
    return _web_news(profile.supply_chain_query(), max_results=6)


# ---------------------------------------------------------------------------
# Cross-cutting research brief
# ---------------------------------------------------------------------------


@tool("get_market_context")
@_memoize_tool
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


def _upstox_options_summary(ticker: str) -> str:
    """Render an options summary from upstox v3 for an Indian F&O ticker.

    Uses the v2 put-call chain endpoint (still current on v3 accounts).
    We pick the nearest listed expiry from the exchange calendar
    implied by ``today + 7d`` — most NSE options list weekly-Thursday
    expiries, so a 30-day forward search picks up the front month
    reliably.  If the ticker isn't F&O-enabled the upstox call
    surfaces an ``UpstoxError`` which we render as a clean
    "not F&O enabled" line.
    """
    # Nearest common NSE weekly-Thursday expiries within the next 45 days.
    # Upstox returns 400 for unlisted expiries, so we walk forward from
    # the next Thursday until we get a non-empty chain.
    today = _dt.date.today()
    days_ahead = (3 - today.weekday()) % 7  # Thursday == 3
    if days_ahead == 0:
        days_ahead = 7
    candidates = [today + _dt.timedelta(days=days_ahead + 7 * i) for i in range(6)]
    last_error: Optional[Exception] = None
    for expiry in candidates:
        try:
            chain = upstox_source.fetch_option_chain(ticker, expiry.isoformat())
        except UpstoxError as exc:
            last_error = exc
            continue
        if not chain.rows:
            continue
        return upstox_source.render_option_chain_summary(
            chain, name=ticker,
        ) + _source_line(
            f"upstox v3 option_chain {ticker} expiry={expiry.isoformat()}"
        )
    return (
        f"upstox v3 returned no option chain for {ticker} across the next "
        f"{len(candidates)} weekly expiries — either the ticker is not "
        f"F&O enabled or the analytics token lacks options data. "
        f"Last error: {last_error}"
    ) + _source_line(f"upstox v3 option_chain {ticker} (no expiry matched)")


@tool("get_options_summary")
@_memoize_tool
def get_options_summary(ticker: str) -> str:
    """Retrieve nearest-expiry options summary (ATM call/put implied
    volatility, total open interest, put/call ratio) for a given ticker.

    For Indian tickers with upstox enabled we use the v3 option-chain
    endpoint — Yahoo does not expose Indian F&O reliably, so this is a
    strictly-additive capability.  For all other tickers we keep the
    existing yfinance path unchanged.
    """
    if upstox_source.is_indian_and_upstox_available(ticker):
        return _upstox_options_summary(ticker)
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
@_memoize_tool
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


@tool("get_forward_estimates")
@_memoize_tool
def get_forward_estimates(ticker: str) -> str:
    """Forward-looking consensus: analyst price targets + upside, the
    valuation *trajectory* (forward vs trailing P/E and EPS), consensus
    earnings / revenue growth for the current and next fiscal year,
    EPS-revision momentum, and the rating distribution.

    This is the deliberate counterweight to the backward-looking
    ``get_long_term_thesis`` (which is trailing statements only).  It
    answers "what does the street expect NEXT" rather than "what did the
    company do over the last N years", so the desk can weigh the FUTURE
    outlook rather than anchoring the call on the past.  These are
    sell-side *estimates* — they herd and are frequently wrong — so treat
    them as one input, not gospel; every line is provenance-tagged.
    """
    canonical = resolve_ticker(ticker)
    t = yf.Ticker(canonical)
    lines: list[str] = [
        f"FORWARD-LOOKING CONSENSUS for {canonical} "
        "(sell-side estimates — directional, can herd / be wrong):"
    ]

    def _pct(x) -> str:
        try:
            return f"{float(x) * 100:+.1f}%"
        except (TypeError, ValueError):
            return "n/a"

    def _num(x):
        try:
            v = float(x)
        except (TypeError, ValueError):
            return None
        return v if v == v else None  # drop NaN

    try:
        info = t.info or {}
    except Exception:
        info = {}

    # -- Price target + implied upside -----------------------------------
    spot = _num(info.get("currentPrice"))
    tgt_mean = _num(info.get("targetMeanPrice"))
    tgt_high = _num(info.get("targetHighPrice"))
    tgt_low = _num(info.get("targetLowPrice"))
    n_an = info.get("numberOfAnalystOpinions")
    if spot and tgt_mean:
        upside = (tgt_mean / spot - 1.0)
        lines.append(
            f"- Price target: spot {spot:.2f} → mean {tgt_mean:.2f} "
            f"({_pct(upside)} implied), range {tgt_low or 'n/a'}..{tgt_high or 'n/a'}"
            f"{f', {n_an} analysts' if n_an else ''} "
            f"[source: yfinance analyst_price_targets {canonical}]"
        )

    # -- Valuation trajectory (forward vs trailing) ----------------------
    fpe = _num(info.get("forwardPE"))
    tpe = _num(info.get("trailingPE"))
    feps = _num(info.get("forwardEps"))
    teps = _num(info.get("trailingEps"))
    if fpe and tpe:
        direction = "compressing (earnings expected to rise)" if fpe < tpe else (
            "expanding (earnings expected to fall)" if fpe > tpe else "flat")
        lines.append(
            f"- Valuation trajectory: forward P/E {fpe:.1f} vs trailing {tpe:.1f} "
            f"— {direction} [source: yfinance .info {canonical}]"
        )
    if feps is not None and teps is not None:
        eps_chg = (feps / teps - 1.0) if teps not in (0, None) else None
        lines.append(
            f"- EPS power: forward EPS {feps:.2f} vs trailing {teps:.2f}"
            f"{f' ({_pct(eps_chg)})' if eps_chg is not None else ''} "
            f"[source: yfinance .info {canonical}]"
        )

    # -- Consensus earnings / revenue growth -----------------------------
    def _est_line(df, label: str, source: str, *, scale: float = 1.0, unit: str = "") -> None:
        if df is None or getattr(df, "empty", True):
            return
        parts = []
        for period, human in (("0y", "FY0"), ("+1y", "FY+1")):
            if period in df.index:
                row = df.loc[period]
                avg = _num(row.get("avg"))
                growth = row.get("growth")
                if avg is not None:
                    val = f"{avg / scale:,.2f}{unit}" if scale != 1.0 else f"{avg:,.2f}{unit}"
                    parts.append(f"{human} {val} (growth {_pct(growth)})")
        if parts:
            lines.append(f"- {label}: " + "; ".join(parts) + f" [source: {source}]")

    try:
        _est_line(t.earnings_estimate, "Earnings estimates (EPS)",
                  f"yfinance earnings_estimate {canonical}")
    except Exception:
        pass
    try:
        _est_line(t.revenue_estimate, "Revenue estimates",
                  f"yfinance revenue_estimate {canonical}", scale=1e9, unit="B")
    except Exception:
        pass

    # -- EPS-revision momentum (last 30 days) ----------------------------
    try:
        rev = t.eps_revisions
        if rev is not None and not rev.empty:
            rparts = []
            for period, human in (("0y", "FY0"), ("+1y", "FY+1")):
                if period in rev.index:
                    up = rev.loc[period].get("upLast30days")
                    dn = rev.loc[period].get("downLast30days")
                    up = int(up) if up == up else 0
                    dn = int(dn) if dn == dn else 0
                    tilt = "↑" if up > dn else ("↓" if dn > up else "→")
                    rparts.append(f"{human} {up}up/{dn}down {tilt}")
            if rparts:
                lines.append(
                    "- EPS revision momentum (30d): " + "; ".join(rparts)
                    + f" [source: yfinance eps_revisions {canonical}]"
                )
    except Exception:
        pass

    # -- Long-term growth estimate ---------------------------------------
    try:
        ge = t.growth_estimates
        if ge is not None and not ge.empty and "LTG" in ge.index:
            ltg = _num(ge.loc["LTG"].get("stockTrend"))
            if ltg is not None:
                lines.append(
                    f"- Long-term growth estimate (LTG): {_pct(ltg)} "
                    f"[source: yfinance growth_estimates {canonical}]"
                )
    except Exception:
        pass

    # -- Rating distribution ---------------------------------------------
    try:
        rs = t.recommendations_summary
        if rs is not None and not rs.empty:
            row = rs.iloc[0]
            lines.append(
                "- Rating distribution (current month): "
                f"strongBuy {int(row.get('strongBuy', 0))}, buy {int(row.get('buy', 0))}, "
                f"hold {int(row.get('hold', 0))}, sell {int(row.get('sell', 0))}, "
                f"strongSell {int(row.get('strongSell', 0))} "
                f"[source: yfinance recommendations_summary {canonical}]"
            )
    except Exception:
        pass

    if len(lines) == 1:
        return (
            f"No forward-looking consensus available via yfinance for {canonical} "
            "(sparse analyst coverage — common for smaller / non-US names). "
            "Do NOT fabricate estimates; rely on the trailing fundamentals and "
            "label any forward view [analyst judgement]."
            + _source_line(f"yfinance forward estimates {canonical} (empty)")
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Trader tools (catalyst proximity + historical base-rate backtest)
# ---------------------------------------------------------------------------


@tool("get_event_proximity")
@_memoize_tool
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


def _simulate_horizon(
    closes,
    horizon: int,
    tgt: float,
    stp: float,
    *,
    direction: str = "long",
):
    """Walk one (target, stop, horizon) combo and return per-trade stats.

    ``direction`` selects the trade side:

    * ``"long"`` (default): entry at close, exit at ``+tgt`` (WIN), exit
      at ``-stp`` (LOSS), or timeout after ``horizon`` bars.
    * ``"short"``: entry at close, exit at ``-tgt`` (WIN — price fell
      the target magnitude), exit at ``+stp`` (LOSS — price rose the
      stop magnitude), or timeout.  Returns are **signed by the short
      side** — i.e. a −8% price move on a short is reported as a
      +8% realised return so ``expectancy`` / ``payoff`` are directly
      comparable to the long case.

    Historically this function only implemented the LONG case, and the
    trader passed positive magnitudes for both target and stop
    regardless of side.  That silently over-estimated SHORT-trade
    expectancy for any ticker with positive long-run drift (which is
    most surviving liquid names) — the "why does the number look
    similar across stocks" pattern the user reported was exactly this:
    ``expectancy ≈ +1 to +2%`` is the ambient LONG per-trade
    expectancy at a 2:1 payoff on a positive-drift universe.

    Returns ``(hit_rate, expectancy, payoff_ratio, avg, median, total,
    wins, losses, timeouts, avg_win, avg_loss)`` so the caller can render
    the table and compute a composite confidence-cap.

    Expectancy (per-trade expected pct return) is the *risk-adjusted*
    base rate the PM should care about — a 25% hit-rate with a 4:1
    payoff ratio still has positive expectancy and should NOT collapse
    confidence to a hard floor.
    """
    is_short = (direction or "long").strip().lower() == "short"
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
            # For SHORT trades the "trade-side return" is -r (price down
            # = short-side profit), and the exit conditions flip: WIN
            # when the price falls to -tgt (r <= -tgt); LOSS when the
            # price rises to +stp (r >= stp).
            if is_short:
                if r <= -tgt:
                    wins += 1
                    side_r = -r * 100  # positive number for SHORT-side win
                    win_returns.append(side_r)
                    all_returns.append(side_r)
                    outcome = "win"
                    break
                if r >= stp:
                    losses += 1
                    side_r = -r * 100  # negative number for SHORT-side loss
                    loss_returns.append(side_r)
                    all_returns.append(side_r)
                    outcome = "loss"
                    break
            else:
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
            terminal_r = (closes[i + horizon] / entry - 1) * 100
            if is_short:
                terminal_r = -terminal_r  # sign by short side
            timeout_returns.append(terminal_r)
            all_returns.append(terminal_r)

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
@_memoize_tool
def backtest_setup(
    ticker: str,
    horizon_days: int,
    target_pct: float,
    stop_pct: float,
    direction: str = "long",
) -> str:
    """Score a proposed trade plan against historical analogous setups.

    Walks up to 15 years of daily price history (whatever the source has).
    For each starting day, simulates entry at close, exit at target %
    move, exit at stop % move, or timeout after horizon_days —
    whichever comes first.

    ``direction`` selects the trade side (**pass it explicitly for
    every call — the default is only for backwards compatibility**):

    * ``"long"``:  WIN condition = price rises ``+target_pct``;
                   LOSS condition = price falls ``-stop_pct``.
    * ``"short"``: WIN condition = price falls ``target_pct``;
                   LOSS condition = price rises ``stop_pct``.
                   Returns are signed by the short side so
                   ``expectancy`` / ``payoff`` are directly comparable
                   to the long case (a short trade with expectancy
                   +2% is a genuine short-side edge, not "the price
                   went up 2%").

    Historically this function was LONG-only and traders passed
    positive magnitudes for both fields regardless of side — that
    silently over-estimated SHORT-trade expectancy for any ticker
    with positive long-run drift (which is most surviving liquid
    names).  The "expectancy ≈ +1-2% per trade" convergence across
    unrelated tickers reflected the ambient long-only per-trade
    expectancy at a 2:1 payoff on a positive-drift universe, not a
    genuine short-side edge.

    Returns the per-horizon table AND a multi-horizon panel that now
    spans SHORT / MEDIUM / LONG bands so the PM can weigh a genuinely
    long-term thesis (5–10 years) alongside the trader's short-term
    window:

    * SHORT tier:  60d
    * MEDIUM tier: 126d (6 months), 189d (9 months), 252d (1 year)
    * LONG tier:   504d (2 years), 1260d (5 years), 2520d (10 years,
                   only when >= 12 years of history is available)

    The 2y / 5y / 10y horizons are simulated with the same target /
    stop as the shorter panels so the PM sees a *risk-adjusted*
    long-hold comparison rather than a naive buy-and-hold CAGR.  When
    a horizon can't be simulated (not enough history), we surface an
    explicit "n=insufficient" line so the LLM does not silently treat
    a missing panel as neutral evidence.

    For each horizon we surface:

    * hit-rate                 — base rate the target gets hit
    * payoff ratio             — avg_win / |avg_loss|
    * expectancy (per trade)   — risk-adjusted base rate; can be positive
                                 even with a sub-40% hit-rate when the
                                 payoff ratio is favourable
    * avg / median realized %

    This richer view is what lets the PM avoid the "hit-rate < 40% =>
    HOLD by default" failure mode.  Whenever expectancy > 0 with
    payoff >= 1.5 on any horizon, the PM is allowed to express full
    conviction (BUY / STRONG_BUY, or SELL / STRONG_SELL for the bearish
    mirror) even if the short-term hit-rate is low.
    """
    _direction = (direction or "long").strip().lower()
    if _direction not in ("long", "short"):
        return (
            f"Invalid direction={direction!r}; must be 'long' or 'short'."
        )
    # Pull the longest history we can — 15y captures the medium-tier
    # (1y windows) with dozens of samples and gives the 5y horizon a
    # handful of overlapping episodes.  Anything shorter starves the
    # long-tier panels.  Sources without 15y just return what they have;
    # the "min-samples" gate below still catches an unusable window.
    try:
        df, source_id = _load_price_history(ticker, period="15y")
    except (UpstoxError, UpstoxAuthError, UpstoxNotFoundError) as exc:
        logger.warning("backtest_setup: upstox failed for %s: %s", ticker, exc)
        return (
            f"upstox v3 returned an error for {ticker}: {exc}. "
            "No backtest panel produced for this analyst turn."
        ) + _source_line(f"upstox v3 historical-candle {ticker} (failed: {type(exc).__name__})")
    if df.empty or len(df) < max(horizon_days + 30, 90):
        return f"Insufficient history for {ticker} backtest." + _source_line(
            f"{source_id} (insufficient for backtest)"
        )
    closes = df["Close"].values
    n = len(closes)
    tgt = abs(float(target_pct)) / 100.0
    stp = abs(float(stop_pct)) / 100.0

    short_h = max(5, int(horizon_days))
    # SHORT band anchors on the trader's chosen horizon + a stable 60d.
    # MEDIUM band = 6m / 9m / 1y (the "few months" bucket).
    # LONG band = 2y / 5y / 10y (the "5-10 year" bucket the user asked
    # for; 10y is dropped automatically when history can't cover it).
    all_horizons = sorted({short_h, 60, 126, 189, 252, 504, 1260, 2520})
    # Require a minimum of 20 non-overlapping starting points for a
    # horizon to be considered statistically meaningful — otherwise the
    # panel is dominated by a single episode and misleads the LLM.
    horizons = [h for h in all_horizons if (n - h) >= 20 and h + 20 <= n]
    dropped = [h for h in all_horizons if h not in horizons]
    if not horizons:
        horizons = [short_h]

    # Tier labels help the LLM cluster horizons by intent ("this is a
    # LONG-tier row, so a positive expectancy here supports the
    # buy-and-hold thesis; a positive expectancy on a SHORT row just
    # means the setup times well").  We prefix the row label so the
    # LLM doesn't have to memorise the tier boundaries.
    def _tier_for(h: int) -> str:
        from trading_crew.horizons import horizon_days_to_horizon
        return horizon_days_to_horizon(h).value

    rows = []
    for h in horizons:
        hit_rate, expectancy, payoff, avg, median, total, wins, losses, timeouts, avg_win, avg_loss = _simulate_horizon(
            closes, h, tgt, stp, direction=_direction
        )
        # Human-readable window: 21d -> "21d ≈ 1m", 252d -> "252d ≈ 1y",
        # 1260d -> "1260d ≈ 5y".  Makes the panel scan-friendly in the
        # LLM prompt without forcing it to convert trading days.
        if h >= 252:
            years = h / 252.0
            human = f"{h}d≈{years:.0f}y" if years >= 2 else f"{h}d≈1y"
        elif h >= 60:
            months = round(h / 21)  # ~21 trading days per month
            human = f"{h}d≈{months}m"
        else:
            human = f"{h}d"
        marker = " *" if h == short_h else ""
        label = f"[{_tier_for(h)}] {human}{marker}"
        rows.append({
            "h": h,
            "tier": _tier_for(h),
            "label": label,
            "human": human,
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
    # For SHORT trades the "win" condition is "price fell target% below
    # entry" — describe the target/stop in trade-side terms so the LLM
    # sees "target=-10% (price falls)" rather than the numerically-flipped
    # LONG framing.
    if _direction == "short":
        tgt_label = f"-{tgt*100:.1f}% price move (short-side target)"
        stp_label = f"+{stp*100:.1f}% price move (short-side stop)"
        side_banner = "SHORT-side simulation — returns are signed by the short side"
    else:
        tgt_label = f"+{tgt*100:.1f}% (long-side target)"
        stp_label = f"-{stp*100:.1f}% (long-side stop)"
        side_banner = "LONG-side simulation"
    legacy = (
        f"Backtest for {ticker} (direction={_direction.upper()}, "
        f"horizon={primary['h']}d, target={tgt_label}, stop={stp_label}):\n"
        f"- N samples: {primary_total}\n"
        f"- Wins (target hit): {primary['wins']} ({primary['hit_rate']*100:.1f}%)\n"
        f"- Stops hit: {primary['losses']} ({(primary['losses']/primary_total*100) if primary_total else 0:.1f}%)\n"
        f"- Timeouts: {primary['timeouts']} ({(primary['timeouts']/primary_total*100) if primary_total else 0:.1f}%)\n"
        f"- Avg realized return ({_direction}-side): {primary['avg']:+.2f}%, median {primary['median']:+.2f}%\n"
        f"- Avg win: {primary['avg_win']:+.2f}%, avg loss: {primary['avg_loss']:+.2f}%\n"
        f"- Payoff ratio: {primary['payoff']:.2f}, per-trade expectancy: {primary['expectancy']:+.2f}%"
    )

    # Multi-horizon panel — PM uses this for the short-vs-medium-vs-long view.
    multi_lines = [
        "",
        f"Multi-horizon panel ({side_banner}; SHORT vs MEDIUM vs LONG, "
        "same target/stop; ``*`` marks the trader's primary window):",
        "  window            | n      | hit-rate | payoff | expectancy | avg     | median",
    ]
    for r in rows:
        multi_lines.append(
            f"  {r['label']:<18}| {r['total']:>5}  | {r['hit_rate']*100:6.1f}%  | "
            f"{r['payoff']:>5.2f}x | {r['expectancy']:+6.2f}%   | "
            f"{r['avg']:+6.2f}% | {r['median']:+6.2f}%"
        )
    if dropped:
        # Surface *which* long-tier panels were dropped so the LLM
        # doesn't silently ignore the absence of e.g. the 10y row.
        multi_lines.append(
            f"  (dropped for insufficient history: "
            f"{', '.join(f'{d}d' for d in dropped)})"
        )

    # Headline: best expectancy + best hit-rate across the panel so the
    # PM can quote them directly without re-doing the math.  We ALSO
    # emit per-tier best-expectancy so the LLM can compare "long-tier
    # edge" vs "short-tier edge" without eyeballing.
    best_expectancy = max(rows, key=lambda r: r["expectancy"])
    best_hit = max(rows, key=lambda r: r["hit_rate"])
    per_tier_best: dict[str, dict] = {}
    for r in rows:
        prev = per_tier_best.get(r["tier"])
        if prev is None or r["expectancy"] > prev["expectancy"]:
            per_tier_best[r["tier"]] = r
    tier_lines = [
        f"  {tier:<7}: best {per_tier_best[tier]['human']:>9} — "
        f"expectancy {per_tier_best[tier]['expectancy']:+.2f}%, "
        f"hit-rate {per_tier_best[tier]['hit_rate']*100:.1f}%, "
        f"payoff {per_tier_best[tier]['payoff']:.2f}x"
        for tier in ("SHORT", "MEDIUM", "LONG")
        if tier in per_tier_best
    ]
    headline = [
        "",
        f"Best expectancy: {best_expectancy['expectancy']:+.2f}% on {best_expectancy['label']} "
        f"(hit-rate {best_expectancy['hit_rate']*100:.1f}%, payoff {best_expectancy['payoff']:.2f}x).",
        f"Best hit-rate:   {best_hit['hit_rate']*100:.1f}% on {best_hit['label']} "
        f"(expectancy {best_hit['expectancy']:+.2f}%).",
        "",
        "Per-tier best expectancy (SECONDARY corroboration — the primary "
        "horizon must come from the analyst thesis character, NOT this "
        "table; do NOT switch horizons to chase a better row):",
    ] + tier_lines

    return (
        legacy
        + "\n"
        + "\n".join(multi_lines)
        + "\n"
        + "\n".join(headline)
        + _source_line(
            f"backtest_setup({ticker}, dir={_direction}, "
            f"h={'+'.join(f'{h}d' for h in horizons)}, "
            f"tgt={tgt*100:.1f}%, stop={stp*100:.1f}%) on {source_id}"
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
    "get_corporate_actions": get_corporate_actions,
    "get_insider_transactions": get_insider_transactions,
    "get_fundamentals": get_fundamentals,
    "get_balance_sheet": get_balance_sheet,
    "get_cashflow": get_cashflow,
    "get_income_statement": get_income_statement,
    "get_shareholding_pattern": get_shareholding_pattern,
    "get_long_term_thesis": get_long_term_thesis,
    "get_macro_data": get_macro_data,
    "get_sector_peers": get_sector_peers,
    "get_geopolitical_news": get_geopolitical_news,
    "get_supply_chain_risk": get_supply_chain_risk,
    "get_options_summary": get_options_summary,
    "get_analyst_recommendations": get_analyst_recommendations,
    "get_forward_estimates": get_forward_estimates,
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
    # ---- 4 merged analyst desks --------------------------------------------
    # Market desk = price/technicals + options/positioning (was market + quant).
    "market_analyst": [
        "get_stock_data", "get_indicators",
        "get_options_summary", "get_analyst_recommendations",
        "retrieve_past_episodes", "rl_policy_recommendation",
    ],
    # News desk = catalysts + corporate actions + sentiment + insider
    # (was news + social).
    "news_analyst": [
        "get_market_context", "get_news", "get_global_news",
        "get_corporate_actions", "get_insider_transactions",
    ],
    # Fundamentals desk = financials + long-term thesis + sector/peers
    # (was fundamentals + sector).
    "fundamentals_analyst": [
        "get_fundamentals", "get_balance_sheet",
        "get_cashflow", "get_income_statement",
        # Indian tickers only: quarterly promoter/FII/DII/public split.
        # For non-Indian names the tool returns a one-line
        # "not-applicable" placeholder so the LLM does not waste a
        # turn on it.
        "get_shareholding_pattern",
        # LONG-horizon aggregator: revenue / EPS CAGR + ROCE trend +
        # debt trajectory + shareholding drift, all in one markdown
        # block.  The PM anchors the horizon=LONG decision on this
        # tool's output, so the fundamentals analyst is responsible
        # for pulling + citing it.
        "get_long_term_thesis",
        # Forward-looking counterweight to the (trailing) long-term thesis:
        # analyst targets + upside, forward vs trailing P/E and EPS, consensus
        # earnings/revenue growth, and revision momentum.  Ensures the desk
        # weighs the FUTURE outlook rather than anchoring only on the past.
        "get_forward_estimates",
        # Sector/peer relative strength (absorbed from the sector desk).
        "get_market_context", "get_sector_peers",
    ],
    # Macro desk = macro regime + geopolitical + supply-chain
    # (was macro + geopolitical).
    "macro_analyst": [
        "get_market_context", "get_macro_data", "get_global_news",
        "get_geopolitical_news", "get_supply_chain_risk",
    ],
    # NOTE: ``backtest_setup`` was intentionally removed from the trader's
    # toolset.  Past-heavy statistical expectancy was dominating the
    # trader + PM decision reasoning and dragging structurally-strong
    # names (esp. Indian growth stocks in an investment/production
    # upcycle) toward HOLD, while adding little forward-looking value.
    # The ``backtest_setup`` @tool and the standalone M6 walk-forward
    # engine (agentic/backtest/) remain available for the Backtest tab
    # and grid search — they are validation features, not decision inputs.
    "trader": ["get_event_proximity", "retrieve_past_episodes", "rl_policy_recommendation"],
}
