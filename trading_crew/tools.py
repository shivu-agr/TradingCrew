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

import datetime as _dt
import functools
import hashlib
import json
import logging
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf
from crewai.tools import tool
from tavily import TavilyClient

from trading_crew.data_sources import (
    get_config as get_data_sources_config,
    is_indian_ticker,
)
from trading_crew.data_sources import screener_source, sec_edgar_source, upstox_source
from trading_crew.data_sources.screener_source import (
    ScreenerError,
    ScreenerRateLimitError,
    ScreenerNotFoundError,
)
from trading_crew.data_sources.sec_edgar_source import (
    EdgarError,
    EdgarNotFoundError,
    EdgarRateLimitError,
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
# Hard cap on live entries.  Without this the cache is unbounded: a 200-name
# batch calls ~10 tools/name and every distinct (tool, args) result — some of
# them large (report text, indicator panels) — lingers here until the SAME key
# is read again, so expired entries are never freed and RSS climbs for hours.
# Cap + purge-on-write keeps it bounded regardless of batch size.
_TOOL_CACHE_MAX_ENTRIES_DEFAULT = 512
_tool_cache_lock = threading.Lock()
_tool_cache: dict[str, tuple[float, object]] = {}


def _tool_cache_ttl() -> float:
    try:
        return float(os.environ.get("TRADINGCREW_TOOL_CACHE_TTL_SECONDS", _TOOL_CACHE_TTL_DEFAULT))
    except ValueError:
        return _TOOL_CACHE_TTL_DEFAULT


def _tool_cache_max_entries() -> int:
    try:
        return int(os.environ.get("TRADINGCREW_TOOL_CACHE_MAX_ENTRIES", _TOOL_CACHE_MAX_ENTRIES_DEFAULT))
    except ValueError:
        return _TOOL_CACHE_MAX_ENTRIES_DEFAULT


def _prune_tool_cache_locked(now: float, ttl: float) -> None:
    """Evict expired entries, then oldest-first down to the size cap.

    Caller must hold ``_tool_cache_lock``.  Cheap: only walks the dict when it
    is over cap or has expired entries, which for a batch is at most once per
    tool call."""
    # 1) drop everything past its TTL (frees the bulk on a long batch).
    expired = [k for k, (ts, _) in _tool_cache.items() if (now - ts) >= ttl]
    for k in expired:
        _tool_cache.pop(k, None)
    # 2) if still over cap, evict oldest-first (dict preserves insertion order
    #    and we only ever append on write, so iteration order ~= age order).
    cap = _tool_cache_max_entries()
    if cap > 0 and len(_tool_cache) > cap:
        for k in list(_tool_cache.keys())[: len(_tool_cache) - cap]:
            _tool_cache.pop(k, None)


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
            _prune_tool_cache_locked(now, ttl)
        return result

    return wrapper


_QUOTA_MARKERS = (
    "usage limit",
    "exceeds this api key",
    "quota exceeded",
    "insufficient_quota",
    "rate limit exceeded",
    "ratelimit",
)


def _is_hard_quota(exc: BaseException) -> bool:
    """True when the provider is out of quota / hard-rate-limited.

    These are not transient — retrying the same key just burns wall time
    (Tavily quota exceeded was 3 sleeps per call on the NTNX run).
    """
    code = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if code == 429:
        return True
    msg = str(exc).lower()
    if any(m in msg for m in _QUOTA_MARKERS):
        return True
    return "429" in msg and any(k in msg for k in ("rate", "quota", "limit", "tavily"))


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

    Quota / 429 errors are not retried — they will not succeed on the
    same key later in the same process.
    """
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if _is_hard_quota(exc) or i == attempts - 1:
                break
            delay = min(max_delay, base_delay * (2 ** i)) + random.uniform(0, base_delay)
            logger.warning(
                "Tool retry %s/%s after %.2fs failed: %s (%s)",
                i + 1, attempts, delay, label or fn.__name__, exc,
            )
            time.sleep(delay)
    # Re-raise the last exception once attempts are exhausted.
    raise last_exc  # type: ignore[misc]


# Process-level: once every configured Tavily key reports quota, later
# news pulls skip Tavily entirely until the process restarts.
_tavily_quota_tripped = False


def reset_tavily_quota_flag() -> None:
    """Test helper — clear the process-level Tavily quota trip."""
    global _tavily_quota_tripped
    _tavily_quota_tripped = False

from trading_crew.market_context import (
    PeerRow,
    discover_peers,
    format_peer_comparison,
    get_market_profile,
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
    from trading_crew.data_sources import pit

    cutoff = pit.cutoff()
    if cutoff is not None:
        # Point-in-time backfill: only bars public on/before ``as_of`` may
        # be used.  yfinance start/end gives an exact cutoff; the upstox
        # "recent candles" path can't be bounded that way, so in backfill we
        # deliberately route through the date-bounded yfinance path (a
        # mode-based routing choice, not a hidden fallback).
        import datetime as _dt

        lookback = {
            "max": 20 * 365, "15y": 15 * 365, "10y": 10 * 365, "5y": 5 * 365,
            "2y": 2 * 365, "1y": 365, "2mo": 60,
        }.get(period, 5 * 365)
        end = cutoff.date()
        start = end - _dt.timedelta(days=lookback)
        df = _with_retry(
            yf.Ticker(ticker).history,
            start=start, end=end,
            label=f"yfinance.history({ticker}, {start}->{end} PIT)",
        )
        return df, f"yfinance OHLCV {ticker} (PIT as_of {end})"
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
    """Return a Tavily client using the next rotated key.

    Reads ``TAVILY_API_KEYS`` (round-robin) with a fallback to the singular
    ``TAVILY_API_KEY``.  Built fresh per call so rotation actually advances;
    the client is cheap to construct.  Kept as a seam so tests can
    monkeypatch the whole factory.
    """
    from trading_crew.data_sources import news_providers

    api_key = news_providers.next_key("tavily") or os.environ["TAVILY_API_KEY"]
    return TavilyClient(api_key=api_key)


def _coerce_absolute_http_url(url: str) -> str:
    """Turn a scheme-less host/path into ``https://…``.

    Search backends sometimes emit ``www.reuters.com/…`` without a
    scheme. Treating that as a relative placeholder dropped the row
    and the desk lost a real article. Relative paths (``/company/x``)
    and bare words (``not-a-url``) stay unchanged and still fail the
    placeholder check.
    """
    if not url or not isinstance(url, str):
        return ""
    u = url.strip()
    if not u:
        return ""
    if u.startswith("//"):
        return "https:" + u
    if u.startswith("http://") or u.startswith("https://"):
        return u
    if u.startswith("/") or " " in u:
        return u
    head = u.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    host = head.split(":", 1)[0]
    if host.count(".") < 1 or not re.search(r"[A-Za-z]", host):
        return u
    return "https://" + u.lstrip("/")


def _is_placeholder_url(url: str) -> bool:
    """True when ``url`` is empty, relative, or points at a known
    placeholder host — i.e. a fabricated citation rather than a real one."""
    if not url or not isinstance(url, str):
        return True
    u = _coerce_absolute_http_url(url).strip().lower()
    if not u:
        return True
    if not (u.startswith("http://") or u.startswith("https://")):
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
    """Drop fabricated links; rewrite scheme-less real hosts to https."""
    out: list[dict] = []
    for row in rows:
        url = _coerce_absolute_http_url(row.get("url", "") or "")
        if _is_placeholder_url(url):
            continue
        kept = dict(row)
        kept["url"] = url
        out.append(kept)
    return out


# Matches an inline provenance tag ``[source: <identifier>]`` (the tag the
# whole workflow uses to attribute a claim to a tool output).
_SOURCE_TAG_RE = re.compile(r"\[source:\s*([^\]]+?)\s*\]", re.IGNORECASE)

# Compound tags: ``url1; url2`` or ``url1, https://url2``. Do not split on
# a comma inside a query string — only when the next token starts a new
# citation (scheme, www., or host.tld).
_CITATION_PART_SPLIT_RE = re.compile(
    r"\s*;\s*|,\s+(?=(?:https?://|www\.|(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/|\s|$)))",
    re.IGNORECASE,
)


def _split_citation_payload(identifier: str) -> list[str]:
    raw = (identifier or "").strip()
    if not raw:
        return []
    parts = [
        p.strip().rstrip(",")
        for p in _CITATION_PART_SPLIT_RE.split(raw)
        if p.strip()
    ]
    return parts or [raw]


def _placeholder_host_in_identifier(ident: str) -> bool:
    """Placeholder host as a host token, not a substring of a real host.

    ``contest.com`` must not match ``test.com``; ``subdomain.com`` must
    not match ``domain.com``. A substring ``bad in ident`` check was
    rewriting those real citations to ``[analyst judgement]``.
    """
    ident_l = (ident or "").strip().lower()
    if not ident_l:
        return False
    for bad in _PLACEHOLDER_URL_HOSTS:
        if re.search(r"(?<![a-z0-9-])" + re.escape(bad) + r"(?![a-z0-9-])", ident_l):
            return True
    return False


def _is_fabricated_part(part: str) -> bool:
    ident = (part or "").strip()
    if not ident:
        return False
    low = ident.lower()
    if low.startswith("http://") or low.startswith("https://"):
        return _is_placeholder_url(ident)
    coerced = _coerce_absolute_http_url(ident)
    if coerced.startswith("http://") or coerced.startswith("https://"):
        return _is_placeholder_url(coerced)
    return _placeholder_host_in_identifier(ident)


def _is_fabricated_identifier(identifier: str) -> bool:
    """True when a ``[source: <identifier>]`` payload is a fabricated /
    placeholder citation rather than a real URL or a real tool identifier.

    We deliberately DO NOT treat every non-URL identifier as fabricated:
    legitimate tool identifiers (``yfinance .info ...``, ``screener.in
    ... #ratios``, ``long_term_thesis ...``) are not URLs and must be
    preserved.  Only two shapes are fabricated:

      1. an ``http(s)://`` URL that resolves to a placeholder host, or
      2. a host-token match on a known placeholder (``example.com`` etc.)
         even without a scheme.

    A compound tag is fabricated only when every part is. Mixed tags
    keep the real part (see ``sanitize_citations``).
    """
    if not identifier or not isinstance(identifier, str):
        return False
    parts = _split_citation_payload(identifier)
    if not parts:
        return False
    return all(_is_fabricated_part(p) for p in parts)


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
    left untouched.  A mixed tag (real URL + ``example.com``) keeps
    only the real part so a valid figure is not stripped.
    """
    if not text or not isinstance(text, str):
        return text

    def _repl(m: "re.Match[str]") -> str:
        identifier = m.group(1)
        parts = _split_citation_payload(identifier)
        kept = [p for p in parts if not _is_fabricated_part(p)]
        if not kept:
            return "[analyst judgement]"
        if kept == parts:
            return m.group(0)
        return "[source: " + "; ".join(kept) + "]"

    return _SOURCE_TAG_RE.sub(_repl, text)


# ---------------------------------------------------------------------------
# Source-provider normalization
# ---------------------------------------------------------------------------
# A ``[source: profit-loss]`` / ``[source: eps_revisions]`` tag names the
# section/field a number came from but hides the DATA PROVIDER the desk
# actually pulled it from.  These maps let us rewrite such a tag to a
# provider-led form — ``[source: screener.in · profit-loss]`` /
# ``[source: yahoo · eps_revisions]`` — deterministically, so the UI (and the
# critic's evidence list) reads the real origin rather than a bare section id.

# Section/field id -> data provider (unambiguous origin).
_SECTION_PROVIDER: dict[str, str] = {
    # screener.in financial-statement sections
    "profit-loss": "screener.in",
    "balance-sheet": "screener.in",
    "cash-flow": "screener.in",
    "ratios": "screener.in",
    "quarters": "screener.in",
    "shareholding": "screener.in",
    "shareholding_pattern": "screener.in",
    # yfinance / Yahoo Finance fields
    "eps_revisions": "yahoo",
    "analyst_price_targets": "yahoo",
    "earnings_estimate": "yahoo",
    "revenue_estimate": "yahoo",
    "recommendations_summary": "yahoo",
    "forward-estimates": "yahoo",
    "forward estimates": "yahoo",
    "insider_transactions": "yahoo",
    "option_chain": "yahoo",
    "sector_peers": "yahoo",
    "ohlcv": "yahoo",
    "info": "yahoo",
}

# Section/field ids whose PROVIDER is genuinely ambiguous (both an Indian
# screener.in path and a yfinance path exist).  We only supply the ``detail``
# and require an explicit provider token in the tag to attribute it.
_AMBIGUOUS_SECTIONS: set[str] = {"long_term_thesis"}

# Substring -> canonical provider label (explicit provider tokens win).
_PROVIDER_TOKENS: tuple[tuple[str, str], ...] = (
    ("screener.in", "screener.in"),
    ("screener", "screener.in"),
    ("yfinance", "yahoo"),
    ("yahoo", "yahoo"),
    ("upstox", "upstox"),
    ("tavily", "tavily"),
    ("serpapi", "serpapi"),
    ("firecrawl", "firecrawl"),
    ("nseindia", "nseindia"),
    ("moneycontrol", "moneycontrol"),
)


def normalize_source_identifier(identifier: str) -> str:
    """Rewrite a single ``[source: …]`` payload to a provider-led form.

    ``profit-loss`` → ``screener.in · profit-loss``; ``yfinance eps_revisions
    RELIANCE.NS`` → ``yahoo · eps_revisions``.  Real URLs and identifiers whose
    provider can't be determined are returned unchanged (idempotent).
    """
    if not identifier or not isinstance(identifier, str):
        return identifier
    raw = identifier.strip()
    low = raw.lower()
    # Real URLs are already provenance-complete.
    if low.startswith("http://") or low.startswith("https://"):
        return raw
    # Already provider-normalized (``screener.in · profit-loss, ratios``) —
    # return verbatim so re-normalization is idempotent and lossless.
    _canonical = {name for _, name in _PROVIDER_TOKENS}
    if " · " in raw and raw.split(" · ", 1)[0].strip().lower() in _canonical:
        return raw

    # Explicit provider token anywhere in the tag wins.
    provider: Optional[str] = None
    for token, name in _PROVIDER_TOKENS:
        if token in low:
            provider = name
            break

    # Find the section/field detail (prefer an explicit ``#section`` anchor).
    detail: Optional[str] = None
    inferred: Optional[str] = None
    if "#" in raw:
        seg = raw.split("#", 1)[1].strip().split()[0].strip().lower() if raw.split("#", 1)[1].strip() else ""
        if seg in _SECTION_PROVIDER:
            detail, inferred = seg, _SECTION_PROVIDER[seg]
        elif seg in _AMBIGUOUS_SECTIONS:
            detail = seg
    if detail is None:
        for token, prov in _SECTION_PROVIDER.items():
            if re.search(r"(?<![\w-])" + re.escape(token) + r"(?![\w-])", low):
                detail, inferred = token, prov
                break
    if detail is None:
        for token in _AMBIGUOUS_SECTIONS:
            if re.search(r"(?<![\w-])" + re.escape(token) + r"(?![\w-])", low):
                detail = token
                break

    provider = provider or inferred
    if provider is None:
        # Unknown origin — leave the identifier untouched.
        return raw
    if not detail or detail == provider:
        return provider
    return f"{provider} · {detail}"


def _normalize_source_payload(payload: str) -> str:
    """Normalize a (possibly ``;``-joined) ``[source: …]`` payload, collapsing
    consecutive same-provider entries (``a · x; a · y`` → ``a · x, y``)."""
    parts = [p.strip() for p in payload.split(";") if p.strip()]
    if not parts:
        return payload
    out: list[str] = []
    for part in parts:
        norm = normalize_source_identifier(part)
        if " · " in norm and out and " · " in out[-1] and \
                out[-1].split(" · ", 1)[0] == norm.split(" · ", 1)[0]:
            out[-1] = out[-1] + ", " + norm.split(" · ", 1)[1]
        else:
            out.append(norm)
    return "; ".join(out)


def normalize_source_providers(text: str) -> str:
    """Rewrite every inline ``[source: <section>]`` tag in ``text`` to the
    provider-led ``[source: <provider> · <section>]`` form.  Deterministic and
    idempotent; URLs and unknown identifiers pass through unchanged."""
    if not text or not isinstance(text, str):
        return text

    def _repl(m: "re.Match[str]") -> str:
        return "[source: " + _normalize_source_payload(m.group(1)) + "]"

    return _SOURCE_TAG_RE.sub(_repl, text)


# ---------------------------------------------------------------------------
# Stamp [source: …] on numeric lines the LLM left bare
# ---------------------------------------------------------------------------
# Quality Review treats an untagged number as UNSUPPORTED. Map/reduce and
# custom-crawl summaries often emit real figures with a prose "Source"
# column ("Annual report", "Q1 call") instead of ``[source: <url>]``.
# This pass attaches the matching filing/article URL — or the tool id —
# so a valid number is not lost downstream.

_HAS_NUMBER_RE = re.compile(
    r"(?<![A-Za-z./])(?:\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+\.\d+|\d{2,}|\d\s*%|\d%)"
)
_PROVENANCE_TAG_RE = re.compile(r"\[(?:source|analyst judgement)\s*:", re.IGNORECASE)
_SIGNAL_INSUFFICIENT_RE = re.compile(
    r"^\s*SIGNAL\s*\|[^|\n]*\|\s*INSUFFICIENT_DATA\b", re.IGNORECASE
)
_MD_SEP_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")

_KIND_HINTS: dict[str, tuple[str, ...]] = {
    "concall": (
        "concall", "earnings call", "transcript",
        "q1 fy", "q2 fy", "q3 fy", "q4 fy",
        "q1-fy", "q2-fy", "q3-fy", "q4-fy",
        "fy26 call", "fy-26 call", "fy 26 call",
    ),
    "earnings": ("earnings", "8-k", "press release", "ex-99", "exhibit 99"),
    "annual_report": ("annual report",),
    "quarterly_report": ("quarterly report", "10-q"),
    "credit_rating": (
        "crisil", "care rating", "rating update", "rating note", "rating docs",
    ),
    "announcement": ("sebi", "lodr", "announcement"),
}


def _host_labels(url: str) -> list[str]:
    host = (url or "").split("://", 1)[-1].split("/", 1)[0].lower()
    host = host.split(":", 1)[0]
    if host.startswith("www."):
        host = host[4:]
    if not host or "." not in host:
        return []
    labels = [host, host.split(".")[0]]
    if host.startswith("care"):
        labels.append("care")
    return labels


def _citation_for_line(
    line: str, default_source: str, citations: list[dict],
) -> str:
    if not citations:
        return default_source
    low = line.lower()
    scored: list[tuple[int, str]] = []
    for cite in citations:
        url = (cite.get("url") or "").strip()
        if not url:
            continue
        kind = (cite.get("kind") or "").strip().lower()
        title = (cite.get("title") or "").strip().lower()
        score = 0
        for hint in _KIND_HINTS.get(kind, ()):
            if hint in low:
                score += 2
                break
        if title and len(title) >= 12 and title[:24] in low:
            score += 2
        for label in _host_labels(url):
            if re.search(
                r"(?<![a-z0-9])" + re.escape(label) + r"(?![a-z0-9])", low,
            ):
                score += 3
                break
        if score:
            scored.append((score, url))
    if not scored:
        return default_source
    best = max(s for s, _ in scored)
    urls: list[str] = []
    seen_kind: set[str] = set()
    # Preserve catalog order (newest first); one URL per kind at best score.
    for cite in citations:
        url = (cite.get("url") or "").strip()
        kind = (cite.get("kind") or "").strip().lower()
        score = next((s for s, u in scored if u == url), 0)
        if score != best or not url:
            continue
        key = kind or url
        if key in seen_kind:
            continue
        seen_kind.add(key)
        urls.append(url)
    return "; ".join(urls) if urls else default_source


def _retag_numeric_claims(
    text: str,
    *,
    default_source: str,
    citations: Optional[list] = None,
) -> str:
    """Append ``[source: …]`` to numeric lines the tool left untagged.

    Already-tagged lines and INSUFFICIENT_DATA signals are unchanged.
    """
    if not text or not isinstance(text, str) or not default_source:
        return text
    cites = citations or []
    out: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if (
            not stripped
            or _PROVENANCE_TAG_RE.search(line)
            or _SIGNAL_INSUFFICIENT_RE.match(stripped)
            or _MD_SEP_RE.match(stripped)
            or not _HAS_NUMBER_RE.search(line)
        ):
            out.append(line)
            continue
        src = _citation_for_line(line, default_source, cites)
        if not src:
            out.append(line)
            continue
        out.append(line.rstrip() + f" [source: {src}]")
    return "\n".join(out)


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
    coverage.  Returns ``[]`` (never raises) so the merge always proceeds.

    Key rotation happens inside :func:`_tavily` (``TAVILY_API_KEYS`` round-
    robin, falling back to the singular ``TAVILY_API_KEY``).  When a backfill
    ``AsOfContext`` is active the search is bounded server-side via
    ``end_date`` and each row carries the ``published_date`` for the
    downstream PIT filter.
    """
    from trading_crew.data_sources import news_providers, pit

    global _tavily_quota_tripped
    if _tavily_quota_tripped:
        return []

    # Try each configured key in round-robin order: a rate-limited / invalid
    # key rolls over to the next rather than failing the whole provider.
    # ``_tavily()`` rotates internally on each call (and stays monkeypatchable
    # as a no-arg factory for the test-suite).
    n_attempts = max(1, len(news_providers.read_keys("tavily")))
    res = None
    quota_hits = 0
    for _ in range(n_attempts):
        try:
            kwargs = dict(max_results=max_results, topic="news", search_depth="advanced")
            cutoff = pit.cutoff()
            if cutoff is not None:
                kwargs["end_date"] = cutoff.strftime("%Y-%m-%d")
            try:
                res = _with_retry(_tavily().search, query, label=f"tavily.search({query!r})", **kwargs)
            except TypeError:
                kwargs.pop("end_date", None)
                res = _with_retry(_tavily().search, query, label=f"tavily.search({query!r})", **kwargs)
            break
        except Exception as exc:  # noqa: BLE001 — try the next key, then give up
            if _is_hard_quota(exc):
                quota_hits += 1
                logger.warning("tavily quota/rate-limit for %r: %s", query, exc)
            else:
                logger.warning("tavily news failed for %r: %s", query, exc)
            res = None
            continue
    if res is None:
        if quota_hits >= n_attempts:
            _tavily_quota_tripped = True
            logger.warning(
                "tavily quota exhausted on all %s key(s); skipping remaining "
                "tavily calls this process",
                n_attempts,
            )
        return []
    rows: list[dict] = []
    for r in res.get("results", []):
        rows.append(
            {
                "title": (r.get("title") or "").strip(),
                "url": (r.get("url") or "").strip(),
                "snippet": (r.get("content") or "").strip().replace("\n", " ")[:240],
                "provider": "tavily",
                "published": r.get("published_date"),
            }
        )
    return rows


def _serpapi_news(query: str, max_results: int = 8) -> list[dict]:
    """Fetch news rows from SerpAPI's Google News engine.  Returns ``[]``
    when the key is unset or the call fails (merge must survive).

    Uses key rotation (``SERPAPI_API_KEYS`` round-robin, falling back to the
    singular ``SERPAPI_API_KEY``) and carries the per-result ``date`` for
    the downstream point-in-time filter."""
    from trading_crew.data_sources import news_providers

    # Round-robin across every configured SerpAPI key, rolling over to the
    # next one when a key is rate-limited / invalid (falls back to the
    # singular ``SERPAPI_API_KEY`` when no plural list is set).
    keys = news_providers.ordered_keys("serpapi")
    if not keys:
        solo = os.environ.get("SERPAPI_API_KEY")
        keys = [solo] if solo else []
    if not keys:
        return []
    data = None
    for api_key in keys:
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
            break
        except Exception as exc:  # noqa: BLE001 — try the next key, then give up
            logger.warning("serpapi news failed for %r: %s", query, exc)
            data = None
            continue
    if data is None:
        return []

    def _row(r: dict) -> dict | None:
        url = (r.get("link") or "").strip()
        title = (r.get("title") or "").strip()
        if not url or not title:
            return None
        src = r.get("source") or {}
        src_name = src.get("name") if isinstance(src, dict) else str(src)
        snippet = (r.get("snippet") or src_name or "").strip().replace("\n", " ")[:240]
        return {"title": title, "url": url, "snippet": snippet, "provider": "serpapi", "published": r.get("date")}

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


def _firecrawl_news(query: str, max_results: int = 8) -> list[dict]:
    """Fetch news rows from Firecrawl web-search (with key rotation).

    Returns ``[]`` when no Firecrawl key is set or the call fails, so the
    merge always proceeds.  Rotation + the HTTP call live in
    :mod:`trading_crew.data_sources.news_providers`.
    """
    from trading_crew.data_sources import news_providers

    try:
        return news_providers.firecrawl_rows(query, max_results)
    except Exception as exc:  # noqa: BLE001 — merge must survive one provider
        logger.warning("firecrawl news failed for %r: %s", query, exc)
        return []


def _pit_and_dedupe_news_rows(rows: list[dict], max_results: int) -> list[dict]:
    """PIT-filter, drop fabricated links, and de-duplicate a provider's rows."""
    from trading_crew.data_sources import pit

    # Point-in-time gate — no-op in live mode; in backfill drops anything
    # published after the as_of cutoff (and unknown-date rows in strict mode).
    rows = pit.filter_before_cutoff(rows, key="published")
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
    return merged[: max_results * 2]


# News providers, tried in a FALLBACK SEQUENCE (not a parallel merge): the
# first provider that returns any usable, PIT-valid rows wins, so we spend the
# cheaper / higher-quota providers first and only reach the next one when the
# prior yields nothing.  Each provider internally rotates its own key list.
# Stored as (label, row-builder attribute name) and resolved on the module at
# call time so the seam stays monkeypatchable in tests.
_NEWS_PROVIDER_SEQUENCE: tuple[tuple[str, str], ...] = (
    ("tavily", "_tavily_news_rows"),
    ("serpapi", "_serpapi_news"),
    ("firecrawl", "_firecrawl_news"),
)


def _web_news_rows(query: str, max_results: int = 6) -> list[dict]:
    """Return news rows from the first provider in the fallback SEQUENCE that
    yields usable, PIT-valid, de-duplicated rows.

    Order: **Tavily → SerpAPI → Firecrawl** (each with its own key rotation).
    This is a fallback chain, not a breadth merge: SerpAPI is only hit when
    Tavily comes back empty, and so on — conserving each provider's quota.
    """
    for label, fn_name in _NEWS_PROVIDER_SEQUENCE:
        fn: "Callable[[str, int], list[dict]]" = globals()[fn_name]
        try:
            rows = fn(query, max_results + 4)
        except Exception as exc:  # noqa: BLE001 — defensive; providers already guard
            logger.warning("news provider %s failed for %r: %s", label, query, exc)
            rows = []
        merged = _pit_and_dedupe_news_rows(rows, max_results)
        if merged:
            return merged
    return []


def _web_news_rows_free(query: str, max_results: int = 6) -> list[dict]:
    """FREE news rows via the custom link-search backends (no paid API).

    Uses :func:`web_research.search_links` (websearch.sh → ddg/brave/mojeek —
    all keyless) and normalises the hits into the same row shape the paid
    providers return, so callers that expect ``[{title,url,snippet,provider}]``
    (e.g. :func:`get_corporate_actions`) can consume it interchangeably.  The
    SearXNG backend supplies a per-result snippet (carried through here);
    scraper backends don't, so those rows have an empty snippet and no publish
    date, and are PIT-filtered fail-closed in strict backfill (kept in live
    mode).  Runs the same placeholder-URL rejection + de-dupe as the paid tier.
    """
    from trading_crew.data_sources import web_research

    try:
        links = web_research.search_links(query, max_results=max_results + 4)
    except Exception as exc:  # noqa: BLE001 — free tier must never raise
        logger.warning("free news search failed for %r: %s", query, exc)
        return []
    rows = [
        {
            "title": (l.get("title") or l.get("url") or "").strip(),
            "url": (l.get("url") or "").strip(),
            # SearXNG supplies a per-result snippet; scraper backends don't, so
            # this is best-effort and may be empty.
            "snippet": (l.get("snippet") or "").strip().replace("\n", " ")[:240],
            "provider": "websearch",
        }
        for l in links
    ]
    return _pit_and_dedupe_news_rows(rows, max_results)


def _corporate_action_rows(query: str, max_results: int = 4) -> list[dict]:
    """Fetch corporate-action rows honouring TRADINGCREW_PREFER_CUSTOM_CRAWL.

    Custom-first (default): try the FREE link-search tier first so a
    large-scale run doesn't burn Tavily/SerpAPI/Firecrawl quota, and only fall
    through to the paid APIs when the free tier is empty.  API-first when the
    toggle is off (better publish-date PIT for historical backfill).
    """
    if _prefer_custom_crawl():
        return _web_news_rows_free(query, max_results) or _web_news_rows(query, max_results)
    return _web_news_rows(query, max_results) or _web_news_rows_free(query, max_results)


def _web_news_via_research(query: str, *, crawler: str = "http") -> str:
    """Custom websearch pipeline (free search → crawl → LLM-summarize).
    Returns a source-cited summary string, or "" on failure.

    ``crawler`` defaults to ``"http"`` (keyless requests+BeautifulSoup) so the
    custom tier consumes NO paid credits — the link search already uses only
    free backends.  Pass ``crawler="auto"`` to allow Firecrawl as the crawler.

    PIT-aware via :func:`web_research` (the search is date-bounded and the
    summarizer is told to ignore post-cutoff info when an ``AsOfContext`` is
    active).  See ``web_research`` for the residual crawl-freshness caveat.
    """
    from trading_crew.data_sources import web_research

    try:
        res = web_research.research_and_summarize(query, max_links=5, crawler=crawler)
    except Exception as exc:  # noqa: BLE001 — last-resort provider must not raise
        logger.warning("websearch news fallback failed for %r: %s", query, exc)
        return ""
    summary = (res.get("summary") or "").strip()
    if not summary:
        # Don't fail silently — record WHY the fallback came back empty so a
        # degraded run is traceable in the logs (search vs. crawl failure).
        logger.warning(
            "websearch news fallback empty for %r (status=%s)",
            query, res.get("status", "unknown"),
        )
        return ""
    citations = [
        {"url": (s.get("url") or "").strip(), "kind": "", "title": s.get("title") or ""}
        for s in (res.get("sources") or [])
        if (s.get("url") or "").strip()
    ]
    return _retag_numeric_claims(
        summary,
        default_source=f"Web News (custom crawl) · q='{query}'",
        citations=citations,
    )


def _prefer_custom_crawl() -> bool:
    """Whether to try the FREE custom-crawl tier before the paid news APIs.

    Default ON (``TRADINGCREW_PREFER_CUSTOM_CRAWL=1``): with limited
    Tavily/SerpAPI/Firecrawl credits, spend the free tier (free link search +
    keyless HTTP crawl + one LLM summarize) first and only fall through to the
    paid APIs when it comes back empty.  Set to ``0`` to restore the old
    API-first order (which carries stronger publish-date PIT metadata for
    backfill).
    """
    return os.getenv("TRADINGCREW_PREFER_CUSTOM_CRAWL", "1").strip().lower() not in (
        "0", "false", "no", "off", ""
    )


# When the free custom-crawl tier crawls generic pages (its keyless HTTP
# crawler is 403-blocked by bot-protected news sites), the summarizer
# correctly reports that the sources are irrelevant — but that self-report is
# itself a *non-empty* string.  Without this check the custom tier would ship
# a "no relevant info" digest and the configured paid news APIs (Tavily /
# SerpAPI / Firecrawl) would never be reached.  So we treat a summary that
# explicitly says the sources are irrelevant/empty as an empty result and fall
# through to the paid tier.
# Whole-digest emptiness only. Bare "irrelevant" / "no news" also match
# a useful crawl that says "ignored irrelevant Wikipedia pages" or
# "no news of a tariff change, but supply strains delayed revenue" —
# those used to discard the digest and fall through to an empty API
# (NTNX get_geopolitical_news → "No news found").
_IRRELEVANCE_MARKERS: tuple[str, ...] = (
    "do not contain any",
    "does not contain any",
    "not contain any information",
    "no relevant information",
    "no relevant recent news found",
    "no factual briefing",
    "cannot be compiled",
    "no articles found",
    "no headlines found",
)


def _is_degraded_news_summary(summary: str) -> bool:
    """True when a custom-crawl summary self-reports that its crawled sources
    were irrelevant / empty, so ``_web_news`` can fall through to the paid
    news APIs instead of shipping a 'no relevant info' digest."""
    low = (summary or "").lower()
    return any(m in low for m in _IRRELEVANCE_MARKERS)


def _web_news(query: str, max_results: int = 6) -> str:
    """Fetch news and render a markdown digest with real article URLs.

    Two clearly-separated tiers:

    * **Custom crawl (free)** — free link search + keyless HTTP crawl +
      one LLM summarize.  No paid API credits.
    * **News APIs (paid)** — Tavily → SerpAPI → Firecrawl fallback sequence,
      each row carrying an inline ``[source: <url>]`` tag and publish date.

    Order is controlled by ``TRADINGCREW_PREFER_CUSTOM_CRAWL`` (default ON =
    free tier first, to conserve API credits).  Each tier is only reached
    when the earlier one comes back empty.
    """

    def _custom() -> str:
        summary = _web_news_via_research(query)
        if summary and not _is_degraded_news_summary(summary):
            return summary + _source_line(f"Web News (custom crawl) · q='{query}'")
        return ""

    def _apis() -> str:
        merged = _web_news_rows(query, max_results)
        if not merged:
            return ""
        body = "\n".join(
            f"- {r['title']}\n  {r['url']}\n  {r['snippet']}\n  [source: {r['url']}]"
            for r in merged
        )
        provider_tag = "+".join(sorted({r["provider"] for r in merged}))
        return body + _source_line(f"Web News ({provider_tag}) · q='{query}'")

    stages = (_custom, _apis) if _prefer_custom_crawl() else (_apis, _custom)
    for stage in stages:
        out = stage()
        if out:
            return out

    logger.warning("all news tiers empty for %r", query[:180])
    return "No news found." + _source_line(f"Web News (none) · q='{query}'")


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

    cite = f" [source: {source_id} · monthly + recent daily]"
    summary_lines = [
        f"5-YEAR OVERVIEW for {ticker}",
        f"  Current price: {last_px:.2f}  (as of {last_date}){cite}",
        f"  52-week high : {hi_val:.2f}  on {hi_date}{cite}",
        f"  52-week low  : {lo_val:.2f}  on {lo_date}{cite}",
        (
            f"  Returns: 1m={_ret_n_days(21)}  3m={_ret_n_days(63)}  "
            f"6m={_ret_n_days(126)}  1y={_ret_n_days(252)}  "
            f"3y={_ret_n_days(252*3)}  5y={ret_5y_pct:+.1f}%{cite}"
        ),
        f"  Annualised volatility (1y trailing): {vol_str}{cite}",
        f"  Max drawdown (5y): {max_dd:.1f}%  (trough {max_dd_date}){cite}",
    ]

    daily_tail = df[["Open", "High", "Low", "Close", "Volume"]].tail(30).round(2)

    return (
        "\n".join(summary_lines)
        + f"\n\nMONTHLY OHLCV ({len(monthly)} months, 5y window):{cite}\n"
        + monthly.round(2).to_string()
        + f"\n\nDAILY OHLCV (last {len(daily_tail)} sessions):{cite}\n"
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

    base = (
        f"{ticker} indicators: price={close.iloc[-1]:.2f}, "
        f"SMA20={_fmt(sma20)}, SMA50={_fmt(sma50)}, SMA200={_fmt(sma200)}, "
        f"RSI14={_fmt(rsi)}, "
        f"60d_return={_fmt_pct(ret60)}, 1y_return={_fmt_pct(ret252)}"
        f" [source: {source_id} · indicators]"
        + _source_line(f"{source_id} · indicators")
    )

    # Append the deterministic PRICE STRUCTURE SIGNALS card (vol-state /
    # regime-age / oversold-dip / Hurst).  These are probe-validated
    # risk/entry signals (see ``trading_crew.price_structure``) that inform
    # SIZE and flag a mean-reversion ENTRY setup — never a direction vote.
    # Best-effort: a failure here must not lose the base indicator panel.
    try:
        from trading_crew.price_structure import compute_price_structure

        ps = compute_price_structure(df, source=f"{source_id} · price-structure (computed)")
        card = ps.render_card() if ps is not None else ""
        if card:
            return base + "\n\n" + card
    except Exception:  # noqa: BLE001 — price-structure card is additive
        logger.warning("get_indicators: price-structure card failed for %s", ticker, exc_info=True)
    return base


# ---------------------------------------------------------------------------
# Social / News / Insider tools
# ---------------------------------------------------------------------------


@tool("get_news")
@_memoize_tool
def get_news(ticker: str, query: str = "") -> str:
    """Retrieve recent company news.

    After ``get_market_context``, pass ``query`` you wrote from the
    brief + your catalyst remit + any gap (6-12 words). If omitted,
    the tool searches the company name only — it does not carry an
    industry phrasebook.
    """
    from trading_crew.market_context import compose_search_query

    canonical = resolve_ticker(ticker)
    profile = get_market_profile(canonical)
    return _web_news(
        compose_search_query(profile, lens="company", agent_query=query),
        max_results=6,
    )


@tool("get_global_news")
@_memoize_tool
def get_global_news(ticker: str = "", query: str = "") -> str:
    """Retrieve home-country macro news.

    Pass ``query`` you wrote from the brief (country, currency, macro
    basket) + your remit. If omitted, searches ``{country} central bank
    inflation currency`` — not a canned Fed/RBI headline list.
    """
    from trading_crew.market_context import compose_search_query

    if query:
        return _web_news(query, max_results=6)
    if not ticker:
        return _web_news(
            "global markets macro overview central bank policy currencies",
            max_results=6,
        )
    canonical = resolve_ticker(ticker)
    profile = get_market_profile(canonical)
    return _web_news(compose_search_query(profile, lens="macro"), max_results=6)


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
# enough that a generic "latest stock news headlines earnings analyst
# ratings" query (``get_news`` above) would routinely miss them — buybacks,
# dividends,
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
            results = _corporate_action_rows(query, max_results=4)
        except Exception as exc:
            logger.warning(
                "get_corporate_actions: %s theme %s failed: %s",
                canonical, theme, exc,
            )
            sections.append(f"### {theme}\n  (search failed: {exc})")
            continue
        rows = []
        for r in results:
            snippet = (r.get("snippet") or "").strip()[:280]
            line = f"- {r['title']}\n  {r['url']}"
            if snippet:
                line += f"\n  {snippet}"
            line += f"\n  [source: {r['url']}]"
            rows.append(line)
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
            # Grouped, dated primary sources (announcements + annual reports +
            # concall transcripts + credit-rating actions).  PIT-filtered in
            # backfill so only filings public as-of the decision date show.
            docs_block = screener_source.render_documents_report(page, limit_per_kind=8)
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


# ---------------------------------------------------------------------------
# Primary-filing report insights (guidance / target-vs-actual)
# ---------------------------------------------------------------------------
#
# The fundamentals desk reads restated *ratio* tables; nobody in the workflow
# reads the company's own PRIMARY filings (earnings-call / concall transcripts,
# investor presentations, annual-report MD&A, credit-rating rationales) where
# management states explicit forward TARGETS and where a prior target can be
# compared to what was actually DELIVERED.  ``get_report_insights`` fills that
# gap market-aware: Indian (NSE/BSE) names via screener.in's dated
# ``#documents`` feed, and US (SEC-registered) names via the SEC EDGAR
# submissions feed (10-K / 10-Q + earnings 8-Ks).  It crawls the (large)
# documents and LLM-summarises them before the digest reaches the agent (so the
# analyst never ingests a 200-page filing), returning a compact guidance /
# target-vs-actual / outlook digest.  It is PIT-safe: filings are first
# date-filtered to on/before the decision cutoff, and the summariser is told to
# ignore anything dated after it.
#
# BIG PDFs are NOT truncated.  Chunk size is derived from
# ``REPORT_PROCESS_CONTEXT_TOKENS`` (128k or 256k) so a filing is split
# only when it will not fit the model — not into many small slices.
# MAP keeps guidance AND the broader briefing (actuals, order book, risks,
# MD&A, operations). REDUCE consolidates that into the digest. The
# newest annual report is enough (prior years are inside it); concalls
# still keep multiple vintages for target-vs-actual.

# Guidance-rich documents first: concall transcripts + investor decks carry the
# most explicit forward targets, then earnings press releases (US 8-K item
# 2.02 — results + guidance), annual reports the MD&A / outlook, quarterly
# reports the interim MD&A, credit ratings the agency's forward view; generic
# announcements are the fallback.  Screener (India) emits concall /
# annual_report / credit_rating / announcement; SEC EDGAR (US) emits earnings /
# annual_report / quarterly_report — the union is ordered here.
_REPORT_DOC_PRIORITY = (
    "concall",
    "earnings",
    "annual_report",
    "quarterly_report",
    "credit_rating",
    "announcement",
)

_REPORT_INSIGHT_PROMPT = (
    "You are an equity analyst briefing a prediction desk from a company's "
    "OWN primary filings (earnings-call / concall transcripts, investor "
    "presentations, annual-report MD&A, credit-rating rationales) for "
    "{name} ({ticker}).\n\n"
    "Read the source excerpts below and produce a briefing the desk can "
    "trade on — guidance AND the broader operating picture (actuals, "
    "order book, risks, strategy, operations). Put an inline "
    "[source: <url>] tag on EVERY concrete figure, using the source's "
    "URL.\n\n"
    "FIRST, emit a machine-readable TREND SIGNAL CARD. This is the headline "
    "deliverable: a short set of DIRECTIONAL, confidence-scored signals a "
    "trader can act on — each derived ONLY from the filings, comparing the "
    "LATEST statement to the PRIOR period wherever the sources allow (that is "
    "what makes it a TREND, not a snapshot). Output this block EXACTLY in this "
    "format, one line per signal, no extra prose:\n\n"
    "## REPORT TREND SIGNALS\n"
    "SIGNAL | <NAME> | <DIRECTION> | conf=<0.00-1.00> | <one-line rationale> "
    "[source: <url>]\n\n"
    "Emit these signals IN THIS ORDER (skip a line only if the filings truly "
    "say nothing about it — then emit DIRECTION=INSUFFICIENT_DATA, conf=0.0, "
    "and NO source):\n"
    "- OUTLOOK_TONE: BULLISH | CAUTIOUS | DEFENSIVE — management's forward "
    "tone/confidence.\n"
    "- REVENUE_GUIDANCE: RAISED | MAINTAINED | CUT — latest revenue/topline "
    "guidance vs the prior guide (or vs last year if that's the only compare).\n"
    "- MARGIN_GUIDANCE: EXPANDING | STABLE | CONTRACTING — margin / "
    "profitability guidance direction.\n"
    "- DELIVERY_TRACK_RECORD: BEAT | IN_LINE | MISS — how recent actuals came "
    "in vs prior guidance (note the streak, e.g. 'beat 3 of last 4').\n"
    "- DEMAND_MOMENTUM: ACCELERATING | STABLE | DECELERATING — order-book / "
    "deal-pipeline / demand-commentary trend.\n"
    "- CREDIT_TRAJECTORY: IMPROVING | STABLE | DETERIORATING — rating action / "
    "outlook direction (skip if no rating info).\n"
    "- NET_GUIDANCE_BIAS: POSITIVE | NEUTRAL | NEGATIVE — your synthesis of the "
    "above into ONE actionable bias, with the confidence you'd stand behind.\n\n"
    "THEN the supporting evidence. Keep material facts — do not drop a "
    "number, risk, or operating detail just because it is not a forward "
    "target:\n\n"
    "1. MANAGEMENT GUIDANCE / TARGETS — explicit forward targets with NUMBER "
    "and PERIOD: revenue / earnings growth, margin, order book / TCV / "
    "book-to-bill, capex, capacity, dividend / payout, debt / net-cash.\n"
    "2. TARGET vs ACTUAL — prior guide vs delivered, labelled BEAT / MISS / "
    "IN-LINE.\n"
    "3. REPORTED ACTUALS & SEGMENTS — latest-period revenue, PAT, EBITDA, "
    "margins, YoY, and segment / geography splits (shipbuilding vs repair, "
    "defence vs commercial, etc.).\n"
    "4. ORDER BOOK / PIPELINE / OPERATIONS — backlog, new orders, "
    "deliveries, capacity, dry-dock / facility status, execution commentary.\n"
    "5. FORWARD OUTLOOK, MD&A & RISKS — demand tone plus named risks, "
    "litigation, contingencies, commodity / forex exposure, strategy "
    "(horizon plans, capex programmes).\n"
    "6. CREDIT / BALANCE-SHEET — rating action + rationale, cash, gearing, "
    "interest cover, working-capital notes if present.\n\n"
    "Rules:\n"
    "- Every concrete figure MUST carry an inline [source: <exact URL from "
    "SOURCES>]. Do not replace that URL with a prose label (Annual report, "
    "Q1 call, CRISIL). Table cells that are numbers need the tag on the "
    "same row.\n"
    "- Use ONLY facts present in the sources. NEVER invent a target, number, "
    "period, or a direction the filings don't support — an unsupported signal "
    "MUST be DIRECTION=INSUFFICIENT_DATA, conf=0.0.\n"
    "- Confidence reflects EVIDENCE STRENGTH: an explicit quantified statement "
    "with a clean prior-period compare = high (0.7-0.9); a qualitative-only or "
    "single-period read = low/medium (0.2-0.5).\n"
    '- If the sources contain no explicit guidance/target, say exactly: "No '
    'explicit management guidance/target found in the available filings." Do '
    "not manufacture one (the signal card is then all INSUFFICIENT_DATA). "
    "Still report actuals / risks / order book if those are present.\n"
    "- Omit a section only if truly absent. Most material first. Keep the "
    "whole briefing under ~8000 chars — prefer keeping a fact over cutting "
    "it for brevity.\n"
)

# MAP step: pull the important facts out of ONE excerpt of a big filing.  Kept
# deliberately extractive (verbatim numbers, no synthesis) so nothing material
# is lost before the REDUCE consolidation.
_REPORT_MAP_PROMPT = (
    "You are extracting facts from PART {idx} of {total} of the filing "
    '"{title}" ({kind}) for {name} ({ticker}). Source URL: {url}\n\n'
    "From THIS EXCERPT ONLY, extract verbatim every material fact a "
    "prediction desk would use. That includes — and is not limited to — "
    "all of the following when present:\n"
    "- management guidance / forward target (number AND period)\n"
    "- prior target vs actual delivery (beat / miss / in-line)\n"
    "- reported actuals: revenue, PAT, EBITDA, margins, YoY, segment splits\n"
    "- order book / backlog / new orders / pipeline / book-to-bill\n"
    "- operations: deliveries, capacity, dry-dock / facility status, capex\n"
    "- MD&A / strategy / multi-year plans\n"
    "- material risks, litigation, contingencies, impairments\n"
    "- credit-rating action + rationale; cash, gearing, interest cover\n"
    "Keep concrete numbers, periods and segment names.\n"
    "- Reply with terse bullet points; preserve figures exactly.\n"
    "- End every bullet that contains a number with [source: {url}].\n"
    "- Reply NONE only if the excerpt is truly empty of substance "
    "(cover letter, contents page, signature block, page numbers). "
    "Accounting notes, risk factors, related-party, and segment tables "
    "ARE substance — extract them.\n"
    "- Do NOT invent anything not present in the excerpt.\n\n"
    "EXCERPT:\n{chunk}"
)


# A crawl returning fewer than this many chars is treated as a failed/gated
# fetch (e.g. a PDF cover stub), not a real document — skipped so it doesn't
# waste a doc slot or dilute the summary.
_REPORT_MIN_DOC_CHARS = 400


def _report_max_docs() -> int:
    """Cap the number of filing PDFs crawled + summarised per call (latency /
    LLM-cost bound).  Override with ``TRADINGCREW_REPORT_MAX_DOCS``."""
    try:
        return max(1, int(os.environ.get("TRADINGCREW_REPORT_MAX_DOCS", "4")))
    except ValueError:
        return 4


def _report_context_chunk_chars(tokens: int) -> int:
    """Map-chunk size so the excerpt plus prompt and completion still fit.

    Assumes ~4 chars/token. Reserve is prompt + completion headroom
    (``min(24000, tokens // 6)``). Set ``REPORT_PROCESS_CONTEXT_TOKENS`` to
    the model's window (128000 or 256000).
    """
    if tokens < 8000:
        raise ValueError(
            f"REPORT_PROCESS_CONTEXT_TOKENS={tokens} is below 8000"
        )
    reserve = min(24000, max(4000, tokens // 6))
    return max(8000, (tokens - reserve) * 4)


def _report_doc_chars() -> int:
    """Chars per LLM map pass. A filing larger than this is chunked, not
    truncated. ``TRADINGCREW_REPORT_DOC_CHARS`` is an explicit size override
    (tests). Otherwise ``REPORT_PROCESS_CONTEXT_TOKENS`` sizes the chunk
    from the report-process window. If neither is set, 20000."""
    raw_chars = (os.environ.get("TRADINGCREW_REPORT_DOC_CHARS") or "").strip()
    if raw_chars:
        return max(2000, int(raw_chars))
    raw_tok = (os.environ.get("REPORT_PROCESS_CONTEXT_TOKENS") or "").strip()
    if raw_tok:
        return _report_context_chunk_chars(int(raw_tok))
    return 20000


def _report_max_per_kind() -> int:
    """Cap how many filings of the SAME kind (e.g. concall) may be selected, so
    a name with dozens of concalls doesn't spend the whole per-call doc budget
    on one kind (and misses the annual report / credit rating).  The budget is
    then spread across kinds and back-filled if slots remain.  Override with
    ``TRADINGCREW_REPORT_MAX_PER_KIND`` (default 3)."""
    try:
        return max(1, int(os.environ.get("TRADINGCREW_REPORT_MAX_PER_KIND", "3")))
    except ValueError:
        return 3


def _report_max_annual_reports() -> int:
    """Newest annual report is enough — prior years already sit in its
    comparatives / MD&A. Override with ``TRADINGCREW_REPORT_MAX_ANNUAL_REPORTS``
    (default 1). Do not set ``MAX_PER_KIND=1`` globally; that would also
    collapse concall vintages."""
    try:
        return max(1, int(os.environ.get("TRADINGCREW_REPORT_MAX_ANNUAL_REPORTS", "1")))
    except ValueError:
        return 1


def _report_max_for_kind(kind: str) -> int:
    if kind == "annual_report":
        return _report_max_annual_reports()
    return _report_max_per_kind()


def _chunk_report_text(text: str, size: int) -> list:
    """Split ``text`` into ``size``-char chunks. Every chunk is kept."""
    if len(text) <= size:
        return [text]
    return [text[i : i + size] for i in range(0, len(text), size)]


def _select_report_documents(documents: list, max_docs: int) -> list:
    """Pick up to ``max_docs`` guidance-rich filings, DE-DUPED by URL and
    DIVERSIFIED across kinds, keeping the source's (reverse-chronological) order
    within each kind so the newest filing of each kind is preferred.

    Two passes:
      1. take up to ``_report_max_for_kind()`` of each kind in priority order
         (concall → earnings → annual → quarterly → credit → announcement), so
         the budget is spread across kinds instead of being consumed by a name
         with dozens of near-identical concalls. Annual reports default to 1
         (the newest); concalls keep the shared per-kind cap.
      2. if slots remain, back-fill with leftover filings in the same
         priority order. ``_try_add`` still enforces the per-kind cap, so
         older annual reports cannot consume leftover slots.

    De-duping is by URL (a screener feed routinely lists the SAME concall PDF
    under several dated titles — without this the doc budget is wasted on
    repeats and the annual report / credit rating never gets read)."""
    selected: list = []
    seen_urls: set = set()

    def _try_add(d) -> bool:
        url = (getattr(d, "url", "") or "").strip()
        if not url or url in seen_urls:
            return False
        kind = getattr(d, "kind", "") or ""
        already = sum(1 for s in selected if getattr(s, "kind", "") == kind)
        if already >= _report_max_for_kind(kind):
            return False
        seen_urls.add(url)
        selected.append(d)
        return True

    # Pass 1 — diversify: cap per kind.
    for kind in _REPORT_DOC_PRIORITY:
        taken = 0
        cap = _report_max_for_kind(kind)
        for d in documents:
            if d.kind != kind:
                continue
            if taken >= cap:
                break
            if _try_add(d):
                taken += 1
                if len(selected) >= max_docs:
                    return selected
    # Pass 2 — back-fill remaining budget with leftovers (still priority order).
    for kind in _REPORT_DOC_PRIORITY:
        for d in documents:
            if d.kind == kind and _try_add(d):
                if len(selected) >= max_docs:
                    return selected
    return selected


# A doc shorter than this is fed to the REDUCE step verbatim; anything larger
# is condensed by the MAP step first.  This is deliberately DECOUPLED from the
# (much larger) chunk size: with a big chunk a typical filing is a single map
# call, but if we let a whole 250k-char filing through raw, several such docs
# would blow past the LLM context window in the reduce prompt.  Keeping the
# raw-passthrough small bounds the reduce input regardless of chunk size.
_REPORT_RAW_PASSTHROUGH_CHARS = 8000


def _map_reduce_report_doc(llm, name: str, ticker: str, doc: dict) -> str:
    """Condense ONE (possibly huge) filing into a compact extract without a
    head-only truncation.

    A small filing is returned as-is (the reduce step reads it whole).  A larger
    filing is chunked and each chunk is summarised extractively (map); the
    concatenated per-chunk extracts become the doc's condensed representation,
    so the whole document — not just its head — feeds the final consolidation.
    With a large chunk size a typical filing is a single map call, yet the
    reduce input stays bounded because only SMALL docs bypass the map step.
    """
    content = doc["content"]
    chunk_size = _report_doc_chars()
    if len(content) <= _REPORT_RAW_PASSTHROUGH_CHARS:
        return content
    chunks = _chunk_report_text(content, chunk_size)
    logger.info(
        "get_report_insights: mapping %d chunks (%d chars) for %s",
        len(chunks), len(content), doc.get("url"),
    )
    notes: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        prompt = _REPORT_MAP_PROMPT.format(
            idx=i, total=len(chunks), title=doc["title"], kind=doc["kind"],
            name=name, ticker=ticker, chunk=chunk,
            url=doc.get("url") or "",
        )
        try:
            raw = (llm.call(messages=[{"role": "user", "content": prompt}]) or "").strip()
        except Exception as exc:  # noqa: BLE001 — a bad chunk must not kill the doc
            logger.warning(
                "get_report_insights: map chunk %d/%d failed for %s: %s",
                i, len(chunks), doc["url"], exc,
            )
            continue
        if raw and raw.strip().upper() != "NONE":
            notes.append(raw)
    # If every chunk came back empty/NONE (or all map calls failed), fall back
    # to the head of the document so the reduce still has raw material to work
    # with rather than nothing (bounded so the reduce prompt stays in-context).
    return "\n".join(notes) if notes else content[:_REPORT_RAW_PASSTHROUGH_CHARS]


def _summarize_report_docs(name: str, ticker: str, crawled: list) -> str:
    """Consolidate the crawled filing texts into a guidance / target-vs-actual
    digest via map-reduce: each big PDF is chunk-summarised (map) so nothing in
    its tail is lost, then all condensed docs are consolidated into the
    briefing (guidance, actuals, order book, risks/MD&A, credit).  PIT-aware:
    when a cutoff is active the model is told to ignore anything dated after it.

    Uses ``REPORT_PROCESS_*`` (not the desk preset / ``VLLM_LLM_*``). A
    crawled item may already carry ``condensed`` from a prior unchanged
    hash — those skip the map step.
    """
    from trading_crew._common import get_report_process_llm

    try:
        llm = get_report_process_llm(temperature=0.2)
    except Exception as exc:  # noqa: BLE001
        logger.warning("get_report_insights: report-process LLM unavailable: %s", exc)
        return ""

    blocks = []
    for d in crawled:
        condensed = (d.get("condensed") or "").strip()
        if not condensed:
            condensed = _map_reduce_report_doc(llm, name, ticker, d)
            d["condensed"] = condensed
        blocks.append(
            f"URL: {d['url']}\nDOC TYPE: {d['kind']}\nTITLE: {d['title']}\n"
            f"CONTENT:\n{condensed}"
        )
    prompt = (
        _REPORT_INSIGHT_PROMPT.format(name=name, ticker=ticker)
        + "\nSOURCES:\n"
        + "\n\n---\n\n".join(blocks)
    )
    try:
        from trading_crew.data_sources import pit

        cutoff = pit.cutoff()
    except Exception:  # noqa: BLE001
        cutoff = None
    if cutoff is not None:
        as_of = cutoff.strftime("%Y-%m-%d")
        prompt += (
            f"\n\nPOINT-IN-TIME CONSTRAINT: Treat 'today' as {as_of}. IGNORE "
            f"and do NOT report anything announced, published, or dated after "
            f"{as_of}, even if it appears in the sources."
        )
    try:
        raw = llm.call(messages=[{"role": "user", "content": prompt}])
    except Exception as exc:  # noqa: BLE001
        logger.warning("get_report_insights: reduce LLM call failed: %s", exc)
        return ""
    return (raw or "").strip()


def _computed_trend_block(ticker: str, screener_quarters) -> str:
    """Render the deterministic financial-trend signal block for ``ticker``.

    Returns an empty string when nothing decisive can be computed OR when a
    point-in-time backfill is active — the computed trends read CURRENT
    quarterly numbers (screener/yfinance today), which are not point-in-time,
    so including them in a historical decision would leak the future.
    """
    try:
        from trading_crew.data_sources import pit as _pit

        if _pit.is_backfill():
            return ""
    except Exception:  # noqa: BLE001 — absence of PIT context = live mode
        pass
    try:
        from trading_crew import report_signals as _rs

        signals = _rs.compute_financial_trend_signals(ticker, screener_quarters)
        return _rs.render_computed_block(signals)
    except Exception as exc:  # noqa: BLE001 — deterministic block is best-effort
        logger.info("get_report_insights: computed trend block failed for %s: %s", ticker, exc)
        return ""


def _report_insights_cache_enabled() -> bool:
    raw = os.environ.get("TRADINGCREW_REPORT_INSIGHTS_CACHE", "1")
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


# First ~3 PDF pages of extracted text — enough to detect a replaced
# file at the same URL without paying for a full map-reduce.
_REPORT_HEAD_HASH_CHARS = 12000


def _report_insights_snapshot_path(ticker: str) -> Path:
    root = Path(os.environ.get("TRADINGCREW_CACHE_DIR") or os.path.expanduser("~/.trading_crew"))
    suffix = ""
    try:
        from trading_crew.data_sources import pit

        if pit.is_backfill():
            cutoff = pit.cutoff()
            if cutoff is not None:
                suffix = "_" + cutoff.strftime("%Y-%m-%d")
    except Exception:  # noqa: BLE001 — live path has no PIT cutoff
        pass
    return root / "report_insights" / f"{ticker}{suffix}.json"


def _report_catalog_entries(selected: list, resolve_url) -> list[dict]:
    """Listing metadata for the selected filings (url + kind + title).

    Screener embeds the filing date in the title (``24 Jun 2024`` or
    ``15h``); EDGAR titles carry ``filed YYYY-MM-DD``. The raw title is
    the website metadata — we do not rewrite relative dates to ``now``.
    """
    entries: list[dict] = []
    for d in selected:
        entries.append(
            {
                "url": resolve_url(d.url),
                "kind": d.kind,
                "title": d.title,
            }
        )
    return entries


def _report_catalog_fingerprint(entries: list[dict]) -> str:
    payload = [
        {"url": e.get("url") or "", "kind": e.get("kind") or "", "title": e.get("title") or ""}
        for e in entries
    ]
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _report_head_sha256(text: str) -> str:
    head = (text or "")[:_REPORT_HEAD_HASH_CHARS]
    if not head.strip():
        return ""
    return hashlib.sha256(head.encode("utf-8", errors="replace")).hexdigest()


def _report_same_head_hashes(entries: list[dict], prev_docs: list) -> bool:
    prev = {
        (d.get("url") or ""): (d.get("head_sha256") or "")
        for d in prev_docs
        if isinstance(d, dict)
    }
    now_urls = {e.get("url") or "" for e in entries}
    if not now_urls or now_urls != set(prev):
        return False
    for e in entries:
        sha = e.get("head_sha256") or ""
        old = prev.get(e.get("url") or "") or ""
        if not sha or not old or sha != old:
            return False
    return True


def _load_report_insights_snapshot(ticker: str) -> Optional[dict]:
    if not _report_insights_cache_enabled():
        return None
    path = _report_insights_snapshot_path(ticker)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("get_report_insights: snapshot read failed %s: %s", path, exc)
        return None
    if not isinstance(data, dict) or not (data.get("llm_body") or "").strip():
        return None
    return data


def _store_report_insights_snapshot(ticker: str, payload: dict) -> None:
    if not _report_insights_cache_enabled():
        return
    if not (payload.get("llm_body") or "").strip():
        return
    path = _report_insights_snapshot_path(ticker)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("get_report_insights: snapshot write failed %s: %s", path, exc)


def _render_report_insights(
    name: str,
    canonical: str,
    llm_body: str,
    filings: list[dict],
    computed_block: str,
    source_tag: str,
    how: str,
) -> str:
    read_list = "\n".join(
        f"- [{d['kind']}] {d['title']} — {d['url']}" for d in filings
    )
    tagged_body = _retag_numeric_claims(
        llm_body,
        default_source=f"get_report_insights · {canonical}",
        citations=filings,
    )
    parts = [
        f"# Management guidance & filings — {name} ({canonical})",
        "",
        tagged_body,
    ]
    if computed_block:
        parts.extend(["", computed_block])
    parts.extend(
        [
            "",
            "## Filings read (LLM-summarised primary sources)",
            read_list,
        ]
    )
    return "\n".join(parts) + _source_line(
        f"{source_tag} ({how}, {len(filings)} filings) · {canonical}"
    )


def _fetch_one_report_doc(
    index: int,
    n_selected: int,
    kind: str,
    crawl_url: str,
    fetch_doc,
) -> tuple[int, Optional[str]]:
    """Download + extract one filing. Safe to run on a worker thread."""
    t0 = time.monotonic()
    logger.info(
        "get_report_insights: crawling %d/%d [%s] %s",
        index, n_selected, kind, crawl_url,
    )
    try:
        content = fetch_doc(crawl_url)
    except Exception as exc:  # noqa: BLE001
        logger.warning("get_report_insights: crawl failed for %s: %s", crawl_url, exc)
        content = None
    logger.info(
        "get_report_insights: crawled %d/%d [%s] %d chars in %.1fs",
        index, n_selected, kind, len((content or "").strip()), time.monotonic() - t0,
    )
    return index, content


def _fetch_report_docs_parallel(selected, catalog, fetch_doc) -> list[tuple]:
    """Extract every selected filing at once (annual reports are the slow ones).

    Same HTTP + pdfplumber path as the sequential loop — only the wall-clock
    overlaps. Returned rows stay in ``selected`` order.
    """
    n = len(selected)
    if n == 0:
        return []
    logger.info("get_report_insights: extracting %d filings in parallel", n)
    slots: list[Optional[str]] = [None] * n
    with ThreadPoolExecutor(max_workers=n) as pool:
        futs = [
            pool.submit(
                _fetch_one_report_doc, i, n, d.kind, entry["url"], fetch_doc,
            )
            for i, (d, entry) in enumerate(zip(selected, catalog), 1)
        ]
        for fut in as_completed(futs):
            index, content = fut.result()
            slots[index - 1] = content
    return [
        (d, entry, entry["url"], slots[i])
        for i, (d, entry) in enumerate(zip(selected, catalog))
    ]


@tool("get_report_insights")
@_memoize_tool
def get_report_insights(ticker: str) -> str:
    """Read the company's OWN primary filings — concall / earnings-call
    transcripts, investor presentations, annual reports and credit-rating
    rationales — and extract MANAGEMENT GUIDANCE, TARGET-vs-ACTUAL delivery,
    reported actuals, order book / operations, MD&A risks, and credit.

    The digest LEADS with a Firefly-style **REPORT TREND SIGNALS** card —
    directional, confidence-scored signals derived only from the filings and
    (where the sources allow) compared latest-vs-prior period so they read as
    TRENDS, not snapshots: OUTLOOK_TONE (bullish/cautious/defensive),
    REVENUE_GUIDANCE (raised/maintained/cut), MARGIN_GUIDANCE, DELIVERY_TRACK_
    RECORD (beat/in-line/miss), DEMAND_MOMENTUM, CREDIT_TRAJECTORY, and a
    synthesised NET_GUIDANCE_BIAS (positive/neutral/negative). Each signal is a
    single line: ``SIGNAL | NAME | DIRECTION | conf=0.00-1.00 | rationale
    [source: url]`` — the desk's headline, actionable read.

    Why this exists
    ---------------
    The fundamentals desk reads restated ratio tables; nothing else in the
    workflow reads the primary filings where management commits to explicit
    forward targets (revenue / margin / order-book / capex guidance) and where
    a prior target can be checked against what was actually delivered.  That is
    exactly the "target / target-vs-actual" signal missing from predictions.

    How it works
    ------------
    Market-aware.  For Indian (NSE/BSE) names the dated primary-filing feed is
    screener.in's ``#documents`` list (concalls, annual reports, credit
    ratings).  For US (SEC-registered) names it is **SEC EDGAR** — the official,
    free, keyless submissions feed: 10-K / 10-Q annual & quarterly reports
    (MD&A / outlook) and earnings 8-Ks (item 2.02 — results + guidance press
    releases, whose numbers live in the Exhibit 99.1 we resolve to before
    crawling).      Either way the tool selects the most guidance-rich recent
    filings (newest annual report only — prior years are already inside it;
    multiple concalls for guidance vintages), downloads those PDFs over HTTP
    in parallel (no Firecrawl), extracts text and tables from every page of
    each filing, and LLM-summarises the WHOLE document in context-sized
    chunks (map-reduce) so broader briefing (actuals, order book, risks)
    is kept, not dropped.  Point-in-time-safe:
    filings are date-filtered to on/before the decision cutoff and the
    summariser is told to ignore anything dated after it.

    Reprocessing is change-gated, not day-gated: the listing metadata
    (url + title + kind, which carry the website filing date) is compared
    to the last snapshot. Unchanged listing → reuse the digest. If titles
    drift but the URL set is the same, the first pages are hashed and
    compared; only changed filings are mapped again.     The map-reduce LLM
    is ``REPORT_PROCESS_*`` from the environment.

    For names in neither feed (e.g. a London / HK listing) the tool returns a
    short not-available note and the desk leans on get_forward_estimates /
    get_analyst_recommendations.
    """
    canonical = resolve_ticker(ticker)

    # Route to the market's primary-filing feed.  ``docs`` is a uniform list of
    # DocumentLink; ``resolve_url`` maps a link to the URL to actually crawl
    # (identity for screener; earnings-8-K -> Ex-99.1 press release for EDGAR);
    # ``source_tag`` names the provenance on the final source line.  ``name`` is
    # taken from whichever feed we land on (so the no-feed path stays offline).
    resolve_url = lambda u: u  # noqa: E731 — trivial default (screener path)
    # Deterministic financial-trend signals are computed from hard quarterly
    # numbers and appended to the digest.  For Indian names we reuse the
    # screener ``#quarters`` table fetched here (no extra request); otherwise
    # the computer reads yfinance.  ``screener_quarters`` stays None off the
    # screener path so the computer falls through to yfinance.
    screener_quarters = None
    if _screener_available_for(canonical):
        try:
            page = screener_source.fetch_company(canonical)
        except (ScreenerRateLimitError, ScreenerNotFoundError, ScreenerError) as exc:
            return (
                f"Report insights unavailable — screener.in fetch failed for "
                f"{canonical}: {exc}."
                + _source_line(f"Report insights (screener fetch failed, {canonical})")
            )
        name = get_market_profile(canonical).long_name or canonical
        # PIT-filter (drops filings published after the cutoff in backfill).
        docs = screener_source.pit_filter_documents(page.documents)
        screener_quarters = page.quarters
        source_kinds = "concalls / annual reports / credit ratings"
        source_tag = "screener.in #documents"
    elif sec_edgar_source.is_edgar_ticker(canonical):
        try:
            company = sec_edgar_source.fetch_filings(canonical)
        except (EdgarRateLimitError, EdgarNotFoundError, EdgarError) as exc:
            return (
                f"Report insights unavailable — SEC EDGAR fetch failed for "
                f"{canonical}: {exc}."
                + _source_line(f"Report insights (EDGAR fetch failed, {canonical})")
            )
        company = sec_edgar_source.pit_filter_documents(company)
        name = company.name or canonical
        docs = sec_edgar_source.to_document_links(company)
        resolve_url = sec_edgar_source.resolve_text_url
        source_kinds = "10-K / 10-Q reports & earnings 8-Ks"
        source_tag = "SEC EDGAR"
    else:
        # No primary-filing feed — but the deterministic financial-trend
        # signals only need quarterly numbers (yfinance), so surface those
        # rather than a bare not-available note.
        computed_block = _computed_trend_block(canonical, None)
        note = (
            f"Report insights read a company's OWN primary filings — sourced from "
            f"screener.in for Indian (NSE/BSE) names and SEC EDGAR for US names. "
            f"{canonical} is in neither feed, so no primary filings (annual / "
            f"quarterly reports, earnings releases, concalls) are available here "
            f"— rely on get_forward_estimates and get_analyst_recommendations for "
            f"the forward/guidance view."
        )
        if computed_block:
            note = f"{computed_block}\n\n{note}"
        return note + _source_line(
            f"Report insights (no filing feed for {canonical}; computed trends only)"
            if computed_block
            else f"Report insights (n/a — no filing feed for {canonical})"
        )

    # Deterministic financial-trend card (hard numbers) — appended to every
    # remaining return so a US name whose filings won't crawl still ships the
    # revenue/earnings/margin trend read.
    computed_block = _computed_trend_block(canonical, screener_quarters)

    if not docs:
        body = (
            f"No dated primary filings ({source_kinds}) available for {name} "
            f"({canonical}) at this point in time."
        )
        if computed_block:
            body = f"{computed_block}\n\n{body}"
        return body + _source_line(f"Report insights (no filings, {canonical})")

    selected = _select_report_documents(docs, _report_max_docs())
    catalog = _report_catalog_entries(selected, resolve_url)
    catalog_fp = _report_catalog_fingerprint(catalog)
    prev = _load_report_insights_snapshot(canonical)

    if (
        prev
        and prev.get("catalog_fp") == catalog_fp
        and prev.get("filings")
    ):
        logger.info("get_report_insights: catalog-hit for %s (no crawl / LLM)", canonical)
        return _render_report_insights(
            name, canonical, prev["llm_body"], prev["filings"],
            computed_block, source_tag, "catalog-hit, digest reused",
        )

    from trading_crew.data_sources import web_research

    # Screener filings are BSE/NSE PDFs. Local HTTP + pdfplumber, whole
    # document (text + tables on every page). No Firecrawl. SEC EDGAR
    # gates the shared crawler (403), so those docs use its client.
    if source_tag == "SEC EDGAR":
        def _fetch_doc(u: str) -> str:
            return sec_edgar_source.fetch_document_text(u, max_chars=None)
    else:
        def _fetch_doc(u: str) -> str:
            return web_research.crawl(
                u,
                crawler="http",
                max_chars=None,
                extract_tables_pages=None,
                timeout=300,
            )

    crawled: list[dict] = []
    for d, entry, crawl_url, content in _fetch_report_docs_parallel(
        selected, catalog, _fetch_doc,
    ):
        # Skip near-empty crawls (a gated/blocked PDF often returns a ~100-char
        # cover stub) so a junk fetch doesn't consume a doc slot / dilute the
        # summary with nothing.
        if content and len(content.strip()) >= _REPORT_MIN_DOC_CHARS:
            entry["head_sha256"] = _report_head_sha256(content)
            crawled.append(
                {
                    "title": d.title,
                    "url": crawl_url,
                    "kind": d.kind,
                    "content": content,
                    "head_sha256": entry["head_sha256"],
                }
            )
        elif content and content.strip():
            logger.info(
                "get_report_insights: skipping near-empty crawl (%d chars) for %s",
                len(content.strip()), crawl_url,
            )

    if not crawled:
        listed = "\n".join(
            f"- [{d.kind}] {d.title} — {resolve_url(d.url)}" for d in selected
        )
        body = (
            f"Could not extract text from the available filings (the crawler "
            f"returned nothing — the links may be gated or a crawler key unset). "
            f"Filings on record:\n{listed}"
        )
        if computed_block:
            body = f"{computed_block}\n\n{body}"
        return (
            f"# Primary filings — {name} ({canonical})\n\n{body}"
            + _source_line(f"Report insights (crawl empty, {canonical})")
        )

    prev_docs = (prev or {}).get("docs") or []
    prev_by_url = {
        (p.get("url") or ""): p
        for p in prev_docs
        if isinstance(p, dict)
    }
    if prev and (prev.get("llm_body") or "").strip() and _report_same_head_hashes(catalog, prev_docs):
        logger.info("get_report_insights: head-hash-hit for %s (skip LLM)", canonical)
        snapshot = {
            "catalog_fp": catalog_fp,
            "llm_body": prev["llm_body"],
            "filings": [
                {"kind": c["kind"], "title": c["title"], "url": c["url"]}
                for c in crawled
            ],
            "docs": [
                {
                    "url": c["url"],
                    "kind": c["kind"],
                    "title": c["title"],
                    "head_sha256": c["head_sha256"],
                    "condensed": (prev_by_url.get(c["url"]) or {}).get("condensed") or "",
                }
                for c in crawled
            ],
        }
        _store_report_insights_snapshot(canonical, snapshot)
        return _render_report_insights(
            name, canonical, prev["llm_body"], snapshot["filings"],
            computed_block, source_tag, "head-hash-hit, digest reused",
        )

    reused = 0
    for c in crawled:
        prior = prev_by_url.get(c["url"]) or {}
        prior_sha = prior.get("head_sha256") or ""
        prior_condensed = (prior.get("condensed") or "").strip()
        if prior_sha and prior_sha == c.get("head_sha256") and prior_condensed:
            c["condensed"] = prior_condensed
            reused += 1
    if reused:
        logger.info(
            "get_report_insights: reusing %d/%d unchanged filing extracts for %s",
            reused, len(crawled), canonical,
        )

    logger.info(
        "get_report_insights: map-reduce starting for %s (%d crawled docs)",
        canonical, len(crawled),
    )
    summary = _summarize_report_docs(name, canonical, crawled)
    if not summary:
        return _render_report_insights(
            name, canonical,
            "(LLM summarisation was unavailable — see the primary filings listed "
            "below and read them directly.)",
            [{"kind": c["kind"], "title": c["title"], "url": c["url"]} for c in crawled],
            computed_block, source_tag, "crawl, LLM unavailable",
        )

    filings = [{"kind": c["kind"], "title": c["title"], "url": c["url"]} for c in crawled]
    _store_report_insights_snapshot(
        canonical,
        {
            "catalog_fp": catalog_fp,
            "llm_body": summary,
            "filings": filings,
            "docs": [
                {
                    "url": c["url"],
                    "kind": c["kind"],
                    "title": c["title"],
                    "head_sha256": c.get("head_sha256") or "",
                    "condensed": (c.get("condensed") or "").strip(),
                }
                for c in crawled
            ],
        },
    )
    return _render_report_insights(
        name, canonical, summary, filings,
        computed_block, source_tag, "crawl + LLM-summarise",
    )


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

    In a backfill ``AsOfContext`` the current fundamentals snapshot
    (screener.in restated ratios / yfinance ``.info``) is **not**
    point-in-time — it reflects today's restated figures, not what was
    public as of the historical decision date — so it is deliberately
    withheld to avoid leakage.  Dated primary sources (filings /
    announcements via ``get_corporate_actions``) remain available and are
    PIT-filtered.
    """
    from trading_crew.data_sources import pit as _pit

    if _pit.is_backfill():
        return (
            f"Point-in-time backfill: current fundamentals snapshot for {ticker} "
            "withheld (restated figures are not as-of-safe). Rely on dated filings "
            "and price/technical evidence for this historical decision."
        ) + _source_line(f"PIT guard (as_of {_pit.cutoff().date() if _pit.cutoff() else 'n/a'})")
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
    cite = f" [source: yfinance .info {ticker} (fundamentals)]"
    lines = [f"{k}: {info.get(k)}{cite}" for k in keys]
    return "\n".join(lines) + _source_line(
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
        f"Balance sheet for {ticker} (latest period) "
        f"[source: yfinance balance_sheet {ticker} (latest period)]:\n"
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
        f"Cash flow for {ticker} (latest period) "
        f"[source: yfinance cashflow {ticker} (latest period)]:\n"
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
        f"Income statement for {ticker} (latest period) "
        f"[source: yfinance income_stmt {ticker} (latest period)]:\n"
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


def _peer_returns(hist) -> Tuple[Optional[float], Optional[float]]:
    """YTD and ~6-month percent returns from a daily Close series."""
    if hist is None or getattr(hist, "empty", True) or "Close" not in hist:
        return None, None
    close = hist["Close"].dropna()
    if close.empty:
        return None, None
    last = float(close.iloc[-1])
    ytd: Optional[float] = None
    try:
        year = close.index[-1].year
        ytd_close = close[close.index.year == year]
        if len(ytd_close) >= 2:
            ytd = (last / float(ytd_close.iloc[0]) - 1.0) * 100.0
    except Exception:
        ytd = None
    ret_6m: Optional[float] = None
    if len(close) >= 126:
        ret_6m = (last / float(close.iloc[-126]) - 1.0) * 100.0
    return ytd, ret_6m


def _resolve_run_ticker(ticker: str) -> str:
    """Use an explicit ticker, else the kickoff symbol on this run.

    CrewAI marks ``ticker`` required even when the desk only wants the
    default live screen (``{}`` or ``{peers: ""}``). That is a schema
    mismatch, not a missing name — AXISBANK and BHARATFORG both burned
    a tool turn this way. The blackboard / scratch pad already hold the
    run symbol. An empty result is fail-closed: we do not invent a
    ticker.
    """
    written = (ticker or "").strip()
    if written:
        return written
    try:
        from trading_crew.agentic.graph.blackboard import current as bb_current

        bb = bb_current()
        if bb is not None and (bb.ticker or "").strip():
            return bb.ticker.strip()
    except Exception:
        pass
    try:
        from trading_crew.agentic.graph.scratchpad import current as pad_current

        pad = pad_current()
        if pad is not None and (pad.ticker or "").strip():
            return pad.ticker.strip()
    except Exception:
        pass
    return ""


def _fetch_peer_row(sym: str) -> PeerRow:
    t = yf.Ticker(sym)
    try:
        hist = t.history(period="1y")
    except Exception:
        hist = None
    ytd, ret_6m = _peer_returns(hist)
    try:
        info = t.info or {}
    except Exception:
        info = {}
    raw_mcap = info.get("marketCap")
    raw_pe = info.get("trailingPE")
    try:
        mcap = float(raw_mcap) if raw_mcap is not None else None
    except (TypeError, ValueError):
        mcap = None
    try:
        pe = float(raw_pe) if raw_pe is not None else None
    except (TypeError, ValueError):
        pe = None
    return PeerRow(
        symbol=sym,
        ytd_pct=ytd,
        ret_6m_pct=ret_6m,
        market_cap=mcap,
        pe=pe,
        industry=str(info.get("industry") or "").strip(),
        country=str(info.get("country") or "").strip(),
    )


@_memoize_tool
def _get_sector_peers_impl(ticker: str, peers: str = "") -> str:
    canonical = resolve_ticker(ticker)
    profile = get_market_profile(canonical)
    universe = discover_peers(canonical, extra_peers=peers, profile=profile)
    rows = [_fetch_peer_row(sym) for sym in universe.symbols]
    body = format_peer_comparison(canonical, universe, rows)
    return body + _source_line(
        f"yfinance OHLCV + .info live peer screen "
        f"[{', '.join(universe.symbols)}] method={universe.method}"
    )


@tool("get_sector_peers")
def get_sector_peers(ticker: str = "", peers: str = "") -> str:
    """Compare a ticker to a live industry universe (same country).

    Always pass ``ticker`` (this run's symbol). ``peers`` is optional:
    omit it for the live Yahoo industry + country screen. If that set
    is the wrong industry or country, pass ``peers`` as space-separated
    yfinance tickers you already saw in a filing or tool output. Do not
    invent tickers.

    The tool prints median YTD / 6m / PE and focus alpha vs the median.
    Quote those lines — do not recompute them.
    """
    resolved = _resolve_run_ticker(ticker)
    if not resolved:
        return (
            "get_sector_peers needs ticker. Pass ticker=<this run's symbol> "
            "(the kickoff name). No peer screen was run."
        )
    return _get_sector_peers_impl(resolved, peers)


# ---------------------------------------------------------------------------
# Geopolitical Analyst tools
# ---------------------------------------------------------------------------


@tool("get_geopolitical_news")
@_memoize_tool
def get_geopolitical_news(ticker: str, query: str = "") -> str:
    """Retrieve geopolitical / regulatory news for the ticker.

    You write ``query`` (6-12 words) from get_market_context facts +
    your remit (tariffs, export controls, sanctions, chokepoints that
    actually apply to THIS country and industry) + any gap on the
    blackboard. Example: ``Nutanix US China AI hardware export controls``.
    There is no canned industry seed list.
    """
    from trading_crew.market_context import compose_search_query

    canonical = resolve_ticker(ticker)
    profile = get_market_profile(canonical)
    return _web_news(
        compose_search_query(profile, lens="geo", agent_query=query),
        max_results=6,
    )


@tool("get_supply_chain_risk")
@_memoize_tool
def get_supply_chain_risk(ticker: str, query: str = "") -> str:
    """Retrieve supply-chain and customer-concentration news.

    You write ``query`` from the brief + your remit (suppliers, OEMs,
    shortages, concentration). Example:
    ``Nutanix hardware appliance OEM NAND SSD shortage``.
    """
    from trading_crew.market_context import compose_search_query

    canonical = resolve_ticker(ticker)
    profile = get_market_profile(canonical)
    return _web_news(
        compose_search_query(profile, lens="supply", agent_query=query),
        max_results=6,
    )


# ---------------------------------------------------------------------------
# Cross-cutting research brief
# ---------------------------------------------------------------------------


@tool("get_market_context")
@_memoize_tool
def get_market_context(ticker: str) -> str:
    """One-shot FACTS brief for ``ticker`` — country, sector, industry,
    market cap, and country macro-ticker symbols.

    Call this FIRST, then WRITE your own 6-12 word search queries from
    these facts + your desk remit + any gap. The brief does not ship a
    canned Hormuz / TSMC / Fed phrase list, and it does not ship a
    per-ticker peer dictionary — peers come from get_sector_peers.
    """
    canonical = resolve_ticker(ticker)
    profile = get_market_profile(canonical)
    return profile.to_brief() + _source_line(
        f"yfinance .info {canonical}"
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


def _gather_crowd_inputs(ticker: str):
    """Fetch the crowd-positioning inputs for ``ticker`` from yfinance.

    Returns ``(CrowdInputs, notes)``.  Each component is wrapped in its own
    try/except so a missing feed simply omits that signal (honest absence,
    never an invented value).  Shared by the ``get_crowd_positioning`` tool
    and the episode-logging path in the runner.
    """
    from trading_crew.agentic.swarm.crowd import CrowdInputs

    inputs = CrowdInputs()
    notes: list[str] = []

    # --- options: put/call ratio + IV skew ---------------------------------
    try:
        t = yf.Ticker(ticker)
        expiries = t.options
        if expiries:
            expiry = expiries[0]
            chain = t.option_chain(expiry)
            spot = float(t.history(period="5d")["Close"].iloc[-1])
            calls, puts = chain.calls, chain.puts
            call_oi = int(calls["openInterest"].fillna(0).sum())
            put_oi = int(puts["openInterest"].fillna(0).sum())
            if call_oi > 0:
                inputs.put_call_ratio = put_oi / call_oi
                notes.append(f"put/call={inputs.put_call_ratio:.2f}")
            atm_call = calls.iloc[(calls["strike"] - spot).abs().argsort()[:1]]
            atm_put = puts.iloc[(puts["strike"] - spot).abs().argsort()[:1]]
            call_iv = float(atm_call["impliedVolatility"].iloc[0])
            put_iv = float(atm_put["impliedVolatility"].iloc[0])
            inputs.iv_skew = put_iv - call_iv
            notes.append(f"IV skew={inputs.iv_skew:+.2%}")
    except Exception:  # noqa: BLE001 — missing options data is fine, just omit
        pass

    # --- insiders: net buy/sell ratio --------------------------------------
    try:
        t = yf.Ticker(ticker)
        df = getattr(t, "insider_transactions", None)
        if df is not None and len(df) > 0 and "Transaction" in df.columns:
            txt = df["Transaction"].astype(str).str.lower()
            buys = int(txt.str.contains("buy|purchase").sum())
            sells = int(txt.str.contains("sale|sell").sum())
            if buys + sells > 0:
                inputs.insider_net_ratio = (buys - sells) / (buys + sells)
                notes.append(f"insider net={inputs.insider_net_ratio:+.2f} ({buys}B/{sells}S)")
    except Exception:  # noqa: BLE001
        pass

    # --- analysts: bullish fraction ----------------------------------------
    try:
        t = yf.Ticker(ticker)
        rec = t.recommendations
        if rec is not None and len(rec) > 0:
            row = rec.iloc[0]
            sb = float(row.get("strongBuy", 0) or 0)
            b = float(row.get("buy", 0) or 0)
            h = float(row.get("hold", 0) or 0)
            s = float(row.get("sell", 0) or 0)
            ss = float(row.get("strongSell", 0) or 0)
            total = sb + b + h + s + ss
            if total > 0:
                inputs.analyst_bullish_frac = (sb + b) / total
                notes.append(f"bullish analysts={inputs.analyst_bullish_frac:.0%}")
    except Exception:  # noqa: BLE001
        pass

    return inputs, notes


def crowd_signal_for(ticker: str):
    """Return the full :class:`CrowdSignal` for ``ticker`` (or ``None`` when no
    positioning data is available).

    Uses the *learned* crowd weights when a fitted weights file exists (see
    :mod:`trading_crew.agentic.swarm.crowd_fit`), otherwise the module default.
    Used by the runner to stamp episodes with the crowd score **and** its
    per-component breakdown at decision time.
    """
    from trading_crew.agentic.swarm.crowd import compute_crowd_signal
    from trading_crew.agentic.swarm.crowd_fit import load_crowd_config

    inputs, _notes = _gather_crowd_inputs(ticker)
    signal = compute_crowd_signal(inputs, load_crowd_config())
    return signal if signal.n_components > 0 else None


def crowd_score_for(ticker: str):
    """Return the fused crowd-positioning score for ``ticker`` (or ``None``
    when no positioning data is available).  Thin wrapper used by the
    runner to stamp episodes with the crowd regime at decision time."""
    signal = crowd_signal_for(ticker)
    return signal.score if signal is not None else None


@tool("get_crowd_positioning")
@_memoize_tool
def get_crowd_positioning(ticker: str) -> str:
    """Swarm-intelligence crowd-positioning read for a ticker.

    Fuses three positioning signals into a single **crowding score** in
    ``[-1, +1]`` (``+`` = the crowd is positioned *long / bullish*, ``-``
    = *short / bearish*):

    - Options put/call ratio + IV skew (downside-demand / fear).
    - Net insider buying vs selling.
    - The sell-side analyst rating distribution (buy-side herding).

    Read it against your trade idea: siding *with* an already-crowded
    consensus means a thinner marginal edge and higher unwind risk;
    going *against* an extreme crowd is the contrarian probability-edge
    play.  Only the components that have real data are used — missing
    signals are omitted, never invented.
    """
    from trading_crew.agentic.swarm.crowd import compute_crowd_signal

    inputs, notes = _gather_crowd_inputs(ticker)
    signal = compute_crowd_signal(inputs)
    src = _source_line(f"get_crowd_positioning({ticker}) yfinance options/insider/analyst")
    if signal.n_components == 0:
        return (
            f"Crowd positioning for {ticker}: no positioning data available "
            f"(no options / insider / analyst coverage)." + src
        )
    comp_lines = "\n".join(
        f"  - {name}: {val:+.2f}" for name, val in sorted(signal.components.items())
    )
    return (
        f"Crowd positioning for {ticker}:\n"
        f"- Crowding score: {signal.score:+.2f} ({signal.label}, "
        f"intensity {signal.intensity:.2f})\n"
        f"- Inputs: {', '.join(notes)}\n"
        f"- Component contributions ({signal.n_components} signals):\n{comp_lines}\n"
        f"- Reading: a score near +1 = crowd is heavily long (thin edge for new longs); "
        f"near -1 = crowd is heavily short (contrarian long may have edge)."
        + src
    )


@tool("research_web")
@_memoize_tool
def research_web(query: str) -> str:
    """Deep open-web research: search → crawl → LLM-summarize.

    Unlike ``get_news`` (which returns ranked headline snippets), this runs
    a full research pipeline for an arbitrary ``query``:

    1. searches the open web: Vertex Gemini + Google Search first when
       ``GEMINI_WEB_SEARCH=1``, otherwise (or on Gemini miss/error) the
       SearXNG ring (websearch.sh / yahoo / optional google-headless),
    2. crawls each page's main content (Firecrawl scrape → raw markdown),
    3. consolidates the raw text into a concise, source-cited briefing via
       the workflow's own LLM.

    Use it when you need the *substance* of what sources say (e.g. "why did
    <company> guidance disappoint in Q2", "read-through of the RBI policy
    for auto lenders") rather than just a list of headlines.  Every claim
    carries an inline ``[source: <URL>]`` tag; missing data is stated
    plainly rather than invented.
    """
    from trading_crew.data_sources import pit, web_research

    # PIT guard: crawled live web pages carry no reliable publish date and
    # routinely contain post-decision information, so this pipeline is
    # unsafe during a historical backfill.  Fail closed rather than leak.
    if pit.is_backfill():
        return (
            f"Web research for {query!r}: skipped in point-in-time / backfill "
            f"mode (crawled pages are not date-bounded). Use the dated news "
            f"providers instead."
        )

    result = web_research.research_and_summarize(query, max_links=5)
    src = _source_line(f"research_web({query!r}) search+crawl+summarize")
    if not result.get("summary"):
        status = result.get("status", "")
        if status == "no_search_results":
            reason = (
                "the search backends returned nothing (likely rate-limited "
                "or offline) — this is a search-infrastructure gap, not an "
                "absence of news"
            )
        elif status == "crawl_blocked":
            reason = (
                "links were found but every page blocked the crawler, so no "
                "content could be read"
            )
        else:
            reason = "search or crawl returned nothing"
        logger.warning("research_web empty for %r (status=%s)", query, status or "unknown")
        return f"Web research for {query!r}: no usable sources — {reason}." + src
    sources = "\n".join(
        f"  - {s.get('title') or s.get('url')}: {s.get('url')}"
        for s in result.get("sources", [])
    )
    citations = [
        {"url": (s.get("url") or "").strip(), "kind": "", "title": s.get("title") or ""}
        for s in (result.get("sources") or [])
        if (s.get("url") or "").strip()
    ]
    briefing = _retag_numeric_claims(
        result["summary"],
        default_source=f"research_web({query!r}) search+crawl+summarize",
        citations=citations,
    )
    return (
        f"Web research briefing for {query!r} "
        f"({result.get('n_crawled', 0)} sources crawled):\n\n"
        f"{briefing}\n\nSources:\n{sources}" + src
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


@tool("lookup_blackboard")
def lookup_blackboard(ticker: str = "") -> str:
    """Read a COMPACT evidence board: each desk's lean, a short TLDR, and
    any filled evidence-request addenda.

    Full analyst reports are NOT dumped here — use read_scratchpad for the
    shared debate history.  Call this when you need a lean snapshot of
    desk leans / addenda.  ``ticker`` is ignored; the board is scoped to
    the current kickoff.
    """
    from trading_crew.agentic.graph.blackboard import current

    bb = current()
    if bb is None:
        return (
            "No evidence blackboard is active for this run. Call "
            "read_scratchpad for the shared debate pad."
        )
    return bb.render_compact()


@tool("read_scratchpad")
def read_scratchpad(ticker: str = "") -> str:
    """Read the shared bull/bear scratch pad for this run.

    Contains compact analyst TLDRs, prior debate turns, and any live
    notes the other side wrote.  Call this ONCE at the start of your
    turn instead of waiting for a dumped discussion in context.
    ``ticker`` is ignored; the pad is scoped to the current kickoff.
    """
    from trading_crew.agentic.graph.scratchpad import current

    pad = current()
    if pad is None:
        return (
            "No debate scratch pad is active for this run. Argue from "
            "tools you call this turn."
        )
    return pad.render()


@tool("write_scratchpad")
def write_scratchpad(note: str, author: str = "") -> str:
    """Append a compact claim or tool finding to the shared scratch pad.

    Use this for NEW evidence you just fetched (web search, news, event
    proximity) so the other side can rebut it without a full redump.
    Keep it to a few bullets.  Your finished 4-6-bullet turn is copied
    onto the pad automatically when the task completes — do not paste
    that whole turn here.

    ``author`` should be ``bull`` or ``bear`` (whichever side you are).
    """
    from trading_crew.agentic.graph.scratchpad import current

    pad = current()
    if pad is None:
        return "Rejected: no debate scratch pad is active."
    body = (note or "").strip()
    if not body:
        return "Rejected: empty note."
    pad.write_note(author or "debate", body)
    return f"Scratch pad updated ({len(body)} chars). Opponent will see this on read_scratchpad."


@tool("request_evidence")
def request_evidence(target_desk: str, question: str, tool_name: str = "") -> str:
    """Queue a bounded re-query of ONE analyst desk's tool.

    Use when a claim in the debate hinges on a number or catalyst that
    is missing or stale on the blackboard.  ``target_desk`` must be an
    analyst key (market_analyst, news_analyst, fundamentals_analyst,
    reports_analyst, macro_analyst).  ``tool_name`` must be owned by
    that desk; omit it to use the desk's primary tool.

    Hard cap: two requests per run.  The owning desk is re-queried
    (tool only, no extra LLM) before the next debate round — then call
    lookup_blackboard or read_scratchpad to read the addendum.  Do NOT
    use this to start a new research saga.
    """
    from trading_crew.agentic.graph.blackboard import current

    bb = current()
    if bb is None:
        return (
            "Rejected: no evidence blackboard is active. Cite the analyst "
            "reports you already have."
        )
    return bb.queue_request(target_desk, question, tool_name)


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
    "get_report_insights": get_report_insights,
    "get_long_term_thesis": get_long_term_thesis,
    "get_macro_data": get_macro_data,
    "get_sector_peers": get_sector_peers,
    "get_geopolitical_news": get_geopolitical_news,
    "get_supply_chain_risk": get_supply_chain_risk,
    "get_options_summary": get_options_summary,
    "get_analyst_recommendations": get_analyst_recommendations,
    "get_crowd_positioning": get_crowd_positioning,
    "research_web": research_web,
    "get_forward_estimates": get_forward_estimates,
    "get_event_proximity": get_event_proximity,
    "backtest_setup": backtest_setup,
    "lookup_blackboard": lookup_blackboard,
    "read_scratchpad": read_scratchpad,
    "write_scratchpad": write_scratchpad,
    "request_evidence": request_evidence,
}

# Default tool ownership per agent (mirrors crew.py wiring).
# Past-episode retrieval is not a live desk tool. Resolve investigates
# the miss and Hermes applies the next experiment.
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
        # Swarm-intelligence crowd/positioning read (options + insider +
        # analyst crowding) so the desk can judge whether an idea sides
        # with an over-crowded consensus or fades it.
        "get_crowd_positioning",
    ],
    # News desk = catalysts + corporate actions + sentiment + insider
    # (was news + social).
    "news_analyst": [
        "get_market_context", "get_news", "get_global_news",
        "get_corporate_actions", "get_insider_transactions",
        # Deep search→crawl→summarize research for catalysts the headline
        # snippet tools miss (auto-skips in point-in-time/backfill mode).
        "research_web",
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
    # Reports desk = the company's OWN primary filings (concall transcripts,
    # investor decks, annual-report MD&A, credit-rating rationales) crawled +
    # LLM-summarised into a management-guidance / target-vs-actual digest.
    # Market-aware: screener.in #documents for Indian names, SEC EDGAR
    # (10-K / 10-Q + earnings 8-Ks) for US names; names in neither feed get a
    # not-available note so the desk defers to the forward-estimates read.
    "reports_analyst": [
        "get_report_insights",
        # Country/industry brief so the desk frames guidance in context.
        "get_market_context",
        # Forward-consensus counterweight (works everywhere, including names
        # with no primary-filing feed where get_report_insights returns n/a).
        "get_forward_estimates",
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
    "trader": ["get_event_proximity"],
    "risk_aggressive": [
        "read_scratchpad", "write_scratchpad",
        "research_web", "get_news", "get_event_proximity",
        "get_stock_data", "get_fundamentals",
        "get_options_summary",
    ],
    "risk_conservative": [
        "read_scratchpad", "write_scratchpad",
        "research_web", "get_news", "get_event_proximity",
        "get_stock_data", "get_fundamentals",
        "get_options_summary",
    ],
    "risk_neutral": [
        "read_scratchpad", "write_scratchpad",
        "research_web", "get_news", "get_event_proximity",
        "get_stock_data", "get_fundamentals",
        "get_options_summary",
    ],
    # Debate desks share a compact scratch pad (no full-report redump) and
    # may fetch NEW evidence with a dynamic query, plus queue at most two
    # request_evidence re-queries (hard-capped).
    "bull_researcher": [
        "read_scratchpad", "write_scratchpad",
        "research_web", "get_news", "get_event_proximity",
        "get_stock_data", "get_fundamentals",
        "get_analyst_recommendations",
        "request_evidence",
    ],
    "bear_researcher": [
        "read_scratchpad", "write_scratchpad",
        "research_web", "get_news", "get_event_proximity",
        "get_stock_data", "get_fundamentals",
        "get_analyst_recommendations",
        "request_evidence",
    ],
    "research_manager": [
        "read_scratchpad", "lookup_blackboard", "request_evidence",
    ],
}
