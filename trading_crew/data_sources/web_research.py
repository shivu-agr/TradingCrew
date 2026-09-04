"""Search → crawl → LLM-summarize web research pipeline.

A three-stage open-web research provider, built on the free websearch tool
vendored at ``scripts/websearch/`` and folding in Firecrawl as the crawler:

1. **Search for links** (:func:`search_links`) — get result URLs for a query.
   When ``GEMINI_WEB_SEARCH=1``, Vertex Gemini + Google Search grounding
   (``GEMINI_SEARCH_HOME`` / ``gemini_ai_search.search``) is tried first.
   If Gemini is off, empty, or errors, the existing SearXNG ring runs
   (``websearch.sh`` + keyless scrapers). Gemini is paid — leave the
   flag at ``0`` to stay on SearXNG only.
2. **Crawl the raw page** (:func:`crawl`) — fetch each link's main content
   (markdown) via Firecrawl ``/v1/scrape``, with a keyless ``requests`` +
   BeautifulSoup extractor as an alternate backend.
3. **LLM-summarize** (:func:`summarize`) — consolidate the crawled raw text
   into a concise, source-cited digest using the workflow's own LLM.

:func:`research_and_summarize` runs all three.  Everything is best-effort and
fail-soft: a dead backend is skipped, and the pipeline returns whatever it
could assemble (the news merge / analyst prompt must survive a bad provider).

Point-in-time handling: the web search itself is date-less, so PIT-awareness
is added in two places when an ``AsOfContext`` cutoff is active — (1) the
search query is bounded with a Google-style ``before:<cutoff>`` operator
(:func:`_pit_bounded_query`) so only results published on/before the decision
date are returned, and (2) the summarizer is told to treat the cutoff as
"today" and ignore anything dated after it.  Residual caveat: a crawled
page's *body* is still fetched live and could contain post-decision edits
(e.g. "last updated" articles / wikis), which cannot be fully eliminated for
arbitrary pages — so for strict training labels the dated news APIs (Tavily
``end_date``, SerpAPI / Firecrawl publish dates) remain the primary PIT source
and this pipeline is the last-resort supplement.
"""

from __future__ import annotations

import contextlib
import importlib.util
import logging
import os
import random
from collections import deque
import re
import shutil
import subprocess
import sys
import threading
import time

try:
    import fcntl  # POSIX file locking (Linux/macOS — google-headless targets these)
except ImportError:  # pragma: no cover — non-POSIX; serialize falls back to in-proc lock
    fcntl = None  # type: ignore[assignment]
from datetime import timedelta
from pathlib import Path
from typing import Callable, List, Optional
from urllib.parse import parse_qs, unquote, urlparse

import requests

from trading_crew.observability.trace import record_search

logger = logging.getLogger(__name__)

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

# Free link-search backends only (no paid search API).  ``searxng`` hits a
# self-hosted SearXNG JSON API; ``yahoo`` scrapes search.yahoo.com (scraper-
# tolerant, real Google-grade results); ``ddg_lite`` hits the lightweight
# lite.duckduckgo.com endpoint (more forgiving than the full HTML one);
# ``websearch`` shells out to the maintained ``websearch.sh`` script; the rest
# are inline keyless-scrape ports.
#
# Scaling to ~200 stocks/day is a *load-spreading* problem, not a single-best-
# backend problem: any one keyless scraper (brave/mojeek/ddg) soft-blocks after
# a handful of rapid queries, and even SearXNG's upstreams throttle it if it
# takes 100% of the traffic.  So instead of always hitting the same backend
# first, we (1) round-robin across a small set of high-quality PRIMARY backends
# so each takes only a share of the load, (2) fall through the remaining
# FALLBACK backends in order, (3) put any backend that reports a rate-limit /
# challenge into a timed cooldown so we stop hammering it, and (4) optionally
# space successive searches out by a global, jittered minimum gap.  All four
# are env-tunable (see the knobs below) so an operator can dial the throughput
# vs. throttle-risk trade-off for their exact daily volume + time window.

# Rotated per query so no single primary backend takes the whole load.  SearXNG
# is the lead aggregator — it already fans out to (and merges) google/yahoo/
# startpage/bing/… internally (see _SEARXNG_ENGINES), so those don't need to be
# separate *primary* backends.  The real-browser google-headless backend is
# prepended at import time when enabled (see TRADINGCREW_GOOGLE_HEADLESS_ENABLED)
# so it shares the primary rotation with searxng; it stays out entirely when
# disabled.  Standalone `yahoo` now lives behind SearXNG, so it drops to a
# fallback (still tried for extra coverage, but no longer double-queried as a
# primary where it mostly 500s on this IP).
_PRIMARY_SEARCH_BACKENDS = ("searxng",)
# Tried in fixed order after the primaries (best-effort extra coverage).
_FALLBACK_SEARCH_BACKENDS = ("yahoo", "ddg_lite", "brave", "ddg", "mojeek", "websearch")
# Legacy flat default (used only when a caller passes an explicit ``backends``
# list of None and rotation is disabled).
_DEFAULT_SEARCH_BACKENDS = _PRIMARY_SEARCH_BACKENDS + _FALLBACK_SEARCH_BACKENDS


def _env_float(name: str, default: float) -> float:
    try:
        raw = (os.environ.get(name) or "").strip()
        return float(raw) if raw else default
    except (TypeError, ValueError):
        return default


def _env_flag(name: str, default: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw not in ("0", "false", "no", "off")


def gemini_web_search_enabled() -> bool:
    """Vertex Gemini + Google Search. Off unless GEMINI_WEB_SEARCH is set."""
    raw = (
        os.environ.get("GEMINI_WEB_SEARCH")
        or os.environ.get("GOOGLE_WEB_SEARCH")
        or ""
    ).strip().strip('"').lower()
    return raw in ("1", "true", "yes", "on")


def gemini_search_home() -> Optional[str]:
    raw = (os.environ.get("GEMINI_SEARCH_HOME") or "").strip().strip('"')
    return raw.rstrip("/") or None


_GEMINI_MOD = None


def _gemini_module():
    """Load ``gemini_ai_search`` from ``GEMINI_SEARCH_HOME`` (same as churn)."""
    global _GEMINI_MOD
    if _GEMINI_MOD is not None:
        return _GEMINI_MOD
    home = gemini_search_home()
    if not home:
        raise RuntimeError(
            "GEMINI_WEB_SEARCH is on. Set GEMINI_SEARCH_HOME to the "
            "gemini-ai-search folder."
        )
    path = Path(home) / "gemini_ai_search.py"
    if not path.is_file():
        raise RuntimeError(f"gemini_ai_search.py not found at {path}")
    spec = importlib.util.spec_from_file_location("gemini_ai_search", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load gemini_ai_search from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gemini_ai_search"] = mod
    spec.loader.exec_module(mod)
    _GEMINI_MOD = mod
    return mod


def reset_gemini_module() -> None:
    """Test helper — drop the cached Gemini client module."""
    global _GEMINI_MOD
    _GEMINI_MOD = None
    sys.modules.pop("gemini_ai_search", None)


def _gemini_source_rows(result: object, limit: int) -> List[dict]:
    rows: List[dict] = []
    seen: set[str] = set()
    for src in getattr(result, "sources", None) or []:
        url = (getattr(src, "url", None) or "").strip()
        title = (getattr(src, "title", None) or "").strip() or url
        if not url.startswith("http"):
            continue
        key = url.split("#", 1)[0].rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        rows.append({"title": title, "url": url, "snippet": ""})
        if len(rows) >= limit:
            return rows
    for url in getattr(result, "links", None) or []:
        href = str(url or "").strip()
        if not href.startswith("http"):
            continue
        key = href.split("#", 1)[0].rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        rows.append({"title": href, "url": href, "snippet": ""})
        if len(rows) >= limit:
            break
    return rows


def _gemini_research(query: str) -> Optional[dict]:
    """Grounded Gemini answer + source URLs. None when the answer is empty."""
    result = _gemini_module().search(query)
    if not getattr(result, "success", False):
        error = getattr(result, "error", None) or "Gemini search failed"
        raise RuntimeError(error)
    content = (getattr(result, "content", None) or "").strip()
    sources = _gemini_source_rows(result, limit=8)
    if not content:
        return None
    return {"summary": content, "sources": sources}


def _search_gemini(query: str, limit: int) -> List[dict]:
    result = _gemini_module().search(query)
    if not getattr(result, "success", False):
        error = getattr(result, "error", None) or "Gemini search failed"
        raise RuntimeError(error)
    return _gemini_source_rows(result, limit=limit)


# --- throughput / throttle-risk knobs (all optional) -----------------------
# Minimum seconds between successive link searches, enforced process-wide
# (across all worker threads) so a big batch can be spread over hours.
_SEARCH_MIN_GAP = _env_float("TRADINGCREW_SEARCH_MIN_GAP_SECONDS", 0.0)
# Extra random 0..J seconds added on top of the min gap to avoid a robotic
# fixed cadence that engines fingerprint.
_SEARCH_GAP_JITTER = _env_float("TRADINGCREW_SEARCH_GAP_JITTER_SECONDS", 0.0)
# How long to skip a backend after it reports a rate-limit / challenge.
# The base (generic) cooldown; the reason-aware tiers below refine it to match
# how long the block ACTUALLY lasts upstream.  SearXNG itself benches an engine
# for `suspended_times` after a failure — 24h on CAPTCHA/access-denied, 1h on
# "too many requests" — so probing sooner than that just burns a ring attempt
# on a guaranteed-empty engine.  These app-side cooldowns are aligned to those
# classes (but capped well below SearXNG's 24h so a SearXNG restart, which
# clears its suspensions, is picked up within a couple of hours).
_SEARCH_COOLDOWN = _env_float("TRADINGCREW_SEARCH_BACKEND_COOLDOWN_SECONDS", 900.0)
# CAPTCHA / access-denied: the upstream is actively challenging this IP.  SearXNG
# holds it 24h; we re-probe every 2h in case SearXNG was restarted.
_SEARCH_COOLDOWN_CAPTCHA = _env_float("TRADINGCREW_SEARCH_COOLDOWN_CAPTCHA_SECONDS", 7200.0)
# HTTP 429 / "too many requests": matches SearXNG's 1h SearxEngineTooManyRequests.
_SEARCH_COOLDOWN_RATELIMIT = _env_float("TRADINGCREW_SEARCH_COOLDOWN_RATELIMIT_SECONDS", 3600.0)
# Timeout: transient/slow (e.g. DDG stalls >3s from a datacenter IP); shorter.
_SEARCH_COOLDOWN_TIMEOUT = _env_float("TRADINGCREW_SEARCH_COOLDOWN_TIMEOUT_SECONDS", 1800.0)


def _cooldown_for_reason(reason: str) -> float:
    """Pick a cooldown length from the block reason, mirroring SearXNG's own
    ``suspended_times`` classes so we don't re-probe an engine the upstream has
    already benched for hours."""
    r = (reason or "").lower()
    if "captcha" in r or "access denied" in r or "denied" in r or "403" in r:
        return _SEARCH_COOLDOWN_CAPTCHA
    if "too many" in r or "429" in r or "rate" in r or "limit" in r:
        return _SEARCH_COOLDOWN_RATELIMIT
    if "timeout" in r:
        return _SEARCH_COOLDOWN_TIMEOUT
    return _SEARCH_COOLDOWN
# Round-robin the primary backends per query (spreads load); set 0 to disable.
_SEARCH_ROTATE = _env_flag("TRADINGCREW_SEARCH_ROTATE", True)
# On a SearXNG HTTP-429, pause this many seconds and retry the SAME instance
# before giving up on it.  Per SearXNG's limiter a short-term block clears in
# ~10-15s, so a brief wait keeps our best (Google-grade) backend in rotation
# instead of sidelining it for the full per-backend cooldown after one blip.
_SEARXNG_RETRY_WAIT = _env_float("TRADINGCREW_SEARXNG_RETRY_WAIT_SECONDS", 12.0)
# Number of retries (after the first attempt) per SearXNG instance on a 429.
_SEARXNG_RETRIES = max(0, int(_env_float("TRADINGCREW_SEARXNG_RETRIES", 1.0)))
# Curated SearXNG engine set — one engine per ring hop.  Working general/news
# engines (duckduckgo web, privacywall, infospace, …) sit in front of the
# older bing/startpage/google/yahoo nodes.  The ring accepts a hop only after
# an LLM URL-relevance check; Bing-style same-site dumps rotate away.  Set
# ``SEARXNG_ENGINES=""`` to use SearXNG's full default set (slower, noisier).
# Measured-working engines first (Bharat Forge Q1 FY27 probe, two passes),
# then the previous ring so a healthy node is tried before Bing/Startpage
# dumps. Names must match SearXNG's ``/config`` engine list (spaces kept).
_DEFAULT_SEARXNG_ENGINES = (
    "duckduckgo web,privacywall,infospace,abcnyheter,"
    "bing news,google cse,duckduckgo news,gmx,"
    "duckduckgo,bing,startpage,google,brave,yahoo"
)
_SEARXNG_ENGINES = (
    os.environ.get("SEARXNG_ENGINES", _DEFAULT_SEARXNG_ENGINES) or ""
).strip()

# Circular-queue engine rotation (default ON).  Instead of fanning EVERY query
# out to ALL engines above — which hammers each upstream and gets them throttled
# (brave/ddg/yahoo were 429-ing ~all day) — send exactly ONE engine per query
# and rotate it head→tail on every call.  So N engines share the load: each sees
# only ~1/N of the query rate.  Together with the Kafka-paced request stream
# (bounded concurrency + launch stagger) this keeps every upstream comfortably
# under its rate-limit, so a single engine reliably answers each query without a
# multi-backend fan-out.  Set TRADINGCREW_SEARXNG_SINGLE_ENGINE=0 to restore the
# old fan-out-and-merge behaviour.
_SEARXNG_SINGLE_ENGINE = _env_flag("TRADINGCREW_SEARXNG_SINGLE_ENGINE", True)
_ENGINE_QUEUE: "deque[str]" = deque(
    e.strip() for e in _SEARXNG_ENGINES.split(",") if e.strip()
)
_ENGINE_QUEUE_LOCK = threading.Lock()


def _next_engine() -> str:
    """Return the head of the circular engine queue and rotate it to the tail.

    Thread-safe: concurrent workers each get the *next* engine, so K parallel
    queries naturally spread across K different engines.
    """
    with _ENGINE_QUEUE_LOCK:
        if not _ENGINE_QUEUE:
            return ""
        engine = _ENGINE_QUEUE[0]
        _ENGINE_QUEUE.rotate(-1)
        return engine

# --- google-headless backend (opt-in) --------------------------------------
# Scrapes real google.com results by shelling out to the vendored Node +
# Playwright(-extra stealth) driver at scripts/google-headless/search.js.  It's
# the highest-quality backend (Google's own ranking) but Google CAPTCHAs a
# datacenter/office IP after a handful of rapid queries, so it (1) runs a real
# (headful) Chromium with a PERSISTENT profile to keep challenge rates low, (2)
# must be PRIMED once (solve one CAPTCHA manually — see README), and (3) fails
# soft on a CAPTCHA (exit 4 → cooldown) so the rotation falls through to
# searxng/yahoo.  Results are URL-only (Google exposes no stable snippet node).
# OFF by default; enable with TRADINGCREW_GOOGLE_HEADLESS_ENABLED=1.
_GOOGLE_HEADLESS_ENABLED = _env_flag("TRADINGCREW_GOOGLE_HEADLESS_ENABLED", False)
_GOOGLE_HEADLESS_SCRIPT = (
    os.environ.get("GOOGLE_HEADLESS_SCRIPT")
    or str(Path(__file__).resolve().parents[2] / "scripts" / "google-headless" / "search.js")
)
_GOOGLE_HEADLESS_TIMEOUT = _env_float("GOOGLE_HEADLESS_TIMEOUT_SECONDS", 45.0)
# Rendering mode passed to the driver (GH_MODE): "offscreen" (invisible headful
# window — the practical macOS default; keeps the low-CAPTCHA headful
# fingerprint without showing a window), "headful" (visible window; use with
# Xvfb on Linux/Docker via GOOGLE_HEADLESS_DISPLAY), or "headless" (Chrome new
# headless — truly windowless but Google CAPTCHAs it ~100% on an office IP, so
# only for residential IPs where you've verified it passes).
_GOOGLE_HEADLESS_MODE = (os.environ.get("GOOGLE_HEADLESS_MODE") or "offscreen").strip()
# X display for Xvfb (Linux/Docker headful-on-virtual-screen); leave empty on
# macOS where a headful window renders directly.
_GOOGLE_HEADLESS_DISPLAY = (os.environ.get("GOOGLE_HEADLESS_DISPLAY") or "").strip()
# CA bundle so node's TLS trusts a corporate proxy (same bundle SearXNG uses).
_GOOGLE_HEADLESS_CA = (
    os.environ.get("GOOGLE_HEADLESS_CA")
    or os.environ.get("NODE_EXTRA_CA_CERTS")
    or str(Path.home() / ".trading_crew" / "ca-bundle.pem")
)
# Serialize google-headless launches machine-wide.  The driver drives ONE
# persistent Chromium profile dir (~/.cache/playwright-google), and Chromium
# refuses to open a profile that's already in use (its SingletonLock) — so
# concurrent batch workers collide and all but one exit 1 ("profile in use").
# A dedicated experiment (scripts/experiments/google_search_experiment.py)
# measured this directly: 4 concurrent launches → 10/12 launch-errors (17%
# success), whereas serialized → 83% success.  So we take a machine-wide file
# lock around the subprocess (plus an in-process lock) so at most one Chromium
# touches the profile at a time; the HTTP backends (searxng/yahoo) stay
# concurrent.  Disable with TRADINGCREW_GOOGLE_HEADLESS_SERIALIZE=0.
_GOOGLE_HEADLESS_SERIALIZE = _env_flag("TRADINGCREW_GOOGLE_HEADLESS_SERIALIZE", True)
# Max seconds to wait for the serialize lock before giving up and falling
# through to the next backend (so a stuck google run never stalls a search).
_GOOGLE_HEADLESS_LOCK_WAIT = _env_float("GOOGLE_HEADLESS_LOCK_WAIT_SECONDS", 60.0)
# Path of the machine-wide lock file (co-located with the persistent profile).
_GOOGLE_HEADLESS_LOCKFILE = (
    os.environ.get("GOOGLE_HEADLESS_LOCKFILE")
    or str(Path(os.environ.get("PW_USER_DATA")
                or (Path.home() / ".cache" / "playwright-google")).parent
           / "playwright-google.singleton.lock")
)
# Google CAPTCHA-cooldown: the same experiment found that once this office/DC IP
# crosses Google's volume budget it stays CAPTCHA-locked for a LONG time — a
# 5-minute break didn't recover it, and light probing every 2 min for 16 min
# never recovered (probing itself resets the clock).  So when google captchas,
# cool it down for much longer than the generic backend cooldown and STOP
# hammering it (letting the IP actually recover) — searxng/yahoo carry the load.
_GOOGLE_HEADLESS_COOLDOWN = _env_float("GOOGLE_HEADLESS_COOLDOWN_SECONDS", 1800.0)
_GOOGLE_HEADLESS_INPROC_LOCK = threading.Lock()
if _GOOGLE_HEADLESS_ENABLED and "google" not in _PRIMARY_SEARCH_BACKENDS:
    # Prepend so it leads the round-robin (best ranking first); rotation still
    # spreads load evenly across google/searxng/yahoo.
    _PRIMARY_SEARCH_BACKENDS = ("google",) + _PRIMARY_SEARCH_BACKENDS
    _DEFAULT_SEARCH_BACKENDS = _PRIMARY_SEARCH_BACKENDS + _FALLBACK_SEARCH_BACKENDS

# Shared throttle/cooldown state (search_links runs in worker threads).
_SEARCH_STATE_LOCK = threading.Lock()
_SEARCH_NEXT_SLOT = 0.0  # monotonic time the next search slot may start
_SEARCH_COOLDOWN_UNTIL: dict[str, float] = {}
_SEARCH_ROTATE_IDX = 0
# Per-backend cooldown overrides (fall back to the generic _SEARCH_COOLDOWN).
# Google recovers slowly once CAPTCHA-locked, so it gets a much longer cooldown.
_BACKEND_COOLDOWN_OVERRIDE: dict[str, float] = {"google": _GOOGLE_HEADLESS_COOLDOWN}


class _SearchRateLimited(Exception):
    """Raised by a backend when it detects a rate-limit / challenge page, so
    ``search_links`` can put that backend into a timed cooldown."""

# Default location of the ``websearch.sh`` tool — the copy vendored into this
# repo (scripts/websearch/).  Override with the ``WEBSEARCH_SH_PATH`` env var.
_DEFAULT_WEBSEARCH_SH = str(
    Path(__file__).resolve().parents[2] / "scripts" / "websearch" / "websearch.sh"
)


def _pit_bounded_query(query: str) -> str:
    """Append a Google-style ``before:<cutoff>`` operator when a point-in-time
    cutoff is active, so the search engine only returns results published on
    or before the decision date.

    This is how the otherwise date-less web search becomes PIT-aware: the
    ``before:`` operator is honoured by google-headless (and ignored
    harmlessly by engines that don't parse it).  Live mode (no cutoff) leaves
    the query untouched.  NOTE: this bounds the *search*; a crawled page's
    body is still fetched live, so strict training labels should prefer the
    dated news APIs (see the module docstring).
    """
    try:
        from trading_crew.data_sources import pit

        cutoff = pit.cutoff()
    except Exception:  # noqa: BLE001
        cutoff = None
    if cutoff is None:
        return query
    # ``before:`` is exclusive on some engines; add a day so the cutoff date
    # itself is included.
    before = (cutoff + timedelta(days=1)).strftime("%Y-%m-%d")
    if "before:" in query:
        return query
    return f"{query} before:{before}"


# ---------------------------------------------------------------------------
# Stage 1 — search for links
# ---------------------------------------------------------------------------


def _dedupe_links(links: List[dict], limit: int) -> List[dict]:
    seen: set[str] = set()
    out: List[dict] = []
    for l in links:
        url = (l.get("url") or "").strip()
        if not url.startswith("http"):
            continue
        key = url.split("#", 1)[0].rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "title": (l.get("title") or "").strip() or url,
            "url": url,
            # Preserve a snippet when a backend provides one (SearXNG does);
            # scraper backends leave this empty.
            "snippet": (l.get("snippet") or "").strip(),
        })
        if len(out) >= limit:
            break
    return out


def _searxng_bases() -> List[str]:
    """Resolve the configured SearXNG instance URL(s).

    ``SEARXNG_URLS`` (comma-separated) lets an operator spread load across
    several instances (self-hosted + public mirrors); ``SEARXNG_URL`` is the
    single-instance fallback, defaulting to ``http://localhost:8080`` to match
    ``websearch.sh``.  Instances are shuffled per call so the load is spread
    rather than always hitting the first one.
    """
    raw = (os.environ.get("SEARXNG_URLS") or "").strip()
    if raw:
        bases = [b.strip().rstrip("/") for b in raw.split(",") if b.strip()]
    else:
        bases = [os.environ.get("SEARXNG_URL", "http://localhost:8080").rstrip("/")]
    random.shuffle(bases)
    return bases


# Reasons SearXNG reports in ``unresponsive_engines`` that mean "the upstream
# blocked/limited us", not "genuinely no results" — e.g. "Suspended: CAPTCHA",
# "too many requests", "timeout".  Used to turn a 200+empty SearXNG response for
# a suspended engine into a throttle (→ cooldown) instead of a false "empty".
_SEARXNG_THROTTLE_HINTS = (
    "429", "too many", "rate", "limit", "captcha", "block", "suspend", "timeout",
)


def _is_throttle_reason(reason: str) -> bool:
    return any(k in (reason or "").lower() for k in _SEARXNG_THROTTLE_HINTS)


def _searxng_unresponsive_reason(data: dict, engine: str) -> Optional[str]:
    """Return SearXNG's ``unresponsive_engines`` reason for ``engine`` (or None).

    ``unresponsive_engines`` is a list of ``[engine, reason]`` pairs.
    """
    eng = (engine or "").lower()
    for u in data.get("unresponsive_engines", []) or []:
        if isinstance(u, (list, tuple)) and u and str(u[0]).lower() == eng:
            return str(u[1]).lower() if len(u) > 1 else ""
        if isinstance(u, str) and u.lower() == eng:
            return ""
    return None


def _record_searxng_engines(data: dict, requested: Optional[List[str]] = None) -> None:
    """Break a SearXNG JSON response down into per-engine outcomes and record
    them as ``searxng/<engine>`` in the search telemetry, so the dashboard can
    *expand* SearXNG's single row into the engines running behind it (bing /
    startpage / google / yahoo / duckduckgo / brave …) — who actually
    contributed vs. who was throttled/errored/empty.  Best-effort: telemetry
    must never break a search.

    ``requested`` is the set of engines actually asked for on this call (one
    engine in circular-queue mode, the full list in fan-out mode); only those
    are marked "empty" when they neither contributed nor errored.
    """
    if requested is None:
        requested = [e.strip() for e in _SEARXNG_ENGINES.split(",")]
    try:
        contributed: set[str] = set()
        for r in data.get("results", []):
            engs = r.get("engines") or ([r.get("engine")] if r.get("engine") else [])
            for e in engs:
                if e:
                    contributed.add(str(e).lower())
        unresponsive: dict[str, str] = {}
        for u in data.get("unresponsive_engines", []):
            if isinstance(u, (list, tuple)) and u:
                unresponsive[str(u[0]).lower()] = (str(u[1]).lower() if len(u) > 1 else "")
            elif isinstance(u, str):
                unresponsive[u.lower()] = ""
        for eng in contributed:
            record_search(f"searxng/{eng}", "success")
        for eng, reason in unresponsive.items():
            if eng in contributed:
                continue
            outcome = "throttled" if _is_throttle_reason(reason) else "error"
            record_search(f"searxng/{eng}", outcome)
        # Requested engines that neither contributed nor errored responded empty.
        for eng in (e.strip().lower() for e in requested):
            if eng and eng not in contributed and eng not in unresponsive:
                record_search(f"searxng/{eng}", "empty")
    except Exception:  # noqa: BLE001 — never let telemetry break a search
        pass


def _search_searxng(query: str, limit: int, engine: Optional[str] = None) -> List[dict]:
    """Query a self-hosted SearXNG instance via its JSON API.

    SearXNG fans each query out to ~70 upstream engines behind the scenes, so
    it degrades slowly under load.  It can still throttle (HTTP 429) once it
    takes 100% of a big batch's traffic — which is why it's now only one of
    several rotated backends.  Supports one or many instances via
    ``SEARXNG_URLS`` (see :func:`_searxng_bases`).  Raises
    :class:`_SearchRateLimited` on a 429 so the caller can cool it down;
    returns ``[]`` on any other error so the chain falls through.
    """
    last_exc: Optional[Exception] = None
    # ``engine`` pins this call to a single upstream (the circular-queue ring
    # passes the exact engine it popped).  Called bare in single-engine mode it
    # self-rotates one engine; in fan-out mode it queries all engines at once.
    if engine is not None:
        requested = [engine] if engine else []
    elif _SEARXNG_SINGLE_ENGINE:
        picked = _next_engine()
        requested = [picked] if picked else []
    else:
        requested = [e.strip() for e in _SEARXNG_ENGINES.split(",") if e.strip()]
    engines_param = (
        f"&engines={requests.utils.quote(','.join(requested))}" if requested else ""
    )
    for base in _searxng_bases():
        url = (
            f"{base}/search?q={requests.utils.quote(query)}"
            f"&format=json&safesearch=0{engines_param}"
        )
        # Retry the same instance on a 429 after a short pause (SearXNG's
        # short-term limiter clears in ~10-15s) before falling through.
        for attempt in range(_SEARXNG_RETRIES + 1):
            try:
                resp = requests.get(url, headers={"User-Agent": _UA}, timeout=15)
            except Exception as exc:  # noqa: BLE001 — try the next instance
                last_exc = exc
                break
            if resp.status_code == 429:
                last_exc = _SearchRateLimited(f"searxng 429 at {base}")
                if attempt < _SEARXNG_RETRIES and _SEARXNG_RETRY_WAIT > 0:
                    logger.info(
                        "searxng 429 at %s — waiting %.0fs then retrying (%d/%d)",
                        base, _SEARXNG_RETRY_WAIT, attempt + 1, _SEARXNG_RETRIES,
                    )
                    time.sleep(_SEARXNG_RETRY_WAIT)
                    continue
                break  # retries exhausted for this instance
            try:
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                break
            # Per-engine breakdown (``searxng/<engine>``) is only meaningful in
            # FAN-OUT mode, where one SearXNG call hits many engines and we want
            # to see which contributed.  In single-engine ring mode each call
            # already has its own top-level ``searxng:<engine>`` node counter, so
            # recording the breakdown too just double-counts and clutters the
            # dashboard (an orphan ``searxng`` parent + duplicate engine rows).
            if len(requested) > 1:
                _record_searxng_engines(data, requested)
            # Single-engine ring mode: SearXNG answers 200+empty when the pinned
            # engine is CAPTCHA-suspended / rate-limited upstream (it lands in
            # ``unresponsive_engines`` — e.g. "Suspended: CAPTCHA", "too many
            # requests", "timeout").  Surface that as a throttle so the ring cools
            # the node down and rotates to a healthy one, instead of mislabeling
            # it "empty" and re-trying the dead engine on every rotation.
            if len(requested) == 1:
                reason = _searxng_unresponsive_reason(data, requested[0])
                if reason is not None and _is_throttle_reason(reason):
                    last_exc = _SearchRateLimited(
                        f"searxng/{requested[0]} unavailable: {reason or 'suspended'}"
                    )
                    break
            rows: List[dict] = []
            for r in data.get("results", []):
                u = (r.get("url") or "").strip()
                if u.startswith("http"):
                    rows.append({
                        "title": (r.get("title") or u).strip(),
                        "url": u,
                        # SearXNG returns a per-result snippet in ``content`` —
                        # keep it so "scan" tools (e.g. get_corporate_actions)
                        # can show text next to each link without a second crawl.
                        "snippet": (r.get("content") or "").strip().replace("\n", " ")[:280],
                    })
                if len(rows) >= limit:
                    break
            if rows:
                return rows
            break  # valid but empty response — move to the next instance
    # All instances failed: surface a rate-limit signal if that's what we saw,
    # otherwise fall through empty.
    if isinstance(last_exc, _SearchRateLimited):
        raise last_exc
    return []


def _search_websearch_sh(query: str, limit: int) -> List[dict]:
    """Run the free ``websearch.sh`` script and parse its stdout URLs.

    The script walks its own auto-fallback chain; we pass ``--disable
    google-headless`` because that Playwright backend soft-CAPTCHAs after
    ~2-3 rapid queries and stalls for seconds per call — fatal at 200-stock
    scale.  Titles aren't emitted, so the URL doubles as the title.  Returns
    ``[]`` when the script is missing / errors / prints nothing.
    """
    script = os.environ.get("WEBSEARCH_SH_PATH", _DEFAULT_WEBSEARCH_SH)
    if not (script and os.path.isfile(script)):
        return []
    bash = shutil.which("bash") or "bash"
    try:
        proc = subprocess.run(
            [bash, script, "--disable", "google-headless", "-n", str(limit), query],
            capture_output=True,
            text=True,
            timeout=90,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("websearch.sh failed for %r: %s", query, exc)
        return []
    rows: List[dict] = []
    for line in (proc.stdout or "").splitlines():
        u = line.strip()
        if u.startswith("http"):
            rows.append({"title": u, "url": u})
    return rows


@contextlib.contextmanager
def _google_headless_serialize():
    """Machine-wide serialization for the single shared Chromium profile.

    Yields ``True`` if the caller may proceed, or ``False`` if the lock could
    not be acquired within ``_GOOGLE_HEADLESS_LOCK_WAIT`` (caller should fall
    through to the next backend).  Chromium enforces one process per profile
    dir via its SingletonLock, so concurrent launches otherwise collide and all
    but one exit 1.  A dedicated experiment measured 4 concurrent launches at
    17% success (10/12 profile-lock failures) vs 83% serialized.  We combine an
    in-process lock (fast path across this process's worker threads) with a
    POSIX file lock (covers cron + web-server as separate processes).  HTTP
    backends stay concurrent — only google is gated here.
    """
    if not _GOOGLE_HEADLESS_SERIALIZE:
        yield True
        return
    wait = max(_GOOGLE_HEADLESS_LOCK_WAIT, 0.0)
    if not _GOOGLE_HEADLESS_INPROC_LOCK.acquire(timeout=wait):
        yield False
        return
    fh = None
    acquired_file = False
    try:
        if fcntl is not None:
            try:
                fh = open(_GOOGLE_HEADLESS_LOCKFILE, "a+")
                deadline = time.monotonic() + wait
                while True:
                    try:
                        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        acquired_file = True
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            break
                        time.sleep(0.5)
            except Exception:  # noqa: BLE001 — file lock unusable: proceed on in-proc lock
                acquired_file = True
        else:
            acquired_file = True  # no fcntl (non-POSIX): in-process lock is best-effort
        yield acquired_file
    finally:
        if fh is not None:
            try:
                if acquired_file and fcntl is not None:
                    fcntl.flock(fh, fcntl.LOCK_UN)
            except Exception:  # noqa: BLE001
                pass
            fh.close()
        _GOOGLE_HEADLESS_INPROC_LOCK.release()


def _search_google_headless(query: str, limit: int) -> List[dict]:
    """Scrape real google.com results via the vendored Node/Playwright driver.

    Shells out to ``scripts/google-headless/search.js`` (stealth-patched headful
    Chromium with a persistent, once-primed profile).  Google's own ranking is
    the best of any backend here, but the trade-offs are: (a) URL-only results —
    Google exposes no stable snippet node, so rows carry no snippet; (b) it
    CAPTCHAs an office/datacenter IP under rapid fire — the driver exits 4 on a
    challenge, which we surface as :class:`_SearchRateLimited` so the caller
    cools it down and rotates to searxng/yahoo.  Exit 5 (nothing parsed) / 3
    (transport) / spawn failure all fall through empty.  Requires a one-time
    prime (see README); disabled unless TRADINGCREW_GOOGLE_HEADLESS_ENABLED=1.
    """
    script = _GOOGLE_HEADLESS_SCRIPT
    if not (script and os.path.isfile(script)):
        logger.warning("google-headless script missing at %s — skipping", script)
        return []
    node = shutil.which("node")
    if not node:
        logger.warning("google-headless: 'node' not on PATH — skipping")
        return []
    env = {**os.environ, "WS_NUM": str(limit)}
    if _GOOGLE_HEADLESS_MODE:
        env["GH_MODE"] = _GOOGLE_HEADLESS_MODE
    if _GOOGLE_HEADLESS_DISPLAY:
        env["DISPLAY"] = _GOOGLE_HEADLESS_DISPLAY
    if _GOOGLE_HEADLESS_CA and os.path.isfile(_GOOGLE_HEADLESS_CA):
        env["NODE_EXTRA_CA_CERTS"] = _GOOGLE_HEADLESS_CA
    try:
        with _google_headless_serialize() as ok:
            if not ok:
                logger.info(
                    "google-headless busy (serialize lock not acquired in %.0fs) — "
                    "falling through to next backend", _GOOGLE_HEADLESS_LOCK_WAIT,
                )
                return []
            proc = subprocess.run(
                [node, script, query],
                cwd=os.path.dirname(script),
                capture_output=True,
                text=True,
                env=env,
                timeout=_GOOGLE_HEADLESS_TIMEOUT,
            )
    except subprocess.TimeoutExpired:
        logger.warning("google-headless timed out for %r", query)
        return []
    except Exception as exc:  # noqa: BLE001 — spawn failure: fall through
        logger.warning("google-headless spawn failed for %r: %s", query, exc)
        return []
    if proc.returncode == 4:
        # CAPTCHA — cool this backend down and rotate away.
        raise _SearchRateLimited("google-headless captcha")
    if proc.returncode != 0:
        logger.info(
            "google-headless exit %d for %r: %s",
            proc.returncode, query, (proc.stderr or "").strip()[:200],
        )
        return []
    rows: List[dict] = []
    for line in (proc.stdout or "").splitlines():
        u = line.strip()
        if u.startswith("http"):
            rows.append({"title": u, "url": u})
        if len(rows) >= limit:
            break
    return rows


def _scrape_html(url: str, timeout: int = 12) -> str:
    resp = requests.get(url, headers={"User-Agent": _UA, "Accept-Language": "en-US,en;q=0.9"}, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def _search_yahoo(query: str, limit: int) -> List[dict]:
    """Scrape search.yahoo.com.  Yahoo is notably scraper-tolerant and returns
    Google-grade organic results; its result links are wrapped in a
    ``/RU=<encoded-target>/RK=`` redirector we unwrap here."""
    try:
        html = _scrape_html(f"https://search.yahoo.com/search?p={requests.utils.quote(query)}")
    except requests.HTTPError as exc:
        # Yahoo soft-rate-limits by returning 5xx / 429 / 999 rather than a
        # challenge page.  Treat those as a throttle (→ cooldown, stop
        # hammering it) instead of a hard error so the fallback chain rotates
        # away cleanly and the metric reads "throttled", not "error".
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status is not None and (status >= 500 or status in (429, 999)):
            raise _SearchRateLimited(f"yahoo http {status}") from exc
        raise
    low = html.lower()
    if "unusual traffic" in low or "verify you are a human" in low:
        raise _SearchRateLimited("yahoo challenge")
    rows: List[dict] = []
    seen: set[str] = set()
    for m in re.finditer(r"/RU=(https?%3a[^/]+)/RK=", html, re.I):
        real = unquote(m.group(1))
        host = urlparse(real).netloc.lower()
        if not real.startswith("http") or "yahoo.com" in host:
            continue
        key = real.split("#", 1)[0]
        if key in seen:
            continue
        seen.add(key)
        rows.append({"title": real, "url": real})
        if len(rows) >= limit:
            break
    return rows


def _search_ddg_lite(query: str, limit: int) -> List[dict]:
    """Scrape lite.duckduckgo.com — the lightweight endpoint is more tolerant
    of scripted queries than the full ``html.duckduckgo.com`` page.  Links come
    through the ``//duckduckgo.com/l/?uddg=<encoded>`` redirector."""
    html = _scrape_html(f"https://lite.duckduckgo.com/lite/?q={requests.utils.quote(query)}")
    low = html.lower()
    if "anomaly" in low or "please solve" in low or "challenge-submit" in html:
        raise _SearchRateLimited("ddg-lite challenge")
    rows: List[dict] = []
    seen: set[str] = set()
    for m in re.finditer(r'href="((?://|https?://)duckduckgo\.com/l/\?uddg=[^"]+)"', html):
        href = m.group(1)
        if href.startswith("//"):
            href = "https:" + href
        qs = parse_qs(urlparse(href).query)
        real = unquote(qs["uddg"][0]) if "uddg" in qs else ""
        if not real.startswith("http") or "duckduckgo.com" in urlparse(real).netloc:
            continue
        key = real.split("#", 1)[0]
        if key in seen:
            continue
        seen.add(key)
        rows.append({"title": real, "url": real})
        if len(rows) >= limit:
            break
    return rows


def _search_ddg(query: str, limit: int) -> List[dict]:
    html = _scrape_html(f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}")
    if any(x in html.lower() for x in ("anomaly", "please solve")) or "challenge-submit" in html:
        raise _SearchRateLimited("ddg challenge")
    rows: List[dict] = []
    for m in re.finditer(r'class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html, re.S):
        href, title = m.group(1), re.sub(r"<[^>]+>", "", m.group(2)).strip()
        if href.startswith("//"):
            href = "https:" + href
        p = urlparse(href)
        real = href
        if "duckduckgo.com" in p.netloc and p.path.startswith("/l/"):
            qs = parse_qs(p.query)
            if "uddg" in qs:
                real = unquote(qs["uddg"][0])
        if real.startswith("http") and "duckduckgo.com" not in urlparse(real).netloc:
            rows.append({"title": title, "url": real})
    return rows


def _search_brave(query: str, limit: int) -> List[dict]:
    html = _scrape_html(f"https://search.brave.com/search?q={requests.utils.quote(query)}&source=web")
    if re.search(r"challenge|verify.you.re.human|captcha", html, re.I) and len(html) < 100000:
        raise _SearchRateLimited("brave challenge")
    rows: List[dict] = []
    for m in re.finditer(r'data-type="web"[\s\S]{0,600}?<a[^>]+href="(https?://[^"]+)"', html):
        u = m.group(1)
        if "brave.com" not in urlparse(u).netloc:
            rows.append({"title": u, "url": u})
    return rows


def _search_mojeek(query: str, limit: int) -> List[dict]:
    html = _scrape_html(f"https://www.mojeek.com/search?q={requests.utils.quote(query)}")
    if "403 - Forbidden" in html or "automated queries" in html.lower():
        raise _SearchRateLimited("mojeek 403")
    rows: List[dict] = []
    for m in re.finditer(r'<a\s+class="title"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html, re.S):
        u = m.group(1)
        title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
        if u.startswith("http"):
            rows.append({"title": title or u, "url": u})
    return rows


_SEARCH_FUNCS: dict[str, Callable[[str, int], List[dict]]] = {
    "google": _search_google_headless,
    "searxng": _search_searxng,
    "yahoo": _search_yahoo,
    "ddg_lite": _search_ddg_lite,
    "websearch": _search_websearch_sh,
    "ddg": _search_ddg,
    "brave": _search_brave,
    "mojeek": _search_mojeek,
}


# --- Unified circular-queue search ring ------------------------------------
# ONE flat ring of "providers", each serving exactly one engine per query:
#   ``searxng:<engine>`` → SearXNG constrained to that single engine (bing,
#                          startpage, google, duckduckgo, brave, yahoo, …)
#   ``google``           → the real-browser google-headless scraper — the one
#                          Google path SearXNG can't replicate — folded into the
#                          same ring (appended last) when it's enabled.
# ``search_links`` pops the head, tries it, and rotates it to the tail on EVERY
# call (success or fail), so N nodes split the traffic ~1/N and no query ever
# fans out to all engines at once.  On failure it keeps rotating DOWN the ring
# (retry with the next node) until one returns links or a full rotation is
# exhausted — so a throttled node costs one retry, not an empty result.
# Nodes currently in a rate-limit cooldown are skipped (still rotated past).
# Override the whole ring verbatim with ``TRADINGCREW_SEARCH_RING`` (CSV).
def _build_search_ring() -> List[str]:
    override = (os.environ.get("TRADINGCREW_SEARCH_RING") or "").strip()
    if override:
        return [p.strip() for p in override.split(",") if p.strip()]
    ring: List[str] = [
        f"searxng:{e.strip()}" for e in _SEARXNG_ENGINES.split(",") if e.strip()
    ]
    if _GOOGLE_HEADLESS_ENABLED:
        ring.append("google")
    return ring


_PROVIDER_RING: "deque[str]" = deque(_build_search_ring())
_PROVIDER_RING_LOCK = threading.Lock()
# Max nodes to try per query before giving up.  0 → one full rotation of the
# ring (each node at most once).  Set lower to cap the retry fan-out per query.
_SEARCH_MAX_ATTEMPTS = int(_env_float("TRADINGCREW_SEARCH_MAX_ATTEMPTS", 0))


def _next_provider() -> Optional[str]:
    """Pop the head of the provider ring and rotate it to the tail (thread-safe).

    Advancing a single GLOBAL ring means the rotation persists across queries
    (so successive queries hit different nodes) AND across the retries within a
    query — K concurrent workers each draw the next distinct node off the ring.
    """
    with _PROVIDER_RING_LOCK:
        if not _PROVIDER_RING:
            return None
        name = _PROVIDER_RING[0]
        _PROVIDER_RING.rotate(-1)
        return name


def _run_provider(name: str, query: str, limit: int) -> List[dict]:
    """Dispatch a ring node: ``searxng:<engine>`` pins SearXNG to one engine,
    anything else maps to a plain backend in ``_SEARCH_FUNCS`` (e.g. ``google``
    → google-headless)."""
    if name.startswith("searxng:"):
        return _search_searxng(query, limit, engine=name.split(":", 1)[1])
    fn = _SEARCH_FUNCS.get(name)
    return fn(query, limit) if fn is not None else []


_RELEVANCE_PROMPT = """\
You judge whether a web-search result set is about the user's query.

QUERY:
{query}

RESULTS:
{results}

Rules:
- RELEVANT if most listed titles/URLs are about the same company, ticker,
  event, or claim as the query.
- IRRELEVANT if the set is a same-site navigation dump, a different topic,
  a random local-pack, or unrelated homepages.
- Use only the titles and URLs given. Do not invent pages.
- The VERY LAST line of your reply MUST be exactly one word: RELEVANT or
  IRRELEVANT.
"""


def parse_relevance_verdict(raw: str) -> Optional[bool]:
    """Parse an explicit last-line RELEVANT / IRRELEVANT verdict.

    Returns True/False only when that token is present. Returns None when
    the model did not emit a verdict — callers must not treat None as a
    yes (that would ship junk).
    """
    lines = [ln.strip() for ln in (raw or "").splitlines() if ln.strip()]
    if not lines:
        return None
    token = lines[-1].rstrip(".!").split()[-1].strip(":").upper()
    if token == "IRRELEVANT":
        return False
    if token == "RELEVANT":
        return True
    return None


def links_relevant_to_query(
    query: str,
    links: List[dict],
    *,
    llm=None,
) -> bool:
    """True only when the workflow LLM marks ``links`` as about ``query``.

    Fail-closed: an empty set, a missing verdict, or a judge error is
    not relevance. The ring then waits and tries the next SearXNG node.
    """
    if not links:
        return False
    blocks = []
    for i, row in enumerate(links, 1):
        title = (row.get("title") or "").strip() or "(no title)"
        url = (row.get("url") or "").strip()
        blocks.append(f"{i}. {title}\n   {url}")
    prompt = _RELEVANCE_PROMPT.format(query=query, results="\n".join(blocks))
    if llm is None:
        from trading_crew._common import get_llm

        llm = get_llm(temperature=0.0)
    try:
        raw = llm.call(messages=[{"role": "user", "content": prompt}])
    except Exception as exc:  # noqa: BLE001 — judge must not accept junk
        logger.warning("search relevance judge failed: %s", exc)
        return False
    verdict = parse_relevance_verdict(raw if isinstance(raw, str) else str(raw or ""))
    if verdict is None:
        logger.warning(
            "search relevance judge emitted no RELEVANT/IRRELEVANT verdict "
            "for %r — treating as irrelevant", query[:120],
        )
        return False
    return verdict


def _acquire_search_slot() -> None:
    """The same search wait used for the first hop (gateway slot or gap)."""
    try:
        from trading_crew.data_sources import rate_gateway

        if rate_gateway.is_enabled():
            rate_gateway.acquire("search")
            return
    except Exception:  # noqa: BLE001 — never let rate limiting break the search
        pass
    _search_throttle()


_SEARCH_THROTTLE_LOGGED = False


def _search_throttle() -> None:
    """Process-wide search rate limiter — an application-level queue that hands
    every link-search a monotonically-increasing time *slot* so requests leave
    at a uniform cadence instead of bursting.

    Why a slot scheduler (not "sleep since last search"): with many concurrent
    workflow workers, the old "hold the lock and sleep until last+gap" approach
    let a whole wave of threads that arrived during an idle valley fire almost
    together (peaks), then go quiet (valleys).  Here each caller *reserves* the
    next slot atomically under the lock (now, now+gap, now+2·gap, …) and then
    sleeps OUTSIDE the lock until its slot — so N callers are spaced by exactly
    one gap each, killing the peak/valley wave.  A no-op when both knobs are 0.
    """
    global _SEARCH_NEXT_SLOT, _SEARCH_THROTTLE_LOGGED
    if not _SEARCH_THROTTLE_LOGGED:
        _SEARCH_THROTTLE_LOGGED = True
        logger.info(
            "search throttle: min_gap=%.1fs jitter=%.1fs (effective spacing "
            "%.1f-%.1fs/search); rotate=%s",
            _SEARCH_MIN_GAP, _SEARCH_GAP_JITTER,
            _SEARCH_MIN_GAP, _SEARCH_MIN_GAP + _SEARCH_GAP_JITTER, _SEARCH_ROTATE,
        )
    if _SEARCH_MIN_GAP <= 0 and _SEARCH_GAP_JITTER <= 0:
        return
    gap = _SEARCH_MIN_GAP
    if _SEARCH_GAP_JITTER > 0:
        gap += random.uniform(0.0, _SEARCH_GAP_JITTER)
    with _SEARCH_STATE_LOCK:
        now = time.monotonic()
        slot = max(now, _SEARCH_NEXT_SLOT)
        _SEARCH_NEXT_SLOT = slot + gap
    wait = slot - time.monotonic()
    if wait > 0:
        time.sleep(wait)


def _resolve_backend_order(backends: Optional[List[str]]) -> List[str]:
    """Build this call's backend order: an explicit list / ``WEBSEARCH_BACKENDS``
    override wins verbatim; otherwise round-robin the PRIMARY backends (to
    spread load) and append the FALLBACK backends.  Backends currently in a
    rate-limit cooldown are dropped (unless that would leave nothing)."""
    if backends is not None:
        order = list(backends)
    else:
        env = (os.environ.get("WEBSEARCH_BACKENDS") or "").strip()
        if env:
            order = [b.strip() for b in env.split(",") if b.strip()]
        else:
            primary = list(_PRIMARY_SEARCH_BACKENDS)
            if _SEARCH_ROTATE and len(primary) > 1:
                global _SEARCH_ROTATE_IDX
                with _SEARCH_STATE_LOCK:
                    k = _SEARCH_ROTATE_IDX % len(primary)
                    _SEARCH_ROTATE_IDX += 1
                primary = primary[k:] + primary[:k]
            order = primary + list(_FALLBACK_SEARCH_BACKENDS)

    now = time.monotonic()
    with _SEARCH_STATE_LOCK:
        live = [b for b in order if _SEARCH_COOLDOWN_UNTIL.get(b, 0.0) <= now]
    # Never let cooldowns starve us completely — fall back to the full order.
    return live or order


def _mark_cooldown(name: str, reason: str = "") -> None:
    # An explicit per-backend override wins; otherwise size the cooldown to the
    # block reason (CAPTCHA→2h, 429→1h, timeout→30m, else base) so we don't
    # re-probe an engine the upstream has benched for hours.
    if name in _BACKEND_COOLDOWN_OVERRIDE:
        cooldown = _BACKEND_COOLDOWN_OVERRIDE[name]
    else:
        cooldown = _cooldown_for_reason(reason)
    if cooldown <= 0:
        return
    with _SEARCH_STATE_LOCK:
        _SEARCH_COOLDOWN_UNTIL[name] = time.monotonic() + cooldown
    logger.info("search backend %s unavailable (%s) — cooling down %.0fs",
                name, reason or "rate-limited", cooldown)


def _search_links_ring(query: str, max_results: int) -> List[dict]:
    """Circular-queue search: try ONE node per attempt, retry down the ring on
    failure until a node returns *relevant* links or a full rotation is exhausted.

    SearXNG hops are accepted only after :func:`links_relevant_to_query`. An
    off-topic URL set (Suzuki dealer nav, weather tags, …) is treated as
    empty: wait the same search slot again, then the next node.  The global
    ring is rotated once per attempt (via :func:`_next_provider`), so the
    starting node advances across queries.  Nodes in a rate-limit cooldown
    are skipped without spending a network call.
    """
    with _PROVIDER_RING_LOCK:
        ring_len = len(_PROVIDER_RING)
    if ring_len == 0:
        return []
    max_attempts = _SEARCH_MAX_ATTEMPTS if _SEARCH_MAX_ATTEMPTS > 0 else ring_len
    max_attempts = min(max_attempts, ring_len)
    tried: List[str] = []
    throttled: List[str] = []
    seen: set[str] = set()
    # Bound the loop to one full rotation so we never spin on a busy ring; the
    # ``seen`` guard also stops early if concurrent rotation wraps us around.
    for _ in range(ring_len):
        if len(tried) >= max_attempts:
            break
        name = _next_provider()
        if name is None:
            break
        if name in seen:
            break  # completed a full cycle — every node has had a turn
        seen.add(name)
        with _SEARCH_STATE_LOCK:
            cooling = _SEARCH_COOLDOWN_UNTIL.get(name, 0.0) > time.monotonic()
        if cooling:
            continue  # already rotated past it; don't spend a call on a cooled node
        tried.append(name)
        try:
            rows = _run_provider(name, query, max_results)
        except _SearchRateLimited as exc:
            _mark_cooldown(name, str(exc))
            throttled.append(name)
            record_search(name, "throttled")
            logger.info("search node %s throttled for %r: %s — rotating to next",
                        name, query, exc)
            continue
        except Exception as exc:  # noqa: BLE001 — retry the next node
            record_search(name, "error")
            logger.warning("search node %s failed for %r: %s — rotating to next",
                           name, query, exc)
            continue
        deduped = _dedupe_links(rows, max_results)
        if deduped and name.startswith("searxng:"):
            if not links_relevant_to_query(query, deduped):
                sample = (deduped[0].get("url") or "")[:120]
                record_search(name, "empty")
                logger.info(
                    "search_links: %s returned %d off-topic links for %r "
                    "(e.g. %s) — waiting, then next node",
                    name, len(deduped), query, sample,
                )
                _acquire_search_slot()
                continue
        if deduped:
            record_search(name, "success")
            logger.info("search_links: %d links from %s (attempt %d)",
                        len(deduped), name, len(tried))
            return deduped
        record_search(name, "empty")
        logger.info("search_links: %s returned 0 links for %r — rotating to next",
                    name, query)
    if throttled:
        logger.warning(
            "search_links: NO results for %r — ring exhausted after %d attempts; "
            "throttled: [%s] tried: [%s]",
            query, len(tried), ",".join(throttled), ",".join(tried),
        )
    else:
        logger.warning(
            "search_links: NO results for %r — %d nodes tried, all empty: [%s]",
            query, len(tried), ",".join(tried),
        )
    return []


def search_links(
    query: str,
    max_results: int = 6,
    backends: Optional[List[str]] = None,
    *,
    skip_gemini: bool = False,
) -> List[dict]:
    """Return up to ``max_results`` ``{title, url}`` for ``query``.

    Circular-queue mode (default, ``TRADINGCREW_SEARXNG_SINGLE_ENGINE=1``):
    each query is served by ONE node popped off a shared circular ring
    (``searxng:<engine>`` nodes + ``google`` headless when enabled), rotated to
    the tail on every call so N nodes split the traffic ~1/N (no fan-out).  On
    failure it retries DOWN the ring — the next node, then the next — until one
    returns links the LLM marks as about the query, or a full rotation is
    exhausted, skipping nodes in cooldown.
    So a throttled node costs one retry, not an empty result.  The Kafka-paced
    request stream (bounded concurrency + launch stagger) keeps each node's ~1/N
    share under its rate-limit, so attempt #1 usually already succeeds.

    Legacy fan-out mode (``TRADINGCREW_SEARXNG_SINGLE_ENGINE=0``): SearXNG
    queries ALL engines at once and merges, then the FALLBACK backends are tried
    in order; the first non-empty result wins.  In either mode a node that
    reports a rate-limit / challenge is cooled down, and successive searches are
    optionally spaced out by a global, jittered minimum gap.  An explicit
    ``backends`` arg or the ``WEBSEARCH_BACKENDS`` env var force the legacy path
    with that exact order.
    """
    # Rate limiting: when the web server's Kafka gateway is running, funnel this
    # search through the shared ``search`` slot queue (centralised + visible in
    # the Console).  Otherwise (headless CLI / cron) use the in-process throttle.
    # The same slot is paid again before a ring retry after off-topic URLs.
    _acquire_search_slot()
    # PIT: bound the query to the decision date when a cutoff is active.
    query = _pit_bounded_query(query)
    # Gemini (paid, opt-in) is first when enabled. An explicit backends list
    # or WEBSEARCH_BACKENDS keeps the caller-chosen order (tests / operators).
    _env_backends = (os.environ.get("WEBSEARCH_BACKENDS") or "").strip()
    if (
        backends is None
        and not _env_backends
        and not skip_gemini
        and gemini_web_search_enabled()
    ):
        try:
            gem_rows = _dedupe_links(_search_gemini(query, max_results), max_results)
        except Exception as exc:  # noqa: BLE001 — SearXNG is the configured backup
            record_search("gemini", "error")
            logger.warning(
                "gemini search failed for %r — falling through to SearXNG: %s",
                query, exc,
            )
            gem_rows = []
        if gem_rows:
            record_search("gemini", "success")
            logger.info("search_links: %d links from gemini", len(gem_rows))
            return gem_rows
        record_search("gemini", "empty")
        logger.info(
            "search_links: gemini returned 0 links for %r — SearXNG next", query
        )
    # Circular-queue ring (default): one node per attempt, retry down the ring
    # on failure.  Skipped when a caller forces an explicit backend order.
    if (
        backends is None
        and not _env_backends
        and _SEARXNG_SINGLE_ENGINE
        and _PROVIDER_RING
    ):
        return _search_links_ring(query, max_results)
    order = _resolve_backend_order(backends)
    throttled: List[str] = []
    for name in order:
        fn = _SEARCH_FUNCS.get(name)
        if fn is None:
            continue
        try:
            rows = fn(query, max_results)
        except _SearchRateLimited as exc:
            _mark_cooldown(name, str(exc))
            throttled.append(name)
            record_search(name, "throttled")
            logger.info("search backend %s throttled for %r: %s", name, query, exc)
            continue
        except Exception as exc:  # noqa: BLE001 — move on to the next backend
            record_search(name, "error")
            logger.warning("search backend %s failed for %r: %s", name, query, exc)
            continue
        deduped = _dedupe_links(rows, max_results)
        if deduped:
            record_search(name, "success")
            logger.info("search_links: %d links from %s", len(deduped), name)
            return deduped
        # Backend answered without error but returned no usable links.
        record_search(name, "empty")
        # A backend that answered without error but with NO usable links used to
        # fall through to the next one silently — so a chronically-empty primary
        # (e.g. searxng returning 200+empty under its own rate limiter) looked
        # like "yahoo only" in the logs.  Surface the empty pass for the primary
        # backends so the fallthrough is visible, not silent.
        if name in _PRIMARY_SEARCH_BACKENDS:
            logger.info(
                "search_links: %s returned 0 usable links for %r — falling through",
                name, query,
            )
    # Nothing worked.  Make the degradation LOUD — an empty web-research result
    # silently weakens the downstream analyst reports, so surface WHY (all
    # rate-limited vs. all genuinely empty) at WARNING level for the operator.
    if throttled:
        logger.warning(
            "search_links: NO results for %r — every backend exhausted; "
            "rate-limited: [%s] (order: %s). Increase gap / add SEARXNG_URLS.",
            query, ",".join(throttled), ",".join(order),
        )
    else:
        logger.warning(
            "search_links: NO results for %r — all backends returned empty "
            "(order: %s).", query, ",".join(order),
        )
    return []


# ---------------------------------------------------------------------------
# Stage 2 — crawl the raw page
# ---------------------------------------------------------------------------


def _table_to_markdown(table: list) -> str:
    """Render a pdfplumber table (list-of-rows) as a pipe-delimited markdown
    table so the LLM sees the tabular structure (columns aligned) rather than a
    de-gridded run of numbers — critical for financial statements where the
    row label + period columns carry the meaning."""
    rows = []
    for row in table or []:
        cells = ["" if c is None else re.sub(r"\s+", " ", str(c)).strip() for c in row]
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
    for r in rows[1:]:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def extract_pdf_text(
    content: bytes,
    max_chars: Optional[int] = 8000,
    *,
    extract_tables_pages: Optional[int] = None,
) -> Optional[str]:
    """Extract text AND tables from a PDF's bytes via pdfplumber.

    Every page is read. Tables are rendered as markdown so quarterly
    financials / guidance tables keep their structure. ``max_chars`` of
    ``None`` means the whole document (no head slice). ``extract_tables_pages``
    of ``None`` means tables on every page. Best-effort: returns ``None``
    on failure so the caller degrades gracefully.
    """
    try:
        import io

        import pdfplumber
    except Exception as exc:  # noqa: BLE001 — parser missing == degrade
        logger.warning("pdf parse unavailable (pdfplumber import failed): %s", exc)
        return None
    parts: list[str] = []
    total = 0
    table_limit = None if extract_tables_pages is None else max(0, int(extract_tables_pages))
    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            n_pages = len(pdf.pages)
            pages_done = 0
            for page_idx, page in enumerate(pdf.pages):
                pages_done = page_idx + 1
                try:
                    txt = page.extract_text() or ""
                except Exception:  # noqa: BLE001 — a bad page must not kill the doc
                    txt = ""
                if txt:
                    parts.append(txt)
                    total += len(txt)
                if table_limit is None or page_idx < table_limit:
                    try:
                        for table in page.extract_tables():
                            md = _table_to_markdown(table)
                            if md:
                                parts.append(md)
                                total += len(md)
                    except Exception:  # noqa: BLE001
                        pass
                if max_chars is not None and total >= max_chars:
                    break
            logger.info(
                "pdf extract: %d/%d pages, %d chars",
                pages_done, n_pages, total,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("pdf parse failed: %s", exc)
        return None
    text = re.sub(r"\n{3,}", "\n\n", "\n\n".join(parts)).strip()
    if not text:
        return None
    if max_chars is None:
        return text
    return text[:max_chars]


def _looks_like_pdf(url: str, content: bytes, content_type: str) -> bool:
    if "application/pdf" in (content_type or "").lower():
        return True
    if (content or b"")[:5] == b"%PDF-":
        return True
    return url.lower().split("?", 1)[0].endswith(".pdf")


def _fetch_url_bytes(url: str, *, timeout: int = 25) -> "tuple[bytes, str]":
    """GET ``url`` and return ``(raw_bytes, content_type)``.

    A single seam so ``_crawl_http`` can inspect the real content type / magic
    bytes and route PDFs to the local parser (BeautifulSoup on decoded text
    can't tell a PDF from HTML, and pdfplumber needs the raw bytes)."""
    resp = requests.get(
        url,
        headers={"User-Agent": _UA, "Accept-Language": "en-US,en;q=0.9"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.content, resp.headers.get("Content-Type", "")


def _crawl_http(
    url: str,
    max_chars: Optional[int],
    *,
    extract_tables_pages: Optional[int] = None,
    timeout: int = 25,
) -> Optional[str]:
    """Keyless crawler: fetch bytes once, then route by type — PDFs go through
    the local pdfplumber parser (text + tables), everything else is tag-stripped
    HTML via BeautifulSoup.  A PDF used to be silently mangled by the HTML path
    (``requests`` ``.text`` on binary → BeautifulSoup → junk); routing on the
    real content type fixes screener / BSE / NSE filing PDFs."""
    try:
        content, ctype = _fetch_url_bytes(url, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        logger.warning("http crawl failed for %s: %s", url, exc)
        return None
    if _looks_like_pdf(url, content, ctype):
        return extract_pdf_text(
            content,
            max_chars=max_chars,
            extract_tables_pages=extract_tables_pages,
        )
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "form", "noscript"]):
            tag.decompose()
        text = re.sub(r"\n{3,}", "\n\n", soup.get_text("\n", strip=True))
        if not text:
            return None
        if max_chars is None:
            return text
        return text[:max_chars]
    except Exception as exc:  # noqa: BLE001
        logger.warning("html parse failed for %s: %s", url, exc)
        return None


def crawl(
    url: str,
    *,
    crawler: str = "auto",
    max_chars: Optional[int] = 8000,
    extract_tables_pages: Optional[int] = None,
    timeout: int = 25,
) -> Optional[str]:
    """Fetch the main content of ``url`` as text.

    ``crawler``: ``"firecrawl"`` (raw markdown via ``/v1/scrape``),
    ``"http"`` (keyless requests + local parsers), or ``"auto"`` (Firecrawl
    first when enabled, then the keyless path).  Filing PDFs must use
    ``http`` with ``max_chars=None`` so the whole document is parsed
    locally.  Returns ``None`` when nothing could be fetched.
    """
    from trading_crew.data_sources import news_providers

    if crawler in ("firecrawl", "auto"):
        md = news_providers.firecrawl_scrape(
            url, max_chars=8000 if max_chars is None else max_chars
        )
        if md:
            return md
        if crawler == "firecrawl":
            return None
    return _crawl_http(
        url,
        max_chars,
        extract_tables_pages=extract_tables_pages,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Stage 3 — LLM summarize
# ---------------------------------------------------------------------------


_SUMMARY_PROMPT = """You are a financial news analyst. You are given raw web sources crawled \
for the query: "{query}".

Some sources are genuine recent news; others may be generic pages (a company's \
own homepage, Wikipedia, product / definition pages, dictionaries, navigation \
menus) that got crawled because the real news site was bot-blocked.

Rules:
- FIRST judge relevance per source: does it actually report recent news about \
the query topic (dates, earnings / guidance, analyst actions, macro events, \
price moves, sentiment)?
- Use ONLY facts present in RELEVANT sources. Do not invent anything.
- IGNORE generic / irrelevant pages (company homepage, Wikipedia, product pages, \
dictionaries, nav menus) — do NOT summarise their content.
- Prefer the MOST RECENT events (last ~7 days); state the date / period of each.
- Attribute every concrete claim with an inline [source: <URL>] tag using the source's URL.
- Prefer specifics (numbers, dates, named events) over generic statements.
- 5-10 bullet points, most material first.
- If NO source contains relevant recent news, say so plainly \
("No relevant recent news found in the provided sources.") — never pad with generic filler.

SOURCES:
{sources}
"""


def summarize(query: str, docs: List[dict], *, llm=None, max_chars: int = 6000) -> str:
    """LLM-consolidate crawled ``docs`` ([{title,url,content}]) into a digest.

    ``llm`` must expose ``.call(messages=[...])`` (a CrewAI LLM).  When None,
    the workflow's default ``get_llm()`` is used.  Returns "" when there is
    nothing to summarize.
    """
    usable = [d for d in docs if (d.get("content") or "").strip()]
    if not usable:
        return ""
    if llm is None:
        from trading_crew._common import get_llm

        llm = get_llm(temperature=0.2)
    blocks = []
    budget = max_chars
    for d in usable:
        content = (d.get("content") or "").strip()
        take = content[: min(len(content), max(500, budget // len(usable)))]
        blocks.append(f"URL: {d.get('url')}\nTITLE: {d.get('title')}\nCONTENT:\n{take}")
    prompt = _SUMMARY_PROMPT.format(query=query, sources="\n\n---\n\n".join(blocks))
    # PIT: instruct the model to ignore anything dated after the cutoff so a
    # crawled page that was updated post-decision can't leak the future.
    try:
        from trading_crew.data_sources import pit

        cutoff = pit.cutoff()
    except Exception:  # noqa: BLE001
        cutoff = None
    if cutoff is not None:
        as_of = cutoff.strftime("%Y-%m-%d")
        prompt += (
            f"\n\nPOINT-IN-TIME CONSTRAINT: Treat 'today' as {as_of}. IGNORE and do "
            f"NOT report anything that happened, was announced, or was published "
            f"after {as_of}, even if it appears in the sources."
        )
    try:
        raw = llm.call(messages=[{"role": "user", "content": prompt}])
    except Exception as exc:  # noqa: BLE001
        logger.warning("summarizer LLM call failed: %s", exc)
        return ""
    return (raw or "").strip()


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def research_and_summarize(
    query: str,
    *,
    max_links: int = 5,
    crawler: str = "auto",
    llm=None,
    backends: Optional[List[str]] = None,
    max_doc_chars: int = 8000,
) -> dict:
    """Run search → crawl → summarize and return a structured result.

    Returns ``{"query", "summary", "sources": [{title,url}], "n_crawled"}``.
    Fail-soft: an empty result simply means no backend produced anything.

    Coverage: bot-protected news sites (e.g. business-standard) return 403 to
    the keyless HTTP crawler, which would otherwise shrink the source set.  We
    therefore OVER-FETCH search candidates and crawl down the list until we
    have ``max_links`` *successful* crawls (or run out) — so a couple of
    blocked pages cost nothing (the free search already surfaced more links
    than we need).

    When ``GEMINI_WEB_SEARCH=1``, a grounded Gemini answer is used as the
    briefing (no crawl) and SearXNG is not called. Off, empty, or error
    falls through to the SearXNG search → crawl → summarize path.
    """
    if gemini_web_search_enabled():
        try:
            gem = _gemini_research(query)
        except Exception as exc:  # noqa: BLE001 — SearXNG is the configured backup
            logger.warning(
                "gemini research failed for %r — falling through to SearXNG: %s",
                query, exc,
            )
            gem = None
        if gem and (gem.get("summary") or "").strip():
            sources = gem.get("sources") or []
            return {
                "query": query,
                "summary": gem["summary"],
                "sources": sources,
                "n_crawled": len(sources),
                "status": "ok",
                "backend": "gemini",
            }
        logger.info(
            "research_and_summarize: gemini empty for %r — SearXNG next", query
        )
    candidates = search_links(
        query,
        max_results=max(max_links * 2, max_links + 3),
        backends=backends,
        skip_gemini=gemini_web_search_enabled(),
    )
    docs: List[dict] = []
    for l in candidates:
        if len(docs) >= max_links:
            break
        content = crawl(l["url"], crawler=crawler, max_chars=max_doc_chars)
        if content:
            docs.append({"title": l["title"], "url": l["url"], "content": content})
    summary = summarize(query, docs, llm=llm) if docs else ""
    # Classify the outcome so callers can distinguish a search-infra failure
    # (all backends throttled/empty) from a crawl-only failure (links found but
    # every page blocked the keyless crawler) instead of both looking identical.
    if not candidates:
        status = "no_search_results"
        logger.warning(
            "research_and_summarize: no search links for %r — search backends "
            "returned nothing (see search_links warning above).", query,
        )
    elif not docs:
        status = "crawl_blocked"
        logger.warning(
            "research_and_summarize: %d link(s) for %r but every crawl "
            "failed/was blocked — no usable content.", len(candidates), query,
        )
    else:
        status = "ok"
    return {
        "query": query,
        "summary": summary,
        "sources": [{"title": d["title"], "url": d["url"]} for d in docs],
        "n_crawled": len(docs),
        "status": status,
    }
