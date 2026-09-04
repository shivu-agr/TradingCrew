"""News-provider key rotation + Firecrawl search (Phase 1).

The Tavily and SerpAPI row-builders live in ``trading_crew.tools`` (that's
where their parsing + the placeholder/dedup pipeline is, and where the
existing test-suite monkeypatches them).  This module adds the two things
that were missing:

1. **Multi-key rotation.**  Each provider may hold several legitimately
   obtained keys, given as a comma-separated ``*_API_KEYS`` env var (or the
   singular ``*_API_KEY``).  :func:`next_key` round-robins across them so
   quota is spread, reading env *fresh* each call so a key added at runtime
   (or in a test) takes effect immediately.
2. **Firecrawl** as a third web-search provider (``/v1/search``).

Point-in-time filtering is applied by the caller via
``trading_crew.data_sources.pit`` — provider rows simply carry a
``published`` field when the vendor exposes one.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

_PROVIDER_ENV = {
    "tavily": ("TAVILY_API_KEYS", "TAVILY_API_KEY"),
    "serpapi": ("SERPAPI_API_KEYS", "SERPAPI_API_KEY"),
    "firecrawl": ("FIRECRAWL_API_KEYS", "FIRECRAWL_API_KEY"),
}

# Persistent round-robin cursor per provider (env is read fresh each call,
# so only the rotation position is stateful).
_INDEX: dict[str, int] = {}
_INDEX_LOCK = threading.Lock()


def read_keys(provider: str) -> List[str]:
    """Return the configured keys for ``provider`` (plural env wins).

    Reads env on every call so runtime/test changes are picked up.  Blanks
    and duplicates are dropped while order is preserved.
    """
    plural, singular = _PROVIDER_ENV[provider]
    raw = os.environ.get(plural, "").strip() or os.environ.get(singular, "").strip()
    keys: List[str] = []
    for part in raw.split(","):
        k = part.strip()
        if k and k not in keys:
            keys.append(k)
    return keys


def next_key(provider: str) -> Optional[str]:
    """Round-robin the next key for ``provider`` (``None`` if none set)."""
    keys = read_keys(provider)
    if not keys:
        return None
    with _INDEX_LOCK:
        i = _INDEX.get(provider, 0)
        _INDEX[provider] = (i + 1) % len(keys)
    return keys[i % len(keys)]


def ordered_keys(provider: str) -> List[str]:
    """Return all keys for ``provider`` starting at the next rotated slot.

    This lets a caller *try every key* in round-robin order: the first
    entry is the freshly-rotated key (so quota spreads across calls), and
    the remainder are the fallbacks to try when a key is rate-limited /
    invalid.  Empty when no key is configured.
    """
    keys = read_keys(provider)
    if not keys:
        return []
    start = next_key(provider)
    if start is None:
        return keys
    return [start] + [k for k in keys if k != start]


def has_keys(provider: str) -> bool:
    return bool(read_keys(provider))


def firecrawl_enabled() -> bool:
    """Whether the paid Firecrawl crawler/search is allowed to run.

    Default OFF: Firecrawl is a paid API (``/v1/search`` + ``/v1/scrape``)
    and an exhausted account answers every call with ``402 Payment
    Required`` — pure log noise + wasted latency.  Set
    ``TRADINGCREW_ENABLE_FIRECRAWL=1`` to opt back in once the account has
    credits.  When off, callers behave as if no Firecrawl key is set (the
    keyless HTTP crawler + free link search still run).
    """
    return os.getenv("TRADINGCREW_ENABLE_FIRECRAWL", "0").strip().lower() in (
        "1", "true", "yes", "on"
    )


def reset_rotation() -> None:
    """Test hook — reset the round-robin cursors (keys re-read from env)."""
    with _INDEX_LOCK:
        _INDEX.clear()


def firecrawl_rows(query: str, max_results: int = 8) -> list:
    """Search the web via Firecrawl ``/v1/search`` with key rotation.

    Returns normalized rows (``title/url/snippet/provider/published``).
    Firecrawl often omits a publish date, so in strict backfill those rows
    are dropped by the PIT guard downstream.  Returns ``[]`` on any error
    (the news merge must survive a dead provider).
    """
    if not firecrawl_enabled():
        return []
    ordered = ordered_keys("firecrawl")
    if not ordered:
        return []
    for key in ordered:
        try:
            resp = requests.post(
                "https://api.firecrawl.dev/v1/search",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"query": query, "limit": max_results},
                timeout=25,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001 — try the next key, then give up
            logger.warning("firecrawl search failed (key …%s): %s", key[-4:], exc)
            continue
        rows = []
        for r in (data.get("data") or []):
            meta = r.get("metadata") or {}
            url = (r.get("url") or meta.get("sourceURL") or "").strip()
            title = (r.get("title") or meta.get("title") or "").strip()
            if not url or not title:
                continue
            desc = (r.get("description") or meta.get("description") or "").strip().replace("\n", " ")[:240]
            published = (
                meta.get("publishedTime")
                or meta.get("published_time")
                or meta.get("article:published_time")
                or meta.get("date")
            )
            rows.append({"title": title, "url": url, "snippet": desc, "provider": "firecrawl", "published": published})
        return rows
    return []


def firecrawl_scrape(url: str, *, timeout: int = 40, max_chars: int = 8000) -> Optional[str]:
    """Crawl a single ``url`` via Firecrawl ``/v1/scrape`` → markdown text.

    Returns the page content as markdown (truncated to ``max_chars``) or
    ``None`` on failure / when no Firecrawl key is set.  Key rotation: the
    rotated key is tried first, then the rest on error.  This is the "raw
    response" crawler whose output the LLM summarizer consolidates.
    """
    if not firecrawl_enabled():
        return None
    ordered = ordered_keys("firecrawl")
    if not ordered or not (url or "").startswith("http"):
        return None
    for key in ordered:
        try:
            resp = requests.post(
                "https://api.firecrawl.dev/v1/scrape",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"url": url, "formats": ["markdown"], "onlyMainContent": True},
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001 — try the next key, then give up
            logger.warning("firecrawl scrape failed (key …%s) for %s: %s", key[-4:], url, exc)
            continue
        payload = data.get("data") or data
        md = payload.get("markdown") or payload.get("content") or ""
        md = (md or "").strip()
        if md:
            return md[:max_chars]
        return None
    return None
