"""screener.in client — Ind-AS fundamentals + shareholding + announcements.

screener.in has no official public API; we scrape the company page HTML
directly.  The scraping surface is small and well-scoped: a single URL
(``/company/{symbol}/consolidated/`` with a ``/company/{symbol}/``
standalone fallback) whose section IDs (``#top``, ``#quarters``,
``#profit-loss``, ``#balance-sheet``, ``#cash-flow``, ``#ratios``,
``#shareholding``, ``#analysis``, ``#documents``) have been stable for
years.  Live-probed on 2026-07-04 against RELIANCE / TCS / BAJAJ-AUTO
and confirmed layout.

Auth
----
screener.in serves company pages anonymously.  A logged-in session
lifts anonymous rate limits — pass ``SCREENER_EMAIL`` /
``SCREENER_PASSWORD`` env vars via the routing config to enable it.
Without credentials the client just runs anonymously; a 429 propagates
to the caller as a ``ScreenerRateLimitError`` (no silent yfinance
fallback, per project rule).

Caching
-------
Every fetch goes through a small in-process TTL cache
(default 15 min).  ``get_market_context`` and the fundamentals /
balance-sheet / cash-flow / income-statement tools all share this
cache so five analysts asking for the same ticker only cost one HTTP
round-trip.
"""

from __future__ import annotations

import logging
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup, Tag

from .routing import get_config, strip_exchange_suffix

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Polite rate-limiting — space out requests to screener.in so we stay a
# well-behaved anonymous/logged-in client instead of hammering the site
# (and getting a 429 / IP ban).  Configurable via env; jittered so bursts
# from parallel analyst threads don't align.
# ---------------------------------------------------------------------------

_THROTTLE_LOCK = threading.Lock()
_LAST_REQUEST_AT: float = 0.0


def _min_request_gap() -> float:
    try:
        return max(0.0, float(os.environ.get("SCREENER_MIN_REQUEST_GAP_SECONDS", "1.5")))
    except (TypeError, ValueError):
        return 1.5


def _throttle() -> None:
    """Block until at least ``SCREENER_MIN_REQUEST_GAP_SECONDS`` (plus a small
    random jitter) has elapsed since the previous request."""
    global _LAST_REQUEST_AT
    gap = _min_request_gap()
    if gap <= 0:
        return
    with _THROTTLE_LOCK:
        now = time.monotonic()
        wait = (_LAST_REQUEST_AT + gap) - now
        if wait > 0:
            time.sleep(wait + random.uniform(0.0, gap * 0.25))
        _LAST_REQUEST_AT = time.monotonic()


# ---------------------------------------------------------------------------
# Public error surface
# ---------------------------------------------------------------------------


class ScreenerError(Exception):
    """Base class for screener.in client errors.

    Deliberately a plain ``Exception`` subclass so callers can catch it
    without pulling the class in — the tool layer generally logs a
    warning and emits a "screener.in returned <status>" line back to
    the LLM instead of retrying.  There is *no* silent yfinance
    fallback (project rule).
    """


class ScreenerRateLimitError(ScreenerError):
    """screener.in returned HTTP 429 or a login-required 403."""


class ScreenerNotFoundError(ScreenerError):
    """Both the consolidated and standalone company pages returned 404."""


# ---------------------------------------------------------------------------
# Data model — deliberately conservative so the LLM sees clean markdown
# ---------------------------------------------------------------------------


@dataclass
class TopRatios:
    """Header-strip ratios: market cap, P/E, book value, ROCE, ROE, etc.

    Keys are the label strings screener.in displays verbatim (e.g.
    ``"Stock P/E"``, ``"Book Value"``, ``"Dividend Yield"``).  Values
    keep the raw display string (``"₹ 17,64,643 Cr."``) so downstream
    formatters don't have to re-format for the LLM.  Numeric parsing
    would be handy for a programmatic backtest but hurts LLM
    readability, so we keep both worlds: raw display + a parsed
    ``numeric`` companion when the value is unambiguously numeric.
    """

    raw: Dict[str, str] = field(default_factory=dict)
    numeric: Dict[str, float] = field(default_factory=dict)


@dataclass
class FinancialTable:
    """A single financial statement (quarters, P&L, BS, CF, ratios).

    ``columns`` is the header row (period labels).  ``rows`` is a list
    of ``(metric_name, values)`` tuples where ``values`` are aligned
    with ``columns``.  Keeping the row order preserves the on-page
    ordering — the LLM sees "Sales / Expenses / Operating Profit /
    OPM / …" in the natural sequence.
    """

    kind: str  # e.g. "profit-loss"
    columns: List[str]
    rows: List[tuple]  # list[tuple[str, list[str]]]


@dataclass
class ShareholdingPattern:
    """Quarterly + yearly promoter / FII / DII / public holding split."""

    quarterly: Optional[FinancialTable] = None
    yearly: Optional[FinancialTable] = None


@dataclass
class DocumentLink:
    """One filing / announcement / concall / rating link.

    ``kind`` categorises the dated primary source so the tool layer can
    group the feed (announcements vs annual reports vs concall
    transcripts vs credit-rating actions).  These dated documents are the
    point-in-time-safe part of screener.in (unlike the restated ratio
    snapshot), which makes them the backbone of offline backfill.
    """

    title: str
    url: str
    # screener.in embeds the date inside the title (``"... 15h"`` or
    # ``"... 24 Jun"``) — we surface it as ``title`` verbatim; a stricter
    # parser could split it but the LLM handles the freeform date fine.
    kind: str = "announcement"  # announcement | annual_report | concall | credit_rating


@dataclass
class ProsCons:
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)


@dataclass
class CompanyPage:
    """Full parsed snapshot of one screener.in company page."""

    symbol: str          # unsuffixed (screener native), e.g. "RELIANCE"
    source_url: str      # the URL we actually fetched (consolidated or standalone)
    mode: str            # "consolidated" or "standalone"
    long_name: str
    about: str
    top: TopRatios
    quarters: Optional[FinancialTable] = None
    profit_loss: Optional[FinancialTable] = None
    balance_sheet: Optional[FinancialTable] = None
    cash_flow: Optional[FinancialTable] = None
    ratios: Optional[FinancialTable] = None
    shareholding: Optional[ShareholdingPattern] = None
    documents: List[DocumentLink] = field(default_factory=list)
    pros_cons: ProsCons = field(default_factory=ProsCons)


# ---------------------------------------------------------------------------
# HTTP session with retry / session-login
# ---------------------------------------------------------------------------


_SESSION_LOCK = threading.Lock()
_SESSION: Optional[requests.Session] = None
_SESSION_LOGGED_IN = False


def _default_headers() -> Dict[str, str]:
    return {
        # A generic but honest UA — screener.in doesn't require a
        # browser-lookalike UA, but a curl/requests default gets rate
        # limited faster than a realistic desktop UA.
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) tradingcrew/1.0 "
            "(+https://github.com/anthropics/experiments)"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-IN,en;q=0.9",
    }


def _get_session() -> requests.Session:
    """Return a process-global ``requests.Session``.

    We keep a single session so cookies (including the optional login
    session cookie) survive across calls.  Locked construction so
    parallel analyst threads don't each create their own session.
    """
    global _SESSION
    with _SESSION_LOCK:
        if _SESSION is None:
            s = requests.Session()
            s.headers.update(_default_headers())
            _SESSION = s
        return _SESSION


def _maybe_login(session: requests.Session) -> None:
    """Log the session in if credentials are configured and we're not
    already logged in.

    The login is best-effort: a failure here surfaces as a warning and
    the session continues as anonymous.  We deliberately do NOT raise
    on a failed login because anonymous access is the primary path;
    the credentials are just an optional rate-limit lift.
    """
    global _SESSION_LOGGED_IN
    if _SESSION_LOGGED_IN:
        return
    cfg = get_config()
    if not (cfg.screener_email and cfg.screener_password):
        return
    try:
        # The screener.in login flow is a standard Django CSRF-protected
        # POST.  We fetch the login page to pick up the CSRF token,
        # then POST the credentials.
        r0 = session.get("https://www.screener.in/login/", timeout=15)
        csrf = None
        soup = BeautifulSoup(r0.text, "html.parser")
        tag = soup.select_one("input[name=csrfmiddlewaretoken]")
        if tag:
            csrf = tag.get("value")
        if not csrf:
            logger.warning("screener.in: no CSRF token on login page; running anonymously")
            return
        r1 = session.post(
            "https://www.screener.in/login/",
            data={
                "csrfmiddlewaretoken": csrf,
                "username": cfg.screener_email,
                "password": cfg.screener_password,
                "next": "/",
            },
            headers={"Referer": "https://www.screener.in/login/"},
            timeout=15,
            allow_redirects=True,
        )
        if "sessionid" in session.cookies:
            _SESSION_LOGGED_IN = True
            logger.info("screener.in: logged in as %s", cfg.screener_email)
        else:
            logger.warning(
                "screener.in: login POST returned %s but no sessionid cookie; "
                "running anonymously",
                r1.status_code,
            )
    except Exception as exc:
        logger.warning("screener.in: login attempt failed (%s); running anonymously", exc)


# ---------------------------------------------------------------------------
# TTL cache for parsed pages
# ---------------------------------------------------------------------------


_CACHE_LOCK = threading.Lock()
_CACHE: Dict[str, tuple] = {}  # symbol -> (parsed_page, fetched_at)
_CACHE_TTL_SECONDS = 15 * 60
# Bound the parsed-page cache.  Keyed by symbol so it's naturally capped by the
# universe size, but a long-running process would still keep every expired
# parsed page (financial tables + doc links) resident until the same symbol is
# re-read.  Purge-on-write frees them and hard-caps total entries.
_CACHE_MAX_ENTRIES = 256


def _cache_get(key: str):
    with _CACHE_LOCK:
        entry = _CACHE.get(key)
        if not entry:
            return None
        parsed, fetched_at = entry
        if time.time() - fetched_at > _CACHE_TTL_SECONDS:
            _CACHE.pop(key, None)
            return None
        return parsed


def _cache_put(key: str, parsed) -> None:
    now = time.time()
    with _CACHE_LOCK:
        _CACHE[key] = (parsed, now)
        expired = [k for k, (_, ts) in _CACHE.items() if now - ts > _CACHE_TTL_SECONDS]
        for k in expired:
            _CACHE.pop(k, None)
        if len(_CACHE) > _CACHE_MAX_ENTRIES:
            for k in list(_CACHE.keys())[: len(_CACHE) - _CACHE_MAX_ENTRIES]:
                _CACHE.pop(k, None)


def clear_cache() -> None:
    """Test hook — drop every cached page."""
    global _SESSION_LOGGED_IN
    with _CACHE_LOCK:
        _CACHE.clear()
    _SESSION_LOGGED_IN = False


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _parse_number(text: str) -> Optional[float]:
    """Parse a screener.in-formatted number string into a float.

    Handles: ``"₹ 17,64,643 Cr."``, ``"22.7"``, ``"8.91 %"``, ``"-3.2"``.
    Rejects anything without at least one digit.  Returns ``None`` if
    the string is fundamentally non-numeric (e.g. ``"..."``, ``"n/a"``).
    """
    if not text:
        return None
    m = _NUMBER_RE.search(text.replace("₹", ""))
    if not m:
        return None
    raw = m.group(0).replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def parse_top_ratios(top: Optional[Tag]) -> TopRatios:
    """Parse the header ratios strip (``section#top``) into a TopRatios."""
    ratios = TopRatios()
    if top is None:
        return ratios
    # screener.in exposes each ratio as ``<li><span class="name">…</span>
    # <span class="value">…</span></li>`` inside the top ul.  Older
    # revisions used raw text so we fall back to a generic li scrape.
    for li in top.select("ul li"):
        name_el = li.select_one(".name")
        value_el = li.select_one(".value")
        if name_el and value_el:
            name = name_el.get_text(" ", strip=True)
            value = value_el.get_text(" ", strip=True)
        else:
            # ``li.get_text()`` gives e.g. ``"Market Cap ₹ 17,64,643 Cr."``;
            # split on the first whitespace-cluster after a label word
            # that ends in a letter / ``)``.
            raw = li.get_text(" ", strip=True)
            parts = raw.rsplit(" ", 1)
            if len(parts) != 2:
                continue
            name, value = parts
        ratios.raw[name] = value
        num = _parse_number(value)
        if num is not None:
            ratios.numeric[name] = num
    return ratios


def _parse_financial_table(section: Optional[Tag], kind: str) -> Optional[FinancialTable]:
    """Parse the first ``table.data-table`` inside ``section`` into a
    ``FinancialTable``.

    Rows are yielded in on-page order; empty cells become empty strings
    (not ``None``) so the LLM sees a stable grid width.
    """
    if section is None:
        return None
    table = section.select_one("table.data-table") or section.select_one("table")
    if table is None:
        return None
    header_cells = [c.get_text(" ", strip=True) for c in table.select("thead th")]
    if not header_cells:
        # Some tables put the header in the first tr without a thead.
        rows = table.select("tr")
        if not rows:
            return None
        header_cells = [c.get_text(" ", strip=True) for c in rows[0].select("th, td")]
        body_rows = rows[1:]
    else:
        body_rows = table.select("tbody tr")
    # First column is the metric label; the rest are period columns.
    columns = header_cells[1:] if header_cells else []
    parsed_rows: List[tuple] = []
    for tr in body_rows:
        cells = [c.get_text(" ", strip=True) for c in tr.select("td, th")]
        if not cells:
            continue
        metric = cells[0]
        values = cells[1:]
        # Right-pad to ``len(columns)`` so rendered markdown stays
        # rectangular even when a metric is missing a period column.
        if len(values) < len(columns):
            values = values + [""] * (len(columns) - len(values))
        parsed_rows.append((metric, values))
    return FinancialTable(kind=kind, columns=columns, rows=parsed_rows)


def parse_shareholding(section: Optional[Tag]) -> Optional[ShareholdingPattern]:
    """Parse the shareholding section into quarterly + yearly tables."""
    if section is None:
        return None
    tables = section.select("table.data-table") or section.select("table")
    if not tables:
        return None
    # Two tables: quarterly (first) then yearly (second) — screener.in's
    # convention.  If only one is present we treat it as quarterly since
    # that's the more informative view for a trading desk.
    quarterly = _parse_financial_table_from_element(tables[0], "shareholding-quarterly")
    yearly = None
    if len(tables) >= 2:
        yearly = _parse_financial_table_from_element(tables[1], "shareholding-yearly")
    return ShareholdingPattern(quarterly=quarterly, yearly=yearly)


def _parse_financial_table_from_element(table: Tag, kind: str) -> Optional[FinancialTable]:
    """Parse a ``<table>`` element directly (helper for shareholding)."""
    header_cells = [c.get_text(" ", strip=True) for c in table.select("thead th")]
    if not header_cells:
        rows = table.select("tr")
        if not rows:
            return None
        header_cells = [c.get_text(" ", strip=True) for c in rows[0].select("th, td")]
        body_rows = rows[1:]
    else:
        body_rows = table.select("tbody tr")
    columns = header_cells[1:] if header_cells else []
    parsed_rows: List[tuple] = []
    for tr in body_rows:
        cells = [c.get_text(" ", strip=True) for c in tr.select("td, th")]
        if not cells:
            continue
        metric = cells[0]
        values = cells[1:]
        if len(values) < len(columns):
            values = values + [""] * (len(columns) - len(values))
        parsed_rows.append((metric, values))
    return FinancialTable(kind=kind, columns=columns, rows=parsed_rows)


def parse_documents(section: Optional[Tag]) -> List[DocumentLink]:
    """Parse the announcements/documents section into a list of links.

    We keep the raw anchor text as ``title`` — screener.in embeds the
    filing date directly in the anchor text ("... 15h", "... 24 Jun"),
    which the LLM handles fine without a dedicated date parser.

    The very first anchor on the page is typically the "All"
    aggregation link back to BSE — we drop it because it isn't a
    filing itself.  Each link is categorised (``announcement`` /
    ``annual_report`` / ``concall`` / ``credit_rating``) so the feed can
    be grouped downstream.
    """
    if section is None:
        return []
    out: List[DocumentLink] = []
    for a in section.select("a"):
        title = a.get_text(" ", strip=True)
        url = a.get("href", "").strip()
        if not (title and url and url.startswith("http")):
            continue
        if title.strip().lower() == "all":
            continue
        out.append(DocumentLink(title=title, url=url, kind=classify_document(title, url)))
    return out


def classify_document(title: str, url: str) -> str:
    """Categorise a screener.in document link by title/URL heuristics.

    Returns one of ``announcement`` / ``annual_report`` / ``concall`` /
    ``credit_rating``.  Deliberately conservative — anything unrecognised
    stays a generic ``announcement``.
    """
    hay = f"{title} {url}".lower()
    if any(k in hay for k in ("concall", "transcript", "conference call", "earnings call", "investor ppt", "investor presentation", "/concalls/")):
        return "concall"
    if any(k in hay for k in ("annual report", "annual-report", "/annual-reports/", "integrated report")):
        return "annual_report"
    if any(k in hay for k in ("credit rating", "rating", "crisil", "icra", "care ratings", "india ratings", "fitch", "brickwork")):
        return "credit_rating"
    return "announcement"


_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_REL_DATE_RE = re.compile(r"\b\d+\s*(h|hr|hrs|hour|hours|d|day|days|w|wk|week|weeks|min|mins)\b", re.IGNORECASE)
_ABS_DATE_RE = re.compile(r"\b(\d{1,2})\s+([A-Za-z]{3})[a-z]*\s+(\d{4})\b")
_DM_DATE_RE = re.compile(r"\b(\d{1,2})\s+([A-Za-z]{3})[a-z]*\b")


def parse_document_date(title: str) -> Optional[datetime]:
    """Best-effort parse of the filing date screener embeds in a doc title.

    Handles ``"24 Jun 2024"`` (absolute) and treats relative markers
    (``"15h"``, ``"3d"``) as *now*.  A bare ``"24 Jun"`` (no year) is
    ambiguous, so we return ``None`` — the PIT filter then drops it in
    strict backfill mode rather than guess a year and risk a leak.
    """
    if not title:
        return None
    m = _ABS_DATE_RE.search(title)
    if m:
        day, mon, year = int(m.group(1)), _MONTHS.get(m.group(2).lower()), int(m.group(3))
        if mon:
            try:
                return datetime(year, mon, day, tzinfo=timezone.utc)
            except ValueError:
                return None
    if _REL_DATE_RE.search(title):
        # "15h" / "3d" → a very recent filing; anchor at now.
        return datetime.now(timezone.utc)
    return None


def pit_filter_documents(documents: List[DocumentLink]) -> List[DocumentLink]:
    """Drop filings published after the current PIT cutoff (backfill only).

    Live mode returns the list unchanged.  In strict backfill, filings
    whose date can't be parsed are also dropped (fail-closed).
    """
    try:
        from . import pit
    except Exception:  # noqa: BLE001
        return documents
    ctx = pit.get_context()
    if not ctx.is_backfill:
        return documents
    kept: List[DocumentLink] = []
    for d in documents:
        dt = parse_document_date(d.title)
        if dt is None:
            if not ctx.strict:
                kept.append(d)
            continue
        if dt <= ctx.as_of:
            kept.append(d)
    return kept


def parse_pros_cons(section: Optional[Tag]) -> ProsCons:
    """Parse the ``#analysis`` section into a pros / cons split.

    screener.in has two ``ul`` children in this section: the first is
    pros, the second cons.  Both are optional (small caps may have only
    cons, or vice versa).  We rely on class hints when available and
    fall back to positional interpretation otherwise.
    """
    result = ProsCons()
    if section is None:
        return result
    pros_ul = section.select_one("ul.pros") or None
    cons_ul = section.select_one("ul.cons") or None
    if pros_ul is None or cons_ul is None:
        # Positional fallback — the two direct-child ULs of #analysis
        uls = section.select("ul")
        if len(uls) >= 1 and pros_ul is None:
            pros_ul = uls[0]
        if len(uls) >= 2 and cons_ul is None:
            cons_ul = uls[1]
    if pros_ul is not None:
        result.pros = [li.get_text(" ", strip=True) for li in pros_ul.select("li")]
    if cons_ul is not None:
        result.cons = [li.get_text(" ", strip=True) for li in cons_ul.select("li")]
    return result


# ---------------------------------------------------------------------------
# High-level fetch
# ---------------------------------------------------------------------------


def _table_has_values(t: Optional[FinancialTable]) -> bool:
    """True when a financial table actually carries period data (not just row
    labels).  screener serves a label-only skeleton (no period columns, blank
    value cells) for a statement a company doesn't publish in the requested
    mode — e.g. the ``/consolidated/`` page of a name with no subsidiaries."""
    if t is None or not t.columns:
        return False
    for _metric, values in t.rows:
        if any((v not in (None, "")) for v in values):
            return True
    return False


def _page_has_financials(page: CompanyPage) -> bool:
    """A page is "real" if either its quarterly or annual P&L table is populated."""
    return _table_has_values(page.quarters) or _table_has_values(getattr(page, "profit_loss", None))


_SEARCH_URL = "https://www.screener.in/api/company/search/"
_SLUG_RE = re.compile(r"/company/([A-Za-z0-9.-]+)/")


def search_company_slug(query: str, session: Optional[requests.Session] = None) -> Optional[str]:
    """Resolve a ticker root to screener.in's listing slug via its search API.

    Yahoo / operator input can differ from the NSE listing code
    (``SBI.NS`` vs ``SBIN``). This is the same source, not another
    vendor. Empty / non-200 search returns None — the caller still
    raises ``ScreenerNotFoundError``.
    """
    q = (query or "").strip()
    if not q:
        return None
    sess = session or _get_session()
    _maybe_login(sess)
    try:
        from trading_crew.data_sources import rate_gateway

        if rate_gateway.is_enabled():
            rate_gateway.acquire("scrape")
        else:
            _throttle()
    except Exception:  # noqa: BLE001
        _throttle()
    try:
        r = sess.get(_SEARCH_URL, params={"q": q}, timeout=20)
    except requests.RequestException as exc:
        raise ScreenerError(f"screener.in search {q!r} failed: {exc}") from exc
    if r.status_code == 429:
        raise ScreenerRateLimitError(
            f"screener.in rate-limited ({r.status_code}) on search {q!r}"
        )
    if r.status_code != 200:
        logger.info("screener.in search %r returned HTTP %s", q, r.status_code)
        return None
    try:
        rows = r.json()
    except ValueError:
        return None
    if not isinstance(rows, list):
        return None
    q_up = q.upper()
    exact: List[str] = []
    prefix: List[str] = []
    name_hit: List[str] = []
    other: List[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        url = str(row.get("url") or "")
        match = _SLUG_RE.search(url)
        if not match:
            continue
        slug = match.group(1).upper()
        name = str(row.get("name") or "").upper()
        tokens = name.replace("-", " ").split()
        if slug == q_up:
            exact.append(slug)
        elif slug.startswith(q_up):
            prefix.append(slug)
        elif q_up in tokens:
            name_hit.append(slug)
        else:
            other.append(slug)
    if exact:
        return exact[0]
    if prefix:
        return sorted(prefix, key=len)[0]
    if name_hit:
        return name_hit[0]
    return other[0] if other else None


def _company_urls(symbol_root: str) -> List[tuple]:
    """Return ``(url, mode)`` candidates for a symbol root.

    We try ``/consolidated/`` first (better for group holding companies
    like RELIANCE, TCS, INFY), then ``/`` (standalone) as the fallback
    for names that don't publish a consolidated view.
    """
    root = symbol_root.strip().upper()
    return [
        (f"https://www.screener.in/company/{root}/consolidated/", "consolidated"),
        (f"https://www.screener.in/company/{root}/", "standalone"),
    ]


def _fetch_html(url: str, session: requests.Session) -> Optional[str]:
    """GET ``url`` and return the response text, or raise / return ``None``.

    ``None`` means "clean 404 — try the next fallback URL".
    ``ScreenerRateLimitError`` propagates when the server rate-limits
    us.  Any other HTTP or transport error is wrapped in a
    ``ScreenerError`` so the tool layer can log a single line and move
    on.

    A 429 triggers a short bounded exponential backoff (respecting a
    ``Retry-After`` header when present) before giving up.
    """
    max_attempts = 3
    for attempt in range(max_attempts):
        # Rate limiting: prefer the Kafka ``scrape`` slot queue when the web
        # server's gateway is up (centralised + visible); else the in-process
        # gap.  Never let the limiter itself break a scrape.
        try:
            from trading_crew.data_sources import rate_gateway

            if rate_gateway.is_enabled():
                rate_gateway.acquire("scrape")
            else:
                _throttle()
        except Exception:  # noqa: BLE001
            _throttle()
        try:
            r = session.get(url, timeout=20)
        except requests.RequestException as exc:
            raise ScreenerError(f"screener.in GET {url} failed: {exc}") from exc
        if r.status_code != 429:
            break
        # Rate-limited — back off and retry a couple of times before failing.
        if attempt < max_attempts - 1:
            retry_after = r.headers.get("Retry-After")
            try:
                sleep_s = float(retry_after) if retry_after else (2.0 ** attempt) * _min_request_gap()
            except (TypeError, ValueError):
                sleep_s = (2.0 ** attempt) * _min_request_gap()
            logger.warning("screener.in 429 on %s; backing off %.1fs (attempt %d)", url, sleep_s, attempt + 1)
            time.sleep(min(sleep_s, 30.0))
    if r.status_code == 404:
        return None
    if r.status_code == 429:
        raise ScreenerRateLimitError(
            f"screener.in rate-limited ({r.status_code}) on {url}; "
            "set SCREENER_EMAIL / SCREENER_PASSWORD to lift limits."
        )
    if r.status_code == 403:
        raise ScreenerRateLimitError(
            f"screener.in returned 403 on {url} — likely bot detection; "
            "set SCREENER_EMAIL / SCREENER_PASSWORD."
        )
    if r.status_code != 200:
        raise ScreenerError(
            f"screener.in GET {url} returned HTTP {r.status_code}"
        )
    return r.text


def fetch_company(symbol: str) -> CompanyPage:
    """Fetch + parse a company page for ``symbol``.

    ``symbol`` accepts either a suffixed yfinance ticker
    (``RELIANCE.NS``) or the unsuffixed screener root (``RELIANCE``);
    we normalise via ``strip_exchange_suffix``.

    Cached for 15 minutes per symbol root.  Cache is per-process and
    thread-safe.

    Raises ``ScreenerNotFoundError`` if neither the consolidated nor
    the standalone URL exists (404 on both).  Rate-limit / 5xx errors
    surface as ``ScreenerRateLimitError`` / ``ScreenerError``
    respectively — the tool layer converts these to a "screener.in
    returned <status>" line back to the LLM (no silent fallback).
    """
    root = strip_exchange_suffix(symbol)
    cached = _cache_get(root)
    if cached is not None:
        return cached

    # Point-in-time backfill: reuse a previously-captured on-disk snapshot
    # of the raw HTML so we scrape each company at most once across an
    # entire backfill run (huge request reduction + politeness).  Only
    # reused in backfill — live runs always fetch fresh (subject to the
    # 15-min in-process TTL cache above).
    try:
        from . import pit as _pit

        if _pit.is_backfill():
            snap = _pit.load_snapshot("screener_html", root)
            if snap and snap.get("html"):
                page = _parse_company_html(root, snap.get("url", ""), snap.get("mode", "consolidated"), snap["html"])
                _cache_put(root, page)
                return page
    except Exception:  # noqa: BLE001 — snapshot reuse is best-effort
        pass

    session = _get_session()
    _maybe_login(session)

    def _finalize(page: CompanyPage, html: str, url: str, mode: str) -> CompanyPage:
        _cache_put(root, page)
        # Persist a raw-HTML snapshot for later PIT reuse (best-effort).
        try:
            from . import pit as _pit

            _pit.save_snapshot("screener_html", root, {"html": html, "url": url, "mode": mode})
        except Exception:  # noqa: BLE001
            pass
        return page

    last_error: Optional[ScreenerError] = None

    def _collect(symbol_root: str) -> list[tuple]:
        nonlocal last_error
        found: list[tuple] = []
        for url, mode in _company_urls(symbol_root):
            try:
                html = _fetch_html(url, session)
            except ScreenerError as exc:
                last_error = exc
                raise
            if html is None:
                continue
            page = _parse_company_html(symbol_root, url, mode, html)
            found.append((page, html, url, mode))
            if _table_has_values(page.quarters):
                return found
            logger.info(
                "screener.in %s page for %s has an empty #quarters skeleton — "
                "trying next candidate URL for populated quarterly data",
                mode, symbol_root,
            )
        return found

    # Rank candidates by usefulness of their QUARTERLY table: a name with no
    # consolidated financials (the COLPAL case) serves a label-only #quarters
    # skeleton on /consolidated/ — yet its ANNUAL P&L is populated with stale
    # history, so a P&L-only check would wrongly keep it.  The computed trend
    # signals need quarters, so prefer the first candidate whose quarters are
    # populated; fetch the next URL (standalone) when they aren't.
    candidates = _collect(root)
    if candidates and _table_has_values(candidates[-1][0].quarters):
        return _finalize(*candidates[-1])

    if not candidates:
        slug = search_company_slug(root, session)
        if slug and slug != root:
            logger.info(
                "screener.in: %s has no /company/%s page; search resolved "
                "listing slug %s",
                root, root, slug,
            )
            candidates = _collect(slug)
            if candidates and _table_has_values(candidates[-1][0].quarters):
                return _finalize(*candidates[-1])

    # No candidate had quarterly data — pick the best of what we fetched:
    # prefer one whose annual P&L is populated, else the first parsed page (its
    # documents / about / ratios are still useful) rather than raising.
    if candidates:
        best = next((c for c in candidates if _page_has_financials(c[0])), candidates[0])
        return _finalize(*best)

    raise ScreenerNotFoundError(
        f"screener.in has no consolidated or standalone page for {root!r}."
    ) if last_error is None else last_error


def _parse_company_html(root: str, source_url: str, mode: str, html: str) -> CompanyPage:
    """Turn the raw HTML into a ``CompanyPage`` snapshot."""
    soup = BeautifulSoup(html, "html.parser")

    # Long name: the <h1> at the top of the page ("Reliance Industries Ltd").
    h1 = soup.select_one("h1")
    long_name = h1.get_text(" ", strip=True) if h1 else root

    # About: the description paragraph, if present.
    about_el = (
        soup.select_one("section#about p")
        or soup.select_one("div.company-profile p")
        or soup.select_one("section#top p")
    )
    about = about_el.get_text(" ", strip=True) if about_el else ""

    top = parse_top_ratios(soup.select_one("#top"))
    quarters = _parse_financial_table(soup.select_one("#quarters"), "quarters")
    profit_loss = _parse_financial_table(soup.select_one("#profit-loss"), "profit-loss")
    balance_sheet = _parse_financial_table(soup.select_one("#balance-sheet"), "balance-sheet")
    cash_flow = _parse_financial_table(soup.select_one("#cash-flow"), "cash-flow")
    ratios = _parse_financial_table(soup.select_one("#ratios"), "ratios")
    shareholding = parse_shareholding(soup.select_one("#shareholding"))
    documents = parse_documents(soup.select_one("#documents"))
    pros_cons = parse_pros_cons(soup.select_one("#analysis"))

    return CompanyPage(
        symbol=root,
        source_url=source_url,
        mode=mode,
        long_name=long_name,
        about=about,
        top=top,
        quarters=quarters,
        profit_loss=profit_loss,
        balance_sheet=balance_sheet,
        cash_flow=cash_flow,
        ratios=ratios,
        shareholding=shareholding,
        documents=documents,
        pros_cons=pros_cons,
    )


# ---------------------------------------------------------------------------
# Markdown formatters (what the tool layer hands to the LLM)
# ---------------------------------------------------------------------------


def _render_table(t: Optional[FinancialTable], title: str, max_rows: int = 20) -> str:
    """Render a ``FinancialTable`` as GitHub-flavored markdown."""
    if t is None or not t.columns:
        return f"### {title}\n_(no data)_"
    header = "| Metric | " + " | ".join(t.columns) + " |"
    sep = "|---|" + "|".join(["---"] * len(t.columns)) + "|"
    body_rows = []
    for i, (metric, values) in enumerate(t.rows):
        if i >= max_rows:
            body_rows.append(f"| _(+{len(t.rows) - max_rows} more rows truncated)_ | " + " | ".join([""] * len(t.columns)) + " |")
            break
        body_rows.append(f"| {metric} | " + " | ".join(values) + " |")
    return "\n".join([f"### {title}", header, sep] + body_rows)


def render_fundamentals_report(page: CompanyPage) -> str:
    """Markdown block replacing yfinance's ``.info`` for Indian tickers.

    Highlights the retail-grade summary metrics + the pros/cons that
    screener.in generates from its rules engine.  Every line carries a
    ``[source: <screener.in URL>]`` tag so the Reflective Critic sees
    real provenance rather than a bare fetch.
    """
    top_lines = [f"- **{k}**: {v}" for k, v in page.top.raw.items()]
    pros = "\n".join(f"- {p}" for p in page.pros_cons.pros) or "_(no pros listed)_"
    cons = "\n".join(f"- {c}" for c in page.pros_cons.cons) or "_(no cons listed)_"
    return (
        f"# {page.long_name} — Fundamentals ({page.mode})\n\n"
        + (f"{page.about}\n\n" if page.about else "")
        + "## Key ratios\n\n"
        + ("\n".join(top_lines) or "_(none)_")
        + "\n\n## Pros (screener.in analysis)\n"
        + pros
        + "\n\n## Cons (screener.in analysis)\n"
        + cons
        + "\n"
    )


def render_shareholding_report(page: CompanyPage) -> str:
    """Markdown block for promoter / FII / DII / public holding trends."""
    sh = page.shareholding
    if sh is None or sh.quarterly is None:
        return (
            f"# {page.long_name} — Shareholding pattern ({page.mode})\n\n"
            "_(no shareholding data available)_\n"
        )
    parts = [
        f"# {page.long_name} — Shareholding pattern ({page.mode})\n",
        _render_table(sh.quarterly, "Quarterly", max_rows=30),
    ]
    if sh.yearly is not None:
        parts.append("")
        parts.append(_render_table(sh.yearly, "Yearly", max_rows=30))
    return "\n\n".join(parts) + "\n"


def render_announcements_report(page: CompanyPage, limit: int = 12) -> str:
    """Markdown digest of the ``#documents`` filings.

    Complements ``get_corporate_actions`` (which uses Tavily news
    search) by surfacing the primary SEBI / BSE / NSE announcement
    feed.  A buyback resolution, dividend record date, board meeting
    outcome, etc. lands here days before it hits the generic news
    ranker — this is the specific fix for the BAJAJ-AUTO buyback-miss
    failure mode observed in the prior BAJAJ-AUTO.NS run.

    In a backfill ``AsOfContext`` the filing list is first filtered to
    those published on/before the cutoff (dated primary sources are the
    PIT-safe part of screener, unlike the restated ratio snapshot).
    """
    docs = pit_filter_documents(page.documents)[:limit]
    if not docs:
        return (
            f"# {page.long_name} — Recent filings & announcements ({page.mode})\n\n"
            "_(no recent filings found)_\n"
        )
    lines = [f"# {page.long_name} — Recent filings & announcements ({page.mode})\n"]
    lines.append(f"Latest {len(docs)} filings from BSE / NSE via screener.in:\n")
    for d in docs:
        lines.append(f"- {d.title}\n  {d.url}\n  [source: {d.url}]")
    return "\n".join(lines) + "\n"


_DOC_KIND_LABELS = {
    "announcement": "Announcements (SEBI / BSE / NSE)",
    "annual_report": "Annual reports",
    "concall": "Concall transcripts & investor presentations",
    "credit_rating": "Credit-rating actions",
}


def render_documents_report(page: CompanyPage, limit_per_kind: int = 6) -> str:
    """Grouped digest of dated primary documents (PIT-filtered).

    Groups the ``#documents`` feed by category so the analyst sees
    announcements, annual reports, concall transcripts, and credit-rating
    actions separately.  These are the point-in-time-safe screener
    surfaces (each carries a filing date), so in a backfill ``AsOfContext``
    the list is first filtered to ``filing_date <= as_of``.
    """
    docs = pit_filter_documents(page.documents)
    if not docs:
        return (
            f"# {page.long_name} — Dated documents ({page.mode})\n\n"
            "_(no dated documents available for this point in time)_\n"
        )
    by_kind: Dict[str, List[DocumentLink]] = {}
    for d in docs:
        by_kind.setdefault(d.kind, []).append(d)
    lines = [f"# {page.long_name} — Dated documents ({page.mode})\n"]
    for kind in ("announcement", "annual_report", "concall", "credit_rating"):
        group = by_kind.get(kind)
        if not group:
            continue
        lines.append(f"\n## {_DOC_KIND_LABELS[kind]}")
        for d in group[:limit_per_kind]:
            lines.append(f"- {d.title}\n  {d.url}\n  [source: {d.url}]")
    return "\n".join(lines) + "\n"


def render_financial_statement_report(page: CompanyPage, statement: str) -> str:
    """Markdown block for a single Ind-AS financial statement.

    ``statement`` picks the section:
    ``"profit-loss"`` / ``"balance-sheet"`` / ``"cash-flow"`` /
    ``"ratios"`` / ``"quarters"``.
    """
    table_by_kind = {
        "profit-loss": page.profit_loss,
        "balance-sheet": page.balance_sheet,
        "cash-flow": page.cash_flow,
        "ratios": page.ratios,
        "quarters": page.quarters,
    }
    if statement not in table_by_kind:
        raise ValueError(
            f"Unknown statement {statement!r}; "
            "expected one of profit-loss / balance-sheet / cash-flow / ratios / quarters."
        )
    return (
        f"# {page.long_name} — {statement.replace('-', ' ').title()} ({page.mode})\n\n"
        + _render_table(table_by_kind[statement], statement.title(), max_rows=25)
        + "\n"
    )
