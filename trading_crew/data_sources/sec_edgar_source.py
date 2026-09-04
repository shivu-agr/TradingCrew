"""SEC EDGAR client — dated primary filings for US (SEC-registered) names.

This is the US analog of :mod:`screener_source`.  ``get_report_insights`` reads
a company's OWN primary filings — the ones where management states explicit
forward TARGETS and where a prior target can be checked against what was
actually DELIVERED.  For Indian names that feed is screener.in's dated
``#documents`` list; for US names the equivalent official, free, keyless feed
is **SEC EDGAR**:

* ticker -> CIK via the public ``company_tickers.json`` map, and
* the per-company ``submissions`` JSON (``data.sec.gov``) which is a dated list
  of every filing (form type, filing date, accession, primary document).

We map the guidance-rich forms onto the SAME :class:`DocumentLink` shape the
tool already consumes, so US tickers get the same guidance / target-vs-actual /
outlook read Indian names get, with zero changes to the summariser:

  * ``10-K`` / ``20-F`` / ``40-F``  -> ``annual_report``  (MD&A, outlook)
  * ``10-Q``                        -> ``quarterly_report`` (interim MD&A)
  * ``8-K`` with item ``2.02``      -> ``earnings``  (results + guidance press
    release; the numbers live in the Exhibit 99.1 that ships alongside the thin
    cover, so we resolve to that exhibit before crawling)

Concall transcripts and credit-rating rationales are NOT on EDGAR (they live on
third-party sites / agency portals), so those two ``screener`` kinds have no
EDGAR equivalent — the web-search desk already surfaces transcripts, and the
tool notes the gap.

Auth / politeness
-----------------
EDGAR needs no key, but its fair-access policy requires a descriptive
``User-Agent`` carrying a contact and asks clients to stay under ~10 req/s.  We
send ``SEC_EDGAR_USER_AGENT`` (env-overridable) and route through the same
polite ``scrape`` rate limiter screener uses.  There is no silent fallback: an
unresolvable ticker / transport error surfaces as an exception the tool layer
turns into a single not-available line (project rule).

Point-in-time
-------------
Every filing carries an exact ``filing_date``, so PIT filtering is precise
(unlike screener's freeform title dates): in a backfill ``AsOfContext`` we drop
anything filed after the cutoff.
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests

from .routing import strip_exchange_suffix
from .screener_source import DocumentLink  # reuse the tool's document shape

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_CIK_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
_ARCHIVES_FOLDER = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}"

# Guidance-rich forms we surface, mapped to the DocumentLink ``kind`` the
# report-insights selector understands.  8-K is handled separately (only the
# earnings item 2.02 subset is kept).
_FORM_KIND = {
    "10-K": "annual_report",
    "10-K/A": "annual_report",
    "20-F": "annual_report",
    "40-F": "annual_report",
    "10-Q": "quarterly_report",
    "10-Q/A": "quarterly_report",
}
# 8-K item that flags an earnings / results release (Results of Operations and
# Financial Condition).  The forward guidance a desk needs is in these.
_EARNINGS_8K_ITEM = "2.02"

# How many of the most-recent relevant filings to keep per company.  The tool
# then selects a smaller guidance-rich subset from these to actually crawl.
_MAX_FILINGS = 12


def _user_agent() -> str:
    """Descriptive UA required by EDGAR's fair-access policy (env-overridable).

    SEC blocks generic library UAs and asks for a contact; this is a required
    request header, not a data fallback.  Override with ``SEC_EDGAR_USER_AGENT``
    to carry your own contact address.
    """
    return (
        os.environ.get("SEC_EDGAR_USER_AGENT", "").strip()
        or "TradingCrew research bot (contact: admin@tradingcrew.local)"
    )


def _enabled() -> bool:
    val = os.environ.get("TRADINGCREW_EDGAR_ENABLED", "1").strip().lower()
    return val not in ("0", "false", "no", "off", "")


# ---------------------------------------------------------------------------
# Error surface (mirrors screener_source so the tool catches uniformly)
# ---------------------------------------------------------------------------


class EdgarError(Exception):
    """Base class for EDGAR client errors (transport / bad status)."""


class EdgarNotFoundError(EdgarError):
    """The ticker has no CIK in EDGAR, or the company has no relevant filings."""


class EdgarRateLimitError(EdgarError):
    """EDGAR returned HTTP 429 (fair-access throttle tripped)."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class EdgarFiling:
    """One dated EDGAR filing normalised to what the tool needs."""

    form: str
    filing_date: datetime          # tz-aware UTC (midnight of the filing day)
    accession: str                 # dashed, e.g. "0000320193-26-000011"
    primary_document: str          # e.g. "aapl-20260328.htm"
    items: str                     # 8-K items csv, "" for periodic reports
    kind: str                      # annual_report | quarterly_report | earnings
    cik: int
    folder_url: str
    primary_url: str

    @property
    def title(self) -> str:
        label = {
            "annual_report": "Annual report",
            "quarterly_report": "Quarterly report",
            "earnings": "Earnings release",
        }.get(self.kind, self.form)
        return f"{label} ({self.form}) — filed {self.filing_date:%Y-%m-%d}"


@dataclass
class EdgarCompany:
    ticker: str
    cik: int
    name: str
    filings: List[EdgarFiling] = field(default_factory=list)


# ---------------------------------------------------------------------------
# HTTP session + polite throttle
# ---------------------------------------------------------------------------

_SESSION_LOCK = threading.Lock()
_SESSION: Optional[requests.Session] = None
_THROTTLE_LOCK = threading.Lock()
_REQUEST_LOCK = threading.Lock()
_LAST_REQUEST_AT: float = 0.0


def _min_gap() -> float:
    try:
        return max(0.0, float(os.environ.get("SEC_EDGAR_MIN_REQUEST_GAP_SECONDS", "0.25")))
    except (TypeError, ValueError):
        return 0.25


def _throttle() -> None:
    global _LAST_REQUEST_AT
    gap = _min_gap()
    if gap <= 0:
        return
    with _THROTTLE_LOCK:
        now = time.monotonic()
        wait = (_LAST_REQUEST_AT + gap) - now
        if wait > 0:
            time.sleep(wait + random.uniform(0.0, gap * 0.25))
        _LAST_REQUEST_AT = time.monotonic()


def _get_session() -> requests.Session:
    global _SESSION
    with _SESSION_LOCK:
        if _SESSION is None:
            s = requests.Session()
            s.headers.update(
                {
                    "User-Agent": _user_agent(),
                    "Accept-Encoding": "gzip, deflate",
                    "Host": "www.sec.gov",
                }
            )
            _SESSION = s
        # Host header is per-request (data.sec.gov vs www.sec.gov); keep UA only.
        _SESSION.headers.pop("Host", None)
        return _SESSION


def _get_json(url: str) -> dict:
    """GET ``url`` and parse JSON, routed through the polite scrape limiter."""
    _pace()
    try:
        r = _get_session().get(url, timeout=20)
    except requests.RequestException as exc:
        raise EdgarError(f"EDGAR GET {url} failed: {exc}") from exc
    if r.status_code == 429:
        raise EdgarRateLimitError(f"EDGAR rate-limited (429) on {url}")
    if r.status_code == 404:
        raise EdgarNotFoundError(f"EDGAR 404 on {url}")
    if r.status_code != 200:
        raise EdgarError(f"EDGAR GET {url} returned HTTP {r.status_code}")
    try:
        return r.json()
    except ValueError as exc:
        raise EdgarError(f"EDGAR returned non-JSON from {url}: {exc}") from exc


def _pace() -> None:
    """Prefer the centralised Kafka ``scrape`` slot queue; else the local gap."""
    try:
        from . import rate_gateway

        if rate_gateway.is_enabled():
            rate_gateway.acquire("scrape")
            return
    except Exception:  # noqa: BLE001 — limiter must never break a scrape
        pass
    _throttle()


def fetch_document_text(url: str, max_chars: Optional[int] = 60000) -> str:
    """Fetch a filing document and return its plain text (HTML stripped).

    EDGAR gates the shared web crawler (``web_research.crawl``) with its
    fair-access User-Agent policy (403 without a descriptive UA + contact), so
    the report-insights US path fetches filing text through THIS client, which
    already carries the compliant UA + polite pacing.  ``.htm`` / ``.html`` is
    tag-stripped to readable text; ``.txt`` is returned as-is.  Best-effort:
    returns ``""`` on any transport / parse failure so the tool degrades to a
    plain filing listing instead of crashing.
    """
    _pace()
    try:
        # Session is not thread-safe; report-insights extracts filings in
        # parallel so the GET must be serialised. Parse happens unlocked.
        with _REQUEST_LOCK:
            r = _get_session().get(url, timeout=300 if max_chars is None else 30)
    except requests.RequestException as exc:
        logger.warning("EDGAR document GET %s failed: %s", url, exc)
        return ""
    if r.status_code != 200:
        logger.warning("EDGAR document GET %s returned HTTP %s", url, r.status_code)
        return ""
    ctype = r.headers.get("Content-Type", "")
    low = url.lower()
    # PDF exhibits: parse locally (text + tables) rather than tag-stripping bytes.
    if "application/pdf" in ctype.lower() or (r.content or b"")[:5] == b"%PDF-" or low.split("?", 1)[0].endswith(".pdf"):
        try:
            from trading_crew.data_sources.web_research import extract_pdf_text

            return extract_pdf_text(r.content, max_chars=max_chars) or ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("EDGAR PDF parse failed for %s: %s", url, exc)
            return ""
    text = r.text or ""
    if low.endswith(".htm") or low.endswith(".html") or "text/html" in ctype:
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(text, "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            text = soup.get_text(" ", strip=True)
        except Exception as exc:  # noqa: BLE001 — fall back to raw HTML text
            logger.warning("EDGAR document parse failed for %s: %s", url, exc)
    return text[:max_chars] if max_chars and len(text) > max_chars else text


# ---------------------------------------------------------------------------
# ticker -> CIK map (cached for the process; the file is ~1 MB)
# ---------------------------------------------------------------------------

_CIK_LOCK = threading.Lock()
_CIK_MAP: Optional[Dict[str, tuple]] = None  # TICKER -> (cik:int, title:str)


def _load_cik_map() -> Dict[str, tuple]:
    global _CIK_MAP
    with _CIK_LOCK:
        if _CIK_MAP is not None:
            return _CIK_MAP
        raw = _get_json(_CIK_MAP_URL)
        mapping: Dict[str, tuple] = {}
        # company_tickers.json is {"0": {"cik_str":320193,"ticker":"AAPL",
        # "title":"Apple Inc."}, ...}
        for row in raw.values():
            tkr = str(row.get("ticker", "")).strip().upper()
            if not tkr_ok(tkr):
                continue
            mapping[tkr] = (int(row["cik_str"]), str(row.get("title", "")))
        _CIK_MAP = mapping
        logger.info("EDGAR ticker->CIK map loaded (%d tickers)", len(mapping))
        return mapping


def tkr_ok(t: str) -> bool:
    return bool(t) and all(c.isalnum() or c in "-." for c in t)


def _normalize_ticker(ticker: str) -> str:
    """EDGAR lists class shares with a hyphen (``BRK-B``), yfinance a dot
    (``BRK.B``); callers may also pass an exchange suffix.  Return the EDGAR
    form."""
    root = strip_exchange_suffix(ticker).strip().upper()
    return root


def lookup_cik(ticker: str) -> Optional[tuple]:
    """Return ``(cik:int, title:str)`` for ``ticker`` or ``None`` if EDGAR has
    no such US-registered symbol."""
    root = _normalize_ticker(ticker)
    cmap = _load_cik_map()
    if root in cmap:
        return cmap[root]
    # class-share spelling: yfinance "BRK.B" -> EDGAR "BRK-B" and vice-versa.
    for alt in (root.replace(".", "-"), root.replace("-", ".")):
        if alt in cmap:
            return cmap[alt]
    return None


def is_edgar_ticker(ticker: str) -> bool:
    """True iff EDGAR has a CIK for this ticker (US SEC-registered name)."""
    if not _enabled():
        return False
    try:
        return lookup_cik(ticker) is not None
    except EdgarError:
        # Map fetch failed — don't claim availability we can't back up.
        return False


# ---------------------------------------------------------------------------
# Filing fetch
# ---------------------------------------------------------------------------


def _parse_date(s: str) -> Optional[datetime]:
    try:
        return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def fetch_filings(ticker: str, max_filings: int = _MAX_FILINGS) -> EdgarCompany:
    """Fetch the most-recent guidance-rich filings for a US ticker.

    Keeps up to ``max_filings`` of the newest 10-K / 20-F / 40-F / 10-Q and
    earnings 8-K (item 2.02) filings.  Raises :class:`EdgarNotFoundError` when
    the ticker isn't in EDGAR or has no relevant filings.
    """
    hit = lookup_cik(ticker)
    if hit is None:
        raise EdgarNotFoundError(f"EDGAR has no CIK for ticker {ticker!r}.")
    cik, title = hit
    cik10 = f"{cik:010d}"
    data = _get_json(_SUBMISSIONS_URL.format(cik10=cik10))
    name = str(data.get("name") or title or ticker)
    recent = (data.get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    dates = recent.get("filingDate") or []
    accs = recent.get("accessionNumber") or []
    prims = recent.get("primaryDocument") or []
    items_l = recent.get("items") or [""] * len(forms)

    filings: List[EdgarFiling] = []
    for i in range(len(forms)):
        form = forms[i]
        items = items_l[i] if i < len(items_l) else ""
        if form in _FORM_KIND:
            kind = _FORM_KIND[form]
        elif form == "8-K" and _EARNINGS_8K_ITEM in (items or ""):
            kind = "earnings"
        else:
            continue
        fdate = _parse_date(dates[i]) if i < len(dates) else None
        if fdate is None:
            continue
        acc = accs[i] if i < len(accs) else ""
        prim = prims[i] if i < len(prims) else ""
        acc_nodash = acc.replace("-", "")
        folder = _ARCHIVES_FOLDER.format(cik=cik, acc=acc_nodash)
        filings.append(
            EdgarFiling(
                form=form,
                filing_date=fdate,
                accession=acc,
                primary_document=prim,
                items=items or "",
                kind=kind,
                cik=cik,
                folder_url=folder,
                primary_url=f"{folder}/{prim}" if prim else folder,
            )
        )
        if len(filings) >= max_filings:
            break

    if not filings:
        raise EdgarNotFoundError(
            f"EDGAR has a CIK for {ticker!r} but no recent 10-K/10-Q/earnings-8-K filings."
        )
    return EdgarCompany(ticker=_normalize_ticker(ticker), cik=cik, name=name, filings=filings)


# ---------------------------------------------------------------------------
# Exhibit resolution (earnings 8-K -> the Ex-99.1 press release)
# ---------------------------------------------------------------------------

# primary_url -> resolved best-text url, so resolve_text_url is a cheap repeat.
_RESOLVE_CACHE: Dict[str, str] = {}
_RESOLVE_LOCK = threading.Lock()
# primary_url -> EdgarFiling, so the tool can resolve by url alone.
_FILING_BY_URL: Dict[str, EdgarFiling] = {}


def _remember(filing: EdgarFiling) -> None:
    _FILING_BY_URL[filing.primary_url] = filing


def _pick_press_release(items: List[dict], primary_document: str) -> Optional[str]:
    """From an 8-K filing folder listing, pick the Exhibit 99.1 press-release
    document (the one carrying revenue / EPS / guidance), else the largest
    non-primary .htm/.txt exhibit."""
    best_ex99: Optional[tuple] = None   # (size, name)
    best_other: Optional[tuple] = None
    for it in items:
        name = str(it.get("name", ""))
        low = name.lower()
        if not (low.endswith(".htm") or low.endswith(".html") or low.endswith(".txt")):
            continue
        if low == primary_document.lower():
            continue
        try:
            size = int(it.get("size") or 0)
        except (TypeError, ValueError):
            size = 0
        # ex99 / ex-99 / a8-kex991... all denote the press release exhibit.
        if "ex99" in low.replace("-", "") or "ex-99" in low:
            if best_ex99 is None or size > best_ex99[0]:
                best_ex99 = (size, name)
        elif best_other is None or size > best_other[0]:
            best_other = (size, name)
    pick = best_ex99 or best_other
    return pick[1] if pick else None


def resolve_text_url(url: str) -> str:
    """Given a filing's ``primary_url``, return the URL whose text a desk should
    actually read.

    For 10-K / 10-Q that's the primary document (full MD&A).  For an earnings
    8-K the numbers live in the Exhibit 99.1 press release that ships beside the
    thin cover, so fetch the filing folder index and swap to that exhibit.
    Best-effort: on any failure the primary URL is returned unchanged.
    """
    with _RESOLVE_LOCK:
        if url in _RESOLVE_CACHE:
            return _RESOLVE_CACHE[url]
    filing = _FILING_BY_URL.get(url)
    resolved = url
    try:
        if filing is not None and filing.kind == "earnings":
            idx = _get_json(f"{filing.folder_url}/index.json")
            items = (idx.get("directory") or {}).get("item") or []
            name = _pick_press_release(items, filing.primary_document)
            if name:
                resolved = f"{filing.folder_url}/{name}"
    except EdgarError as exc:
        logger.warning("EDGAR exhibit resolution failed for %s: %s", url, exc)
        resolved = url
    with _RESOLVE_LOCK:
        _RESOLVE_CACHE[url] = resolved
    return resolved


# ---------------------------------------------------------------------------
# DocumentLink adaptation + PIT
# ---------------------------------------------------------------------------


def to_document_links(company: EdgarCompany) -> List[DocumentLink]:
    """Adapt EDGAR filings to the :class:`DocumentLink` shape the tool consumes.

    The url is the filing's primary document; earnings-8-K links are resolved to
    their press-release exhibit lazily by :func:`resolve_text_url` (keyed on the
    primary url), so the tool crawls the richest text without an upfront request
    per filing.
    """
    docs: List[DocumentLink] = []
    for f in company.filings:
        _remember(f)
        docs.append(DocumentLink(title=f.title, url=f.primary_url, kind=f.kind))
    return docs


def pit_filter_documents(company: EdgarCompany) -> EdgarCompany:
    """Drop filings filed after the current PIT cutoff (backfill only).

    Live mode returns the company unchanged.  Filing dates are exact, so there
    is no ambiguous-date fail-closed case (unlike screener's freeform titles).
    """
    try:
        from . import pit
    except Exception:  # noqa: BLE001
        return company
    ctx = pit.get_context()
    if not ctx.is_backfill:
        return company
    kept = [f for f in company.filings if f.filing_date <= ctx.as_of]
    return EdgarCompany(ticker=company.ticker, cik=company.cik, name=company.name, filings=kept)


def clear_cache() -> None:
    """Test hook — drop the CIK map + resolution caches."""
    global _CIK_MAP
    with _CIK_LOCK:
        _CIK_MAP = None
    with _RESOLVE_LOCK:
        _RESOLVE_CACHE.clear()
    _FILING_BY_URL.clear()
