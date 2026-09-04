"""Company metadata + symbol search for the Prediction dashboard.

Two concerns, both live/authoritative (no static tables):

  * ``enrich_ticker`` — resolve a ticker's display **name** and **homepage**
    from Yahoo Finance (via ``yfinance``), plus a **Wikipedia** deep link built
    deterministically from the resolved name. Used to turn the seeded universe
    (which only has tickers) into named, linkable tiles; results are cached back
    into the prediction DB so it runs once per name.
  * ``search_symbols`` — resolve a free-text company query to real, runnable
    tickers via Yahoo Finance's symbol-search endpoint. This is what powers the
    "add a company" search box: the user types a name, picks a real symbol
    (with exchange + market), and it gets added under the right sector.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)

# NSE / BSE listings map to the Indian market; everything else is treated as US
# for this dashboard's two-market split.
_INDIA_EXCHANGES = {"NSI", "BSE", "NSE"}


def _wikipedia_link(name: str) -> str:
    return f"https://en.wikipedia.org/wiki/Special:Search?search={quote(name)}"


def _market_for(symbol: str, exchange: str = "") -> str:
    if symbol.upper().endswith(".NS") or symbol.upper().endswith(".BO"):
        return "IN"
    if (exchange or "").upper() in _INDIA_EXCHANGES:
        return "IN"
    return "US"


def enrich_ticker(ticker: str) -> Dict[str, Optional[str]]:
    """Resolve {name, homepage, wikipedia, sector} for a ticker via yfinance."""
    import yfinance as yf

    info: Dict[str, Any] = {}
    try:
        info = yf.Ticker(ticker).get_info() or {}
    except Exception as exc:  # noqa: BLE001 — a flaky lookup must not break the page
        logger.warning("yfinance info failed for %s: %s", ticker, exc)

    name = (
        info.get("longName")
        or info.get("shortName")
        or info.get("displayName")
        or ""
    ).strip()
    homepage = (info.get("website") or "").strip() or None
    wikipedia = _wikipedia_link(name) if name else None
    sector = (info.get("sector") or "").strip() or None
    return {"name": name or None, "homepage": homepage, "wikipedia": wikipedia, "sector": sector}


_ALLOWED_TYPES = ("EQUITY", "ETF", "MUTUALFUND", "INDEX")


def _yahoo_quotes(query: str) -> List[Dict[str, Any]]:
    import yfinance as yf

    try:
        return yf.Search(query, max_results=20).quotes or []
    except Exception as exc:  # noqa: BLE001
        logger.warning("yfinance search failed for %q: %s", query, exc)
        return []


def search_symbols(query: str, limit: int = 8, market: str = "") -> List[Dict[str, str]]:
    """Resolve a company-name query to candidate tickers.

    Uses ``yfinance.Search`` (it manages the cookie+crumb the raw endpoint now
    requires — hitting the URL directly returns HTTP 429).

    Yahoo search is unreliable for Indian names: for a query like "infosys" it
    returns the US ADR (``INFY``) or the Frankfurt line, and the primary NSE
    symbol (``INFY.NS`` / ``RELIANCE.NS``) is frequently missing entirely. So
    when ``market == "IN"`` we *construct* NSE candidates — the compacted query
    (``reliance`` -> ``RELIANCE.NS``, ``tata motors`` -> ``TATAMOTORS.NS``) and
    the root of each Yahoo hit (``INFY`` -> ``INFY.NS``) — validate them against
    yfinance, and surface the real NSE listings first.
    """
    query = (query or "").strip()
    if not query:
        return []
    market = (market or "").upper()
    quotes = _yahoo_quotes(query)
    directs = [q for q in quotes if (q.get("quoteType") or "").upper() in _ALLOWED_TYPES]

    out: List[Dict[str, str]] = []
    seen: set[str] = set()

    def _add(symbol: str, name: str, exchange: str, qtype: str) -> None:
        s = (symbol or "").strip()
        if not s or s.upper() in seen:
            return
        seen.add(s.upper())
        out.append({
            "symbol": s,
            "name": (name or s).strip(),
            "exchange": exchange or "",
            "type": qtype or "EQUITY",
            "market": _market_for(s, exchange),
        })

    if market == "IN":
        # 1) construct + validate NSE candidates (bounded to keep the box snappy)
        candidates: List[str] = []
        compact = re.sub(r"[^A-Za-z0-9]", "", query).upper()
        if compact:
            candidates.append(f"{compact}.NS")
        for q in directs[:3]:
            root = (q.get("symbol") or "").split(".", 1)[0].upper()
            if root:
                candidates.append(f"{root}.NS")
        validated = 0
        for cand in dict.fromkeys(candidates):
            if validated >= 4:
                break
            meta = enrich_ticker(cand)
            validated += 1
            if meta.get("name"):
                _add(cand, meta["name"], "NSE", "EQUITY")
        # 2) any real NSE/BSE hits Yahoo did return
        for q in directs:
            sym = (q.get("symbol") or "").upper()
            if sym.endswith(".NS") or sym.endswith(".BO"):
                _add(q["symbol"], q.get("longname") or q.get("shortname") or sym,
                     q.get("exchDisp") or q.get("exchange") or "", (q.get("quoteType") or "").upper())
        # 3) fall back to remaining hits so nothing is hidden
        for q in directs:
            _add(q["symbol"], q.get("longname") or q.get("shortname") or q["symbol"],
                 q.get("exchDisp") or q.get("exchange") or "", (q.get("quoteType") or "").upper())
    else:
        # US / auto: US listings first, foreign (Frankfurt .F etc.) last
        def _rank(q: Dict[str, Any]) -> int:
            exch = (q.get("exchange") or "").upper()
            if exch in ("NMS", "NYQ", "NGM", "PCX", "ASE", "BATS"):
                return 0
            if _market_for(q.get("symbol", ""), q.get("exchange", "")) == "IN":
                return 1
            return 2
        for q in sorted(directs, key=_rank):
            _add(q["symbol"], q.get("longname") or q.get("shortname") or q["symbol"],
                 q.get("exchDisp") or q.get("exchange") or "", (q.get("quoteType") or "").upper())

    return out[:limit]
