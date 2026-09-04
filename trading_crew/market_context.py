"""Ticker resolution + dynamic research context.

This module is the *single source of truth* for "what does this ticker
actually represent?" — the country, currency, sector, industry, macro
instruments, and the live peer screen the analyst team grounds work in.

Why it exists
-------------
Before this module the news / macro / geopolitical tools hard-coded
US-centric defaults ("Fed rates", "TSMC China dependency", "10y
treasury yield"). That makes sense for an NYSE/Nasdaq name but produces
nonsense for an Indian shipbuilder like MAZDOCK.NS — the analyst still
talks about the Federal Reserve while the trade is driven by RBI policy,
INR/USD, Strait of Hormuz, Indian defence-budget allocations, etc.

Two responsibilities live here:

1. ``resolve_ticker(raw)`` — turn a user-typed symbol into the canonical
   yfinance ticker (``MAZDOCK`` → ``MAZDOCK.NS``). yfinance silently
   returns empty frames for unsuffixed Indian / European tickers; we
   probe the common suffixes once and cache the answer.

2. ``get_market_profile(symbol)`` — build a structured ``MarketProfile``
   the tool layer reuses to generate **dynamic, ticker-specific news
   queries**.  Each profile carries:

   - ``country`` / ``currency`` / ``exchange`` (from yfinance ``.info``).
   - ``sector`` / ``industry`` / ``long_name``.
   - ``macro_tickers`` — yfinance symbols for the rates / FX / vol /
     commodity baskets that actually move the stock (e.g. ``^TNX``
     for a US name, ``^INDIAVIX`` + ``INR=X`` for an NSE name).
   - ``peer_index`` — broad-market benchmark (S&P for the US, Nifty
     for India, FTSE for the UK, …) used by alpha calculations.
   - facts the analyst uses to *write* a search query (country,
     industry, name). Search prose is not stored here. Peers are not
     a per-ticker dictionary — ``discover_peers`` screens Yahoo's
     live industry universe in the listing's country.

Provenance is preserved end-to-end: tools that build queries from a
profile still emit ``Source: …`` footers so the Reflective Critic can
audit every claim back to a real fetch.
"""

from __future__ import annotations

import logging
import math
import re
import statistics
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yfinance as yf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Suffix probing for non-US exchanges
# ---------------------------------------------------------------------------

# Order matters: the first suffix that yields non-empty OHLCV wins.
# ".NS" (NSE) comes before ".BO" (BSE) because NSE is the deeper book
# for most Indian names. London (".L") + Hong Kong (".HK") + Tokyo
# (".T") are included so the same resolver handles HSBA, 0700.HK, 7203.T.
_PROBE_SUFFIXES: Tuple[str, ...] = ("", ".NS", ".BO", ".L", ".HK", ".T", ".SI", ".TO", ".AX", ".DE", ".PA")

# In-process cache so resolve_ticker doesn't pay the yfinance round-trip
# on every tool call within a single run.  Keyed by the raw user input
# (uppercased) — the cached value is the canonical yfinance symbol.
_RESOLVE_CACHE: Dict[str, str] = {}


def _has_history(symbol: str) -> bool:
    """Return True iff yfinance has any recent OHLCV for ``symbol``.

    Uses a short window (1mo) so the probe is cheap.  We deliberately
    swallow exceptions — yfinance occasionally raises on bad-symbol
    lookups instead of returning empty, and we treat both the same.

    yfinance's own logger emits ERROR-level lines for every "no data
    found" response (one per suffix we try).  Most probes are *expected*
    to miss; logging them as ERROR floods ``logs/web.log`` with red
    herrings that distract from real failures.  We temporarily bump its
    logger up to CRITICAL for the duration of the probe so only its
    INFO/WARNING/ERROR lines go quiet — our own logger keeps the
    "resolver did not find a match" signal at WARNING level (see
    :func:`resolve_ticker`).
    """
    yf_logger = logging.getLogger("yfinance")
    saved_level = yf_logger.level
    yf_logger.setLevel(logging.CRITICAL)
    try:
        df = yf.Ticker(symbol).history(period="1mo")
    except Exception:
        return False
    finally:
        yf_logger.setLevel(saved_level)
    return df is not None and not df.empty


# Characters that *cannot* appear in a real ticker symbol.  yfinance
# tickers are alphanumeric with at most ``.`` (exchange suffix),
# ``-``  (BRK-B style class suffix), ``=`` (futures, e.g. ``CL=F``)
# or ``^`` (indices, e.g. ``^GSPC``).  Anything outside that set is
# almost certainly a company *name* the LLM is passing in by mistake
# (e.g. ``"LARSEN & TOUBRO LIMITED"``); probing 11 exchange suffixes
# on a value like that wastes ~11 network round-trips and floods the
# log with yfinance error chatter.
_TICKER_ALLOWED = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-=^")


def _looks_like_ticker(sym: str) -> bool:
    """Cheap structural validator — does ``sym`` look like a yfinance ticker?

    Bare symbols are ≤ 12 characters. Suffixed listings may run longer
    (``COCHINSHIP.NS``, ``TATAMOTORS.NS``, ``BHARTIARTL.NS``) — those
    stay accepted up to 16 chars so the live peer screen does not drop
    real NSE names. Whitespace / punctuation still fail so a company
    *name* does not trigger an exchange-suffix probe storm.
    """
    if not sym or " " in sym:
        return False
    if not all(ch in _TICKER_ALLOWED for ch in sym):
        return False
    if "." in sym or "=" in sym or "^" in sym:
        return len(sym) <= 16
    return len(sym) <= 12


def resolve_ticker(raw: str) -> str:
    """Return the canonical yfinance symbol for a user-typed ticker.

    Examples
    --------
    >>> resolve_ticker("MAZDOCK")    # -> "MAZDOCK.NS"
    >>> resolve_ticker("AAPL")       # -> "AAPL"
    >>> resolve_ticker("RELIANCE")   # -> "RELIANCE.NS"
    >>> resolve_ticker("0700")       # -> "0700.HK"

    The resolver probes a short list of common exchange suffixes
    (NSE / BSE / LSE / HKEX / TSE / SGX / TSX / ASX / XETRA / Euronext)
    and returns the first one with data.  If the input already carries
    a suffix (e.g. ``MAZDOCK.NS`` or ``CL=F``) it's returned unchanged.

    On a structurally-invalid input (whitespace, ``&``, lower-case-only,
    > 12 chars) we return the raw string immediately without probing.
    This is the common case where an LLM passes in a company *name*
    instead of a ticker — running 11 yfinance round-trips on garbage
    is pure waste and floods the log with red herrings.

    On total probe miss we return the raw input — the downstream tool
    surfaces an explicit "no data" message rather than silently masking
    the failure with a wrong symbol.
    """
    if not raw:
        return raw
    sym = raw.strip().upper()
    if sym in _RESOLVE_CACHE:
        return _RESOLVE_CACHE[sym]

    if not _looks_like_ticker(sym):
        # Skip the probe entirely — saves 11 network calls + 11 ERROR
        # log lines per malformed call.  Log once at WARNING so the
        # caller can still see they passed in a non-ticker without
        # having to scrape the network noise.
        logger.warning(
            "resolve_ticker: %r is not a ticker (contains whitespace / "
            "punctuation / too long); skipping suffix probe.",
            raw,
        )
        _RESOLVE_CACHE[sym] = sym
        return sym

    # Already-suffixed symbols (e.g. ``MAZDOCK.NS``, ``CL=F``, ``BRK-B``)
    # are passed through untouched.  We only probe symbols that contain
    # no dot / equals / caret so we don't accidentally turn ``BRK-B``
    # into ``BRK-B.NS``.
    if "." in sym or "=" in sym or "^" in sym:
        _RESOLVE_CACHE[sym] = sym
        return sym

    for suffix in _PROBE_SUFFIXES:
        candidate = f"{sym}{suffix}"
        if _has_history(candidate):
            _RESOLVE_CACHE[sym] = candidate
            return candidate

    # Total miss — still cache so we don't re-probe repeatedly.
    logger.warning(
        "resolve_ticker: no yfinance match for %r after probing %d "
        "exchange suffixes; returning input as-is.",
        raw, len(_PROBE_SUFFIXES),
    )
    _RESOLVE_CACHE[sym] = sym
    return sym


def clear_resolve_cache() -> None:
    """Test hook — empty the in-process resolver cache."""
    _RESOLVE_CACHE.clear()


# ---------------------------------------------------------------------------
# Country / exchange → macro basket + benchmark + news themes
# ---------------------------------------------------------------------------

# Each country bucket carries *instrument identifiers*, not search prose:
#   - macro_tickers: yfinance symbols for rates / FX / vol / commodities
#   - peer_index: broad-market benchmark for alpha
#   - currency: ISO-4217
# Search queries are written by the analyst from role + these facts +
# any gap on the blackboard — not from a canned theme list.
_COUNTRY_PROFILE: Dict[str, Dict[str, object]] = {
    "United States": {
        "currency": "USD",
        "peer_index": "^GSPC",  # S&P 500
        "macro_tickers": {
            "us_10y_treasury_yield": "^TNX",
            "us_dollar_index": "DX-Y.NYB",
            "vix_volatility": "^VIX",
            "crude_oil_wti": "CL=F",
            "gold": "GC=F",
        },
    },
    "India": {
        "currency": "INR",
        "peer_index": "^NSEI",  # Nifty 50
        "macro_tickers": {
            "nifty_50": "^NSEI",
            "india_vix": "^INDIAVIX",
            "usd_inr": "INR=X",
            "sensex": "^BSESN",
            "brent_crude": "BZ=F",
            "gold": "GC=F",
        },
    },
    "United Kingdom": {
        "currency": "GBP",
        "peer_index": "^FTSE",
        "macro_tickers": {
            "uk_10y_gilt_yield": "^TNX",  # yfinance lacks a clean UK 10y; TNX as proxy
            "gbp_usd": "GBPUSD=X",
            "ftse_100": "^FTSE",
            "brent_crude": "BZ=F",
            "gold": "GC=F",
        },
    },
    "Hong Kong": {
        "currency": "HKD",
        "peer_index": "^HSI",
        "macro_tickers": {
            "hang_seng": "^HSI",
            "usd_hkd": "HKD=X",
            "csi_300_shanghai": "000300.SS",
            "brent_crude": "BZ=F",
        },
    },
    "Japan": {
        "currency": "JPY",
        "peer_index": "^N225",
        "macro_tickers": {
            "nikkei_225": "^N225",
            "usd_jpy": "JPY=X",
            "topix": "^TPX",
            "brent_crude": "BZ=F",
        },
    },
}

# Default fallback when ``.info["country"]`` is missing or unmapped — we
# still want *something* better than US-only.
_DEFAULT_COUNTRY_KEY = "United States"


# ---------------------------------------------------------------------------
# Live peer universe (Yahoo industry screen — no per-ticker dictionary)
# ---------------------------------------------------------------------------

# Listing suffix → Yahoo exchange code. Instrument identifiers only;
# used to invert yfinance's own exchange→region map so an NSE name
# screens India, not the US mega-cap list.
_SUFFIX_TO_EXCHANGE: Dict[str, str] = {
    ".NS": "NSI", ".BO": "BSE",
    ".L": "LSE", ".HK": "HKG", ".T": "JPX",
    ".SI": "SES", ".TO": "TOR", ".AX": "ASX",
    ".DE": "GER", ".PA": "PAR",
}

# English country name → Yahoo screener region. Same five buckets as
# ``_COUNTRY_PROFILE`` — routing keys, not search prose.
_COUNTRY_TO_YF_REGION: Dict[str, str] = {
    "United States": "us",
    "India": "in",
    "United Kingdom": "gb",
    "Hong Kong": "hk",
    "Japan": "jp",
}

_PEER_SPLIT = re.compile(r"[,;\s]+")
_MAX_PEERS = 6          # focus + 5 comparables
_MIN_OTHERS = 2         # widen the *live* screen if fewer
_MCAP_LO = 0.15
_MCAP_HI = 8.0
_PEER_CACHE: Dict[Tuple[str, str], "PeerUniverse"] = {}


@dataclass(frozen=True)
class PeerUniverse:
    """Live (or agent-written) comparable set for one focus ticker."""

    symbols: List[str]
    method: str
    industry: str
    region: str
    notes: str = ""


@dataclass(frozen=True)
class PeerRow:
    """One priced row in the sector-peer table."""

    symbol: str
    ytd_pct: Optional[float]
    ret_6m_pct: Optional[float]
    market_cap: Optional[float]
    pe: Optional[float]
    industry: str
    country: str


def _compact_alnum(value: str) -> str:
    return "".join(ch.lower() for ch in (value or "") if ch.isalnum())


def _listing_suffix(symbol: str) -> str:
    if "." in symbol:
        return symbol.rsplit(".", 1)[-1].upper()
    return ""


def _region_for_exchange(exchange: str) -> str:
    """Invert yfinance's exchange→region map (NSI→in, NMS→us, …)."""
    if not exchange:
        return ""
    try:
        from yfinance.const import EQUITY_SCREENER_EQ_MAP
    except Exception:
        return ""
    needle = exchange.strip().upper()
    for region, exchanges in (EQUITY_SCREENER_EQ_MAP.get("exchange") or {}).items():
        if needle in {str(e).upper() for e in exchanges}:
            return str(region)
    return ""


def yahoo_region_for(
    *,
    exchange: str = "",
    symbol: str = "",
    country: str = "",
) -> str:
    """Yahoo screener region for a listing (``in``, ``us``, ``gb``, …)."""
    region = _region_for_exchange(exchange)
    if region:
        return region
    sym = (symbol or "").upper()
    for suffix, exch in _SUFFIX_TO_EXCHANGE.items():
        if sym.endswith(suffix):
            region = _region_for_exchange(exch)
            if region:
                return region
    mapped = _COUNTRY_TO_YF_REGION.get((country or "").strip())
    if mapped:
        return mapped
    return "us"


def _screener_industry_name(industry: str, industry_key: str, sector: str) -> str:
    """Map yfinance ``.info`` industry text onto Yahoo's screener enum.

    Uses Yahoo's own ``SECTOR_INDUSTY_MAPPING``, not a project phrasebook.
    """
    try:
        from yfinance.const import SECTOR_INDUSTY_MAPPING
    except Exception:
        return ""
    pool: List[str] = []
    if sector and sector in SECTOR_INDUSTY_MAPPING:
        pool.extend(SECTOR_INDUSTY_MAPPING[sector])
    else:
        for names in SECTOR_INDUSTY_MAPPING.values():
            pool.extend(names)
    targets = [_compact_alnum(industry), _compact_alnum(industry_key)]
    targets = [t for t in targets if t]
    for name in pool:
        if _compact_alnum(name) in targets:
            return name
    return ""


def _screener_sector_name(sector: str) -> str:
    try:
        from yfinance.const import EQUITY_SCREENER_EQ_MAP
    except Exception:
        return sector
    valid = EQUITY_SCREENER_EQ_MAP.get("sector") or set()
    if sector in valid:
        return sector
    compact = _compact_alnum(sector)
    for name in valid:
        if _compact_alnum(str(name)) == compact:
            return str(name)
    return ""


def _parse_agent_peers(raw: str) -> List[str]:
    tokens = [tok.strip().upper() for tok in _PEER_SPLIT.split(raw or "") if tok.strip()]
    out: List[str] = []
    seen = set()
    for tok in tokens:
        if not _looks_like_ticker(tok) or tok in seen:
            continue
        seen.add(tok)
        out.append(resolve_ticker(tok))
    return out


def _run_equity_screen(query, size: int = 25):
    """Yahoo equity screener. Tests patch this — no network in CI."""
    from yfinance import screen as yf_screen

    return yf_screen(query, size=size, sortField="intradaymarketcap", sortAsc=False)


def _quotes_from_screen(query) -> List[Tuple[str, Optional[float]]]:
    try:
        result = _run_equity_screen(query)
    except Exception as exc:
        logger.warning("equity screen failed: %s", exc)
        return []
    quotes = (result or {}).get("quotes") if isinstance(result, dict) else None
    if not quotes:
        return []
    out: List[Tuple[str, Optional[float]]] = []
    for row in quotes:
        if not isinstance(row, dict):
            continue
        sym = str(row.get("symbol") or "").strip().upper()
        if not sym or not _looks_like_ticker(sym) or sym.startswith("^") or "=" in sym:
            continue
        raw_mcap = row.get("marketCap")
        if raw_mcap is None:
            raw_mcap = row.get("intradayMarketCap")
        mcap: Optional[float]
        try:
            mcap = float(raw_mcap) if raw_mcap is not None else None
        except (TypeError, ValueError):
            mcap = None
        out.append((sym, mcap))
    return out


def _build_equity_query(
    *,
    region: str,
    industry: str = "",
    sector: str = "",
    market_cap: Optional[float] = None,
    mcap_band: bool = False,
):
    from yfinance import EquityQuery

    clauses = []
    if region:
        clauses.append(EquityQuery("eq", ["region", region]))
    if industry:
        clauses.append(EquityQuery("eq", ["industry", industry]))
    elif sector:
        clauses.append(EquityQuery("eq", ["sector", sector]))
    if mcap_band and market_cap and market_cap > 0:
        clauses.append(EquityQuery("btwn", [
            "intradaymarketcap",
            market_cap * _MCAP_LO,
            market_cap * _MCAP_HI,
        ]))
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return EquityQuery("and", clauses)


def _rank_peers(
    focus: str,
    candidates: Iterable[Tuple[str, Optional[float]]],
    focus_mcap: Optional[float],
    limit: int,
) -> List[str]:
    focus_u = focus.upper()
    scored: List[Tuple[int, float, str]] = []
    seen = {focus_u}
    focus_sfx = _listing_suffix(focus_u)
    for raw_sym, mcap in candidates:
        sym = raw_sym.upper()
        if sym in seen:
            continue
        seen.add(sym)
        same_listing = 0 if _listing_suffix(sym) == focus_sfx else 1
        if focus_mcap and mcap and focus_mcap > 0 and mcap > 0:
            dist = abs(math.log(mcap) - math.log(focus_mcap))
        else:
            dist = 50.0
        scored.append((same_listing, dist, sym))
    scored.sort()
    out = [focus_u]
    for _, _, sym in scored:
        out.append(sym)
        if len(out) >= limit:
            break
    return out


def discover_peers(
    symbol: str,
    *,
    extra_peers: str = "",
    profile: Optional["MarketProfile"] = None,
    info: Optional[Dict[str, object]] = None,
) -> PeerUniverse:
    """Build a comparable set without a per-ticker dictionary.

    * ``extra_peers`` (agent-written yfinance tickers) always wins.
    * Otherwise screen Yahoo's live industry universe in this listing's
      country, ranked by listing family then log-market-cap distance.
    * If the industry+country screen is thinner than ``_MIN_OTHERS``,
      widen the *same live screen* (drop mcap band, then sector+country).
      That is query widening, not a hardcoded basket.
    """
    focus = (symbol or "").strip().upper()
    extra_key = (extra_peers or "").strip()
    cache_key = (focus, extra_key)
    if cache_key in _PEER_CACHE:
        return _PEER_CACHE[cache_key]

    agent_list = _parse_agent_peers(extra_key)
    if agent_list:
        symbols = [focus]
        for p in agent_list:
            if p.upper() not in {s.upper() for s in symbols}:
                symbols.append(p)
            if len(symbols) >= _MAX_PEERS + 2:
                break
        universe = PeerUniverse(
            symbols=symbols,
            method="agent",
            industry=(profile.industry if profile else ""),
            region=yahoo_region_for(
                exchange=profile.exchange if profile else "",
                symbol=focus,
                country=profile.country if profile else "",
            ),
            notes="analyst-supplied peers=; live screen not used",
        )
        _PEER_CACHE[cache_key] = universe
        return universe

    info = dict(info or {})
    if profile is not None:
        industry = profile.industry if profile.industry not in ("", "(unknown)") else ""
        sector = profile.sector if profile.sector not in ("", "(unknown)") else ""
        industry_key = profile.industry_key
        country = profile.country
        exchange = profile.exchange
        market_cap = profile.market_cap
    else:
        if not info:
            info = _info_safe(focus)
        industry = str(info.get("industry") or "").strip()
        sector = str(info.get("sector") or "").strip()
        industry_key = str(info.get("industryKey") or "").strip()
        country = str(info.get("country") or "").strip()
        exchange = str(info.get("exchange") or "").strip()
        raw_mcap = info.get("marketCap")
        try:
            market_cap = float(raw_mcap) if raw_mcap is not None else None
        except (TypeError, ValueError):
            market_cap = None

    region = yahoo_region_for(exchange=exchange, symbol=focus, country=country)
    industry_enum = _screener_industry_name(industry, industry_key, sector)
    sector_enum = _screener_sector_name(sector)

    attempts: List[Tuple[str, object]] = []
    if industry_enum and region:
        q = _build_equity_query(
            region=region, industry=industry_enum,
            market_cap=market_cap, mcap_band=bool(market_cap),
        )
        if q is not None:
            attempts.append(("industry+region+mcap" if market_cap else "industry+region", q))
        if market_cap:
            q2 = _build_equity_query(region=region, industry=industry_enum)
            if q2 is not None:
                attempts.append(("industry+region", q2))
    if sector_enum and region:
        q3 = _build_equity_query(
            region=region, sector=sector_enum,
            market_cap=market_cap, mcap_band=bool(market_cap),
        )
        if q3 is not None:
            attempts.append(("sector+region+mcap" if market_cap else "sector+region", q3))
    if industry_enum and market_cap:
        q4 = _build_equity_query(
            region="", industry=industry_enum,
            market_cap=market_cap, mcap_band=True,
        )
        if q4 is not None:
            attempts.append(("industry+mcap", q4))

    method = "focus_only"
    notes = "live industry screen returned no comparables"
    symbols = [focus]
    for label, query in attempts:
        picked = _rank_peers(focus, _quotes_from_screen(query), market_cap, _MAX_PEERS)
        others = [s for s in picked if s != focus]
        if len(others) >= _MIN_OTHERS:
            symbols = picked
            method = label
            notes = f"Yahoo equity screen ({label})"
            break
        if len(others) > len(symbols) - 1:
            symbols = picked
            method = label
            notes = f"Yahoo equity screen ({label}); thinner than {_MIN_OTHERS} others"

    if len(symbols) == 1:
        notes = (
            "live industry screen returned no comparables — pass peers= "
            "with yfinance tickers you already saw in a filing or tool output"
        )
        method = "focus_only"

    universe = PeerUniverse(
        symbols=symbols,
        method=method,
        industry=industry_enum or industry,
        region=region,
        notes=notes,
    )
    logger.info(
        "discover_peers %s method=%s region=%s industry=%s n=%d",
        focus, method, region, universe.industry, len(symbols),
    )
    _PEER_CACHE[cache_key] = universe
    return universe


def peer_basket(symbol: str, extra_peers: str = "") -> List[str]:
    """Symbols in the live (or agent-written) peer universe."""
    return list(discover_peers(symbol, extra_peers=extra_peers).symbols)


def _fmt_mcap(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    abs_v = abs(value)
    if abs_v >= 1e12:
        return f"{value / 1e12:.2f}T"
    if abs_v >= 1e9:
        return f"{value / 1e9:.2f}B"
    if abs_v >= 1e6:
        return f"{value / 1e6:.2f}M"
    return f"{value:.0f}"


def _fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2f}%"


def _fmt_pe(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}"


def _median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.median(values))


def format_peer_comparison(
    canonical: str,
    universe: PeerUniverse,
    rows: Sequence[PeerRow],
) -> str:
    """Render the peer table + tool-computed alpha (no mental arithmetic)."""
    focus = canonical.upper()
    focus_industry = ""
    for row in rows:
        if row.symbol.upper() == focus:
            focus_industry = row.industry
            break
    focus_compact = _compact_alnum(focus_industry)

    lines = [
        f"Sector peer comparison (method={universe.method}; "
        f"industry={universe.industry or '(unknown)'}; "
        f"region={universe.region or '(unknown)'}):",
    ]
    if universe.notes:
        lines.append(f"Universe: {universe.notes}")

    ytd_others: List[float] = []
    ret6_others: List[float] = []
    pe_others: List[float] = []
    focus_ytd: Optional[float] = None
    focus_6m: Optional[float] = None
    mismatch_count = 0

    for row in rows:
        marker = "  <-- focus" if row.symbol.upper() == focus else ""
        mismatch = ""
        if (
            focus_compact
            and row.industry
            and _compact_alnum(row.industry) != focus_compact
            and row.symbol.upper() != focus
        ):
            mismatch = "  INDUSTRY_MISMATCH"
            mismatch_count += 1
        lines.append(
            f"- {row.symbol}: YTD={_fmt_pct(row.ytd_pct)}, "
            f"6m={_fmt_pct(row.ret_6m_pct)}, "
            f"mcap={_fmt_mcap(row.market_cap)}, PE={_fmt_pe(row.pe)}, "
            f"industry={row.industry or '(unknown)'}, "
            f"country={row.country or '(unknown)'}"
            f"{marker}{mismatch}"
        )
        if row.symbol.upper() == focus:
            focus_ytd = row.ytd_pct
            focus_6m = row.ret_6m_pct
            continue
        if mismatch:
            continue
        if row.ytd_pct is not None:
            ytd_others.append(row.ytd_pct)
        if row.ret_6m_pct is not None:
            ret6_others.append(row.ret_6m_pct)
        if row.pe is not None and row.pe > 0:
            pe_others.append(row.pe)

    med_ytd = _median(ytd_others)
    med_6m = _median(ret6_others)
    med_pe = _median(pe_others)
    lines.append(
        f"Median peer YTD (ex-focus, matched industry): {_fmt_pct(med_ytd)}"
    )
    if focus_ytd is not None and med_ytd is not None:
        lines.append(
            f"Focus alpha vs median YTD: {focus_ytd - med_ytd:+.2f} pp"
        )
    else:
        lines.append(
            "Focus alpha vs median YTD: n/a — need the focus YTD and "
            "at least one matched-industry peer YTD"
        )
    lines.append(
        f"Median peer 6m (ex-focus, matched industry): {_fmt_pct(med_6m)}"
    )
    if focus_6m is not None and med_6m is not None:
        lines.append(
            f"Focus alpha vs median 6m: {focus_6m - med_6m:+.2f} pp"
        )
    lines.append(
        f"Median peer trailing PE (ex-focus, matched industry): {_fmt_pe(med_pe)}"
    )
    if mismatch_count:
        lines.append(
            f"{mismatch_count} INDUSTRY_MISMATCH row(s) — do not treat those "
            "as comparables. Re-call with peers= if the live set is wrong."
        )
    if universe.method == "focus_only":
        lines.append(
            "No live comparables. Pass peers= with yfinance tickers you "
            "already saw in a filing or tool output. Do not invent tickers."
        )
    else:
        lines.append(
            "Quote the alpha lines above — do not recompute. If this set "
            "is the wrong industry or country, re-call with peers=."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MarketProfile — the structured handle the tool layer consumes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketProfile:
    """Everything a downstream tool needs to query the right context.

    All fields are deterministic given the symbol — no LLM calls happen
    here, which is the whole point: the LLM should be reasoning about
    the *evidence*, not synthesising the country/sector classification.
    """

    symbol: str                         # canonical yfinance ticker
    raw_input: str                      # what the user typed
    long_name: str                      # ``Mazagon Dock Shipbuilders Ltd``
    country: str                        # ``India``
    currency: str                       # ``INR``
    exchange: str                       # ``NSI`` / ``NMS`` / …
    sector: str                         # ``Industrials``
    industry: str                       # ``Aerospace & Defense``
    industry_key: str = ""              # Yahoo kebab key (``software-infrastructure``)
    market_cap: Optional[float] = None
    peer_index: str = ""                # ``^NSEI``
    macro_tickers: Dict[str, str] = field(default_factory=dict)
    macro_themes: List[str] = field(default_factory=list)  # unused; kept for RunRecord shape
    industry_themes: List[str] = field(default_factory=list)
    peers: List[str] = field(default_factory=list)  # not a dictionary; live set is get_sector_peers

    def country_news_query(self, focus: str = "") -> str:
        base = compose_search_query(self, lens="company")
        return f"{base} {focus}".strip() if focus else base

    def macro_news_query(self) -> str:
        return compose_search_query(self, lens="macro")

    def industry_news_query(self) -> str:
        return compose_search_query(self, lens="industry")

    def geopolitical_query(self, agent_query: str = "") -> str:
        return compose_search_query(self, lens="geo", agent_query=agent_query)

    def supply_chain_query(self, agent_query: str = "") -> str:
        return compose_search_query(self, lens="supply", agent_query=agent_query)

    def to_brief(self) -> str:
        """Facts-only brief. Analysts write their own search queries from
        this + their desk remit + any gap on the blackboard."""
        macro_lines = "\n".join(
            f"  - {k}: {v}" for k, v in self.macro_tickers.items()
        )
        mcap_line = _fmt_mcap(self.market_cap)
        return (
            f"## Market context for {self.symbol}\n"
            f"- Name: {self.long_name or '(unknown)'}\n"
            f"- Country / currency / exchange: {self.country} / {self.currency} / {self.exchange}\n"
            f"- Sector / industry: {self.sector} / {self.industry}\n"
            f"- Market cap: {mcap_line}\n"
            f"- Benchmark index: {self.peer_index}\n"
            f"- Peer basket: live Yahoo industry + country screen via "
            f"get_sector_peers (not a ticker dictionary). Pass peers= if "
            f"filings already named comparables.\n"
            f"- Macro basket (yfinance symbols):\n{macro_lines}\n"
            f"- Query contract: after this brief, YOU write a 6-12 word "
            f"search query from (1) these facts, (2) your desk remit, "
            f"(3) any gap not already on the blackboard. Pass it as "
            f"the tool's ``query`` argument. There is no canned theme list.\n"
        )

    def to_dict(self) -> dict:
        return asdict(self)


def compose_search_query(
    profile: MarketProfile,
    *,
    lens: str,
    agent_query: str = "",
) -> str:
    """Build a web-search string.

    The analyst's ``query`` always wins — that is the self-service path
    (role remit + brief facts + knowledge gap). When the agent omits it,
    we compose only from live yfinance facts (name / country / industry)
    plus the *tool's* job verbs (tariff, supply chain, …). No industry
    phrasebook.
    """
    written = (agent_query or "").strip()
    if written:
        return written
    name = profile.long_name or profile.symbol
    country = profile.country or ""
    industry = profile.industry if profile.industry not in ("", "(unknown)") else (profile.sector or "")
    if lens == "geo":
        return f"{name} {country} {industry} tariff sanction export control".strip()
    if lens == "supply":
        return f"{name} {country} {industry} suppliers customers supply chain".strip()
    if lens == "macro":
        return f"{country or 'global'} economy latest central bank inflation currency policy news"
    if lens == "industry":
        return f"{name} {industry}".strip()
    # company / default
    return f"{name} ({profile.symbol}) latest stock news headlines this week"


def _info_safe(symbol: str) -> Dict[str, object]:
    """Fetch ``yfinance.Ticker(symbol).info`` and never raise.

    yfinance occasionally hits a 404 or a JSON parse error and surfaces
    it as an exception — we'd rather return an empty dict and degrade
    gracefully than fail an entire analyst run.
    """
    try:
        info = yf.Ticker(symbol).info or {}
    except Exception:
        info = {}
    return info


# Exchange-suffix → country bucket.  Deterministic and network-free (mirrors
# the resolver's ``_PROBE_SUFFIXES`` convention) so callers that only need the
# broad-market benchmark — e.g. the cascade controller distinguishing a
# systemic crisis from an idiosyncratic single-name shock — don't pay for a
# yfinance ``.info`` round-trip.
_SUFFIX_COUNTRY: Dict[str, str] = {
    ".NS": "India", ".BO": "India",
    ".L": "United Kingdom",
    ".HK": "Hong Kong",
    ".T": "Japan",
}


def benchmark_index_for(ticker: str) -> str:
    """Broad-market benchmark (``peer_index``) for ``ticker`` by exchange suffix.

    ``RELIANCE.NS`` → ``^NSEI`` (Nifty 50), ``AAPL`` → ``^GSPC`` (S&P 500),
    ``VOD.L`` → ``^FTSE``.  US is the default bucket for unsuffixed / ``=F`` /
    ``^`` symbols — matching :func:`_country_profile_for`'s default — so this
    is a pure ticker→index map, not a data fallback.
    """
    sym = (ticker or "").strip()
    country = _DEFAULT_COUNTRY_KEY
    for suffix, ctry in _SUFFIX_COUNTRY.items():
        if sym.upper().endswith(suffix.upper()):
            country = ctry
            break
    return str(_COUNTRY_PROFILE[country]["peer_index"])


def _country_profile_for(info: Dict[str, object]) -> Tuple[str, Dict[str, object]]:
    """Resolve ``info`` → country bucket from :data:`_COUNTRY_PROFILE`.

    yfinance reports ``country`` (``"India"``) directly for most listed
    equities.  When it's missing we look at the exchange suffix the
    resolver already added (``NSI`` -> India, ``NMS`` -> US) so we
    still pick the right macro basket.
    """
    country = str(info.get("country") or "").strip()
    if country in _COUNTRY_PROFILE:
        return country, _COUNTRY_PROFILE[country]

    exch = str(info.get("exchange") or "").upper()
    # yfinance exchange codes: NMS / NYQ → US ; NSI / BSE → India ;
    # LSE → UK ; HKG → Hong Kong ; JPX → Japan.
    if exch in {"NMS", "NYQ", "NCM", "NGM", "PCX", "BATS", "ASE"}:
        return "United States", _COUNTRY_PROFILE["United States"]
    if exch in {"NSI", "BSE"}:
        return "India", _COUNTRY_PROFILE["India"]
    if exch in {"LSE", "LON"}:
        return "United Kingdom", _COUNTRY_PROFILE["United Kingdom"]
    if exch in {"HKG"}:
        return "Hong Kong", _COUNTRY_PROFILE["Hong Kong"]
    if exch in {"JPX", "TSE"}:
        return "Japan", _COUNTRY_PROFILE["Japan"]
    return _DEFAULT_COUNTRY_KEY, _COUNTRY_PROFILE[_DEFAULT_COUNTRY_KEY]


# In-process MarketProfile cache so the same run doesn't pay the
# ``.info`` round-trip for every analyst tool that needs the brief.
_PROFILE_CACHE: Dict[str, MarketProfile] = {}


def get_market_profile(symbol: str) -> MarketProfile:
    """Build (and cache) the structured ``MarketProfile`` for ``symbol``.

    ``symbol`` should already be the canonical yfinance ticker — call
    :func:`resolve_ticker` first if you have a user-typed input.  The
    cache is keyed on the canonical symbol so cross-tool calls within
    one run reuse the same fetch.
    """
    if symbol in _PROFILE_CACHE:
        return _PROFILE_CACHE[symbol]

    info = _info_safe(symbol)
    country, country_profile = _country_profile_for(info)
    sector = str(info.get("sector") or "").strip()
    industry = str(info.get("industry") or "").strip()
    industry_key = str(info.get("industryKey") or "").strip()
    long_name = str(info.get("longName") or info.get("shortName") or symbol)
    raw_mcap = info.get("marketCap")
    try:
        market_cap = float(raw_mcap) if raw_mcap is not None else None
    except (TypeError, ValueError):
        market_cap = None

    profile = MarketProfile(
        symbol=symbol,
        raw_input=symbol,
        long_name=long_name,
        country=country,
        currency=str(country_profile.get("currency") or info.get("currency") or "USD"),
        exchange=str(info.get("exchange") or ""),
        sector=sector or "(unknown)",
        industry=industry or "(unknown)",
        industry_key=industry_key,
        market_cap=market_cap,
        peer_index=str(country_profile["peer_index"]),
        macro_tickers=dict(country_profile["macro_tickers"]),  # type: ignore[arg-type]
    )
    _PROFILE_CACHE[symbol] = profile
    return profile


def clear_profile_cache() -> None:
    """Test hook — empty the in-process profile + peer caches."""
    _PROFILE_CACHE.clear()
    _PEER_CACHE.clear()
