"""Ticker resolution + dynamic research context.

This module is the *single source of truth* for "what does this ticker
actually represent?" — the country, currency, sector, peer set, macro
basket and news themes the analyst team should ground its queries in.

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
   - ``news_themes`` — country/industry-relevant search seeds the
     news/geopolitical tools fan out over (e.g. "India shipbuilding
     budget", "Strait of Hormuz", "Make in India defence").

Provenance is preserved end-to-end: tools that build queries from a
profile still emit ``Source: …`` footers so the Reflective Critic can
audit every claim back to a real fetch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

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

    Real tickers are ≤ 12 characters, alphanumeric (plus a handful of
    structural characters), and never contain whitespace.  Anything
    else is rejected up front so we don't trigger an exchange-suffix
    probe storm on values that are obviously *not* tickers.
    """
    if not sym or len(sym) > 12:
        return False
    return all(ch in _TICKER_ALLOWED for ch in sym)


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

# Each country bucket carries:
#   - macro_tickers: yfinance symbols for the rates / FX / vol / commodity
#     baskets that actually move stocks in that country.
#   - peer_index: broad-market benchmark for alpha calculations.
#   - currency: ISO-4217 code (used to label macro figures).
#   - macro_themes: country-level macro news seeds (central bank,
#     fiscal policy, currency, key bilateral relationships).
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
        "macro_themes": [
            "Federal Reserve interest rate decision",
            "US treasury yield curve",
            "US CPI inflation print",
            "US dollar index DXY",
            "US-China trade policy",
        ],
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
        "macro_themes": [
            "Reserve Bank of India RBI repo rate",
            "India CPI inflation WPI",
            "INR USD exchange rate",
            "India Union Budget capex allocation",
            "FII DII flows Indian equities",
            "India crude oil import bill",
        ],
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
        "macro_themes": [
            "Bank of England BoE base rate",
            "UK CPI inflation",
            "GBP USD exchange rate",
            "UK fiscal statement gilts",
            "EU UK trade",
        ],
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
        "macro_themes": [
            "PBOC monetary policy China",
            "Hong Kong China policy risk",
            "HKD USD peg",
            "Hang Seng index flows",
        ],
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
        "macro_themes": [
            "Bank of Japan BoJ yield curve control",
            "JPY USD exchange rate",
            "Japan CPI core inflation",
            "Nikkei index foreign flows",
        ],
    },
}

# Default fallback when ``.info["country"]`` is missing or unmapped — we
# still want *something* better than US-only.
_DEFAULT_COUNTRY_KEY = "United States"


# ---------------------------------------------------------------------------
# Industry → product-level news seeds
# ---------------------------------------------------------------------------
# Mapping is intentionally coarse — the LLM still does the final phrasing.
# Each key matches a substring (case-insensitive) of the yfinance
# ``.info["industry"]`` or ``.info["sector"]`` field.
_INDUSTRY_SEEDS: Dict[str, List[str]] = {
    # Defence + shipbuilding (MAZDOCK, BEL, HAL, BAE, Lockheed, …)
    "ship": [
        "shipbuilding order book",
        "naval defence contract",
        "submarine destroyer frigate procurement",
        "ship insurance war risk premium",
        "Strait of Hormuz shipping disruption",
        "Red Sea Suez Canal shipping",
    ],
    "aerospace & defense": [
        "defence budget allocation",
        "defence export contract",
        "aircraft procurement order",
        "drone missile contract",
    ],
    "defense": [
        "defence budget allocation",
        "defence export contract",
        "aircraft procurement order",
        "drone missile contract",
    ],
    # Semis (NVDA, AMD, TSM, ASML, INTC)
    "semiconductor": [
        "chip export controls",
        "TSMC fab capacity",
        "ASML EUV equipment",
        "GPU AI accelerator demand",
        "Taiwan strait risk",
    ],
    # Software (NTNX, MSFT, ORCL)
    "software": [
        "enterprise IT spending",
        "cloud capex hyperscaler",
        "AI software demand",
    ],
    # Oil & gas (XOM, RELIANCE, BP)
    "oil & gas": [
        "OPEC production quota",
        "crude oil inventory EIA",
        "natural gas storage",
        "refinery margins crack spread",
    ],
    # Banks (HDFCBANK, JPM, HSBC)
    "bank": [
        "central bank policy rate",
        "credit growth NPA",
        "net interest margin",
        "bank earnings deposit growth",
    ],
    # Pharma (SUN, DRREDDY, PFE)
    "pharmaceutical": [
        "FDA approval drug pipeline",
        "USFDA inspection observations",
        "generic drug pricing",
    ],
    "drug": [
        "FDA approval drug pipeline",
        "USFDA inspection observations",
        "generic drug pricing",
    ],
    # Auto (TSLA, M&M, MARUTI)
    "auto": [
        "EV adoption subsidies",
        "lithium battery supply",
        "monthly vehicle sales",
        "auto chip supply",
    ],
    # Steel / metals (TATASTEEL, JSWSTEEL, X)
    "steel": [
        "steel prices HRC",
        "iron ore China demand",
        "steel export duty",
    ],
    "metals & mining": [
        "metals prices LME",
        "iron ore copper aluminum demand",
        "mining royalty policy",
    ],
}


def _seeds_for_industry(industry: str, sector: str, name: str = "") -> List[str]:
    """Return the union of seeds matching the industry / sector / name text.

    We deliberately also look at the company's long name because
    yfinance industry tags are coarse — a shipbuilder like Mazagon Dock
    is classified ``Aerospace & Defense``, so the "ship" seed bucket
    is only reachable via the name string.  Sub-industry granularity
    via the name is the cheapest way to keep the seed taxonomy small.
    """
    hay = f"{industry} {sector} {name}".lower()
    out: List[str] = []
    for key, seeds in _INDUSTRY_SEEDS.items():
        if key in hay:
            out.extend(seeds)
    # Dedupe while preserving order.
    seen = set()
    deduped = []
    for s in out:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped


# ---------------------------------------------------------------------------
# Sector peer baskets (kept ticker-agnostic so works for both US + India)
# ---------------------------------------------------------------------------

_SECTOR_PEERS: Dict[str, List[str]] = {
    # --- US ---
    "NVDA": ["NVDA", "AMD", "AVGO", "INTC", "TSM", "QCOM"],
    "AAPL": ["AAPL", "MSFT", "GOOGL", "META"],
    "TSLA": ["TSLA", "GM", "F", "RIVN"],
    "NTNX": ["NTNX", "DELL", "HPE", "PSTG", "NTAP", "IBM"],
    # --- India: shipbuilding + defence ---
    "MAZDOCK.NS": ["MAZDOCK.NS", "COCHINSHIP.NS", "GRSE.NS", "BEL.NS", "HAL.NS"],
    "COCHINSHIP.NS": ["COCHINSHIP.NS", "MAZDOCK.NS", "GRSE.NS", "BEL.NS", "HAL.NS"],
    "GRSE.NS": ["GRSE.NS", "MAZDOCK.NS", "COCHINSHIP.NS", "BEL.NS", "HAL.NS"],
    "BEL.NS": ["BEL.NS", "HAL.NS", "MAZDOCK.NS", "BDL.NS"],
    "HAL.NS": ["HAL.NS", "BEL.NS", "MAZDOCK.NS", "BDL.NS"],
    # --- India: banks ---
    "HDFCBANK.NS": ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "SBIN.NS"],
    "ICICIBANK.NS": ["ICICIBANK.NS", "HDFCBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "SBIN.NS"],
    # --- India: IT services ---
    "TCS.NS": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
    "INFY.NS": ["INFY.NS", "TCS.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
    # --- India: energy / refining ---
    "RELIANCE.NS": ["RELIANCE.NS", "ONGC.NS", "IOC.NS", "BPCL.NS", "HINDPETRO.NS"],
}


def peer_basket(symbol: str) -> List[str]:
    """Return the peer basket for ``symbol`` (canonical, suffixed).

    Falls back to ``[symbol]`` when no curated basket exists — the
    Sector Analyst tool then degrades gracefully to a single-line
    "no peers configured" entry, which is still better than an empty
    response that hides the gap.
    """
    sym = symbol.upper()
    if sym in _SECTOR_PEERS:
        return list(_SECTOR_PEERS[sym])
    # Strip the exchange suffix and retry — lets us match a US-style
    # bare symbol against the curated baskets without duplicating keys.
    base = sym.split(".")[0]
    if base in _SECTOR_PEERS:
        return list(_SECTOR_PEERS[base])
    return [sym]


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
    peer_index: str                     # ``^NSEI``
    macro_tickers: Dict[str, str]       # ordered macro basket
    macro_themes: List[str]             # country-level news seeds
    industry_themes: List[str]          # industry-level news seeds
    peers: List[str]                    # peer basket

    def country_news_query(self, focus: str = "") -> str:
        """Compose a country-aware news query.

        ``focus`` is an optional sub-topic (e.g. ``"earnings"``) that
        gets prefixed.  The macro themes are appended so the search
        engine returns truly country-relevant results — for an Indian
        name we don't want "US treasury yields" dominating.
        """
        parts = [self.long_name or self.symbol]
        if focus:
            parts.append(focus)
        # Up to 3 themes; more than that confuses Tavily ranking.
        parts.extend(self.macro_themes[:3])
        return " ".join(parts)

    def industry_news_query(self) -> str:
        """News query keyed off the company's industry, not generic terms."""
        seeds = self.industry_themes[:4] if self.industry_themes else [self.industry]
        return f"{self.long_name or self.symbol} {' '.join(seeds)}"

    def geopolitical_query(self) -> str:
        """Country + industry-specific geopolitical / regulatory query.

        Replaces the old hard-coded ``"tariffs export controls sanctions
        China Taiwan"`` query.  An Indian shipbuilder gets ``"Strait of
        Hormuz Red Sea shipping naval defence procurement"`` instead.

        We use the FULL industry-theme list (not the first 3 only)
        because the geopolitical lens is exactly where the rarer
        chokepoint themes (Hormuz, Suez, Red Sea, Black Sea) live —
        they would be the first to be dropped by an aggressive slice.
        """
        country = self.country or ""
        industry_words = " ".join(self.industry_themes) if self.industry_themes else ""
        return (
            f"{self.long_name or self.symbol} "
            f"{country} regulation tariff sanction policy "
            f"{industry_words}"
        ).strip()

    def supply_chain_query(self) -> str:
        """Supply-chain / customer-concentration query, sector-aware."""
        industry_words = " ".join(self.industry_themes[:2]) if self.industry_themes else ""
        return (
            f"{self.long_name or self.symbol} suppliers customers "
            f"revenue concentration supply chain {industry_words}"
        ).strip()

    def to_brief(self) -> str:
        """Render the human-readable brief surfaced via ``get_market_context``.

        Analysts read this *first* and use it to phrase their searches —
        so the brief is short, citable, and lists themes the analyst
        layer can keep / discard rather than inventing its own.
        """
        macro_lines = "\n".join(
            f"  - {k}: {v}" for k, v in self.macro_tickers.items()
        )
        peer_str = ", ".join(self.peers)
        macro_seeds = "; ".join(self.macro_themes[:5])
        ind_seeds = "; ".join(self.industry_themes) if self.industry_themes else "(no industry seeds configured — phrase your own)"
        return (
            f"## Market context for {self.symbol}\n"
            f"- Name: {self.long_name or '(unknown)'}\n"
            f"- Country / currency / exchange: {self.country} / {self.currency} / {self.exchange}\n"
            f"- Sector / industry: {self.sector} / {self.industry}\n"
            f"- Benchmark index: {self.peer_index}\n"
            f"- Peer basket: {peer_str}\n"
            f"- Macro basket (yfinance symbols):\n{macro_lines}\n"
            f"- Country news themes: {macro_seeds}\n"
            f"- Industry news themes: {ind_seeds}\n"
        )

    def to_dict(self) -> dict:
        return asdict(self)


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
    long_name = str(info.get("longName") or info.get("shortName") or symbol)
    industry_themes = _seeds_for_industry(industry, sector, long_name)

    profile = MarketProfile(
        symbol=symbol,
        raw_input=symbol,
        long_name=long_name,
        country=country,
        currency=str(country_profile.get("currency") or info.get("currency") or "USD"),
        exchange=str(info.get("exchange") or ""),
        sector=sector or "(unknown)",
        industry=industry or "(unknown)",
        peer_index=str(country_profile["peer_index"]),
        macro_tickers=dict(country_profile["macro_tickers"]),  # type: ignore[arg-type]
        macro_themes=list(country_profile["macro_themes"]),     # type: ignore[arg-type]
        industry_themes=industry_themes,
        peers=peer_basket(symbol),
    )
    _PROFILE_CACHE[symbol] = profile
    return profile


def clear_profile_cache() -> None:
    """Test hook — empty the in-process profile cache."""
    _PROFILE_CACHE.clear()
