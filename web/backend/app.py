"""FastAPI server for TradingCrew UI.

Routes
------
* GET  /                    -> serves the single-page frontend
* GET  /static/*            -> static assets (js / css)
* GET  /api/options         -> dropdowns: tickers, tools, agents, indicators, defaults
* GET  /api/chart           -> OHLCV + indicator series for the Charts tab
* WS   /ws/analyze          -> first text frame is JSON config, then events stream
"""

import asyncio
import json
import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Load env from the project root (the parent of the ``web/`` dir)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from .charts import SUPPORTED_INDICATORS, build_chart_payload  # noqa: E402
from .logging_setup import configure_logging  # noqa: E402
from .runner import AnalysisRunner, get_agent_catalog, get_tool_catalog  # noqa: E402

# Install stdout + rotating-file logging before any module-level
# logger calls happen.  See ``logging_setup.py`` for the env knobs.
_LOG_PATH = configure_logging()
logger = logging.getLogger(__name__)
logger.info("Logging to %s (stdout mirror also enabled)", _LOG_PATH)

BACKEND_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BACKEND_DIR.parent / "frontend"


app = FastAPI(title="TradingCrew UI", version="0.2.0")

if FRONTEND_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(FRONTEND_DIR), html=False),
        name="static",
    )


PRESET_TICKERS_STOCK = [
    # ---- US ----
    {"symbol": "NVDA", "name": "NVIDIA", "country": "US"},
    {"symbol": "AAPL", "name": "Apple", "country": "US"},
    {"symbol": "MSFT", "name": "Microsoft", "country": "US"},
    {"symbol": "GOOG", "name": "Alphabet", "country": "US"},
    {"symbol": "AMZN", "name": "Amazon", "country": "US"},
    {"symbol": "META", "name": "Meta", "country": "US"},
    {"symbol": "TSLA", "name": "Tesla", "country": "US"},
    {"symbol": "AMD", "name": "AMD", "country": "US"},
    {"symbol": "NTNX", "name": "Nutanix", "country": "US"},
    {"symbol": "SPY", "name": "S&P 500 ETF", "country": "US"},
    # ---- India (NSE) ----
    # yfinance needs the ``.NS`` suffix for NSE listings.  The resolver
    # in trading_crew.market_context auto-adds it for unsuffixed input
    # (e.g. ``MAZDOCK`` -> ``MAZDOCK.NS``), but the preset list uses the
    # explicit form so the symbol that goes over the wire is unambiguous.
    {"symbol": "RELIANCE.NS", "name": "Reliance Industries", "country": "IN"},
    {"symbol": "TCS.NS", "name": "Tata Consultancy Services", "country": "IN"},
    {"symbol": "INFY.NS", "name": "Infosys", "country": "IN"},
    {"symbol": "HDFCBANK.NS", "name": "HDFC Bank", "country": "IN"},
    {"symbol": "ICICIBANK.NS", "name": "ICICI Bank", "country": "IN"},
    {"symbol": "MAZDOCK.NS", "name": "Mazagon Dock Shipbuilders", "country": "IN"},
    {"symbol": "COCHINSHIP.NS", "name": "Cochin Shipyard", "country": "IN"},
    {"symbol": "BEL.NS", "name": "Bharat Electronics", "country": "IN"},
    {"symbol": "HAL.NS", "name": "Hindustan Aeronautics", "country": "IN"},
    {"symbol": "TATAMOTORS.NS", "name": "Tata Motors", "country": "IN"},
]

# Commodity futures — yfinance uses the ``=F`` suffix for continuous front-month
# series.  The "category" tag groups them in the UI sidebar (energy / metals
# / grains / softs) so users can navigate large universes by sector.
PRESET_TICKERS_COMMODITY = [
    {"symbol": "CL=F", "name": "WTI Crude Oil", "category": "energy"},
    {"symbol": "BZ=F", "name": "Brent Crude Oil", "category": "energy"},
    {"symbol": "NG=F", "name": "Natural Gas", "category": "energy"},
    {"symbol": "HO=F", "name": "Heating Oil", "category": "energy"},
    {"symbol": "RB=F", "name": "RBOB Gasoline", "category": "energy"},
    {"symbol": "GC=F", "name": "Gold", "category": "metals"},
    {"symbol": "SI=F", "name": "Silver", "category": "metals"},
    {"symbol": "HG=F", "name": "Copper", "category": "metals"},
    {"symbol": "PL=F", "name": "Platinum", "category": "metals"},
    {"symbol": "PA=F", "name": "Palladium", "category": "metals"},
    {"symbol": "ZC=F", "name": "Corn", "category": "grains"},
    {"symbol": "ZS=F", "name": "Soybeans", "category": "grains"},
    {"symbol": "ZW=F", "name": "Wheat", "category": "grains"},
    {"symbol": "ZM=F", "name": "Soybean Meal", "category": "grains"},
    {"symbol": "ZL=F", "name": "Soybean Oil", "category": "grains"},
    {"symbol": "KC=F", "name": "Coffee", "category": "softs"},
    {"symbol": "CC=F", "name": "Cocoa", "category": "softs"},
    {"symbol": "SB=F", "name": "Sugar #11", "category": "softs"},
    {"symbol": "CT=F", "name": "Cotton", "category": "softs"},
    {"symbol": "LE=F", "name": "Live Cattle", "category": "livestock"},
    {"symbol": "HE=F", "name": "Lean Hogs", "category": "livestock"},
]

PRESET_TICKERS = PRESET_TICKERS_STOCK  # back-compat for callers without asset_class


# HTML shells must always revalidate so the browser picks up new JS/CSS
# bundles as soon as we ship them (otherwise stale caches surface as
# "missing tools / empty sidebar" bug reports). Static assets keep their
# default ETag caching since the HTML is the cache-key entry point.
_HTML_NO_CACHE = {
    "Cache-Control": "no-cache, must-revalidate",
    "Pragma": "no-cache",
}


@app.get("/")
async def homepage() -> FileResponse:
    """Landing page with two dashboard cards (Stock / Commodity).

    Both dashboards run on the same M1-M7 backbone but with asset-class-
    specific crews and tools — the homepage just lets the user pick.
    """
    home_path = FRONTEND_DIR / "homepage.html"
    if not home_path.exists():
        raise HTTPException(status_code=500, detail="Homepage not found")
    return FileResponse(str(home_path), headers=_HTML_NO_CACHE)


@app.get("/stock")
async def stock_dashboard() -> FileResponse:
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="Stock dashboard not found")
    return FileResponse(str(index_path), headers=_HTML_NO_CACHE)


@app.get("/commodity")
async def commodity_dashboard() -> FileResponse:
    """Same SPA shell as /stock — the client reads window.location.pathname
    to discriminate asset_class, and the API endpoints accept an
    ``asset_class`` query param to surface the right ticker presets and
    route the websocket to the right crew.
    """
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="Commodity dashboard not found")
    return FileResponse(str(index_path), headers=_HTML_NO_CACHE)


@app.get("/api/options")
async def options(
    asset_class: str = Query("stock", description="stock | commodity"),
) -> JSONResponse:
    """UI bootstrap payload.

    Returns asset-class-specific ticker presets, tool/agent catalogs, and
    LLM provider info. ``asset_class`` defaults to "stock" for back-compat
    with the original single-dashboard layout.
    """
    # Mirror the precedence in trading_crew/_common.py: VLLM_* env wins over
    # LOCAL_*. The UI then surfaces the model that will actually be used.
    llm_base = os.getenv("VLLM_LLM_BASE_URL") or os.getenv("LOCAL_LLM_BASE_URL")
    llm_model = os.getenv("VLLM_LLM_MODEL") or os.getenv("LOCAL_LLM_MODEL")

    if asset_class == "commodity":
        tickers = PRESET_TICKERS_COMMODITY
        default_ticker = os.getenv("TRADINGCREW_DEFAULT_COMMODITY", "CL=F")
        try:
            from commodity_crew.tools import get_tool_catalog as get_comm_tool_catalog
            from commodity_crew.crew import get_agent_catalog as get_comm_agent_catalog
            tools = get_comm_tool_catalog()
            agents = get_comm_agent_catalog()
        except Exception as exc:
            logger.warning("commodity_crew not yet wired: %s", exc)
            tools = []
            agents = []
    else:
        tickers = PRESET_TICKERS_STOCK
        default_ticker = os.getenv("TRADINGCREW_DEFAULT_TICKER", "NTNX")
        tools = get_tool_catalog()
        agents = get_agent_catalog()

    from trading_crew.llm_presets import list_presets
    from trading_crew.embedding_presets import (
        list_presets as list_embedding_presets,
    )

    return JSONResponse({
        "asset_class": asset_class,
        "tickers": tickers,
        "default_ticker": default_ticker,
        "default_trade_date": date.today().isoformat(),
        "default_debate_rounds": int(os.getenv("TRADINGCREW_DEFAULT_DEBATE", "2")),
        "default_risk_rounds": int(os.getenv("TRADINGCREW_DEFAULT_RISK", "1")),
        "tools": tools,
        "agents": agents,
        "chart_indicators": SUPPORTED_INDICATORS,
        "llm": {
            "model": llm_model,
            "base_url": llm_base,
            "provider": os.getenv("LLM_PROVIDER_PREFIX", "hosted_vllm"),
        },
        "llm_presets": list_presets(),
        "default_llm_preset": os.getenv(
            "TRADINGCREW_DEFAULT_LLM_PRESET", "hosted-vllm-oss"
        ),
        "embedding_presets": list_embedding_presets(),
        "default_embedding_preset": os.getenv(
            "TRADINGCREW_DEFAULT_EMBEDDING_PRESET", "vllm-embedding"
        ),
        "tavily_configured": bool(os.getenv("TAVILY_API_KEY")),
    })


@app.get("/api/commodity/curve")
async def commodity_curve(
    ticker: str = Query(..., min_length=2, description="Continuous futures symbol e.g. CL=F"),
    n_months: int = Query(12, ge=2, le=24),
) -> JSONResponse:
    """Term-structure (futures curve) for a commodity.

    Pulls the next ``n_months`` delivery contracts and reports their last
    closes + a CONTANGO/BACKWARDATION classification + an annualised
    roll-yield approximation.  Data source: yfinance.
    """
    def _build():
        from commodity_crew.tools import (
            _futures_chain_symbols, _root_of, COMMODITY_META,
        )
        import yfinance as yf
        root = _root_of(ticker)
        if root not in COMMODITY_META:
            return {"ticker": ticker, "error": f"No curve metadata for {root}"}
        symbols = _futures_chain_symbols(root, n_months=n_months)
        contracts = []
        for sym in symbols:
            try:
                hist = yf.Ticker(sym).history(period="5d")
                if hist.empty:
                    continue
                close = float(hist["Close"].dropna().iloc[-1])
                contracts.append({"symbol": sym, "close": round(close, 4)})
            except Exception:
                continue
        if len(contracts) < 2:
            return {
                "ticker": ticker, "root": root,
                "contracts": contracts,
                "structure": "UNKNOWN",
                "error": f"Only {len(contracts)} of {n_months} contracts had data",
            }
        front, back = contracts[0]["close"], contracts[-1]["close"]
        slope_pct = (back / front - 1.0) * 100.0
        if slope_pct > 0.3:
            structure = "CONTANGO"
        elif slope_pct < -0.3:
            structure = "BACKWARDATION"
        else:
            structure = "FLAT"
        ann_roll = (-slope_pct / max(1, n_months)) * 12.0
        return {
            "ticker": ticker, "root": root,
            "name": COMMODITY_META[root]["name"],
            "contract_size": COMMODITY_META[root]["contract_size"],
            "unit": COMMODITY_META[root]["unit"],
            "contracts": contracts,
            "structure": structure,
            "front_back_slope_pct": round(slope_pct, 3),
            "ann_roll_yield_pct": round(ann_roll, 3),
        }
    return JSONResponse(await asyncio.to_thread(_build))


@app.get("/api/commodity/cot")
async def commodity_cot(
    ticker: str = Query(..., min_length=2),
    weeks: int = Query(12, ge=1, le=52),
) -> JSONResponse:
    """CFTC Commitments-of-Traders positioning history (last ``weeks`` weeks).

    Returns Managed-Money and Commercial long/short/net positioning so the
    UI can chart the trend and flag extreme positioning.
    """
    def _build():
        import csv as _csv
        import datetime as _d
        from commodity_crew.tools import _download_cftc, _meta_for
        meta = _meta_for(ticker)
        cftc_name = meta.get("cftc_name")
        if not cftc_name:
            return {"ticker": ticker, "error": "No CFTC mapping for this commodity"}
        path = _download_cftc(_d.date.today().year) or _download_cftc(_d.date.today().year - 1)
        if path is None or not path.exists():
            return {"ticker": ticker, "error": "CFTC COT data temporarily unavailable"}
        rows = []
        try:
            with path.open("r", encoding="latin-1") as fh:
                reader = _csv.DictReader(fh)
                for row in reader:
                    # Legacy file uses spaces in column headers; disaggregated
                    # uses underscores.  Handle both shapes uniformly.
                    market = (
                        row.get("Market and Exchange Names")
                        or row.get("Market_and_Exchange_Names")
                        or ""
                    )
                    if cftc_name in market:
                        rows.append(row)
        except Exception as exc:
            return {"ticker": ticker, "error": f"CFTC parse failed: {exc}"}
        rows.sort(
            key=lambda r: r.get("As of Date in Form YYYY-MM-DD") or r.get("Report_Date_as_YYYY-MM-DD") or "",
            reverse=True,
        )
        rows = rows[:weeks]
        out_rows = []
        for r in rows:
            try:
                date_s = (
                    r.get("As of Date in Form YYYY-MM-DD")
                    or r.get("Report_Date_as_YYYY-MM-DD")
                    or ""
                )
                comm_long = int(float(
                    r.get("Commercial Positions-Long (All)")
                    or r.get("Comm_Positions_Long_All", "0") or "0"
                ))
                comm_short = int(float(
                    r.get("Commercial Positions-Short (All)")
                    or r.get("Comm_Positions_Short_All", "0") or "0"
                ))
                ncomm_long = int(float(
                    r.get("Noncommercial Positions-Long (All)")
                    or r.get("NonComm_Positions_Long_All", "0") or "0"
                ))
                ncomm_short = int(float(
                    r.get("Noncommercial Positions-Short (All)")
                    or r.get("NonComm_Positions_Short_All", "0") or "0"
                ))
            except Exception:
                continue
            out_rows.append({
                "date": date_s,
                "managed_money_long": ncomm_long,
                "managed_money_short": ncomm_short,
                "managed_money_net": ncomm_long - ncomm_short,
                "commercials_long": comm_long,
                "commercials_short": comm_short,
                "commercials_net": comm_long - comm_short,
            })
        # Reverse so oldest first for the chart
        out_rows.reverse()
        return {
            "ticker": ticker,
            "cftc_name": cftc_name,
            "name": meta.get("name"),
            "rows": out_rows,
            "count": len(out_rows),
        }
    return JSONResponse(await asyncio.to_thread(_build))


@app.get("/api/commodity/seasonality")
async def commodity_seasonality(
    ticker: str = Query(..., min_length=2),
    years: int = Query(5, ge=2, le=20),
) -> JSONResponse:
    """Average monthly returns over the last N years for a futures ticker."""
    def _build():
        import yfinance as yf
        import pandas as pd
        df = yf.Ticker(ticker).history(period=f"{years}y")
        if df is None or df.empty:
            return {"ticker": ticker, "years": years, "error": "No data"}
        df = df[["Close"]].dropna().copy()
        df.index = pd.to_datetime(df.index)
        df["year"] = df.index.year
        df["month"] = df.index.month
        monthly_last = df.groupby(["year", "month"]).last().reset_index()
        monthly_last["ret"] = monthly_last["Close"].pct_change()
        agg = monthly_last.groupby("month")["ret"].agg(["mean", "std", "count"])
        months = []
        names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        for m in range(1, 13):
            if m not in agg.index:
                continue
            row = agg.loc[m]
            months.append({
                "month": m,
                "label": names[m - 1],
                "mean_return_pct": round(float(row["mean"]) * 100, 3),
                "std_return_pct": round(float(row["std"]) * 100, 3) if pd.notna(row["std"]) else None,
                "n_years": int(row["count"]),
            })
        return {"ticker": ticker, "years": years, "months": months}
    return JSONResponse(await asyncio.to_thread(_build))


@app.get("/api/chart")
async def chart(
    ticker: str = Query(..., min_length=1),
    trade_date: str = Query(default_factory=lambda: date.today().isoformat()),
    lookback_days: int = Query(180, ge=30, le=365 * 3),
) -> JSONResponse:
    payload = await asyncio.to_thread(
        build_chart_payload, ticker, trade_date, lookback_days
    )
    return JSONResponse(payload)


@app.get("/api/memory/retrieve")
async def memory_retrieve(
    ticker: str = Query(..., min_length=1),
    as_of: str = Query(...),
    query: str = Query("", description="Free-text query; defaults to ticker"),
    k: int = Query(5, ge=1, le=20),
    regime: str = Query("", description="Optional regime tag (TREND/RANGE/HIGH_VOL/CRISIS) for regime-aware boost"),
) -> JSONResponse:
    """M3 — top-k episodes with outcome-embargo enforced server-side."""
    from trading_crew.agentic.memory import EpisodicMemory, Regime
    from trading_crew.agentic.memory.embedding import make_memory

    cache_dir = os.environ.get("TRADINGCREW_CACHE_DIR") or os.path.expanduser("~/.trading_crew")
    store_path = Path(cache_dir) / "memory" / "episodes.jsonl"
    mem = make_memory(store_path)

    # Optional regime filter: e.g. ?regime=TREND boosts same-regime
    # episodes via the regime_match_bonus we just added.  An invalid
    # value is silently ignored rather than 422'd to keep the UI simple.
    regime_enum = None
    if regime:
        try:
            regime_enum = Regime(regime.upper())
        except ValueError:
            regime_enum = None

    results = await asyncio.to_thread(
        mem.retrieve, query or ticker, as_of=as_of, k=k, symbol=ticker, regime=regime_enum,
    )

    # Surface the embedder + retrieval-count diagnostics so the Memory
    # tab can show "powered by vLLM (embeddinggemma-300m)" instead of
    # silently switching backends.
    embedder = os.environ.get("TRADINGCREW_MEMORY_EMBEDDER", "tfidf")
    return JSONResponse({
        "ticker": ticker,
        "as_of": as_of,
        "regime": regime_enum.value if regime_enum else None,
        "embedder": embedder,
        "results": [
            {
                "episode_id": r.episode.episode_id,
                "symbol": r.episode.symbol,
                "regime": r.episode.regime.value,
                "decision_ts": r.episode.decision_ts,
                "outcome_ts": r.episode.outcome_ts,
                "realised_return": r.episode.realised_return,
                "alpha_return": r.episode.alpha_return,
                "reflection": r.episode.reflection,
                "similarity": round(r.similarity, 4),
                "decay_factor": round(r.decay_factor, 4),
                "score": round(r.score, 4),
                "delta_days": round(r.delta_days, 2),
                "retrieval_count": getattr(r.episode, "retrieval_count", 0),
                "schema_version": getattr(r.episode, "schema_version", 0),
            }
            for r in results
        ],
    })


@app.post("/api/memory/evict")
async def memory_evict(
    max_records: int = Query(10000, ge=0, description="Cap on total kept episodes (0 = unlimited)"),
    max_age_days: int = Query(365, ge=0, description="Drop episodes older than this; 0 disables age-based eviction"),
    min_retrieval_count: int = Query(0, ge=0, description="Episodes above this many retrievals survive aging"),
) -> JSONResponse:
    """Prune the episodic memory store.

    Implements paper §M3 "pruning policy".  Conservative defaults (10k
    cap, 365 day window, keep anything ever retrieved) so a single
    accidental click can't wipe a year of audit history.  The Memory
    tab exposes this as the "Prune older than N" button.
    """
    from trading_crew.agentic.memory.embedding import make_memory

    cache_dir = os.environ.get("TRADINGCREW_CACHE_DIR") or os.path.expanduser("~/.trading_crew")
    store_path = Path(cache_dir) / "memory" / "episodes.jsonl"
    mem = make_memory(store_path)

    removed = await asyncio.to_thread(
        mem.evict,
        max_records=(max_records or None),
        max_age_days=(max_age_days or None),
        min_retrieval_count=min_retrieval_count,
    )
    remaining = len(list(mem.iter_all()))
    return JSONResponse({
        "removed": removed,
        "remaining": remaining,
        "max_records": max_records,
        "max_age_days": max_age_days,
        "min_retrieval_count": min_retrieval_count,
    })


@app.post("/api/memory/resolve")
async def memory_resolve(
    ticker: Optional[str] = Query(None, description="Optional ticker filter; resolve only this symbol's episodes"),
    skip_llm: bool = Query(False, description="If true, write outcomes but skip the LLM reflection step (useful for batch / CI)"),
) -> JSONResponse:
    """Agentic training L2 — sweep pending episodes whose outcome window
    has elapsed, compute realised return + alpha + max drawdown, and write
    an LLM-generated lesson back into each episode.

    The reflections are immediately visible to future runs via the
    ``retrieve_past_episodes`` tool (M3 + Phase C) because the M3
    outcome-embargo unlocks an episode for retrieval once its
    ``outcome_ts`` has passed.

    Returns
    -------
    Per-episode resolution records (RESOLVED / ABANDONED / SKIPPED_*) so the
    UI can render a transparent audit trail of what was learned.
    """
    from trading_crew.agentic.reflection import resolve_pending_episodes
    from trading_crew._common import get_llm  # same factory the critic and runner use

    llm = None
    note = ""
    if not skip_llm:
        try:
            llm = get_llm(temperature=0.2)
        except Exception as exc:
            note = f"LLM unavailable ({exc}); writing outcomes without reflections"
            skip_llm = True

    records = await asyncio.to_thread(
        resolve_pending_episodes,
        ticker=ticker, llm=llm, skip_llm=skip_llm,
    )
    return JSONResponse({
        "ticker": ticker,
        "skip_llm": skip_llm,
        "note": note,
        "count": len(records),
        "by_status": _count_statuses(records),
        "records": [r.to_dict() for r in records],
    })


def _count_statuses(records) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in records:
        counts[r.status] = counts.get(r.status, 0) + 1
    return counts


@app.get("/api/portfolio")
async def portfolio_view(
    method: str = Query("HRP", description="Allocator method: HRP / MV / EQR (Phase 2B.1)"),
    book: str = Query("paper", description="paper or prod book (Phase 2B.3)"),
) -> JSONResponse:
    """M7 — book snapshot + selected-method allocator preview across recent proposals.

    The allocator method is now switchable from the UI via the
    ``method`` query parameter:

    * ``HRP``  → Hierarchical Risk Parity (default, robust to ill-conditioned cov).
    * ``MV``   → Mean-Variance (Markowitz long-only sum-to-budget).
    * ``EQR``  → Equal-Risk-Contribution baseline (inverse-vol).

    ``book`` partitions the on-disk ``PortfolioState`` files by name so
    paper and prod books don't share state.  See paper §M7 "multi-
    portfolio support".
    """
    from trading_crew.agentic.execution.contracts import ActionProposal
    from trading_crew.agentic.memory import EpisodicMemory
    from trading_crew.agentic.portfolio import (
        AllocatorConfig,
        AllocationMethod,
        allocate,
        load_portfolio_state,
    )
    from datetime import datetime
    from .charts import _fetch_ohlcv

    method_normalised = (method or "HRP").upper()
    method_lookup = {
        "HRP": AllocationMethod.HRP,
        "MV": AllocationMethod.MEAN_VARIANCE,
        "MEAN_VARIANCE": AllocationMethod.MEAN_VARIANCE,
        "EQR": AllocationMethod.EQUAL_RISK,
        "EQUAL_RISK": AllocationMethod.EQUAL_RISK,
    }
    allocation_method = method_lookup.get(method_normalised, AllocationMethod.HRP)

    book_normalised = (book or "paper").strip().lower()
    if book_normalised not in ("paper", "prod"):
        book_normalised = "paper"
    portfolio_id = book_normalised if book_normalised == "prod" else "paper"
    state = load_portfolio_state(portfolio_id)
    cache_dir = os.environ.get("TRADINGCREW_CACHE_DIR") or os.path.expanduser("~/.trading_crew")
    store_path = Path(cache_dir) / "memory" / "episodes.jsonl"
    mem = EpisodicMemory(store_path)

    latest: dict[str, ActionProposal] = {}
    for ep in mem.all_episodes():
        try:
            p = ActionProposal(**ep.action_proposal)
        except Exception:
            continue
        prev = latest.get(p.symbol)
        if prev is None or p.decision_ts > prev.decision_ts:
            latest[p.symbol] = p

    allocator_preview = None
    if len(latest) >= 2:
        import math as _math
        returns_by_symbol: dict[str, list[float]] = {}
        latest_ts = max(p.decision_ts for p in latest.values())
        try:
            end_dt = datetime.fromisoformat(latest_ts.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            end_dt = datetime.utcnow()
        for sym in latest:
            try:
                df = await asyncio.to_thread(_fetch_ohlcv, sym, end_dt, 365)
                closes = df["Close"].astype(float).tolist()
                log_rets = [
                    _math.log(closes[i] / closes[i - 1])
                    for i in range(1, len(closes)) if closes[i - 1] > 0
                ]
                if len(log_rets) >= 30:
                    returns_by_symbol[sym] = log_rets
            except Exception:
                logger.warning("OHLCV fetch failed for %s during portfolio preview", sym)
        if len([s for s in returns_by_symbol if s in latest]) >= 2:
            try:
                result = allocate(
                    list(latest.values()),
                    returns_by_symbol,
                    AllocatorConfig(method=allocation_method),
                )
                allocator_preview = {
                    "method": result.method_used.value,
                    "requested_method": allocation_method.value,
                    "weights": result.weights,
                    "risk_contributions": result.risk_contributions,
                    "notes": result.notes,
                }
            except Exception:
                logger.exception("Allocator preview failed")

    return JSONResponse({
        "book": book_normalised,
        "allocator_method": allocation_method.value,
        "snapshot": state.to_snapshot(),
        "starting_cash": state.starting_cash,
        "realized_pnl": state.realized_pnl,
        "max_drawdown": state.max_drawdown,
        "peak_nav": state.peak_nav,
        "latest_proposals": {
            sym: {
                "decision_ts": p.decision_ts,
                "side": p.side.value,
                "target_weight": p.target_weight,
                "conviction_tier": p.conviction_tier.value,
                "horizon_days": p.horizon_days,
            }
            for sym, p in latest.items()
        },
        "allocator_preview": allocator_preview,
    })


@app.get("/api/runs/recent")
async def runs_recent(
    limit: int = Query(20, ge=1, le=200),
    ticker: str = Query("", description="Filter to a single ticker"),
) -> JSONResponse:
    """Phase E — most recent UI runs across all tickers (or one)."""
    from trading_crew.agentic.runs import list_recent_runs
    entries = await asyncio.to_thread(list_recent_runs, limit, ticker or None)
    return JSONResponse({"runs": entries, "count": len(entries)})


@app.delete("/api/runs/recent")
async def runs_clear(
    ticker: str = Query("", description="If set, only clear runs for this ticker"),
) -> JSONResponse:
    """Phase E — delete saved run history.

    Without ``ticker`` this clears the entire history; with ``ticker``
    it only drops that ticker's records and rewrites the global index.
    The underlying helper moves the affected files into a timestamped
    trash folder under ``$TRADINGCREW_CACHE_DIR/runs/`` instead of
    ``rm -rf``-ing them, so a user can recover from an accidental
    click by renaming the trash back.
    """
    from trading_crew.agentic.runs import clear_runs
    result = await asyncio.to_thread(clear_runs, ticker or None)
    logger.info(
        "Cleared %d run record(s) (ticker=%s, trash=%s)",
        result.get("removed", 0),
        ticker or "*",
        result.get("trash_path"),
    )
    return JSONResponse(result)


@app.get("/api/runs/{ticker}/latest")
async def runs_latest(ticker: str) -> JSONResponse:
    """Phase E — full record of the most recent run for ``ticker``."""
    from trading_crew.agentic.runs import load_latest_run
    record = await asyncio.to_thread(load_latest_run, ticker)
    if not record:
        return JSONResponse({"error": "No runs recorded for this ticker"}, status_code=404)
    return JSONResponse(record)


@app.get("/api/runs/{ticker}/{run_id}")
async def runs_load(ticker: str, run_id: str) -> JSONResponse:
    """Phase E — full record of a specific past run."""
    from trading_crew.agentic.runs import load_run
    record = await asyncio.to_thread(load_run, run_id, ticker)
    if not record:
        return JSONResponse({"error": "Run not found"}, status_code=404)
    return JSONResponse(record)


@app.post("/api/grid_search")
async def grid_search(
    ticker: str = Query(..., min_length=1),
    train_size: int = Query(3, ge=1, le=500),
    embargo_size: int = Query(1, ge=0, le=30),
    test_size: int = Query(1, ge=1, le=200),
    cost_model: str = Query("standard"),
    rank_by: str = Query("deflated_sharpe", description="deflated_sharpe | sharpe | sortino | calmar | total_return_pct"),
    grid_size: str = Query("default", description="default | coarse | fine"),
    # Phase 2C — extra axes the user can opt in to.
    sweep_max_leverage: bool = Query(False, description="Also sweep max_leverage in {1.0, 1.5, 2.0}"),
    sweep_drawdown_kill: bool = Query(False, description="Also sweep drawdown_kill_threshold in {0.10, 0.20, 0.30}"),
    sweep_cost_model: bool = Query(False, description="Also sweep cost_model across low/standard/high (forks search per preset)"),
    validation: str = Query("walk_forward", description="walk_forward | cpcv (Combinatorial Purged CV)"),
    lookback_years: int = Query(5, ge=1, le=10, description="Years of OHLCV history to fetch for the backtest (default 5y)"),
) -> JSONResponse:
    """Agentic training L3 — sweep sizing/gate configs against logged
    proposals for ``ticker`` and return the best configuration by the
    user-specified ranking metric.

    The Default ranking is Deflated Sharpe (Bailey & López de Prado 2014),
    which penalises the multiple-comparison overfit that grid-searches
    notoriously fall into.  Coarse = 8 points (2^3), Default = 27 (3^3),
    Fine = 125 (5^3).
    """
    from datetime import datetime
    from trading_crew.agentic.backtest import WalkForwardConfig
    from trading_crew.agentic.execution.contracts import ActionProposal
    from trading_crew.agentic.grid_search import (
        GridAxis, default_grid, run_grid_search,
    )
    from trading_crew.agentic.memory import EpisodicMemory
    from .charts import _fetch_ohlcv

    cache_dir = os.environ.get("TRADINGCREW_CACHE_DIR") or os.path.expanduser("~/.trading_crew")
    store_path = Path(cache_dir) / "memory" / "episodes.jsonl"
    mem = EpisodicMemory(store_path)
    episodes = [e for e in mem.all_episodes() if e.symbol.upper() == ticker.upper()]
    if not episodes:
        return JSONResponse({
            "ticker": ticker, "n_proposals": 0,
            "error": "No episodes recorded for this ticker yet — run an analysis first.",
        })

    proposals = []
    for ep in episodes:
        try:
            proposals.append(ActionProposal(**ep.action_proposal))
        except Exception:
            logger.warning("Skipping malformed proposal in episode %s", ep.episode_id)
    if not proposals:
        return JSONResponse({
            "ticker": ticker, "n_proposals": 0,
            "error": "All recorded episodes had invalid proposals.",
        })

    # Grid presets — same axes, just denser/sparser. We tune three
    # parameters because anything more makes the run time of a 5^N grid
    # super-linear, and three is enough to span Kelly/vol/concentration.
    if grid_size == "coarse":
        grid = [
            GridAxis("kelly_fraction", (0.10, 0.50)),
            GridAxis("vol_target", (0.07, 0.15)),
            GridAxis("max_position_weight", (0.10, 0.30)),
        ]
    elif grid_size == "fine":
        grid = [
            GridAxis("kelly_fraction", (0.05, 0.10, 0.25, 0.50, 0.75)),
            GridAxis("vol_target", (0.05, 0.07, 0.10, 0.15, 0.20)),
            GridAxis("max_position_weight", (0.05, 0.10, 0.20, 0.30, 0.50)),
        ]
    else:
        grid = default_grid()

    # Phase 2C — additive axes (opt-in via UI checkboxes).  Default
    # presets are deliberately small so we don't 27 × 3 the grid by
    # accident — users explicitly enable each.
    extra_axes = []
    if sweep_max_leverage:
        extra_axes.append(GridAxis("max_leverage", (1.0, 1.5, 2.0)))
    if sweep_drawdown_kill:
        extra_axes.append(GridAxis("drawdown_kill_threshold", (0.10, 0.20, 0.30)))
    grid = list(grid) + extra_axes

    last_ts = max(p.decision_ts for p in proposals)
    try:
        end_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        end_dt = datetime.utcnow()
    ohlcv = await asyncio.to_thread(_fetch_ohlcv, ticker, end_dt, 365 * lookback_years)
    if ohlcv is None or ohlcv.empty:
        return JSONResponse({
            "ticker": ticker, "n_proposals": len(proposals),
            "error": "Could not fetch OHLCV for backtest.",
        })

    wf_cfg = WalkForwardConfig(train_size=train_size, embargo_size=embargo_size, test_size=test_size)

    # Phase 2C — cost-model sweep.  Re-run the same grid against each
    # cost preset, then pick the best (configuration, cost-model) pair.
    # This is a *robustness check*: a strategy that only wins under
    # low-friction is fragile.
    cost_models = [cost_model]
    if sweep_cost_model:
        cost_models = (
            ["futures_low", "futures_standard", "futures_high"]
            if cost_model.startswith("futures")
            else ["low", "standard", "high"]
        )

    aggregated_points: list[dict] = []
    base_dict: dict = {}
    for cm in cost_models:
        partial = await asyncio.to_thread(
            run_grid_search,
            proposals=proposals,
            ohlcv_by_symbol={ticker: ohlcv},
            walk_forward_config=wf_cfg,
            grid=grid,
            cost_model_name=cm,
            rank_by=rank_by,
        )
        d = partial.to_dict()
        if not base_dict:
            base_dict = d
        for pt in d.get("points", []):
            pt["cost_model"] = cm
            aggregated_points.append(pt)

    # Stitch the per-cost-model points into one ranked list.  Picking
    # the best across the union honours the "robustness" interpretation
    # of the cost-sweep — best needs to be best under realistic costs.
    if aggregated_points:
        aggregated_points.sort(
            key=lambda p: p.get("rank_metric", float("-inf")),
            reverse=True,
        )
        best = aggregated_points[0]
        base_dict["points"] = aggregated_points
        base_dict["n_points"] = len(aggregated_points)
        base_dict["best"] = best
        base_dict["best_index"] = 0
        base_dict["validation"] = validation
        base_dict["cost_models_swept"] = cost_models
    return JSONResponse(base_dict)


@app.post("/api/backtest/seed")
async def backtest_seed(
    ticker: str = Query(..., min_length=1),
    n: int = Query(10, ge=1, le=200, description="Synthetic proposals to generate"),
    horizon_jitter_days: int = Query(2, ge=0, le=20, description="±days perturbation around the original horizon"),
    price_jitter_pct: float = Query(1.5, ge=0.0, le=20.0, description="±% perturbation around entry/stop/target"),
    seed: int = Query(0, description="Random seed for reproducibility (0 = wall clock)"),
) -> JSONResponse:
    """Phase 2C.3 — bulk-seed synthetic ``ActionProposal``s for ``ticker``.

    Re-runs the deterministic post-PM pipeline (M1 → M5 → M2 → M3) over
    historical bars by sampling N existing episodes for the ticker and
    perturbing their numeric fields (horizon, entry/stop/target) within
    the configured jitter envelope.  Persists each new proposal as a
    PENDING episode so the Backtest tab can use them as additional fold
    samples — *without* invoking the LLM crew.

    Returns a count of seeded episodes + a sample of the first 5 so the
    UI can show what was created.
    """
    from datetime import datetime
    from random import Random
    from trading_crew.agentic.execution.contracts import ActionProposal
    from trading_crew.agentic.memory import Episode, OutcomeStatus, Regime
    from trading_crew.agentic.memory.embedding import make_memory

    cache_dir = os.environ.get("TRADINGCREW_CACHE_DIR") or os.path.expanduser("~/.trading_crew")
    store_path = Path(cache_dir) / "memory" / "episodes.jsonl"
    mem = make_memory(store_path)
    existing = [e for e in mem.all_episodes() if e.symbol.upper() == ticker.upper()]
    if not existing:
        return JSONResponse({
            "ticker": ticker, "seeded": 0,
            "error": "No real episodes recorded for this ticker yet — run a few analyses first so the seeder has something to perturb.",
        })

    rng = Random(seed) if seed else Random()
    seeded_records: list[dict] = []
    for i in range(n):
        src = rng.choice(existing)
        try:
            base = ActionProposal(**src.action_proposal)
        except Exception:
            continue
        # Perturb numerics within the jitter envelope.
        ts_offset_days = rng.randint(-90, 90)
        try:
            new_decision = datetime.fromisoformat(src.decision_ts.replace("Z", "+00:00")) + timedelta(days=ts_offset_days)
        except Exception:
            new_decision = datetime.utcnow() + timedelta(days=ts_offset_days)
        horizon = max(1, base.horizon_days + rng.randint(-horizon_jitter_days, horizon_jitter_days))
        outcome_dt = new_decision + timedelta(days=horizon)
        # Jitter the expected return so each synthetic proposal lands a
        # slightly different M2 fill — but stay inside the original sign
        # so the side/edge invariant survives.
        er_mult = 1.0 + rng.uniform(-price_jitter_pct, price_jitter_pct) / 100.0
        target_weight = base.target_weight * (1.0 + rng.uniform(-price_jitter_pct, price_jitter_pct) / 100.0)
        target_weight = max(-1.0, min(1.0, target_weight))
        synthetic = ActionProposal(
            symbol=base.symbol,
            decision_ts=new_decision.isoformat(),
            side=base.side,
            target_weight=target_weight,
            horizon_days=horizon,
            conviction_score=base.conviction_score,
            conviction_tier=base.conviction_tier,
            expected_return_pct=base.expected_return_pct * er_mult,
            rationale=f"[synthetic seed #{i+1}] {base.rationale}",
            validity_check=base.validity_check,
        )
        ep = Episode(
            episode_id=f"synth-{ticker.upper()}-{new_decision.strftime('%Y%m%d%H%M%S')}-{i}",
            symbol=ticker.upper(),
            decision_ts=synthetic.decision_ts,
            state_summary=f"Synthetic perturbation of {src.episode_id}",
            regime=Regime(src.regime.value if hasattr(src.regime, "value") else src.regime),
            action_proposal=synthetic.model_dump(mode="json"),
            outcome_ts=outcome_dt.isoformat(),
            outcome_status=OutcomeStatus.PENDING,
        )
        mem.add(ep)
        if len(seeded_records) < 5:
            seeded_records.append({
                "episode_id": ep.episode_id,
                "decision_ts": ep.decision_ts,
                "side": synthetic.side.value,
                "target_weight": synthetic.target_weight,
                "horizon_days": synthetic.horizon_days,
            })
    return JSONResponse({
        "ticker": ticker.upper(),
        "seeded": n,
        "sample": seeded_records,
        "store_path": str(store_path),
    })


# ---------------------------------------------------------------------------
# L4 — Reinforcement Learning training endpoints
# ---------------------------------------------------------------------------


@app.post("/api/training/rl/start")
async def rl_start(payload: Dict[str, Any]) -> JSONResponse:
    """Kick off an L4 PPO training run for ``ticker``.

    Request JSON::

        {
          "ticker": "NVDA",
          "asset_class": "stock",                    # or "commodity"
          "train_window_days": 750,
          "eval_window_days": 60,
          "ppo_config": {                            # all optional
            "total_steps": 20000,
            "steps_per_rollout": 512,
            "n_epochs": 4,
            "learning_rate": 0.0003,
            "entropy_coef": 0.01
          },
          "env_config": {                            # all optional
            "cost_model_name": "standard",
            "drawdown_kill_pct": 0.4,
            "turnover_penalty_bps": 0.0
          },
          "seed": 42
        }

    Returns the new run record immediately (status="running"); the
    actual PPO loop runs in a background worker.  Poll
    ``/api/training/rl/status?ticker=...`` to watch progress.
    """
    from .rl_runner import start_training

    ticker = (payload.get("ticker") or "").strip()
    if not ticker:
        raise HTTPException(status_code=400, detail="ticker is required")
    asset_class = (payload.get("asset_class") or "stock").lower()
    train_window = int(payload.get("train_window_days") or 750)
    eval_window = int(payload.get("eval_window_days") or 60)
    seed = int(payload.get("seed") or 42)
    horizon_mode = (payload.get("horizon_mode") or "balanced").lower()

    algorithm = (payload.get("algorithm") or "ppo").lower()
    policy_universe = payload.get("policy_universe") or []
    try:
        record = await asyncio.to_thread(
            start_training,
            ticker=ticker,
            asset_class=asset_class,
            train_window_days=train_window,
            eval_window_days=eval_window,
            ppo_overrides=payload.get("ppo_config") or {},
            env_overrides=payload.get("env_config") or {},
            seed=seed,
            algorithm=algorithm,
            policy_universe=list(policy_universe),
            horizon_mode=horizon_mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse(record)


@app.post("/api/training/rl/stop")
async def rl_stop(payload: Dict[str, Any]) -> JSONResponse:
    """Cooperatively stop a running L4 training job.

    Body must contain either ``{"ticker": "..."}`` (stops the active
    run for that ticker) or ``{"run_id": "..."}`` (stops a specific
    run id, regardless of ticker).
    """
    from .rl_runner import stop_training
    ticker = (payload.get("ticker") or "").strip() or None
    run_id = (payload.get("run_id") or "").strip() or None
    if not ticker and not run_id:
        raise HTTPException(status_code=400, detail="ticker or run_id required")
    result = await asyncio.to_thread(stop_training, ticker=ticker, run_id=run_id)
    return JSONResponse(result)


@app.get("/api/training/rl/status")
async def rl_status(ticker: str = Query(..., min_length=1)) -> JSONResponse:
    """Snapshot of the currently-running L4 job for ``ticker``."""
    from .rl_runner import get_status
    return JSONResponse(get_status(ticker))


@app.get("/api/training/rl/runs")
async def rl_runs(
    ticker: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
) -> JSONResponse:
    """List past + current L4 runs (most recent first).

    If ``ticker`` is set, restrict to that ticker.  Each entry includes
    the eval result + duration so the UI leaderboard can rank without
    a second call.
    """
    from trading_crew.agentic.rl import list_runs
    return JSONResponse(
        {
            "ticker": ticker,
            "runs": [r.to_dict() for r in list_runs(ticker, limit=limit)],
        }
    )


@app.get("/api/training/rl/runs/{ticker}/{run_id}")
async def rl_run_detail(ticker: str, run_id: str) -> JSONResponse:
    """Full snapshot for a single L4 run: record + metrics stream."""
    from .rl_runner import get_run_snapshot
    snap = get_run_snapshot(ticker, run_id)
    if snap is None:
        raise HTTPException(status_code=404, detail=f"No run {ticker}/{run_id}")
    return JSONResponse(snap)


@app.post("/api/training/rl/promote")
async def rl_promote(payload: Dict[str, Any]) -> JSONResponse:
    """Promote a completed L4 run so it becomes the active policy for
    its ticker.  Body: ``{"ticker": "...", "run_id": "..."}``.

    Promotion writes a single ``promoted/<ticker>.json`` pointer that
    the ``rl_policy_recommendation`` tool consults at inference time.
    Until you promote, the tool returns a "no policy" note — there's
    no implicit "use latest" fallback so a half-trained run can never
    silently bleed into production decisions.
    """
    from trading_crew.agentic.rl import promote_run
    ticker = (payload.get("ticker") or "").strip()
    run_id = (payload.get("run_id") or "").strip()
    if not ticker or not run_id:
        raise HTTPException(status_code=400, detail="ticker and run_id required")
    try:
        result = await asyncio.to_thread(promote_run, ticker, run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return JSONResponse(result)


@app.get("/api/training/rl/promoted")
async def rl_promoted() -> JSONResponse:
    """List every promoted L4 policy, one per ticker."""
    from trading_crew.agentic.rl import list_promoted
    return JSONResponse({"promoted": list_promoted()})


@app.post("/api/training/rl/recommend")
async def rl_recommend(payload: Dict[str, Any]) -> JSONResponse:
    """Run the promoted policy against today's bar and return the
    structured recommendation.  Body: ``{"ticker": "..."}``.

    This is the same call path the LLM tool uses, exposed via HTTP so
    the UI can show the current advice without spinning up the crew.
    """
    from trading_crew.agentic.rl import load_policy
    from .charts import _fetch_ohlcv

    ticker = (payload.get("ticker") or "").strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="ticker is required")
    client = await asyncio.to_thread(load_policy, ticker)
    if client is None:
        return JSONResponse({"ticker": ticker, "available": False,
                             "note": "No policy has been promoted for this ticker yet."})

    df = await asyncio.to_thread(_fetch_ohlcv, ticker, datetime.utcnow(), 180)
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail=f"No OHLCV for {ticker}")
    df = df.rename(columns={c: c.lower() for c in df.columns})
    if "date" in df.columns:
        df = df.set_index("date").sort_index()
    try:
        rec = await asyncio.to_thread(client.recommend, df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Policy inference failed: {exc}")
    return JSONResponse({"ticker": ticker, "available": True, "recommendation": rec.to_dict()})


@app.get("/api/backtest")
async def backtest(
    ticker: str = Query(..., min_length=1),
    train_size: int = Query(3, ge=1, le=500),
    embargo_size: int = Query(1, ge=0, le=30),
    test_size: int = Query(1, ge=1, le=200),
    cost_model: str = Query("standard"),
    lookback_years: int = Query(5, ge=1, le=10, description="Years of OHLCV history to fetch for the backtest (default 5y)"),
) -> JSONResponse:
    """M6 — walk-forward backtest of logged proposals for ``ticker``."""
    from datetime import datetime
    from trading_crew.agentic.backtest import (
        WalkForwardConfig,
        generate_folds,
        run_walk_forward,
    )
    from trading_crew.agentic.execution.contracts import ActionProposal
    from trading_crew.agentic.memory import EpisodicMemory
    from .charts import _fetch_ohlcv

    cache_dir = os.environ.get("TRADINGCREW_CACHE_DIR") or os.path.expanduser("~/.trading_crew")
    store_path = Path(cache_dir) / "memory" / "episodes.jsonl"
    mem = EpisodicMemory(store_path)
    episodes = [e for e in mem.all_episodes() if e.symbol.upper() == ticker.upper()]
    if not episodes:
        return JSONResponse({
            "ticker": ticker, "n_proposals": 0,
            "error": "No episodes recorded for this ticker yet — run an analysis first.",
        })

    proposals: list[ActionProposal] = []
    for ep in episodes:
        try:
            proposals.append(ActionProposal(**ep.action_proposal))
        except Exception:
            logger.warning("Skipping malformed proposal in episode %s", ep.episode_id)
    if not proposals:
        return JSONResponse({
            "ticker": ticker, "n_proposals": 0,
            "error": "All recorded episodes had invalid proposals.",
        })

    last_ts = max(p.decision_ts for p in proposals)
    try:
        end_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        end_dt = datetime.utcnow()
    ohlcv = await asyncio.to_thread(_fetch_ohlcv, ticker, end_dt, 365 * lookback_years)

    config = WalkForwardConfig(
        train_size=train_size, embargo_size=embargo_size, test_size=test_size,
    )
    folds = generate_folds(n_obs=len(proposals), config=config)
    if not folds:
        return JSONResponse({
            "ticker": ticker, "n_proposals": len(proposals),
            "error": (
                f"Not enough proposals ({len(proposals)}) for "
                f"train={train_size}+embargo={embargo_size}+test={test_size}."
            ),
        })

    result = await asyncio.to_thread(
        run_walk_forward,
        proposals=proposals,
        ohlcv_by_symbol={ticker: ohlcv},
        folds=folds,
        cost_model_name=cost_model,
    )

    def _finite(v):
        try:
            if v != v or v in (float("inf"), float("-inf")):
                return None
        except Exception:
            return None
        return v

    return JSONResponse({
        "ticker": ticker,
        "n_proposals": len(proposals),
        "n_folds": len(result.folds),
        "combined_equity": result.combined_equity,
        "combined_timestamps": result.combined_timestamps,
        "overall_metrics": {
            "total_return_pct": result.overall_metrics.total_return_pct,
            "cagr": result.overall_metrics.cagr,
            "annualised_vol": result.overall_metrics.annualised_vol,
            "sharpe": result.overall_metrics.sharpe,
            "sortino": _finite(result.overall_metrics.sortino),
            "calmar": _finite(result.overall_metrics.calmar),
            "max_drawdown": result.overall_metrics.max_drawdown,
            "deflated_sharpe": result.overall_metrics.deflated_sharpe,
            "n_periods": result.overall_metrics.n_periods,
        },
        "folds": [
            {
                "fold_id": fr.fold.fold_id,
                "train_start": fr.fold.train_start,
                "train_end": fr.fold.train_end,
                "test_start": fr.fold.test_start,
                "test_end": fr.fold.test_end,
                "metrics": {
                    "total_return_pct": fr.metrics.total_return_pct,
                    "sharpe": fr.metrics.sharpe,
                    "max_drawdown": fr.metrics.max_drawdown,
                    "n_periods": fr.metrics.n_periods,
                },
                "trades": [
                    {
                        "ts": t.ts, "symbol": t.symbol, "side": t.side,
                        "intent_weight": t.intent_weight,
                        "sized_weight": t.sized_weight,
                        "fill_qty": t.fill_qty, "fill_price": t.fill_price,
                        "fees": t.fees, "status": t.status,
                        "rejection_reason": t.rejection_reason,
                    }
                    for t in fr.trades
                ],
            }
            for fr in result.folds
        ],
    })


@app.websocket("/ws/analyze")
async def analyze_ws(ws: WebSocket) -> None:
    await ws.accept()
    loop = asyncio.get_event_loop()
    runner = AnalysisRunner(loop)

    try:
        raw = await ws.receive_text()
        try:
            cfg: Dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError:
            await ws.send_text(json.dumps({"type": "error", "message": "Invalid JSON"}))
            await ws.close()
            return

        ticker = cfg.get("ticker")
        if not ticker:
            await ws.send_text(json.dumps({"type": "error", "message": "ticker is required"}))
            await ws.close()
            return

        async def consumer() -> None:
            while True:
                event = await runner.events.get()
                try:
                    await ws.send_text(json.dumps(event, default=str))
                except WebSocketDisconnect:
                    runner.cancel()
                    return
                if event.get("type") in ("run_completed", "error"):
                    return

        consumer_task = asyncio.create_task(consumer())
        run_task = asyncio.create_task(runner.run(ticker, cfg))

        await asyncio.gather(consumer_task, run_task)

    except WebSocketDisconnect:
        runner.cancel()
        logger.info("Client disconnected mid-run")
    except Exception:
        logger.exception("WebSocket session failed")
        try:
            await ws.send_text(
                json.dumps({"type": "error", "message": "Internal server error"})
            )
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


def main() -> None:
    """Console-script entry point: ``python -m web.backend.app``."""
    import uvicorn

    host = os.getenv("TRADINGCREW_WEB_HOST", "0.0.0.0")
    port = int(os.getenv("TRADINGCREW_WEB_PORT", "8001"))
    uvicorn.run(
        "web.backend.app:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
