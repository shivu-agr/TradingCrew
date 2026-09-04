"""FastAPI server for TradingCrew UI.

Routes
------
* GET  /                    -> serves the single-page frontend
* GET  /static/*            -> static assets (js / css)
* GET  /api/options         -> dropdowns: tickers, tools, agents, indicators, defaults
* GET  /api/chart           -> OHLCV + indicator series for the Charts tab
* /api/v1/*                 -> programmatic Prediction API (start / status / stop)
* WS   /ws/analyze          -> first text frame is JSON config, then events stream
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

# Load env from the project root (the parent of the ``web/`` dir).
# override=True makes .env authoritative: any stale/empty value already present
# in the launching shell's environment (which load_dotenv would otherwise keep)
# is replaced by the .env value.  Without this, a leftover shell export silently
# shadows .env — e.g. the search min-gap reading 0 while .env says 6.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

from .charts import SUPPORTED_INDICATORS, build_chart_payload  # noqa: E402
from .logging_setup import configure_logging  # noqa: E402
from .runner import AnalysisRunner, get_agent_catalog, get_tool_catalog  # noqa: E402
from .api_v1 import router as api_v1_router  # noqa: E402
from .session_manager import (  # noqa: E402
    _EOF,
    RunStatus,
    get_session_manager,
    init_session_manager,
)

# Install stdout + rotating-file logging before any module-level
# logger calls happen.  See ``logging_setup.py`` for the env knobs.
_LOG_PATH = configure_logging()
logger = logging.getLogger(__name__)
logger.info("Logging to %s (stdout mirror also enabled)", _LOG_PATH)

BACKEND_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BACKEND_DIR.parent / "frontend"


@asynccontextmanager
async def _lifespan(app_: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan hook.

    Binds the ``SessionManager`` singleton to the running event loop at startup,
    then brings up the Kafka (Redpanda) pub-sub bus + the in-process workflow
    consumer pool so ``/api/scale/*`` can publish work that consumers pull
    continuously.  If the broker is unreachable we log a clear, actionable error
    and leave the rest of the UI usable — the scale endpoints will report the
    same until ``scripts/redpanda.sh start`` is run and the server restarts.
    """
    init_session_manager(asyncio.get_running_loop())
    logger.info("SessionManager initialised on the FastAPI event loop.")

    from . import kafka_bus, workflow_consumer

    try:
        await kafka_bus.start_bus()
        try:
            n = int(os.environ.get("TRADINGCREW_WORKFLOW_CONSUMERS", "6"))
        except ValueError:
            n = 6
        await workflow_consumer.start_consumers(n)
        logger.info(
            "Kafka workflow pub-sub ready: %d consumers on '%s'. Console: %s",
            n, kafka_bus.workflow_requests_topic(), kafka_bus.console_url(),
        )
    except Exception:  # noqa: BLE001
        logger.exception(
            "Kafka bus/consumers failed to start — is Redpanda up? Run "
            "`scripts/redpanda.sh start`. Scale (batch) runs are unavailable "
            "until the broker is reachable and the server is restarted."
        )

    # Rate-limit gateway: route outbound web search + screener scraping through
    # a shared Kafka slot queue so their upstream rate limits are enforced
    # centrally (and visible in the Console).
    try:
        from trading_crew.data_sources import rate_gateway

        await rate_gateway.start(bootstrap=kafka_bus.bootstrap_servers())
        logger.info("Rate-limit gateway ready for domains: %s", list(rate_gateway.domains()))
    except Exception:  # noqa: BLE001
        logger.exception(
            "Rate-limit gateway failed to start — search/scrape will fall back "
            "to in-process throttling until Redpanda is reachable."
        )

    try:
        yield
    finally:
        try:
            from trading_crew.data_sources import rate_gateway
            await rate_gateway.stop()
        except Exception:  # noqa: BLE001
            logger.exception("error stopping rate-limit gateway")
        try:
            await workflow_consumer.stop_consumers()
        except Exception:  # noqa: BLE001
            logger.exception("error stopping workflow consumers")
        try:
            await kafka_bus.stop_bus()
        except Exception:  # noqa: BLE001
            logger.exception("error stopping Kafka bus")


app = FastAPI(title="TradingCrew UI", version="0.2.0", lifespan=_lifespan)
app.include_router(api_v1_router)


class _NoCacheStaticFiles(StaticFiles):
    """StaticFiles that tells the browser to always revalidate.

    The default ``StaticFiles`` lets the browser cache ``app.js`` /
    ``style.css`` aggressively (heuristic + ETag), so a soft page
    reload after a deploy can keep running the *old* JS while the
    no-cache ``index.html`` already renders the new template — a
    confusing mismatch (e.g. the Run button behaves like the previous
    build).  Serving assets ``no-cache`` forces a conditional GET; the
    304 path keeps it cheap while guaranteeing freshness.
    """

    async def get_response(self, path: str, scope):  # type: ignore[override]
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response


if FRONTEND_DIR.exists():
    app.mount(
        "/static",
        _NoCacheStaticFiles(directory=str(FRONTEND_DIR), html=False),
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


@app.get("/api/health")
async def health() -> dict:
    """Liveness probe for Docker / compose. No auth, no I/O."""
    return {"ok": True, "service": "tradingcrew"}


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


@app.get("/settings")
async def settings_page() -> FileResponse:
    """Operator Settings page: left pane is run config, right pane is
    strategy + prompt overlays. Same SPA as ``/stocks``; the client reads
    the path (and optional ``?book=commodity``)."""
    return _predict_page()


@app.get("/stocks")
async def stocks_dashboard() -> FileResponse:
    """High-level stock dashboard (unified SPA): the live prediction universe +
    Training / Episodes, with the History left pane.  This is the canonical
    stock landing (renamed from ``/predict``); opening a tile deep-links to
    ``/run/{ticker}``.  Dated prediction runs live at
    ``/stocks/history/{date}`` (Names + Portfolio)."""
    return _predict_page()


@app.get("/stocks/history/{run_date}")
async def stocks_history_page(run_date: str) -> FileResponse:
    """Prediction-run review: the SPA reads the date off the path and shows
    that day's ran names plus the virtual portfolio.  Opening a name
    deep-links to ``/run/{ticker}?date=...``."""
    return _predict_page()


@app.get("/stock")
async def stock_dashboard() -> RedirectResponse:
    """Back-compat alias — the stock dashboard is now ``/stocks``."""
    return RedirectResponse(url="/stocks", status_code=307)


@app.get("/commodity")
async def commodity_dashboard() -> FileResponse:
    """High-level commodity dashboard (unified SPA): the Contracts picker +
    Training / Episodes, with the History/Settings left pane.  Mirrors the
    stock ``/stocks`` flow — opening a contract deep-links to
    ``/commodity/run/{contract}`` for a focused per-contract workbench.

    The client reads ``window.location.pathname`` to discriminate asset_class,
    and the API endpoints accept an ``asset_class`` query param to surface the
    right ticker presets and route the websocket to the right crew.
    """
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="Commodity dashboard not found")
    return FileResponse(str(index_path), headers=_HTML_NO_CACHE)


@app.get("/commodity/run/{contract:path}")
async def commodity_run_page(contract: str) -> FileResponse:
    """Focused per-contract workbench (commodity analog of ``/run/{ticker}``):
    the SPA drops the left pane and shows only the instrument tabs (Workflow ·
    Charts · Futures · Backtest) with a Back button to ``/commodity``.  Uses a
    ``:path`` converter because futures symbols contain ``=`` (e.g. ``CL=F``)."""
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="Commodity dashboard not found")
    return FileResponse(str(index_path), headers=_HTML_NO_CACHE)


def _predict_page() -> FileResponse:
    # The Prediction dashboard is now the front page of the unified SPA
    # (index.html) — it reads the URL path to open the Prediction tab (or a
    # historical snapshot) and reveals stock-specific tabs on tile click.
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="Prediction dashboard not found")
    return FileResponse(str(index_path), headers=_HTML_NO_CACHE)


@app.get("/predict")
async def predict_page() -> RedirectResponse:
    """Back-compat alias — the high-level dashboard was renamed to ``/stocks``."""
    return RedirectResponse(url="/stocks", status_code=307)


@app.get("/predict/history/{run_date}")
async def predict_history_page(run_date: str) -> RedirectResponse:
    """Back-compat alias — history snapshots now live under ``/stocks``."""
    return RedirectResponse(url=f"/stocks/history/{run_date}", status_code=307)


@app.get("/run/{ticker}")
async def run_page(ticker: str) -> FileResponse:
    """Per-stock workbench (the /stock SPA) deep-linked to one ticker with the
    Workflow tab focused — this is where a Prediction tile opens to."""
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="Stock dashboard not found")
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

    # Desks running ReAct-style planning (empty when disabled). Only the
    # trading_crew wires planning today, so commodity stays empty.
    react_agents: list = []
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
        try:
            from trading_crew.crew import planning_agent_roles
            react_agents = planning_agent_roles()
        except Exception:
            react_agents = []

    from trading_crew.llm_presets import list_presets

    return JSONResponse({
        "asset_class": asset_class,
        "tickers": tickers,
        "default_ticker": default_ticker,
        "default_trade_date": date.today().isoformat(),
        "default_debate_rounds": int(os.getenv("TRADINGCREW_DEFAULT_DEBATE", "2")),
        "default_risk_rounds": int(os.getenv("TRADINGCREW_DEFAULT_RISK", "1")),
        "tools": tools,
        "agents": agents,
        "react_agents": react_agents,
        "chart_indicators": SUPPORTED_INDICATORS,
        "llm": {
            "model": llm_model,
            "base_url": llm_base,
            "provider": os.getenv("LLM_PROVIDER_PREFIX", "hosted_vllm"),
        },
        "llm_presets": list_presets(),
        "default_llm_preset": os.getenv("TRADINGCREW_LLM_PRESET") or "",
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
                "reflection_meta": getattr(r.episode, "reflection_meta", None),
                "action": (r.episode.action_proposal or {}).get("action"),
                "similarity": round(r.similarity, 4),
                "decay_factor": round(r.decay_factor, 4),
                "score": round(r.score, 4),
                "delta_days": round(r.delta_days, 2),
                "retrieval_count": getattr(r.episode, "retrieval_count", 0),
                "schema_version": getattr(r.episode, "schema_version", 0),
                "review_ts": getattr(r.episode, "review_ts", None),
                "review_status": getattr(r.episode, "review_status", None),
                "review": getattr(r.episode, "review", None),
                "horizon_scores": getattr(r.episode, "horizon_scores", None),
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


def _strategy_memory():
    from trading_crew.agentic.memory.embedding import make_memory

    cache_dir = os.environ.get("TRADINGCREW_CACHE_DIR") or os.path.expanduser("~/.trading_crew")
    store_path = Path(cache_dir) / "memory" / "episodes.jsonl"
    return make_memory(store_path)


@app.get("/api/settings")
async def get_settings(
    book: str = Query("equity", description="'equity' or 'commodity'"),
) -> JSONResponse:
    """One Settings document: live strategy, prompt overlays, and bulk run snapshot."""
    from trading_crew.agentic.live_settings import SettingsError, settings_payload
    from trading_crew.agentic.prompt_overlay import PromptOverlayError
    from trading_crew.agentic.strategy import StrategyError

    try:
        payload = await asyncio.to_thread(settings_payload, book)
    except (SettingsError, PromptOverlayError, StrategyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(payload)


@app.put("/api/settings")
async def put_settings(
    payload: Dict[str, Any],
    book: str = Query("equity", description="'equity' or 'commodity'"),
) -> JSONResponse:
    """Persist only the keys sent: strategy, prompts, unset_prompts, revert_strategy, bulk.

    Existing ``/api/strategy*`` endpoints stay for Hermes experiment ops and
    programmatic edits.
    """
    from trading_crew.agentic.live_settings import SettingsError, save_settings
    from trading_crew.agentic.prompt_overlay import PromptOverlayError
    from trading_crew.agentic.strategy import StrategyError

    try:
        out = await asyncio.to_thread(save_settings, book, payload)
    except (SettingsError, PromptOverlayError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except StrategyError as exc:
        code = 409 if payload.get("revert_strategy") else 400
        raise HTTPException(status_code=code, detail=str(exc)) from exc
    return JSONResponse(out)


@app.get("/api/strategy")
async def desk_strategy(
    book: str = Query("equity", description="'equity' or 'commodity'"),
) -> JSONResponse:
    """Editable desk strategy + live unified harness scoreboard."""
    from trading_crew.agentic.harness import dashboard_payload

    mem = _strategy_memory()
    payload = await asyncio.to_thread(
        dashboard_payload, mem.all_episodes(), book=book,
    )
    return JSONResponse(payload)


@app.put("/api/strategy")
async def save_desk_strategy_api(
    payload: Dict[str, Any],
    book: str = Query("equity"),
) -> JSONResponse:
    """Validate and write the live strategy file used by the next run / Resolve."""
    from trading_crew.agentic.harness import dashboard_payload
    from trading_crew.agentic.strategy import StrategyError, save_desk_strategy

    try:
        await asyncio.to_thread(save_desk_strategy, payload)
    except StrategyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    mem = _strategy_memory()
    board = await asyncio.to_thread(
        dashboard_payload, mem.all_episodes(), book=book,
    )
    return JSONResponse(board)


@app.post("/api/strategy/revert")
async def revert_desk_strategy_api(
    book: str = Query("equity"),
) -> JSONResponse:
    """Delete the live strategy file so the packaged yaml is used."""
    from trading_crew.agentic.harness import dashboard_payload
    from trading_crew.agentic.strategy import StrategyError, revert_live_strategy

    try:
        reverted = await asyncio.to_thread(revert_live_strategy)
    except StrategyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    mem = _strategy_memory()
    board = await asyncio.to_thread(
        dashboard_payload, mem.all_episodes(), book=book,
    )
    board["reverted"] = reverted
    return JSONResponse(board)


@app.put("/api/strategy/prompts")
async def save_strategy_prompt(
    payload: Dict[str, Any],
    book: str = Query("equity"),
) -> JSONResponse:
    """Set a per-agent goal/backstory overlay. Unset via DELETE."""
    from trading_crew.agentic.prompt_overlay import PromptOverlayError, set_prompt

    key = str(payload.get("key") or "").strip()
    try:
        row = await asyncio.to_thread(
            set_prompt,
            book,
            key,
            goal=str(payload.get("goal") or ""),
            backstory=str(payload.get("backstory") or ""),
        )
    except PromptOverlayError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(row)


@app.delete("/api/strategy/prompts/{agent_key}")
async def delete_strategy_prompt(
    agent_key: str,
    book: str = Query("equity"),
) -> JSONResponse:
    from trading_crew.agentic.prompt_overlay import PromptOverlayError, unset_prompt

    try:
        removed = await asyncio.to_thread(unset_prompt, book, agent_key)
    except PromptOverlayError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse({"key": agent_key, "removed": removed})


@app.post("/api/strategy/experiment")
async def strategy_experiment_api(
    payload: Dict[str, Any],
    book: str = Query("equity"),
) -> JSONResponse:
    """Close, open, or clear used knobs from the Hermes UI."""
    from trading_crew.agentic.harness import dashboard_payload
    from trading_crew.agentic.self_learn import (
        clear_used_knobs_ui,
        close_experiment_ui,
        open_knob_ui,
    )
    from trading_crew.agentic.strategy import StrategyError

    action = str(payload.get("action") or "").strip()
    mem = _strategy_memory()
    try:
        if action == "close":
            await asyncio.to_thread(close_experiment_ui)
        elif action == "clear_used":
            await asyncio.to_thread(clear_used_knobs_ui)
        elif action == "open":
            knob = str(payload.get("knob") or "").strip()
            if not knob:
                raise HTTPException(status_code=400, detail="open requires knob")
            await asyncio.to_thread(
                open_knob_ui, knob, mem.all_episodes(), book=book,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="action must be close, open, or clear_used",
            )
    except HTTPException:
        raise
    except (ValueError, StrategyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    board = await asyncio.to_thread(
        dashboard_payload, mem.all_episodes(), book=book,
    )
    return JSONResponse(board)


@app.post("/api/memory/resolve")
async def memory_resolve(
    ticker: Optional[str] = Query(None, description="Optional ticker filter; resolve only this symbol's episodes"),
    skip_llm: bool = Query(False, description="If true, write outcomes but skip the LLM reflection step (useful for batch / CI)"),
) -> JSONResponse:
    """Agentic training L2 — sweep pending episodes whose 30-day learning
    review or thesis ``outcome_ts`` has elapsed, compute realised return +
    alpha + max drawdown, and write an LLM-generated lesson on the thesis
    clock.

    A 30-day ``REVIEW_RESOLVED`` does not close a MEDIUM/LONG thesis.
    The review lesson (and any tool investigation) is written for Hermes.

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


@app.get("/api/calibration")
async def calibration(
    ticker: Optional[str] = Query(None, description="Optional ticker filter; omit for the whole book"),
    n_bins: int = Query(10, ge=2, le=20, description="Reliability bins over [0,1]"),
    basis: str = Query("return", description="'return' (raw) or 'alpha' (vs SPY)"),
    cohort: str = Query("current", description="'current' (this workflow version) or 'all'"),
) -> JSONResponse:
    """Conviction calibration — is the desk's stated confidence honest?

    Scores resolved directional episodes into a Brier score, reliability bins
    (predicted conviction vs realised hit-rate), and a signed over/under-
    confidence gap.  Measurement-only; empty (``n == 0``) until episodes
    resolve.
    """
    from trading_crew.agentic.calibration import compute_calibration

    workflow_version = None
    if (cohort or "current").strip().lower() != "all":
        try:
            from trading_crew.agentic.versioning import workflow_version as _wv

            workflow_version = _wv()
        except Exception:
            workflow_version = None

    report = await asyncio.to_thread(
        compute_calibration,
        symbol=ticker,
        n_bins=n_bins,
        use_alpha=(basis or "return").strip().lower() == "alpha",
        workflow_version=workflow_version,
    )
    return JSONResponse(report.to_dict())


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
async def runs_latest(
    ticker: str,
    full: bool = Query(False, description="Include agent report bodies"),
) -> JSONResponse:
    """Most recent run for ``ticker``. Lite header by default."""
    from trading_crew.agentic.runs import load_latest_run
    record = await asyncio.to_thread(load_latest_run, ticker, full)
    if not record:
        return JSONResponse({"error": "No runs recorded for this ticker"}, status_code=404)
    return JSONResponse(record)


@app.get("/api/runs/{ticker}/{run_id}/reports")
async def runs_reports(
    ticker: str,
    run_id: str,
    role: str = Query("", description="If set, only this agent role"),
) -> JSONResponse:
    """Agent report bodies for a saved run (one role, or all)."""
    from . import prediction_db as db
    payload = await asyncio.to_thread(db.get_run_reports, run_id, ticker, role or None)
    if payload is None:
        return JSONResponse({"error": "Run not found"}, status_code=404)
    return JSONResponse(payload)


@app.get("/api/runs/{ticker}/{run_id}")
async def runs_load(
    ticker: str,
    run_id: str,
    full: bool = Query(False, description="Include agent report bodies"),
) -> JSONResponse:
    """Saved run header (diagram + decision). Bodies via ``/reports`` or ``?full=1``."""
    from trading_crew.agentic.runs import load_run
    record = await asyncio.to_thread(load_run, run_id, ticker, full)
    if not record:
        return JSONResponse({"error": "Run not found"}, status_code=404)
    return JSONResponse(record)


@app.get("/api/search_stats")
async def search_stats_list(limit: int = Query(50, ge=1, le=500)) -> JSONResponse:
    """Recent sessions (trace ids) that recorded web-search calls."""
    from web.backend import prediction_db as db

    rows = await asyncio.to_thread(db.list_search_trace_ids, limit)
    return JSONResponse({"sessions": rows})


@app.get("/api/search_stats/{trace_id}")
async def search_stats_detail(trace_id: str) -> JSONResponse:
    """Per-backend web-search counters for one session (batch job_id / run_id)."""
    from web.backend import prediction_db as db

    stats = await asyncio.to_thread(db.get_search_stats, trace_id)
    return JSONResponse(stats)


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
    cpcv_groups: int = Query(6, ge=2, le=12, description="CPCV: number of contiguous groups (validation=cpcv only)"),
    cpcv_k: int = Query(2, ge=1, le=6, description="CPCV: groups held out per fold (validation=cpcv only)"),
    optimizer: str = Query("grid", description="grid (brute-force) | pso (particle-swarm)"),
    swarm_size: int = Query(12, ge=2, le=60, description="PSO: number of particles (optimizer=pso only)"),
    swarm_iterations: int = Query(8, ge=1, le=40, description="PSO: iterations (optimizer=pso only)"),
    swarm_seed: int = Query(0, description="PSO: RNG seed for reproducibility (optimizer=pso only)"),
    consensus: bool = Query(False, description="Also fuse the top-K configs into a consensus ensemble backtest"),
    consensus_top_k: int = Query(5, ge=2, le=25, description="How many configs to fuse when consensus=true"),
    lookback_years: int = Query(5, ge=1, le=10, description="Years of OHLCV history to fetch for the backtest (default 5y)"),
    workflow_version: str = Query("", description="Phase 0 cohort filter — restrict to this workflow_version (blank = all)"),
    source: str = Query("", description="Phase 0 cohort filter — 'live' | 'backfill' (blank = all)"),
    apply: bool = Query(False, description="Persist the winning config as this ticker's tuned sizing/gate config — applied at live inference AND walk-forward"),
) -> JSONResponse:
    """Agentic training L3 — sweep sizing/gate configs against logged
    proposals for ``ticker`` and return the best configuration by the
    user-specified ranking metric.

    The Default ranking is Deflated Sharpe (Bailey & López de Prado 2014),
    which penalises the multiple-comparison overfit that grid-searches
    notoriously fall into.  Coarse = 8 points (2^3), Default = 27 (3^3),
    Fine = 125 (5^3).  With ``validation=cpcv`` each config is scored on
    its combinatorial out-of-sample paths and the sweep also reports a
    Probability of Backtest Overfitting (PBO).
    """
    from datetime import datetime
    from trading_crew.agentic.backtest import CPCVConfig, WalkForwardConfig
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
    if workflow_version or source:
        from trading_crew.agentic.versioning import filter_episodes_by_cohort
        episodes = filter_episodes_by_cohort(
            episodes,
            workflow_version=workflow_version or None,
            source=source or None,
        )
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
    cpcv_cfg = CPCVConfig(n_groups=cpcv_groups, k_test=cpcv_k, embargo_size=embargo_size)

    def _persist_best(best: dict | None) -> dict | None:
        """When ``apply=1``, save the winning config as this ticker's tuned
        sizing/gate config so live inference + walk-forward both use it."""
        if not apply or not best:
            return None
        try:
            from trading_crew.agentic.tuning import save_tuned_config
            meta = {
                "rank_by": rank_by,
                "rank_metric": best.get("rank_metric"),
                "deflated_sharpe": (best.get("metrics") or {}).get("deflated_sharpe"),
                "sharpe": (best.get("metrics") or {}).get("sharpe"),
                "n_proposals": len(proposals),
                "validation": validation,
                "optimizer": optimizer,
                "cohort_workflow_version": workflow_version or None,
            }
            path = save_tuned_config(
                ticker,
                sizing_config=best.get("sizing_config") or {},
                gate_config=best.get("gate_config") or {},
                meta=meta,
                cache_dir=cache_dir,
            )
            return {"saved_to": str(path), "meta": meta}
        except Exception as exc:  # noqa: BLE001 — persistence is best-effort
            logger.warning("Failed to persist tuned config for %s: %s", ticker, exc)
            return {"error": str(exc)}

    def _consensus_block(members: list[tuple]) -> dict | None:
        """Fuse the supplied (sizing, gate, score) members into a consensus
        ensemble backtest and return its serialised result (or None)."""
        if not consensus or len(members) < 2:
            return None
        from trading_crew.agentic.swarm import ConsensusConfig, run_consensus_backtest
        cons = run_consensus_backtest(
            proposals=proposals,
            ohlcv_by_symbol={ticker: ohlcv},
            members=members,
            config=ConsensusConfig(top_k=consensus_top_k),
            cost_model_name=cost_model,
        )
        return cons.to_dict()

    # ---- Swarm (PSO) optimizer branch ------------------------------------
    if optimizer == "pso":
        from trading_crew.agentic.swarm import PSOConfig, run_swarm_search
        pso = await asyncio.to_thread(
            run_swarm_search,
            proposals=proposals,
            ohlcv_by_symbol={ticker: ohlcv},
            pso_config=PSOConfig(
                swarm_size=swarm_size, iterations=swarm_iterations, seed=swarm_seed,
            ),
            walk_forward_config=wf_cfg,
            cpcv_config=cpcv_cfg,
            cost_model_name=cost_model,
            rank_by=rank_by,
            validation=validation,
        )
        d = pso.to_dict()
        # Re-shape the swarm evaluations into the same "points" schema the
        # grid returns, so the UI table renders either optimizer uniformly.
        points = [
            {
                "sizing_config": e["sizing_config"],
                "gate_config": e["gate_config"],
                "metrics": e["metrics"],
                "rank_metric": e["score"],
                "iteration": e.get("iteration", 0),
                "cost_model": cost_model,
            }
            for e in d.get("evaluations", [])
        ]
        points.sort(key=lambda p: p.get("rank_metric", float("-inf")), reverse=True)
        members = [
            (e["sizing_config"], e["gate_config"], e["score"])
            for e in d.get("evaluations", [])
            if e.get("metrics")
        ]
        resp = {
            "ticker": ticker,
            "n_proposals": len(proposals),
            "optimizer": "pso",
            "validation": validation,
            "n_points": len(points),
            "points": points,
            "best": points[0] if points else None,
            "best_index": 0 if points else None,
            "best_params": d.get("best_params"),
            "history": d.get("history"),
            "n_evaluations": d.get("n_evaluations"),
        }
        cons_block = _consensus_block(members)
        if cons_block is not None:
            resp["consensus"] = cons_block
        applied = _persist_best(resp.get("best"))
        if applied is not None:
            resp["applied_tune"] = applied
        return JSONResponse(resp)

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
            validation=validation,
            cpcv_config=cpcv_cfg,
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
        base_dict["optimizer"] = "grid"
        base_dict["cost_models_swept"] = cost_models
        cons_members = [
            (p.get("sizing_config", {}), p.get("gate_config", {}), p.get("rank_metric", float("-inf")))
            for p in aggregated_points
            if p.get("metrics")
        ]
        cons_block = _consensus_block(cons_members)
        if cons_block is not None:
            base_dict["consensus"] = cons_block
        applied = _persist_best(base_dict.get("best"))
        if applied is not None:
            base_dict["applied_tune"] = applied
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
        "store_path": str(mem.db_path),
    })


# ---------------------------------------------------------------------------
# L4 — removed from the live prediction path (410, not a silent 404)
# ---------------------------------------------------------------------------

_L4_GONE = (
    "L4 PPO is removed from the live prediction path. "
    "Live sizing is the L3 applied tune composed with an open Hermes "
    "size_scale. The trainer library remains at trading_crew.agentic.rl."
)


def _l4_removed() -> None:
    raise HTTPException(status_code=410, detail=_L4_GONE)


@app.post("/api/training/rl/start")
async def rl_start(payload: Dict[str, Any]) -> JSONResponse:
    _l4_removed()


@app.post("/api/training/rl/stop")
async def rl_stop(payload: Dict[str, Any]) -> JSONResponse:
    _l4_removed()


@app.get("/api/training/rl/status")
async def rl_status(ticker: str = Query(..., min_length=1)) -> JSONResponse:
    _l4_removed()


@app.get("/api/training/rl/runs")
async def rl_runs(
    ticker: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
) -> JSONResponse:
    _l4_removed()


@app.get("/api/training/rl/runs/{ticker}/{run_id}")
async def rl_run_detail(ticker: str, run_id: str) -> JSONResponse:
    _l4_removed()


@app.post("/api/training/rl/promote")
async def rl_promote(payload: Dict[str, Any]) -> JSONResponse:
    _l4_removed()


@app.get("/api/training/rl/promoted")
async def rl_promoted() -> JSONResponse:
    _l4_removed()


@app.post("/api/training/rl/recommend")
async def rl_recommend(payload: Dict[str, Any]) -> JSONResponse:
    _l4_removed()


# ---------------------------------------------------------------------------
# Phase 4 — signal fusion (advisory). LLM + L3 expectancy + crowd.
# L4 policy is not assembled on the live path.
# ---------------------------------------------------------------------------


# Moved to web/backend/fusion.py so the workflow runner can reuse it for the
# optional in-run fusion step without importing app (circular).
from .fusion import assemble_fusion_inputs as _assemble_fusion_inputs


@app.get("/api/signals/fusion")
async def signals_fusion(
    ticker: str = Query(..., min_length=1),
    include_crowd: bool = Query(False, description="Also fetch the live crowd signal (slower)"),
    expectancy_source: str = Query(
        "pso",
        description="L3 expectancy source: 'pso' (swarm-tuned M5 config), "
                    "'proxy' (mean resolved-episode edge), or 'none' (disable gate)",
    ),
) -> JSONResponse:
    """Fuse LLM + L3 expectancy + crowd into one guardrailed signal (advisory)."""
    from trading_crew.agentic.signals import fuse_signals

    ticker = ticker.strip().upper()
    inputs, sources, err = await asyncio.to_thread(
        _assemble_fusion_inputs, ticker, include_crowd, expectancy_source
    )
    if err:
        return JSONResponse({"ticker": ticker, "available": False, "note": err, "sources": sources})
    fused = fuse_signals(
        inputs["llm"],
        policy=inputs["policy"],
        expectancy=inputs["expectancy"],
        crowd=inputs["crowd"],
    )
    return JSONResponse({
        "ticker": ticker,
        "available": True,
        "sources": sources,
        "fused": fused.to_dict(),
    })


# ---------------------------------------------------------------------------
# Harness — synthetic 30-day book so Hermes diagnosis can be exercised without
# waiting for live clocks.  Tagged SAMPLE-… and removable.
# ---------------------------------------------------------------------------


@app.post("/api/harness/sample/seed")
async def harness_sample_seed(payload: Dict[str, Any]) -> JSONResponse:
    """Seed synthetic 30-day reviews and run one harness step on those rows."""
    from trading_crew.agentic.harness import seed_and_step
    from trading_crew.agentic.memory.embedding import make_memory

    cache_dir = os.environ.get("TRADINGCREW_CACHE_DIR") or os.path.expanduser("~/.trading_crew")
    mem = make_memory(Path(cache_dir) / "memory" / "episodes.jsonl")
    return JSONResponse(await asyncio.to_thread(
        seed_and_step,
        count=int(payload.get("count") or 12),
        symbol=(payload.get("symbol") or "SAMPLE.NS").strip() or "SAMPLE.NS",
        memory=mem,
        fail=bool(payload.get("fail", True)),
        pattern=(payload.get("pattern") or None),
    ))


@app.post("/api/harness/sample/clear")
async def harness_sample_clear() -> JSONResponse:
    """Remove ``SAMPLE-…`` episodes and restore the pre-seed journal."""
    from trading_crew.agentic.harness import clear_sample_and_restore
    from trading_crew.agentic.memory.embedding import make_memory

    cache_dir = os.environ.get("TRADINGCREW_CACHE_DIR") or os.path.expanduser("~/.trading_crew")
    mem = make_memory(Path(cache_dir) / "memory" / "episodes.jsonl")
    return JSONResponse(await asyncio.to_thread(clear_sample_and_restore, memory=mem))


# ---------------------------------------------------------------------------
# Offline PIT decision-label backfill (Phase 2).  Runs the crew "as of" past
# dates to generate labeled decisions now; resumable + rate-limited.
# ---------------------------------------------------------------------------


@app.post("/api/backfill/start")
async def backfill_start(payload: Dict[str, Any]) -> JSONResponse:
    """Start (or resume) an offline point-in-time backfill.

    Body::

        {"tickers": ["RELIANCE.NS", "TCS.NS"], "start_date": "2023-01-01",
         "end_date": "2024-06-01", "spacing_days": 30, "horizon_days": 90,
         "debate_rounds": 1, "risk_rounds": 1, "cell_gap_seconds": 5,
         "resolve": true, "reflect": false, "include_unresolvable": false}
    """
    from .backfill_runner import start_backfill

    tickers = payload.get("tickers") or []
    if isinstance(tickers, str):
        tickers = [t.strip() for t in tickers.split(",") if t.strip()]
    try:
        snap = await asyncio.to_thread(
            start_backfill,
            tickers=tickers,
            start_date=(payload.get("start_date") or "").strip(),
            end_date=(payload.get("end_date") or "").strip(),
            spacing_days=int(payload.get("spacing_days") or 30),
            horizon_days=int(payload.get("horizon_days") or 90),
            debate_rounds=int(payload.get("debate_rounds") or 1),
            risk_rounds=int(payload.get("risk_rounds") or 1),
            cell_gap_seconds=float(payload.get("cell_gap_seconds") or 5.0),
            resolve=bool(payload.get("resolve", True)),
            reflect=bool(payload.get("reflect", False)),
            include_unresolvable=bool(payload.get("include_unresolvable", False)),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse(snap)


@app.post("/api/backfill/stop")
async def backfill_stop() -> JSONResponse:
    from .backfill_runner import stop_backfill
    return JSONResponse(await asyncio.to_thread(stop_backfill))


@app.get("/api/backfill/status")
async def backfill_status() -> JSONResponse:
    from .backfill_runner import get_backfill_status
    return JSONResponse(await asyncio.to_thread(get_backfill_status))


# ---------------------------------------------------------------------------
# Forward-collection scale job (UI-driven).  Runs the crew LIVE over the
# 200-stock universe with parallel workers; start / pause / resume / kill.
# Cross-process flock ensures only one collection runs at once.
# ---------------------------------------------------------------------------


@app.post("/api/scale/start")
async def scale_start(payload: Dict[str, Any]) -> JSONResponse:
    """Start (or resume-from-scratch) a forward-collection run.

    Body::

        {"universe": "all200", "workers": 6}
        {"tickers": ["RELIANCE.NS", "AAPL"], "workers": 4}
    """
    from .scale_runner import start_scale

    tickers = payload.get("tickers") or None
    if isinstance(tickers, str):
        tickers = [t.strip() for t in tickers.replace("\n", ",").split(",") if t.strip()]
    try:
        # start_scale is async — it schedules the orchestrator task on this loop
        # (needed so it can drive the SessionManager for live streaming).
        snap = await start_scale(
            universe=(payload.get("universe") or None),
            tickers=tickers,
            workers=int(payload.get("workers") or 5),
            day=(payload.get("day") or None),
            run_cfg=(payload.get("run_cfg") or None),
            force_rerun=bool(payload.get("force_rerun")),
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse(snap)


@app.post("/api/scale/pause")
async def scale_pause() -> JSONResponse:
    from .scale_runner import pause_scale
    return JSONResponse(await pause_scale())


@app.post("/api/scale/resume")
async def scale_resume() -> JSONResponse:
    from .scale_runner import resume_scale
    return JSONResponse(await resume_scale())


@app.post("/api/scale/stop")
async def scale_stop() -> JSONResponse:
    from .scale_runner import stop_scale
    return JSONResponse(await stop_scale())


@app.get("/api/scale/status")
async def scale_status() -> JSONResponse:
    from .scale_runner import get_scale_status
    return JSONResponse(await asyncio.to_thread(get_scale_status))


# ---------------------------------------------------------------------------
# Prediction dashboard (multi-page /predict UI, SQLite-backed)
# ---------------------------------------------------------------------------


@app.get("/api/predict/universe")
async def predict_universe(run_date: str = Query("", description="YYYY-MM-DD; defaults to today")) -> JSONResponse:
    """Editable universe (categories + companies) with per-day prediction status."""
    from . import prediction_db as db
    data = await asyncio.to_thread(db.list_universe, run_date or None)
    return JSONResponse(data)


@app.get("/api/predict/search")
async def predict_search(
    q: str = Query(..., min_length=1),
    limit: int = Query(8, ge=1, le=20),
    market: str = Query("", description="IN | US | '' (auto) — IN constructs/validates .NS candidates"),
) -> JSONResponse:
    """Resolve a company-name query to real, runnable tickers (Yahoo search)."""
    from .company_meta import search_symbols
    try:
        results = await asyncio.to_thread(search_symbols, q, limit, market)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"symbol search failed: {exc}")
    return JSONResponse({"query": q, "results": results})


@app.post("/api/predict/company")
async def predict_add_company(payload: Dict[str, Any]) -> JSONResponse:
    """Add a company to the universe under a sector (from the search box)."""
    from . import prediction_db as db
    from .company_meta import enrich_ticker

    ticker = (payload.get("ticker") or "").strip()
    if not ticker:
        raise HTTPException(status_code=400, detail="ticker is required")
    name = (payload.get("name") or "").strip()
    homepage = payload.get("homepage")
    wikipedia = payload.get("wikipedia")
    sector = (payload.get("sector") or "").strip()
    # Fill missing name/links/sector live so the new tile lands in the right
    # place and is immediately complete (name + hover links).
    if not name or not homepage or not sector:
        meta = await asyncio.to_thread(enrich_ticker, ticker)
        name = name or (meta.get("name") or "")
        homepage = homepage or meta.get("homepage")
        wikipedia = wikipedia or meta.get("wikipedia")
        sector = sector or (meta.get("sector") or "")
    try:
        company = await asyncio.to_thread(
            db.add_company,
            ticker=ticker,
            name=name,
            sector=(sector or "Other"),
            market=(payload.get("market") or "US"),
            homepage=homepage,
            wikipedia=wikipedia,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse({"ok": True, "company": company})


@app.delete("/api/predict/company/{ticker}")
async def predict_remove_company(ticker: str) -> JSONResponse:
    from . import prediction_db as db
    removed = await asyncio.to_thread(db.remove_company, ticker)
    if not removed:
        raise HTTPException(status_code=404, detail=f"{ticker} not in universe")
    return JSONResponse({"ok": True, "ticker": ticker})


@app.post("/api/predict/enrich")
async def predict_enrich(limit: int = Query(20, ge=1, le=100)) -> JSONResponse:
    """Resolve name/homepage/wikipedia for up to ``limit`` un-enriched companies.

    The frontend calls this in a loop after loading the universe (each yfinance
    lookup is ~0.5–1s), refreshing tiles until ``remaining`` hits zero.
    """
    from . import prediction_db as db
    from .company_meta import enrich_ticker

    def _run() -> Dict[str, Any]:
        pending = db.companies_needing_meta(limit)
        enriched = 0
        for row in pending:
            meta = enrich_ticker(row["ticker"])
            db.set_company_meta(
                row["ticker"],
                name=meta.get("name"),
                homepage=meta.get("homepage"),
                wikipedia=meta.get("wikipedia"),
            )
            enriched += 1
        remaining = len(db.companies_needing_meta(1))
        return {"enriched": enriched, "remaining": remaining}

    return JSONResponse(await asyncio.to_thread(_run))


@app.get("/api/predict/prediction/{ticker}")
async def predict_prediction(
    ticker: str,
    run_date: str = Query("", description="YYYY-MM-DD; defaults to latest"),
) -> JSONResponse:
    """The stored prediction (decision + report) for a ticker (latest or dated)."""
    from . import prediction_db as db
    pred = await asyncio.to_thread(db.get_prediction, ticker, run_date or None)
    company = await asyncio.to_thread(db.get_company, ticker)
    return JSONResponse({"ticker": ticker, "company": company, "prediction": pred})


@app.get("/api/predict/history")
async def predict_history() -> JSONResponse:
    """List historical run dates for the front-page history panel."""
    from . import prediction_db as db
    return JSONResponse({"days": await asyncio.to_thread(db.list_run_days)})


@app.get("/api/predict/history/{run_date}")
async def predict_history_day(run_date: str) -> JSONResponse:
    """Read-only snapshot of all predictions made on ``run_date``."""
    from . import prediction_db as db
    return JSONResponse(await asyncio.to_thread(db.get_day_snapshot, run_date))


@app.get("/api/portfolio/virtual")
async def virtual_portfolio_days() -> JSONResponse:
    """Run dates that have a virtual book (== any run day with predictions)."""
    from . import prediction_db as db
    return JSONResponse({"days": await asyncio.to_thread(db.list_run_days)})


@app.get("/api/portfolio/virtual/{run_date}")
async def virtual_portfolio_day(
    run_date: str,
    market: str = Query("all", description="Investable universe: all | IN | US"),
) -> JSONResponse:
    """Growth-weighted ₹10L virtual book for ``run_date``, marked to the latest
    known price of each held name (see web/backend/vportfolio.py).  ``market``
    restricts the universe and re-deploys the full capital within it."""
    from . import vportfolio
    return JSONResponse(await asyncio.to_thread(
        vportfolio.compute_portfolio, run_date, vportfolio.DEFAULT_CAPITAL_INR, market))


@app.get("/api/backtest")
async def backtest(
    ticker: str = Query(..., min_length=1),
    train_size: int = Query(3, ge=1, le=500),
    embargo_size: int = Query(1, ge=0, le=30),
    test_size: int = Query(1, ge=1, le=200),
    cost_model: str = Query("standard"),
    lookback_years: int = Query(5, ge=1, le=10, description="Years of OHLCV history to fetch for the backtest (default 5y)"),
    validation: str = Query("walk_forward", description="walk_forward | cpcv (Combinatorial Purged CV)"),
    cpcv_groups: int = Query(6, ge=2, le=12, description="CPCV: number of contiguous groups (validation=cpcv only)"),
    cpcv_k: int = Query(2, ge=1, le=6, description="CPCV: groups held out per fold (validation=cpcv only)"),
    workflow_version: str = Query("", description="Phase 0 cohort filter — restrict to episodes stamped with this workflow_version (blank = latest version, see all_versions)"),
    source: str = Query("", description="Phase 0 cohort filter — 'live' | 'backfill' (blank = all)"),
    all_versions: bool = Query(False, description="Backtest across ALL workflow versions instead of only the latest (default: latest-only, the trustworthy cohort)"),
) -> JSONResponse:
    """M6 — walk-forward (or CPCV) backtest of logged proposals for ``ticker``.

    By default this restricts to the **latest workflow version** so the
    walk-forward measures the *current* workflow rather than a mix of drifting
    past versions (whose proposals are not comparable).  Pass ``all_versions=1``
    (or an explicit ``workflow_version``) to override.
    """
    from datetime import datetime
    from trading_crew.agentic.backtest import (
        CPCVConfig,
        WalkForwardConfig,
        generate_cpcv_folds,
        generate_folds,
        run_cpcv,
        run_walk_forward,
    )
    from trading_crew.agentic.execution.contracts import ActionProposal
    from .charts import _fetch_ohlcv

    # Shared library path (also used by `python -m trading_crew.agentic.validate`)
    # so the web app and the CLI harness build the cohort + proposal timeline
    # identically instead of re-implementing it.
    from trading_crew.agentic.evaluation import (
        latest_workflow_version,
        load_cohort,
        proposals_from_episodes,
    )

    cache_dir = os.environ.get("TRADINGCREW_CACHE_DIR") or os.path.expanduser("~/.trading_crew")
    episodes = load_cohort(
        ticker,
        cache_dir=cache_dir,
        workflow_version=workflow_version or None,
        source=source or None,
    )
    if not episodes:
        return JSONResponse({
            "ticker": ticker, "n_proposals": 0,
            "error": "No episodes recorded for this ticker yet — run an analysis first.",
        })

    # Default to the LATEST workflow version so the walk-forward reflects the
    # current workflow, not a blend of older (incomparable) versions.  Skipped
    # when the caller pins an explicit workflow_version or asks for all_versions.
    n_all_versions = len(episodes)
    effective_version = workflow_version or None
    if not workflow_version and not all_versions:
        from trading_crew.agentic.versioning import filter_episodes_by_cohort

        effective_version = latest_workflow_version(episodes)
        if effective_version:
            episodes = filter_episodes_by_cohort(episodes, workflow_version=effective_version)
    cohort = {
        "workflow_version": effective_version,
        "all_versions": bool(all_versions),
        "n_all_versions": n_all_versions,
    }

    proposals: list[ActionProposal] = proposals_from_episodes(episodes)
    if not proposals:
        return JSONResponse({
            "ticker": ticker, "n_proposals": 0,
            "error": "All recorded episodes had invalid proposals.",
            **cohort,
        })

    last_ts = max(p.decision_ts for p in proposals)
    try:
        end_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        end_dt = datetime.utcnow()
    ohlcv = await asyncio.to_thread(_fetch_ohlcv, ticker, end_dt, 365 * lookback_years)

    def _finite(v):
        try:
            if v != v or v in (float("inf"), float("-inf")):
                return None
        except Exception:
            return None
        return v

    def _metrics_dict(m) -> dict:
        return {
            "total_return_pct": m.total_return_pct,
            "cagr": m.cagr,
            "annualised_vol": m.annualised_vol,
            "sharpe": m.sharpe,
            "sortino": _finite(m.sortino),
            "calmar": _finite(m.calmar),
            "max_drawdown": m.max_drawdown,
            "deflated_sharpe": m.deflated_sharpe,
            "n_periods": m.n_periods,
            "win_rate": m.win_rate,
            "expectancy": m.expectancy,
            "payoff_ratio": _finite(m.payoff_ratio),
            "profit_factor": _finite(m.profit_factor),
            "benchmark_return_pct": m.benchmark_return_pct,
            "alpha": m.alpha,
            "beta": m.beta,
            "sharpe_ci_low": m.sharpe_ci_low,
            "sharpe_ci_high": m.sharpe_ci_high,
        }

    # L3 tune (if applied) composed with an open Hermes size experiment —
    # same composition the live runner uses.
    from trading_crew.agentic.tuning import load_tuned_entry, resolve_live_sizing

    _sizing, _gate, _sz_meta = resolve_live_sizing(ticker, cache_dir=cache_dir)
    _cfg_kwargs: dict = {}
    tuned_info: dict = {
        "applied": bool(_sz_meta.get("l3_applied")),
        "hermes_size_scale": _sz_meta.get("hermes_size_scale"),
    }
    if _sizing is not None:
        _cfg_kwargs["sizing_config"] = _sizing
        _cfg_kwargs["gate_config"] = _gate
        _entry = load_tuned_entry(ticker, cache_dir=cache_dir) or {}
        tuned_info["meta"] = _entry.get("meta")
        tuned_info["saved_ts"] = _entry.get("saved_ts")

    # ---- CPCV branch ------------------------------------------------------
    if validation == "cpcv":
        cpcv_cfg = CPCVConfig(
            n_groups=cpcv_groups, k_test=cpcv_k, embargo_size=embargo_size,
        )
        cpcv_folds = generate_cpcv_folds(n_obs=len(proposals), config=cpcv_cfg)
        if not cpcv_folds:
            return JSONResponse({
                "ticker": ticker, "n_proposals": len(proposals),
                "error": (
                    f"Not enough proposals ({len(proposals)}) for CPCV with "
                    f"n_groups={cpcv_groups}, k_test={cpcv_k}."
                    + (
                        f" (latest workflow version only — {n_all_versions - len(proposals)} "
                        f"older-version proposals excluded; pass all_versions=1 to include them.)"
                        if effective_version and not all_versions and n_all_versions > len(proposals)
                        else ""
                    )
                ),
                **cohort,
            })
        cpcv = await asyncio.to_thread(
            run_cpcv,
            proposals=proposals,
            ohlcv_by_symbol={ticker: ohlcv},
            cpcv_folds=cpcv_folds,
            cost_model_name=cost_model,
            **_cfg_kwargs,
        )
        return JSONResponse({
            "ticker": ticker,
            "n_proposals": len(proposals),
            **cohort,
            "tuned_config": tuned_info,
            "validation": "cpcv",
            "n_folds": len(cpcv.paths),
            "combined_equity": cpcv.combined_equity,
            "combined_timestamps": cpcv.combined_timestamps,
            "combined_benchmark": cpcv.combined_benchmark,
            "overall_metrics": _metrics_dict(cpcv.aggregate_metrics),
            "cpcv": {
                "oos_sharpes": [_finite(s) for s in cpcv.oos_sharpes],
                "mean_oos_sharpe": _finite(cpcv.mean_oos_sharpe),
                "median_oos_sharpe": _finite(cpcv.median_oos_sharpe),
                "sharpe_std": _finite(cpcv.sharpe_std),
                "n_paths": len(cpcv.paths),
            },
            "folds": [
                {
                    "fold_id": p.fold_id,
                    "metrics": {
                        "total_return_pct": p.metrics.total_return_pct,
                        "sharpe": p.metrics.sharpe,
                        "max_drawdown": p.metrics.max_drawdown,
                        "n_periods": p.metrics.n_periods,
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
                        for t in p.trades
                    ],
                }
                for p in cpcv.paths
            ],
        })

    # ---- Walk-forward branch ---------------------------------------------
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
                + (
                    f" (latest workflow version only — {n_all_versions - len(proposals)} "
                    f"older-version proposals excluded; pass all_versions=1 to include them.)"
                    if effective_version and not all_versions and n_all_versions > len(proposals)
                    else ""
                )
            ),
            **cohort,
        })

    result = await asyncio.to_thread(
        run_walk_forward,
        proposals=proposals,
        ohlcv_by_symbol={ticker: ohlcv},
        folds=folds,
        cost_model_name=cost_model,
        **_cfg_kwargs,
    )

    # Phase 3 — decompose realised PnL into decision / sizing / cost legs so
    # the desk can see WHICH layer to fix, not just the headline Sharpe.
    attribution = None
    try:
        from trading_crew.agentic.backtest import (
            attribute_decisions,
            sized_weights_from_trades,
        )

        all_trades = [t for fr in result.folds for t in fr.trades]
        attribution = await asyncio.to_thread(
            attribute_decisions,
            proposals,
            {ticker: ohlcv},
            sized_weights=sized_weights_from_trades(all_trades),
            cost_model_name=cost_model,
        )
    except Exception:  # noqa: BLE001 — attribution is diagnostic, never fatal
        attribution = None

    return JSONResponse({
        "ticker": ticker,
        "n_proposals": len(proposals),
        **cohort,
        "tuned_config": tuned_info,
        "validation": "walk_forward",
        "n_folds": len(result.folds),
        "attribution": attribution.to_dict() if attribution is not None else None,
        "combined_equity": result.combined_equity,
        "combined_timestamps": result.combined_timestamps,
        "combined_benchmark": result.combined_benchmark,
        "overall_metrics": _metrics_dict(result.overall_metrics),
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


@app.get("/api/portfolio_backtest")
async def portfolio_backtest(
    tickers: str = Query(..., min_length=1, description="Comma-separated tickers, e.g. AAPL,MSFT,RELIANCE.NS"),
    cost_model: str = Query("standard"),
    allocation_method: str = Query("HRP", description="HRP | MEAN_VARIANCE | EQUAL_RISK"),
    gross_budget: float = Query(0.80, ge=0.05, le=2.0, description="Max sum of |weights| across the book"),
    max_position_weight: float = Query(0.20, ge=0.01, le=1.0),
    lookback_years: int = Query(5, ge=1, le=10),
    all_versions: bool = Query(False, description="Replay across ALL workflow versions instead of only the latest (default: latest-only, matches Walk forward)"),
) -> JSONResponse:
    """M6 + M7 — multi-asset portfolio backtest.

    Loads every ticker's logged proposals, then replays them as one
    coherent book: at each rebalance date the M7 allocator sizes the
    cross-sectional weight vector (HRP / MV / ERC) before execution.
    """
    from datetime import datetime
    from trading_crew.agentic.backtest import run_portfolio_backtest
    from trading_crew.agentic.execution.contracts import ActionProposal
    from trading_crew.agentic.memory import EpisodicMemory
    from trading_crew.agentic.portfolio.allocator import (
        AllocationMethod, AllocatorConfig,
    )
    from trading_crew.agentic.risk import GateConfig
    from .charts import _fetch_ohlcv

    symbols = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if len(symbols) < 2:
        return JSONResponse({
            "tickers": symbols,
            "error": "Portfolio backtest needs at least 2 tickers.",
        })

    cache_dir = os.environ.get("TRADINGCREW_CACHE_DIR") or os.path.expanduser("~/.trading_crew")
    store_path = Path(cache_dir) / "memory" / "episodes.jsonl"
    mem = EpisodicMemory(store_path)

    from trading_crew.agentic.evaluation import (
        latest_workflow_version,
        proposals_from_episodes,
    )
    from trading_crew.agentic.versioning import filter_episodes_by_cohort

    symbol_set = set(symbols)
    all_eps = [e for e in mem.all_episodes() if e.symbol.upper() in symbol_set]
    n_all_versions = len(all_eps)

    # Default to the LATEST workflow version so the book is one coherent agent
    # build across all names (matches Walk forward).  A single global version is
    # used — not per-ticker — so cross-sectional weights stay comparable.
    effective_version = None
    if not all_versions:
        effective_version = latest_workflow_version(all_eps)
        if effective_version:
            all_eps = filter_episodes_by_cohort(all_eps, workflow_version=effective_version)
    cohort = {
        "workflow_version": effective_version,
        "all_versions": bool(all_versions),
        "n_all_versions": n_all_versions,
    }

    proposals: list[ActionProposal] = []
    per_ticker_counts: dict[str, int] = {}
    ohlcv_by_symbol: dict = {}
    for sym in symbols:
        eps = [e for e in all_eps if e.symbol.upper() == sym]
        ps = proposals_from_episodes(eps)
        proposals.extend(ps)
        per_ticker_counts[sym] = len(ps)

    if not proposals:
        return JSONResponse({
            "tickers": symbols, "n_proposals": 0,
            "error": (
                "No episodes recorded for these tickers yet — run analyses first."
                + (
                    f" (latest workflow version only — {n_all_versions} older-version "
                    f"proposals excluded; pass all_versions=1 to include them.)"
                    if effective_version and n_all_versions > 0
                    else ""
                )
            ),
            **cohort,
        })

    try:
        last_ts = max(p.decision_ts for p in proposals)
        end_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        end_dt = datetime.utcnow()

    for sym in symbols:
        df = await asyncio.to_thread(_fetch_ohlcv, sym, end_dt, 365 * lookback_years)
        if df is not None and not df.empty:
            ohlcv_by_symbol[sym] = df

    if len(ohlcv_by_symbol) < 2:
        return JSONResponse({
            "tickers": symbols, "n_proposals": len(proposals),
            "error": "Could not fetch OHLCV for at least 2 of the tickers.",
        })

    try:
        method = AllocationMethod(allocation_method)
    except ValueError:
        method = AllocationMethod.HRP
    alloc_cfg = AllocatorConfig(
        method=method,
        gross_budget=gross_budget,
        max_position_weight=max_position_weight,
    )

    result = await asyncio.to_thread(
        run_portfolio_backtest,
        proposals=proposals,
        ohlcv_by_symbol=ohlcv_by_symbol,
        cost_model_name=cost_model,
        gate_config=GateConfig(),
        allocator_config=alloc_cfg,
    )

    def _finite(v):
        try:
            if v != v or v in (float("inf"), float("-inf")):
                return None
        except Exception:
            return None
        return v

    return JSONResponse({
        "tickers": symbols,
        **cohort,
        "per_ticker_proposals": per_ticker_counts,
        "n_proposals": len(proposals),
        "method_used": result.method_used,
        "gross_budget": gross_budget,
        "equity_curve": result.equity_curve,
        "timestamps": result.timestamps,
        "benchmark_curve": result.benchmark_curve,
        "weights_over_time": result.weights_over_time,
        "metrics": {
            "total_return_pct": result.metrics.total_return_pct,
            "cagr": result.metrics.cagr,
            "annualised_vol": result.metrics.annualised_vol,
            "sharpe": result.metrics.sharpe,
            "sortino": _finite(result.metrics.sortino),
            "calmar": _finite(result.metrics.calmar),
            "max_drawdown": result.metrics.max_drawdown,
            "deflated_sharpe": result.metrics.deflated_sharpe,
            "n_periods": result.metrics.n_periods,
            "win_rate": result.metrics.win_rate,
            "expectancy": result.metrics.expectancy,
            "payoff_ratio": _finite(result.metrics.payoff_ratio),
            "profit_factor": _finite(result.metrics.profit_factor),
            "benchmark_return_pct": result.metrics.benchmark_return_pct,
            "alpha": result.metrics.alpha,
            "beta": result.metrics.beta,
            "sharpe_ci_low": result.metrics.sharpe_ci_low,
            "sharpe_ci_high": result.metrics.sharpe_ci_high,
        },
        "trades": [
            {
                "ts": t.ts, "symbol": t.symbol, "side": t.side,
                "intent_weight": t.intent_weight, "sized_weight": t.sized_weight,
                "fill_qty": t.fill_qty, "fill_price": t.fill_price,
                "fees": t.fees, "status": t.status,
                "rejection_reason": t.rejection_reason,
            }
            for t in result.trades
        ],
    })


@app.get("/api/crowd_positioning")
async def crowd_positioning(
    ticker: str = Query(..., min_length=1),
) -> JSONResponse:
    """Swarm crowd-positioning signal for ``ticker`` (options + insider +
    analyst crowding fused into a score in [-1, +1])."""
    from trading_crew.agentic.swarm.crowd import compute_crowd_signal
    from trading_crew.agentic.swarm.crowd_fit import load_crowd_config
    from trading_crew.tools import _gather_crowd_inputs

    def _work():
        inputs, notes = _gather_crowd_inputs(ticker)
        signal = compute_crowd_signal(inputs, load_crowd_config())
        return signal, notes

    try:
        signal, notes = await asyncio.to_thread(_work)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"ticker": ticker, "error": str(exc)})

    return JSONResponse({
        "ticker": ticker,
        "score": signal.score,
        "intensity": signal.intensity,
        "label": signal.label,
        "components": signal.components,
        "n_components": signal.n_components,
        "notes": notes,
    })


# ---------------------------------------------------------------------------
# Session-aware run API
# ---------------------------------------------------------------------------
# The endpoints below decouple "the run" from "the viewer".  A run is
# owned by the process-wide ``SessionManager`` and lives on until it
# terminates (or is evicted after cap); WebSocket connections attach
# and detach freely without ever cancelling the underlying kickoff.
#
# The paths are namespaced under ``/api/sessions`` (and ``/ws/session``)
# to avoid any resolution ambiguity with the pre-existing on-disk
# ``/api/runs/{ticker}/{run_id}`` GET route that reads persisted
# RunRecords — a session id passed to that path would otherwise
# accidentally match ``{ticker}=<id>, {run_id}=cancel``.
#
# Flow (new client):
#   1. POST /api/sessions            {ticker, ...cfg}       -> {run_id}
#   2. WS   /ws/session/{run_id}                             -> replay + tail
#   3. POST /api/sessions/{run_id}/cancel                    -> stop mid-run
#   4. GET  /api/sessions                                    -> resume after refresh
#
# The legacy ``/ws/analyze`` endpoint still works but is now a thin
# shim over the same manager (starts a session, subscribes the caller,
# then intentionally *does not* cancel on client disconnect).

@app.post("/api/sessions")
async def api_sessions_start(payload: Dict[str, Any]) -> JSONResponse:
    """Kick off a new analysis in the background.

    Returns immediately with a ``run_id`` the client can subscribe to
    over ``/ws/run/{run_id}``.  Sessions beyond the configured
    concurrency cap sit in QUEUED and promote automatically when a
    running slot frees up.
    """
    ticker = (payload.get("ticker") or "").strip()
    if not ticker:
        raise HTTPException(status_code=400, detail="ticker is required")
    asset = str(payload.get("asset_class") or "stock").lower()
    force_rerun = bool(payload.get("force_rerun"))
    # Same-day stock cache: a completed Prediction-index row is returned
    # as-is. Failed rows are not hits. Commodity never uses this index.
    if not force_rerun and asset != "commodity":
        from . import prediction_db as db

        hit = db.cached_session_payload(ticker)
        if hit:
            return JSONResponse(hit)
    sm = get_session_manager()
    session = await sm.start(ticker, payload)
    return JSONResponse({
        "run_id": session.run_id,
        "status": session.status.value,
        "ticker": session.ticker,
        "started_at": session.started_at.isoformat(),
    })


@app.get("/api/sessions")
async def api_sessions_list() -> JSONResponse:
    """List every in-memory run (queued / running / recently terminated)
    so a reloaded page can rebuild the sidebar without losing state."""
    sm = get_session_manager()
    return JSONResponse({"runs": sm.list()})


@app.post("/api/sessions/{run_id}/cancel")
async def api_sessions_cancel(run_id: str) -> JSONResponse:
    sm = get_session_manager()
    ok = await sm.cancel(run_id)
    if not ok:
        # 404 covers both "no such run" and "already terminated" — the
        # frontend just refetches /api/sessions to reconcile.
        raise HTTPException(status_code=404, detail="No cancellable run with that id")
    return JSONResponse({"ok": True, "run_id": run_id})


@app.delete("/api/sessions/{run_id}")
async def api_sessions_delete(run_id: str) -> JSONResponse:
    """Evict a *terminal* session from the manager's memory."""
    sm = get_session_manager()
    session = sm.get(run_id)
    if session is None:
        raise HTTPException(status_code=404, detail="No such run")
    if not sm.remove(run_id):
        raise HTTPException(
            status_code=409,
            detail="Session is still active — cancel it before deleting",
        )
    return JSONResponse({"ok": True, "run_id": run_id})


async def _pipe_session_to_ws(ws: WebSocket, run_id: str) -> None:
    """Attach a WebSocket to a live session.

    Sends every buffered event first (so a page reload rebuilds state
    from scratch), then tails the subscriber queue until the run
    terminates OR the WebSocket disconnects.  Disconnect is *not*
    treated as cancel — the underlying kickoff keeps running.
    """
    sm = get_session_manager()
    session, replay, q = sm.subscribe(run_id)
    if session is None or q is None:
        await ws.send_text(json.dumps({"type": "error", "message": f"unknown run_id {run_id}"}))
        return
    try:
        for ev in replay:
            await ws.send_text(json.dumps(ev, default=str))
        while True:
            ev = await q.get()
            if ev is _EOF:
                return
            await ws.send_text(json.dumps(ev, default=str))
    except WebSocketDisconnect:
        # Client walked away — the manager keeps the run going for
        # anyone else who's subscribed and for future reconnects.
        logger.info("WebSocket detached from %s (run continues)", run_id)
    finally:
        sm.unsubscribe(session, q)


@app.websocket("/ws/session/{run_id}")
async def ws_session(ws: WebSocket, run_id: str) -> None:
    """Preferred WebSocket endpoint for the new session-aware UI."""
    await ws.accept()
    try:
        await _pipe_session_to_ws(ws, run_id)
    except Exception:
        logger.exception("ws_run failed for %s", run_id)
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


@app.websocket("/ws/analyze")
async def analyze_ws(ws: WebSocket) -> None:
    """Legacy compat: start a run *and* subscribe on the same socket.

    Behavioural change vs. pre-session versions: closing this
    WebSocket no longer cancels the run.  The client must issue
    ``POST /api/runs/{run_id}/cancel`` explicitly.  This is a
    deliberate migration path — old bookmarks and integrations keep
    working, but they get the new "refresh-safe" semantics for free.
    """
    await ws.accept()
    try:
        raw = await ws.receive_text()
        try:
            cfg: Dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError:
            await ws.send_text(json.dumps({"type": "error", "message": "Invalid JSON"}))
            return

        ticker = (cfg.get("ticker") or "").strip()
        if not ticker:
            await ws.send_text(json.dumps({"type": "error", "message": "ticker is required"}))
            return

        sm = get_session_manager()
        session = await sm.start(ticker, cfg)
        # Announce the id so clients that care can reconnect later.
        await ws.send_text(json.dumps({"type": "session_started", "run_id": session.run_id}))
        await _pipe_session_to_ws(ws, session.run_id)
    except WebSocketDisconnect:
        logger.info("analyze_ws client disconnected (run continues in session manager)")
    except Exception:
        logger.exception("analyze_ws failed")
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
    port = int(os.getenv("TRADINGCREW_WEB_PORT", "8002"))
    uvicorn.run(
        "web.backend.app:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
