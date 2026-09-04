"""Programmatic Prediction API (``/api/v1``).

Same FastAPI process and SessionManager as the UI. Callers send ticker
(and debate/risk rounds when ``api_run.yaml`` is not harness); every
other kickoff knob comes from ``trading_crew/config/api_run.yaml``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from trading_crew.api_run import (
    ApiRunConfigError,
    build_session_cfg,
    config_path,
    load_api_run_config,
    prompt_files,
)
from web.backend.session_manager import get_session_manager

router = APIRouter(prefix="/api/v1", tags=["prediction-api"])


class StartRunBody(BaseModel):
    ticker: str = Field(..., min_length=1, description="Equity ticker, e.g. NTNX or RELIANCE.NS")
    debate_rounds: Optional[int] = Field(default=None, ge=1, le=8)
    risk_rounds: Optional[int] = Field(default=None, ge=1, le=8)
    force_rerun: Optional[bool] = Field(
        default=None,
        description="If set, overrides force_rerun in the API run config",
    )


def _cfg_error(exc: ApiRunConfigError) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


def _live_status(run_id: str) -> Optional[Dict[str, Any]]:
    sm = get_session_manager()
    session = sm.get(run_id)
    if session is None:
        return None
    snap = session.snapshot()
    snap["source"] = "live"
    return snap


def _persisted_status(run_id: str) -> Optional[Dict[str, Any]]:
    from web.backend import prediction_db as db

    row = db.find_run(run_id)
    if row is None:
        return None
    return {
        "run_id": row["run_id"],
        "ticker": row["ticker"],
        "status": row["status"] or "completed",
        "started_at": row.get("started_at"),
        "completed_at": row.get("completed_at"),
        "final_action": row.get("final_action"),
        "error": None,
        "event_count": None,
        "source": "persisted",
        "cfg": {
            "cascade_route": row.get("cascade_route"),
            "final_size_pct": row.get("final_size_pct"),
        },
    }


@router.get("/health")
async def api_health() -> dict:
    """Liveness plus whether the LLM endpoint and API config are present."""
    llm = bool(
        (os.getenv("VLLM_LLM_BASE_URL") and os.getenv("VLLM_LLM_MODEL"))
        or (os.getenv("LOCAL_LLM_BASE_URL") and os.getenv("LOCAL_LLM_MODEL"))
    )
    cfg_path = config_path()
    live = 0
    try:
        live = len(get_session_manager().list())
    except RuntimeError:
        live = 0
    return {
        "ok": True,
        "service": "tradingcrew",
        "api": "v1",
        "llm_configured": llm,
        "config_path": str(cfg_path),
        "config_present": cfg_path.is_file(),
        "live_sessions": live,
    }


@router.get("/config")
async def api_config() -> JSONResponse:
    """Resolved run knobs and prompt-file locations."""
    try:
        cfg = load_api_run_config()
    except ApiRunConfigError as exc:
        raise _cfg_error(exc) from exc
    return JSONResponse({
        "config": cfg,
        "prompts": prompt_files(),
    })


@router.post("/runs")
async def api_start_run(body: StartRunBody) -> JSONResponse:
    """Queue a stock prediction. Returns immediately with ``run_id``."""
    try:
        cfg = build_session_cfg(
            ticker=body.ticker,
            debate_rounds=body.debate_rounds,
            risk_rounds=body.risk_rounds,
            force_rerun=body.force_rerun,
        )
    except ApiRunConfigError as exc:
        raise _cfg_error(exc) from exc
    ticker = cfg["ticker"]
    if not cfg["force_rerun"] and cfg["asset_class"] != "commodity":
        from web.backend import prediction_db as db

        hit = db.cached_session_payload(ticker)
        if hit:
            return JSONResponse({
                "cached": True,
                "run_id": hit.get("run_id") or "",
                "ticker": ticker,
                "status": hit.get("status") or "completed",
                "run_date": hit.get("run_date"),
            })
    sm = get_session_manager()
    session = await sm.start(ticker, cfg)
    body_out: Dict[str, Any] = {
        "cached": False,
        "run_id": session.run_id,
        "ticker": session.ticker,
        "status": session.status.value,
        "started_at": session.started_at.isoformat(),
        "run_mode": cfg["run_mode"],
    }
    if "debate_rounds" in cfg:
        body_out["debate_rounds"] = cfg["debate_rounds"]
        body_out["risk_rounds"] = cfg["risk_rounds"]
    return JSONResponse(body_out)


@router.get("/runs")
async def api_list_runs() -> JSONResponse:
    """Live and recently-finished sessions still held in memory."""
    sm = get_session_manager()
    rows = sm.list()
    return JSONResponse({"runs": rows, "count": len(rows)})


@router.get("/runs/{run_id}")
async def api_run_status(run_id: str) -> JSONResponse:
    """Live session first; persisted row after the process has dropped it."""
    live = _live_status(run_id)
    if live is not None:
        return JSONResponse(live)
    persisted = _persisted_status(run_id)
    if persisted is not None:
        return JSONResponse(persisted)
    raise HTTPException(status_code=404, detail="run not found")


@router.post("/runs/{run_id}/stop")
async def api_stop_run(run_id: str) -> JSONResponse:
    """Cancel a queued or running session."""
    sm = get_session_manager()
    ok = await sm.cancel(run_id)
    if not ok:
        raise HTTPException(status_code=404, detail="No cancellable run with that id")
    return JSONResponse({"ok": True, "run_id": run_id, "status": "cancelled"})


@router.get("/runs/{run_id}/result")
async def api_run_result(run_id: str) -> JSONResponse:
    """Lite persisted record (decision + report stubs). 409 while still live."""
    live = _live_status(run_id)
    if live is not None and live["status"] in {"queued", "running"}:
        raise HTTPException(status_code=409, detail="run is still in progress")
    from web.backend import prediction_db as db

    row = db.find_run(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="no persisted result for this run")
    record = db.get_run(run_id, row["ticker"], full=False)
    if record is None:
        raise HTTPException(status_code=404, detail="no persisted result for this run")
    return JSONResponse({
        "run_id": run_id,
        "ticker": row["ticker"],
        "status": record.get("status") or row.get("status"),
        "final_decision": record.get("final_decision"),
        "action_proposal": record.get("action_proposal"),
        "reports": record.get("reports") or {},
        "reports_by_round": record.get("reports_by_round") or {},
        "degraded_roles": record.get("degraded_roles") or [],
    })


@router.get("/runs/{run_id}/reports")
async def api_run_reports(
    run_id: str,
    role: str = Query("", description="If set, only this agent role"),
) -> JSONResponse:
    """Agent report bodies (one role, or all)."""
    from web.backend import prediction_db as db

    row = db.find_run(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="run not found")
    payload = db.get_run_reports(run_id, row["ticker"], role or None)
    if payload is None:
        raise HTTPException(status_code=404, detail="run not found")
    return JSONResponse(payload)
