"""UI-driven forward-collection — now a Kafka pub-sub *producer* + run callback.

Previously this owned the whole batch: it created one asyncio task per ticker,
gated them behind a single ``asyncio.Semaphore``, and refused to accept new work
until the batch drained.  It now **publishes** each ticker to the Kafka topic
``workflow.requests`` and returns immediately.  A process-wide consumer pool
(:mod:`web.backend.workflow_consumer`, in the ``workflow-runners`` group) pulls
the next name the instant a slot frees and drives it here through
:func:`consume_workflow` — so:

  * you can submit more names at any time (each :func:`start_scale` just appends
    to the same live stream), and
  * a finished workflow immediately frees a consumer to pick up the next one —
    no waiting for a "batch" boundary.

What is reused vs. kept
-----------------------
* **Reused**: the full session pipeline via the ``SessionManager`` (streaming
  callbacks, M1-M7, critic, episode recording).  Batch sessions stay tagged
  ``episode_source="forward"`` for the clean RL cohort.
* **Kept**: the cross-process ``flock`` (single-flight so two dashboard
  collectors can't race the episode store), the per-day collect-state file for
  **resumability** (already-done names are skipped, never re-run), the pollable
  status snapshot, and the prediction-DB persistence (tile colours + history).

Concurrency
-----------
Concurrency = the number of consumers in the pool (``TRADINGCREW_WORKFLOW_CONSUMERS``);
the SessionManager's global cap (``TRADINGCREW_MAX_CONCURRENT_RUNS``) still
applies, so effective LLM concurrency is ``min(consumers, cap)``.  Launches are
still spaced by ``_LAUNCH_STAGGER_SEC`` to avoid a thundering herd.

Pause vs. stop: pause tells the consumers to stop pulling new names (in-flight
finish); stop also discards the queued backlog and cancels the in-flight runs.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# This file lives at ``<repo>/web/backend/scale_runner.py`` → repo root is parents[2].
_UNIVERSE_DIR = Path(__file__).resolve().parents[2] / "universe"

# Seconds between successive session launches (smooths the LLM-endpoint burst
# and, together with the per-search gap in web_research, spreads a big batch
# over hours).  Override with ``TRADINGCREW_SCALE_LAUNCH_STAGGER_SECONDS``.
try:
    _LAUNCH_STAGGER_SEC = float(
        os.environ.get("TRADINGCREW_SCALE_LAUNCH_STAGGER_SECONDS", "").strip() or 2.0
    )
except (TypeError, ValueError):
    _LAUNCH_STAGGER_SEC = 2.0

# Max age (hours) a queued workflow request may be consumed after it was
# published.  Guards against a durable-topic backlog being resurrected after a
# restart and silently attributed to a stale date (a batch for day D drained on
# day D+1).  Generous enough (24h) to tolerate a normal overnight overrun of the
# 8-hour window, but skips a request that's been sitting for a full day+.  A
# legacy message with no ``published_ts`` is treated as stale when its ``day`` is
# already in the past.  Set 0 to disable the guard.
try:
    _WORKFLOW_MAX_AGE_HOURS = float(
        os.environ.get("TRADINGCREW_WORKFLOW_MAX_AGE_HOURS", "").strip() or 24.0
    )
except (TypeError, ValueError):
    _WORKFLOW_MAX_AGE_HOURS = 24.0

# Bounded wait for a LOCAL OpenAI-compatible endpoint (the llama.cpp server on
# 8001) to become ready before publishing a batch.  A (re)start leaves the port
# unresponsive for minutes while the model loads + KV cache initialises;
# without this, a batch launched in that window fails every cell with
# "Failed to connect ..."/503 "Loading model", which downstream surfaces as a
# failed Pydantic conversion.  Remote/hosted endpoints are not gated (handled
# elsewhere).  Override with LLM_READY_TIMEOUT / LLM_READY_PROBE_INTERVAL.
try:
    _LLM_READY_TIMEOUT = int(os.environ.get("LLM_READY_TIMEOUT", "480"))
except (TypeError, ValueError):
    _LLM_READY_TIMEOUT = 480
try:
    _LLM_READY_PROBE_INTERVAL = int(
        os.environ.get("LLM_READY_PROBE_INTERVAL", "5")
    )
except (TypeError, ValueError):
    _LLM_READY_PROBE_INTERVAL = 5


def _is_local_llm(base_url: str) -> bool:
    from urllib.parse import urlparse

    host = (urlparse(base_url).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1")


def _llm_health_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return base[: -len("/v1")] + "/health"
    return base.rstrip("/") + "/health"


def _batch_llm_base_url(run_cfg: Optional[Dict[str, Any]]) -> Optional[str]:
    """The LLM endpoint the batch will actually use — its saved preset's
    base_url — so the readiness gate probes the RIGHT server.

    Returns ``None`` to mean "no pinned preset / closed-source preset /
    unknown preset": the gate then falls back to the env-chain default.
    """
    preset_id = (run_cfg or {}).get("llm_preset")
    if not preset_id:
        return None
    from trading_crew import llm_presets

    preset = llm_presets.resolve_preset(preset_id)
    if preset is None:
        logger.warning(
            "run_cfg pins unknown llm_preset %r — LLM readiness gate falls "
            "back to the env default endpoint", preset_id,
        )
        return None
    return preset.base_url


async def _wait_llm_ready(
    base_url: Optional[str] = None,
    timeout: Optional[int] = None,
) -> Optional[str]:
    """Wait for an OpenAI-compatible endpoint to be ready.

    ``base_url`` is the endpoint the specific run/batch will use (normally
    ``_batch_llm_base_url(run_cfg)``).  When omitted we fall back to the
    thread-local active preset, then the env chain — the historical
    behaviour.  Remote (or unresolvable) endpoints are never gated.
    Returns ``None`` once ready, else a human-readable reason string after
    ``timeout`` (default ``LLM_READY_TIMEOUT``).
    """
    import httpx

    if base_url is None:
        from trading_crew.llm_presets import get_active_overrides

        base_url = (
            get_active_overrides().get("base_url")
            or os.environ.get("LLAMACPP_LLM_BASE_URL")
            or "http://localhost:8001/v1"
        )
    if not base_url or not _is_local_llm(base_url):
        return None  # remote endpoint — don't gate the batch here.
    health = _llm_health_endpoint(base_url)
    timeout = timeout or _LLM_READY_TIMEOUT
    start = time.time()
    while True:
        try:
            r = await asyncio.to_thread(httpx.get, health, timeout=2.0)
            # vLLM's /health answers 200 with an EMPTY body while
            # llama-server answers 200 + {"status":"ok"} — treat any 2xx as
            # "up".  503 (vLLM "loading model") or a connection error means
            # not ready yet.
            if 200 <= r.status_code < 300:
                return None
        except Exception:  # noqa: BLE001 — transient / not-ready
            pass
        if time.time() - start >= timeout:
            return (
                f"LLM endpoint {health} not ready after {int(timeout)}s. "
                "Check the local LLM server is up and finished loading the "
                "model (its /health endpoint must return 2xx)."
            )
        await asyncio.sleep(_LLM_READY_PROBE_INTERVAL)


def _cache_dir() -> Path:
    return Path(os.environ.get("TRADINGCREW_CACHE_DIR") or os.path.expanduser("~/.trading_crew"))


def _lock_path() -> Path:
    return _cache_dir() / "forward" / "collect.lock"


# ---------------------------------------------------------------------------
# Universe helpers (shipped ticker files — used by the status payload)
# ---------------------------------------------------------------------------

def _read_universe_file(name_or_path: str) -> List[str]:
    """Resolve a universe by bare name (``all200``) or path and read its tickers."""
    from trading_crew.agentic.scale import _read_universe

    p = Path(name_or_path)
    candidates = [p]
    if not p.is_absolute():
        candidates += [_UNIVERSE_DIR / p.name, _UNIVERSE_DIR / f"{p.name}.txt"]
    for c in candidates:
        if c.exists():
            return _read_universe(str(c))
    raise FileNotFoundError(f"universe not found: {name_or_path}")


def available_universes() -> List[Dict[str, Any]]:
    """List the shipped universe files with ticker counts (for the UI select)."""
    from trading_crew.agentic.scale import _read_universe

    out: List[Dict[str, Any]] = []
    if _UNIVERSE_DIR.exists():
        for f in sorted(_UNIVERSE_DIR.glob("*.txt")):
            try:
                n = len(_read_universe(str(f)))
            except Exception:
                n = 0
            out.append({"name": f.stem, "file": f.name, "count": n})
    return out


# ---------------------------------------------------------------------------
# Stream state (a single live pub-sub "stream", extended by each start_scale)
# ---------------------------------------------------------------------------


@dataclass
class _ScaleJob:
    job_id: str
    config: Dict[str, Any]
    day: str
    # control lifecycle: running | paused | stopped
    control: str = "running"
    started_ts: float = field(default_factory=time.time)
    finished_ts: Optional[float] = None
    total: int = 0          # cumulative names published to this stream
    processed: int = 0
    done: int = 0
    failed: int = 0
    error: Optional[str] = None
    # Health URL the start request is waiting on (local LLM warm-up).  Set
    # while the readiness gate blocks, cleared once it passes/fails — the UI
    # shows a "waiting" banner from this so the dashboard isn't blind while
    # the start POST is still in flight.
    llm_wait: Optional[str] = None
    logs: List[Dict[str, Any]] = field(default_factory=list)
    # ticker -> live session run_id (only names currently mid-run)
    active_runs: Dict[str, str] = field(default_factory=dict)
    _lock_fd: Optional[int] = None

    def log(self, payload: Dict[str, Any]) -> None:
        self.logs.append({**payload, "ts": datetime.now(timezone.utc).isoformat()})
        if len(self.logs) > 600:
            self.logs = self.logs[-600:]

    def derived_status(self) -> str:
        """UI status derived from control + progress (pub-sub has no hard
        'batch complete' — a stream goes idle when it drains but can accept
        more work at any time).

        A batch is only "running" while work is actually in flight (a name
        mid-run) or still pending (published but not yet processed).  Once the
        stream drains it reads "completed" (something ran) or "idle" (nothing
        did) — never "running".  This must not depend on ``total`` alone:
        counters can be inconsistent (e.g. an old in-memory job with total=0
        but processed>0), and a job with no active runs and no backlog is not
        running regardless of the totals."""
        if self.control == "stopped":
            return "stopped"
        if self.control == "paused":
            return "paused"
        pending = self.total - self.processed
        if self.active_runs or pending > 0:
            return "running"
        if self.processed > 0 or self.total > 0:
            return "completed"
        return "idle"

    def snapshot(self) -> Dict[str, Any]:
        from . import kafka_bus
        from . import workflow_consumer

        status = self.derived_status()
        return {
            "job_id": self.job_id,
            "config": self.config,
            "status": status,
            "active": status == "running",
            "can_pause": status == "running",
            "can_resume": status == "paused",
            "can_kill": status in ("running", "paused"),
            "started_ts": self.started_ts,
            "finished_ts": self.finished_ts,
            "elapsed_sec": round((self.finished_ts or time.time()) - self.started_ts, 1),
            "day": self.day,
            "workers": workflow_consumer.concurrency(),
            "total": self.total,
            "processed": self.processed,
            "done": self.done,
            "failed": self.failed,
            "error": self.error,
            "llm_wait": self.llm_wait,
            "active_runs": dict(self.active_runs),
            # Pub-sub / visualization surface.
            "transport": "kafka",
            "kafka": {
                "bootstrap": kafka_bus.bootstrap_servers(),
                "requests_topic": kafka_bus.workflow_requests_topic(),
                "events_topic": kafka_bus.workflow_events_topic(),
                "group": kafka_bus.workflow_group(),
                "console_url": kafka_bus.console_url(),
                "consumers": workflow_consumer.consumer_count(),
                "concurrency": workflow_consumer.concurrency(),
                "consumers_paused": workflow_consumer.is_paused(),
            },
            "logs": self.logs[-300:],
        }


_LOCK = threading.RLock()
_JOB: Optional[_ScaleJob] = None

# Global launch pacing (shared by all consumers) — reserve a monotonic slot so
# N consumers freeing at once still start staggered.
_LAUNCH_LOCK = asyncio.Lock()
_next_launch: float = 0.0


# ---------------------------------------------------------------------------
# Cross-process file lock (single-flight across dashboard collectors)
# ---------------------------------------------------------------------------

def _acquire_flock() -> Optional[int]:
    """Non-blocking flock on the shared collect lock. Returns fd or None if busy."""
    import fcntl

    path = _lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return fd


def _release_flock(fd: Optional[int]) -> None:
    if fd is None:
        return
    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        os.close(fd)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Prediction persistence (tile colours + history)
# ---------------------------------------------------------------------------

def _render_report(payload: dict) -> str:
    """Render a compact markdown report from a completed cell's decision."""
    d = payload.get("decision") or {}
    ticker = payload.get("ticker", "")
    side = payload.get("side") or d.get("side") or "N/A"
    conv = payload.get("conviction")
    tw = payload.get("target_weight")
    er = payload.get("expected_return_pct")
    lines = [f"# {ticker} — forward prediction", "", f"- **Action**: {str(side).upper()}"]
    if conv is not None:
        lines.append(f"- **Conviction**: {conv:.2f}")
    if tw is not None:
        lines.append(f"- **Target weight**: {tw:+.2%}")
    if er is not None:
        lines.append(f"- **Expected return**: {er:+.2%} over {d.get('horizon_days', '?')} trading days")
    rationale = (d.get("rationale") or "").strip()
    if rationale:
        lines += ["", "## Rationale", "", rationale]
    tags = d.get("tags") or []
    if tags:
        lines += ["", "**Tags**: " + ", ".join(str(t) for t in tags)]
    return "\n".join(lines)


def _persist_prediction(job: "_ScaleJob", payload: dict) -> None:
    """Best-effort write of a completed/failed cell into the prediction DB."""
    from . import prediction_db as db

    evt = payload.get("event")
    ticker = payload.get("ticker")
    if not ticker:
        return
    try:
        if evt == "cell_done":
            d = payload.get("decision") or {}
            db.upsert_prediction(
                ticker=ticker,
                run_date=job.day,
                run_id=payload.get("run_id") or job.job_id,
                status="done",
                action=(payload.get("side") or d.get("side") or None),
                conviction=payload.get("conviction"),
                size=payload.get("target_weight"),
                expected_return=payload.get("expected_return_pct"),
                decision=d or None,
                report=_render_report(payload),
                signals=payload.get("report_signals") or None,
            )
        elif evt == "cell_failed":
            db.upsert_prediction(
                ticker=ticker,
                run_date=job.day,
                run_id=payload.get("run_id") or job.job_id,
                status="failed",
                error=payload.get("error"),
            )
        db.touch_run_day(job.day)
    except Exception:  # noqa: BLE001
        logger.exception("failed to persist prediction for %s", ticker)


def _batch_cfg(job: "_ScaleJob", ticker: str, run_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Session config for a batch name — lean (no memory) so the 200-name run
    stays inside the LLM budget, tagged ``forward`` for the RL cohort."""
    cfg = dict(run_cfg or job.config.get("run_cfg") or {})
    cfg.setdefault("asset_class", "stock")
    cfg.setdefault("run_mode", "autonomous")
    cfg.setdefault("debate_rounds", 1)
    cfg.setdefault("risk_rounds", 1)
    cfg.setdefault("memory", False)
    cfg.setdefault("tools_enabled", {})   # {} -> each agent uses DEFAULT_AGENT_TOOLS
    cfg.setdefault("book", "paper")
    cfg["episode_source"] = "forward"
    cfg["decision_ts"] = datetime.utcnow().replace(microsecond=0).isoformat()
    cfg["trade_date"] = job.day
    # All tickers in a batch share the stream's job_id as their trace id so
    # logs + web-search counters group per batch (not per ticker).
    cfg["trace_id"] = job.job_id
    return cfg


def _extract_report_signals(session: Any) -> Optional[Dict[str, Any]]:
    """Pull the structured report trend-signal card from a session's events."""
    for ev in reversed(session.events or []):
        if ev.get("type") == "report_signals" and ev.get("signals"):
            return {
                "signals": ev.get("signals") or [],
                "net_bias": ev.get("net_bias"),
                "net_confidence": ev.get("net_confidence"),
                "bias_score": ev.get("bias_score"),
            }
    return None


def _extract_decision(session: Any) -> Dict[str, Any]:
    """Pull the final ActionProposal (side/conviction/size/return + full dump)
    out of a finished session's event buffer."""
    proposal = None
    for ev in reversed(session.events or []):
        if ev.get("type") == "action_proposal" and ev.get("proposal"):
            proposal = ev["proposal"]
            break
    if not proposal:
        for ev in reversed(session.events or []):
            if ev.get("type") == "final_decision" and ev.get("decision"):
                dec = ev["decision"]
                return {
                    "side": dec.get("action"),
                    "conviction": dec.get("confidence"),
                    "target_weight": None,
                    "expected_return_pct": (dec.get("expected_return_pct") or 0) / 100.0
                    if dec.get("expected_return_pct") is not None else None,
                    "decision": dec,
                }
        return {"side": None, "conviction": None, "target_weight": None,
                "expected_return_pct": None, "decision": None}
    side = proposal.get("side")
    if isinstance(side, dict):
        side = side.get("value")
    return {
        "side": side,
        "conviction": proposal.get("conviction_score"),
        "target_weight": proposal.get("target_weight"),
        "expected_return_pct": proposal.get("expected_return_pct"),
        "decision": proposal,
    }


def _finish_cell(job, ticker, *, ok, t0, decision=None, error=None, run_id=None) -> None:
    """Update collect-state (resumability) + counters + DB + logs for one name."""
    from trading_crew.agentic.scale import _load_collect_state, _save_collect_state

    with _LOCK:
        state = _load_collect_state(job.day)
        if ok:
            state[ticker.upper()] = {
                "status": "done", "run_id": run_id,
                "elapsed_sec": round(time.time() - t0, 1),
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            job.done += 1
        else:
            state[ticker.upper()] = {
                "status": "failed", "error": error,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            job.failed += 1
        job.processed += 1
        _save_collect_state(job.day, state)

    if ok:
        payload = {"event": "cell_done", "ticker": ticker, **(decision or {}),
                   "elapsed_sec": round(time.time() - t0, 1), "run_id": run_id}
    else:
        payload = {"event": "cell_failed", "ticker": ticker, "error": error,
                   "run_id": run_id}
    _persist_prediction(job, payload)
    job.log({k: v for k, v in payload.items() if k != "decision"})


# ---------------------------------------------------------------------------
# Launch pacing
# ---------------------------------------------------------------------------

async def _throttle_launch() -> None:
    """Space session launches out by ``_LAUNCH_STAGGER_SEC`` across all
    consumers (reserve a monotonic slot under a lock, sleep outside it)."""
    global _next_launch
    if _LAUNCH_STAGGER_SEC <= 0:
        return
    async with _LAUNCH_LOCK:
        loop = asyncio.get_running_loop()
        now = loop.time()
        wait = max(0.0, _next_launch - now)
        _next_launch = max(now, _next_launch) + _LAUNCH_STAGGER_SEC
    if wait > 0:
        await asyncio.sleep(wait)


# ---------------------------------------------------------------------------
# Kafka event emission (visualization / observability on workflow.events)
# ---------------------------------------------------------------------------

async def _emit_event(payload: Dict[str, Any]) -> None:
    from . import kafka_bus

    if not kafka_bus.is_started():
        return
    try:
        await kafka_bus.publish(
            kafka_bus.workflow_events_topic(), payload, key=payload.get("ticker")
        )
    except Exception:  # noqa: BLE001
        logger.debug("emit workflow event failed", exc_info=True)


# ---------------------------------------------------------------------------
# Consumer callback — run ONE workflow request (invoked by workflow_consumer)
# ---------------------------------------------------------------------------

def _is_stale_request(payload: Dict[str, Any], day: str) -> bool:
    """A queued request is stale if it was published more than
    ``_WORKFLOW_MAX_AGE_HOURS`` ago.  Messages predating this field (no
    ``published_ts``) are stale only once their ``day`` is already in the past —
    so legacy backlog for a bygone date is dropped, but same-day work isn't."""
    pub = payload.get("published_ts")
    if isinstance(pub, (int, float)) and pub > 0:
        return (time.time() - float(pub)) > _WORKFLOW_MAX_AGE_HOURS * 3600.0
    from .prediction_db import prediction_day

    today = prediction_day()
    return bool(day) and day < today


def _ensure_stream(day: str, *, run_cfg: Optional[Dict[str, Any]] = None) -> "_ScaleJob":
    """Return the live stream for ``day``, reconstructing one if the process
    restarted with messages still queued (durable topic → work survives)."""
    global _JOB
    with _LOCK:
        if _JOB is None or _JOB.control == "stopped" or _JOB.day != day:
            fd = _acquire_flock()  # may be None if another collector holds it
            _JOB = _ScaleJob(
                job_id=f"scale-{day}-{int(time.time())}",
                config={"source": "recovered", "day": day, "run_cfg": run_cfg or {}},
                day=day,
                _lock_fd=fd,
            )
            _JOB.log({"event": "stream_recovered", "day": day})
        return _JOB


def _should_skip_name(day: str, ticker: str, *, force_rerun: bool) -> bool:
    """True when this name already has a ``done`` result for ``day``.

    Checks collect-state (bulk resumability) and the Prediction index
    (single completions). ``force_rerun`` disables both. A ``failed``
    row is not a skip.
    """
    if force_rerun:
        return False
    from trading_crew.agentic.scale import _load_collect_state
    from . import prediction_db as db

    key = (ticker or "").strip().upper()
    if not key:
        return False
    if _load_collect_state(day).get(key, {}).get("status") == "done":
        return True
    return db.done_prediction(key, day) is not None


async def consume_workflow(payload: Dict[str, Any]) -> None:
    """Run a single workflow request pulled from ``workflow.requests``.

    Idempotent w.r.t. the per-day collect-state and the Prediction index:
    a name already ``done`` is skipped (so re-delivery after a crash, a
    resume, or a same-day single run never re-runs it) unless the
    published message carries ``force_rerun``.
    """
    from .session_manager import get_session_manager
    from .prediction_db import prediction_day

    ticker = (payload.get("ticker") or "").strip()
    if not ticker:
        return
    day = payload.get("day") or prediction_day()
    run_cfg = payload.get("run_cfg") or None
    force_rerun = bool(payload.get("force_rerun"))

    # Stale-backlog guard: drop a request that missed its window (e.g. a
    # durable-topic backlog resurrected after a restart) so it isn't silently
    # attributed to a past date.  See _WORKFLOW_MAX_AGE_HOURS.
    if _WORKFLOW_MAX_AGE_HOURS > 0 and _is_stale_request(payload, day):
        logger.warning(
            "skipping stale workflow request ticker=%s day=%s (published %s) — "
            "older than %.0fh; not attributing to a past date",
            ticker, day, payload.get("published_ts"), _WORKFLOW_MAX_AGE_HOURS,
        )
        return

    job = _ensure_stream(day, run_cfg=run_cfg)

    # Resumability / dedup: skip names already collected for this day
    # (collect-state or a completed single prediction) unless forced.
    if _should_skip_name(day, ticker, force_rerun=force_rerun):
        with _LOCK:
            job.processed = min(job.total, job.processed + 1) if job.total else job.processed + 1
        job.log({"event": "cell_skipped", "ticker": ticker, "reason": "already done"})
        return

    await _throttle_launch()
    await _emit_event({"event": "cell_start", "ticker": ticker, "day": day, "job_id": job.job_id})

    t0 = time.time()
    job.log({"event": "cell_start", "ticker": ticker})
    sm = get_session_manager()
    try:
        session = await sm.start(ticker, _batch_cfg(job, ticker, run_cfg))
    except Exception as exc:  # noqa: BLE001
        _finish_cell(job, ticker, ok=False, error=str(exc), t0=t0)
        await _emit_event({"event": "cell_failed", "ticker": ticker, "error": str(exc), "job_id": job.job_id})
        return

    with _LOCK:
        job.active_runs[ticker.upper()] = session.run_id
    try:
        await session.done_event.wait()
    finally:
        with _LOCK:
            job.active_runs.pop(ticker.upper(), None)

    status = getattr(session.status, "value", str(session.status))
    decision = _extract_decision(session)
    decision["report_signals"] = _extract_report_signals(session)
    if status == "completed" and decision.get("side") is not None:
        _finish_cell(job, ticker, ok=True, decision=decision, t0=t0, run_id=session.run_id)
        await _emit_event({"event": "cell_done", "ticker": ticker, "side": decision.get("side"),
                           "job_id": job.job_id, "elapsed_sec": round(time.time() - t0, 1)})
    else:
        err = session.error or f"session ended '{status}' with no decision"
        _finish_cell(job, ticker, ok=False, error=err, t0=t0)
        await _emit_event({"event": "cell_failed", "ticker": ticker, "error": err, "job_id": job.job_id})


# ---------------------------------------------------------------------------
# Public API (async — the /api/scale/* endpoints await these directly)
# ---------------------------------------------------------------------------

async def start_scale(
    *,
    universe: Optional[str] = None,
    tickers: Optional[List[str]] = None,
    workers: int = 5,
    day: Optional[str] = None,
    run_cfg: Optional[Dict[str, Any]] = None,
    force_rerun: bool = False,
) -> Dict[str, Any]:
    """Publish a forward-collection stream to Kafka.

    Unlike the old batch runner this never rejects a concurrent call — each
    invocation appends its names to the same live stream and the consumer pool
    picks them up continuously.
    """
    global _JOB
    from . import kafka_bus
    from . import workflow_consumer
    from .prediction_db import prediction_day, start_run_day

    if not kafka_bus.is_started():
        raise ValueError(
            "Kafka bus is not running. Start the broker with "
            "`scripts/redpanda.sh start` and restart the web server."
        )

    if tickers:
        names = [t.strip() for t in tickers if t and t.strip()]
        source = "custom"
    elif universe:
        names = _read_universe_file(universe)
        source = universe
    else:
        raise ValueError("provide a universe name or a custom ticker list")
    if not names:
        raise ValueError("no tickers to collect")
    day = day or prediction_day()
    force_rerun = bool(force_rerun)

    # Set how many workflows run at once (UI worker count).
    if workers:
        workflow_consumer.set_concurrency(max(1, min(int(workers), 24)))
    workflow_consumer.resume()

    with _LOCK:
        if _JOB is None or _JOB.control == "stopped" or _JOB.day != day:
            fd = _acquire_flock()
            if fd is None and (_JOB is None or _JOB._lock_fd is None):
                raise ValueError(
                    "the collect lock is held by another process (another "
                    "dashboard collection). Wait for it to finish."
                )
            _JOB = _ScaleJob(
                job_id=f"scale-{day}-{int(time.time())}",
                config={"source": source, "day": day, "run_cfg": run_cfg or {}},
                day=day,
                _lock_fd=fd if fd is not None else (_JOB._lock_fd if _JOB else None),
            )
        else:
            _JOB.control = "running"
            _JOB.finished_ts = None
        job = _JOB

    # Only publish names not already collected for this day (resumability
    # + same-day single-run cache) unless the operator forced a rerun.
    if force_rerun:
        todo = list(names)
        already = 0
    else:
        todo = [t for t in names if not _should_skip_name(day, t, force_rerun=False)]
        already = len(names) - len(todo)
    start_run_day(day, total=len(names), source=source)

    # Gate the batch on the LOCAL LLM endpoint the batch will actually use
    # (its saved preset) — otherwise a server (re)start in progress makes
    # every cell fail with a connection / 503 "Loading model" error, which
    # downstream surfaces as a failed Pydantic conversion.  Remote presets
    # skip the gate entirely.  Wait (bounded), then fail cleanly if it never
    # comes up.
    gate_url = _batch_llm_base_url(run_cfg)
    if gate_url and _is_local_llm(gate_url):
        health = _llm_health_endpoint(gate_url)
        with _LOCK:
            job.llm_wait = health
        job.log({"event": "llm_wait", "health": health,
                 "preset": (run_cfg or {}).get("llm_preset") or "",
                 "timeout_sec": _LLM_READY_TIMEOUT})
    try:
        not_ready = await _wait_llm_ready(base_url=gate_url)
    finally:
        with _LOCK:
            job.llm_wait = None
    if not_ready:
        with _LOCK:
            job.error = not_ready
            job.log({"event": "llm_wait_failed", "error": not_ready})
            # Don't strand a zero-progress job as a zombie "running" stream —
            # the UI would show nothing and no way out.  A stream that already
            # has work in flight (reused by this append) is left untouched.
            fd: Optional[int] = None
            if not job.total and not job.active_runs and job.processed == 0:
                job.control = "stopped"
                job.finished_ts = time.time()
                fd = job._lock_fd
                job._lock_fd = None
        if fd is not None:
            _release_flock(fd)
        raise ValueError(not_ready)

    with _LOCK:
        job.total += len(todo)
        job.config["source"] = source
        job.log({"event": "publish", "source": source, "day": day,
                 "submitted": len(todo), "skipped_done": already})

    # Round-robin ticker -> partition so a burst of N names lands on N
    # partitions and genuinely runs min(N, concurrency) at once (rather than
    # piling onto one partition, which would serialise them).
    parts = kafka_bus.num_partitions()
    published_ts = time.time()
    for i, t in enumerate(todo):
        await kafka_bus.publish(
            kafka_bus.workflow_requests_topic(),
            {"ticker": t, "day": day, "run_cfg": run_cfg or {},
             "job_id": job.job_id, "published_ts": published_ts,
             "force_rerun": force_rerun},
            key=t,
            partition=i % parts,
        )

    logger.info("published %d workflow requests (day=%s, skipped %d already-done)",
                len(todo), day, already)
    return job.snapshot()


async def pause_scale() -> Dict[str, Any]:
    from . import workflow_consumer

    with _LOCK:
        job = _JOB
        if job is None or job.control == "stopped":
            return {"ok": False, "reason": "no active scale stream to pause"}
        job.control = "paused"
    workflow_consumer.pause()
    return {"ok": True, "job_id": job.job_id, "status": job.derived_status()}


async def resume_scale() -> Dict[str, Any]:
    from . import workflow_consumer

    with _LOCK:
        job = _JOB
        if job is None or job.control != "paused":
            return {"ok": False, "reason": "no paused scale stream to resume"}
        job.control = "running"
        job.finished_ts = None
    workflow_consumer.resume()
    return {"ok": True, "job_id": job.job_id, "status": job.derived_status()}


async def stop_scale() -> Dict[str, Any]:
    """Hard stop: discard the queued backlog + cancel in-flight sessions."""
    from .session_manager import get_session_manager
    from . import workflow_consumer

    with _LOCK:
        job = _JOB
        if job is None or job.control == "stopped":
            return {"ok": False, "reason": "no scale stream"}
        job.control = "stopped"
        run_ids = list(job.active_runs.values())
        fd = job._lock_fd
        job._lock_fd = None
        job.finished_ts = time.time()

    # Halt new starts + stop pulling.  Halt first so any name already fetched
    # and waiting on a slot is skipped (not started) once the slot frees.
    workflow_consumer.halt()
    workflow_consumer.pause()

    # Cancel in-flight sessions — this frees the slots those runs held, letting
    # the waiting consumer loops observe the halt and skip their fetched names.
    sm = get_session_manager()
    for rid in run_ids:
        try:
            await sm.cancel(rid)
        except Exception:  # noqa: BLE001
            logger.exception("failed to cancel session %s", rid)

    # Drop everything still queued for this stream.
    try:
        skipped = await workflow_consumer.discard_backlog()
        job.log({"event": "backlog_discarded", "skipped": skipped})
    except Exception:  # noqa: BLE001
        logger.exception("failed to discard workflow backlog")

    _release_flock(fd)
    return {"ok": True, "job_id": job.job_id, "status": "stopped"}


def get_scale_status() -> Dict[str, Any]:
    with _LOCK:
        job = _JOB
    base = {"universes": available_universes()}
    # Don't surface a *stale previous-day* job as the current "Today's forward
    # collection": once it has drained (not actively running/paused) and its day
    # is not today, report idle so the dashboard (batch panel, Pause button and
    # the search-backends panel keyed off job_id) starts each day clean.  A
    # batch still genuinely running across midnight is preserved.
    from .prediction_db import prediction_day

    _today = prediction_day()
    if job is not None and job.day != _today and job.derived_status() not in ("running", "paused"):
        job = None
    if job is None:
        from . import kafka_bus
        from . import workflow_consumer
        return {
            **base, "active": False, "status": "idle", "job_id": None,
            "logs": [], "active_runs": {}, "transport": "kafka",
            "kafka": {
                "bootstrap": kafka_bus.bootstrap_servers(),
                "requests_topic": kafka_bus.workflow_requests_topic(),
                "events_topic": kafka_bus.workflow_events_topic(),
                "group": kafka_bus.workflow_group(),
                "console_url": kafka_bus.console_url(),
                "consumers": workflow_consumer.consumer_count(),
                "concurrency": workflow_consumer.concurrency(),
                "consumers_paused": workflow_consumer.is_paused(),
            },
        }
    return {**base, **job.snapshot()}
