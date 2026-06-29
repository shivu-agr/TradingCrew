"""Run TradingCrew in a background thread and stream events.

Architecture
------------
* CrewAI's ``step_callback`` / ``task_callback`` fire on the kickoff thread.
* We push each callback into a thread-safe ``queue.Queue`` (``self._sync_q``).
* An asyncio pump task drains ``_sync_q`` (via ``asyncio.to_thread``) into
  ``_async_q`` so the WebSocket coroutine can stream events to the browser.

Streaming-gap fix
-----------------
With ``async_execution=True`` analyst tasks, all 8 analysts fire ``step_callback``
near-simultaneously and the UI lights them up. After they all complete, the next
task (Bullish Researcher round 1) is a *sequential, tool-less* task: its first
step_callback fires only when the LLM returns its full answer — typically 30–60 s
later. The UI sat completely idle in that gap and looked stuck.

We now precompute the expected agent ordering from ``crew.tasks``. Whenever a
``task_callback`` fires we increment a counter and look up the *next* expected
agent role in the list — if it has not been announced yet we synthesize a
``node_started`` event for it. That keeps the diagram "alive" even while a
sequential, tool-less agent is mid-LLM-call.
"""

import asyncio
import json
import logging
import queue
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from crewai.events import crewai_event_bus
from crewai.events.types.tool_usage_events import (
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)
from trading_crew import TradingCrew
from trading_crew._common import get_llm
from trading_crew._patches import DEGRADED_OUTPUT_MARKER
from trading_crew.critic import run_reflective_critic, records_to_payload
from trading_crew.tools import ALL_TOOLS, DEFAULT_AGENT_TOOLS

# Agentic-trading roadmap (M1-M7) — deterministic post-LLM pipeline.
from trading_crew.agentic.bridge import portfolio_decision_to_action_proposal
from trading_crew.agentic.execution.pipeline import run_pipeline
from trading_crew.agentic.memory import (
    EpisodicMemory,
    Episode,
    OutcomeStatus,
    Regime,
    detect_regime,
)
from trading_crew.agentic.runs import RunRecord, write_run

from .charts import _fetch_ohlcv  # reused so the pipeline sees the same OHLCV the chart UI does
from .events import NODE_KIND, make_event

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_text(value: Any, limit: int = 4000) -> str:
    """Coerce a CrewAI step / task object to a readable string."""
    if value is None:
        return ""
    if isinstance(value, str):
        out = value
    elif hasattr(value, "raw") and value.raw:
        out = value.raw
    elif hasattr(value, "output") and value.output is not None:
        out = str(value.output)
    elif hasattr(value, "text") and value.text:
        out = value.text
    elif hasattr(value, "content"):
        out = str(value.content)
    else:
        out = str(value)
    if len(out) > limit:
        return out[: limit - 20] + f"\n…[truncated {len(out) - limit} chars]"
    return out


def _agent_role_from_step(step: Any) -> Optional[str]:
    for attr in ("agent", "role", "agent_role"):
        v = getattr(step, attr, None)
        if isinstance(v, str):
            return v
        if v is not None and hasattr(v, "role"):
            return v.role
    return None


def _agent_role_from_task_output(task_output: Any) -> Optional[str]:
    """Walk the task_output object graph to find the agent's role string."""
    role = getattr(task_output, "agent", None)
    if role is None:
        t = getattr(task_output, "task", None)
        if t is not None:
            role = getattr(t, "agent", None)
    if role is None:
        return None
    if isinstance(role, str):
        return role
    return getattr(role, "role", None)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class AnalysisRunner:
    """Bridges synchronous CrewAI callbacks to an asyncio event consumer."""

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._async_q: asyncio.Queue = asyncio.Queue()
        self._sync_q: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._pump_task: Optional[asyncio.Task] = None
        self._cancel_event = threading.Event()
        # State driving the streaming-gap fix.
        self._expected_role_order: List[str] = []
        # Parallel array of ``async_execution`` flags per slot.  Async
        # batches dispatch in parallel, so when we announce the first
        # async slot we also pre-announce every contiguous async sibling
        # — otherwise an out-of-order completion inside the batch (e.g.
        # Conservative finishing before Aggressive) would fire a DONE
        # event with no matching prior START.
        self._expected_async_flags: List[bool] = []
        # ``async_execution=True`` siblings inside a batch complete
        # out-of-order, so we can't use a single completion counter as
        # a slot pointer.  ``_completed_slot_indices`` tracks WHICH task
        # slots have finished so the lookahead can advance to the next
        # not-yet-completed slot (instead of pointing back at an
        # already-finished sibling and emitting a "ghost" START).
        self._completed_slot_indices: set[int] = set()
        self._completed_count: int = 0
        # Roles that have an in-flight node_started but no matching
        # node_completed yet — used to dedupe step_callback announcements
        # within a single task. The same role can re-enter this set in
        # subsequent rounds (debate / risk).
        self._running_roles: set[str] = set()
        # Roles whose per-task output was substituted with the degraded
        # placeholder by ``trading_crew/_patches.py`` (OSS LLM emitted a
        # tool-call list as its final answer instead of synthesised
        # text).  Persisted into the saved run record so the operator
        # can see exactly which analysts came back empty after the run
        # is over.  Order-preserving via dict-keyset isn't worth it
        # here — the UI sorts by ``expected_role_order``.
        self._degraded_roles: set[str] = set()
        # Tool-call ledger so tool_result can be attributed back to its tool_call.
        self._tool_call_seq: int = 0
        self._open_tool_calls: Dict[str, Dict[str, Any]] = {}
        # Phase E — run persistence. Populated by _emit's snapshotter
        # and flushed to disk on run_completed / error.
        self._run_record: Optional[RunRecord] = None
        # Default to stock; _run_blocking updates this from ui_cfg so the
        # M2 pipeline picks asset-class-appropriate cost presets.
        self._asset_class: str = "stock"
        # Phase 2B.3 — paper/prod book partition.  Defaults to "paper" so
        # iterative experimentation can't pollute the audit trail of a
        # live ``prod`` book.  Set from ui_cfg via the sidebar selector.
        self._book: str = "paper"
        # Phase 2E — analyst cache metadata (filled at run start).
        self._cache_meta: Optional[Dict[str, Any]] = None

    @property
    def events(self) -> asyncio.Queue:
        return self._async_q

    def cancel(self) -> None:
        self._cancel_event.set()

    # ---- pump --------------------------------------------------------------

    async def start_pump(self) -> None:
        async def _pump() -> None:
            while True:
                try:
                    item = await asyncio.to_thread(self._sync_q.get, True, 0.25)
                except queue.Empty:
                    continue
                except Exception:
                    continue
                await self._async_q.put(item)
                if item.get("type") in ("run_completed", "error"):
                    return

        self._pump_task = asyncio.create_task(_pump())

    # ---- event helpers -----------------------------------------------------

    def _emit(self, event: Dict[str, Any]) -> None:
        self._sync_q.put(event)
        self._snapshot_event(event)

    def _snapshot_event(self, event: Dict[str, Any]) -> None:
        """Mirror UI events into the in-flight RunRecord so Phase E can
        persist the full audit trail on completion.

        Failures here are swallowed — persistence is a best-effort
        sidecar that must never break the main event stream.
        """
        rec = self._run_record
        if rec is None:
            return
        try:
            t = event.get("type")
            if t == "cascade_status":
                rec.cascade_status = {
                    "regime": event.get("regime"),
                    "route": event.get("route"),
                    "reason": event.get("reason"),
                }
                rec.cascade_route = event.get("route")
            elif t == "node_completed":
                role = event.get("node") or event.get("role")
                if role:
                    rec.reports[role] = (event.get("output") or "")[:8000]
                    if event.get("degraded"):
                        rec.degraded_roles.append(role)
            elif t == "final_decision":
                rec.final_decision = event.get("decision")
                rec.final_decision_source = event.get("source") or rec.final_decision_source or "crew"
            elif t == "reflection_records":
                rec.reflection_records = {
                    "records": event.get("records") or [],
                    "final_action": event.get("final_action"),
                    "final_size_pct": event.get("final_size_pct"),
                    "revised": event.get("revised"),
                }
            elif t == "action_proposal":
                rec.action_proposal = event.get("proposal")
                rec.action_proposal_markdown = event.get("markdown")
            elif t == "execution_result":
                rec.execution_result = {
                    k: v for k, v in event.items() if k not in ("type", "ts")
                }
            elif t == "episode_recorded":
                rec.episode_meta = {
                    "episode_id": event.get("episode_id"),
                    "regime": event.get("regime"),
                    "decision_ts": event.get("decision_ts"),
                    "outcome_ts": event.get("outcome_ts"),
                }
            elif t == "tool_call":
                rec.tool_calls += 1
            elif t == "run_completed":
                rec.status = "completed"
                rec.completed_at = datetime.now(timezone.utc).isoformat()
            elif t == "error":
                rec.status = "error"
                rec.error = event.get("message")
                rec.completed_at = datetime.now(timezone.utc).isoformat()
        except Exception:
            logger.exception("Run snapshotter failed (non-fatal)")

    def _announce(self, role: str) -> None:
        """Send a ``node_started`` event for ``role`` if it isn't already
        marked as running. Re-entry after node_completed is allowed so
        debate-round repeats (Bull/Bear, Aggressive/Neutral/Conservative)
        can flip the UI back to running.
        """
        if not role or role in self._running_roles:
            return
        self._running_roles.add(role)
        self._emit(make_event(
            "node_started",
            node=role, role=role, kind=NODE_KIND.get(role, "node"),
        ))

    # ---- callbacks (run on CrewAI thread) ----------------------------------

    def _on_step(self, step: Any) -> None:
        if self._cancel_event.is_set():
            return
        # CrewAI's step_callback receives an AgentAction (post-tool, with .result)
        # or an AgentFinish dataclass. Neither has an `agent` attribute, so
        # ``_agent_role_from_step`` will return None — and step_callback only
        # ever fires once per task with the AgentFinish, since the local LLM
        # uses native tool calling. We therefore skip step events when the
        # role can't be attributed (the per-tool events come from the
        # CrewAI event bus instead, see _on_tool_started / _on_tool_finished).
        role = _agent_role_from_step(step)
        if not role:
            return
        self._announce(role)
        text = _safe_text(step, limit=2000)
        if text:
            self._emit(make_event(
                "agent_step",
                node=role, role=role, kind=NODE_KIND.get(role, "node"),
                content=text,
            ))

    # ---- tool events from the CrewAI event bus -----------------------------

    def _on_tool_started(self, _source: Any, event: ToolUsageStartedEvent) -> None:
        if self._cancel_event.is_set():
            return
        role = event.agent_role or "Agent"
        self._announce(role)
        self._tool_call_seq += 1
        call_id = f"{role}-{event.tool_name}-{self._tool_call_seq}"
        self._open_tool_calls[call_id] = {
            "tool": event.tool_name, "node": role, "started_at": time.time(),
        }
        # Map this tool's started_at -> call_id so finished/error can match.
        # CrewAI's ToolUsageStartedEvent doesn't carry a call_id of its own;
        # we collapse on (role, tool_name) for the most-recent open call.
        args = event.tool_args
        if isinstance(args, str):
            args_text = args[:600]
        else:
            try:
                args_text = json.dumps(args, default=str)[:600]
            except Exception:
                args_text = str(args)[:600]
        self._emit(make_event(
            "tool_call",
            node=role, tool=event.tool_name,
            call_id=call_id,
            args=args_text,
        ))

    def _on_tool_finished(self, _source: Any, event: ToolUsageFinishedEvent) -> None:
        if self._cancel_event.is_set():
            return
        role = event.agent_role or "Agent"
        # Match the most recent open call from this (role, tool).
        call_id = self._pop_matching_call(role, event.tool_name)
        elapsed_ms = 0
        try:
            if event.started_at and event.finished_at:
                elapsed_ms = int((event.finished_at - event.started_at).total_seconds() * 1000)
        except Exception:
            pass
        self._emit(make_event(
            "tool_result",
            node=role, tool=event.tool_name, call_id=call_id,
            output=_safe_text(event.output, limit=4000),
            elapsed_ms=elapsed_ms,
        ))

    def _on_tool_error(self, _source: Any, event: ToolUsageErrorEvent) -> None:
        if self._cancel_event.is_set():
            return
        role = event.agent_role or "Agent"
        call_id = self._pop_matching_call(role, event.tool_name)
        self._emit(make_event(
            "tool_result",
            node=role, tool=event.tool_name, call_id=call_id,
            error=str(event.error),
            output=str(event.error)[:1000],
            elapsed_ms=0,
        ))

    def _pop_matching_call(self, role: str, tool_name: str) -> Optional[str]:
        for cid in list(self._open_tool_calls.keys())[::-1]:
            meta = self._open_tool_calls[cid]
            if meta["node"] == role and meta["tool"] == tool_name:
                del self._open_tool_calls[cid]
                return cid
        return None

    _ANALYST_ROLE_TO_TASK_ID = {
        "Market Analyst": "market_task",
        "Social Analyst": "social_task",
        "News Analyst": "news_task",
        "Fundamentals Analyst": "fundamentals_task",
        "Macro Analyst": "macro_task",
        "Geopolitical Analyst": "geopolitical_task",
        "Sector Analyst": "sector_task",
        "Quant Analyst": "quant_task",
    }

    def _on_task(self, task_output: Any) -> None:
        if self._cancel_event.is_set():
            return
        role = _agent_role_from_task_output(task_output) or "Agent"
        self._completed_count += 1
        # Map this completion back to a SLOT index — the lookahead below
        # needs to know which expected slot just finished so it can skip
        # past it.  Within an async batch each role appears at most once,
        # and across the whole task list repeated roles (Bullish / Bearish
        # debate rounds, risk personas) get assigned to successive
        # un-completed slots — so "first un-completed slot whose role
        # matches" is unambiguous.
        for i, expected in enumerate(self._expected_role_order):
            if i in self._completed_slot_indices:
                continue
            if expected == role:
                self._completed_slot_indices.add(i)
                break
        # Allow the agent to be re-announced if it runs again later (debate
        # rounds re-use the same agent). We only remove from the running set
        # after we've emitted node_completed.
        self._running_roles.discard(role)
        raw_text = _safe_text(task_output, limit=8000)
        # When the OSS-LLM tool-call-as-final-answer patch fires (see
        # ``trading_crew/_patches.py``) it prefixes the per-task output
        # with ``DEGRADED_OUTPUT_MARKER``.  Detect it so the UI can flag
        # the affected report as degraded instead of rendering an empty
        # / nonsense body, and so the saved run record persists the
        # signal for audit.
        degraded = raw_text.startswith(DEGRADED_OUTPUT_MARKER)
        if degraded:
            self._degraded_roles.add(role)
        self._emit(make_event(
            "node_completed",
            node=role, role=role, kind=NODE_KIND.get(role, "node"),
            output=raw_text,
            degraded=degraded,
        ))

        # Phase 2E — populate the analyst output cache.  We only cache
        # the analyst phase because (debate, risk, PM) outputs depend on
        # the *interaction* between agents and on the current portfolio
        # state, neither of which is captured by the cache key.
        task_id = self._ANALYST_ROLE_TO_TASK_ID.get(role)
        if task_id and getattr(self, "_cache_meta", None) and raw_text:
            try:
                from .analyst_cache import make_cache_key, save_entry
                key = make_cache_key(
                    ticker=self._cache_meta["ticker"],
                    trade_date=self._cache_meta["trade_date"],
                    tools_enabled=self._cache_meta["tools_enabled"],
                    task_id=task_id,
                )
                save_entry(key, agent_role=role, raw=raw_text)
            except Exception:  # never let cache writes break the run
                logger.exception("Failed to persist analyst cache for %s", role)

        # ---- streaming-gap fix ------------------------------------------------
        # Announce the next not-yet-completed slot so the diagram never sits
        # idle while CrewAI is mid-LLM-call on a tool-less task.
        #
        # Three regimes are handled here:
        #
        # 1. **Sequential tool-less task** (Bullish / Bearish debate,
        #    Research Manager, Quality Reviewer, Compliance, PM) — the
        #    next slot is sequential, the role isn't running, so
        #    ``_announce`` emits the START 30-60 s before step_callback
        #    would otherwise fire it at the end of the LLM call.
        # 2. **Async analyst batch already in flight** — scanning from
        #    slot 0 lands on a slot that the tool-use bus already
        #    announced; ``_announce`` is a no-op via ``_running_roles``,
        #    and no "ghost" START gets emitted for an already-finished
        #    sibling (the Macro / Sector / Quant bug).
        # 3. **Entering an async batch** (e.g. Trader → 3 risk analysts)
        #    — the first un-completed slot is async, so we pre-announce
        #    the ENTIRE contiguous async batch.  Risk tasks are tool-less
        #    so the tool-use bus never fires for them; without this
        #    fan-out, an out-of-order completion (Conservative finishing
        #    before Aggressive) would emit a DONE event with no matching
        #    prior START.
        for next_idx in range(len(self._expected_role_order)):
            if next_idx in self._completed_slot_indices:
                continue
            self._announce(self._expected_role_order[next_idx])
            if (
                next_idx < len(self._expected_async_flags)
                and self._expected_async_flags[next_idx]
            ):
                j = next_idx + 1
                while (
                    j < len(self._expected_role_order)
                    and j < len(self._expected_async_flags)
                    and j not in self._completed_slot_indices
                    and self._expected_async_flags[j]
                ):
                    self._announce(self._expected_role_order[j])
                    j += 1
            break

    # ---- main blocking work ------------------------------------------------

    # ---- M4 Cascaded Controller ----------------------------------------
    def _run_cascade_controller(self, ticker: str, decision_ts: str) -> Tuple[Optional[str], str]:
        """Detect regime *before* the crew kicks off.

        Returns ``(regime, route)`` so the caller can decide whether to
        proceed with the full debate, run with reduced budget, or short-
        circuit entirely on CRISIS. Emits ``cascade_status`` so the UI
        lights up the new node even before the analysts begin.
        """
        regime = Regime.UNKNOWN.value
        reason = "OHLCV unavailable"
        route = "FULL_DEBATE"
        try:
            df = _fetch_ohlcv(ticker, datetime.fromisoformat(decision_ts.replace("Z", "+00:00")).replace(tzinfo=None), 252)
            if df is not None and not df.empty:
                closes = df["Close"].astype(float).tolist()
                r = detect_regime(closes)
                regime = r.value
                if r == Regime.CRISIS:
                    route = "CRISIS_OVERRIDE"
                    reason = (
                        "Crisis regime (paper §5.3) — skipping the 18-agent "
                        "debate and routing directly to an ABSTAIN proposal. "
                        "M2 + M5 + M3 still run on the override so the "
                        "risk gates can decide whether to flatten exposure."
                    )
                elif r == Regime.HIGH_VOL_RANGE:
                    # Phase 2E — choppy high-vol with no trend.  Route
                    # to a risk-only mini-crew (the analyst fan-out
                    # rarely uncovers actionable signal in this regime).
                    route = "RISK_ONLY"
                    reason = (
                        "High-vol RANGE regime — chopping with no direction. "
                        "Skipping the analyst fan-out and routing to a "
                        "risk-only mini-crew (Phase 2E)."
                    )
                elif r in (Regime.HIGH_VOL, Regime.HIGH_VOL_TREND):
                    route = "RISK_HEAVY"
                    reason = "Elevated volatility — risk team gets priority."
                else:
                    reason = f"Regime={r.value}; running full debate."
        except Exception as exc:
            logger.warning("Cascade controller failed: %s", exc)
            reason = f"Detector error: {exc}"
        self._emit(make_event("cascade_status", regime=regime, route=route, reason=reason))
        return regime, route

    @staticmethod
    def _crisis_override_decision(ticker: str) -> Any:
        """Build a synthetic ABSTAIN ``PortfolioDecision`` for CRISIS runs.

        This skips the 18-agent debate but still goes through the
        deterministic M1->M2->M5->M3 pipeline so the audit trail and
        episodic memory record *why* we didn't trade.
        """
        from trading_crew.schemas import PortfolioDecision
        return PortfolioDecision(
            action="NEUTRAL",
            confidence=0.0,
            size_pct_of_book=0.0,
            entry_price=0.0,
            stop_loss=0.0,
            target_price=0.0,
            horizon_days=1,
            expected_return_pct=0.0,
            rationale=(
                f"[CASCADE OVERRIDE] Market regime classified CRISIS for {ticker.upper()}. "
                "Reflective debate skipped (paper §5.3) to prevent the LLM ensemble "
                "from rationalising into a contrarian trade against broken microstructure "
                "(widening spreads, gappy fills, correlated risk-off). The deterministic "
                "M5 risk gate will separately decide whether to flatten existing exposure."
            ),
            key_drivers=[],
            key_risks=["crisis regime", "elevated volatility", "widening spreads"],
            falsifiers=["regime detector returns to TREND or RANGE"],
            geopolitical_flags=["cascade_override"],
            compliance_status="FLAGGED",
        )

    # ---- M2 + M5 Execution pipeline ------------------------------------
    def _run_execution_pipeline(
        self,
        ticker: str,
        proposal,
        ohlcv,
    ) -> Optional[Dict[str, Any]]:
        """Run ActionProposal -> Sizer -> RiskGate -> Simulator after the crew.

        Emits the ``action_proposal`` and ``execution_result`` events.
        Returns the result envelope as a dict so the caller can decide
        whether to record an outcome to episodic memory.
        """
        try:
            self._emit(make_event(
                "action_proposal",
                proposal=proposal.model_dump(),
                # The Pydantic helper is named ``render_markdown`` —
                # ``to_markdown`` was a stale call that raised
                # ``AttributeError``, was swallowed by this try/except, and
                # ended up shipping an ``action_proposal`` event with no
                # markdown body to the UI.  See
                # ``trading_crew/agentic/execution/contracts.py``.
                markdown=proposal.render_markdown(),
            ))
        except Exception:
            logger.exception("Failed to emit action_proposal event")

        # For commodity runs, pick the futures cost-model presets so the M2
        # sweep ("low/standard/high") shows realistic exchange-fee and
        # tighter half-spreads instead of equity-style assumptions.
        cost_model_name = (
            "futures_standard" if self._asset_class == "commodity" else "standard"
        )
        try:
            result = run_pipeline(
                proposal,
                ohlcv=ohlcv,
                portfolio_id=self._book,
                cost_model_name=cost_model_name,
                persist=True,
            )
        except Exception as exc:
            logger.exception("Execution pipeline failed")
            self._emit(make_event(
                "execution_result",
                rejected=True,
                note=f"Pipeline crash: {exc}",
            ))
            return None

        # Cost scenarios carry a nested CostModel dataclass — flatten so
        # the WebSocket JSON serializer doesn't choke on the dataclass field.
        cost_scenarios_payload = []
        for s in (result.cost_scenarios or []):
            try:
                model_dict = s.model.__dict__ if hasattr(s.model, "__dict__") else {}
            except Exception:
                model_dict = {}
            cost_scenarios_payload.append({
                "label": getattr(s, "label", "?"),
                "notional": getattr(s, "notional", 0.0),
                "participation": getattr(s, "participation", 0.0),
                "cost_breakdown": getattr(s, "cost_breakdown", {}) or {},
                "model": {
                    k: v for k, v in model_dict.items()
                    if isinstance(v, (int, float, str, bool, type(None)))
                },
            })

        payload: Dict[str, Any] = {
            "rejected": result.rejected,
            "note": result.note,
            "state_after": result.state_after,
            "cost_scenarios": cost_scenarios_payload,
        }
        if result.order is not None:
            payload["order"] = {
                "symbol": result.order.symbol,
                "qty": result.order.qty,
                "side": result.order.side.value,
                "order_type": result.order.order_type.value,
                "limit_price": result.order.limit_price,
            }
        if result.fill is not None:
            payload["fill"] = {
                "status": result.fill.status.value,
                "qty_filled": result.fill.qty_filled,
                "avg_price": result.fill.avg_price,
                "cost_breakdown": result.fill.cost_breakdown,
                "slippage_bps": result.fill.slippage_bps,
            }
        if result.sizing is not None:
            payload["sizing"] = {
                "final_weight": result.sizing.final_weight,
                "binding_constraint": result.sizing.binding_constraint,
                "kelly_cap": result.sizing.kelly_cap,
                "vol_cap": result.sizing.vol_cap,
                "cvar_cap": result.sizing.cvar_cap,
                "hard_cap": result.sizing.hard_cap,
                "risk_mult": result.sizing.risk_mult,
                "notes": result.sizing.notes,
            }
        if result.risk_gate is not None:
            payload["risk_gate"] = {
                "passed": result.risk_gate.passed,
                "failures": list(result.risk_gate.failures),
                "kill_switch_triggered": result.risk_gate.kill_switch_triggered,
            }
        self._emit(make_event("execution_result", **payload))
        return payload

    # ---- M3 Episode recording ------------------------------------------
    def _record_episode(self, ticker: str, proposal, decision_ts: str, regime: Optional[str]) -> None:
        """Append a PENDING episode for outcome resolution later.

        The walk-forward backtester or a scheduled outcome job will fill
        in ``outcome`` + flip status to RESOLVED once ``horizon_days``
        have elapsed.
        """
        try:
            import os
            cache_dir = os.environ.get("TRADINGCREW_CACHE_DIR") or os.path.expanduser("~/.trading_crew")
            store_path = Path(cache_dir) / "memory" / "episodes.jsonl"
            mem = EpisodicMemory(store_path)
            # outcome_ts is mandatory in the Episode schema even when the
            # outcome hasn't materialised yet — we set it to decision_ts +
            # horizon_days so the embargo gate is correctly delayed.
            try:
                d = datetime.fromisoformat(decision_ts.replace("Z", "+00:00"))
            except Exception:
                d = datetime.now(timezone.utc)
            outcome_dt = d + timedelta(days=max(proposal.horizon_days, 1))
            ep = Episode(
                episode_id=f"{ticker}-{decision_ts}",
                symbol=ticker.upper(),
                decision_ts=decision_ts,
                state_summary=proposal.rationale[:1000],
                regime=Regime(regime) if regime else Regime.UNKNOWN,
                action_proposal=proposal.model_dump(),
                outcome_ts=outcome_dt.isoformat(),
                outcome_status=OutcomeStatus.PENDING,
                realised_return=None,
                alpha_return=None,
                reflection=None,
                embargo_days=max(proposal.horizon_days, 1),
            )
            mem.add(ep)
            self._emit(make_event(
                "episode_recorded",
                episode_id=ep.episode_id,
                regime=ep.regime.value,
                decision_ts=ep.decision_ts,
                outcome_ts=ep.outcome_ts,
            ))
        except Exception:
            logger.exception("Could not record episode")

    def _run_blocking(self, ticker: str, ui_cfg: Dict[str, Any]) -> None:
        # Resolve the user-typed ticker (e.g. "MAZDOCK") to the canonical
        # yfinance symbol ("MAZDOCK.NS") so every downstream tool, chart,
        # and episode write uses the same symbol. Commodity tickers
        # (``CL=F``) and already-suffixed inputs pass through unchanged.
        asset_class_pre = (ui_cfg.get("asset_class") or "stock").lower()
        if asset_class_pre != "commodity":
            try:
                from trading_crew.market_context import resolve_ticker
                canonical = resolve_ticker(ticker)
                if canonical and canonical != ticker:
                    logger.info("Resolved ticker %s -> %s", ticker, canonical)
                    ticker = canonical
            except Exception:
                logger.exception("Ticker resolution failed; using raw input")

        # Phase 2F — apply the UI-selected LLM preset for the duration
        # of this run.  The preset is a thread-local override that
        # ``get_llm()`` consults BEFORE the env-var chain, so concurrent
        # WS sessions can each pick a different LLM without touching
        # ``os.environ``.  Cleared in the matching ``finally`` block.
        try:
            from trading_crew import llm_presets as _llm_presets
            llm_preset_id = ui_cfg.get("llm_preset")
            applied = _llm_presets.set_active(llm_preset_id) if llm_preset_id else None
            if applied is not None:
                logger.info(
                    "LLM preset for this run: %s (%s · %s)",
                    applied.id, applied.label, applied.model,
                )
        except Exception:
            logger.exception("Failed to apply UI LLM preset; falling back to .env defaults")

        # Same pattern for the embedding preset — used by
        # ``TruncatingOpenAIEmbedder`` and ``get_embedder_config()``.
        # The crew partitions its LanceDB store by preset id so vector
        # dim changes don't crash subsequent runs.
        try:
            from trading_crew import embedding_presets as _embedding_presets
            embedding_preset_id = ui_cfg.get("embedding_preset")
            applied_emb = (
                _embedding_presets.set_active(embedding_preset_id)
                if embedding_preset_id
                else None
            )
            if applied_emb is not None:
                logger.info(
                    "Embedding preset for this run: %s (%s · %s)",
                    applied_emb.id, applied_emb.label,
                    applied_emb.resolve_model() or "(env-driven)",
                )
        except Exception:
            logger.exception(
                "Failed to apply UI embedding preset; falling back to .env defaults"
            )

        # Phase E — open a RunRecord that the _emit snapshotter populates.
        run_started_at = datetime.now(timezone.utc).isoformat()
        run_id = run_started_at.replace(":", "-").replace("+00:00", "Z")
        self._run_record = RunRecord(
            run_id=run_id,
            ticker=ticker.upper(),
            started_at=run_started_at,
            config={
                "debate_rounds": ui_cfg.get("debate_rounds"),
                "risk_rounds": ui_cfg.get("risk_rounds"),
                "memory": ui_cfg.get("memory"),
                "tools_enabled": ui_cfg.get("tools_enabled"),
                "critic_iterations": ui_cfg.get("critic_iterations"),
                "critic_samples": ui_cfg.get("critic_samples"),
                "llm_preset": ui_cfg.get("llm_preset"),
                "embedding_preset": ui_cfg.get("embedding_preset"),
            },
        )
        try:
            debate_rounds = int(ui_cfg.get("debate_rounds", 2))
            risk_rounds = int(ui_cfg.get("risk_rounds", 1))
            use_memory = bool(ui_cfg.get("memory", True))
            tools_enabled = ui_cfg.get("tools_enabled") or {}
            asset_class = (ui_cfg.get("asset_class") or "stock").lower()
            self._asset_class = asset_class
            book = (ui_cfg.get("book") or "paper").strip().lower()
            if book not in ("paper", "prod"):
                book = "paper"
            self._book = book

            # Phase 2E — record metadata used by the analyst cache.
            trade_date_meta = (ui_cfg.get("trade_date") or ui_cfg.get("decision_ts") or datetime.utcnow().strftime("%Y-%m-%d"))[:10]
            self._cache_meta = {
                "ticker": ticker.upper(),
                "trade_date": trade_date_meta,
                "tools_enabled": tools_enabled,
            }

            # Route to the asset-class-appropriate crew. The downstream M1-M7
            # pipeline is identical for both (the deterministic layer is
            # asset-agnostic by design) — only the LLM debate differs.
            if asset_class == "commodity":
                from commodity_crew import CommodityCrew
                tc = CommodityCrew(
                    ticker=ticker,
                    debate_rounds=debate_rounds,
                    step_callback=self._on_step,
                    task_callback=self._on_task,
                    memory=use_memory,
                    tools_enabled=tools_enabled,
                )
            else:
                tc = TradingCrew(
                    ticker=ticker,
                    debate_rounds=debate_rounds,
                    risk_rounds=risk_rounds,
                    step_callback=self._on_step,
                    task_callback=self._on_task,
                    memory=use_memory,
                    tools_enabled=tools_enabled,
                )
            crew = tc.crew()
            if self._run_record is not None:
                self._run_record.config["asset_class"] = asset_class

            # Precompute expected role ordering + async flags so we can
            # synthesize node_started events while sequential agents are
            # mid-LLM-call AND pre-announce contiguous async batches
            # (so out-of-order completions inside a batch never fire a
            # DONE without a prior START).
            self._expected_role_order = [
                getattr(t.agent, "role", "Agent") for t in crew.tasks
            ]
            self._expected_async_flags = [
                bool(getattr(t, "async_execution", False)) for t in crew.tasks
            ]
            # Reset per-run slot tracking (in case the runner instance is
            # reused across analyses on the same WS session).
            self._completed_slot_indices = set()
            self._completed_count = 0
            logger.info("Expected role order: %s", self._expected_role_order)
            if self._run_record is not None:
                self._run_record.expected_role_order = list(self._expected_role_order)

            self._emit(make_event(
                "run_started",
                ticker=ticker,
                debate_rounds=debate_rounds,
                risk_rounds=risk_rounds,
                agent_count=len(crew.agents),
                task_count=len(crew.tasks),
                memory=use_memory,
                expected_order=self._expected_role_order,
                tools_enabled=tools_enabled,
            ))

            # M4: Cascaded Controller — detect regime before debate fires.
            # decision_ts is intentionally tz-naive (UTC implicit) so the
            # M2 pipeline can compare it directly against tz-naive OHLCV
            # dates without a tz coercion pass.
            decision_ts = (
                ui_cfg.get("decision_ts")
                or datetime.utcnow().replace(microsecond=0).isoformat()
            )
            regime_label, cascade_route = self._run_cascade_controller(ticker, decision_ts)

            # CRISIS short-circuit: skip the 18-agent debate entirely and
            # build a synthetic ABSTAIN decision. The rest of the M1-M5
            # pipeline (sizer, risk gate, simulator, episodic memory)
            # still runs so the audit trail is identical to a normal run.
            short_circuited = cascade_route == "CRISIS_OVERRIDE"
            result = None
            if short_circuited:
                # Mark every agent as "skipped" so the diagram doesn't
                # appear stuck. We send node_completed without a matching
                # node_started — the frontend tolerates this.
                for role in self._expected_role_order:
                    self._emit(make_event(
                        "node_completed",
                        node=role, role=role, kind=NODE_KIND.get(role, "node"),
                        output="(skipped — cascade controller routed around debate)",
                        skipped=True,
                    ))
                override = self._crisis_override_decision(ticker)
                self._emit(make_event(
                    "final_decision",
                    decision=override.model_dump(),
                    raw=override.rationale,
                    source="cascade_override",
                ))

                class _Result:
                    pydantic = override
                    raw = override.rationale
                result = _Result()
            else:
                # Announce the first task's agent right away — keeps the UI alive
                # even before CrewAI's first step_callback returns.
                if self._expected_role_order:
                    self._announce(self._expected_role_order[0])

                # Subscribe to the CrewAI event bus for per-tool events. We use
                # ``register_handler`` (not the @on decorator) so we can unregister
                # cleanly between runs.
                crewai_event_bus.register_handler(ToolUsageStartedEvent, self._on_tool_started)
                crewai_event_bus.register_handler(ToolUsageFinishedEvent, self._on_tool_finished)
                crewai_event_bus.register_handler(ToolUsageErrorEvent, self._on_tool_error)

                try:
                    result = crew.kickoff(inputs={"ticker": ticker})
                finally:
                    # Best-effort detach so handlers don't leak across runs in the
                    # same process (FastAPI keeps the bus singleton alive).
                    self._detach_event_handlers()

            decision_payload: Optional[Dict[str, Any]] = None
            if result is not None and result.pydantic is not None:
                decision_payload = result.pydantic.model_dump()
            if not short_circuited:
                # On short-circuit we already emitted final_decision with
                # source="cascade_override"; don't re-emit here.
                self._emit(make_event(
                    "final_decision",
                    decision=decision_payload,
                    raw=_safe_text(result, limit=8000),
                ))

            # M1 + M2 + M5 + M3: deterministic continuation of the LLM loop.
            if result is not None and result.pydantic is not None:
                try:
                    # M4: Reflective Critic — 5-stage protocol + 3-temp
                    # consistency vote. Runs *before* the bridge so the
                    # ActionProposal is built from the post-critique
                    # PortfolioDecision (which may have been revised
                    # downwards or collapsed to ABSTAIN). Skipped on
                    # cascade override: no debate happened so there's
                    # nothing to critique.
                    pm_decision = result.pydantic
                    # If this was a commodity run, the trader emitted a
                    # FuturesDecision — adapt it to PortfolioDecision so
                    # the critic + M1 bridge work unchanged.
                    if asset_class == "commodity":
                        try:
                            from commodity_crew.schemas import FuturesDecision
                            from commodity_crew.bridge import futures_decision_to_portfolio_decision
                            if isinstance(pm_decision, FuturesDecision):
                                pm_decision = futures_decision_to_portfolio_decision(pm_decision)
                        except Exception:
                            logger.exception("FuturesDecision -> PortfolioDecision adapter failed; using raw")
                    try:
                        if short_circuited:
                            raise RuntimeError("skip critic on cascade override")
                        critic_llm = get_llm(temperature=0.0)
                        critiqued, reflection_records = run_reflective_critic(
                            pm_decision, ticker=ticker, llm=critic_llm,
                            max_iterations=int(ui_cfg.get("critic_iterations", 2)),
                            consistency_samples=int(ui_cfg.get("critic_samples", 3)),
                        )
                        if reflection_records:
                            self._emit(make_event(
                                "reflection_records",
                                records=records_to_payload(reflection_records),
                                final_action=critiqued.action,
                                final_size_pct=critiqued.size_pct_of_book,
                                revised=(
                                    critiqued.action != pm_decision.action
                                    or critiqued.size_pct_of_book != pm_decision.size_pct_of_book
                                ),
                            ))
                        # Re-emit a final_decision so the UI re-renders the
                        # report card with the critiqued version.
                        if (
                            critiqued.action != pm_decision.action
                            or critiqued.size_pct_of_book != pm_decision.size_pct_of_book
                        ):
                            self._emit(make_event(
                                "final_decision",
                                decision=critiqued.model_dump(),
                                raw=critiqued.rationale,
                                source="critic",
                            ))
                        pm_decision = critiqued
                    except Exception as critic_exc:
                        if not short_circuited:
                            logger.exception("Reflective critic failed; using PM decision unchanged")

                    proposal = portfolio_decision_to_action_proposal(
                        pm_decision,
                        symbol=ticker,
                        decision_ts=decision_ts,
                    )
                    # Reuse the OHLCV that the chart UI already fetched.
                    try:
                        end_dt = datetime.fromisoformat(decision_ts.replace("Z", "+00:00")).replace(tzinfo=None)
                    except Exception:
                        end_dt = datetime.utcnow()
                    ohlcv = _fetch_ohlcv(ticker, end_dt, lookback_days=365)
                    self._run_execution_pipeline(ticker, proposal, ohlcv)
                    self._record_episode(ticker, proposal, decision_ts, regime_label)
                except Exception:
                    logger.exception("M1-M5 post-LLM stage failed")

            usage: Dict[str, Any] = {}
            try:
                usage = (
                    crew.usage_metrics.model_dump()
                    if hasattr(crew.usage_metrics, "model_dump")
                    else dict(crew.usage_metrics)
                )
            except Exception:
                usage = {}

            self._emit(make_event("run_completed", usage=usage))
        except Exception as exc:
            logger.exception("Analysis run failed")
            self._emit(make_event(
                "error",
                message=str(exc),
                traceback=traceback.format_exc(),
            ))
        finally:
            # Phase E — persist whatever we captured.  Even partial runs
            # (e.g. an error mid-debate) get a record so users can see
            # what happened before the failure.
            if self._run_record is not None:
                try:
                    write_run(self._run_record)
                except Exception:
                    logger.exception("Failed to persist run record (non-fatal)")
            # Phase 2F — clear the UI preset override so the next run on
            # this thread (or the next thread reusing the worker) starts
            # from a clean slate.
            try:
                from trading_crew import llm_presets as _llm_presets
                _llm_presets.clear_active()
            except Exception:
                pass
            try:
                from trading_crew import embedding_presets as _embedding_presets
                _embedding_presets.clear_active()
            except Exception:
                pass

    async def run(self, ticker: str, ui_cfg: Dict[str, Any]) -> None:
        await self.start_pump()
        await asyncio.to_thread(self._run_blocking, ticker, ui_cfg)
        if self._pump_task is not None:
            try:
                await asyncio.wait_for(self._pump_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._pump_task.cancel()

    def _detach_event_handlers(self) -> None:
        """Remove our handlers from the CrewAI singleton event bus so they
        don't accumulate / fire across multiple kickoffs in the same process.
        """
        for evt_type, handler in (
            (ToolUsageStartedEvent, self._on_tool_started),
            (ToolUsageFinishedEvent, self._on_tool_finished),
            (ToolUsageErrorEvent, self._on_tool_error),
        ):
            try:
                # CrewAIEventsBus stores frozensets keyed by event_type
                with crewai_event_bus._instance_lock:
                    s = crewai_event_bus._sync_handlers.get(evt_type, frozenset())
                    crewai_event_bus._sync_handlers[evt_type] = s - {handler}
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Catalogs used by /api/options to render the sidebar
# ---------------------------------------------------------------------------

def get_tool_catalog() -> Dict[str, Dict[str, str]]:
    """Tool name -> {name, description} for the UI sidebar."""
    out: Dict[str, Dict[str, str]] = {}
    for name, fn in ALL_TOOLS.items():
        doc = (fn.description or "").strip().splitlines()
        short = next((ln.strip() for ln in doc if ln.strip()), name)
        out[name] = {"name": name, "description": short}
    return out


def get_agent_catalog() -> List[Dict[str, Any]]:
    """Agent layout for the workflow diagram + tool checkboxes."""
    layout = [
        ("market_analyst", "Market Analyst", "analyst"),
        ("social_analyst", "Social Analyst", "analyst"),
        ("news_analyst", "News Analyst", "analyst"),
        ("fundamentals_analyst", "Fundamentals Analyst", "analyst"),
        ("macro_analyst", "Macro Analyst", "analyst"),
        ("geopolitical_analyst", "Geopolitical Analyst", "analyst"),
        ("sector_analyst", "Sector / Peer Analyst", "analyst"),
        ("quant_analyst", "Quant / Options Analyst", "analyst"),
        ("bull_researcher", "Bullish Researcher", "bull"),
        ("bear_researcher", "Bearish Researcher", "bear"),
        ("research_manager", "Research Manager", "manager"),
        ("quality_reviewer", "Quality Reviewer", "reviewer"),
        ("trader", "Trader", "trader"),
        ("risk_aggressive", "Aggressive Risk Analyst", "risk_a"),
        ("risk_neutral", "Neutral Risk Analyst", "risk_n"),
        ("risk_conservative", "Conservative Risk Analyst", "risk_c"),
        ("compliance_officer", "Compliance Officer", "reviewer"),
        ("portfolio_manager", "Portfolio Manager", "manager"),
    ]
    out = []
    for key, role, kind in layout:
        tools = DEFAULT_AGENT_TOOLS.get(key, [])
        out.append({"key": key, "role": role, "kind": kind, "tools": tools})
    return out
