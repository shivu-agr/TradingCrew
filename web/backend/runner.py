"""Run TradingCrew in a background thread and stream events.

Architecture
------------
* CrewAI's ``step_callback`` / ``task_callback`` fire on the kickoff thread.
* We push each callback into a thread-safe ``queue.Queue`` (``self._sync_q``).
* An asyncio pump task drains ``_sync_q`` (via ``asyncio.to_thread``) into
  ``_async_q`` so the WebSocket coroutine can stream events to the browser.

Streaming-gap fix
-----------------
With ``async_execution=True`` analyst tasks (deterministic v2 crew), all
five analysts fire ``step_callback`` near-simultaneously and the UI lights
them up. After they all complete, the next task (Bullish Researcher
round 1) is a *sync, tool-using* task: its first step_callback can sit
30–60 s later. The UI sat completely idle in that gap and looked stuck.

We now precompute the expected agent ordering from ``crew.tasks``. Whenever a
``task_callback`` fires we increment a counter and look up the *next* expected
agent role in the list — if it has not been announced yet we synthesize a
``node_started`` event for it. That keeps the diagram "alive" even while a
sequential, tool-less agent is mid-LLM-call.
"""

import asyncio
import json
import logging
import os
import queue
import re
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
from trading_crew._patches import DEGRADED_OUTPUT_MARKER, DEGRADED_TIMEOUT_MARKER
from trading_crew.critic import run_reflective_critic, records_to_payload
from trading_crew.tools import (
    ALL_TOOLS,
    DEFAULT_AGENT_TOOLS,
    normalize_source_identifier,
    normalize_source_providers,
    sanitize_citations,
)

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

# ``EpisodicMemory`` is now SQLite-backed (WAL + busy-timeout), but we still
# serialize episode writes process-wide so interactive sessions AND the batch
# forward-collection (several sessions in parallel) share one clean writer and
# never contend on the same row mid-upsert.
_EPISODE_WRITE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_text(value: Any, limit: Optional[int] = 4000) -> str:
    """Coerce a CrewAI step / task object to a readable string.

    ``limit=None`` disables truncation entirely — used for the report / tool /
    decision bodies the UI renders in full (analysts + PM outputs must never be
    silently clipped, or the analyst sees a partial thesis)."""
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
    if limit is not None and len(out) > limit:
        return out[: limit - 20] + f"\n…[truncated {len(out) - limit} chars]"
    return out


def _critic_hard_fail_reason(pm_decision: Any) -> Optional[str]:
    """Return a non-None reason ONLY when the PM decision has a hard integrity
    failure the critic is allowed to override even in advisory mode:

      * ``compliance_status == "BLOCKED"`` — must not trade, and
      * a fabricated / placeholder source citation in the rationale/sources
        (detected deterministically: ``sanitize_citations`` relabels fake
        ``[source: example.com/…]`` tags, so a changed string means one was
        present).

    A weak / mixed signal, a qualitative ``[analyst judgement]`` claim, or a
    plain REVISE-to-downgrade is NOT a hard failure — those are advisory only.
    """
    if getattr(pm_decision, "compliance_status", None) == "BLOCKED":
        return "compliance BLOCKED"
    prov = "\n".join(
        [getattr(pm_decision, "rationale", "") or ""]
        + list(getattr(pm_decision, "sources", None) or [])
    )
    if prov and sanitize_citations(prov) != prov:
        return "fabricated source citation"
    return None


def _resolve_critic_outcome(pm_decision: Any, critiqued: Any, advisory: bool):
    """Decide whether the critic's revised decision replaces the PM's.

    Returns ``(final_decision, apply_override, hard_fail_reason)``.

    Advisory (default): keep the PM's action/size — the critic is a
    one-directional dampener (only ever holds / downgrades / abstains) so
    letting it override both flips the decision at the last moment and biases
    the whole book toward HOLD/SELL.  Override ONLY on a hard integrity
    failure.  Non-advisory restores the legacy behaviour (critic always wins).
    """
    reason = _critic_hard_fail_reason(pm_decision)
    apply_override = (not advisory) or (reason is not None)
    return (critiqued if apply_override else pm_decision), apply_override, reason


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
# Per-desk bias board (direct signal → PM)
# ---------------------------------------------------------------------------
# Compact display label for each analyst role, in canonical lens order.
_DESK_LABELS: List[Tuple[str, str]] = [
    ("Market Analyst", "Market"),
    ("News Analyst", "News"),
    ("Fundamentals Analyst", "Fundamentals"),
    ("Reports Analyst", "Reports/Guidance"),
    ("Macro Analyst", "Macro/Geopolitical"),
]
# The analyst TLDR is required to end with a ``Bias: bullish/bearish/neutral``
# line (see config/tasks.yaml).  Models render it with assorted markdown
# leaders — ``- Bias:``, ``- **Bias:**``, ``* Bias -`` — and sometimes prepend
# one qualifier word (``Long-term bias:``, ``Overall bias:``, ``Net bias:``).
# The pattern tolerates markdown markers + an OPTIONAL single safe qualifier,
# but is anchored to line-start (multiline) so it can't latch onto free prose
# like "net positive bias." or the reports desk's ``NET_GUIDANCE_BIAS:`` line.
# We read the FIRST such line (it lives in the TLDR near the top).
_BIAS_LINE_RE = re.compile(
    r"(?im)^[\s\-*#>•\u2022]*"
    r"(?:\*\*\s*)?"                                                    # optional opening bold
    r"(?:(?:overall|net|long[\s-]?term|final|directional|composite)\s+)?"  # optional qualifier
    r"bias\b\s*(?:\*\*)?\s*[:\-=]\s*(.+)$"
)
# Canonical machine tag each analyst task is now required to emit as its final
# line (``DESK_BIAS: BULLISH|BEARISH|NEUTRAL``).  This is the RELIABLE source —
# the free-text TLDR bias line varies too much run-to-run on the OSS model to
# populate a 5/5 board.  Parsed strictly (uppercase verdict), preferred over
# the prose line below.  Same structured-output discipline as the reports
# desk's ``NET_GUIDANCE_BIAS:``.
_DESK_BIAS_TAG_RE = re.compile(
    r"(?im)^[\s\-*#>•\u2022]*\**\s*DESK_BIAS\s*\**\s*[:=]\s*\**\s*(BULLISH|BEARISH|NEUTRAL)\b"
)
_BULL_WORDS = ("bull", "positive", "long", "constructive", "overweight", "buy")
_BEAR_WORDS = ("bear", "negative", "short", "cautious", "underweight", "sell")


def _pm_bias_board_enabled() -> bool:
    """Direct per-desk bias board into the PM prompt.  Costs no extra LLM/tool
    calls (pure parse of already-captured TLDRs), so default ON; flip
    ``TRADINGCREW_PM_DESK_BIAS_BOARD=0`` for A/B or rollback."""
    raw = os.environ.get("TRADINGCREW_PM_DESK_BIAS_BOARD", "1")
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def _pm_price_structure_enabled() -> bool:
    """Direct the deterministic price-structure card (vol-state / regime-age /
    DIP_BUY) into the PM prompt + emit it for the UI.  One cheap OHLCV read +
    a BOCPD pass (~1s), no LLM, so default ON; flip
    ``TRADINGCREW_PM_PRICE_STRUCTURE=0`` for A/B or rollback."""
    raw = os.environ.get("TRADINGCREW_PM_PRICE_STRUCTURE", "1")
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def _pm_align_guard_enabled() -> bool:
    """Post-PM enforcement of the desk-bias board + computed trends.
    Default ON; flip ``TRADINGCREW_PM_ALIGN_GUARD=0`` for A/B or rollback."""
    raw = os.environ.get("TRADINGCREW_PM_ALIGN_GUARD", "1")
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def _parse_desk_bias(raw_text: str) -> Optional[str]:
    """Classify an analyst report's self-declared bias into ``bullish`` /
    ``bearish`` / ``neutral``.

    Reads the analyst's own explicit declaration in one of two forms: the
    canonical ``DESK_BIAS: BULLISH|BEARISH|NEUTRAL`` machine tag (preferred —
    reliable), else the free-text TLDR ``Bias:`` line.  Returns ``None`` when
    the desk declared neither (nothing to surface — we never guess a lean)."""
    if not raw_text:
        return None
    tag = _DESK_BIAS_TAG_RE.search(raw_text)
    if tag:
        return {"BULLISH": "bullish", "BEARISH": "bearish", "NEUTRAL": "neutral"}[
            tag.group(1).upper()
        ]
    m = _BIAS_LINE_RE.search(raw_text)
    if not m:
        return None
    val = m.group(1).strip().lower()
    # Strip any trailing rationale / source tag after the verdict word.
    has_bull = any(w in val for w in _BULL_WORDS)
    has_bear = any(w in val for w in _BEAR_WORDS)
    if has_bull and not has_bear:
        return "bullish"
    if has_bear and not has_bull:
        return "bearish"
    if "neutral" in val or "mixed" in val or "balanced" in val:
        return "neutral"
    # Ambiguous line (both/neither keyword) — treat as neutral rather than drop,
    # so the desk still shows up on the board as "did not lean".
    return "neutral"


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
        # Parallel array: round number for each slot in
        # ``_expected_role_order``.  1-based; single-round roles get 1.
        # Populated alongside ``_expected_role_order`` in ``_run_blocking``.
        self._role_round_per_slot: List[int] = []
        # Map role → total number of rounds that role will fire.
        self._role_total_rounds: Dict[str, int] = {}
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
        # CrewAI 1.14+ dispatches sync event handlers through a
        # ``ThreadPoolExecutor`` (see
        # ``crewai/events/event_bus.py::CrewAIEventsBus.emit``).  When a
        # tool call is fast (e.g. cache hit → 0 ms elapsed) or when the
        # experimental agent-executor runs several tool calls in
        # parallel via its own ``ThreadPoolExecutor`` (see
        # ``crewai/experimental/agent_executor.py::_execute_native_tool_calls``),
        # our ``_on_tool_started`` and ``_on_tool_finished`` handlers
        # for the *same call* race — and the "finished" one can win.
        # The result was a UI where each ``tool_result`` card rendered
        # *before* its own ``tool_call`` card.
        #
        # ``_tool_lock`` + ``_tool_cv`` serialise our access to
        # ``_open_tool_calls`` AND let a Finished handler block briefly
        # until its matching Started has actually registered the call.
        # The Started handler always emits the UI event while holding
        # the lock, so the sync queue receives ``tool_call`` before
        # ``tool_result`` for every pair.
        self._tool_lock: threading.Lock = threading.Lock()
        self._tool_cv: threading.Condition = threading.Condition(self._tool_lock)
        # Concurrency-safety filter.  CrewAI's event bus is a *process
        # singleton* (see crewai.events.event_bus.CrewAIEventsBus) so
        # when the SessionManager runs two kickoffs in parallel, our
        # ``_on_tool_started`` / _on_tool_finished handlers fire for
        # *every* crew's tool events — a run for BAJAJ-AUTO.NS would
        # see tool_call cards from a concurrent RELIANCE.NS run and
        # attribute them to the wrong session.
        #
        # ``ToolUsageEvent.agent_id`` is a per-Agent UUID assigned at
        # construction time, so populating this set from
        # ``crew.agents`` at run-start and rejecting events whose
        # ``agent_id`` isn't in it gives us clean per-session isolation
        # without patching the singleton bus.  Populated in
        # ``_run_blocking`` immediately after ``tc.crew()`` returns.
        self._own_agent_ids: set[str] = set()
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
        # Optional episode source override (e.g. "forward" when driven by the
        # batch forward-collection).  None -> use the versioning default.
        self._episode_source: Optional[str] = None
        # Set once a ``report_signals`` event has been emitted for this run so
        # we don't emit it twice (the primary source is the get_report_insights
        # TOOL output — which carries both the LLM filing card AND the
        # deterministic computed-trend block; the Reports Analyst node output
        # is only a fallback, since the analyst may drop the computed block).
        self._report_signals_emitted: bool = False
        # PM task handle (set after the crew is built) + a guard so the
        # deterministic reports-desk guidance ledger is appended to the PM
        # prompt exactly once, before the PM node runs.
        self._pm_task: Any = None
        self._graph_edges: List[Dict[str, str]] = []
        self._graph_mode: bool = False
        self._harness_mode: bool = False
        self._run_mode: str = "autonomous"
        self._graph_agents: Dict[str, Any] = {}
        # The live crew for the current run, held so ``_run_blocking``'s
        # ``finally`` can tear its CrewAI Memory (LanceDB + save pool) down.
        self._crew: Any = None
        self._pm_ledger_injected: bool = False
        self._pm_regime_injected: bool = False
        # Deterministic price-structure card (vol-state / regime-age / DIP_BUY)
        # computed once and injected into the PM prompt + emitted for the UI.
        self._pm_price_injected: bool = False
        self._price_structure: Any = None
        # Per-desk directional lean, parsed from each analyst's TLDR ``Bias:``
        # line as it completes; consolidated into a compact board that is
        # handed to the PM the moment its node is announced (all analysts have
        # finished by then).  See ``_inject_pm_bias_board``.
        self._desk_biases: Dict[str, str] = {}
        self._pm_bias_board_injected: bool = False

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

    _GRAPH_ASYNC_ROLES = frozenset({
        "Market Analyst",
        "News Analyst",
        "Fundamentals Analyst",
        "Reports Analyst",
        "Macro Analyst",
        "Aggressive Risk Analyst",
        "Conservative Risk Analyst",
        "Quality Reviewer",
        "Compliance Officer",
    })

    def _rebuild_role_rounds(self) -> None:
        seen: Dict[str, int] = {}
        self._role_round_per_slot = []
        for role in self._expected_role_order:
            seen[role] = seen.get(role, 0) + 1
            self._role_round_per_slot.append(seen[role])
        self._role_total_rounds = {
            r: self._expected_role_order.count(r) for r in set(self._expected_role_order)
        }

    def _append_expected_roles(self, roles: List[str]) -> None:
        self._expected_role_order.extend(roles)
        self._expected_async_flags.extend([False] * len(roles))
        self._rebuild_role_rounds()

    def _insert_role_before(self, role: str, before: str) -> None:
        for i, existing in enumerate(self._expected_role_order):
            if existing == before and i not in self._completed_slot_indices:
                self._expected_role_order.insert(i, role)
                self._expected_async_flags.insert(i, False)
                self._completed_slot_indices = {
                    (idx + 1 if idx >= i else idx) for idx in self._completed_slot_indices
                }
                self._rebuild_role_rounds()
                return
        self._append_expected_roles([role])

    def _on_graph_edge(self, src: str, dst: str, kind: str) -> None:
        self._graph_edges.append({"src": src, "dst": dst, "kind": kind})
        self._emit(make_event("graph_edge", src=src, dst=dst, kind=kind))
        if kind == "cycled" and src == "Bearish Researcher" and dst == "Bullish Researcher":
            self._append_expected_roles(["Bullish Researcher", "Bearish Researcher"])
        elif kind == "cycled" and dst == "Trader":
            self._insert_role_before("Trader", "Portfolio Manager")

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
                    output = event.get("output") or ""
                    rec.reports[role] = output
                    # Full per-round audit so the Reports tab can render
                    # round 1 / round 2 side-by-side instead of only the
                    # last round (which is what ``rec.reports`` keeps).
                    rec.reports_by_round.setdefault(role, []).append({
                        "round": int(event.get("round") or 1),
                        "total_rounds": int(event.get("total_rounds") or 1),
                        "output": output,
                        "degraded": bool(event.get("degraded")),
                    })
                    if event.get("degraded"):
                        rec.degraded_roles.append(role)
            elif t == "final_decision":
                rec.final_decision = event.get("decision")
                rec.final_decision_source = event.get("source") or rec.final_decision_source or "crew"
            elif t == "report_signals":
                rec.report_signals = {
                    "signals": event.get("signals") or [],
                    "net_bias": event.get("net_bias"),
                    "net_confidence": event.get("net_confidence"),
                    "bias_score": event.get("bias_score"),
                }
            elif t == "price_signals":
                rec.price_signals = {
                    "signals": event.get("signals") or [],
                    "vol_pct": event.get("vol_pct"),
                    "ann_vol": event.get("ann_vol"),
                    "atr_pct": event.get("atr_pct"),
                    "pos_in_range": event.get("pos_in_range"),
                    "drawdown_from_high": event.get("drawdown_from_high"),
                    "regime_age_days": event.get("regime_age_days"),
                    "hurst": event.get("hurst"),
                    "dip_buy": event.get("dip_buy"),
                    "size_risk_mult": event.get("size_risk_mult"),
                    "size_note": event.get("size_note"),
                }
            elif t == "desk_bias_board":
                rec.desk_bias_board = event.get("desks") or []
            elif t == "evidence_blackboard":
                rec.evidence_blackboard = {
                    "ticker": event.get("ticker"),
                    "markdown": event.get("markdown") or "",
                    "desks": event.get("desks") or [],
                    "requests": event.get("requests") or [],
                }
            elif t == "debate_scratchpad":
                rec.debate_scratchpad = {
                    "ticker": event.get("ticker"),
                    "markdown": event.get("markdown") or "",
                    "entries": event.get("entries") or [],
                }
            elif t == "signal_alignment":
                rec.signal_alignment = {
                    "applied": event.get("applied"),
                    "original_action": event.get("original_action"),
                    "final_action": event.get("final_action"),
                    "notes": event.get("notes") or [],
                }
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

    def _inject_pm_regime(self, regime_label: Optional[str]) -> None:
        """Prepend the machine-detected market regime onto the PM prompt.

        The regime is the deterministic price-regime read at decision time
        (M4 cascade).  The market analyst usually mentions it in prose, but
        that lens only reaches the final decision ~80% of the time; injecting
        it directly guarantees the decision desk always sees the *measured*
        regime.  Compact (two lines) and PM-only, so it can't overwhelm the
        mid-pipeline agents.  Non-directive by design — no side is prescribed.
        """
        if self._pm_task is None or self._pm_regime_injected:
            return
        if not regime_label or str(regime_label).upper() == "UNKNOWN":
            return
        try:
            block = (
                f"## MARKET REGIME (machine-detected, deterministic): {str(regime_label).upper()}\n"
                "This is the measured price regime at decision time. Weigh it "
                "alongside the desks; it does not by itself dictate a side.\n"
            )
            self._pm_task.description = f"{block}\n{self._pm_task.description}"
            self._pm_regime_injected = True
        except Exception:  # noqa: BLE001 — never break a run on injection
            logger.warning("PM regime injection failed", exc_info=True)

    def _bias_board_rows(self) -> List[Tuple[str, str]]:
        """Captured per-desk biases as ``(display_label, bias)`` in canonical
        lens order — the shared source for the prompt block and the UI event."""
        return [
            (label, self._desk_biases[role])
            for role, label in _DESK_LABELS
            if role in self._desk_biases
        ]

    def _render_bias_board(self, rows: List[Tuple[str, str]]) -> str:
        """Render ``rows`` as the compact PM-facing prompt block (``""`` if empty)."""
        if not rows:
            return ""
        width = max(len(label) for label, _ in rows)
        lines = [f"- {label + ':':<{width + 1}} {bias}" for label, bias in rows]
        return (
            "## DESK BIAS BOARD (machine-parsed from each analyst's TLDR)\n"
            "Every analyst desk's self-declared directional lean at a glance. "
            "Consider ALL of them — a lens you don't address is a lens you "
            "dropped. This does not prescribe a side, but disagreement IS "
            "binding: a split board (bullish count == bearish count, both > 0) "
            "or a majority of directional desks against your call is HOLD, "
            "not a weak BUY/SELL. Neutrals do not create a split "
            "(3 bullish / 2 neutral is a bullish majority). STRONG_* "
            "requires a one-sided board (no opposing desk). Name specific "
            "tagged evidence only if you override a dissenting desk; "
            "otherwise sit out.\n"
            + "\n".join(lines)
            + "\n"
        )

    def _build_bias_board(self) -> str:
        """Compact PM-facing board from the captured biases, or ``""``."""
        return self._render_bias_board(self._bias_board_rows())

    def _inject_pm_bias_board(self) -> None:
        """Emit the per-desk bias board for the UI AND prepend it onto the PM
        prompt exactly once.

        Called when the PM node is announced — by then all analyst desks have
        completed, so ``_desk_biases`` is fully populated.  This is a *direct*
        structured signal to the decision point only (never broadcast
        mid-pipeline), complementing the diluted prose that flows through the
        normal analyst→manager→PM path.  Non-directive by construction.
        """
        if self._pm_task is None or self._pm_bias_board_injected:
            return
        if not _pm_bias_board_enabled():
            return
        rows = self._bias_board_rows()
        if not rows:
            return
        # Surface the board to the dashboard (Decision-node panel) — the same
        # structured signal the PM is about to see, so the operator can audit it.
        self._emit(make_event(
            "desk_bias_board",
            desks=[{"desk": label, "bias": bias} for label, bias in rows],
        ))
        board = self._render_bias_board(rows)
        try:
            self._pm_task.description = f"{board}\n{self._pm_task.description}"
            self._pm_bias_board_injected = True
        except Exception:  # noqa: BLE001 — never break a run on injection
            logger.warning("PM desk-bias-board injection failed", exc_info=True)

    def _emit_blackboard(self) -> None:
        """Push the full evidence board to the UI (no clip)."""
        try:
            from trading_crew.agentic.graph.blackboard import current
            bb = current()
            if bb is None or (not bb.desk_reports and not bb.desk_leans):
                return
            self._emit(make_event("evidence_blackboard", **bb.to_payload()))
        except Exception:
            logger.debug("evidence_blackboard emit failed", exc_info=True)

    def _emit_scratchpad(self) -> None:
        """Push the shared bull/bear scratch pad to the UI."""
        try:
            from trading_crew.agentic.graph.scratchpad import current as pad_current
            pad = pad_current()
            if pad is None or not pad.entries:
                return
            self._emit(make_event("debate_scratchpad", **pad.to_payload()))
        except Exception:
            logger.debug("debate_scratchpad emit failed", exc_info=True)

    def _apply_signal_alignment(self, pm_decision):
        """Enforce desk-board + computed-trend disagreement on the PM call.

        Prompt injection is a completeness guarantee; this is the utilization
        guarantee.  Symmetric (long/short), never invents a side, only
        HOLDs / caps STRONG when the signals the PM already saw disagree.
        """
        if not _pm_align_guard_enabled() or pm_decision is None:
            return pm_decision
        from trading_crew.agentic.signals.alignment import apply_signal_alignment

        rec = self._run_record
        board = (rec.desk_bias_board if rec is not None else None) or [
            {"desk": label, "bias": bias} for label, bias in self._bias_board_rows()
        ]
        report_signals = rec.report_signals if rec is not None else None
        aligned, notes = apply_signal_alignment(
            pm_decision, board=board, report_signals=report_signals,
        )
        if not notes:
            return pm_decision
        changed = (
            aligned.action != pm_decision.action
            or aligned.size_pct_of_book != pm_decision.size_pct_of_book
        )
        self._emit(make_event(
            "signal_alignment",
            applied=changed,
            original_action=pm_decision.action,
            final_action=aligned.action,
            notes=notes,
        ))
        if changed:
            self._emit(make_event(
                "final_decision",
                decision=aligned.model_dump(),
                raw=aligned.rationale,
                source="signal_alignment",
            ))
        return aligned

    def _inject_pm_price_structure(self) -> None:
        """Compute the deterministic price-structure card and (a) emit it for the
        UI and (b) prepend its authoritative, non-directive ledger onto the PM
        prompt exactly once.

        Called when the PM node is announced.  Unlike the report signals (which
        are parsed out of analyst prose that often drops the card), this is
        computed straight from OHLCV so it ALWAYS reaches the decision.  It
        informs SIZE (vol-state) and flags a measured mean-reversion ENTRY edge
        (DIP_BUY); it never prescribes a side.  Best-effort — a failure here
        must never break the run.
        """
        if self._pm_price_injected or self._pm_task is None:
            return
        if not _pm_price_structure_enabled():
            return
        ticker = getattr(self._run_record, "ticker", None) if self._run_record else None
        if not ticker:
            return
        try:
            from trading_crew.tools import _load_price_history
            from trading_crew.price_structure import compute_price_structure

            df, source_id = _load_price_history(ticker, period="5y")
            last_close = None
            try:
                if df is not None and not getattr(df, "empty", True):
                    last_close = float(df["Close"].iloc[-1])
            except Exception:
                last_close = None
            blocks: list[str] = []
            if last_close and last_close > 0:
                blocks.append(
                    f"LAST_CLOSE: {last_close:.4f} [source: {source_id} · ohlcv]. "
                    "Ground entry_price to this print or to a limit within a few "
                    "percent of it. Do not copy a trader entry that is tens of "
                    "percent away, and do not invent a round placeholder "
                    "(100 / 50 / 10)."
                )
            ps = compute_price_structure(df, source=f"{source_id} · price-structure (computed)")
            if ps is not None and ps.signals:
                self._price_structure = ps
                rm, rm_why = ps.risk_mult()

                def _fin(v: Any) -> Any:
                    return v if isinstance(v, (int, float)) and v == v and v not in (float("inf"), float("-inf")) else None

                self._emit(make_event(
                    "price_signals",
                    signals=[s.model_dump() for s in ps.signals],
                    vol_pct=_fin(ps.vol_pct),
                    ann_vol=_fin(ps.ann_vol),
                    atr_pct=_fin(ps.atr_pct),
                    pos_in_range=_fin(ps.pos_in_range),
                    drawdown_from_high=_fin(ps.drawdown_from_high),
                    regime_age_days=_fin(ps.regime_age_days),
                    hurst=_fin(ps.hurst),
                    dip_buy=bool(ps.dip_buy),
                    size_risk_mult=round(rm, 3),
                    size_note=rm_why,
                    last_close=_fin(last_close),
                ))
                ledger = ps.render_pm_ledger()
                if ledger:
                    blocks.append(ledger)
            if not blocks:
                return
            self._pm_price_injected = True
            try:
                self._pm_task.description = (
                    "\n\n".join(blocks) + "\n\n" + self._pm_task.description
                )
            except Exception:  # noqa: BLE001 — never break a run on injection
                logger.warning("PM price-structure injection failed", exc_info=True)
        except Exception:  # noqa: BLE001 — price structure is advisory
            logger.exception("Failed to compute/inject price-structure signals (non-fatal)")

    def _emit_report_signals(self, reports_text: str) -> None:
        """Parse the reports desk output into structured signals and emit them.

        Idempotent (emits at most once per run) and advisory / best-effort —
        a parse miss must never break the run.
        """
        if self._report_signals_emitted:
            return
        try:
            from trading_crew.report_signals import (
                guidance_bias_score,
                parse_signal_card,
                render_pm_ledger,
            )

            signals = parse_signal_card(reports_text)
            if signals.is_empty():
                return
            self._report_signals_emitted = True
            self._emit(make_event(
                "report_signals",
                signals=[s.model_dump() for s in signals.signals],
                net_bias=signals.net_bias,
                net_confidence=signals.net_confidence,
                bias_score=guidance_bias_score(signals),
            ))
            # Guarantee the parsed guidance signal reaches the PM decision.
            # The reports desk's prose historically dropped this card ~77% of
            # the time; we append the machine-parsed ledger straight onto the
            # PM prompt (PM runs last, so the mutation lands before it fires).
            if self._pm_task is not None and not self._pm_ledger_injected:
                ledger = render_pm_ledger(signals)
                if ledger:
                    try:
                        self._pm_task.description = (
                            f"{ledger}\n\n{self._pm_task.description}"
                        )
                        self._pm_ledger_injected = True
                    except Exception:  # noqa: BLE001 — never break a run on injection
                        logger.warning("PM guidance-ledger injection failed", exc_info=True)
        except Exception:  # noqa: BLE001 — report signals are advisory
            logger.exception("Failed to parse report trend signals (non-fatal)")

    def _announce(self, role: str) -> None:
        """Send a ``node_started`` event for ``role`` if it isn't already
        marked as running. Re-entry after node_completed is allowed so
        debate-round repeats (Bull/Bear, Aggressive/Neutral/Conservative)
        can flip the UI back to running.
        """
        if not role or role in self._running_roles:
            return
        # The PM is announced only after every analyst desk has completed, so
        # this is the last-moment, all-biases-available point to hand the PM
        # its compact desk-bias board (mutation lands before CrewAI builds the
        # PM prompt, same as the guidance-ledger injection).
        if role == "Portfolio Manager":
            self._inject_pm_bias_board()
            self._inject_pm_price_structure()
        self._running_roles.add(role)
        # Look up the round number this announcement corresponds to:
        # the first un-completed slot whose role matches.  Lets the UI
        # render "Bullish Researcher — round 2 of 2" on the diagram
        # subtitle without having to re-derive round_num from the
        # completion counter on the frontend.
        round_num = 1
        total_rounds = self._role_total_rounds.get(role, 1)
        for i, expected in enumerate(self._expected_role_order):
            if i in self._completed_slot_indices:
                continue
            if expected == role and i < len(self._role_round_per_slot):
                round_num = self._role_round_per_slot[i]
                break
        self._emit(make_event(
            "node_started",
            node=role, role=role, kind=NODE_KIND.get(role, "node"),
            round=round_num,
            total_rounds=total_rounds,
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
        if not self._is_own_event(event):
            return
        role = event.agent_role or "Agent"
        self._announce(role)
        args = event.tool_args
        if isinstance(args, str):
            args_text = args[:600]
        else:
            try:
                args_text = json.dumps(args, default=str)[:600]
            except Exception:
                args_text = str(args)[:600]
        # Register the open call AND emit the ``tool_call`` UI event
        # atomically while holding the lock.  Any concurrent Finished
        # handler for the same (role, tool) blocked in ``_on_tool_finished``
        # is woken via ``_tool_cv.notify_all()`` so it can pop the call
        # and emit ``tool_result`` immediately after us — guaranteeing
        # the sync queue sees ``tool_call`` before ``tool_result``.
        with self._tool_cv:
            self._tool_call_seq += 1
            call_id = f"{role}-{event.tool_name}-{self._tool_call_seq}"
            self._open_tool_calls[call_id] = {
                "tool": event.tool_name, "node": role, "started_at": time.time(),
            }
            self._emit(make_event(
                "tool_call",
                node=role, tool=event.tool_name,
                call_id=call_id,
                args=args_text,
            ))
            self._tool_cv.notify_all()

    def _on_tool_finished(self, _source: Any, event: ToolUsageFinishedEvent) -> None:
        if self._cancel_event.is_set():
            return
        if not self._is_own_event(event):
            return
        role = event.agent_role or "Agent"
        elapsed_ms = 0
        try:
            if event.started_at and event.finished_at:
                elapsed_ms = int((event.finished_at - event.started_at).total_seconds() * 1000)
        except Exception:
            pass
        # Wait up to ~5 s for the matching Started to register the
        # open call.  Fast (cached) tools + CrewAI's threaded event
        # dispatch mean the Finished handler often reaches us first;
        # blocking here keeps the emitted order ``tool_call`` →
        # ``tool_result`` regardless of scheduling.  If nothing ever
        # arrives we still emit with ``call_id=None`` so the UI can
        # render an orphan result instead of dropping data.
        call_id = self._wait_for_matching_call(role, event.tool_name)
        output = _safe_text(event.output, limit=None)
        self._emit(make_event(
            "tool_result",
            node=role, tool=event.tool_name, call_id=call_id,
            output=output,
            elapsed_ms=elapsed_ms,
        ))
        # The reports desk's tool output is the canonical signal source — it
        # carries BOTH the LLM filing card and the deterministic computed
        # financial-trend block (which the analyst's own synthesis may drop).
        if event.tool_name == "get_report_insights" and output:
            self._emit_report_signals(output)

    def _on_tool_error(self, _source: Any, event: ToolUsageErrorEvent) -> None:
        if self._cancel_event.is_set():
            return
        if not self._is_own_event(event):
            return
        role = event.agent_role or "Agent"
        call_id = self._wait_for_matching_call(role, event.tool_name)
        self._emit(make_event(
            "tool_result",
            node=role, tool=event.tool_name, call_id=call_id,
            error=str(event.error),
            output=str(event.error)[:1000],
            elapsed_ms=0,
        ))

    def _is_own_event(self, event: Any) -> bool:
        """Filter for the CrewAI singleton event bus.

        When multiple ``AnalysisRunner``s are alive concurrently (see
        ``session_manager.py``), each one's handler fires for *every*
        crew's tool events — the bus is process-global.  If we haven't
        recorded our own agent-id set yet (init race, or a run that
        never called ``_run_blocking``) we conservatively accept the
        event so single-run behaviour is unchanged.
        """
        if not self._own_agent_ids:
            return True
        agent_id = getattr(event, "agent_id", None)
        if agent_id is None:
            return True
        return str(agent_id) in self._own_agent_ids

    def _pop_matching_call(self, role: str, tool_name: str) -> Optional[str]:
        """Pop the most recent open call for ``(role, tool_name)``.

        Caller must hold ``_tool_cv`` / ``_tool_lock``.
        """
        for cid in list(self._open_tool_calls.keys())[::-1]:
            meta = self._open_tool_calls[cid]
            if meta["node"] == role and meta["tool"] == tool_name:
                del self._open_tool_calls[cid]
                return cid
        return None

    def _wait_for_matching_call(
        self, role: str, tool_name: str, *, timeout: float = 5.0,
    ) -> Optional[str]:
        """Return the ``call_id`` of a matching open call, waiting if needed.

        Blocks up to ``timeout`` seconds for a Started handler running
        concurrently in CrewAI's thread pool to register the open call.
        Returns ``None`` on timeout (orphan Finished) so the UI still
        renders the tool_result instead of silently dropping it.
        """
        deadline = time.monotonic() + timeout
        with self._tool_cv:
            while True:
                cid = self._pop_matching_call(role, tool_name)
                if cid is not None:
                    return cid
                if self._cancel_event.is_set():
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._tool_cv.wait(timeout=remaining)

    # Post-consolidation: 5 merged analyst desks (was 8, then 4 + reports).
    _ANALYST_ROLE_TO_TASK_ID = {
        "Market Analyst": "market_task",
        "News Analyst": "news_task",
        "Fundamentals Analyst": "fundamentals_task",
        "Reports Analyst": "reports_task",
        "Macro Analyst": "macro_task",
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
        completed_slot: Optional[int] = None
        for i, expected in enumerate(self._expected_role_order):
            if i in self._completed_slot_indices:
                continue
            if expected == role:
                self._completed_slot_indices.add(i)
                completed_slot = i
                break
        # Look up the round number for this completion so the UI / saved
        # run record can keep round 1 and round 2 outputs side-by-side
        # instead of overwriting round 1 with round 2 on the same role
        # key (the cause of the "multi-turn discussion shows no change"
        # bug — the Bullish Researcher always rendered its last round
        # only because every prior round was clobbered by role-keyed
        # persistence).
        round_num = 1
        total_rounds = 1
        if completed_slot is not None and completed_slot < len(self._role_round_per_slot):
            round_num = self._role_round_per_slot[completed_slot]
            total_rounds = self._role_total_rounds.get(role, 1)
        # Allow the agent to be re-announced if it runs again later (debate
        # rounds re-use the same agent). We only remove from the running set
        # after we've emitted node_completed.
        self._running_roles.discard(role)
        # No truncation: analyst / researcher / PM reports render in full in the
        # UI (a clipped thesis misleads both the reader and the downstream PM).
        raw_text = _safe_text(task_output, limit=None)
        # Deterministically relabel any fabricated ``[source: example.com/…]``
        # citation the LLM invented in its own prose to an honest
        # ``[analyst judgement]`` tag BEFORE the text is shown, persisted,
        # or fed to the PM / critic.  This fixes the root cause of sound
        # theses being downgraded for a model-invented URL (the tool
        # layer already rejects fake URLs, but the model can still type
        # one into free text).
        raw_text = sanitize_citations(raw_text)
        # Rewrite bare section/field ``[source: profit-loss]`` tags to the
        # provider-led ``[source: screener.in · profit-loss]`` form so the UI
        # (and the downstream PM / critic, which read this text) attribute each
        # figure to the DATA PROVIDER it came from, not just its section id.
        raw_text = normalize_source_providers(raw_text)
        # When the OSS-LLM tool-call-as-final-answer patch fires (see
        # ``trading_crew/_patches.py``) it prefixes the per-task output
        # with ``DEGRADED_OUTPUT_MARKER``.  The companion LLM-timeout
        # patch uses ``DEGRADED_TIMEOUT_MARKER``.  Detect either so the
        # UI can flag the affected report as degraded instead of
        # rendering an empty / nonsense body, and so the saved run
        # record persists the signal for audit.
        degraded = raw_text.startswith(DEGRADED_OUTPUT_MARKER) or raw_text.startswith(
            DEGRADED_TIMEOUT_MARKER
        )
        if degraded:
            self._degraded_roles.add(role)
        self._emit(make_event(
            "node_completed",
            node=role, role=role, kind=NODE_KIND.get(role, "node"),
            output=raw_text,
            degraded=degraded,
            round=round_num,
            total_rounds=total_rounds,
        ))

        # Reports desk — lift its Firefly-style trend-signal card into
        # structure so the dashboard/PM see machine-readable signals (and a
        # single numeric guidance-bias) rather than having to re-parse prose.
        if role == "Reports Analyst" and raw_text and not degraded:
            self._emit_report_signals(raw_text)

        # Capture each analyst desk's self-declared directional lean so the PM
        # gets a compact "every lens at a glance" board (see
        # ``_inject_pm_bias_board``).  Cheap: a single regex over text we've
        # already got in hand — no extra model/tool call.
        if role in self._ANALYST_ROLE_TO_TASK_ID and raw_text and not degraded:
            bias = _parse_desk_bias(raw_text)
            if bias:
                self._desk_biases[role] = bias

        try:
            from trading_crew.agentic.graph.blackboard import current, drain_queued_requests
            from trading_crew.agentic.graph.scratchpad import current as pad_current
            bb = current()
            pad = pad_current()
            if bb is not None and raw_text and not degraded:
                bb.record_desk(role, raw_text, self._desk_biases.get(role))
                self._emit_blackboard()
            if pad is not None and raw_text and not degraded:
                if role in self._ANALYST_ROLE_TO_TASK_ID:
                    pad.seed_analyst(role, self._desk_biases.get(role), raw_text)
                    self._emit_scratchpad()
                elif role in ("Bullish Researcher", "Bearish Researcher"):
                    pad.append_debate(role, round_num, raw_text)
                    self._emit_scratchpad()
            if (
                role in ("Bullish Researcher", "Bearish Researcher")
                and round_num == 1
            ):
                done = getattr(self, "_debate_r1_completed", None)
                if done is None:
                    self._debate_r1_completed = set()
                    done = self._debate_r1_completed
                done.add(role)
                if len(done) >= 2 and bb is not None:
                    n = drain_queued_requests(bb.ticker)
                    if n:
                        self._emit(make_event("evidence_drained", filled=n))
                    self._emit_blackboard()
                    if pad is not None:
                        self._emit_scratchpad()
        except Exception:
            logger.debug("blackboard update failed (non-fatal)", exc_info=True)

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
    def _detect_market_regime(self, ticker: str, decision_ts: str) -> Tuple[Any, str]:
        """Regime of ``ticker``'s broad-market benchmark index.

        Used to tell a *systemic* crisis (skip the debate) apart from an
        *idiosyncratic* single-name shock (evaluate it).  Returns
        ``(Regime, index_symbol)``; ``Regime.UNKNOWN`` on any failure — we
        never fabricate a regime, and the caller treats UNKNOWN as
        "not systemic" so a data gap can't silently trigger an abstain.
        """
        idx = "?"
        try:
            from trading_crew.market_context import benchmark_index_for

            idx = benchmark_index_for(ticker)
            asof = datetime.fromisoformat(decision_ts.replace("Z", "+00:00")).replace(tzinfo=None)
            df = _fetch_ohlcv(idx, asof, 252)
            if df is not None and not df.empty:
                closes = df["Close"].astype(float).tolist()
                return detect_regime(closes), idx
        except Exception as exc:  # noqa: BLE001 — regime is advisory
            logger.warning("market-regime detect failed for %s (idx=%s): %s", ticker, idx, exc)
        return Regime.UNKNOWN, idx

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
                    # A single-name CRISIS only justifies skipping the debate
                    # when the *market* is also in crisis (systemic risk-off).
                    # A lone name gapping down while the benchmark is calm is an
                    # idiosyncratic shock — exactly the oversold-dip case we WANT
                    # the full desk (incl. the DIP_BUY read) to evaluate — so we
                    # don't auto-ABSTAIN it.
                    mkt_regime, idx = self._detect_market_regime(ticker, decision_ts)
                    systemic = mkt_regime in (
                        Regime.CRISIS, Regime.HIGH_VOL,
                        Regime.HIGH_VOL_TREND, Regime.HIGH_VOL_RANGE,
                    )
                    if systemic:
                        route = "CRISIS_OVERRIDE"
                        reason = (
                            f"Systemic crisis — name CRISIS and benchmark {idx} "
                            f"regime={mkt_regime.value} (paper §5.3). Skipping the "
                            "18-agent debate and routing to an ABSTAIN proposal; "
                            "M2 + M5 + M3 still run so the risk gates can flatten."
                        )
                    else:
                        route = "RISK_ONLY"
                        reason = (
                            f"Idiosyncratic single-name shock — name CRISIS but "
                            f"benchmark {idx} regime={mkt_regime.value} (not systemic). "
                            "Running the full desk with risk priority instead of "
                            "auto-abstaining, so an oversold-dip entry isn't vetoed."
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
            action="HOLD",
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
    def _normalize_stops(self, ticker: str, decision, ohlcv):
        """Widen an implausibly tight / wrong-side / missing stop to a
        deterministic volatility+horizon floor.

        Pure-ish: reads OHLCV, computes ATR + annualised vol, and (only when
        the stop is too tight) returns a ``model_copy`` of ``decision`` with a
        corrected ``entry_price`` / ``stop_loss`` and an audit note appended to
        the rationale.  Re-emits ``final_decision`` so the report card shows the
        grounded levels.  On any failure it returns the decision untouched — we
        never fabricate a stop when volatility can't be measured.
        """
        try:
            from trading_crew.agentic.risk.stops import (
                assess_stop,
                compute_atr,
                annualised_vol_from_close,
                entry_needs_grounding,
            )
        except Exception:
            return decision
        try:
            if ohlcv is None or getattr(ohlcv, "empty", True) or len(ohlcv) < 30:
                return decision
            close = [float(x) for x in ohlcv["Close"].tolist()]
            high = [float(x) for x in ohlcv["High"].tolist()]
            low = [float(x) for x in ohlcv["Low"].tolist()]
            atr = compute_atr(high, low, close)
            ann_vol = annualised_vol_from_close(close)
            current_price = close[-1] if close else 0.0

            # Ground a corrupt entry BEFORE it freezes into limit_price.
            # The previous 0.5×–2.0× band left a $100 placeholder on a
            # ~$68 name (1.48×). A marketable limit is a few percent
            # from spot — snap anything outside ±20%.
            entry_in = float(decision.entry_price or 0.0)
            grounded_entry = False
            if entry_needs_grounding(entry_in, current_price):
                logger.warning(
                    "%s: grounding implausible entry_price %.4f -> market close %.4f",
                    ticker, entry_in, current_price,
                )
                decision = decision.model_copy(update={
                    "entry_price": current_price,
                    "rationale": (
                        f"{(decision.rationale or '').strip()}\n"
                        f"[risk] Entry grounded to market close {current_price:.2f} "
                        f"(model emitted {entry_in:.2f}, implausible vs spot)."
                    ).strip(),
                })
                grounded_entry = True

            assessment = assess_stop(
                action=str(decision.action),
                entry=float(decision.entry_price or 0.0),
                stop=float(decision.stop_loss or 0.0),
                target=float(decision.target_price or 0.0),
                horizon_days=int(decision.horizon_days or 21),
                current_price=current_price,
                atr=atr,
                ann_vol=ann_vol,
            )
        except Exception:
            logger.exception("stop normalization failed; leaving levels unchanged")
            return decision

        # No fabricated levels when volatility can't be measured.
        if not assessment.assessable:
            if grounded_entry:
                try:
                    self._emit(make_event(
                        "final_decision",
                        decision=decision.model_dump(),
                        raw=decision.rationale,
                        source="risk_stop_normalizer",
                    ))
                except Exception:
                    logger.exception("failed to re-emit final_decision after entry grounding")
            return decision

        update: Dict[str, Any] = {}
        changes = []
        if assessment.any_adjusted:
            if assessment.adjusted:
                changes.append(assessment.reason)
            if assessment.target_adjusted:
                changes.append(assessment.target_reason)
            vol_ctx = (
                f" (ATR14={atr:.2f}, ann_vol={ann_vol * 100:.0f}%"
                + (f", reward:risk now ~{assessment.reward_risk:.1f}:1" if assessment.reward_risk is not None else "")
                + ")"
            )
            note = f"[risk] Levels normalized deterministically: {'; '.join(changes)}.{vol_ctx}"
            update["entry_price"] = assessment.entry
            update["stop_loss"] = assessment.stop
            update["rationale"] = f"{(decision.rationale or '').strip()}\n{note}".strip()
            if assessment.target_adjusted:
                update["target_price"] = assessment.target
                if assessment.expected_return_pct is not None:
                    update["expected_return_pct"] = assessment.expected_return_pct

        # Attach scale-out tiers whenever we have a valid directional target —
        # even if no level needed correcting — so the trade plan always ships
        # with staged profit targets (Firefly-style).
        if assessment.staged_targets:
            from trading_crew.schemas import StagedTarget

            update["staged_targets"] = [
                StagedTarget(**t) for t in assessment.staged_targets
            ]

        if not update:
            if grounded_entry:
                try:
                    self._emit(make_event(
                        "final_decision",
                        decision=decision.model_dump(),
                        raw=decision.rationale,
                        source="risk_stop_normalizer",
                    ))
                except Exception:
                    logger.exception("failed to re-emit final_decision after entry grounding")
            return decision

        try:
            updated = decision.model_copy(update=update)
        except Exception:
            logger.exception("could not apply normalized levels to decision")
            return decision

        if changes:
            logger.info("levels normalized for %s: %s", ticker, "; ".join(changes))
        try:
            self._emit(make_event(
                "final_decision",
                decision=updated.model_dump(),
                raw=updated.rationale,
                source="risk_stop_normalizer",
            ))
        except Exception:
            logger.exception("failed to re-emit final_decision after stop normalization")
        return updated

    def _apply_hold_watch_plan(self, ticker: str, decision, ohlcv, decision_ts: str):
        """ATR watch stop/target + as-if Kelly size on HOLD. Book size stays 0."""
        if str(getattr(decision, "action", "") or "").upper() != "HOLD":
            return decision
        try:
            if ohlcv is None or getattr(ohlcv, "empty", True) or len(ohlcv) < 30:
                return decision
            from trading_crew.agentic.risk.hold_watch import apply_hold_watch_plan

            updated = apply_hold_watch_plan(
                decision,
                close=[float(x) for x in ohlcv["Close"].tolist()],
                high=[float(x) for x in ohlcv["High"].tolist()],
                low=[float(x) for x in ohlcv["Low"].tolist()],
                desk_biases=getattr(self, "_desk_biases", {}) or {},
                ticker=ticker,
                decision_ts=str(decision_ts or ""),
            )
        except Exception:
            logger.exception("HOLD watch plan failed; leaving sit-out levels unchanged")
            return decision
        if updated is decision:
            return decision
        try:
            self._emit(make_event(
                "final_decision",
                decision=updated.model_dump(),
                raw=updated.rationale,
                source="hold_watch_plan",
            ))
        except Exception:
            logger.exception("failed to re-emit final_decision after HOLD watch plan")
        return updated

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
        # L3 tune (if applied) composed with an open Hermes size experiment.
        sizing_cfg = gate_cfg = None
        tuned_config_applied = False
        hermes_size_scale = None
        try:
            from trading_crew.agentic.tuning import resolve_live_sizing
            sizing_cfg, gate_cfg, _sz_meta = resolve_live_sizing(proposal.symbol)
            tuned_config_applied = bool(_sz_meta.get("l3_applied"))
            hermes_size_scale = _sz_meta.get("hermes_size_scale")
        except Exception:  # noqa: BLE001 — never block execution on tune load
            logger.exception("Live sizing resolve failed; using default sizing/gates")
        try:
            result = run_pipeline(
                proposal,
                ohlcv=ohlcv,
                portfolio_id=self._book,
                cost_model_name=cost_model_name,
                persist=True,
                sizing_config=sizing_cfg,
                gate_config=gate_cfg,
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
            "tuned_config_applied": tuned_config_applied,
            "hermes_size_scale": hermes_size_scale,
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
            review_dt = None
            review_embargo = 0
            try:
                from trading_crew.agentic.strategy import load_desk_strategy
                _strat = load_desk_strategy()
                review_dt = d + timedelta(days=int(_strat.learning.calendar_days))
                review_embargo = int(_strat.learning.embargo_days)
            except Exception:
                logger.exception("desk strategy missing; cannot stamp 30-day review clock")
                raise
            # Optional swarm crowd-positioning stamp (off by default — it
            # costs an extra data fetch).  Lets the backtest segment
            # performance by the crowd regime at decision time.
            crowd_score = None
            crowd_components = None
            if os.environ.get("TRADINGCREW_CROWD_SIGNAL_ENABLED") == "1":
                try:
                    from trading_crew.tools import crowd_signal_for
                    _sig = crowd_signal_for(ticker)
                    if _sig is not None:
                        crowd_score = _sig.score
                        # Per-component breakdown → feeds the crowd-weight learner.
                        crowd_components = dict(_sig.components) or None
                except Exception:  # noqa: BLE001 — never block episode recording
                    crowd_score = None
                    crowd_components = None
            # Phase 0 — stamp the workflow provenance so this episode can
            # later be filtered into a trustworthy cohort (a drifting
            # workflow's proposals must not silently pollute the backtest).
            try:
                from trading_crew.agentic.versioning import (
                    code_git_sha as _code_git_sha,
                    current_source as _current_source,
                    prompts_hash as _prompts_hash,
                    workflow_version as _workflow_version,
                )
                wf_version = _workflow_version()
                wf_prompts_hash = _prompts_hash()
                wf_git_sha = _code_git_sha()
                wf_source = _current_source()
            except Exception:  # noqa: BLE001 — never block episode recording
                wf_version = wf_prompts_hash = wf_git_sha = None
                wf_source = "live"
            # The batch forward-collection tags its episodes ``forward`` so they
            # land in the clean RL cohort (mirrors ``scale.run_collect``).
            if self._episode_source:
                wf_source = self._episode_source
            desk_leans: Optional[Dict[str, str]] = None
            try:
                from trading_crew.agentic.graph.spec import ROLE_TO_AGENT_KEY
                from trading_crew.agentic.graph.blackboard import current as _bb_current
                leans: Dict[str, str] = {}
                for role, lean in (self._desk_biases or {}).items():
                    key = ROLE_TO_AGENT_KEY.get(role)
                    if key and lean:
                        leans[key] = lean
                bb = _bb_current()
                if bb is not None:
                    leans.update(bb.desk_leans or {})
                desk_leans = leans or None
            except Exception:
                desk_leans = None
            ep = Episode(
                episode_id=f"{ticker}-{decision_ts}",
                symbol=ticker.upper(),
                decision_ts=decision_ts,
                state_summary=proposal.rationale[:1000],
                regime=Regime(regime) if regime else Regime.UNKNOWN,
                action_proposal=proposal.model_dump(),
                outcome_ts=outcome_dt.isoformat(),
                outcome_status=OutcomeStatus.PENDING,
                review_ts=review_dt.isoformat() if review_dt is not None else None,
                review_status="PENDING",
                review_embargo_days=review_embargo,
                realised_return=None,
                alpha_return=None,
                reflection=None,
                embargo_days=max(proposal.horizon_days, 1),
                crowd_score=crowd_score,
                crowd_components=crowd_components,
                workflow_version=wf_version,
                prompts_hash=wf_prompts_hash,
                code_git_sha=wf_git_sha,
                source=wf_source,
                desk_leans=desk_leans,
            )
            with _EPISODE_WRITE_LOCK:
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

    def _run_fusion_step(self, ticker: str, *, include_crowd: bool = True) -> None:
        """Optional in-run signal fusion (folded in from the old Signals tab).

        Combines this run's LLM decision (the episode just recorded) with the
        L3 expectancy edge and — when ``include_crowd`` — the
        live crowd-positioning signal into one guardrailed :class:`FusedSignal`,
        then emits a ``fusion`` event the Workflow tab renders next to the final
        decision.  Advisory only: it never changes the PM's action/size, it just
        surfaces cross-model agreement / disagreement.  Uses the cheap ``proxy``
        expectancy (mean resolved-episode edge) rather than a 5y PSO sweep so it
        stays fast enough to run inline.
        """
        try:
            from .fusion import assemble_fusion_inputs
            from trading_crew.agentic.signals import fuse_signals

            inputs, sources, err = assemble_fusion_inputs(
                ticker, include_crowd, expectancy_source="proxy"
            )
            if err:
                self._emit(make_event("fusion", available=False, note=err, sources=sources))
                return
            fused = fuse_signals(
                inputs["llm"],
                policy=inputs["policy"],
                expectancy=inputs["expectancy"],
                crowd=inputs["crowd"],
            )
            # Advisory context only: surface the reports desk's guidance-bias
            # alongside the fused signal so the operator can see whether the
            # fundamental/filing read AGREES with the cross-model fusion.  We
            # deliberately do NOT feed this into ``fuse_signals``: the fusion
            # layer is opt-in + advisory (it never changes the PM's action or
            # size) and is not yet validated to improve prediction/sizing, so
            # altering its tested math for no decision impact isn't worth the
            # risk.  The actionable guidance signal already reaches the PM /
            # dashboard via the ``report_signals`` event.
            guidance = None
            rec = self._run_record
            if rec is not None and rec.report_signals:
                guidance = {
                    "net_bias": rec.report_signals.get("net_bias"),
                    "bias_score": rec.report_signals.get("bias_score"),
                    "net_confidence": rec.report_signals.get("net_confidence"),
                }
            self._emit(make_event(
                "fusion",
                available=True,
                sources=sources,
                fused=fused.to_dict(),
                guidance=guidance,
            ))
        except Exception:  # noqa: BLE001 — fusion is advisory, never fail the run
            logger.exception("Signal fusion step failed (non-fatal)")

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
        # of this run.  The preset is a ContextVar that ``get_llm()``
        # consults BEFORE the env-var chain, so concurrent WS sessions
        # can each pick a different LLM without touching ``os.environ``.
        # Graph thread-pool workers copy it via
        # ``submit_in_active_context``.  Cleared in the matching
        # ``finally`` block.
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
                "run_mode": ui_cfg.get("run_mode") or "autonomous",
                "memory": ui_cfg.get("memory"),
                "quality_review": ui_cfg.get("quality_review"),
                "compliance": ui_cfg.get("compliance"),
                "tools_enabled": ui_cfg.get("tools_enabled"),
                "critic_enabled": ui_cfg.get("critic_enabled"),
                "critic_advisory": ui_cfg.get("critic_advisory", True),
                "fusion_enabled": ui_cfg.get("fusion_enabled", False),
                "fusion_include_crowd": ui_cfg.get("fusion_include_crowd", True),
                "critic_iterations": ui_cfg.get("critic_iterations"),
                "critic_samples": ui_cfg.get("critic_samples"),
                "llm_preset": ui_cfg.get("llm_preset"),
                "embedding_preset": ui_cfg.get("embedding_preset"),
            },
        )
        try:
            run_mode = str(ui_cfg.get("run_mode") or "autonomous").strip().lower()
            if run_mode not in ("autonomous", "deterministic", "harness"):
                raise ValueError(
                    f"run_mode must be autonomous, deterministic, or harness, not {run_mode!r}"
                )
            debate_rounds = int(ui_cfg.get("debate_rounds", 2))
            risk_rounds = int(ui_cfg.get("risk_rounds", 1))
            # Hermes playbook (debate_rounds_1/2) overrides the UI slider on
            # LangGraph and sequential crews. Harness has no round quota.
            if run_mode != "harness":
                from trading_crew.data_sources import is_indian_ticker
                from trading_crew.graph.routers import (
                    debate_rounds_from_action,
                    resolve_playbook_action,
                )

                _pb_market = "IN" if is_indian_ticker(ticker) else "US"
                debate_rounds = debate_rounds_from_action(
                    resolve_playbook_action(_pb_market), debate_rounds
                )
            # Memory defaults OFF now — CrewAI's per-task analyze passes were
            # the single biggest source of LLM calls (~80/run).  The toggle
            # is still exposed in the UI so a user can opt back in.
            use_memory = bool(ui_cfg.get("memory", False))
            tools_enabled = ui_cfg.get("tools_enabled") or {}
            # Optional advisory reviewers — default OFF for a lean run.
            quality_review_enabled = bool(ui_cfg.get("quality_review", False))
            compliance_enabled = bool(ui_cfg.get("compliance", False))
            asset_class = (ui_cfg.get("asset_class") or "stock").lower()
            self._asset_class = asset_class
            book = (ui_cfg.get("book") or "paper").strip().lower()
            if book not in ("paper", "prod"):
                book = "paper"
            self._book = book
            # Batch forward-collection stamps its episodes with a dedicated
            # source so the RL cohort stays clean (see scale_runner).
            self._episode_source = (ui_cfg.get("episode_source") or None)

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
            # Stock: autonomous = LangGraph; deterministic = sequential CrewAI;
            # harness = observe/dispatch loop. Commodity is always sequential.
            from trading_crew.graph.routers import use_stock_graph

            run_mode = str(ui_cfg.get("run_mode") or "autonomous").strip().lower()
            if run_mode not in ("autonomous", "deterministic", "harness"):
                raise ValueError(
                    f"run_mode must be autonomous, deterministic, or harness, not {run_mode!r}"
                )
            harness_mode = asset_class == "stock" and run_mode == "harness"
            graph_mode = use_stock_graph(asset_class, run_mode)
            self._graph_mode = graph_mode
            self._harness_mode = harness_mode
            self._run_mode = run_mode
            # Isolated one-task Crews never attach CrewAI Memory. The
            # toggle only applies to sequential ``crew()`` (deterministic
            # stock + commodity).
            if graph_mode or harness_mode:
                use_memory = False
            if self._run_record is not None:
                self._run_record.config["memory"] = use_memory
            self._graph_edges = []
            init: Dict[str, Any] = {}
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
                crew = tc.crew()
                self._crew = crew
                self._pm_task = getattr(tc, "_pm_task", None)
                self._graph_agents = {}
            elif graph_mode or harness_mode:
                tc = TradingCrew(
                    ticker=ticker,
                    debate_rounds=debate_rounds,
                    risk_rounds=risk_rounds,
                    step_callback=self._on_step,
                    task_callback=self._on_task,
                    memory=use_memory,
                    tools_enabled=tools_enabled,
                    quality_review_enabled=quality_review_enabled,
                    compliance_enabled=compliance_enabled,
                    run_mode=run_mode,
                )
                agents = tc.materialize_agents()
                self._graph_agents = agents
                self._pm_task = getattr(tc, "_pm_task", None)
                crew = type(
                    "GraphCrew",
                    (),
                    {"agents": list(agents.values()), "tasks": []},
                )()
                self._crew = crew
            else:
                tc = TradingCrew(
                    ticker=ticker,
                    debate_rounds=debate_rounds,
                    risk_rounds=risk_rounds,
                    step_callback=self._on_step,
                    task_callback=self._on_task,
                    memory=use_memory,
                    tools_enabled=tools_enabled,
                    quality_review_enabled=quality_review_enabled,
                    compliance_enabled=compliance_enabled,
                    run_mode=run_mode,
                )
                crew = tc.crew()
                self._crew = crew
                self._graph_agents = {}
                self._pm_task = getattr(tc, "_pm_task", None)
            self._pm_ledger_injected = False
            self._pm_regime_injected = False
            self._pm_price_injected = False
            self._price_structure = None
            self._desk_biases = {}
            self._pm_bias_board_injected = False
            self._debate_r1_completed = set()
            try:
                from trading_crew.agentic.graph.blackboard import begin_run
                from trading_crew.agentic.graph.scratchpad import begin_run as begin_pad
                begin_run(ticker)
                begin_pad(ticker)
            except Exception:
                logger.warning("evidence blackboard failed to start", exc_info=True)
            # Record our own agent ids so ``_on_tool_*`` can reject
            # events from a *sibling* run sharing the CrewAI singleton
            # event bus.  Agents get a UUID at construction time so
            # even two runs on the same ticker have disjoint sets.
            try:
                self._own_agent_ids = {
                    str(getattr(a, "id", None) or getattr(a, "_id", None) or id(a))
                    for a in getattr(crew, "agents", []) or []
                }
            except Exception:
                self._own_agent_ids = set()
            if self._run_record is not None:
                self._run_record.config["asset_class"] = asset_class
                self._run_record.config["run_mode"] = run_mode
                self._run_record.config["orchestrator"] = (
                    "harness" if harness_mode else ("langgraph" if graph_mode else "crewai")
                )
            # Precompute expected role ordering + async flags so we can
            # synthesize node_started events while sequential agents are
            # mid-LLM-call AND pre-announce contiguous async batches
            # (so out-of-order completions inside a batch never fire a
            # DONE without a prior START).
            if graph_mode:
                from trading_crew.data_sources import is_indian_ticker
                from trading_crew.graph.app import initial_state
                from trading_crew.graph.routers import preview_role_order

                market = "IN" if is_indian_ticker(ticker) else "US"
                self._graph_market = market
                init = initial_state(
                    ticker,
                    ui_debate_rounds=debate_rounds,
                    ui_risk_rounds=risk_rounds,
                    quality_review_enabled=quality_review_enabled,
                    compliance_enabled=compliance_enabled,
                    market=market,
                )
                self._expected_role_order = preview_role_order(
                    skip_reports=bool(init["skip_reports"]),
                    debate_rounds_target=int(init["debate_rounds_target"]),
                    allow_extra_debate=False,
                    allow_trader_reopen=False,
                    risk_rounds=int(init.get("ui_risk_rounds") or risk_rounds),
                    quality_review=quality_review_enabled,
                    compliance=compliance_enabled,
                )
                self._expected_async_flags = [
                    r in self._GRAPH_ASYNC_ROLES for r in self._expected_role_order
                ]
            elif harness_mode:
                self._expected_role_order = [
                    "Conductor",
                    "Market Analyst",
                    "News Analyst",
                    "Fundamentals Analyst",
                    "Reports Analyst",
                    "Macro Analyst",
                    "Bullish Researcher",
                    "Bearish Researcher",
                    "Research Manager",
                    "Trader",
                    "Aggressive Risk Analyst",
                    "Conservative Risk Analyst",
                    "Neutral Risk Analyst",
                    "Portfolio Manager",
                    "Investigator",
                ]
                self._expected_async_flags = [True] * len(self._expected_role_order)
            else:
                self._expected_role_order = [
                    getattr(t.agent, "role", "Agent") for t in crew.tasks
                ]
                self._expected_async_flags = [
                    bool(getattr(t, "async_execution", False)) for t in crew.tasks
                ]
            # Pre-compute the round number AND total-rounds for every
            # task slot so ``node_started`` / ``node_completed`` events
            # can carry ``round`` / ``total_rounds`` fields.  For roles
            # that fire once (analysts, manager, trader, compliance,
            # PM) round=1, total_rounds=1.  For multi-round roles
            # (Bullish / Bearish researcher, the three Risk personas)
            # the slot's position among same-role siblings becomes
            # round_num, and ``total_rounds`` is just how many times
            # the role appears in expected_role_order.
            role_seen_count: Dict[str, int] = {}
            self._role_round_per_slot: List[int] = []
            for r in self._expected_role_order:
                role_seen_count[r] = role_seen_count.get(r, 0) + 1
                self._role_round_per_slot.append(role_seen_count[r])
            self._role_total_rounds: Dict[str, int] = {
                r: self._expected_role_order.count(r)
                for r in set(self._expected_role_order)
            }
            # Reset per-run slot tracking (in case the runner instance is
            # reused across analyses on the same WS session).
            self._completed_slot_indices = set()
            self._completed_count = 0
            self._report_signals_emitted = False
            logger.info("Expected role order: %s", self._expected_role_order)
            if self._run_record is not None:
                self._run_record.expected_role_order = list(self._expected_role_order)
                if graph_mode and init:
                    self._run_record.config["playbook_action"] = init.get("playbook_action")
                    self._run_record.config["skip_reports"] = init.get("skip_reports")

            self._emit(make_event(
                "run_started",
                ticker=ticker,
                trade_date=self._cache_meta.get("trade_date"),
                debate_rounds=None if harness_mode else debate_rounds,
                risk_rounds=None if harness_mode else risk_rounds,
                agent_count=len(getattr(crew, "agents", []) or []),
                task_count=len(self._expected_role_order),
                memory=use_memory,
                expected_order=self._expected_role_order,
                tools_enabled=tools_enabled,
                orchestrator=(
                    "harness" if harness_mode else ("langgraph" if graph_mode else "crewai")
                ),
                run_mode=run_mode,
                quality_review=quality_review_enabled,
                compliance=compliance_enabled,
                playbook_action=(init.get("playbook_action") if graph_mode else None),
                skip_reports=(init.get("skip_reports") if graph_mode else None),
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
            # Direct signal → PM: the deterministic regime is known before the
            # debate fires; hand it straight to the decision desk (see method).
            self._inject_pm_regime(regime_label)

            # CRISIS short-circuit: skip the 18-agent debate entirely and
            # build a synthetic ABSTAIN decision. The rest of the M1-M5
            # pipeline (sizer, risk gate, simulator, episodic memory)
            # still runs so the audit trail is identical to a normal run.
            short_circuited = cascade_route == "CRISIS_OVERRIDE"
            result = None
            if short_circuited:
                # Even though the debate is skipped, still compute + emit the
                # deterministic price-structure card so the Workflow tab / UI
                # isn't blank for the very names most likely to be oversold
                # (vol-state, regime age, DIP_BUY).  Emit-only here — there's no
                # PM prompt to prepend onto on the override path.
                try:
                    self._inject_pm_price_structure()
                except Exception:  # noqa: BLE001 — advisory, never break the override
                    logger.debug("price-structure emit on crisis override failed", exc_info=True)
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
                    if harness_mode:
                        from trading_crew.harness import run_harness

                        def _harness_event(typ: str, **kw: Any) -> None:
                            self._emit(make_event(typ, **kw))

                        result = run_harness(
                            tc,
                            self._graph_agents,
                            step_callback=self._on_step,
                            task_callback=self._on_task,
                            on_edge=self._on_graph_edge,
                            on_event=_harness_event,
                            should_stop=self._cancel_event.is_set,
                        )
                        self._emit(make_event("graph_trace", edges=list(self._graph_edges)))
                    elif graph_mode:
                        from trading_crew.graph import run_stock_graph

                        result = run_stock_graph(
                            tc,
                            self._graph_agents,
                            step_callback=self._on_step,
                            task_callback=self._on_task,
                            on_edge=self._on_graph_edge,
                            ui_debate_rounds=debate_rounds,
                            ui_risk_rounds=risk_rounds,
                            quality_review_enabled=quality_review_enabled,
                            compliance_enabled=compliance_enabled,
                            market=getattr(self, "_graph_market", ""),
                        )
                        self._emit(make_event("graph_trace", edges=list(self._graph_edges)))
                    else:
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
                    raw=_safe_text(result, limit=None),
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
                    # Normalize provenance tags to their DATA PROVIDER (screener.in
                    # / yahoo / …) on BOTH the rationale prose and the sources list
                    # in one deterministic pass, so the persisted decision card and
                    # the critic's evidence check stay mutually consistent.
                    try:
                        _dec_update: dict[str, Any] = {}
                        _rat = getattr(pm_decision, "rationale", None)
                        if isinstance(_rat, str) and _rat:
                            _dec_update["rationale"] = normalize_source_providers(_rat)
                        _srcs = getattr(pm_decision, "sources", None)
                        if isinstance(_srcs, list) and _srcs:
                            _dec_update["sources"] = [
                                normalize_source_identifier(s) if isinstance(s, str) else s
                                for s in _srcs
                            ]
                        if _dec_update:
                            pm_decision = pm_decision.model_copy(update=_dec_update)
                    except Exception:
                        logger.exception("source-provider normalization of PM decision failed; using raw")
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
                        # Critic is on by default but SINGLE-PASS now: the
                        # 3-temperature consistency vote defaulted to 3
                        # extra LLM calls per run for marginal benefit, so
                        # ``critic_samples`` defaults to 1.  A user can
                        # disable the critic entirely via ``critic_enabled``.
                        if not bool(ui_cfg.get("critic_enabled", True)):
                            raise RuntimeError("critic disabled by config")
                        critic_llm = get_llm(temperature=0.0)
                        critiqued, reflection_records = run_reflective_critic(
                            pm_decision, ticker=ticker, llm=critic_llm,
                            max_iterations=int(ui_cfg.get("critic_iterations", 2)),
                            consistency_samples=int(ui_cfg.get("critic_samples", 1)),
                        )
                        # ADVISORY-ONLY (default): the critic is a *one-directional*
                        # operator — it can only hold, downgrade, cut size or
                        # ABSTAIN, never upgrade — so letting it override the PM
                        # both (a) flips the decision at the last moment and
                        # (b) biases the whole book toward HOLD/SELL.  In advisory
                        # mode we still run the full audit (and surface its verdict
                        # in the UI) but we KEEP the PM's action/size, overriding
                        # ONLY on a hard integrity failure: compliance BLOCKED, or
                        # a fabricated/placeholder source in the decision.  Set
                        # ``critic_advisory=false`` to restore the old override.
                        advisory = bool(ui_cfg.get("critic_advisory", True))
                        final_dec, apply_override, hard_fail_reason = _resolve_critic_outcome(
                            pm_decision, critiqued, advisory
                        )
                        changed = (
                            critiqued.action != pm_decision.action
                            or critiqued.size_pct_of_book != pm_decision.size_pct_of_book
                        )
                        if reflection_records:
                            self._emit(make_event(
                                "reflection_records",
                                records=records_to_payload(reflection_records),
                                final_action=final_dec.action,
                                final_size_pct=final_dec.size_pct_of_book,
                                revised=bool(apply_override and changed),
                                advisory=advisory,
                                override_reason=hard_fail_reason,
                            ))
                        # Only re-emit final_decision (the visible "flip") when we
                        # actually apply the override — i.e. never in advisory mode
                        # unless a hard integrity failure was found.
                        if apply_override and changed:
                            self._emit(make_event(
                                "final_decision",
                                decision=critiqued.model_dump(),
                                raw=critiqued.rationale,
                                source="critic",
                            ))
                        pm_decision = final_dec
                    except Exception as critic_exc:
                        # Intentional skips (cascade override, critic
                        # disabled) pass the PM decision through untouched
                        # without a scary traceback in the logs.
                        msg = str(critic_exc)
                        if short_circuited or "critic disabled" in msg:
                            logger.info("Reflective critic skipped: %s", msg)
                        else:
                            logger.exception("Reflective critic failed; using PM decision unchanged")

                    # Desk-board / computed-trend utilization (after the
                    # advisory critic so the audit is of the PM's own call,
                    # then we enforce disagreement the LLM can ignore).
                    try:
                        if not short_circuited:
                            pm_decision = self._apply_signal_alignment(pm_decision)
                    except Exception:  # noqa: BLE001 — never fail the run
                        logger.exception("Signal alignment failed (non-fatal)")

                    # Reuse the OHLCV that the chart UI already fetched.
                    try:
                        end_dt = datetime.fromisoformat(decision_ts.replace("Z", "+00:00")).replace(tzinfo=None)
                    except Exception:
                        end_dt = datetime.utcnow()
                    ohlcv = _fetch_ohlcv(ticker, end_dt, lookback_days=365)

                    # Deterministic stop sanity: widen an implausibly tight /
                    # wrong-side / missing stop to a volatility+horizon floor
                    # before the bridge freezes the levels. Runs on the
                    # post-critic decision so the UI + episode see the grounded
                    # entry/stop rather than the LLM's free-form guess.
                    pm_decision = self._normalize_stops(ticker, pm_decision, ohlcv)
                    pm_decision = self._apply_hold_watch_plan(
                        ticker, pm_decision, ohlcv, decision_ts,
                    )

                    proposal = portfolio_decision_to_action_proposal(
                        pm_decision,
                        symbol=ticker,
                        decision_ts=decision_ts,
                    )
                    self._run_execution_pipeline(ticker, proposal, ohlcv)
                    self._record_episode(ticker, proposal, decision_ts, regime_label)
                    # Optional in-run signal fusion (opt-in, like the reviewers /
                    # critic).  Runs after the episode is recorded so it fuses
                    # THIS decision with the L3 edge / crowd view.
                    if bool(ui_cfg.get("fusion_enabled", False)):
                        self._run_fusion_step(
                            ticker,
                            include_crowd=bool(ui_cfg.get("fusion_include_crowd", True)),
                        )
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
            try:
                from trading_crew.agentic.graph.blackboard import end_run
                from trading_crew.agentic.graph.scratchpad import end_run as end_pad
                end_run()
                end_pad()
            except Exception:
                pass
            # Release this run's CrewAI Memory (LanceDB + save pool) and hand
            # the freed heap back to the OS so a long batch's RSS stays flat.
            try:
                from trading_crew.memutil import close_crew_memory, release_memory
                close_crew_memory(getattr(self, "_crew", None))
                self._crew = None
                release_memory()
            except Exception:
                logger.debug("post-run memory release failed (non-fatal)", exc_info=True)

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
        ("news_analyst", "News Analyst", "analyst"),
        ("fundamentals_analyst", "Fundamentals Analyst", "analyst"),
        ("reports_analyst", "Reports Analyst", "analyst"),
        ("macro_analyst", "Macro Analyst", "analyst"),
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
