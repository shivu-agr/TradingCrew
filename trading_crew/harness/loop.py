"""Observe → dispatch → LLM until finish or budget."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from trading_crew._common import get_llm
from trading_crew.graph.nodes import ANALYST_KEYS, GraphRuntime
from trading_crew.horizons import Horizon
from trading_crew.schemas import HorizonForecasts, PortfolioDecision

from .conductor import SpawnSpec, build_closing_prompt, build_system_prompt, next_turn
from .config import HarnessConfigError, load_harness_config, load_standing_memory
from .ingest import ingest_corpus
from .safety import SKEPTICAL_TOOLS, coverage_gaps, HarnessGateError
from .spawn import spawn_many, spawn_one
from .state import HarnessState, is_context_overflow
from .tools import family_for, invoke_tool

logger = logging.getLogger(__name__)


@dataclass
class HarnessKickoffResult:
    """CrewAI-shaped result so the runner M1–M7 path stays intact."""

    pydantic: Any = None
    raw: str = ""
    state: Optional[HarnessState] = None

    def __str__(self) -> str:
        return self.raw or ""


def _stance_matches_action(stance: str, action: str) -> bool:
    side = (action or "").upper()
    st = (stance or "").lower()
    if st == "bullish":
        return side in ("BUY", "STRONG_BUY")
    if st == "bearish":
        return side in ("SELL", "STRONG_SELL")
    if st in ("mixed", "neutral"):
        return side == "HOLD"
    return False


def _primary_for_action(hf: HorizonForecasts, action: str) -> Optional[Horizon]:
    """Pick a horizon whose stance is allowed to pair with ``action``."""
    side = (action or "").upper()
    if side in ("BUY", "STRONG_BUY"):
        want = {"bullish"}
    elif side in ("SELL", "STRONG_SELL"):
        want = {"bearish"}
    elif side == "HOLD":
        want = {"mixed", "neutral"}
    else:
        return None
    for horizon in (Horizon.MEDIUM, Horizon.LONG, Horizon.SHORT):
        if getattr(hf, horizon.value).stance in want:
            return horizon
    return None


_CLOSING_PM_PREFIX = (
    "CLOSING. The investigation step budget is spent. Write the book ticket "
    "from the desk reports below. If bull/bear or horizon stances conflict, "
    "HOLD with size 0 is allowed. rationale, key_drivers, key_risks, and "
    "falsifiers must cite those specific conflicts. Do not write that the "
    "step budget failed as the investment thesis."
)


def _excerpt(text: str, n: int = 280) -> str:
    return " ".join((text or "").split())[:n]


def _exhausted_hold(ticker: str, max_steps: int) -> PortfolioDecision:
    """Last-resort ticket when no desk reported and PM never parsed."""
    return PortfolioDecision(
        action="HOLD",
        confidence=0.4,
        size_pct_of_book=0.0,
        entry_price=0.0,
        stop_loss=0.0,
        target_price=0.0,
        horizon_days=21,
        expected_return_pct=0.0,
        rationale=(
            f"Harness used all {max_steps} steps on {ticker} without a gated "
            "finish, and the portfolio manager did not emit a valid ticket. "
            "HOLD with size 0 so the run still returns a book response."
        ),
        key_drivers=["harness step budget exhausted"],
        key_risks=["ticket written without a gated finish"],
        falsifiers=["a later gated finish with matching horizon stance"],
        geopolitical_flags=[],
        compliance_status="CLEAR",
        long_term_view=(
            "Long-term view not locked — harness returned after max_steps."
        ),
    )


def _hold_from_reports(
    ticker: str,
    max_steps: int,
    state: HarnessState,
) -> PortfolioDecision:
    """HOLD that cites desk reports instead of a canned budget stub."""
    excerpts: list[str] = []
    drivers: list[str] = []
    risks: list[str] = []
    for key, text in state.reports.items():
        body = (text or "").strip()
        if not body:
            continue
        excerpts.append(f"- {key}: {_excerpt(body)}")
        drivers.append(f"{key}: {_excerpt(body, 80)}")
        risks.append(f"{key} conflict unresolved")
    if not excerpts:
        return _exhausted_hold(ticker, max_steps)
    rationale = (
        f"After {max_steps} investigation steps on {ticker}, the desks that "
        f"reported did not converge on one side:\n"
        + "\n".join(excerpts[:12])
        + "\n\nHOLD, size 0. The conflict above was not reconciled into a "
        "directional primary ticket."
    )
    return PortfolioDecision(
        action="HOLD",
        confidence=0.4,
        size_pct_of_book=0.0,
        entry_price=0.0,
        stop_loss=0.0,
        target_price=0.0,
        horizon_days=21,
        expected_return_pct=0.0,
        rationale=rationale,
        key_drivers=drivers[:8] or ["desks did not converge"],
        key_risks=risks[:8] or ["unresolved desk conflict"],
        falsifiers=["a later gated finish with matching horizon stance"],
        geopolitical_flags=[],
        compliance_status="CLEAR",
        long_term_view=(
            "Long-term view not locked — desks remained conflicted after "
            "the step budget."
        ),
    )


def _attach_forecasts(
    pm: PortfolioDecision,
    hf: Optional[HorizonForecasts],
    primary: Optional[Horizon],
) -> PortfolioDecision:
    chosen = primary or Horizon.MEDIUM
    if hf is not None:
        if not _stance_matches_action(getattr(hf, chosen.value).stance, pm.action):
            chosen = _primary_for_action(hf, pm.action) or chosen
        return pm.model_copy(update={"horizon_forecasts": hf, "horizon": chosen})
    return pm.model_copy(update={"horizon": chosen})


def _copied_forecasts(hf: HorizonForecasts) -> bool:
    short_d = tuple(hf.SHORT.drivers)
    med_d = tuple(hf.MEDIUM.drivers)
    long_d = tuple(hf.LONG.drivers)
    if short_d and short_d == med_d == long_d:
        return True
    if hf.SHORT.stance == hf.MEDIUM.stance == hf.LONG.stance and short_d == med_d:
        return True
    return False


def _analyst_reports(state: HarnessState) -> Dict[str, str]:
    return {k: state.reports.get(k, "") for k in ANALYST_KEYS}


def _risk_notes(state: HarnessState) -> Dict[str, str]:
    return {
        "Aggressive Risk Analyst": state.reports.get("risk_aggressive", ""),
        "Conservative Risk Analyst": state.reports.get("risk_conservative", ""),
        "Neutral Risk Analyst": state.reports.get("risk_neutral", ""),
    }


def _pm_thesis(state: HarnessState, *, closing: bool) -> str:
    thesis = state.reports.get("research_manager", "")
    if not closing:
        return thesis
    return f"{_CLOSING_PM_PREFIX}\n\n{thesis}".strip()


def _call_pm(runtime: GraphRuntime, state: HarnessState, *, closing: bool) -> str:
    return runtime.run_pm(
        reports=_analyst_reports(state),
        thesis=_pm_thesis(state, closing=closing),
        trade_plan=state.reports.get("trader", ""),
        risk_notes=_risk_notes(state),
    )


def _read_turn(
    conductor_llm: Any,
    system: str,
    state: HarnessState,
    cfg: Dict[str, Any],
    *,
    closing: bool = False,
) -> Any:
    transcript = state.render(int(cfg["context_chars"]))
    try:
        return next_turn(conductor_llm, system, transcript, closing=closing)
    except Exception as exc:
        if is_context_overflow(exc):
            saved = state.compact_in_place()
            state.add_observation(
                "compact",
                "context overflow",
                f"Conductor prompt overflowed; compacted {saved} chars and retrying.",
            )
            try:
                transcript = state.render(int(cfg["context_chars"]))
                return next_turn(conductor_llm, system, transcript, closing=closing)
            except Exception:
                logger.exception("conductor turn failed after compact")
                state.add_observation(
                    "error",
                    "conductor parse",
                    "Could not parse a JSON turn after compact. Retry.",
                )
                return None
        logger.exception("conductor turn failed")
        state.add_observation(
            "error",
            "conductor parse",
            "Could not parse a JSON turn. Retry.",
        )
        return None


def _apply_finish(
    runtime: GraphRuntime,
    state: HarnessState,
    turn: Any,
    cfg: Dict[str, Any],
    *,
    closing: bool,
) -> Dict[str, Any]:
    """Run PM on a finish turn. Closing skips coverage/copy gates and rematches stance."""
    gaps: list[str] = []
    if not closing:
        gaps = coverage_gaps(
            families=state.families,
            skeptical=state.skeptical,
            spawned=state.spawned,
            reports=state.reports,
            tool_names=state.tool_names,
            steps=state.steps,
            cfg=cfg,
        )
    hf = turn.horizon_forecasts
    if hf is None:
        gaps.append("finish requires horizon_forecasts for SHORT, MEDIUM, and LONG")
    elif not closing and _copied_forecasts(hf):
        gaps.append(
            "horizon_forecasts look copied across tiers — write independent "
            "drivers and allow stances to disagree"
        )
    primary = turn.primary_horizon or Horizon.MEDIUM
    if gaps:
        state.add_observation("gate", "cannot finish", "; ".join(gaps))
        return {"retry": True}
    try:
        raw = _call_pm(runtime, state, closing=closing)
    except Exception as exc:
        logger.exception("harness PM failed")
        state.add_observation("error", "portfolio_manager", str(exc))
        return {"retry": True}
    pm = runtime.last_pydantic
    if not isinstance(pm, PortfolioDecision):
        state.add_observation(
            "gate",
            "portfolio_manager",
            "PM did not emit a valid PortfolioDecision; gather more evidence or retry finish",
        )
        return {"retry": True}
    raw = raw or pm.rationale
    if not _stance_matches_action(getattr(hf, primary.value).stance, pm.action):
        if closing:
            rematch = _primary_for_action(hf, pm.action) or primary
            decision = pm.model_copy(
                update={"horizon_forecasts": hf, "horizon": rematch}
            )
            return {
                "retry": False,
                "decision": decision,
                "last_pm": pm,
                "last_pm_raw": raw or decision.rationale,
                "last_hf": hf,
                "last_primary": rematch,
                "raw": raw or decision.rationale,
            }
        state.add_observation(
            "gate",
            "horizon mismatch",
            f"primary {primary.value} stance is "
            f"{getattr(hf, primary.value).stance} but PM action is {pm.action}. "
            "Reconcile or change primary_horizon.",
        )
        return {
            "retry": True,
            "last_pm": pm,
            "last_pm_raw": raw,
            "last_hf": hf,
            "last_primary": primary,
        }
    decision = pm.model_copy(update={"horizon_forecasts": hf, "horizon": primary})
    return {
        "retry": False,
        "decision": decision,
        "last_pm": pm,
        "last_pm_raw": raw or decision.rationale,
        "last_hf": hf,
        "last_primary": primary,
        "raw": raw or decision.rationale,
    }


def run_harness(
    trading_crew: Any,
    agents: Dict[str, Any],
    *,
    step_callback: Optional[Callable] = None,
    task_callback: Optional[Callable] = None,
    on_edge: Optional[Callable[[str, str, str], None]] = None,
    on_event: Optional[Callable[..., None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    llm: Any = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> HarnessKickoffResult:
    cfg = cfg or load_harness_config()
    standing = load_standing_memory(cfg)
    ingest_corpus(cfg["corpus_dir"])

    ticker = trading_crew.ticker
    state = HarnessState(ticker=ticker)

    def emit_edge(src: str, dst: str, kind: str = "activated") -> None:
        state.add_edge(src, dst, kind)
        if on_edge:
            on_edge(src, dst, kind)

    runtime = GraphRuntime(
        trading_crew,
        agents,
        step_callback=step_callback,
        task_callback=task_callback,
        on_edge=emit_edge,
    )
    conductor_llm = llm or get_llm(temperature=float(cfg["conductor_temperature"]))

    if on_event:
        on_event("node_started", node="Conductor", role="Conductor", kind="router")
    emit_edge("Conductor", "Shared State", "activated")

    try:
        brief = invoke_tool("get_market_context", {"ticker": ticker}, cfg=cfg)
    except HarnessGateError as exc:
        raise HarnessConfigError(str(exc)) from exc
    state.tool_names.append("get_market_context")
    fam = family_for("get_market_context")
    if fam:
        state.families.add(fam)
    state.add_observation("tool", "get_market_context", brief)

    max_steps = int(cfg["max_steps"])
    decision: Optional[PortfolioDecision] = None
    raw = ""
    last_pm: Optional[PortfolioDecision] = None
    last_pm_raw = ""
    last_hf: Optional[HorizonForecasts] = None
    last_primary: Optional[Horizon] = None

    for step in range(max_steps):
        if should_stop and should_stop():
            raise HarnessConfigError("harness run cancelled")
        state.steps = step + 1
        system = build_system_prompt(
            ticker=ticker,
            standing=standing,
            allowlist=list(cfg["tool_allowlist"]),
            catalog=list(cfg["catalog_desks"]),
            spawn_left=max(0, int(cfg["max_spawn"]) - state.spawn_used),
            steps_left=max_steps - step,
            coverage=state.coverage_board(),
        )
        turn = _read_turn(conductor_llm, system, state, cfg)
        if turn is None:
            continue

        logger.info(
            "harness step %s action=%s tool=%s spawn=%s",
            state.steps,
            turn.action,
            turn.tool,
            [a.desk or a.kind for a in turn.agents],
        )
        print(
            f"[harness] step {state.steps} {turn.action} "
            f"tool={turn.tool or '-'} spawn={[a.desk or a.kind for a in turn.agents]}",
            flush=True,
        )
        thought = (turn.thought or "").strip()
        if thought:
            state.add_observation("thought", f"step {state.steps}", thought)

        if turn.action == "compact":
            saved = state.compact_in_place()
            state.add_observation(
                "compact",
                f"step {state.steps}",
                f"Compacted earlier observations ({saved} chars saved). Continue the investigation.",
            )
            continue

        if turn.action == "call_tool":
            name = (turn.tool or "").strip()
            args = dict(turn.arguments or {})
            args.setdefault("ticker", ticker)
            try:
                out = invoke_tool(name, args, cfg=cfg)
            except HarnessGateError as exc:
                state.add_observation("gate", f"refused {name}", str(exc))
                continue
            except Exception as exc:
                logger.exception("tool %s failed", name)
                state.add_observation("error", name, str(exc))
                continue
            state.tool_names.append(name)
            fam = family_for(name)
            if fam:
                state.families.add(fam)
            if name in SKEPTICAL_TOOLS:
                state.skeptical = True
            emit_edge("Conductor", name, "activated")
            state.add_observation("tool", name, out)
            continue

        if turn.action in ("spawn", "spawn_parallel"):
            specs = list(turn.agents or [])
            if turn.action == "spawn" and not specs:
                desk = (turn.tool or "").strip()
                if desk:
                    specs = [SpawnSpec(kind="desk", desk=desk, brief=turn.notes)]
            if not specs:
                state.add_observation("gate", "spawn", "spawn requires agents[]")
                continue
            prior = state.render(int(cfg["context_chars"]))
            try:
                if turn.action == "spawn_parallel" and len(specs) > 1:
                    results = spawn_many(
                        specs,
                        runtime=runtime,
                        state=state,
                        cfg=cfg,
                        depth=0,
                        prior=prior,
                    )
                else:
                    results = [
                        spawn_one(
                            spec,
                            runtime=runtime,
                            state=state,
                            cfg=cfg,
                            depth=0,
                            prior=prior,
                        )
                        for spec in specs[: int(cfg["max_parallel"])]
                    ]
            except HarnessGateError as exc:
                state.add_observation("gate", "spawn refused", str(exc))
                continue
            except Exception as exc:
                logger.exception("spawn failed")
                state.add_observation("error", "spawn", str(exc))
                continue
            for key, text in results:
                state.add_observation("spawn", key, text)
            continue

        if turn.action == "finish":
            outcome = _apply_finish(runtime, state, turn, cfg, closing=False)
            if outcome.get("last_pm") is not None:
                last_pm = outcome["last_pm"]
                last_pm_raw = outcome.get("last_pm_raw") or last_pm_raw
                last_hf = outcome.get("last_hf") or last_hf
                last_primary = outcome.get("last_primary") or last_primary
            if outcome.get("retry"):
                continue
            decision = outcome["decision"]
            raw = outcome.get("raw") or decision.rationale
            break

        state.add_observation("gate", "unknown action", f"{turn.action}")

    closing_used = 0
    exhaust_steps = int(cfg["exhaust_steps"])
    if decision is None and exhaust_steps > 0:
        closing_compacted = False
        for i in range(exhaust_steps):
            if should_stop and should_stop():
                raise HarnessConfigError("harness run cancelled")
            closing_used += 1
            state.steps = max_steps + closing_used
            system = build_closing_prompt(
                ticker=ticker,
                standing=standing,
                coverage=state.coverage_board(),
                steps_left=exhaust_steps - i,
            )
            turn = _read_turn(conductor_llm, system, state, cfg, closing=True)
            if turn is None:
                continue
            logger.info(
                "harness closing %s action=%s",
                closing_used,
                turn.action,
            )
            print(
                f"[harness] closing {closing_used}/{exhaust_steps} {turn.action}",
                flush=True,
            )
            thought = (turn.thought or "").strip()
            if thought:
                state.add_observation(
                    "thought",
                    f"closing {closing_used}",
                    thought,
                )
            if turn.action == "compact":
                if closing_compacted:
                    state.add_observation(
                        "gate",
                        "closing",
                        "compact already used; finish with horizon_forecasts",
                    )
                    continue
                closing_compacted = True
                saved = state.compact_in_place()
                state.add_observation(
                    "compact",
                    f"closing {closing_used}",
                    f"Compacted earlier observations ({saved} chars saved). Finish now.",
                )
                continue
            if turn.action in ("spawn", "spawn_parallel", "call_tool"):
                state.add_observation(
                    "gate",
                    "closing",
                    f"{turn.action} is closed; finish with horizon_forecasts "
                    "from the desks that already ran",
                )
                continue
            if turn.action == "finish":
                outcome = _apply_finish(runtime, state, turn, cfg, closing=True)
                if outcome.get("last_pm") is not None:
                    last_pm = outcome["last_pm"]
                    last_pm_raw = outcome.get("last_pm_raw") or last_pm_raw
                    last_hf = outcome.get("last_hf") or last_hf
                    last_primary = outcome.get("last_primary") or last_primary
                if outcome.get("retry"):
                    continue
                decision = outcome["decision"]
                raw = outcome.get("raw") or decision.rationale
                state.add_observation(
                    "gate",
                    "closing finish",
                    "accepted ticket from the desks that ran after the step budget",
                )
                break
            state.add_observation("gate", "unknown action", f"{turn.action}")

    if decision is None:
        if last_pm is None:
            try:
                last_pm_raw = _call_pm(runtime, state, closing=True)
                if isinstance(runtime.last_pydantic, PortfolioDecision):
                    last_pm = runtime.last_pydantic
            except Exception:
                logger.exception("harness PM failed on exhausted finish")
        if last_pm is None:
            last_pm = _hold_from_reports(ticker, max_steps, state)
            last_pm_raw = last_pm.rationale
        if closing_used:
            note = (
                f"[harness] closed after {max_steps} investigation steps plus "
                f"{closing_used} closing turns without a gated finish. "
                "Ticket is from the desks that ran."
            )
        else:
            note = (
                f"[harness] returned after {max_steps} steps without a gated finish."
            )
        decision = _attach_forecasts(last_pm, last_hf, last_primary)
        if note not in (decision.rationale or ""):
            decision = decision.model_copy(
                update={"rationale": f"{(decision.rationale or '').rstrip()}\n\n{note}"}
            )
        raw = last_pm_raw or decision.rationale
        state.add_observation("gate", "budget exhausted", note)
        logger.warning(
            "harness exhausted %s steps for %s — returning %s",
            max_steps,
            ticker,
            decision.action,
        )

    if on_event:
        on_event(
            "node_completed",
            node="Conductor",
            role="Conductor",
            kind="router",
            output=raw or "harness loop ended",
        )

    return HarnessKickoffResult(pydantic=decision, raw=raw, state=state)
