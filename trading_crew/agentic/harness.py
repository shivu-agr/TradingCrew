"""Hermes prediction harness — one experiment after Resolve.

Hermes owns selection: desk_prompt, topology, or synthesis prompt, chosen
from Resolve evidence (hurt desks, split boards, hit-rate vs alpha/DD).
The apply path is unchanged (``apply_harness`` + graph router).

Mass 1PO/2TO/3PO and ``promote_graph.py`` are **legacy** — still imported
so an existing ``promoted.json`` can apply, but they are not the live
training step.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from trading_crew.agentic.mass import (  # legacy apply-only
    load_promoted,
    load_promoted_prompts,
)
from trading_crew.agentic.objectives import review_payload
from trading_crew.agentic.self_learn import (
    _hypothesis_for,
    active_experiment,
    dashboard_payload as _scoreboard,
    journal_path,
    load_journal,
    save_journal,
    step_self_learn,
)
from trading_crew.agentic.strategy import (
    StrategyError,
    apply_strategy_backstory,
    load_desk_strategy,
    render_prompt_block,
)

logger = logging.getLogger(__name__)

SAMPLE_PREFIX = "SAMPLE-"

_DESK_ROLE_NEEDLES = (
    ("curve_analyst", ("term structure", "curve analyst")),
    ("inventories_analyst", ("inventories", "inventory")),
    ("supply_demand_analyst", ("supply & demand", "supply and demand")),
    ("geopolitical_analyst", ("geopolitical",)),
    ("quant_analyst", ("seasonality quant", "positioning & seasonality")),
    ("quality_reviewer", ("quality review",)),
    ("compliance_officer", ("compliance officer",)),
    ("market_analyst", ("market analyst",)),
    ("news_analyst", ("news analyst",)),
    ("fundamentals_analyst", ("fundamentals analyst",)),
    ("reports_analyst", ("reports analyst",)),
    ("macro_analyst", ("macro analyst",)),
    ("bull_researcher", ("bullish researcher", "bullish futures")),
    ("bear_researcher", ("bearish researcher", "bearish futures")),
    ("research_manager", ("research manager",)),
    ("trader", ("trader",)),
    ("risk_aggressive", ("aggressive risk",)),
    ("risk_conservative", ("conservative risk",)),
    ("risk_neutral", ("neutral risk",)),
    ("portfolio_manager", ("portfolio manager",)),
)


def _agent_key(ag: Any, explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    role = (getattr(ag, "role", "") or "").lower()
    for key, needles in _DESK_ROLE_NEEDLES:
        if any(n in role for n in needles):
            return key
    return ""


def _prepend(ag: Any, block: str) -> None:
    if not block.strip():
        return
    try:
        ag.backstory = f"{block.strip()}\n\n{ag.backstory or ''}"
    except (AttributeError, TypeError):
        return


def _iter_keyed(agents: Any) -> List[tuple]:
    if agents is None:
        return []
    if isinstance(agents, dict):
        return list(agents.items())
    if isinstance(agents, (list, tuple)):
        return [(None, a) for a in agents]
    return [(None, agents)]


def apply_harness(
    agents: Any,
    *,
    book: str = "equity",
    journal_file: Optional[Path] = None,
    run_mode: str = "autonomous",
) -> Dict[str, Any]:
    """Prepend the unified harness onto a live crew (autonomous or sequential)."""
    from trading_crew.agentic.run_mode_prompts import apply_run_mode_prompts

    apply_prompt_overlay(agents, book=book)
    apply_run_mode_prompts(agents, book=book, run_mode=run_mode)
    skip_playbook = (run_mode or "").strip().lower() == "harness"
    exp = None if skip_playbook else active_experiment(journal_file)
    strategy_text = apply_strategy_backstory(agents, book=book, experiment=exp)
    if skip_playbook:
        return {
            "experiment": None,
            "strategy_chars": len(strategy_text),
            "desk_prompts": [],
            "applied_desks": [],
        }
    desk_prompts = dict((exp or {}).get("desk_prompts") or {})
    promoted = load_promoted_prompts()
    global_add = str((load_promoted().get("global_addendum") or "")).strip()
    applied_desks: List[str] = []
    for key, ag in _iter_keyed(agents):
        desk = _agent_key(ag, key)
        if not desk:
            continue
        parts: List[str] = []
        live = (desk_prompts.get(desk) or "").strip()
        if live:
            parts.append(f"## HARNESS DESK ADDENDUM ({desk})\n{live}")
        # Legacy Mass promote — apply if an older promoted.json is still on disk.
        perm = (promoted.get(desk) or "").strip()
        if desk in ("research_manager", "portfolio_manager") and global_add:
            perm = f"{perm}\n\n{global_add}".strip() if perm else global_add
        if perm:
            parts.append(f"## PROMOTED DESK ADDENDUM ({desk}) [legacy Mass]\n{perm}")
        if parts:
            _prepend(ag, "\n\n".join(parts))
            applied_desks.append(desk)
    return {
        "experiment": exp,
        "strategy_chars": len(strategy_text),
        "desk_prompts": sorted(desk_prompts),
        "applied_desks": applied_desks,
    }


def _iter_metas(ep: Any) -> List[Dict[str, Any]]:
    metas: List[Dict[str, Any]] = []
    status = getattr(ep, "outcome_status", None)
    if str(getattr(status, "value", status) or "").upper() == "RESOLVED":
        thesis = getattr(ep, "reflection_meta", None)
        if isinstance(thesis, dict):
            metas.append(thesis)
    rev = review_payload(ep)
    if rev and isinstance(rev.get("reflection_meta"), dict):
        metas.append(rev["reflection_meta"])
    return metas


def _board_is_split(ep: Any, metas: Sequence[Dict[str, Any]]) -> bool:
    for meta in metas:
        if str(meta.get("stance") or "").upper() == "MIXED":
            return True
        if meta.get("desk_split"):
            return True
    leans = getattr(ep, "desk_leans", None) or {}
    bull = sum(1 for v in leans.values() if str(v).lower() in ("bullish", "buy"))
    bear = sum(1 for v in leans.values() if str(v).lower() in ("bearish", "sell"))
    return bull > 0 and bear > 0


def _loss_label(ep: Any) -> bool:
    rev = review_payload(ep)
    if rev:
        return str(rev.get("outcome_label") or "") == "LOSS"
    return str(getattr(ep, "outcome_label", "") or "") == "LOSS"


def collect_evidence(episodes: Sequence[Any]) -> Dict[str, Any]:
    """Summarize Resolve lessons and board shape. Never invents change text."""
    hurt_counts: Dict[str, int] = {}
    changes: Dict[str, List[str]] = {}
    wrongs: Dict[str, List[str]] = {}
    unhelpful: Dict[str, int] = {}
    overemphasized: Dict[str, int] = {}
    playbook_votes: Dict[str, int] = {}
    n_loss = 0
    n_split_loss = 0
    n_reports_hurt = 0
    for ep in episodes:
        metas = _iter_metas(ep)
        loss = _loss_label(ep)
        if loss:
            n_loss += 1
            if _board_is_split(ep, metas):
                n_split_loss += 1
        for meta in metas:
            for les in meta.get("desk_lessons") or []:
                if not isinstance(les, dict):
                    continue
                if (les.get("verdict") or "").strip().lower() != "hurt":
                    continue
                desk = (les.get("desk") or "").strip()
                change = (les.get("recommended_change") or "").strip()
                wrong = (les.get("what_they_got_wrong") or "").strip()
                if not desk:
                    continue
                hurt_counts[desk] = hurt_counts.get(desk, 0) + 1
                if desk == "reports_analyst" and loss:
                    n_reports_hurt += 1
                if change:
                    bucket = changes.setdefault(desk, [])
                    if change not in bucket:
                        bucket.append(change)
                if wrong:
                    wbucket = wrongs.setdefault(desk, [])
                    if wrong not in wbucket:
                        wbucket.append(wrong)
            if not loss:
                continue
            for tool in meta.get("unhelpful_tools") or []:
                name = str(tool).strip()
                if name:
                    unhelpful[name] = unhelpful.get(name, 0) + 1
            for lens in meta.get("overemphasized") or []:
                name = str(lens).strip().lower()
                if name:
                    overemphasized[name] = overemphasized.get(name, 0) + 1
            vote = str(meta.get("suggested_playbook") or "").strip()
            if vote:
                playbook_votes[vote] = playbook_votes.get(vote, 0) + 1
    dominant = None
    if hurt_counts:
        top_n = max(hurt_counts.values())
        leaders = [d for d, n in hurt_counts.items() if n == top_n]
        if len(leaders) == 1 and (changes.get(leaders[0]) or wrongs.get(leaders[0])):
            dominant = leaders[0]
    share = (n_split_loss / n_loss) if n_loss else 0.0
    top_playbook = None
    if playbook_votes:
        best = max(playbook_votes, key=playbook_votes.get)
        if playbook_votes[best] * 2 > n_loss and n_loss:
            top_playbook = best
    return {
        "hurt_counts": hurt_counts,
        "changes": changes,
        "wrongs": wrongs,
        "dominant_desk": dominant,
        "n_loss": n_loss,
        "n_split_loss": n_split_loss,
        "split_loss_share": share,
        "unhelpful_tools": unhelpful,
        "overemphasized": overemphasized,
        "n_reports_hurt": n_reports_hurt,
        "suggested_playbook": top_playbook,
        "playbook_votes": playbook_votes,
    }


def apply_prompt_overlay(agents: Any, *, book: str = "equity") -> List[str]:
    """Replace packaged goal/backstory when a Hermes overlay key is set."""
    from trading_crew.agentic.prompt_overlay import load_overlay

    if book not in ("equity", "commodity"):
        return []
    overlay = load_overlay().get(book) or {}
    if not overlay:
        return []
    applied: List[str] = []
    for key, ag in _iter_keyed(agents):
        desk = _agent_key(ag, key)
        row = overlay.get(desk)
        if not row:
            continue
        goal = (row.get("goal") or "").strip()
        backstory = (row.get("backstory") or "").strip()
        try:
            if goal:
                ag.goal = goal
            if backstory:
                ag.backstory = backstory
        except (AttributeError, TypeError):
            continue
        applied.append(desk)
    return applied


def _from_yaml_knob(strategy, knob_id: str, score, diagnosis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        knob = strategy.knob(knob_id)
    except StrategyError:
        return None
    if not knob.enabled:
        return None
    payload = {
        "knob": knob.id,
        "apply": knob.apply,
        "kind": "topology" if knob.apply == "playbook" else "synthesis_prompt",
        "playbook_action": knob.playbook_action,
        "hypothesis": _hypothesis_for(knob, score),
        "diagnosis": diagnosis,
    }
    if knob.size_scale is not None:
        payload["size_scale"] = float(knob.size_scale)
    return payload


def propose_experiment(
    episodes: Sequence[Any],
    score,
    strategy,
    used_ids: Sequence[str],
) -> Optional[Dict[str, Any]]:
    """Pick one experiment from Resolve evidence. YAML catalog is a menu, not order."""
    used = set(used_ids or [])
    ev = collect_evidence(episodes)
    fail = strategy.objectives["short"].failure
    metrics = score.metrics
    desk = ev.get("dominant_desk")
    if desk:
        kid = f"desk:{desk}"
        lines = list(ev.get("changes", {}).get(desk) or [])
        for wrong in ev.get("wrongs", {}).get(desk) or []:
            if wrong not in lines:
                lines.append(wrong)
        text = "\n".join(f"- {line}" for line in lines[:8])
        if kid not in used and text:
            return {
                "knob": kid,
                "apply": "desk_prompt",
                "kind": "desk_prompt",
                "desk": desk,
                "desk_prompts": {desk: text},
                "hypothesis": (
                    f"Change only '{kid}'. Use this desk's Resolve "
                    f"recommended_change. Triggered because {score.verdict}: "
                    f"{'; '.join(score.reasons) or 'hurt desk dominates LOSSes'}."
                ),
                "diagnosis": {
                    "rule": "dominant_hurt_desk",
                    "desk": desk,
                    "hurt_counts": ev["hurt_counts"],
                    "n_loss": ev["n_loss"],
                },
            }
    n_loss = int(ev.get("n_loss") or 0)
    reports_blamed = n_loss > 0 and (
        int(ev.get("n_reports_hurt") or 0) * 2 > n_loss
        or int((ev.get("unhelpful_tools") or {}).get("get_report_insights") or 0) * 2 > n_loss
        or int((ev.get("overemphasized") or {}).get("reports_analyst") or 0) * 2 > n_loss
        or int((ev.get("overemphasized") or {}).get("filings") or 0) * 2 > n_loss
    )
    if reports_blamed and "skip_reports" not in used:
        return _from_yaml_knob(
            strategy,
            "skip_reports",
            score,
            {
                "rule": "reports_unhelpful",
                "n_reports_hurt": ev.get("n_reports_hurt"),
                "unhelpful_tools": ev.get("unhelpful_tools"),
            },
        )
    evidence_blamed = (
        (ev.get("unhelpful_tools") or {}).get("request_evidence", 0)
        + (ev.get("unhelpful_tools") or {}).get("research_web", 0)
    )
    if n_loss and evidence_blamed * 2 > n_loss and "evidence_budget_0" not in used:
        return _from_yaml_knob(
            strategy,
            "evidence_budget_0",
            score,
            {
                "rule": "evidence_noise",
                "unhelpful_tools": ev.get("unhelpful_tools"),
            },
        )
    if (
        ev["n_loss"] > 0
        and ev["split_loss_share"] >= 0.5
        and "extra_debate_on_split" not in used
    ):
        return _from_yaml_knob(
            strategy,
            "extra_debate_on_split",
            score,
            {
                "rule": "split_boards",
                "split_loss_share": ev["split_loss_share"],
                "n_loss": ev["n_loss"],
                "n_split_loss": ev["n_split_loss"],
            },
        )
    hit_ok = (
        metrics.hit_rate is not None
        and metrics.hit_rate >= float(fail["hit_rate_below"])
    )
    alpha_fail = (
        metrics.mean_alpha is not None
        and metrics.mean_alpha < float(fail["mean_alpha_below"])
    )
    dd_fail = (
        metrics.mean_drawdown is not None
        and metrics.mean_drawdown < float(fail["mean_drawdown_below"])
    )
    if hit_ok and (alpha_fail or dd_fail) and "size_scale_down" not in used:
        return _from_yaml_knob(
            strategy,
            "size_scale_down",
            score,
            {
                "rule": "size_not_side",
                "hit_rate": metrics.hit_rate,
                "mean_alpha": metrics.mean_alpha,
                "mean_drawdown": metrics.mean_drawdown,
            },
        )
    hit_fail = (
        metrics.hit_rate is not None
        and metrics.hit_rate < float(fail["hit_rate_below"])
    )
    if hit_fail and "debate_rounds_2" not in used:
        return _from_yaml_knob(
            strategy,
            "debate_rounds_2",
            score,
            {
                "rule": "wrong_side",
                "hit_rate": metrics.hit_rate,
            },
        )
    reopen_blamed = (
        (ev.get("overemphasized") or {}).get("trader", 0)
        + (ev.get("unhelpful_tools") or {}).get("reopen_trader", 0)
    )
    if n_loss and reopen_blamed * 2 > n_loss and "no_trader_reopen" not in used:
        return _from_yaml_knob(
            strategy,
            "no_trader_reopen",
            score,
            {"rule": "reopen_hurt", "overemphasized": ev.get("overemphasized")},
        )
    vote = ev.get("suggested_playbook")
    if vote and vote not in used:
        picked = _from_yaml_knob(
            strategy, vote, score,
            {
                "rule": "lesson_playbook_vote",
                "playbook_votes": ev.get("playbook_votes"),
            },
        )
        if picked:
            return picked
    return None


def enrich_experiment(
    experiment: Dict[str, Any],
    episodes: Sequence[Any],
    *,
    skip_llm: bool = True,
) -> Dict[str, Any]:
    """LEGACY Mass attach. Not called on the live Hermes step."""
    from trading_crew.agentic.mass import (
        desk_addenda_from_episodes,
        stage_1po_instruction,
        stage_2to_propose_action,
    )

    addenda = desk_addenda_from_episodes(episodes)
    if not skip_llm:
        for desk in list(addenda):
            llm_text = stage_1po_instruction(desk)
            if llm_text:
                addenda[desk] = llm_text
    if addenda:
        experiment["desk_prompts"] = addenda
    if (experiment.get("apply") or "") == "playbook":
        topo = stage_2to_propose_action()
        if topo and not experiment.get("playbook_action"):
            experiment["playbook_action"] = topo
        if topo:
            experiment["mass_2to"] = topo
    return experiment


def step_harness(
    episodes: Sequence[Any],
    *,
    strategy=None,
    journal_file: Optional[Path] = None,
    book: str = "equity",
    skip_llm: bool = True,
) -> Dict[str, Any]:
    """Score the book and open at most one Hermes experiment. Mass is not stepped."""
    del skip_llm  # kept so Resolve callers do not break; unused on the live path
    out = step_self_learn(
        episodes,
        strategy=strategy,
        journal_file=journal_file,
        book=book,
        propose=propose_experiment,
    )
    journal = load_journal(journal_file)
    out["diagnosis"] = journal.get("last_diagnosis")
    return out


def _live_layers(
    exp: Optional[Dict[str, Any]],
    size_scale: Optional[float],
) -> List[Dict[str, str]]:
    """What the open experiment actually changes on the next prediction."""
    if not exp:
        return []
    layers = [{
        "id": "synthesis_prompt",
        "label": "Prompts",
        "detail": str(exp.get("knob") or ""),
    }]
    raw_desks = exp.get("desk_prompts") or {}
    desks = list(raw_desks) if isinstance(raw_desks, dict) else list(raw_desks)
    if desks:
        layers.append({
            "id": "desk_prompt",
            "label": "Desk",
            "detail": ", ".join(str(d) for d in desks),
        })
    action = str(exp.get("playbook_action") or "").strip()
    if (exp.get("apply") or "") == "playbook" or action:
        layers.append({
            "id": "topology",
            "label": "Topology",
            "detail": action or "playbook",
        })
    if size_scale is not None:
        layers.append({
            "id": "size",
            "label": "Size",
            "detail": f"×{size_scale:g}",
        })
    return layers


def dashboard_payload(
    episodes: Sequence[Any],
    *,
    book: str = "equity",
    strategy=None,
    journal_file: Optional[Path] = None,
) -> Dict[str, Any]:
    payload = _scoreboard(
        episodes, book=book, strategy=strategy, journal_file=journal_file,
    )
    exp = payload.get("experiment") if isinstance(payload.get("experiment"), dict) else None
    from trading_crew.agentic.tuning import hermes_size_scale

    scale = hermes_size_scale(journal_file)
    payload["desk_prompts"] = list((exp or {}).get("desk_prompts") or {})
    payload["diagnosis"] = payload.get("last_diagnosis") or (exp or {}).get("diagnosis")
    payload["hermes_size_scale"] = scale
    payload["live_layers"] = _live_layers(exp, scale)
    payload["unified"] = True
    payload["sample_count"] = sum(
        1 for e in episodes if str(getattr(e, "episode_id", "")).startswith(SAMPLE_PREFIX)
    )
    from trading_crew.agentic.prompt_overlay import list_prompts
    from trading_crew.agentic.strategy import (
        live_strategy_path,
        load_desk_strategy,
        packaged_strategy_path,
        render_prompt_block,
        strategy_source,
        strategy_to_raw,
    )

    s = strategy or load_desk_strategy()
    payload["strategy_source"] = strategy_source()
    payload["packaged_path"] = str(packaged_strategy_path())
    payload["live_path"] = str(live_strategy_path())
    payload["strategy_edit"] = strategy_to_raw(s)
    payload["prompt_block"] = render_prompt_block(
        s, book=book, experiment=exp, path=None,
    )
    payload["prompts"] = list_prompts(book)
    payload["knob_catalog"] = [
        {
            "id": k.id,
            "enabled": k.enabled,
            "apply": k.apply,
            "playbook_action": k.playbook_action,
            "size_scale": k.size_scale,
            "description": k.description,
        }
        for k in s.knobs
    ]
    return payload


def _seed_lessons(pattern: str, *, loss: bool) -> List[Dict[str, Any]]:
    if pattern == "alpha" or not loss:
        return []
    fund = {
        "desk": "fundamentals_analyst",
        "lean_at_decision": "bullish",
        "verdict": "hurt",
        "what_they_got_wrong": "Over-weighted earnings momentum versus guidance.",
        "recommended_change": "Weight NET_GUIDANCE_BIAS at least as heavily as the last print.",
    }
    macro = {
        "desk": "macro_analyst",
        "lean_at_decision": "bearish" if pattern == "split" else "bullish",
        "verdict": "hurt",
        "what_they_got_wrong": "Ignored a one-sided rates/FX headwind.",
        "recommended_change": "Stand down when the macro lens contradicts the tape regime.",
    }
    if pattern == "desk":
        return [fund]
    return [fund, macro]


def _seed_leans(pattern: str) -> Dict[str, str]:
    if pattern == "split":
        return {
            "fundamentals_analyst": "bullish",
            "macro_analyst": "bearish",
            "news_analyst": "neutral",
        }
    return {
        "fundamentals_analyst": "bullish",
        "macro_analyst": "bullish",
        "news_analyst": "neutral",
    }


def seed_harness_book(
    *,
    count: int = 12,
    symbol: str = "SAMPLE.NS",
    memory: Any = None,
    fail: bool = True,
    pattern: Optional[str] = None,
) -> Dict[str, Any]:
    """Synthetic directional 30-day reviews so Hermes diagnosis can be exercised."""
    from trading_crew.agentic.memory import EpisodicMemory
    from trading_crew.agentic.memory.episodic import Episode, OutcomeStatus, Regime
    from trading_crew.agentic.reflection import _default_memory_path

    count = max(1, min(int(count), 200))
    kind = (pattern or ("side" if fail else "mixed")).strip().lower()
    if kind not in ("side", "alpha", "desk", "split", "mixed"):
        raise ValueError(f"unknown harness seed pattern: {kind}")
    sym = (symbol or "SAMPLE.NS").strip().upper()
    mem = memory or EpisodicMemory(_default_memory_path())
    s = load_desk_strategy()
    fail_alpha = float(s.objectives["short"].failure["mean_alpha_below"]) - 0.04
    win_alpha = 0.01 if kind == "alpha" else 0.03
    n_win = max(int(round(count * 0.58)), int(count * 0.45) + 1) if kind == "alpha" else 0
    base = datetime(2026, 5, 1, tzinfo=timezone.utc)
    added = 0
    for i in range(count):
        if kind == "alpha":
            loss = i >= n_win
        else:
            loss = fail or (i % 3 != 0)
        label = "LOSS" if loss else "WIN"
        alpha = fail_alpha if loss else win_alpha
        realised = -0.05 if loss else 0.02
        decision = base + timedelta(days=i)
        review_ts = (decision + timedelta(days=int(s.learning.calendar_days))).isoformat()
        lessons = _seed_lessons(kind, loss=loss)
        mem.add(
            Episode(
                episode_id=f"{SAMPLE_PREFIX}{sym}-{i:03d}",
                symbol=sym,
                decision_ts=decision.isoformat(),
                state_summary=(
                    f"[SAMPLE DATA] Synthetic {label} 30-day path on {sym} "
                    f"({kind}) for harness testing — not a live tape print."
                ),
                regime=Regime.TREND,
                action_proposal={
                    "side": "BUY" if i % 2 == 0 else "SELL",
                    "conviction_score": 0.6,
                    "horizon_days": 180,
                    "target_weight": 0.04,
                },
                outcome_ts=(decision + timedelta(days=180)).isoformat(),
                outcome_status=OutcomeStatus.PENDING,
                review_ts=review_ts,
                review_status="RESOLVED",
                review={
                    "outcome_label": label,
                    "alpha_return": alpha,
                    "realised_return": realised,
                    "max_drawdown": -0.10 if loss else -0.03,
                    "market_return": realised,
                    "reflection": (
                        f"[short_review | {label}] synthetic 30-day path "
                        f"{realised:+.2%} (α {alpha:+.2%})."
                    ),
                    "reflection_meta": {
                        "clock": "short_review",
                        "outcome_label": label,
                        "stance": "MIXED" if kind == "split" else "ALIGNED",
                        "desk_split": kind == "split",
                        "market_direction": "bearish" if loss else "bullish",
                        "desks": [row["desk"] for row in lessons],
                        "desk_lessons": lessons,
                    },
                },
                desk_leans=_seed_leans(kind),
                source="sample",
            )
        )
        added += 1
    store = mem.db_path if not callable(getattr(mem, "db_path", None)) else mem.db_path()
    return {
        "seeded": added,
        "symbol": sym,
        "fail": fail,
        "pattern": kind,
        "store_path": str(store),
    }


def _journal_snapshot_path(journal_file: Optional[Path] = None) -> Path:
    return (journal_file or journal_path()).with_name("journal.pre_sample.json")


def _sample_episodes(memory: Any) -> List[Any]:
    return [
        e for e in memory.all_episodes()
        if str(getattr(e, "episode_id", "")).startswith(SAMPLE_PREFIX)
    ]


def _clear_sample_candidates() -> int:
    from trading_crew.agentic.mass import candidates_dir

    removed = 0
    root = candidates_dir()
    if not root.is_dir():
        return 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        name = p.name
        if name.startswith("SAMPLE") or "SAMPLE.NS" in name or SAMPLE_PREFIX in name:
            p.unlink()
            removed += 1
    return removed


def snapshot_journal(journal_file: Optional[Path] = None) -> bool:
    """Copy the live journal aside so a SAMPLE seed can be undone."""
    src = journal_file or journal_path()
    dest = _journal_snapshot_path(journal_file)
    if src.is_file():
        dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        return True
    if dest.is_file():
        dest.unlink()
    return False


def restore_journal_snapshot(journal_file: Optional[Path] = None) -> bool:
    dest = journal_file or journal_path()
    snap = _journal_snapshot_path(journal_file)
    if not snap.is_file():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(snap.read_text(encoding="utf-8"), encoding="utf-8")
    snap.unlink()
    return True


def seed_and_step(
    *,
    count: int = 12,
    symbol: str = "SAMPLE.NS",
    memory: Any = None,
    journal_file: Optional[Path] = None,
    fail: bool = True,
    pattern: Optional[str] = None,
) -> Dict[str, Any]:
    """Write SAMPLE reviews, run Hermes diagnosis on those rows only, snapshot journal."""
    from trading_crew.agentic.memory import EpisodicMemory
    from trading_crew.agentic.reflection import _default_memory_path

    mem = memory or EpisodicMemory(_default_memory_path())
    jf = journal_file or journal_path()
    snapped = snapshot_journal(jf)
    save_journal(
        {
            "experiment": None,
            "history": [],
            "used_knobs": [],
            "last_score": None,
        },
        jf,
    )
    seeded = seed_harness_book(
        count=count, symbol=symbol, memory=mem, fail=fail, pattern=pattern,
    )
    trained = step_harness(_sample_episodes(mem), journal_file=jf)
    seeded["journal_snapshotted"] = snapped
    seeded["harness"] = {
        "action": trained.get("action"),
        "note": trained.get("note"),
        "verdict": (trained.get("score") or {}).get("verdict"),
        "n": (trained.get("score") or {}).get("n"),
        "experiment": trained.get("experiment"),
        "desk_prompts": list((trained.get("experiment") or {}).get("desk_prompts") or {}),
        "diagnosis": trained.get("diagnosis"),
    }
    return seeded


def clear_sample_episodes(*, memory: Any = None) -> Dict[str, Any]:
    from trading_crew.agentic.memory import EpisodicMemory
    from trading_crew.agentic.reflection import _default_memory_path

    mem = memory or EpisodicMemory(_default_memory_path())
    ids = [
        e.episode_id for e in mem.all_episodes()
        if str(e.episode_id).startswith(SAMPLE_PREFIX)
    ]
    for eid in ids:
        mem.remove(eid)
    return {"removed": len(ids), "candidates_removed": _clear_sample_candidates()}


def clear_sample_and_restore(
    *,
    memory: Any = None,
    journal_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """Drop SAMPLE rows and put the pre-seed journal back."""
    out = clear_sample_episodes(memory=memory)
    out["journal_restored"] = restore_journal_snapshot(journal_file)
    return out


def render_harness_block(book: str = "equity", journal_file: Optional[Path] = None) -> str:
    return render_prompt_block(
        book=book, experiment=active_experiment(journal_file),
    )
