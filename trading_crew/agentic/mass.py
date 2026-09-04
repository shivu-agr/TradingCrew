"""LEGACY Mass 1PO/2TO/3PO + gated promote.

Hermes now owns live experiment selection (``harness.propose_experiment``).
This module is kept for A/B against Mass-alone and for applying an existing
``promoted.json``. ``step_harness`` does not ingest or stage Mass anymore.
``scripts/promote_graph.py`` is the legacy install path.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from trading_crew.agentic.graph.playbook import PLAYBOOK_ACTIONS, load_playbook, save_playbook
from trading_crew.agentic.outcome import DIRECTIONAL_SIDES, OutcomeLabel, normalize_side

logger = logging.getLogger(__name__)


def _root() -> Path:
    cache = os.environ.get("TRADINGCREW_CACHE_DIR") or os.path.expanduser("~/.trading_crew")
    return Path(cache) / "v3"


def candidates_dir() -> Path:
    p = _root() / "candidates"
    p.mkdir(parents=True, exist_ok=True)
    return p


def promoted_path() -> Path:
    return _root() / "promoted.json"


def _atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".mass.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            if isinstance(payload, str):
                fh.write(payload)
            else:
                json.dump(payload, fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def fetch_tape_reason(
    ticker: str,
    outcome_date: str,
    market_return: Optional[float],
    side: str,
) -> str:
    """Live web search on the resolve day for why the tape moved.

    Must not run inside a point-in-time backfill decision. Resolve is after
    ``outcome_ts`` — that is the Mass ``y`` text, not a live feature.
    """
    from trading_crew.data_sources import pit

    if pit.is_backfill():
        logger.info("mass: skip tape-reason search during PIT backfill")
        return ""

    ret = float(market_return or 0.0)
    direction = "rose" if ret > 0 else ("fell" if ret < 0 else "were flat")
    pct = abs(ret) * 100.0
    query = (
        f"{ticker} {outcome_date} why shares {direction} "
        f"{pct:.1f} percent stock move {side or ''}"
    ).strip()
    from trading_crew.tools import research_web

    raw = ""
    try:
        raw = str(research_web.func(query) or "")
    except Exception:
        logger.warning("mass: research_web failed for %s", ticker, exc_info=True)
        return ""
    if not raw.strip():
        return ""
    return _extract_reason(ticker, outcome_date, side, ret, raw)


def _extract_reason(
    ticker: str, outcome_date: str, side: str, market_return: float, briefing: str
) -> str:
    from trading_crew._common import get_memory_llm

    llm = get_memory_llm()
    prompt = (
        f"Ticker {ticker} on {outcome_date}: predicted side {side or '?'}, "
        f"unsigned market return {market_return:+.2%}.\n"
        "From the briefing below, write 4-6 bullets: what the tape actually "
        "priced (the catalyst), and which desk lens (news / filings / macro / "
        "fundamentals / positioning) that reason belongs to. Keep every "
        "[source: …] tag. Do not invent numbers.\n\n"
        f"{briefing[:12000]}"
    )
    try:
        out = llm.call(messages=[{"role": "user", "content": prompt}])
    except Exception:
        logger.warning("mass: tape-reason extract LLM failed for %s", ticker, exc_info=True)
        return ""
    return (out or "").strip()


def on_episode_resolved(episode: Any, *, skip_search: bool = False) -> str:
    """LEGACY: persist ``market_reason`` and write 1PO candidate snippets.

    Hermes does not consume these files. Kept so Mass-alone A/B still has data.
    """
    reason = (getattr(episode, "market_reason", None) or "").strip()
    if not reason and not skip_search:
        outcome_date = str(getattr(episode, "outcome_ts", "") or "")[:10]
        if not outcome_date:
            outcome_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        ap = getattr(episode, "action_proposal", None) or {}
        side = str(ap.get("side") or ap.get("action") or "")
        reason = fetch_tape_reason(
            getattr(episode, "symbol", ""),
            outcome_date,
            getattr(episode, "market_return", None),
            side,
        )
        episode.market_reason = reason
    if reason:
        _write_1po_example(episode, reason)
    return reason


def _write_1po_example(episode: Any, reason: str, *, dest_dir: Optional[Path] = None) -> None:
    label = str(getattr(episode, "outcome_label", "") or "")
    leans = getattr(episode, "desk_leans", None) or {}
    meta = getattr(episode, "reflection_meta", None) or {}
    lessons = list(meta.get("desk_lessons") or [])
    symbol = getattr(episode, "symbol", "")
    day = str(getattr(episode, "decision_ts", "") or "")[:10]
    for les in lessons:
        desk = (les.get("desk") or "").strip()
        if not desk:
            continue
        verdict = (les.get("verdict") or "").strip()
        if label == OutcomeLabel.WIN.value and verdict != "helped":
            continue
        if label == OutcomeLabel.LOSS.value and verdict != "hurt":
            continue
        if label not in (OutcomeLabel.WIN.value, OutcomeLabel.LOSS.value):
            continue
        rec = {
            "symbol": symbol,
            "decision_ts": day,
            "label": label,
            "desk": desk,
            "lean": les.get("lean_at_decision") or leans.get(desk),
            "what_they_got_wrong": les.get("what_they_got_wrong") or "",
            "recommended_change": les.get("recommended_change") or "",
            "market_reason": reason,
            "alpha_return": getattr(episode, "alpha_return", None),
        }
        dest = dest_dir if dest_dir is not None else candidates_dir()
        path = dest / desk / f"{symbol}_{day}_{label}.json"
        _atomic_write(path, rec)


def _overlay_episode(episode: Any, *, label: str, alpha: Any, reason: str, meta: Dict[str, Any]) -> Any:
    """Thin view so 1PO write can use a 30-day review as the teaching row."""
    class _View:
        def __init__(self) -> None:
            self.symbol = getattr(episode, "symbol", "")
            self.decision_ts = getattr(episode, "decision_ts", "")
            self.action_proposal = getattr(episode, "action_proposal", None) or {}
            self.desk_leans = getattr(episode, "desk_leans", None) or {}
            self.outcome_label = label
            self.alpha_return = alpha
            self.market_reason = reason
            self.reflection_meta = meta
    return _View()


def ingest_demonstrations(episodes: Sequence[Any], *, dest_dir: Optional[Path] = None) -> int:
    """Write 1PO examples from thesis resolves and from 30-day reviews."""
    from trading_crew.agentic.objectives import review_payload

    wrote = 0
    for ep in episodes:
        proposal = getattr(ep, "action_proposal", None) or {}
        side = normalize_side(proposal.get("side") or proposal.get("action"))
        if side not in DIRECTIONAL_SIDES:
            continue
        status = getattr(ep, "outcome_status", None)
        status_s = str(getattr(status, "value", status) or "").upper()
        if status_s == "RESOLVED":
            label = str(getattr(ep, "outcome_label", "") or "")
            if label in (OutcomeLabel.WIN.value, OutcomeLabel.LOSS.value):
                reason = (getattr(ep, "market_reason", None) or "").strip()
                if not reason:
                    reason = (getattr(ep, "reflection", None) or "").strip()
                if reason:
                    _write_1po_example(ep, reason, dest_dir=dest_dir)
                    wrote += 1
        rev = review_payload(ep)
        if rev is None:
            continue
        label = str(rev.get("outcome_label") or "")
        if label not in (OutcomeLabel.WIN.value, OutcomeLabel.LOSS.value):
            continue
        reason = (rev.get("reflection") or getattr(ep, "market_reason", None) or "").strip()
        if not reason:
            continue
        meta = rev.get("reflection_meta") if isinstance(rev.get("reflection_meta"), dict) else {}
        view = _overlay_episode(
            ep,
            label=label,
            alpha=rev.get("alpha_return"),
            reason=reason,
            meta=meta,
        )
        _write_1po_example(view, reason, dest_dir=dest_dir)
        wrote += 1
    return wrote


def desk_addenda_from_episodes(episodes: Sequence[Any]) -> Dict[str, str]:
    """Build live desk addenda from hurt-desk lessons (review or thesis).

    Uses the reflection's own ``recommended_change`` text — no invented
    default instruction.
    """
    from trading_crew.agentic.objectives import review_payload

    buckets: Dict[str, List[str]] = {}
    for ep in episodes:
        metas: List[Dict[str, Any]] = []
        status = getattr(ep, "outcome_status", None)
        if str(getattr(status, "value", status) or "").upper() == "RESOLVED":
            thesis_meta = getattr(ep, "reflection_meta", None)
            if isinstance(thesis_meta, dict):
                metas.append(thesis_meta)
        rev = review_payload(ep)
        if rev and isinstance(rev.get("reflection_meta"), dict):
            metas.append(rev["reflection_meta"])
        for meta in metas:
            for les in meta.get("desk_lessons") or []:
                if not isinstance(les, dict):
                    continue
                if (les.get("verdict") or "").strip().lower() != "hurt":
                    continue
                desk = (les.get("desk") or "").strip()
                change = (les.get("recommended_change") or "").strip()
                if not desk or not change:
                    continue
                buckets.setdefault(desk, [])
                if change not in buckets[desk]:
                    buckets[desk].append(change)
    return {
        desk: "\n".join(f"- {line}" for line in lines[:8])
        for desk, lines in buckets.items()
        if lines
    }


def stage_1po_instruction(desk: str) -> Optional[str]:
    """Propose a short instruction addendum for ``desk`` from its examples."""
    folder = candidates_dir() / desk
    if not folder.is_dir():
        return None
    examples = []
    for p in sorted(folder.glob("*.json"))[-12:]:
        try:
            examples.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    if not examples:
        return None
    from trading_crew._common import get_memory_llm

    lines = []
    for ex in examples:
        lines.append(
            f"- {ex.get('symbol')} {ex.get('decision_ts')} {ex.get('label')} "
            f"lean={ex.get('lean')} reason={str(ex.get('market_reason') or '')[:400]}"
        )
    prompt = (
        f"You are Mass stage 1PO for the {desk} desk. Using these resolved "
        "episodes (WIN = positive exemplar, LOSS + tape reason = negative), "
        "write a 6-10 line instruction addendum this desk should prepend to "
        "its backstory next time. Do not mention specific future prices. "
        "Cite patterns, not one-off noise.\n\n" + "\n".join(lines)
    )
    llm = get_memory_llm()
    try:
        text = (llm.call(messages=[{"role": "user", "content": prompt}]) or "").strip()
    except Exception:
        logger.warning("mass 1PO LLM failed for %s", desk, exc_info=True)
        return None
    if not text:
        return None
    from trading_crew.agentic.versioning import workflow_version

    ver = workflow_version() or "unknown"
    payload = {
        "desk": desk,
        "workflow_version": ver,
        "instruction_addendum": text,
        "few_shots": [
            {
                "symbol": ex.get("symbol"),
                "decision_ts": ex.get("decision_ts"),
                "label": ex.get("label"),
                "lean": ex.get("lean"),
                "market_reason": ex.get("market_reason"),
            }
            for ex in examples[:6]
        ],
    }
    import yaml

    _atomic_write(candidates_dir() / desk / f"{ver}.yaml", yaml.safe_dump(payload, sort_keys=False))
    return text


def stage_2to_propose_action() -> Optional[str]:
    """Pick a PLAYBOOK_ACTION from recorded key win-rates (influence-weighted)."""
    data = load_playbook()
    keys = data.get("keys") or {}
    if not keys:
        return None
    wins = sum(int(r.get("wins") or 0) for r in keys.values())
    losses = sum(int(r.get("losses") or 0) for r in keys.values())
    n = wins + losses
    if n < 3:
        return None
    # Tiny Mass 2TO space. Influence: more debate when the book is losing;
    # extra debate on split when keys already disagree (win_rate near 0.5).
    rates = []
    for row in keys.values():
        w, L = int(row.get("wins") or 0), int(row.get("losses") or 0)
        if (w + L) >= 3:
            rates.append(w / (w + L))
    mean_rate = (sum(rates) / len(rates)) if rates else (wins / n)
    if losses > wins:
        best_action = "debate_rounds_2"
    elif mean_rate < 0.55:
        best_action = "extra_debate_on_split"
    else:
        best_action = "default"
    payload = {
        "proposed_action": best_action,
        "wins": wins,
        "losses": losses,
        "mean_win_rate": mean_rate,
        "space": list(PLAYBOOK_ACTIONS),
    }
    _atomic_write(candidates_dir() / "topology_2to.json", payload)
    return best_action


def stage_3po_global() -> Optional[str]:
    """Light global instruction for RM/PM from recent tape reasons."""
    snippets: List[str] = []
    for p in sorted(candidates_dir().glob("*/*.json"))[-20:]:
        if p.name.startswith("topology"):
            continue
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        reason = (rec.get("market_reason") or "").strip()
        if reason:
            snippets.append(f"{rec.get('symbol')} {rec.get('label')}: {reason[:300]}")
    if len(snippets) < 3:
        return None
    from trading_crew._common import get_memory_llm

    prompt = (
        "Mass 3PO: write a 5-8 line global addendum for the Research Manager "
        "and Portfolio Manager from these resolved tape reasons. Patterns only.\n\n"
        + "\n".join(snippets)
    )
    llm = get_memory_llm()
    try:
        text = (llm.call(messages=[{"role": "user", "content": prompt}]) or "").strip()
    except Exception:
        logger.warning("mass 3PO LLM failed", exc_info=True)
        return None
    if not text:
        return None
    _atomic_write(candidates_dir() / "global_3po.md", text + "\n")
    return text


def evaluate_cohort(episodes: Sequence[Any]) -> Dict[str, float]:
    wins = losses = 0
    alphas: List[float] = []
    for ep in episodes:
        lab = str(getattr(ep, "outcome_label", "") or "")
        if lab == OutcomeLabel.WIN.value:
            wins += 1
        elif lab == OutcomeLabel.LOSS.value:
            losses += 1
        a = getattr(ep, "alpha_return", None)
        if isinstance(a, (int, float)):
            alphas.append(float(a))
    n = wins + losses
    return {
        "n": float(n),
        "win_rate": (wins / n) if n else 0.0,
        "mean_alpha": (sum(alphas) / len(alphas)) if alphas else 0.0,
    }


def load_promoted() -> Dict[str, Any]:
    p = promoted_path()
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_promoted_prompts() -> Dict[str, str]:
    return dict((load_promoted().get("prompts") or {}))


def collect_candidate_prompts() -> Dict[str, str]:
    """Latest 1PO yaml addendum per desk under the candidates tree."""
    out: Dict[str, str] = {}
    import yaml

    for p in sorted(candidates_dir().glob("*/*.yaml")):
        desk = p.parent.name
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        text = str(data.get("instruction_addendum") or "").strip()
        if text:
            out[desk] = text
    return out


def promote(
    *,
    candidate_action: Optional[str] = None,
    desk_instructions: Optional[Dict[str, str]] = None,
    global_addendum: Optional[str] = None,
    candidate_metrics: Optional[Dict[str, float]] = None,
    current_metrics: Optional[Dict[str, float]] = None,
    playbook_key_name: str = "",
) -> bool:
    """Install candidates only if they beat the current held-out cohort."""
    cand = candidate_metrics or {}
    cur = current_metrics or {}
    if float(cand.get("n") or 0) < 3:
        logger.info("mass promote refused: candidate n < 3")
        return False
    if float(cand.get("win_rate") or 0) < float(cur.get("win_rate") or 0):
        logger.info("mass promote refused: win_rate did not beat current")
        return False
    if float(cand.get("mean_alpha") or 0) < float(cur.get("mean_alpha") or 0):
        logger.info("mass promote refused: mean_alpha did not beat current")
        return False
    action = candidate_action or (stage_2to_propose_action() or "default")
    if action not in PLAYBOOK_ACTIONS:
        action = "default"
    payload = {
        "promoted_action": action,
        "prompts": desk_instructions or {},
        "global_addendum": global_addendum or "",
        "metrics": cand,
        "promoted_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
    }
    _atomic_write(promoted_path(), payload)
    if playbook_key_name:
        data = load_playbook()
        row = (data.get("keys") or {}).setdefault(playbook_key_name, {})
        row["promoted_action"] = action
        save_playbook(data)
    logger.info("mass promoted action=%s", action)
    return True
