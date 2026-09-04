"""Scientific-method self-learn loop over the 30-day review book.

After short reviews resolve, score the cohort against desk_strategy.yaml.
On FAILURE, open exactly one experiment. Hermes may propose desk_prompt /
topology / synthesis from Resolve evidence; otherwise the YAML catalog is
walked in order. On SUCCESS, keep the open experiment (do not auto-install).
A look-ahead-suspect book or a single LOSS never rewrites production.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

from trading_crew.agentic.objectives import BookVerdict, ObjectiveScore, score_episodes
from trading_crew.agentic.strategy import DeskStrategy, Knob, load_desk_strategy, strategy_path

logger = logging.getLogger(__name__)

_JOURNAL_NAME = "journal.json"


def journal_path() -> Path:
    cache = os.environ.get("TRADINGCREW_CACHE_DIR") or os.path.expanduser("~/.trading_crew")
    return Path(cache) / "v3" / "self_learn" / _JOURNAL_NAME


def _atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".self_learn.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_journal(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or journal_path()
    if not p.is_file():
        return {
            "experiment": None,
            "history": [],
            "used_knobs": [],
            "last_score": None,
        }
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"self-learn journal is not a mapping: {p}")
    data.setdefault("experiment", None)
    data.setdefault("history", [])
    data.setdefault("used_knobs", [])
    data.setdefault("last_score", None)
    return data


def save_journal(data: Dict[str, Any], path: Optional[Path] = None) -> None:
    _atomic_write(path or journal_path(), data)


def active_experiment(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    exp = load_journal(path).get("experiment")
    if not isinstance(exp, dict):
        return None
    if (exp.get("status") or "") != "open":
        return None
    if not exp.get("knob"):
        return None
    return exp


def _hypothesis_for(knob: Knob, score: ObjectiveScore) -> str:
    reasons = "; ".join(score.reasons) or "30-day book missed the success bars"
    return (
        f"Change only '{knob.id}'. {knob.description} "
        f"Triggered because {score.verdict}: {reasons}."
    )


def _close_experiment(journal: Dict[str, Any], *, status: str, note: str) -> None:
    exp = journal.get("experiment")
    if not isinstance(exp, dict):
        return
    exp["status"] = status
    exp["closed_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    exp["close_note"] = note
    history = list(journal.get("history") or [])
    history.append(dict(exp))
    journal["history"] = history[-40:]
    journal["experiment"] = None


def _stamp_open(journal: Dict[str, Any], exp: Dict[str, Any], score: ObjectiveScore) -> Dict[str, Any]:
    exp = dict(exp)
    exp["status"] = "open"
    exp["opened_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    exp["n_at_open"] = score.metrics.n
    exp["verdict_at_open"] = score.verdict
    journal["experiment"] = exp
    used = list(journal.get("used_knobs") or [])
    kid = str(exp.get("knob") or "")
    if kid and kid not in used:
        used.append(kid)
    journal["used_knobs"] = used
    return exp


def _open_experiment(journal: Dict[str, Any], knob: Knob, score: ObjectiveScore) -> Dict[str, Any]:
    payload = {
        "knob": knob.id,
        "apply": knob.apply,
        "kind": "topology" if knob.apply == "playbook" else "synthesis_prompt",
        "playbook_action": knob.playbook_action,
        "hypothesis": _hypothesis_for(knob, score),
    }
    if knob.size_scale is not None:
        payload["size_scale"] = float(knob.size_scale)
    return _stamp_open(journal, payload, score)


def step_self_learn(
    episodes: Sequence[Any],
    *,
    strategy: Optional[DeskStrategy] = None,
    journal_file: Optional[Path] = None,
    book: str = "equity",
    propose: Optional[Callable[..., Optional[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """Score the 30-day review book and advance at most one experiment."""
    s = strategy or load_desk_strategy()
    score = score_episodes(episodes, tier="short", strategy=s, prefer_review=True)
    journal = load_journal(journal_file)
    journal["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    journal["book"] = book
    journal["strategy_name"] = s.name
    journal["strategy_path"] = str(strategy_path())
    journal["last_score"] = score.to_dict()

    exp = journal.get("experiment") if isinstance(journal.get("experiment"), dict) else None
    if exp and (exp.get("status") or "") != "open":
        exp = None

    action = "hold"
    note = ""

    if score.verdict == BookVerdict.LOOKAHEAD_SUSPECT:
        if exp:
            _close_experiment(journal, status="rejected_lookahead", note="; ".join(score.reasons))
            action = "reject_lookahead"
        else:
            action = "discard_lookahead"
        note = "; ".join(score.reasons)

    elif score.verdict == BookVerdict.INSUFFICIENT:
        action = "wait"
        note = "; ".join(score.reasons)
        if exp is None and s.seed_experiment.knob:
            knob = s.knob(s.seed_experiment.knob)
            opened = _open_experiment(journal, knob, score)
            if s.seed_experiment.hypothesis:
                opened["hypothesis"] = s.seed_experiment.hypothesis
            journal["experiment"] = opened
            action = "seed"
            note = f"seeded knob {knob.id} from desk_strategy.yaml"

    elif score.verdict == BookVerdict.SUCCESS:
        if exp:
            action = "confirm"
            note = "30-day book cleared success bars; keep the open knob (not auto-installed)"
        else:
            action = "keep"
            note = "30-day book cleared success bars; no experiment open"

    elif score.verdict == BookVerdict.FAILURE:
        if s.never_promote_from_single_loss and score.metrics.n < s.min_reviews_to_change:
            action = "wait"
            note = (
                f"FAILURE on n={score.metrics.n} < min_reviews_to_change="
                f"{s.min_reviews_to_change}; one LOSS does not rewrite the book"
            )
        elif exp:
            _close_experiment(journal, status="rejected", note="; ".join(score.reasons))
            action = "reject"
            note = "open knob did not clear the failure bars; closed, no second knob this step"
        else:
            used = list(journal.get("used_knobs") or [])
            proposed = None
            if propose is not None:
                proposed = propose(episodes, score, s, used)
            if proposed and proposed.get("knob"):
                opened = _stamp_open(journal, proposed, score)
                action = "open"
                note = f"opened one experiment: {opened['knob']}"
                journal["last_diagnosis"] = opened.get("diagnosis")
            else:
                nxt = s.next_unused_knob(used)
                if nxt is None:
                    action = "exhausted"
                    note = "all knobs in desk_strategy.yaml have been tried; edit the file to add more"
                    journal["last_diagnosis"] = {
                        "rule": "yaml_catalog",
                        "reason": "no unused evidence-backed or catalog experiment",
                    }
                else:
                    opened = _open_experiment(journal, nxt, score)
                    action = "open"
                    note = f"opened one knob: {opened['knob']}"
                    journal["last_diagnosis"] = {
                        "rule": "yaml_catalog",
                        "knob": opened["knob"],
                        "reason": "no unused evidence-backed experiment; next catalog knob",
                    }

    else:  # MIXED
        action = "hold_mixed"
        note = "; ".join(score.reasons)

    journal["last_action"] = action
    journal["last_note"] = note
    save_journal(journal, journal_file)

    candidate: Optional[Dict[str, Any]] = None
    live = active_experiment(journal_file)
    if live and live.get("apply") == "playbook" and live.get("playbook_action"):
        candidate = {
            "proposed_action": live["playbook_action"],
            "knob": live["knob"],
            "hypothesis": live.get("hypothesis"),
            "score": score.to_dict(),
            "note": "Hermes playbook is live via the graph router; Mass promote is not required",
        }
        cand_path = (journal_file or journal_path()).parent / "candidate.json"
        _atomic_write(cand_path, candidate)

    logger.info("self-learn action=%s verdict=%s n=%s", action, score.verdict, score.metrics.n)
    return {
        "action": action,
        "note": note,
        "score": score.to_dict(),
        "experiment": live or journal.get("experiment"),
        "candidate": candidate,
    }


def close_experiment_ui(
    *,
    journal_file: Optional[Path] = None,
    note: str = "closed from Hermes UI",
) -> Dict[str, Any]:
    """Operator close. Does not invent a replacement knob."""
    journal = load_journal(journal_file)
    exp = journal.get("experiment") if isinstance(journal.get("experiment"), dict) else None
    if not exp or (exp.get("status") or "") != "open":
        raise ValueError("no open experiment to close")
    _close_experiment(journal, status="closed_ui", note=note)
    journal["last_action"] = "close_ui"
    journal["last_note"] = note
    journal["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    save_journal(journal, journal_file)
    return {"action": "close_ui", "note": note, "experiment": None}


def open_knob_ui(
    knob_id: str,
    episodes: Sequence[Any],
    *,
    journal_file: Optional[Path] = None,
    book: str = "equity",
) -> Dict[str, Any]:
    """Operator-open one catalog knob. Rejects if one is already open."""
    if active_experiment(journal_file):
        raise ValueError("an experiment is already open; close it first")
    s = load_desk_strategy()
    knob = s.knob(knob_id)
    if not knob.enabled:
        raise ValueError(f"knob {knob_id!r} is disabled")
    score = score_episodes(episodes, tier="short", strategy=s, prefer_review=True)
    journal = load_journal(journal_file)
    opened = _open_experiment(journal, knob, score)
    journal["last_action"] = "open_ui"
    journal["last_note"] = f"opened {knob.id} from Hermes UI"
    journal["last_diagnosis"] = {"rule": "ui_open", "knob": knob.id}
    journal["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    journal["book"] = book
    save_journal(journal, journal_file)
    return {
        "action": "open_ui",
        "note": journal["last_note"],
        "experiment": opened,
        "score": score.to_dict(),
    }


def clear_used_knobs_ui(*, journal_file: Optional[Path] = None) -> Dict[str, Any]:
    journal = load_journal(journal_file)
    journal["used_knobs"] = []
    journal["last_action"] = "clear_used"
    journal["last_note"] = "cleared used knobs from Hermes UI"
    journal["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    save_journal(journal, journal_file)
    return {"action": "clear_used", "used_knobs": []}


def dashboard_payload(
    episodes: Sequence[Any],
    *,
    book: str = "equity",
    strategy: Optional[DeskStrategy] = None,
    journal_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """Read-side snapshot for the Training tab Hermes board."""
    s = strategy or load_desk_strategy()
    score = score_episodes(episodes, tier="short", strategy=s, prefer_review=True)
    journal = load_journal(journal_file)
    return {
        "name": s.name,
        "version": s.version,
        "path": str(strategy_path()),
        "editable": True,
        "book": book,
        "mandate": s.mandate,
        "learning": {
            "calendar_days": s.learning.calendar_days,
            "trading_days": s.learning.trading_days,
            "embargo_days": s.learning.embargo_days,
        },
        "thesis": {
            tier: {
                "trading_days": clock.trading_days,
                "calendar_days": clock.calendar_days,
                "note": clock.note,
            }
            for tier, clock in s.thesis.items()
        },
        "objectives": {
            tier: {
                "min_n": band.min_n,
                "success": dict(band.success),
                "failure": dict(band.failure),
                "look_ahead": dict(band.look_ahead),
            }
            for tier, band in s.objectives.items()
        },
        "knobs": [k.id for k in s.knobs],
        "one_variable_at_a_time": s.one_variable_at_a_time,
        "min_reviews_to_change": s.min_reviews_to_change,
        "score": score.to_dict(),
        "experiment": active_experiment(journal_file),
        "last_action": journal.get("last_action"),
        "last_note": journal.get("last_note"),
        "last_diagnosis": journal.get("last_diagnosis"),
        "used_knobs": list(journal.get("used_knobs") or []),
    }
