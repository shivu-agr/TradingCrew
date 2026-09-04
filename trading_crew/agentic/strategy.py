"""Load the editable desk strategy and render it for the desks.

Source of truth, in order:
  1. ``TRADINGCREW_STRATEGY_PATH`` if set
  2. the live file saved from Settings
     (``$TRADINGCREW_CACHE_DIR/v3/desk_strategy.yaml``)
  3. the packaged ``trading_crew/config/desk_strategy.yaml``

Missing file or missing required keys raise — there is no silent default book.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml


_PACKAGE_STRATEGY = (
    Path(__file__).resolve().parents[1] / "config" / "desk_strategy.yaml"
)
_LIVE_HEADER = (
    "# Live desk strategy — saved from Settings.\n"
    "# Packaged default: trading_crew/config/desk_strategy.yaml\n"
    "# Revert in the UI to delete this file and use the packaged yaml.\n"
)

_REQUIRED_OBJECTIVE_KEYS = (
    "min_n",
    "success",
    "failure",
    "look_ahead",
)
_REQUIRED_SUCCESS = (
    "min_hit_rate",
    "min_mean_alpha",
    "max_mean_drawdown",
    "min_expectancy",
    "min_profit_factor",
)
_REQUIRED_FAILURE = (
    "hit_rate_below",
    "mean_alpha_below",
    "mean_drawdown_below",
)
_REQUIRED_LOOKAHEAD = (
    "min_n_for_trust",
    "hit_rate_above",
    "mean_alpha_above",
)


class StrategyError(ValueError):
    """The strategy file is missing or does not match the required schema."""


def packaged_strategy_path() -> Path:
    return _PACKAGE_STRATEGY


def live_strategy_path() -> Path:
    cache = os.environ.get("TRADINGCREW_CACHE_DIR") or os.path.expanduser("~/.trading_crew")
    return Path(cache) / "v3" / "desk_strategy.yaml"


def strategy_path() -> Path:
    override = (os.environ.get("TRADINGCREW_STRATEGY_PATH") or "").strip()
    if override:
        return Path(override)
    live = live_strategy_path()
    if live.is_file():
        return live
    return _PACKAGE_STRATEGY


def strategy_source() -> str:
    override = (os.environ.get("TRADINGCREW_STRATEGY_PATH") or "").strip()
    if override:
        return "env"
    if live_strategy_path().is_file():
        return "live"
    return "packaged"


def _require(data: Dict[str, Any], key: str, *, ctx: str) -> Any:
    if key not in data:
        raise StrategyError(f"{ctx}: missing required key {key!r}")
    return data[key]


def _require_keys(data: Dict[str, Any], keys: Sequence[str], *, ctx: str) -> None:
    missing = [k for k in keys if k not in data]
    if missing:
        raise StrategyError(f"{ctx}: missing required keys {missing}")


@dataclass(frozen=True)
class ClockSpec:
    calendar_days: int
    trading_days: int
    note: str = ""


@dataclass(frozen=True)
class LearningClock:
    calendar_days: int
    trading_days: int
    embargo_days: int
    label: str


@dataclass(frozen=True)
class ObjectiveBand:
    min_n: int
    success: Dict[str, float]
    failure: Dict[str, float]
    look_ahead: Dict[str, float]


@dataclass(frozen=True)
class Knob:
    id: str
    apply: str
    description: str
    playbook_action: str = ""
    size_scale: Optional[float] = None
    enabled: bool = True


@dataclass(frozen=True)
class SeedExperiment:
    knob: str
    hypothesis: str


@dataclass(frozen=True)
class DeskStrategy:
    name: str
    version: int
    mandate: str
    asset_classes: List[str]
    learning: LearningClock
    thesis: Dict[str, ClockSpec]
    objectives: Dict[str, ObjectiveBand]
    rules: Dict[str, List[str]]
    one_variable_at_a_time: bool
    min_reviews_to_change: int
    never_promote_from_single_loss: bool
    seed_experiment: SeedExperiment
    knobs: List[Knob] = field(default_factory=list)

    def knob(self, knob_id: str) -> Knob:
        for k in self.knobs:
            if k.id == knob_id:
                return k
        raise StrategyError(f"unknown knob {knob_id!r}")

    def next_unused_knob(self, used_ids: Iterable[str]) -> Optional[Knob]:
        used = set(used_ids)
        for k in self.knobs:
            if k.id not in used and k.enabled:
                return k
        return None


def _clock(raw: Dict[str, Any], *, ctx: str, embargo: bool = False) -> ClockSpec | LearningClock:
    _require_keys(raw, ("calendar_days", "trading_days"), ctx=ctx)
    cal = int(raw["calendar_days"])
    trd = int(raw["trading_days"])
    if cal < 1 or trd < 1:
        raise StrategyError(f"{ctx}: calendar_days and trading_days must be >= 1")
    if embargo:
        if "embargo_days" not in raw or "label" not in raw:
            raise StrategyError(f"{ctx}: missing embargo_days or label")
        return LearningClock(
            calendar_days=cal,
            trading_days=trd,
            embargo_days=int(raw["embargo_days"]),
            label=str(raw["label"]),
        )
    return ClockSpec(
        calendar_days=cal,
        trading_days=trd,
        note=str(raw.get("note") or ""),
    )


def _band(raw: Dict[str, Any], *, ctx: str) -> ObjectiveBand:
    _require_keys(raw, _REQUIRED_OBJECTIVE_KEYS, ctx=ctx)
    success = dict(raw["success"])
    failure = dict(raw["failure"])
    look = dict(raw["look_ahead"])
    _require_keys(success, _REQUIRED_SUCCESS, ctx=f"{ctx}.success")
    _require_keys(failure, _REQUIRED_FAILURE, ctx=f"{ctx}.failure")
    _require_keys(look, _REQUIRED_LOOKAHEAD, ctx=f"{ctx}.look_ahead")
    return ObjectiveBand(
        min_n=int(raw["min_n"]),
        success={k: float(success[k]) for k in _REQUIRED_SUCCESS},
        failure={k: float(failure[k]) for k in _REQUIRED_FAILURE},
        look_ahead={k: float(look[k]) for k in _REQUIRED_LOOKAHEAD},
    )


def _knob(raw: Dict[str, Any], *, ctx: str) -> Knob:
    _require_keys(raw, ("id", "apply", "description", "enabled"), ctx=ctx)
    apply = str(raw["apply"])
    if apply not in ("prompt", "playbook"):
        raise StrategyError(f"{ctx}: apply must be 'prompt' or 'playbook'")
    action = str(raw.get("playbook_action") or "")
    if apply == "playbook" and not action:
        raise StrategyError(f"{ctx}: playbook knobs need playbook_action")
    scale_raw = raw.get("size_scale")
    size_scale = None
    if scale_raw is not None and str(scale_raw).strip() != "":
        size_scale = float(scale_raw)
        if size_scale <= 0:
            raise StrategyError(f"{ctx}: size_scale must be > 0")
    return Knob(
        id=str(raw["id"]),
        apply=apply,
        description=" ".join(str(raw["description"]).split()),
        playbook_action=action,
        size_scale=size_scale,
        enabled=bool(raw["enabled"]),
    )


def parse_desk_strategy(raw: Any) -> DeskStrategy:
    if not isinstance(raw, dict):
        raise StrategyError("strategy root must be a mapping")
    name = str(_require(raw, "name", ctx="root"))
    version = int(_require(raw, "version", ctx="root"))
    mandate = " ".join(str(_require(raw, "mandate", ctx="root")).split())
    assets = list(_require(raw, "asset_classes", ctx="root"))
    if not assets:
        raise StrategyError("asset_classes must not be empty")

    learning_raw = _require(raw, "learning", ctx="root")
    learning = _clock(learning_raw, ctx="learning", embargo=True)
    assert isinstance(learning, LearningClock)

    thesis_raw = _require(raw, "thesis", ctx="root")
    if not isinstance(thesis_raw, dict):
        raise StrategyError("thesis must be a mapping")
    thesis: Dict[str, ClockSpec] = {}
    for tier in ("short", "medium", "long"):
        clock = _clock(_require(thesis_raw, tier, ctx="thesis"), ctx=f"thesis.{tier}")
        assert isinstance(clock, ClockSpec)
        thesis[tier] = clock

    obj_raw = _require(raw, "objectives", ctx="root")
    objectives = {
        tier: _band(_require(obj_raw, tier, ctx="objectives"), ctx=f"objectives.{tier}")
        for tier in ("short", "medium", "long")
    }

    rules_raw = _require(raw, "rules", ctx="root")
    rules: Dict[str, List[str]] = {}
    for section in ("shared", "equity", "commodity"):
        items = _require(rules_raw, section, ctx="rules")
        rules[section] = [str(x).strip() for x in items if str(x).strip()]

    sci = _require(raw, "scientific_method", ctx="root")
    _require_keys(
        sci,
        ("one_variable_at_a_time", "min_reviews_to_change", "never_promote_from_single_loss", "seed_experiment"),
        ctx="scientific_method",
    )
    seed_raw = sci["seed_experiment"] or {}
    seed = SeedExperiment(
        knob=str(seed_raw.get("knob") or "").strip(),
        hypothesis=str(seed_raw.get("hypothesis") or "").strip(),
    )
    knobs_raw = _require(raw, "knobs", ctx="root")
    knobs = [_knob(item, ctx=f"knobs[{i}]") for i, item in enumerate(knobs_raw)]
    ids = [k.id for k in knobs]
    if len(ids) != len(set(ids)):
        raise StrategyError("knob ids must be unique")

    return DeskStrategy(
        name=name,
        version=version,
        mandate=mandate,
        asset_classes=[str(a) for a in assets],
        learning=learning,
        thesis=thesis,
        objectives=objectives,
        rules=rules,
        one_variable_at_a_time=bool(sci["one_variable_at_a_time"]),
        min_reviews_to_change=int(sci["min_reviews_to_change"]),
        never_promote_from_single_loss=bool(sci["never_promote_from_single_loss"]),
        seed_experiment=seed,
        knobs=knobs,
    )


def load_desk_strategy(path: Optional[Path] = None) -> DeskStrategy:
    p = path or strategy_path()
    if not p.is_file():
        raise StrategyError(f"desk strategy file not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    return parse_desk_strategy(raw)


def render_prompt_block(
    strategy: Optional[DeskStrategy] = None,
    *,
    book: str = "equity",
    experiment: Optional[Dict[str, Any]] = None,
    path: Optional[Path] = None,
) -> str:
    """Compact block prepended to trader / PM / research-manager backstories."""
    s = strategy or load_desk_strategy(path)
    book_key = "commodity" if book == "commodity" else "equity"
    lines = [
        "## DESK STRATEGY (editable from Settings)",
        f"Book: {s.name} v{s.version} · asset={book_key}",
        f"Learning clock: {s.learning.calendar_days} calendar days "
        f"(~{s.learning.trading_days} trading) — this is scored. "
        f"Thesis clocks stay SHORT {s.thesis['short'].trading_days}d / "
        f"MEDIUM {s.thesis['medium'].trading_days}d / "
        f"LONG {s.thesis['long'].trading_days}d.",
        "",
        f"Mandate: {s.mandate}",
        "",
        "30-day SUCCESS (book, after min_n reviews): "
        f"hit-rate ≥ {s.objectives['short'].success['min_hit_rate']:.0%}, "
        f"mean α ≥ {s.objectives['short'].success['min_mean_alpha']:+.2%}, "
        f"mean DD ≥ {s.objectives['short'].success['max_mean_drawdown']:.0%}, "
        f"expectancy ≥ {s.objectives['short'].success['min_expectancy']:+.2%}, "
        f"PF ≥ {s.objectives['short'].success['min_profit_factor']:.2f}.",
        "30-day FAILURE: "
        f"hit-rate < {s.objectives['short'].failure['hit_rate_below']:.0%} OR "
        f"mean α < {s.objectives['short'].failure['mean_alpha_below']:+.2%} OR "
        f"mean DD < {s.objectives['short'].failure['mean_drawdown_below']:.0%}.",
        "Do not shrink a MEDIUM/LONG thesis to 30 days — the loop snapshots "
        "the 30-day path without changing your horizon_days.",
        "",
        "Rules:",
    ]
    for rule in list(s.rules.get("shared") or []) + list(s.rules.get(book_key) or []):
        lines.append(f"- {rule}")
    if experiment and experiment.get("knob"):
        lines.append("")
        lines.append("## ACTIVE LEARNING EXPERIMENT (one knob only)")
        lines.append(f"Knob: {experiment.get('knob')}")
        hyp = (experiment.get("hypothesis") or "").strip()
        if hyp:
            lines.append(f"Hypothesis: {hyp}")
        lines.append("Do not open a second experiment.  Judge this knob on the 30-day bars.")
    return "\n".join(lines)


def apply_strategy_backstory(
    agents: Any,
    *,
    book: str = "equity",
    experiment: Optional[Dict[str, Any]] = None,
    path: Optional[Path] = None,
) -> str:
    """Prepend the strategy block to trader / PM / research-manager agents."""
    text = render_prompt_block(book=book, experiment=experiment, path=path)
    needles = ("trader", "portfolio manager", "research manager")
    for ag in _iter_agents(agents):
        role = (getattr(ag, "role", "") or "").lower()
        if not any(n in role for n in needles):
            continue
        try:
            ag.backstory = f"{text}\n\n{ag.backstory or ''}"
        except (AttributeError, TypeError):
            continue
    return text


def _iter_agents(agents: Any) -> List[Any]:
    if agents is None:
        return []
    if isinstance(agents, dict):
        return list(agents.values())
    if isinstance(agents, (list, tuple)):
        return list(agents)
    return [agents]


def strategy_to_raw(strategy: DeskStrategy) -> Dict[str, Any]:
    """Round-trip mapping the parser accepts. Used by the Hermes editor."""
    knobs: List[Dict[str, Any]] = []
    for k in strategy.knobs:
        row: Dict[str, Any] = {
            "id": k.id,
            "enabled": bool(k.enabled),
            "apply": k.apply,
            "description": k.description,
        }
        if k.playbook_action:
            row["playbook_action"] = k.playbook_action
        if k.size_scale is not None:
            row["size_scale"] = float(k.size_scale)
        knobs.append(row)
    return {
        "name": strategy.name,
        "version": int(strategy.version),
        "mandate": strategy.mandate,
        "asset_classes": list(strategy.asset_classes),
        "learning": {
            "calendar_days": strategy.learning.calendar_days,
            "trading_days": strategy.learning.trading_days,
            "embargo_days": strategy.learning.embargo_days,
            "label": strategy.learning.label,
        },
        "thesis": {
            tier: {
                "trading_days": clock.trading_days,
                "calendar_days": clock.calendar_days,
                "note": clock.note,
            }
            for tier, clock in strategy.thesis.items()
        },
        "objectives": {
            tier: {
                "min_n": band.min_n,
                "success": dict(band.success),
                "failure": dict(band.failure),
                "look_ahead": dict(band.look_ahead),
            }
            for tier, band in strategy.objectives.items()
        },
        "rules": {section: list(items) for section, items in strategy.rules.items()},
        "scientific_method": {
            "one_variable_at_a_time": bool(strategy.one_variable_at_a_time),
            "min_reviews_to_change": int(strategy.min_reviews_to_change),
            "never_promote_from_single_loss": bool(strategy.never_promote_from_single_loss),
            "seed_experiment": {
                "knob": strategy.seed_experiment.knob,
                "hypothesis": strategy.seed_experiment.hypothesis,
            },
        },
        "knobs": knobs,
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".desk_strategy.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def save_desk_strategy(raw: Dict[str, Any], path: Optional[Path] = None) -> DeskStrategy:
    """Validate then write the live (or given) strategy file."""
    if path is None and (os.environ.get("TRADINGCREW_STRATEGY_PATH") or "").strip():
        raise StrategyError(
            "TRADINGCREW_STRATEGY_PATH is set; unset it to save from Settings"
        )
    parsed = parse_desk_strategy(raw)
    dest = path or live_strategy_path()
    dump = yaml.safe_dump(
        strategy_to_raw(parsed),
        sort_keys=False,
        allow_unicode=True,
        width=88,
    )
    _atomic_write_text(dest, _LIVE_HEADER + dump)
    return parsed


def revert_live_strategy() -> bool:
    """Delete the live file. Packaged yaml is used on the next load.

    Raises if ``TRADINGCREW_STRATEGY_PATH`` is set — that path is not ours to delete.
    """
    if (os.environ.get("TRADINGCREW_STRATEGY_PATH") or "").strip():
        raise StrategyError(
            "TRADINGCREW_STRATEGY_PATH is set; unset it before reverting to packaged"
        )
    live = live_strategy_path()
    if not live.is_file():
        return False
    live.unlink()
    return True
