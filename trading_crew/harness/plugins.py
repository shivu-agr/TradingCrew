"""In-process plugin kernel. Missing compose ids are an error."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from .config import HarnessConfigError

_KNOWN = {
    "loop": frozenset({"conductor_v1"}),
    "session": frozenset({"harness_state"}),
    "planner": frozenset({"spawn_v1"}),
}


@dataclass
class PluginContext:
    cfg: Dict[str, Any]
    registry: Dict[Tuple[str, str], Any] = field(default_factory=dict)

    def register(self, kind: str, ident: str, obj: Any) -> None:
        self.registry[(kind, ident)] = obj

    def get(self, kind: str, ident: str) -> Any:
        key = (kind, ident)
        if key not in self.registry:
            raise HarnessConfigError(f"plugin not mounted: {kind}/{ident}")
        return self.registry[key]


def load_plugins(cfg: Dict[str, Any]) -> PluginContext:
    """Mount loop / session / planner / tools / skills from cfg['plugins']."""
    compose = cfg.get("plugins")
    if not isinstance(compose, dict):
        raise HarnessConfigError("harness plugins compose is missing")
    ctx = PluginContext(cfg=cfg)
    loop_id = str(compose.get("loop") or "").strip()
    session_id = str(compose.get("session") or "").strip()
    planner_id = str(compose.get("planner") or "").strip()
    if loop_id not in _KNOWN["loop"]:
        raise HarnessConfigError(f"unknown loop plugin: {loop_id!r}")
    if session_id not in _KNOWN["session"]:
        raise HarnessConfigError(f"unknown session plugin: {session_id!r}")
    if planner_id not in _KNOWN["planner"]:
        raise HarnessConfigError(f"unknown planner plugin: {planner_id!r}")
    from . import planner as planner_mod

    ctx.register("loop", loop_id, "run_harness")
    ctx.register("session", session_id, "harness_state")
    ctx.register("planner", planner_id, planner_mod.plan_spawn)
    for name in compose.get("tools") or []:
        ctx.register("tool", str(name), str(name))
    skills = compose.get("skills") or {}
    if not isinstance(skills, dict):
        raise HarnessConfigError("plugins.skills must be a mapping")
    for sid, skill in skills.items():
        ctx.register("skill", str(sid), skill)
    return ctx


def skill_for(cfg: Dict[str, Any], desk: str) -> Dict[str, Any]:
    skills = (cfg.get("plugins") or {}).get("skills") or {}
    skill = skills.get(desk)
    if not isinstance(skill, dict):
        raise HarnessConfigError(f"no skill plugin for desk {desk!r}")
    return skill
