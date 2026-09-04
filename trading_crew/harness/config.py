"""Load harness.yaml. Missing file or keys is an error — no Python defaults."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import yaml

_ROOT = Path(__file__).resolve().parent.parent
_PACKAGED = _ROOT / "config" / "harness.yaml"

REQUIRED_KEYS = (
    "max_steps",
    "exhaust_steps",
    "max_parallel",
    "max_spawn_depth",
    "max_spawn",
    "context_chars",
    "knowledge_k",
    "retriever",
    "corpus_dir",
    "conductor_temperature",
    "min_tool_families",
    "require_skeptical_lens",
    "require_debate_pair",
    "require_bundled_debate",
    "require_reports_lens",
    "require_market_lens",
    "require_news_lens",
    "require_fundamentals_lens",
    "require_macro_lens",
    "require_risk_lens",
    "require_bundled_risk",
    "min_steps_before_finish",
    "query_required_tools",
    "deny_tools",
    "tool_allowlist",
    "catalog_desks",
    "standing_rules",
    "plugins_file",
)

INT_KEYS = (
    "max_steps",
    "exhaust_steps",
    "max_parallel",
    "max_spawn_depth",
    "max_spawn",
    "context_chars",
    "knowledge_k",
    "min_steps_before_finish",
)

BOOL_KEYS = (
    "require_skeptical_lens",
    "require_debate_pair",
    "require_bundled_debate",
    "require_reports_lens",
    "require_market_lens",
    "require_news_lens",
    "require_fundamentals_lens",
    "require_macro_lens",
    "require_risk_lens",
    "require_bundled_risk",
)

LIST_KEYS = (
    "query_required_tools",
    "deny_tools",
    "tool_allowlist",
    "catalog_desks",
)


class HarnessConfigError(ValueError):
    """Harness config file is missing or invalid."""


def config_path() -> Path:
    raw = (os.environ.get("TRADINGCREW_HARNESS_CONFIG") or "").strip()
    if raw:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = _ROOT.parent / path
        return path
    return _PACKAGED


def memory_overlay_path() -> Path | None:
    raw = (os.environ.get("TRADINGCREW_HARNESS_MEMORY") or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = _ROOT.parent / path
    return path


def _str_list(raw: Any, key: str) -> List[str]:
    if not isinstance(raw, list) or any(not isinstance(n, str) or not n.strip() for n in raw):
        raise HarnessConfigError(f"{key} must be a list of strings")
    return [str(n).strip() for n in raw]


def load_harness_config() -> Dict[str, Any]:
    path = config_path()
    if not path.is_file():
        raise HarnessConfigError(f"harness config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise HarnessConfigError(f"harness config must be a mapping: {path}")
    missing = [k for k in REQUIRED_KEYS if k not in raw]
    if missing:
        raise HarnessConfigError(f"harness config missing keys {missing}: {path}")
    out: Dict[str, Any] = {"config_path": str(path)}
    for key in INT_KEYS:
        try:
            val = int(raw[key])
        except (TypeError, ValueError) as exc:
            raise HarnessConfigError(f"{key} must be an integer") from exc
        if val < 0:
            raise HarnessConfigError(f"{key} must be >= 0")
        out[key] = val
    try:
        out["conductor_temperature"] = float(raw["conductor_temperature"])
    except (TypeError, ValueError) as exc:
        raise HarnessConfigError("conductor_temperature must be a number") from exc
    for key in BOOL_KEYS:
        if not isinstance(raw[key], bool):
            raise HarnessConfigError(f"{key} must be a boolean")
        out[key] = raw[key]
    try:
        out["min_tool_families"] = int(raw["min_tool_families"])
    except (TypeError, ValueError) as exc:
        raise HarnessConfigError("min_tool_families must be an integer") from exc
    retriever = str(raw["retriever"] or "").strip()
    if retriever not in ("tfidf", "embedder"):
        raise HarnessConfigError("retriever must be tfidf or embedder")
    out["retriever"] = retriever
    rules = str(raw["standing_rules"] or "").strip()
    if not rules:
        raise HarnessConfigError("standing_rules must be set")
    out["standing_rules"] = rules
    for key in LIST_KEYS:
        out[key] = _str_list(raw[key], key)
    corpus = Path(str(raw["corpus_dir"]).strip()).expanduser()
    if not corpus.is_absolute():
        corpus = _ROOT.parent / corpus
    out["corpus_dir"] = corpus
    overlay = memory_overlay_path()
    if overlay is not None and not overlay.is_file():
        raise HarnessConfigError(f"TRADINGCREW_HARNESS_MEMORY is set but missing: {overlay}")
    out["memory_overlay_path"] = overlay
    out["plugins"] = _load_plugins_compose(
        raw["plugins_file"],
        allowlist=out["tool_allowlist"],
        catalog=out["catalog_desks"],
    )
    from .plugins import load_plugins

    out["plugin_ctx"] = load_plugins(out)
    return out


def _resolve_under_root(raw_path: Any, key: str) -> Path:
    text = str(raw_path or "").strip()
    if not text:
        raise HarnessConfigError(f"{key} must be set")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = _ROOT.parent / path
    return path


def _skill_ok(desk: str, skill: Any) -> Dict[str, Any]:
    if not isinstance(skill, dict):
        raise HarnessConfigError(f"plugins.skills.{desk} must be a mapping")
    tools = skill.get("required_tools")
    if not isinstance(tools, list) or any(not isinstance(n, str) or not n.strip() for n in tools):
        raise HarnessConfigError(f"plugins.skills.{desk}.required_tools must be a string list")
    success = str(skill.get("success") or "").strip()
    if not success:
        raise HarnessConfigError(f"plugins.skills.{desk}.success must be set")
    ret = skill.get("return_to_parent")
    if not isinstance(ret, list) or not ret or any(not str(x).strip() for x in ret):
        raise HarnessConfigError(
            f"plugins.skills.{desk}.return_to_parent must be a non-empty string list"
        )
    hints = str(skill.get("hints") or "").strip()
    return {
        "required_tools": [str(n).strip() for n in tools],
        "success": success,
        "return_to_parent": [str(x).strip() for x in ret],
        "hints": hints,
    }


def _load_plugins_compose(
    raw_path: Any,
    *,
    allowlist: List[str],
    catalog: List[str],
) -> Dict[str, Any]:
    path = _resolve_under_root(raw_path, "plugins_file")
    if not path.is_file():
        raise HarnessConfigError(f"harness plugins file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise HarnessConfigError(f"harness plugins file must be a mapping: {path}")
    for key in ("loop", "session", "planner", "tools", "skills"):
        if key not in raw:
            raise HarnessConfigError(f"harness plugins missing {key}: {path}")
    tools = _str_list(raw["tools"], "plugins.tools")
    expected = list(allowlist) + (["query_knowledge"] if "query_knowledge" not in allowlist else [])
    if tools != expected:
        raise HarnessConfigError(
            "plugins.tools must equal tool_allowlist plus query_knowledge "
            f"(expected {expected}, got {tools})"
        )
    skills_raw = raw["skills"]
    if not isinstance(skills_raw, dict):
        raise HarnessConfigError("plugins.skills must be a mapping")
    need = list(catalog) + (["generic_investigator"] if "generic_investigator" not in catalog else [])
    missing = [d for d in need if d not in skills_raw]
    if missing:
        raise HarnessConfigError(f"plugins.skills missing desks {missing}: {path}")
    skills = {str(desk): _skill_ok(str(desk), skills_raw[desk]) for desk in need}
    extra = [k for k in skills_raw if k not in skills]
    if extra:
        raise HarnessConfigError(f"plugins.skills has unknown desks {extra}: {path}")
    return {
        "loop": str(raw["loop"] or "").strip(),
        "session": str(raw["session"] or "").strip(),
        "planner": str(raw["planner"] or "").strip(),
        "tools": tools,
        "skills": skills,
        "path": str(path),
    }


def load_standing_memory(cfg: Dict[str, Any]) -> str:
    parts = [str(cfg["standing_rules"]).strip()]
    overlay = cfg.get("memory_overlay_path")
    if overlay is None:
        return parts[0]
    raw = yaml.safe_load(Path(overlay).read_text(encoding="utf-8"))
    if raw is None:
        raise HarnessConfigError(f"harness memory overlay is empty: {overlay}")
    if isinstance(raw, str):
        text = raw.strip()
    elif isinstance(raw, dict):
        text = str(raw.get("standing_rules") or raw.get("rules") or "").strip()
    else:
        raise HarnessConfigError(f"harness memory overlay must be a string or mapping: {overlay}")
    if not text:
        raise HarnessConfigError(f"harness memory overlay has no standing_rules: {overlay}")
    parts.append(text)
    return "\n\n".join(parts)
