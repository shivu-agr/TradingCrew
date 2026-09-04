"""Load the programmatic Prediction API run config.

The request supplies ticker, plus debate/risk rounds when the configured
``run_mode`` is not harness. Every other kickoff knob comes from
``trading_crew/config/api_run.yaml`` (or ``TRADINGCREW_API_CONFIG``).
Agent prompts stay in the packaged yaml files — they are not inlined here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_ROOT = Path(__file__).resolve().parent
_PACKAGED = _ROOT / "config" / "api_run.yaml"
_AGENTS = _ROOT / "config" / "agents.yaml"
_TASKS = _ROOT / "config" / "tasks.yaml"
_RUN_MODE = _ROOT / "config" / "run_mode_prompts.yaml"

REQUIRED_KEYS = (
    "asset_class",
    "run_mode",
    "quality_review",
    "compliance",
    "critic_enabled",
    "fusion_enabled",
    "fusion_include_crowd",
    "book",
    "llm_preset",
    "force_rerun",
    "tools_enabled",
)

BOOL_KEYS = (
    "quality_review",
    "compliance",
    "critic_enabled",
    "fusion_enabled",
    "fusion_include_crowd",
    "force_rerun",
)

RUN_MODES = ("autonomous", "deterministic", "harness")


class ApiRunConfigError(ValueError):
    """API run config file is missing or invalid."""


def config_path() -> Path:
    raw = (os.environ.get("TRADINGCREW_API_CONFIG") or "").strip()
    if raw:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = _ROOT.parent / path
        return path
    return _PACKAGED


def prompt_files() -> Dict[str, str]:
    cache = os.environ.get("TRADINGCREW_CACHE_DIR") or os.path.expanduser("~/.trading_crew")
    overlay = Path(cache) / "v3" / "prompt_overlay.yaml"
    return {
        "agents": str(_AGENTS),
        "tasks": str(_TASKS),
        "run_mode_prompts": str(_RUN_MODE),
        "overlay": str(overlay),
        "overlay_exists": overlay.is_file(),
    }


def load_api_run_config() -> Dict[str, Any]:
    path = config_path()
    if not path.is_file():
        raise ApiRunConfigError(f"API run config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ApiRunConfigError(f"API run config must be a mapping: {path}")
    missing = [k for k in REQUIRED_KEYS if k not in raw]
    if missing:
        raise ApiRunConfigError(f"API run config missing keys {missing}: {path}")
    mode = str(raw["run_mode"] or "").strip()
    if mode not in RUN_MODES:
        raise ApiRunConfigError(f"run_mode must be one of {list(RUN_MODES)}, not {mode!r}")
    tools = raw["tools_enabled"]
    if not isinstance(tools, dict):
        raise ApiRunConfigError("tools_enabled must be a mapping of desk -> tool list")
    cleaned_tools: Dict[str, list] = {}
    for key, names in tools.items():
        if names is None:
            raise ApiRunConfigError(f"tools_enabled.{key} is empty — use [] to disable tools")
        if not isinstance(names, list) or any(not isinstance(n, str) or not n.strip() for n in names):
            raise ApiRunConfigError(f"tools_enabled.{key} must be a list of tool names")
        cleaned_tools[str(key)] = [str(n).strip() for n in names]
    asset = str(raw["asset_class"] or "").strip()
    book = str(raw["book"] or "").strip()
    preset = (os.environ.get("TRADINGCREW_LLM_PRESET") or "").strip()
    if not preset:
        preset = str(raw.get("llm_preset") or "").strip()
    if not asset:
        raise ApiRunConfigError("asset_class must be set in the API run config")
    if not book:
        raise ApiRunConfigError("book must be set in the API run config")
    if not preset:
        raise ApiRunConfigError(
            "TRADINGCREW_LLM_PRESET must be set in .env "
            "(or llm_preset in the API run config)"
        )
    out: Dict[str, Any] = {
        "asset_class": asset,
        "run_mode": mode,
        "book": book,
        "llm_preset": preset,
        "tools_enabled": cleaned_tools,
        "config_path": str(path),
    }
    for key in BOOL_KEYS:
        val = raw[key]
        if not isinstance(val, bool):
            raise ApiRunConfigError(f"{key} must be a boolean, not {val!r}")
        out[key] = val
    return out


def build_session_cfg(
    *,
    ticker: str,
    debate_rounds: Optional[int] = None,
    risk_rounds: Optional[int] = None,
    force_rerun: Optional[bool] = None,
) -> Dict[str, Any]:
    """Merge the API request with the config file into a SessionManager payload."""
    symbol = (ticker or "").strip().upper()
    if not symbol:
        raise ApiRunConfigError("ticker is required")
    cfg = load_api_run_config()
    rerun = cfg["force_rerun"] if force_rerun is None else bool(force_rerun)
    out = {
        "asset_class": cfg["asset_class"],
        "ticker": symbol,
        "run_mode": cfg["run_mode"],
        "quality_review": cfg["quality_review"],
        "compliance": cfg["compliance"],
        "critic_enabled": cfg["critic_enabled"],
        "fusion_enabled": cfg["fusion_enabled"],
        "fusion_include_crowd": cfg["fusion_include_crowd"],
        "tools_enabled": cfg["tools_enabled"],
        "book": cfg["book"],
        "llm_preset": cfg["llm_preset"],
        "force_rerun": rerun,
    }
    if cfg["run_mode"] == "harness":
        return out
    if debate_rounds is None or debate_rounds < 1:
        raise ApiRunConfigError("debate_rounds must be >= 1")
    if risk_rounds is None or risk_rounds < 1:
        raise ApiRunConfigError("risk_rounds must be >= 1")
    out["debate_rounds"] = int(debate_rounds)
    out["risk_rounds"] = int(risk_rounds)
    return out
