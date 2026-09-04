"""Packaged autonomous / deterministic / harness prompt contracts.

``agents.yaml`` is desk identity. This module appends the orchestrator
contract after operator overlays so Settings cannot wipe it. Missing
desk/task keys are skipped — apply does not invent copy.

Harness is a third contract: it **replaces** ``tasks.yaml`` descriptions
(mandatory tool order, round N of M) instead of appending an addendum.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_CONFIG = Path(__file__).resolve().parents[1] / "config" / "run_mode_prompts.yaml"
_MODES = ("autonomous", "deterministic", "harness")
_HARNESS_TASKS = (
    "market_task",
    "news_task",
    "fundamentals_task",
    "reports_task",
    "macro_task",
    "bull_round",
    "bear_round",
    "debate_bull_open",
    "debate_bear_rebut",
    "debate_bull_close",
    "research_synthesis",
    "trader_task",
    "risk_aggressive_round",
    "risk_conservative_round",
    "risk_neutral_round",
    "risk_aggressive_open",
    "risk_conservative_open",
    "risk_neutral_open",
    "risk_aggressive_rebut",
    "risk_conservative_rebut",
    "risk_neutral_rebut",
    "pm_decision",
    "quality_review",
    "compliance_review",
)


class RunModePromptError(ValueError):
    """Run-mode prompt file is missing or invalid."""


def prompt_run_mode(book: str, run_mode: str) -> str:
    if (book or "").strip().lower() == "commodity":
        return "deterministic"
    mode = (run_mode or "autonomous").strip().lower()
    if mode == "deterministic":
        return "deterministic"
    if mode == "harness":
        return "harness"
    return "autonomous"


@lru_cache(maxsize=1)
def load_run_mode_spec() -> Dict[str, Any]:
    if not _CONFIG.is_file():
        raise RunModePromptError(f"run mode prompts file is missing: {_CONFIG}")
    raw = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RunModePromptError(f"run mode prompts root must be a mapping: {_CONFIG}")
    for mode in _MODES:
        block = raw.get(mode)
        if not isinstance(block, dict):
            raise RunModePromptError(f"run_mode_prompts.yaml missing {mode!r} mapping")
        agents = block.get("agents")
        if agents is not None and not isinstance(agents, dict):
            raise RunModePromptError(f"run_mode_prompts.yaml {mode}.agents must be a mapping")
        tasks = block.get("tasks")
        if tasks is not None and not isinstance(tasks, dict):
            raise RunModePromptError(f"run_mode_prompts.yaml {mode}.tasks must be a mapping")
        if mode == "harness":
            if block.get("replace_tasks") is not True:
                raise RunModePromptError("run_mode_prompts.yaml harness.replace_tasks must be true")
            for name in _HARNESS_TASKS:
                if not _task_description(tasks.get(name) if tasks else None):
                    raise RunModePromptError(
                        f"run_mode_prompts.yaml harness.tasks.{name} must have a description"
                    )
    return raw


def _task_description(raw: Any) -> str:
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, dict):
        return str(raw.get("description") or "").strip()
    return ""


def _append_backstory(ag: Any, block: str) -> None:
    text = (block or "").strip()
    if not text:
        return
    try:
        current = ag.backstory or ""
        ag.backstory = f"{current}\n\n{text}".strip() if current else text
    except (AttributeError, TypeError):
        return


def apply_run_mode_prompts(
    agents: Any,
    *,
    book: str = "equity",
    run_mode: str = "autonomous",
) -> List[str]:
    from trading_crew.agentic.harness import _agent_key, _iter_keyed

    mode = prompt_run_mode(book, run_mode)
    spec = load_run_mode_spec()[mode]
    rows = spec.get("agents") or {}
    applied: List[str] = []
    for key, ag in _iter_keyed(agents):
        desk = _agent_key(ag, key)
        if not desk:
            continue
        row = rows.get(desk)
        if not isinstance(row, dict):
            continue
        extra = (row.get("backstory") or "").strip()
        if not extra:
            continue
        _append_backstory(ag, extra)
        applied.append(desk)
    return applied


def task_addendum(
    task_name: str,
    *,
    book: str = "equity",
    run_mode: str = "autonomous",
) -> str:
    mode = prompt_run_mode(book, run_mode)
    spec = load_run_mode_spec()[mode]
    if spec.get("replace_tasks"):
        return ""
    extra = (spec.get("tasks") or {}).get(task_name)
    return extra.strip() if isinstance(extra, str) and extra.strip() else ""


def task_override(
    task_name: str,
    *,
    book: str = "equity",
    run_mode: str = "autonomous",
) -> Optional[Dict[str, str]]:
    """Full task body when the mode replaces ``tasks.yaml`` (harness)."""
    mode = prompt_run_mode(book, run_mode)
    spec = load_run_mode_spec()[mode]
    if not spec.get("replace_tasks"):
        return None
    raw = (spec.get("tasks") or {}).get(task_name)
    desc = _task_description(raw)
    if not desc:
        return None
    out: Dict[str, str] = {"description": desc}
    if isinstance(raw, dict):
        expected = str(raw.get("expected_output") or "").strip()
        if expected:
            out["expected_output"] = expected
    return out
