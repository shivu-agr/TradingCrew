"""Invoke allowlisted desk tools from the conductor."""

from __future__ import annotations

import inspect
from typing import Any, Dict

from trading_crew.tools import ALL_TOOLS

from .knowledge import query_knowledge
from .safety import HarnessGateError, check_tool, tool_family


def _filter_args(fn: Any, args: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return dict(args)
    accepted = {
        name
        for name, p in sig.parameters.items()
        if name not in ("self", "cls") and p.kind != inspect.Parameter.VAR_KEYWORD
    }
    has_var = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if has_var:
        return dict(args)
    return {k: v for k, v in args.items() if k in accepted}


def invoke_tool(name: str, args: Dict[str, Any], *, cfg: Dict[str, Any]) -> str:
    payload = dict(args or {})
    if name == "query_knowledge":
        check_tool(name, payload, cfg)
        tags = payload.get("tags")
        if isinstance(tags, str):
            tags = [tags]
        return query_knowledge(
            str(payload.get("query") or ""),
            cfg=cfg,
            tags=tags,
        )
    if name not in ALL_TOOLS:
        raise HarnessGateError(f"unknown tool: {name}")
    check_tool(name, payload, cfg)
    tool = ALL_TOOLS[name]
    fn = getattr(tool, "func", None) or getattr(tool, "_python_fn", None) or tool
    filtered = _filter_args(fn, payload)
    if hasattr(tool, "run"):
        try:
            return str(tool.run(**filtered) or "")
        except TypeError:
            if "ticker" in filtered and len(filtered) == 1:
                return str(tool.run(filtered["ticker"]) or "")
            raise
    return str(fn(**filtered) or "")


def family_for(name: str) -> str | None:
    return tool_family(name)
