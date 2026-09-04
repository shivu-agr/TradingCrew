"""Operator prompt overlays — Settings page, not the packaged yaml.

Packaged ``agents.yaml`` is the checked-in persona. An overlay key replaces
goal/backstory for that desk on the next run. Unset removes the key.
Unknown agent keys are rejected; empty goal/backstory is rejected.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import yaml

_ROOT = Path(__file__).resolve().parents[2]
_PACKAGED = {
    "equity": _ROOT / "trading_crew" / "config" / "agents.yaml",
    "commodity": _ROOT / "commodity_crew" / "config" / "agents.yaml",
}

_BOOKS = ("equity", "commodity")


class PromptOverlayError(ValueError):
    """Overlay file or payload is invalid."""


def overlay_path() -> Path:
    cache = os.environ.get("TRADINGCREW_CACHE_DIR") or os.path.expanduser("~/.trading_crew")
    return Path(cache) / "v3" / "prompt_overlay.yaml"


def packaged_agents_path(book: str) -> Path:
    key = _book(book)
    return _PACKAGED[key]


def _book(book: str) -> str:
    key = (book or "equity").strip().lower()
    if key not in _BOOKS:
        raise PromptOverlayError(f"book must be equity or commodity, not {book!r}")
    return key


def load_packaged_prompts(book: str = "equity") -> Dict[str, Dict[str, str]]:
    path = packaged_agents_path(book)
    if not path.is_file():
        raise PromptOverlayError(f"packaged agents file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise PromptOverlayError(f"agents yaml root must be a mapping: {path}")
    out: Dict[str, Dict[str, str]] = {}
    for key, row in raw.items():
        if not isinstance(row, dict):
            continue
        if "role" not in row or "goal" not in row or "backstory" not in row:
            continue
        out[str(key)] = {
            "key": str(key),
            "role": str(row["role"]).strip(),
            "goal": str(row["goal"]).strip(),
            "backstory": str(row["backstory"]).strip(),
        }
    if not out:
        raise PromptOverlayError(f"no agent personas in {path}")
    return out


def load_overlay() -> Dict[str, Dict[str, Dict[str, str]]]:
    path = overlay_path()
    if not path.is_file():
        return {book: {} for book in _BOOKS}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {book: {} for book in _BOOKS}
    if not isinstance(raw, dict):
        raise PromptOverlayError(f"prompt overlay is not a mapping: {path}")
    out: Dict[str, Dict[str, Dict[str, str]]] = {book: {} for book in _BOOKS}
    for book in _BOOKS:
        block = raw.get(book) or {}
        if not isinstance(block, dict):
            raise PromptOverlayError(f"prompt overlay {book} must be a mapping")
        for key, row in block.items():
            if not isinstance(row, dict):
                raise PromptOverlayError(f"prompt overlay {book}.{key} must be a mapping")
            out[book][str(key)] = {
                "goal": str(row["goal"]).strip() if "goal" in row else "",
                "backstory": str(row["backstory"]).strip() if "backstory" in row else "",
            }
    return out


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".prompt_overlay.", suffix=".tmp")
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


def save_overlay(data: Dict[str, Dict[str, Dict[str, str]]]) -> None:
    payload = {book: dict(data.get(book) or {}) for book in _BOOKS}
    dump = yaml.safe_dump(payload, sort_keys=True, allow_unicode=True, width=88)
    _atomic_write_text(overlay_path(), dump)


def list_prompts(book: str = "equity") -> List[Dict[str, Any]]:
    packaged = load_packaged_prompts(book)
    overlay = load_overlay().get(_book(book)) or {}
    rows: List[Dict[str, Any]] = []
    for key, base in packaged.items():
        over = overlay.get(key) or {}
        goal = (over.get("goal") or "").strip() or base["goal"]
        backstory = (over.get("backstory") or "").strip() or base["backstory"]
        rows.append({
            "key": key,
            "role": base["role"],
            "goal": goal,
            "backstory": backstory,
            "overlay": bool(over),
        })
    return rows


def set_prompt(
    book: str,
    key: str,
    *,
    goal: str,
    backstory: str,
) -> Dict[str, Any]:
    book = _book(book)
    packaged = load_packaged_prompts(book)
    if key not in packaged:
        raise PromptOverlayError(f"unknown agent {key!r} for {book}")
    goal_t = (goal or "").strip()
    back_t = (backstory or "").strip()
    if not goal_t or not back_t:
        raise PromptOverlayError("goal and backstory must be non-empty")
    data = load_overlay()
    data[book][key] = {"goal": goal_t, "backstory": back_t}
    save_overlay(data)
    return {
        "key": key,
        "role": packaged[key]["role"],
        "goal": goal_t,
        "backstory": back_t,
        "overlay": True,
    }


def unset_prompt(book: str, key: str) -> bool:
    book = _book(book)
    packaged = load_packaged_prompts(book)
    if key not in packaged:
        raise PromptOverlayError(f"unknown agent {key!r} for {book}")
    data = load_overlay()
    if key not in data[book]:
        return False
    del data[book][key]
    save_overlay(data)
    return True
