"""Consolidated Settings page payload (strategy + prompts + bulk run snapshot).

Existing ``/api/strategy*`` endpoints stay; this is the one document the
``/settings`` page GET/PUT. Missing bulk file is ``None`` — not invented.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from trading_crew.agentic.prompt_overlay import (
    list_prompts,
    set_prompt,
    unset_prompt,
)
from trading_crew.agentic.self_learn import active_experiment
from trading_crew.agentic.strategy import (
    StrategyError,
    live_strategy_path,
    load_desk_strategy,
    packaged_strategy_path,
    render_prompt_block,
    revert_live_strategy,
    save_desk_strategy,
    strategy_source,
    strategy_to_raw,
)

BULK_KEYS = (
    "debate_rounds",
    "risk_rounds",
    "run_mode",
    "quality_review",
    "compliance",
    "critic_enabled",
    "fusion_enabled",
    "fusion_include_crowd",
    "tools_enabled",
    "book",
    "llm_preset",
)

# Operator sliders apply to commodity (always sequential) and deterministic
# stock. Autonomous and harness stock omit them — graph extra-debate + Hermes
# own loops, or the conductor chooses desks with no round quota.
DIALOGUE_KEYS = ("debate_rounds", "risk_rounds")
DROPPED_BULK_KEYS = ("embedding_preset", "memory")


class SettingsError(ValueError):
    """Settings payload is invalid."""


def settings_book(book: str) -> str:
    key = (book or "equity").strip().lower()
    if key in ("stock", "equity"):
        return "equity"
    if key == "commodity":
        return "commodity"
    raise SettingsError(f"book must be equity or commodity, not {book!r}")


def bulk_path() -> Path:
    cache = os.environ.get("TRADINGCREW_CACHE_DIR") or os.path.expanduser("~/.trading_crew")
    return Path(cache) / "v3" / "bulk_settings.json"


def operator_picks_dialogue(book: str, bulk: Dict[str, Any]) -> bool:
    if book == "commodity":
        return True
    return (bulk.get("run_mode") or "autonomous") == "deterministic"


def required_bulk_keys(book: str, bulk: Dict[str, Any]) -> tuple:
    if operator_picks_dialogue(book, bulk):
        return BULK_KEYS
    return tuple(k for k in BULK_KEYS if k not in DIALOGUE_KEYS)


def sanitize_bulk(book: str, bulk: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: bulk[k] for k in BULK_KEYS if k in bulk}
    for dropped in DROPPED_BULK_KEYS:
        out.pop(dropped, None)
    if not operator_picks_dialogue(book, out):
        for key in DIALOGUE_KEYS:
            out.pop(key, None)
    return out


def _read_bulk_file(path: Path) -> Dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SettingsError(f"bulk settings file is not JSON: {path}") from exc
    if not isinstance(raw, dict):
        raise SettingsError(f"bulk settings file is not a mapping: {path}")
    return raw


def _write_bulk_file(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".bulk_settings.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_bulk(book: str) -> Optional[Dict[str, Any]]:
    book = settings_book(book)
    path = bulk_path()
    if not path.is_file():
        return None
    raw = _read_bulk_file(path)
    row = raw.get(book)
    if row is None:
        return None
    if not isinstance(row, dict):
        raise SettingsError(f"bulk settings {book} is not a mapping")
    cleaned = sanitize_bulk(book, row)
    if any(k in row for k in DROPPED_BULK_KEYS):
        raw[book] = cleaned
        _write_bulk_file(path, raw)
    return cleaned


def save_bulk(book: str, bulk: Dict[str, Any]) -> Dict[str, Any]:
    book = settings_book(book)
    if not isinstance(bulk, dict):
        raise SettingsError("bulk must be a mapping")
    missing = [k for k in required_bulk_keys(book, bulk) if k not in bulk]
    if missing:
        raise SettingsError(f"bulk missing required keys {missing}")
    path = bulk_path()
    data: Dict[str, Any] = {}
    if path.is_file():
        data = _read_bulk_file(path)
    data[book] = sanitize_bulk(book, bulk)
    _write_bulk_file(path, data)
    return data[book]


def settings_payload(book: str) -> Dict[str, Any]:
    book = settings_book(book)
    s = load_desk_strategy()
    return {
        "book": book,
        "strategy": strategy_to_raw(s),
        "strategy_source": strategy_source(),
        "packaged_path": str(packaged_strategy_path()),
        "live_path": str(live_strategy_path()),
        "prompt_block": render_prompt_block(s, book=book, experiment=active_experiment()),
        "prompts": list_prompts(book),
        "knob_catalog": [
            {
                "id": k.id,
                "enabled": k.enabled,
                "apply": k.apply,
                "playbook_action": k.playbook_action,
                "size_scale": k.size_scale,
                "description": k.description,
            }
            for k in s.knobs
        ],
        "bulk": load_bulk(book),
    }


def save_settings(book: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    book = settings_book(book)
    if not isinstance(payload, dict):
        raise SettingsError("settings body must be a mapping")
    if payload.get("revert_strategy"):
        revert_live_strategy()
    elif "strategy" in payload:
        save_desk_strategy(payload["strategy"])
    if "prompts" in payload:
        rows = payload["prompts"]
        if not isinstance(rows, list):
            raise SettingsError("prompts must be a list of {key, goal, backstory}")
        for row in rows:
            if not isinstance(row, dict):
                raise SettingsError("each prompt must be a mapping")
            set_prompt(
                book,
                str(row.get("key") or ""),
                goal=str(row.get("goal") or ""),
                backstory=str(row.get("backstory") or ""),
            )
    if "unset_prompts" in payload:
        keys = payload["unset_prompts"]
        if not isinstance(keys, list):
            raise SettingsError("unset_prompts must be a list of agent keys")
        for key in keys:
            unset_prompt(book, str(key))
    if "bulk" in payload:
        save_bulk(book, payload["bulk"])
    return settings_payload(book)
