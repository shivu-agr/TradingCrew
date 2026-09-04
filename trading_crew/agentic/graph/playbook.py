"""Offline topology playbook: credit/debit routing keys from resolved WIN/LOSS.

Online routing reads the weights; a single LOSS never mutates the live crew.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from trading_crew.agentic.outcome import OutcomeLabel

logger = logging.getLogger(__name__)

_DEFAULT_NAME = "topology_playbook.json"

# Bounded online actions the router may take for a playbook key.
PLAYBOOK_ACTIONS = (
    "default",
    "skip_reports",
    "debate_rounds_1",
    "debate_rounds_2",
    "evidence_budget_0",
    "evidence_budget_1",
    "extra_debate_on_split",
    "no_trader_reopen",
    "risk_rounds_2",
)


def _path() -> Path:
    cache = os.environ.get("TRADINGCREW_CACHE_DIR") or os.path.expanduser(
        "~/.trading_crew"
    )
    return Path(cache) / _DEFAULT_NAME


def playbook_key(
    *,
    market: str = "",
    has_filings: Optional[bool] = None,
    options_listed: Optional[bool] = None,
    eventful: Optional[bool] = None,
) -> str:
    parts = [ (market or "UNK").upper()[:2] ]
    if has_filings is True:
        parts.append("filings")
    elif has_filings is False:
        parts.append("nofile")
    if options_listed is True:
        parts.append("opt")
    elif options_listed is False:
        parts.append("noopt")
    if eventful is True:
        parts.append("event")
    return "|".join(parts)


def load_playbook(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or _path()
    if not p.is_file():
        return {"keys": {}}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("topology playbook unreadable at %s", p, exc_info=True)
        return {"keys": {}}


def save_playbook(data: Dict[str, Any], path: Optional[Path] = None) -> None:
    p = path or _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".playbook.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
        os.replace(tmp, p)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def record_outcome(
    key: str,
    label: OutcomeLabel | str,
    *,
    path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Credit a WIN / debit a LOSS on ``key``.  HOLD_* is a no-op vote."""
    data = load_playbook(path)
    keys: Dict[str, Any] = data.setdefault("keys", {})
    row = keys.setdefault(key, {"wins": 0, "losses": 0, "flat": 0, "n": 0})
    lab = label.value if isinstance(label, OutcomeLabel) else str(label).upper()
    row["n"] = int(row.get("n") or 0) + 1
    if lab == OutcomeLabel.WIN.value:
        row["wins"] = int(row.get("wins") or 0) + 1
    elif lab == OutcomeLabel.LOSS.value:
        row["losses"] = int(row.get("losses") or 0) + 1
    else:
        row["flat"] = int(row.get("flat") or 0) + 1
    w, L = row["wins"], row["losses"]
    row["win_rate"] = (w / (w + L)) if (w + L) else None
    keys[key] = row
    save_playbook(data, path)
    return row


def preferred_action(key: str, *, path: Optional[Path] = None) -> str:
    """Best bounded action for this key — currently just ``default``.

    Weights are recorded so a later Mass topology search can pick among
    ``PLAYBOOK_ACTIONS``; until that search promotes a winner we do not
    change the live DAG from a cold playbook.
    """
    data = load_playbook(path)
    row = (data.get("keys") or {}).get(key) or {}
    # Explicit promoted action only — never infer a skip from a small n.
    action = row.get("promoted_action")
    if action in PLAYBOOK_ACTIONS:
        return str(action)
    return "default"
