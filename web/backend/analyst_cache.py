"""Phase 2E — analyst output cache.

Persists each analyst task's final markdown payload keyed by
``(ticker, tools_enabled hash, prompt_hash, trade_date)`` so subsequent
runs with the *same* configuration can skip the analyst LLM calls.

Cache layout::

    ~/.trading_crew/cache/analyst/<sha1>.json

Each entry::

    {
      "key": {
        "ticker": "NVDA",
        "trade_date": "2026-06-12",
        "tools_hash": "5d…",
        "prompt_hash": "a1…",
      },
      "agent_role": "Market Analyst",
      "task_id": "market_task",
      "raw": "<markdown body>",
      "written_at": "2026-06-12T16:33:21Z",
    }

Public API:

- :func:`make_cache_key` — deterministic key from ``(ticker, …)``.
- :func:`load_entry`     — return the cached payload for a key (or None).
- :func:`save_entry`     — persist a payload at a key.
- :func:`cache_path_for` — file path of a key (useful for debugging).
- :func:`clear`          — bulk-delete the cache directory.

Read-side wiring lives in :mod:`runner` — the runner consults
:func:`load_entry` before kicking off the crew and short-circuits the
analyst phase when every analyst task has a cache hit.  Debate / risk
/ PM still re-run regardless (their outputs depend on the upstream
research + the current portfolio state, which are not cacheable in the
same way).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


def _cache_root() -> Path:
    base = os.environ.get("TRADINGCREW_CACHE_DIR") or os.path.expanduser("~/.trading_crew")
    p = Path(base) / "cache" / "analyst"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _stable_hash(payload: Any) -> str:
    """SHA-1 of a JSON-serialised, sort-key-stable payload."""
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


@dataclass(frozen=True)
class CacheKey:
    ticker: str
    trade_date: str
    tools_hash: str
    prompt_hash: str
    task_id: str

    def digest(self) -> str:
        return _stable_hash({
            "ticker": self.ticker.upper(),
            "trade_date": self.trade_date,
            "tools_hash": self.tools_hash,
            "prompt_hash": self.prompt_hash,
            "task_id": self.task_id,
        })

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker.upper(),
            "trade_date": self.trade_date,
            "tools_hash": self.tools_hash,
            "prompt_hash": self.prompt_hash,
            "task_id": self.task_id,
        }


def make_cache_key(
    *,
    ticker: str,
    trade_date: str,
    tools_enabled: Dict[str, List[str]],
    task_id: str,
    prompt: str = "",
) -> CacheKey:
    """Build a :class:`CacheKey` from the runner's UI config + task id.

    ``tools_enabled`` is normalised (keys sorted, values sorted) so two
    configs that differ only in iteration order produce the same hash.
    ``prompt`` is optional — pass the task description (with any
    parametrised round numbers expanded) when caching debate / risk
    tasks; the analyst tasks ship a fixed prompt so the empty string
    works.
    """
    norm_tools = {k: sorted(v or []) for k, v in sorted((tools_enabled or {}).items())}
    return CacheKey(
        ticker=ticker.upper(),
        trade_date=trade_date,
        tools_hash=_stable_hash(norm_tools),
        prompt_hash=_stable_hash(prompt),
        task_id=task_id,
    )


def cache_path_for(key: CacheKey) -> Path:
    return _cache_root() / f"{key.digest()}.json"


def load_entry(key: CacheKey) -> Optional[Dict[str, Any]]:
    path = cache_path_for(key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def save_entry(key: CacheKey, *, agent_role: str, raw: str) -> Path:
    path = cache_path_for(key)
    payload = {
        "key": key.to_dict(),
        "agent_role": agent_role,
        "task_id": key.task_id,
        "raw": raw,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def clear() -> int:
    root = _cache_root()
    n = 0
    for f in root.glob("*.json"):
        try:
            f.unlink()
            n += 1
        except OSError:
            pass
    return n
