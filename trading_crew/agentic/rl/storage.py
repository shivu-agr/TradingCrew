"""Persistence layout for L4 RL training runs.

Layout
------
``$TRADINGCREW_CACHE_DIR/rl_runs/`` (default: ``~/.trading_crew/rl_runs/``)::

    <ticker>/
      <run_id>/
        record.json          # RLRunRecord serialised
        policy.pt            # torch checkpoint
        metrics.jsonl        # one TrainingMetrics per line (for streaming)

``$TRADINGCREW_CACHE_DIR/rl_runs/promoted/<ticker>.json``::

    { "ticker": "...", "run_id": "...", "promoted_ts": "..." }

The promoted symlink (well, json pointer — symlinks are portability
trouble on Windows) is what the inference client + CrewAI tool look at
when answering "do you have a trained policy for NVDA?".  Promotion is
an explicit user action so a half-trained run never silently leaks into
production decisions.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _cache_root() -> Path:
    root = os.environ.get("TRADINGCREW_CACHE_DIR") or os.path.expanduser("~/.trading_crew")
    return Path(root)


RL_RUN_DIR: Path = _cache_root() / "rl_runs"
PROMOTED_LINK_DIR: Path = RL_RUN_DIR / "promoted"


# ---------------------------------------------------------------------------
# RLRunRecord
# ---------------------------------------------------------------------------


@dataclass
class RLRunRecord:
    """Everything we want to keep about a single L4 training run.

    Persisted as JSON next to the torch checkpoint.  All numeric fields
    are JSON-serialisable directly (no numpy dtypes).
    """

    run_id: str
    ticker: str
    asset_class: str
    created_ts: str
    status: str  # "running" | "completed" | "stopped" | "failed"

    # Config snapshots — full enough that the run is reproducible.
    env_config: Dict[str, Any] = field(default_factory=dict)
    ppo_config: Dict[str, Any] = field(default_factory=dict)
    data_window: Dict[str, Any] = field(default_factory=dict)
    """Keys: ``train_start``, ``train_end``, ``eval_start``, ``eval_end``,
    ``bars_train``, ``bars_eval``."""

    # Outputs — populated as training progresses / finishes.
    metrics: List[Dict[str, Any]] = field(default_factory=list)
    """Each entry is a serialised TrainingMetrics."""
    eval_result: Optional[Dict[str, Any]] = None
    """The evaluate() result on the held-out window."""
    duration_sec: float = 0.0
    error: Optional[str] = None

    # Optional buy-and-hold baseline computed from the eval window
    # OHLCV.  Lets the UI say "your policy returned +12% vs +8% B&H".
    baseline_buy_and_hold: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RLRunRecord":
        return cls(**data)


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def _run_dir(ticker: str, run_id: str) -> Path:
    return RL_RUN_DIR / ticker.upper() / run_id


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Atomic JSON write via tmp+rename — same pattern as PortfolioState."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(tmp, path)


def save_run(record: RLRunRecord) -> Path:
    """Write the run record JSON.  Idempotent — overwrites in place."""
    target = _run_dir(record.ticker, record.run_id) / "record.json"
    _atomic_write_json(target, record.to_dict())
    return target


def load_run(ticker: str, run_id: str) -> Optional[RLRunRecord]:
    """Load a run by (ticker, run_id) or ``None`` if missing."""
    path = _run_dir(ticker, run_id) / "record.json"
    if not path.exists():
        return None
    try:
        return RLRunRecord.from_dict(json.loads(path.read_text()))
    except Exception:
        return None


def list_runs(ticker: Optional[str] = None, limit: int = 50) -> List[RLRunRecord]:
    """List runs across all tickers (most recent first).

    If ``ticker`` is given, restrict to that ticker.  Anything that
    fails to deserialise is skipped silently — the leaderboard should
    degrade gracefully rather than 500 on a single bad record.
    """
    out: List[RLRunRecord] = []
    if not RL_RUN_DIR.exists():
        return out

    tickers: List[str] = (
        [ticker.upper()] if ticker
        else sorted(d.name for d in RL_RUN_DIR.iterdir() if d.is_dir() and d.name != "promoted")
    )
    for tk in tickers:
        tk_dir = RL_RUN_DIR / tk
        if not tk_dir.exists():
            continue
        for run_dir in sorted(tk_dir.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            rec_path = run_dir / "record.json"
            if not rec_path.exists():
                continue
            try:
                out.append(RLRunRecord.from_dict(json.loads(rec_path.read_text())))
            except Exception:
                continue
            if len(out) >= limit:
                return out
    return out


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------


def promote_run(ticker: str, run_id: str) -> Dict[str, Any]:
    """Mark ``run_id`` as the active policy for ``ticker``.

    Writes ``promoted/<ticker>.json`` pointing at the run.  Raises if
    the run or its checkpoint is missing — promoting a run whose
    checkpoint failed to save would silently bake a broken pointer.
    """
    record = load_run(ticker, run_id)
    if record is None:
        raise FileNotFoundError(f"No run record for {ticker}/{run_id}")
    ckpt = _run_dir(ticker, run_id) / "policy.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Run {ticker}/{run_id} has no policy.pt checkpoint")
    PROMOTED_LINK_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "ticker": ticker.upper(),
        "run_id": run_id,
        "promoted_ts": datetime.now(timezone.utc).isoformat(),
        "asset_class": record.asset_class,
        "summary": record.eval_result,
    }
    _atomic_write_json(PROMOTED_LINK_DIR / f"{ticker.upper()}.json", payload)
    return payload


def get_promoted(ticker: str) -> Optional[Dict[str, Any]]:
    """Return the promoted-policy pointer for ``ticker``, or ``None``."""
    path = PROMOTED_LINK_DIR / f"{ticker.upper()}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def list_promoted() -> List[Dict[str, Any]]:
    """List every promoted policy across all tickers."""
    out: List[Dict[str, Any]] = []
    if not PROMOTED_LINK_DIR.exists():
        return out
    for path in sorted(PROMOTED_LINK_DIR.glob("*.json")):
        try:
            out.append(json.loads(path.read_text()))
        except Exception:
            continue
    return out


def policy_checkpoint_path(ticker: str, run_id: str) -> Path:
    """Path where the torch checkpoint lives for ``(ticker, run_id)``.

    Just a path builder — callers can pre-construct the path before the
    checkpoint exists (e.g. to hand to ``PPOTrainer.save_checkpoint``).
    """
    p = _run_dir(ticker, run_id) / "policy.pt"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def append_metric_jsonl(ticker: str, run_id: str, metric: Dict[str, Any]) -> None:
    """Append one metric line to the streaming JSONL log.

    The UI polling endpoint reads the tail of this file so the live
    chart updates without the run record JSON being re-written every
    rollout (which would be ~50 KB per rewrite).
    """
    target = _run_dir(ticker, run_id) / "metrics.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a") as f:
        f.write(json.dumps(metric, default=str) + "\n")


def read_metrics_jsonl(ticker: str, run_id: str) -> List[Dict[str, Any]]:
    """Read the full metrics stream for a run.  Empty list if missing."""
    path = _run_dir(ticker, run_id) / "metrics.jsonl"
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def new_run_id() -> str:
    """Generate a sortable run id — UTC timestamp down to milliseconds."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
