"""L3 per-ticker tunes composed with an open Hermes size experiment.

L3 grid/PSO (:mod:`trading_crew.agentic.grid_search`) finds a ticker's
``SizingConfig`` / ``GateConfig``. Hermes may also open ``size_scale_down``
with an explicit ``size_scale`` from ``desk_strategy.yaml``. Live sizing
and walk-forward use the same composition:

- L3 applied tune is the per-ticker base (or built-in defaults if none).
- An open Hermes ``size_scale`` multiplies Kelly and max position weight.

Absence is honest: no tune and no Hermes scale ⇒ callers keep defaults.
``size_scale`` comes from the open experiment, or the YAML catalog
entry for that same knob id — never a hardcoded default.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from trading_crew.agentic.risk import GateConfig, SizingConfig

logger = logging.getLogger(__name__)

_FILENAME = "tuned_sizing.json"


def tuned_config_path(cache_dir: Optional[str] = None) -> Path:
    """Location of the per-ticker tuned-config store."""
    override = os.environ.get("TRADINGCREW_TUNED_CONFIG")
    if override:
        return Path(override)
    root = cache_dir or os.environ.get("TRADINGCREW_CACHE_DIR") or os.path.expanduser("~/.trading_crew")
    return Path(root) / _FILENAME


def _load_store(cache_dir: Optional[str] = None) -> dict:
    path = tuned_config_path(cache_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001 — a corrupt file must not break sizing
        logger.warning("tuning: could not parse %s — ignoring persisted tunes", path)
        return {}


def _filter_kwargs(cls, kwargs: Optional[dict]) -> dict:
    """Keep only kwargs that are real fields of ``cls`` (so a stale/renamed
    parameter in the store can never crash config construction)."""
    valid = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in (kwargs or {}).items() if k in valid}


def save_tuned_config(
    ticker: str,
    *,
    sizing_config: dict,
    gate_config: dict,
    meta: Optional[dict] = None,
    cache_dir: Optional[str] = None,
) -> Path:
    """Persist the chosen (sizing, gate) tune for ``ticker``.

    ``sizing_config`` / ``gate_config`` are the kwargs dicts from the winning
    grid/PSO point.  ``meta`` records provenance (rank metric, sample count,
    validation mode, …) for auditability.
    """
    store = _load_store(cache_dir)
    store[ticker.upper()] = {
        "sizing_config": dict(sizing_config or {}),
        "gate_config": dict(gate_config or {}),
        "meta": meta or {},
        "saved_ts": datetime.now(timezone.utc).isoformat(),
    }
    path = tuned_config_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, indent=2))
    return path


def load_tuned_entry(ticker: str, cache_dir: Optional[str] = None) -> Optional[dict]:
    """Return the raw persisted entry for ``ticker`` (config + meta) or None."""
    return _load_store(cache_dir).get(ticker.upper())


def load_tuned_config(
    ticker: str,
    cache_dir: Optional[str] = None,
) -> Optional[Tuple[SizingConfig, GateConfig]]:
    """Return ``(SizingConfig, GateConfig)`` for ``ticker`` when a tune exists.

    Returns ``None`` when the ticker has no saved tune (or the entry can't be
    reconstructed) — callers then keep their own defaults.
    """
    entry = load_tuned_entry(ticker, cache_dir)
    if not entry:
        return None
    try:
        sizing = SizingConfig(**_filter_kwargs(SizingConfig, entry.get("sizing_config")))
        gate = GateConfig(**_filter_kwargs(GateConfig, entry.get("gate_config")))
        return sizing, gate
    except Exception:  # noqa: BLE001 — never let a bad tune block sizing
        logger.warning("tuning: could not build configs for %s — using defaults", ticker)
        return None


def hermes_size_scale(journal_file: Optional[Path] = None) -> Optional[float]:
    """Return the open Hermes ``size_scale`` when present and valid.

    Prefer the value stamped on the experiment. If the journal predates
    that field, use the YAML catalog entry for the same knob id — that
    is the experiment's contract, not an invented default.
    """
    from trading_crew.agentic.self_learn import active_experiment
    from trading_crew.agentic.strategy import load_desk_strategy

    exp = active_experiment(journal_file)
    if not exp:
        return None
    raw = exp.get("size_scale")
    if raw is None or str(raw).strip() == "":
        kid = str(exp.get("knob") or "")
        knob = next((k for k in load_desk_strategy().knobs if k.id == kid), None)
        if knob is None or knob.size_scale is None:
            return None
        raw = knob.size_scale
    scale = float(raw)
    if scale <= 0:
        raise ValueError(f"open experiment size_scale must be > 0, got {scale}")
    return scale


def resolve_live_sizing(
    ticker: str,
    *,
    cache_dir: Optional[str] = None,
    journal_file: Optional[Path] = None,
) -> Tuple[Optional[SizingConfig], Optional[GateConfig], Dict[str, Any]]:
    """Compose L3 tune (if applied) with an open Hermes size experiment.

    Returns ``(sizing, gate, meta)``. Both configs are ``None`` when neither
    an L3 tune nor a Hermes scale is active — callers then use defaults.
    """
    tuned = load_tuned_config(ticker, cache_dir)
    scale = hermes_size_scale(journal_file)
    meta: Dict[str, Any] = {
        "l3_applied": tuned is not None,
        "hermes_size_scale": scale,
    }
    if tuned is None and scale is None:
        return None, None, meta
    if tuned is not None:
        sizing, gate = tuned
    else:
        sizing, gate = SizingConfig(), GateConfig()
    if scale is not None:
        sizing = dataclasses.replace(
            sizing,
            kelly_fraction=sizing.kelly_fraction * scale,
            max_position_weight=sizing.max_position_weight * scale,
        )
        gate = dataclasses.replace(
            gate,
            max_position_weight=gate.max_position_weight * scale,
        )
    return sizing, gate, meta
