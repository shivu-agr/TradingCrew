"""Data-driven crowd-weight learner.

The base crowd signal (:mod:`trading_crew.agentic.swarm.crowd`) fuses its
components with **equal, hand-set weights** (all ``w_* = 1.0``).  This module
replaces those with weights *learned from realised outcomes*: it regresses the
per-component crowd contributions logged on each episode against the episode's
realised ``alpha_return`` and turns each component's predictive strength into a
fusion weight.

Design decisions
----------------
- **Importance weighting, not sign-flipping.**  Each component contribution is
  already signed toward "bullish crowd" (+).  We keep that structure and only
  learn *how much* each component should count, so the downstream fade logic is
  unchanged.  A component's weight ∝ ``|standardised ridge coefficient|`` — how
  much it actually moves forward alpha.  The (signed) coefficient is persisted
  as a diagnostic so the desk can see contrarian (−) vs momentum (+) behaviour.
- **Cold-start safe.**  Fitting requires a minimum corpus (``min_samples``) and
  a minimum per-component observation count.  Components below the threshold, or
  a corpus that is too small, are simply *not learned* — the existing equal
  weight (1.0) is retained.  No weights are invented.
- **Opt-in load.**  :func:`load_crowd_config` returns the module default until a
  weights file has been fitted and saved; nothing changes silently.

CLI::

    python -m trading_crew.agentic.swarm.crowd_fit --min-samples 30
    python -m trading_crew.agentic.swarm.crowd_fit --dry-run   # fit + report only
"""

from __future__ import annotations

import dataclasses
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from trading_crew.agentic.swarm.crowd import CrowdConfig

logger = logging.getLogger(__name__)

# Stable ordering — must match the keys emitted by ``compute_crowd_signal``.
COMPONENT_KEYS: List[str] = [
    "put_call",
    "iv_skew",
    "insider",
    "analyst",
    "institutional",
    "cot",
]

_WEIGHTS_FILENAME = "crowd_weights.json"


def weights_path(cache_dir: Optional[str] = None) -> Path:
    """Location of the fitted weights file."""
    import os

    override = os.environ.get("TRADINGCREW_CROWD_WEIGHTS")
    if override:
        return Path(override)
    root = cache_dir or os.environ.get("TRADINGCREW_CACHE_DIR") or os.path.expanduser("~/.trading_crew")
    return Path(root) / _WEIGHTS_FILENAME


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


def _collect_rows(episodes: Sequence["object"]):
    """Extract ``(components_dict, alpha_return)`` pairs from resolved episodes
    that carry a per-component crowd breakdown."""
    rows = []
    for ep in episodes:
        comps = getattr(ep, "crowd_components", None)
        alpha = getattr(ep, "alpha_return", None)
        status = getattr(ep, "outcome_status", None)
        status_val = getattr(status, "value", status)
        if not comps or alpha is None or status_val != "RESOLVED":
            continue
        try:
            alpha = float(alpha)
        except (TypeError, ValueError):
            continue
        rows.append((comps, alpha))
    return rows


def fit_crowd_weights(
    episodes: Sequence["object"],
    *,
    min_samples: int = 30,
    min_component_obs: int = 10,
    ridge_lambda: float = 1.0,
) -> Optional[dict]:
    """Fit per-component crowd fusion weights from realised alpha.

    Returns a result dict (see keys below) or ``None`` when the corpus is too
    small / carries no usable signal — in which case callers keep the equal
    default weights.

    Result keys: ``weights`` (``w_*`` → float), ``coefficients`` (signed,
    diagnostic), ``component_obs``, ``n_samples``, ``ridge_lambda``,
    ``target``, ``fit_ts``.
    """
    import numpy as np

    rows = _collect_rows(episodes)
    n = len(rows)
    if n < min_samples:
        logger.info("crowd_fit: %d resolved+labelled rows < min_samples=%d — skipping", n, min_samples)
        return None

    # Per-component observation count decides which components are learnable.
    comp_obs: Dict[str, int] = {k: 0 for k in COMPONENT_KEYS}
    for comps, _alpha in rows:
        for k in COMPONENT_KEYS:
            if comps.get(k) is not None:
                comp_obs[k] += 1
    learnable = [k for k in COMPONENT_KEYS if comp_obs[k] >= min_component_obs]
    if not learnable:
        logger.info("crowd_fit: no component reached min_component_obs=%d — skipping", min_component_obs)
        return None

    # Design matrix over learnable components.  Standardise each column on its
    # observed values; missing entries map to 0 (the column mean after
    # centering) so they neither pull nor push the fit.
    y = np.array([a for _c, a in rows], dtype=float)
    y = y - y.mean()

    X = np.zeros((n, len(learnable)), dtype=float)
    col_stats: Dict[str, dict] = {}
    for j, k in enumerate(learnable):
        vals = np.array([r[0].get(k) for r in rows], dtype=object)
        present = np.array([v is not None for v in vals])
        obs = np.array([float(v) for v in vals[present]], dtype=float)
        mu = float(obs.mean()) if obs.size else 0.0
        sd = float(obs.std()) if obs.size else 0.0
        col_stats[k] = {"mean": mu, "std": sd}
        if sd <= 1e-9:
            continue  # constant column carries no information
        col = np.zeros(n, dtype=float)
        col[present] = (obs - mu) / sd
        X[:, j] = col

    # Ridge closed form: (XᵀX + λI)⁻¹ Xᵀy.
    k = X.shape[1]
    coef = np.linalg.solve(X.T @ X + ridge_lambda * np.eye(k), X.T @ y)
    abs_coef = np.abs(coef)
    mean_abs = float(abs_coef.mean())
    if mean_abs <= 1e-9:
        logger.info("crowd_fit: fitted coefficients are ~0 (no signal) — skipping")
        return None

    # Normalise so the mean learned weight = 1.0, preserving the scale of the
    # old equal-weight default while re-distributing emphasis by importance.
    norm_weights = abs_coef / mean_abs

    weights: Dict[str, float] = {}
    coefficients: Dict[str, float] = {}
    for j, key in enumerate(learnable):
        weights[f"w_{key}"] = round(float(norm_weights[j]), 6)
        coefficients[key] = round(float(coef[j]), 6)

    return {
        "weights": weights,
        "coefficients": coefficients,
        "component_obs": comp_obs,
        "n_samples": n,
        "ridge_lambda": ridge_lambda,
        "target": "alpha_return",
        "fit_ts": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Persist / load
# ---------------------------------------------------------------------------


def save_weights(result: dict, cache_dir: Optional[str] = None) -> Path:
    path = weights_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    return path


def load_weights(cache_dir: Optional[str] = None) -> Optional[dict]:
    path = weights_path(cache_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001 — a corrupt file must not break scoring
        logger.warning("crowd_fit: could not parse %s — using default weights", path)
        return None


def load_crowd_config(
    cache_dir: Optional[str] = None,
    base: Optional[CrowdConfig] = None,
) -> CrowdConfig:
    """Return a :class:`CrowdConfig` with *learned* weights when available.

    Falls back to ``base`` (or the module default) when no weights file has been
    fitted yet — i.e. the existing equal-weight behaviour, nothing invented.
    Only the ``w_*`` fusion weights are overridden; enabled / threshold / fade
    parameters come from ``base``.
    """
    cfg = base or CrowdConfig()
    result = load_weights(cache_dir)
    if not result:
        return cfg
    learned = result.get("weights") or {}
    overrides = {
        field: float(learned[field])
        for field in (f"w_{k}" for k in COMPONENT_KEYS)
        if field in learned
    }
    if not overrides:
        return cfg
    return dataclasses.replace(cfg, **overrides)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    from trading_crew.agentic.evaluation import default_cache_dir
    from trading_crew.agentic.memory import EpisodicMemory

    parser = argparse.ArgumentParser(description="Fit data-driven crowd fusion weights from realised alpha.")
    parser.add_argument("--cache-dir", default=None, help="Episode/memory root (default: TRADINGCREW_CACHE_DIR or ~/.trading_crew)")
    parser.add_argument("--min-samples", type=int, default=30, help="Minimum resolved+labelled episodes required to fit")
    parser.add_argument("--min-component-obs", type=int, default=10, help="Minimum per-component observations to learn a weight")
    parser.add_argument("--ridge-lambda", type=float, default=1.0, help="Ridge (L2) regularisation strength")
    parser.add_argument("--dry-run", action="store_true", help="Fit + print the report but do NOT save the weights file")
    args = parser.parse_args(argv)

    cache_dir = args.cache_dir or default_cache_dir()
    mem = EpisodicMemory(Path(cache_dir) / "memory" / "episodes.jsonl")
    episodes = list(mem.all_episodes())
    labelled = _collect_rows(episodes)
    print(f"episodes total={len(episodes)}  resolved+crowd-labelled={len(labelled)}")

    result = fit_crowd_weights(
        episodes,
        min_samples=args.min_samples,
        min_component_obs=args.min_component_obs,
        ridge_lambda=args.ridge_lambda,
    )
    if result is None:
        print(
            "Not enough labelled corpus to fit yet. Keep running analyses with "
            "TRADINGCREW_CROWD_SIGNAL_ENABLED=1 and resolve outcomes; the default "
            "equal weights stay in effect until then."
        )
        return 1

    print(f"\nFitted on {result['n_samples']} samples (target={result['target']}, λ={result['ridge_lambda']}):")
    for k in COMPONENT_KEYS:
        wkey = f"w_{k}"
        if wkey in result["weights"]:
            w = result["weights"][wkey]
            c = result["coefficients"].get(k, 0.0)
            direction = "contrarian" if c < 0 else "momentum"
            print(f"  {k:<14} weight={w:<8} coef={c:<+10} ({direction}, obs={result['component_obs'][k]})")
        else:
            print(f"  {k:<14} weight=1.0 (default — obs={result['component_obs'][k]} < min)")

    if args.dry_run:
        print("\n--dry-run: not saved.")
        return 0
    path = save_weights(result, cache_dir)
    print(f"\nSaved → {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
