"""Walk-forward backtest harness (paper §8).

Three submodules:

- ``metrics``   — Performance / risk-adjusted return metrics (Sharpe,
                  Sortino, Calmar, deflated Sharpe).
- ``walk_forward`` — Walk-forward fold generator with mandatory embargo
                  between train and test windows.  No future leakage.
- ``manifest``  — Run manifest writer: git SHA, prompt hash, seed, cost
                  params, data hashes.  Every backtest result is paired
                  with a manifest so it can be replayed bit-for-bit.

The harness does *not* call the LLM agents — it consumes pre-recorded
``ActionProposal`` decisions (one per ticker per bar) and walks them
through the M2 execution pipeline + M5 risk gates, producing an equity
curve and metrics.  This split lets us backtest deterministically once
the LLM decisions have been logged, without paying the LLM cost on every
parameter sweep.
"""

from .metrics import (
    BacktestMetrics,
    compute_metrics,
    deflated_sharpe,
    max_drawdown_pct,
)
from .walk_forward import (
    CPCVConfig,
    CPCVFold,
    Fold,
    WalkForwardConfig,
    generate_cpcv_folds,
    generate_folds,
    walk_forward_cpcv,
)
from .manifest import (
    RunManifest,
    build_manifest,
    write_manifest,
    load_manifest,
    hash_data_dir,
    hash_text,
)
from .engine import (
    BacktestResult,
    FoldResult,
    TradeRecord,
    run_backtest,
    run_walk_forward,
)

__all__ = [
    "BacktestMetrics",
    "compute_metrics",
    "deflated_sharpe",
    "max_drawdown_pct",
    "Fold",
    "WalkForwardConfig",
    "generate_folds",
    "CPCVConfig",
    "CPCVFold",
    "generate_cpcv_folds",
    "walk_forward_cpcv",
    "RunManifest",
    "build_manifest",
    "write_manifest",
    "load_manifest",
    "hash_data_dir",
    "hash_text",
    "BacktestResult",
    "FoldResult",
    "TradeRecord",
    "run_backtest",
    "run_walk_forward",
]
