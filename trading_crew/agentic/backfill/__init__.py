"""Offline LLM decision-label backfill (Phase 2).

Re-runs the crew "as of" historical decision dates so the desk can build a
corpus of labeled decisions *now* instead of waiting months for live
decisions to resolve.  Every backfill run:

- executes the full crew inside a point-in-time :class:`AsOfContext`
  (Phase 1) so only information public on the decision date is used,
- records a ``source="backfill"`` episode (Phase 0 provenance stamped),
- immediately resolves the outcome (the horizon is already in the past),
  producing an alpha-labeled training row for L2 / L5.

Runs are **resumable** (a job-state checkpoint records which cells are
done) and **rate-limited** (a configurable inter-cell gap on top of the
screener throttle), and use the fast ``VLLM_LLM_*`` endpoint from ``.env``
by default.
"""

from __future__ import annotations

from .harness import (
    BackfillCell,
    BackfillConfig,
    generate_cells,
    job_id_for,
    load_state,
    run_backfill,
)

__all__ = [
    "BackfillCell",
    "BackfillConfig",
    "generate_cells",
    "job_id_for",
    "load_state",
    "run_backfill",
]
