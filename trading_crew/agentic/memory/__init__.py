"""Audit-grade memory — paper §4.2 (episodic) and §4.3 (semantic).

This package is separate from ``tradingagents.agents.utils.memory`` so the
existing append-only markdown log keeps working for current users while the
new audit-grade memory is opt-in via config (``memory_v2: true``).

Components:

- ``episodic``  — (State, Action, Outcome, Timestamp) episodes with
                  outcome-embargo, time-decay retrieval, and regime tags.
- ``semantic``  — curated knowledge base with (doc_id, version_ts,
                  source_url) provenance.
- ``store``     — JSON-backed atomic persistence (same crash-safety pattern
                  as the portfolio store).

The audit-grade memory exists to close paper §4.2's "outcome embargo" gap:
without it, an episode whose outcome occurred *after* the current decision
date can silently leak into the retrieval pool, inflating backtest results.
"""

from .episodic import (
    Episode,
    EpisodicMemory,
    OutcomeStatus,
    Regime,
    RetrievedEpisode,
)
from .semantic import (
    KnowledgeDoc,
    RetrievedDoc,
    SemanticKnowledgeBase,
)
from .regime import detect_regime

__all__ = [
    "Episode",
    "EpisodicMemory",
    "OutcomeStatus",
    "Regime",
    "RetrievedEpisode",
    "KnowledgeDoc",
    "RetrievedDoc",
    "SemanticKnowledgeBase",
    "detect_regime",
]
