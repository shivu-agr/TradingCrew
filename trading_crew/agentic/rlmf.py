"""RLMF — Reinforcement Learning from Market Feedback.

Inspired by the FinGPT project (AI4Finance-Foundation/FinGPT) which
adapted the RLHF framework used to align ChatGPT so that the *human
feedback* is replaced with *market feedback* — the realised return of
a decision over its stated horizon.  The insight is:

    RLHF: reward = human preference over pairs of responses
    RLMF: reward = signed realised return over the decision's horizon,
                    scaled by conviction and clipped for tail risk

We don't fine-tune the base LLM in-process here (that's a
separately-runnable job on a GPU; see the README for the
FinGPT-Forecaster LoRA recipe adapted to this codebase).  What this
module *does* run every workflow is the cheaper, prompt-only surrogate
that FinGPT calls "instruction distillation":

1.  **Dataset build.**  Walk the resolved episodes on disk (M3 already
    writes them) and emit a JSONL that carries ``(state_summary,
    action, realised_return, alpha_return, horizon, reward)`` per row.
    The dataset shape mirrors ``fingpt-forecaster-dow30-202305-202405``
    on Hugging Face so the same fine-tuning notebook can consume it
    without adaptation.
2.  **Reward ranking.**  Score each past decision on a composite
    "market-alignment" reward:

        reward = alpha_return * conviction_weight * horizon_weight

    where ``conviction_weight`` up-weights high-conviction wins /
    losses (a HIGH-conviction miss is a more informative training
    signal than a LOW-conviction one), and ``horizon_weight`` gives
    LONG-tier decisions more weight than SHORT (matches the desk's
    mid-to-long-term bias — a good long-hold thesis deserves more
    credit than a lucky earnings scalp).

3.  **Few-shot exemplar pool.**  Pick the top-K rewarded past
    decisions and format them into a compact block the PM prompt can
    prefix (``PAST WINS TO EMULATE`` + ``PAST LOSSES TO AVOID``).
    This is the prompt-only RLHF surrogate that actually runs in
    production — it needs no GPU, no LoRA training, and gives the LLM
    the same shape of feedback FinGPT-Forecaster consumes at fine-tune
    time.

4.  **Optional: LoRA fine-tune.**  ``export_fingpt_forecaster_jsonl``
    writes a file the standard FinGPT-Forecaster training notebook can
    ingest directly.  When a user runs that notebook offline they get
    a LoRA adapter that can be merged into the base LLM served by vLLM
    — closing the loop the FinGPT paper describes.

Everything below is dependency-light (stdlib + pandas) so the
prompt-only path runs in every workflow without adding weight to
the hot loop.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

from trading_crew.agentic.memory import (
    Episode,
    EpisodicMemory,
    OutcomeStatus,
)
from trading_crew.horizons import Horizon, horizon_days_to_horizon

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------


# Horizon-aware credit assignment.  A LONG-tier decision is more
# strategically informative than a SHORT one — the LLM's mid-to-long-
# term bias is exactly what we want to reinforce — so we scale up the
# reward on LONG-tier wins and down on SHORT-tier scalps.  Mirrors
# FinGPT-Forecaster's per-week aggregation weight but tier-based
# rather than time-based (our decisions are one-shot, not rolling).
_HORIZON_REWARD_WEIGHT: dict[Horizon, float] = {
    Horizon.SHORT: 0.75,
    Horizon.MEDIUM: 1.00,
    Horizon.LONG: 1.25,
}


def _conviction_weight(conviction_score: float) -> float:
    """Higher-conviction outcomes carry more RLMF signal.

    A HIGH conviction call that pays off is a strong reinforcement
    signal; a HIGH conviction call that flops is a strong *negative*
    reinforcement signal (i.e. the LLM should *not* replicate that
    prompt pattern).  Both benefit from being weighted heavier than
    a LOW conviction outcome which was basically an abstention with
    dressing.

    Mapping is intentionally smooth (not tiered) so we don't create
    hard cliffs at conviction = 0.35 / 0.65 — the fine-tune notebook
    can consume the continuous signal directly.
    """
    c = max(0.0, min(1.0, float(conviction_score)))
    # 0.5 -> 1.0 (baseline); 0.8 -> 1.4 (up-weight); 0.2 -> 0.6 (down-weight)
    return 0.4 + 1.2 * c


def _clip_return(r: float, cap: float = 2.0) -> float:
    """Winsorize the reward so a single 500% moonshot doesn't dominate.

    Set to 200% by default — matches the FinGPT-Forecaster paper's
    convention of clipping at the 1st / 99th percentile of realised
    return.  Below the cap the signal is linear so ranking is
    preserved.
    """
    return max(-cap, min(cap, float(r)))


def compute_reward(ep: Episode) -> Optional[float]:
    """Compute the RLMF reward for a single resolved episode.

    Returns ``None`` for episodes that are not yet resolved (or
    resolved without a realised return — e.g. delisted / abandoned).
    Callers should filter those out before ranking.
    """
    if ep.outcome_status != OutcomeStatus.RESOLVED:
        return None
    if ep.realised_return is None:
        return None

    # Prefer alpha over benchmark when available — a +3% return during
    # a +8% market rally is not a reward signal we want to reinforce;
    # the LLM should learn to beat SPY, not track it.
    signal = (
        ep.alpha_return
        if ep.alpha_return is not None
        else ep.realised_return
    )
    signal = _clip_return(signal)

    proposal = ep.action_proposal or {}
    conviction = float(proposal.get("conviction_score") or 0.5)
    conv_w = _conviction_weight(conviction)

    horizon_days = int(proposal.get("horizon_days") or 0)
    tier = horizon_days_to_horizon(horizon_days)
    horizon_w = _HORIZON_REWARD_WEIGHT.get(tier, 1.0)

    return signal * conv_w * horizon_w


# ---------------------------------------------------------------------------
# Dataset row schema
# ---------------------------------------------------------------------------


@dataclass
class RLMFRow:
    """One (state, action, reward) training row.

    Mirrors FinGPT-Forecaster's JSONL shape so a user can point the
    Hugging Face fine-tuning notebook at ``rlmf/dataset.jsonl``
    without any adapter code.  We keep the extra fields
    (``alpha_return``, ``horizon``, ``reward``) even though FinGPT's
    tokeniser ignores them — they are useful for the frontend RL panel
    and for downstream analytics.
    """

    episode_id: str
    symbol: str
    decision_ts: str
    horizon: str  # SHORT / MEDIUM / LONG
    horizon_days: int
    action: str   # BUY / SELL / HOLD / ABSTAIN
    conviction_score: float
    state_summary: str
    realised_return: float
    alpha_return: Optional[float]
    reward: float
    reflection: Optional[str] = None

    def to_jsonl_dict(self) -> dict:
        """Return a dict shaped like FinGPT-Forecaster's JSONL rows.

        FinGPT uses ``prompt`` / ``response`` / ``chosen`` /
        ``rejected`` keys for its DPO-style dataset.  We emit both the
        FinGPT-compatible keys (``prompt``, ``response``) and our own
        analytics fields so the same file drives fine-tuning + the UI.
        """
        return {
            # FinGPT-Forecaster compatibility keys
            "prompt": self.state_summary,
            "response": (
                f"Action: {self.action}\n"
                f"Horizon: {self.horizon} ({self.horizon_days}d)\n"
                f"Conviction: {self.conviction_score:.2f}\n"
                f"Realised return: {self.realised_return:+.2%}"
                + (
                    f"\nAlpha vs benchmark: {self.alpha_return:+.2%}"
                    if self.alpha_return is not None
                    else ""
                )
                + (f"\nReflection: {self.reflection}" if self.reflection else "")
            ),
            # RLMF-specific analytics fields
            "episode_id": self.episode_id,
            "symbol": self.symbol,
            "decision_ts": self.decision_ts,
            "horizon": self.horizon,
            "horizon_days": self.horizon_days,
            "action": self.action,
            "conviction_score": self.conviction_score,
            "realised_return": self.realised_return,
            "alpha_return": self.alpha_return,
            "reward": self.reward,
            "reflection": self.reflection,
        }


def _episode_to_row(ep: Episode) -> Optional[RLMFRow]:
    """Project one ``Episode`` into an ``RLMFRow`` — or ``None`` if the
    episode isn't yet a valid training row.
    """
    reward = compute_reward(ep)
    if reward is None:
        return None
    proposal = ep.action_proposal or {}
    horizon_days = int(proposal.get("horizon_days") or 0)
    return RLMFRow(
        episode_id=ep.episode_id,
        symbol=ep.symbol,
        decision_ts=ep.decision_ts,
        horizon=horizon_days_to_horizon(horizon_days).value,
        horizon_days=horizon_days,
        action=str(proposal.get("side") or "ABSTAIN").upper(),
        conviction_score=float(proposal.get("conviction_score") or 0.0),
        state_summary=ep.state_summary,
        realised_return=float(ep.realised_return or 0.0),
        alpha_return=(
            float(ep.alpha_return) if ep.alpha_return is not None else None
        ),
        reward=reward,
        reflection=ep.reflection,
    )


# ---------------------------------------------------------------------------
# Dataset build + export
# ---------------------------------------------------------------------------


def build_dataset(
    memory: Optional[EpisodicMemory] = None,
    *,
    symbol: Optional[str] = None,
    horizon_tier: Optional[Horizon] = None,
    now: Optional[datetime] = None,
) -> List[RLMFRow]:
    """Return every resolved episode as an RLMF training row.

    Filters:
      * ``symbol`` — restrict to a single ticker (case-insensitive).
      * ``horizon_tier`` — restrict to SHORT / MEDIUM / LONG (helpful
        when you want to fine-tune a long-only or short-only adapter).
      * ``now`` — respected only for the ``embargo`` check downstream;
        this function itself only touches RESOLVED episodes.

    The dataset is sorted by descending reward so a caller building
    a few-shot exemplar pool can just ``take(k)`` from the top.
    """
    from trading_crew.agentic.reflection import _default_memory_path  # local import — avoids startup cost when RLMF unused

    memory = memory or EpisodicMemory(_default_memory_path())
    rows: List[RLMFRow] = []
    for ep in memory.all_episodes():
        if symbol and ep.symbol.upper() != symbol.upper():
            continue
        row = _episode_to_row(ep)
        if row is None:
            continue
        if horizon_tier is not None and row.horizon != horizon_tier.value:
            continue
        rows.append(row)

    rows.sort(key=lambda r: r.reward, reverse=True)
    return rows


def default_dataset_path() -> Path:
    """Where the JSONL is written on disk.

    Reads ``TRADINGCREW_CACHE_DIR`` (same env var episodic-memory uses)
    so the RLMF dataset always lands alongside the episode store — one
    directory per operator, one file per learning artefact.
    """
    cache_dir = os.environ.get("TRADINGCREW_CACHE_DIR") or os.path.expanduser("~/.trading_crew")
    return Path(cache_dir) / "rlmf" / "dataset.jsonl"


def export_dataset(
    rows: Iterable[RLMFRow],
    path: Optional[Path] = None,
) -> Path:
    """Write ``rows`` to a JSONL file the FinGPT training notebook can
    ingest directly.  Returns the path written.

    We overwrite (not append) so the file is always a snapshot of the
    latest resolved set — appending would drift out of sync with the
    episodic store after an abandonment / re-resolution.
    """
    path = path or default_dataset_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_jsonl_dict(), ensure_ascii=False))
            f.write("\n")
    logger.info("RLMF: wrote %d rows to %s", len(list(rows) if isinstance(rows, list) else []), path)
    return path


def export_fingpt_forecaster_jsonl(
    memory: Optional[EpisodicMemory] = None,
    path: Optional[Path] = None,
) -> Path:
    """Convenience: build + export in one call, defaulting to the
    canonical dataset path.  Suitable for a cron / CLI step:

        python -m trading_crew.agentic.rlmf --export
    """
    rows = build_dataset(memory)
    return export_dataset(rows, path)


# ---------------------------------------------------------------------------
# Few-shot exemplar pool
# ---------------------------------------------------------------------------


def _format_exemplar(row: RLMFRow, *, sign: str) -> str:
    """One-line-per-field markdown block a PM prompt can inline.

    ``sign`` is one of ``"WIN"`` / ``"LOSS"`` — controls the "TO EMULATE"
    vs "TO AVOID" framing.  We deliberately keep the block short (< 20
    lines) so a pool of K=5 exemplars still fits in a single-turn
    prompt budget.
    """
    return (
        f"### {sign}: {row.symbol} ({row.decision_ts[:10]}) — "
        f"action={row.action}, horizon={row.horizon} ({row.horizon_days}d), "
        f"realised={row.realised_return:+.2%}"
        + (
            f", alpha={row.alpha_return:+.2%}"
            if row.alpha_return is not None
            else ""
        )
        + f", reward={row.reward:+.3f}\n"
        f"State summary (what the desk saw):\n"
        f"> {row.state_summary[:400]}"
        + ("\n> …\n" if len(row.state_summary) > 400 else "\n")
        + (
            f"Post-hoc reflection: {row.reflection}\n"
            if row.reflection
            else ""
        )
    )


def build_few_shot_pool(
    memory: Optional[EpisodicMemory] = None,
    *,
    symbol: Optional[str] = None,
    horizon_tier: Optional[Horizon] = None,
    k_wins: int = 3,
    k_losses: int = 2,
) -> str:
    """Build a compact few-shot prompt block from resolved episodes.

    The block has two sections:

    * ``## PAST WINS TO EMULATE`` — the top ``k_wins`` decisions by
      reward.  The PM should re-use the *shape* of these prompts
      (evidence, sizing, horizon choice) when it sees a similar
      state.
    * ``## PAST LOSSES TO AVOID`` — the bottom ``k_losses`` decisions
      by reward, i.e. high-conviction misses.  These teach the LLM
      what NOT to do (e.g. "high-conviction BUY on a broken compounder").

    Returns an empty string when there are no resolved episodes yet
    — the caller can then skip the RLMF prefix and prompt as usual.

    This is the *prompt-only* RLHF surrogate.  For the full LoRA
    fine-tune, run ``export_fingpt_forecaster_jsonl`` and use the
    FinGPT-Forecaster training notebook.
    """
    rows = build_dataset(
        memory=memory,
        symbol=symbol,
        horizon_tier=horizon_tier,
    )
    if not rows:
        return ""

    wins = rows[:k_wins]
    losses = list(reversed(rows))[:k_losses]

    parts = [
        "## RLMF PROMPT PREFIX — past decisions ranked by market-alignment reward",
        "",
        "The desk has a growing case-book of resolved decisions.  The "
        "highest-reward calls (below, ``PAST WINS``) are prompt patterns "
        "worth EMULATING when you see similar evidence.  The lowest-reward "
        "calls (``PAST LOSSES``) are patterns to AVOID — they were high-"
        "conviction bets that turned out to be badly calibrated.",
        "",
    ]
    if wins:
        parts.append("## PAST WINS TO EMULATE")
        parts.append("")
        parts.extend(_format_exemplar(r, sign="WIN") for r in wins)
    if losses:
        parts.append("")
        parts.append("## PAST LOSSES TO AVOID")
        parts.append("")
        parts.extend(_format_exemplar(r, sign="LOSS") for r in losses)

    parts.append("")
    parts.append(
        "END OF RLMF PROMPT PREFIX.  Use it as a similarity anchor: "
        "if the *current* evidence resembles a WIN pattern, incline "
        "toward the same action/horizon shape; if it resembles a LOSS "
        "pattern, prefer the opposite side or a tighter-sized HOLD."
    )
    parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def summarise_dataset(rows: List[RLMFRow]) -> dict:
    """Aggregate stats for the RL Training panel in the web UI.

    Returns a dict shaped for a Vue / Alpine table:

        {
            "n_resolved": int,
            "by_tier": {"SHORT": {"n": int, "mean_reward": float}, ...},
            "by_action": {"BUY": {"n": int, "mean_reward": float}, ...},
            "top_wins": [row dicts, up to 3],
            "top_losses": [row dicts, up to 3],
        }
    """
    if not rows:
        return {
            "n_resolved": 0,
            "by_tier": {},
            "by_action": {},
            "top_wins": [],
            "top_losses": [],
            "as_of": datetime.now(timezone.utc).isoformat(),
        }
    by_tier: dict[str, list[float]] = {}
    by_action: dict[str, list[float]] = {}
    for r in rows:
        by_tier.setdefault(r.horizon, []).append(r.reward)
        by_action.setdefault(r.action, []).append(r.reward)
    return {
        "n_resolved": len(rows),
        "by_tier": {
            k: {"n": len(v), "mean_reward": (sum(v) / len(v)) if v else 0.0}
            for k, v in by_tier.items()
        },
        "by_action": {
            k: {"n": len(v), "mean_reward": (sum(v) / len(v)) if v else 0.0}
            for k, v in by_action.items()
        },
        "top_wins": [row.to_jsonl_dict() for row in rows[:3]],
        "top_losses": [row.to_jsonl_dict() for row in list(reversed(rows))[:3]],
        "as_of": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _main(argv: Optional[List[str]] = None) -> int:
    """RLMF utilities entry point.

    Subcommands (all optional; if none are passed we run ``--export``
    against the default path):

      --export         write dataset.jsonl in the FinGPT-Forecaster JSONL
                       shape.  Default path is
                       ``$TRADINGCREW_CACHE_DIR/rlmf/dataset.jsonl``.
      --summary        print aggregate stats (n_resolved, per-tier and
                       per-action mean reward, top 3 wins / losses) as
                       JSON to stdout.
      --few-shot       dry-run the prompt-only few-shot pool the PM
                       task would see if TRADINGCREW_RLMF_ENABLED=1.
                       Useful to inspect what your desk has "learned"
                       before flipping the switch on for real runs.
      --symbol SYM     restrict every op above to a single ticker.
      --horizon TIER   restrict to SHORT / MEDIUM / LONG.
      --path FILE      override the export path (only affects --export).
      --k-wins N       # of top-reward wins for --few-shot (default 3).
      --k-losses N     # of bottom-reward losses for --few-shot (default 2).
    """
    import argparse
    parser = argparse.ArgumentParser(
        description="RLMF utilities — build a FinGPT-Forecaster-compatible "
        "dataset from resolved episodes and inspect the prompt-only "
        "few-shot pool used at PM decision time.",
    )
    parser.add_argument("--export", action="store_true", help="write dataset.jsonl")
    parser.add_argument("--summary", action="store_true", help="print summary JSON")
    parser.add_argument(
        "--few-shot",
        action="store_true",
        help="print the prompt-only RLMF prefix the PM would see",
    )
    parser.add_argument("--symbol", default=None, help="restrict to one symbol")
    parser.add_argument(
        "--horizon",
        default=None,
        choices=("SHORT", "MEDIUM", "LONG"),
        help="restrict to SHORT / MEDIUM / LONG",
    )
    parser.add_argument(
        "--path",
        default=None,
        help="override the export path (only affects --export)",
    )
    parser.add_argument(
        "--k-wins",
        type=int,
        default=3,
        help="# of top-reward wins for --few-shot (default 3)",
    )
    parser.add_argument(
        "--k-losses",
        type=int,
        default=2,
        help="# of bottom-reward losses for --few-shot (default 2)",
    )
    args = parser.parse_args(argv)

    tier = Horizon(args.horizon) if args.horizon else None

    rows = build_dataset(symbol=args.symbol, horizon_tier=tier)

    ran_something = False
    if args.export or not (args.export or args.summary or args.few_shot):
        target = Path(args.path) if args.path else default_dataset_path()
        path = export_dataset(rows, target)
        print(f"wrote {len(rows)} rows to {path}")
        ran_something = True
    if args.summary:
        print(json.dumps(summarise_dataset(rows), indent=2, default=str))
        ran_something = True
    if args.few_shot:
        prefix = build_few_shot_pool(
            symbol=args.symbol,
            horizon_tier=tier,
            k_wins=args.k_wins,
            k_losses=args.k_losses,
        )
        if prefix:
            print(prefix)
        else:
            print(
                "no resolved episodes yet — the RLMF prefix is empty.  "
                "Run the workflow and wait for outcome resolution before "
                "expecting few-shot exemplars."
            )
        ran_something = True
    if not ran_something:  # defensive
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_main(sys.argv[1:]))
