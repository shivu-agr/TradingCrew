"""Gym-style trading environment wrapping the M2 simulator.

This is the **environment half** of L4 RL.  The other half (the policy
+ PPO update) lives in :mod:`trading_crew.agentic.rl.ppo`.

Why our own env instead of ``gymnasium``?
-----------------------------------------
The M2 simulator already enforces:

* next-bar fill semantics,
* participation cap (5% of ADV),
* fees + half-spread + Almgren-Chriss impact,
* cash / margin checks,
* mark-to-market + drawdown bookkeeping.

We don't want a different execution model during training vs live —
that would *create* the train/test gap the paper warns about (§6.1
"implicit execution") rather than close it.  So this env is a *thin*
adapter: at each step the policy chooses a discrete target weight, we
build an ``ActionProposal`` + compile to an ``Order``, hand it to the
``ExecutionSimulator``, mark the book to the next close, and the reward
is the realised Δ NAV.

Action space
------------
7 discrete buckets mapped to target portfolio weights.  Discrete (not
continuous) so:

* PPO update is simpler (categorical entropy, no reparam),
* a categorical distribution is what the LLM eventually emits ("LONG /
  NEUTRAL / SHORT" + a confidence bucket), so the policy and the LLM
  share the same action grammar.

Observation space
-----------------
:data:`trading_crew.agentic.rl.state.FEATURE_DIM` features — see that
module for the full schema.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from trading_crew.agentic.execution.contracts import (
    ActionSide,
    OrderTimeInForce,
)
from trading_crew.agentic.execution.cost import get_cost_model
from trading_crew.agentic.execution.simulator import (
    Bar,
    ExecutionSimulator,
    FillStatus,
    Order,
)
from trading_crew.agentic.portfolio.state import PortfolioState
from trading_crew.agentic.rl.state import FEATURE_DIM, FeatureExtractor


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

# The discrete action set.  Index → target weight ∈ [-1, 1].  Tuned so
# that:
#  * Index 0–2 are short of increasing size,
#  * Index 3 is FLAT (the "do nothing" action — vital, otherwise the
#    policy is forced to always hold a position),
#  * Index 4–6 are long of increasing size.
# We deliberately cap at ±20% per ticker; concentration risk is the
# single biggest failure mode in retail RL trading bots and ±20% is
# what M5's default ``max_position_weight`` will clamp to anyway.
ACTION_WEIGHTS: tuple[float, ...] = (
    -0.20, -0.10, -0.05,
    0.00,
    0.05, 0.10, 0.20,
)
N_ACTIONS: int = len(ACTION_WEIGHTS)
FLAT_ACTION: int = ACTION_WEIGHTS.index(0.0)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TradingEnvConfig:
    """Hyperparameters for the env that *aren't* policy hyperparameters.

    These live separate from PPO config because they describe the
    *world* the agent is in — different ones would mean a different
    learning problem entirely (e.g. raise ``transaction_cost_mult`` and
    you're training for a higher-friction venue).
    """

    starting_cash: float = 100_000.0
    """Initial cash in the simulated book."""

    horizon_days: int = 1
    """How many bars each ActionProposal is held for before reconsidering.
    1 means the policy can change its mind every bar; larger values give
    the env a slower clock and reduce transaction costs."""

    cost_model_name: str = "standard"
    """One of ``standard / low / high / futures_standard / futures_low /
    futures_high`` — picks the M2 cost preset used during training."""

    transaction_cost_mult: float = 1.0
    """Multiplier applied to the M2 cost model's fees + spread before
    they're charged.  Set >1.0 to make the policy more cost-aware during
    training (paper §6.1 "explicit cost awareness")."""

    drawdown_kill_pct: float = 0.40
    """If realised drawdown exceeds this, the episode terminates early
    with a large negative reward.  Saves training time and bakes in a
    hard risk gate the policy can never override."""

    reward_scaling: float = 100.0
    """Returns are tiny per bar (~0.001), and PPO needs reward values in
    a manageable range or the value loss dominates the policy loss.  We
    multiply by 100 so a 1% bar return shows up as reward=1.0."""

    turnover_penalty_bps: float = 0.0
    """Optional extra penalty per unit of |Δweight| (in basis points).
    The M2 simulator already charges fees + slippage; this is a separate
    shaping term to push toward lower turnover when desired."""

    drawdown_penalty_coef: float = 0.5
    """Coefficient on the additional reward penalty applied when
    drawdown deepens within the episode.  ``0.0`` disables shaping;
    higher values produce a more risk-averse policy."""

    volatility_curriculum: bool = False
    """Phase 2D — when True, ``reset()`` samples the starting bar from a
    *low-vol-first* schedule that ramps up to high-vol regimes as
    training progresses.  Faster + more stable early learning; the
    policy doesn't have to wrestle with high-vol noise before its
    value head has any grip on the dynamics.  No effect when
    ``random_starts=False``."""


# ---------------------------------------------------------------------------
# StepResult — Gym-style tuple, named for readability
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    done: bool
    info: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# TradingEnv
# ---------------------------------------------------------------------------


class TradingEnv:
    """Gym-style env wrapping a single ticker's OHLCV history.

    Lifecycle::

        env = TradingEnv(symbol="NTNX", ohlcv=df, config=TradingEnvConfig())
        obs = env.reset()
        while True:
            action = policy(obs)
            step = env.step(action)
            if step.done: break

    Single-ticker on purpose — a multi-ticker policy is an M7 problem
    (portfolio allocator) on top of N single-ticker policies.  Keeping
    this env single-asset makes the action space tiny and the training
    signal strong.
    """

    # -- construction ------------------------------------------------------

    def __init__(
        self,
        symbol: str,
        ohlcv: pd.DataFrame,
        config: Optional[TradingEnvConfig] = None,
        feature_extractor: Optional[FeatureExtractor] = None,
        random_starts: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """Construct an env around ``ohlcv`` (oldest first).

        Args:
            symbol: ticker symbol (used in PortfolioState bookkeeping).
            ohlcv: DataFrame indexed by Date with columns
                ``open, high, low, close, volume``.  Must contain at
                least ``warmup + horizon + 5`` rows.
            config: optional env hyperparameters.
            feature_extractor: optional extractor (default ``lookback=60``).
            random_starts: if True, ``reset()`` samples a random starting
                index.  Drastically improves sample efficiency by
                exposing the policy to many regimes per epoch.
            rng: optional ``numpy.random.Generator`` for deterministic
                sampling in tests.
        """
        self.symbol = symbol.upper().strip()
        self.config = config or TradingEnvConfig()
        self.fe = feature_extractor or FeatureExtractor()
        self.random_starts = random_starts
        self._rng = rng or np.random.default_rng()

        self._ohlcv = _normalise_ohlcv(ohlcv)
        warmup = self.fe.warmup_bars()
        # We need warmup bars to compute the *first* feature vector, +1
        # for the next-bar fill, + horizon_days for at least one
        # decision.  Anything less and the env can never finish a
        # single episode.
        min_rows = warmup + self.config.horizon_days + 2
        if len(self._ohlcv) < min_rows:
            raise ValueError(
                f"OHLCV too short: need at least {min_rows} bars, "
                f"got {len(self._ohlcv)}"
            )

        # Build the cost model once.  We multiply fees + half_spread by
        # transaction_cost_mult so the env can punish the policy with
        # *worse* costs than live — a safety margin baked into training.
        base = get_cost_model(self.config.cost_model_name)
        if self.config.transaction_cost_mult != 1.0:
            from trading_crew.agentic.execution.cost import CostModel
            m = self.config.transaction_cost_mult
            base = CostModel(
                fee_bps=base.fee_bps * m,
                flat_fee=base.flat_fee * m,
                half_spread_bps=base.half_spread_bps * m,
                impact_k=base.impact_k,
                name=f"{base.name}_x{m:g}",
            )
        self._cost_model = base

        self._simulator = ExecutionSimulator(
            cost_model=self._cost_model,
            participation_cap=0.05,
            latency_ms=50,
        )

        # Set by reset().
        self._t: int = 0  # current bar index (the bar the policy is *deciding* on).
        self._state: Optional[PortfolioState] = None
        self._open_ts_idx: Optional[int] = None
        self._open_avg_cost: float = 0.0
        self._episode_steps: int = 0
        self._max_steps: int = 0
        self._prev_drawdown: float = 0.0
        self._last_nav: float = 0.0

        # Phase 2D — volatility-curriculum bookkeeping.  ``_curriculum_progress``
        # is incremented externally (or via ``set_curriculum_progress``) and
        # goes from 0.0 (lowest-vol bars only) to 1.0 (all bars).
        self._curriculum_progress: float = 0.0
        self._vol_sorted_starts: Optional[np.ndarray] = None
        if self.config.volatility_curriculum and self.random_starts:
            self._vol_sorted_starts = self._build_vol_sorted_starts()

    # -- observation / action shapes (Gym-compatible API) ------------------

    @property
    def observation_dim(self) -> int:
        return FEATURE_DIM

    @property
    def n_actions(self) -> int:
        return N_ACTIONS

    # -- core lifecycle ---------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset to a fresh portfolio and a (possibly random) start bar.

        Returns the initial observation.  Always returns a finite vector
        with shape ``(observation_dim,)``.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        warmup = self.fe.warmup_bars()
        last_valid = len(self._ohlcv) - self.config.horizon_days - 2
        if self.random_starts:
            if self._vol_sorted_starts is not None and len(self._vol_sorted_starts) > 0:
                # Curriculum: start from the lowest-vol bars and expand
                # the pool linearly with ``_curriculum_progress``.  At
                # progress=0 the sample pool is the bottom 10% of vol
                # buckets; at progress=1 it's the full sortable range.
                pool_size = max(
                    1,
                    int(len(self._vol_sorted_starts) * (0.1 + 0.9 * min(1.0, max(0.0, self._curriculum_progress)))),
                )
                pool = self._vol_sorted_starts[:pool_size]
                start = int(pool[self._rng.integers(0, pool_size)])
            else:
                start = int(self._rng.integers(warmup, max(warmup + 1, last_valid)))
        else:
            start = warmup
        self._t = start
        self._max_steps = max(1, last_valid - start)
        self._episode_steps = 0
        self._open_ts_idx = None
        self._open_avg_cost = 0.0
        self._prev_drawdown = 0.0

        self._state = PortfolioState(
            portfolio_id=f"rl-{self.symbol}",
            base_currency="USD",
            starting_cash=self.config.starting_cash,
            cash=self.config.starting_cash,
        )
        self._last_nav = self._state.nav

        return self._observe()

    def step(self, action: int) -> StepResult:
        """Apply ``action`` (index into ``ACTION_WEIGHTS``) for one bar.

        Algorithm:

        1. Map ``action`` to a target weight.
        2. Build an ``ActionProposal`` with that weight.
        3. Compile to an Order using current state + the *current* bar's
           close as a reference price.
        4. Hand the order + the *next* bar to the M2 simulator.
        5. Mark the book to the next bar's close.
        6. Reward = Δ NAV / starting_cash, scaled.  Hard stop on
           drawdown_kill_pct.
        """
        if self._state is None:
            raise RuntimeError("step() called before reset().")
        if not 0 <= action < N_ACTIONS:
            raise ValueError(f"action {action} out of range [0, {N_ACTIONS})")

        target_w = ACTION_WEIGHTS[action]
        current_bar = self._ohlcv.iloc[self._t]
        next_bar_row = self._ohlcv.iloc[self._t + 1]
        reference_price = float(current_bar["close"])

        # Build the M2 order directly.  We deliberately bypass the
        # ``ActionProposal -> proposal_to_order`` path here because the
        # proposal schema enforces ``side == sign(target_weight)`` (it's
        # an LLM-intent contract), whereas the env needs to express
        # *delta-aware* trades — e.g. trimming a +10% long to +5% is a
        # SELL even though target_weight stays positive.  Constructing
        # the Order ourselves uses the same downstream simulator code
        # path (cost model, partial fills, mark-to-market) without
        # tripping the LLM-side invariant check.
        order = self._build_order(target_w, reference_price)

        # The order is None when the target weight equals the current
        # weight (no delta) or rounds below one share.
        fill = None
        if order is not None:
            volume = float(next_bar_row.get("volume", 0.0) or 0.0)
            adv = max(volume, 1.0)  # avoid div/0 in simulator participation
            next_bar = Bar(
                ts=str(self._ohlcv.index[self._t + 1]),
                open=float(next_bar_row["open"]),
                high=float(next_bar_row["high"]),
                low=float(next_bar_row["low"]),
                close=float(next_bar_row["close"]),
                volume=volume,
                adv=adv,
            )
            fill = self._simulator.execute(order, next_bar, self._state)
            if fill.status in (FillStatus.FILLED, FillStatus.PARTIAL_FILL):
                if self._open_ts_idx is None:
                    self._open_ts_idx = self._t + 1
                    self._open_avg_cost = fill.avg_price or reference_price
        else:
            # Even without a fill, mark the book to the next close so
            # mark-to-market PnL is recorded for any existing position.
            self._state.mark_to_market(
                {self.symbol: float(next_bar_row["close"])},
                ts=str(self._ohlcv.index[self._t + 1]),
            )

        # If the position has closed (no longer in state.positions), clear
        # tracking — otherwise the next "open" still thinks it's a re-add.
        if self.symbol not in self._state.positions:
            self._open_ts_idx = None
            self._open_avg_cost = 0.0

        # --- reward -------------------------------------------------------
        new_nav = self._state.nav
        bar_return = (new_nav - self._last_nav) / max(self.config.starting_cash, 1e-8)
        reward = bar_return * self.config.reward_scaling

        # Optional shaping: turnover penalty.  Encourages the policy to
        # not flip every bar even if costs are zeroed in low-cost regimes.
        if order is not None and self.config.turnover_penalty_bps > 0.0:
            notional_pct = abs(order.qty_signed * reference_price) / max(new_nav, 1e-8)
            reward -= notional_pct * self.config.turnover_penalty_bps * 1e-4 * self.config.reward_scaling

        # Drawdown-shaping reward: punish *increases* in drawdown so the
        # policy learns risk control without us hard-coding stops.
        if self.config.drawdown_penalty_coef > 0.0:
            dd_delta = max(self._state.max_drawdown - self._prev_drawdown, 0.0)
            if dd_delta > 0.0:
                reward -= dd_delta * self.config.drawdown_penalty_coef * self.config.reward_scaling
        self._prev_drawdown = self._state.max_drawdown

        self._last_nav = new_nav
        self._t += 1
        self._episode_steps += 1

        # --- termination --------------------------------------------------
        done = False
        kill_reason = ""
        if self._state.max_drawdown >= self.config.drawdown_kill_pct:
            done = True
            kill_reason = f"drawdown {self._state.max_drawdown:.1%} >= kill_pct"
            # One-off large penalty so the policy *really* hates blowups.
            reward -= 5.0
        elif self._t >= len(self._ohlcv) - 1:
            done = True
            kill_reason = "out of bars"
        elif self._episode_steps >= self._max_steps:
            done = True
            kill_reason = "max_steps reached"

        info = {
            "action_weight": target_w,
            "nav": new_nav,
            "drawdown": self._state.max_drawdown,
            "fill_status": fill.status.value if fill else "NO_ORDER",
            "fees": (fill.cost_breakdown.get("total", 0.0) if fill else 0.0),
            "kill_reason": kill_reason,
            "bar_return": bar_return,
            "step": self._episode_steps,
        }
        return StepResult(self._observe(), reward, done, info)

    # -- volatility curriculum --------------------------------------------

    def set_curriculum_progress(self, progress: float) -> None:
        """Move the curriculum pointer along ``[0.0, 1.0]``.

        Trainers call this once per rollout (typically
        ``progress = steps_done / total_steps``).  At ``progress=0`` the
        env samples starts only from the bottom 10% of bars by realised
        vol; at ``progress=1`` the full distribution is sampled.
        """
        self._curriculum_progress = float(progress)

    def _build_vol_sorted_starts(self) -> np.ndarray:
        """Return candidate start indices sorted ascending by realised vol.

        Uses a 20-bar rolling stdev of close-to-close log returns.  Only
        bars in the valid ``[warmup, last_valid)`` window are returned.
        """
        warmup = self.fe.warmup_bars()
        last_valid = len(self._ohlcv) - self.config.horizon_days - 2
        if last_valid <= warmup + 1:
            return np.array([], dtype=np.int64)
        close = self._ohlcv["close"].astype(float).to_numpy()
        # log returns, then 20-bar rolling stdev.
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ret = np.diff(np.log(np.where(close > 0, close, np.nan)))
        log_ret = np.concatenate([[0.0], log_ret])
        window = 20
        # Pad-front so rolling[i] aligns with bar i.
        roll_vol = np.zeros_like(log_ret)
        for i in range(len(log_ret)):
            lo = max(0, i - window + 1)
            seg = log_ret[lo:i + 1]
            roll_vol[i] = float(np.nanstd(seg)) if seg.size > 1 else 0.0
        candidate = np.arange(warmup, last_valid)
        if candidate.size == 0:
            return np.array([], dtype=np.int64)
        vols = roll_vol[candidate]
        order = np.argsort(vols, kind="stable")
        return candidate[order]

    # -- order construction (env-internal, delta-aware) -------------------

    def _build_order(self, target_weight: float, reference_price: float):
        """Compile a delta-aware ``Order`` for the M2 simulator.

        Computes ``delta_dollars = (target_weight - current_weight) * NAV``,
        converts to shares at ``reference_price``, and returns the
        Order with the matching side.  Skips zero / sub-1-share deltas.
        """
        if self._state is None:
            return None
        current_dollars = (
            self._state.positions[self.symbol].market_value
            if self.symbol in self._state.positions
            else 0.0
        )
        target_dollars = target_weight * self._state.nav
        delta_dollars = target_dollars - current_dollars
        qty_signed = delta_dollars / max(reference_price, 1e-8)
        if abs(qty_signed) < 1.0:
            return None
        qty_signed = math.copysign(round(abs(qty_signed)), qty_signed)
        side = ActionSide.BUY if qty_signed > 0 else ActionSide.SELL
        return Order(
            symbol=self.symbol,
            side=side,
            qty_signed=qty_signed,
            limit_price=None,
            tif=OrderTimeInForce.DAY,
            decision_ts=str(self._ohlcv.index[self._t]),
        )

    # -- observation builder ----------------------------------------------

    def _observe(self) -> np.ndarray:
        """Build the observation vector for the *current* bar (``self._t``)."""
        warmup = self.fe.warmup_bars()
        history_start = max(0, self._t + 1 - warmup)
        history = self._ohlcv.iloc[history_start:self._t + 1]

        if self.symbol in self._state.positions:
            pos = self._state.positions[self.symbol]
            pos_weight = self._state.weight(self.symbol)
            unrealised = ((pos.last_price - pos.avg_cost) / pos.avg_cost) if pos.avg_cost else 0.0
            bars_in_trade = self._t - (self._open_ts_idx or self._t)
            # Shorts: a falling price is profit, so flip the sign.
            if pos.qty < 0:
                unrealised = -unrealised
        else:
            pos_weight = 0.0
            unrealised = 0.0
            bars_in_trade = 0

        return self.fe.extract(
            history,
            position_weight=pos_weight,
            unrealised_pct=unrealised,
            bars_in_trade=bars_in_trade,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the input is uniformly indexed + lowercase-columned.

    Accepts either yfinance-style (Date / Open / …) or already-clean
    (date / open / …) inputs so the env doesn't care which source
    fetched the data.
    """
    if df is None or df.empty:
        raise ValueError("ohlcv is empty")
    out = df.copy()
    if "Date" in out.columns:
        out = out.set_index("Date")
    elif "date" in out.columns:
        out = out.set_index("date")
    out.columns = [c.lower() for c in out.columns]
    required = {"open", "high", "low", "close"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"ohlcv missing columns: {sorted(missing)}")
    if "volume" not in out.columns:
        out["volume"] = 0.0
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out = out.dropna(subset=["open", "high", "low", "close"])
    return out
