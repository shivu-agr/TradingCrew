"""Deterministic portfolio state — the audit-grade ``Layer A`` from §4.1.

`PortfolioState` is the ground truth of the trading book.  It is updated *only*
by the execution layer (``Fill`` events from ``tradingagents.execution``) and
exposed to the LLM exclusively through read-only tool calls.  The LLM cannot
mutate cash, positions, or P&L through generated text — that gap is what makes
the state auditable across runs (paper §4.1, Layer A vs Layer B distinction).

Persistence
-----------
State is stored as a single JSON file per portfolio, atomically rewritten on
every update.  We avoid the SQLite/parquet route on purpose: a flat JSON file
is trivially diffable, trivially replayable, and survives partial-write
failures because we always write to ``<path>.tmp`` and ``os.replace`` into
place (the same atomic-write pattern that protected the OHLCV cache after the
earlier disk-full event).

Schema versioning is explicit (``schema_version`` field) so future milestones
(M2 adds ``open_orders``, M5 adds risk metrics, M7 adds multi-portfolio) can
migrate forward without silently corrupting a live book.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------


@dataclass
class Position:
    """A single open position in the book.

    ``avg_cost`` is the running cost basis updated on every fill; it is *not*
    the historical entry price (which would only be valid for a single-fill
    position).  ``last_price`` and ``last_mark_ts`` are written by the
    mark-to-market step and consulted when computing ``market_value`` and
    ``unrealized_pnl``.
    """

    symbol: str
    qty: float
    avg_cost: float
    last_price: float
    last_mark_ts: str
    opened_ts: str

    @property
    def market_value(self) -> float:
        """Mark-to-market value at ``last_price``.  Signed: long positive, short negative."""
        return self.qty * self.last_price

    @property
    def cost_basis(self) -> float:
        """Capital deployed in the position at average cost."""
        return self.qty * self.avg_cost

    @property
    def unrealized_pnl(self) -> float:
        """Mark-to-market profit/loss against the running cost basis."""
        return (self.last_price - self.avg_cost) * self.qty

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Position":
        return cls(
            symbol=data["symbol"],
            qty=float(data["qty"]),
            avg_cost=float(data["avg_cost"]),
            last_price=float(data["last_price"]),
            last_mark_ts=data["last_mark_ts"],
            opened_ts=data["opened_ts"],
        )


# ---------------------------------------------------------------------------
# Portfolio state
# ---------------------------------------------------------------------------


@dataclass
class PortfolioState:
    """The full book at a single point in time.

    Invariants:

    - ``cash`` is in account base currency (USD by default).
    - ``positions`` is keyed by symbol; a symbol is removed entirely when its
      qty hits zero (the closed P&L is rolled into ``realized_pnl``).
    - ``nav = cash + sum(p.market_value for p in positions)``.
    - ``peak_nav`` is the running maximum of NAV over the portfolio's
      lifetime; ``max_drawdown`` is the worst observed ``(peak − nav)/peak``.

    These invariants are enforced by ``apply_fill`` and ``mark_to_market`` —
    callers should never poke fields directly.
    """

    portfolio_id: str
    base_currency: str
    starting_cash: float
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    realized_pnl: float = 0.0
    peak_nav: float = 0.0
    max_drawdown: float = 0.0
    created_ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_update_ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: int = SCHEMA_VERSION

    # -- derived metrics ---------------------------------------------------

    @property
    def market_value(self) -> float:
        """Sum of mark-to-market across all positions (long + short)."""
        return sum(p.market_value for p in self.positions.values())

    @property
    def nav(self) -> float:
        """Net asset value: cash plus mark-to-market of every position."""
        return self.cash + self.market_value

    @property
    def gross_exposure(self) -> float:
        """Sum of ``|market_value|`` over all positions.  Equal to NAV when 100% invested long."""
        return sum(abs(p.market_value) for p in self.positions.values())

    @property
    def net_exposure(self) -> float:
        """Signed sum of position market values (positive = net long)."""
        return self.market_value

    @property
    def unrealized_pnl(self) -> float:
        """Sum of unrealized P&L across all open positions."""
        return sum(p.unrealized_pnl for p in self.positions.values())

    @property
    def gross_leverage(self) -> float:
        """``gross_exposure / nav`` — useful for the risk-gate concentration check (M5)."""
        if self.nav == 0:
            return 0.0
        return self.gross_exposure / self.nav

    def weight(self, symbol: str) -> float:
        """Portfolio weight of ``symbol`` as ``market_value / nav`` (signed)."""
        if self.nav == 0 or symbol not in self.positions:
            return 0.0
        return self.positions[symbol].market_value / self.nav

    # -- mutators (only the execution layer should call these) -------------

    def mark_to_market(self, prices: Dict[str, float], ts: str) -> None:
        """Update ``last_price``/``last_mark_ts`` for every position with a new price.

        Symbols missing from ``prices`` are left at their previous mark — we
        do *not* invent a price (which would silently bias NAV).  Callers
        should always pass the full price set when computing risk metrics or
        a daily snapshot.

        ``peak_nav`` and ``max_drawdown`` are updated atomically with the
        new NAV.
        """
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].last_price = float(price)
                self.positions[symbol].last_mark_ts = ts
        new_nav = self.nav
        if new_nav > self.peak_nav:
            self.peak_nav = new_nav
        if self.peak_nav > 0:
            dd = (self.peak_nav - new_nav) / self.peak_nav
            if dd > self.max_drawdown:
                self.max_drawdown = dd
        self.last_update_ts = ts

    def apply_fill(
        self,
        symbol: str,
        qty_delta: float,
        fill_price: float,
        fees: float,
        ts: str,
    ) -> None:
        """Apply a single ``Fill`` to the book.

        ``qty_delta`` is signed: positive for a buy (increases long / covers
        short), negative for a sell.  ``fees`` are always paid in cash.

        Cost-basis update rules:

        - Adding to a long (or short) position rolls a fresh weighted average
          of the new fill into the existing ``avg_cost``.
        - Reducing or closing a position realises P&L against the old
          ``avg_cost`` and leaves ``avg_cost`` unchanged for the remaining
          qty.
        - Crossing zero (long -> short or short -> long) realises P&L on the
          closing slice and starts a new average on the opening slice.

        Cash is decremented by ``qty_delta * fill_price + fees`` so a buy
        debits cash and a sell credits it; fees are *always* a debit.

        Raises ``ValueError`` on a zero-qty fill — we treat that as a caller
        bug rather than silently no-op'ing.
        """
        if qty_delta == 0:
            raise ValueError(f"apply_fill called with qty_delta=0 for {symbol}")

        cash_delta = -(qty_delta * fill_price) - fees
        self.cash += cash_delta

        existing = self.positions.get(symbol)
        if existing is None:
            self.positions[symbol] = Position(
                symbol=symbol,
                qty=qty_delta,
                avg_cost=fill_price,
                last_price=fill_price,
                last_mark_ts=ts,
                opened_ts=ts,
            )
        else:
            new_qty = existing.qty + qty_delta
            same_side = (existing.qty > 0 and qty_delta > 0) or (existing.qty < 0 and qty_delta < 0)
            crossing = (existing.qty > 0 and new_qty < 0) or (existing.qty < 0 and new_qty > 0)

            if same_side:
                total_cost = existing.qty * existing.avg_cost + qty_delta * fill_price
                existing.qty = new_qty
                existing.avg_cost = total_cost / new_qty
            elif crossing:
                # Close the full existing position then open a new one on the
                # opposite side.  Realised P&L is (fill - avg_cost) * existing.qty
                # — signed by ``existing.qty`` so it works for both
                # long->short (positive existing.qty) and short->long
                # (negative existing.qty) crossings.
                self.realized_pnl += (fill_price - existing.avg_cost) * existing.qty
                existing.qty = new_qty
                existing.avg_cost = fill_price
            else:
                # Partial close on the same side.  The closed slice has the
                # opposite sign of ``existing.qty``; ``-qty_delta`` is the
                # number of shares closed in the *direction of existing*, so
                # P&L = (fill - avg) * (closed-slice in existing's sign) =
                # (fill - avg) * (-qty_delta) when existing>0, and
                # (fill - avg) * qty_delta when existing<0.  The signed
                # ``-qty_delta`` already encodes that because qty_delta and
                # existing.qty have opposite signs in this branch.
                self.realized_pnl += (fill_price - existing.avg_cost) * (-qty_delta)
                existing.qty = new_qty

            if existing.qty == 0:
                del self.positions[symbol]
            else:
                existing.last_price = fill_price
                existing.last_mark_ts = ts

        self.last_update_ts = ts

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "portfolio_id": self.portfolio_id,
            "base_currency": self.base_currency,
            "starting_cash": self.starting_cash,
            "cash": self.cash,
            "positions": {s: p.to_dict() for s, p in self.positions.items()},
            "realized_pnl": self.realized_pnl,
            "peak_nav": self.peak_nav,
            "max_drawdown": self.max_drawdown,
            "created_ts": self.created_ts,
            "last_update_ts": self.last_update_ts,
        }

    def to_snapshot(self) -> dict:
        """Read-only view safe for the LLM context window.

        Strips internal fields the model doesn't need and adds derived
        metrics that are easier for the LLM to reason about than raw avg
        cost / cash.  Used by the ``get_portfolio_state`` tool in M2.
        """
        return {
            "nav": round(self.nav, 2),
            "cash": round(self.cash, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "gross_exposure": round(self.gross_exposure, 2),
            "net_exposure": round(self.net_exposure, 2),
            "gross_leverage": round(self.gross_leverage, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "positions": [
                {
                    "symbol": p.symbol,
                    "qty": p.qty,
                    "avg_cost": round(p.avg_cost, 4),
                    "last_price": round(p.last_price, 4),
                    "market_value": round(p.market_value, 2),
                    "weight": round(self.weight(p.symbol), 4),
                    "unrealized_pnl": round(p.unrealized_pnl, 2),
                }
                for p in self.positions.values()
            ],
            "last_update_ts": self.last_update_ts,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PortfolioState":
        version = data.get("schema_version", 0)
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"PortfolioState schema_version mismatch: file has {version}, "
                f"code expects {SCHEMA_VERSION}.  Run the migration tool before loading."
            )
        return cls(
            portfolio_id=data["portfolio_id"],
            base_currency=data["base_currency"],
            starting_cash=float(data["starting_cash"]),
            cash=float(data["cash"]),
            positions={s: Position.from_dict(p) for s, p in data.get("positions", {}).items()},
            realized_pnl=float(data.get("realized_pnl", 0.0)),
            peak_nav=float(data.get("peak_nav", 0.0)),
            max_drawdown=float(data.get("max_drawdown", 0.0)),
            created_ts=data["created_ts"],
            last_update_ts=data["last_update_ts"],
            schema_version=version,
        )


# ---------------------------------------------------------------------------
# Store: atomic JSON persistence
# ---------------------------------------------------------------------------


class PortfolioStateStore:
    """Atomic file-backed store for ``PortfolioState``.

    Every save writes to ``<path>.tmp`` then ``os.replace`` it into place so a
    partial write never leaves a half-corrupted JSON file on disk — the same
    crash-safety pattern the OHLCV cache uses.  Reads use a simple JSON load.

    The store is intentionally thread-safe via instance-level lock; concurrent
    runs against the same portfolio file should still serialise their writes,
    which is exactly the right behaviour for a single book.
    """

    def __init__(self, path: str | os.PathLike) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def exists(self) -> bool:
        return self.path.is_file() and self.path.stat().st_size > 0

    def load(self) -> PortfolioState:
        if not self.exists():
            raise FileNotFoundError(f"No portfolio state at {self.path}")
        with self.path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return PortfolioState.from_dict(data)

    def save(self, state: PortfolioState) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(state.to_dict(), fh, indent=2, sort_keys=True)
            os.replace(tmp, self.path)
        except OSError as exc:
            logger.warning("could not persist portfolio state to %s: %s", self.path, exc)
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            raise


# ---------------------------------------------------------------------------
# Convenience helpers (used by graph runner / web backend)
# ---------------------------------------------------------------------------


def _default_path(portfolio_id: str) -> Path:
    base = os.environ.get("TRADINGAGENTS_CACHE_DIR")
    root = Path(base) if base else Path.home() / ".tradingagents"
    return root / "portfolios" / f"{portfolio_id}.json"


def load_portfolio_state(
    portfolio_id: str = "default",
    *,
    starting_cash: float = 100_000.0,
    base_currency: str = "USD",
    path: Optional[str | os.PathLike] = None,
) -> PortfolioState:
    """Load a portfolio from disk, or initialise a fresh one.

    The optional ``starting_cash`` / ``base_currency`` are used **only** when
    no state file exists yet — once the file is created, those fields come
    from the JSON.  This avoids silently re-initialising a live book to its
    config defaults if the user changes ``starting_cash`` between runs.
    """
    store_path = Path(path) if path is not None else _default_path(portfolio_id)
    store = PortfolioStateStore(store_path)
    if store.exists():
        return store.load()
    state = PortfolioState(
        portfolio_id=portfolio_id,
        base_currency=base_currency,
        starting_cash=starting_cash,
        cash=starting_cash,
        peak_nav=starting_cash,
    )
    store.save(state)
    return state


def save_portfolio_state(
    state: PortfolioState,
    *,
    path: Optional[str | os.PathLike] = None,
) -> None:
    """Persist ``state`` atomically to disk.

    Uses the same path resolution as ``load_portfolio_state`` so the two
    helpers always agree on where a portfolio lives.
    """
    store_path = Path(path) if path is not None else _default_path(state.portfolio_id)
    PortfolioStateStore(store_path).save(state)
