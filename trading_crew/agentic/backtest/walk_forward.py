"""Walk-forward folds with mandatory embargo (paper §8.1).

The paper §8.1's central finding on backtest integrity:

  "Most reported agentic-trading results are leaked: the same returns
   feed both the calibration window and the held-out test window, so
   the strategy appears to anticipate signals it was tuned on."

The remedy is a strict walk-forward scheme:

1. A ``train`` window is used to tune any free parameter (prompt
   variants, regime thresholds, sizing config) or to seed any in-context
   memory.
2. An ``embargo`` gap follows the train window — no data from the gap
   is used by either the calibration or the evaluation.  Its length must
   be at least the longest horizon any signal looks ahead for outcomes.
3. The ``test`` window immediately follows the embargo.  Decisions made
   on test bars use only information that became public *before* that
   bar's timestamp.

Subsequent folds slide forward by ``test_size`` so the test windows
form a contiguous, non-overlapping out-of-sample period.  This module
generates the (train, embargo, test) index triples; the caller is
responsible for keeping their state (prompt cache, memory store) consistent
with the fold boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class WalkForwardConfig:
    """Fold geometry for walk-forward evaluation.

    All sizes are in *bars* (typically trading days).

    - ``train_size``: bars used for parameter tuning / warm-up.
    - ``embargo_size``: bars discarded between train and test to prevent
                       outcome leakage.  Must be >= the max look-ahead
                       used by any feature or label.  The paper §8.1
                       recommends embargo >= 2 × max signal horizon.
    - ``test_size``: bars in each out-of-sample evaluation block.  Folds
                     slide forward by this amount, so the per-fold test
                     windows tile the dataset.
    - ``min_train_size``: optional minimum size for the first train
                          window if the user wants an expanding initial
                          window (default: same as ``train_size``).
    - ``expanding``: if True, each subsequent train window grows by
                     ``test_size``; if False, train window is rolling.
    """

    train_size: int
    embargo_size: int
    test_size: int
    min_train_size: Optional[int] = None
    expanding: bool = False

    def __post_init__(self):
        if self.train_size <= 0:
            raise ValueError(f"train_size must be > 0; got {self.train_size}")
        if self.embargo_size < 0:
            raise ValueError(f"embargo_size must be >= 0; got {self.embargo_size}")
        if self.test_size <= 0:
            raise ValueError(f"test_size must be > 0; got {self.test_size}")
        mts = self.min_train_size if self.min_train_size is not None else self.train_size
        if mts <= 0 or mts > self.train_size:
            raise ValueError(
                f"min_train_size must be in (0, train_size]; got {mts} (train_size={self.train_size})"
            )


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold.

    Indices are inclusive-start, exclusive-end (``range``-compatible).

    Invariants enforced by ``generate_folds``:

    - ``train_end <= embargo_start``
    - ``embargo_end == test_start``     (no gap, the embargo is the gap)
    - ``test_start - train_end == embargo_size``
    - ``train`` and ``test`` indices are disjoint
    """

    fold_id: int
    train_start: int
    train_end: int
    embargo_start: int
    embargo_end: int
    test_start: int
    test_end: int

    @property
    def train_indices(self) -> range:
        return range(self.train_start, self.train_end)

    @property
    def embargo_indices(self) -> range:
        return range(self.embargo_start, self.embargo_end)

    @property
    def test_indices(self) -> range:
        return range(self.test_start, self.test_end)


def generate_folds(n_obs: int, config: WalkForwardConfig) -> List[Fold]:
    """Materialise the list of walk-forward folds for an ``n_obs``-bar dataset.

    The first fold uses ``[0, train_size)`` for train, ``[train_size,
    train_size + embargo_size)`` for embargo, ``[train_size +
    embargo_size, train_size + embargo_size + test_size)`` for test.

    Subsequent folds shift the test window forward by ``test_size``:

    - In **rolling** mode (``expanding=False``) the train window also
      slides forward by ``test_size`` so the training set stays the
      same size — useful when you want to detect parameter drift.
    - In **expanding** mode (``expanding=True``) the train window grows
      so every fold sees more history — useful when you trust the older
      data and just want more samples.

    Folds stop when the next test window would extend past ``n_obs``;
    no partial folds are generated (we refuse to evaluate on a
    fractional test window).
    """
    folds: List[Fold] = []
    fold_id = 0
    train_start = 0
    train_end = config.train_size

    while True:
        embargo_start = train_end
        embargo_end = embargo_start + config.embargo_size
        test_start = embargo_end
        test_end = test_start + config.test_size

        if test_end > n_obs:
            break  # not enough data for a full test window

        folds.append(
            Fold(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                embargo_start=embargo_start,
                embargo_end=embargo_end,
                test_start=test_start,
                test_end=test_end,
            )
        )

        fold_id += 1
        if config.expanding:
            train_end = train_end + config.test_size
            # train_start stays 0
        else:
            train_start = train_start + config.test_size
            train_end = train_end + config.test_size

    return folds


def assert_no_leakage(folds: Sequence[Fold]) -> None:
    """Raise ``AssertionError`` if any fold's train and test indices overlap.

    Cheap belt-and-suspenders check for callers that compose folds
    themselves; ``generate_folds`` already preserves this invariant.
    """
    for f in folds:
        train_set = set(f.train_indices)
        test_set = set(f.test_indices)
        embargo_set = set(f.embargo_indices)
        assert train_set.isdisjoint(test_set), f"fold {f.fold_id}: train overlaps test"
        assert train_set.isdisjoint(embargo_set), f"fold {f.fold_id}: train overlaps embargo"
        assert test_set.isdisjoint(embargo_set), f"fold {f.fold_id}: test overlaps embargo"


# ---------------------------------------------------------------------------
# Combinatorial Purged Cross-Validation (López de Prado, AFML ch. 12)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CPCVConfig:
    """N-choose-k partition geometry for CPCV.

    - ``n_groups``: number of contiguous groups the observations are
                    split into (default 6 — gives 15 folds with k=2).
    - ``k_test``:   number of groups held out for test in each fold (k=2
                    is the LdP default; k=1 collapses to standard CV).
    - ``embargo_size``: bars discarded on each *side* of every test
                    group to prevent label leakage from contiguous
                    train neighbours.

    Yields N-choose-k folds, each with a contiguous-train index
    sequence (union of the non-test groups, minus the embargo).
    """

    n_groups: int = 6
    k_test: int = 2
    embargo_size: int = 1

    def __post_init__(self):
        if self.n_groups < 2:
            raise ValueError("n_groups must be >= 2")
        if not (1 <= self.k_test < self.n_groups):
            raise ValueError("k_test must be in [1, n_groups-1]")
        if self.embargo_size < 0:
            raise ValueError("embargo_size must be >= 0")


@dataclass(frozen=True)
class CPCVFold:
    """One CPCV fold.

    Unlike :class:`Fold`, the test set is a *union* of contiguous
    groups (which can be non-adjacent in the original time axis).  We
    represent it as a tuple of ``range`` objects so callers can iterate
    each contiguous run and still see the natural ordering.
    """

    fold_id: int
    train_ranges: Tuple[range, ...]
    test_ranges: Tuple[range, ...]
    embargo_ranges: Tuple[range, ...]

    @property
    def train_indices(self) -> Tuple[int, ...]:
        return tuple(i for r in self.train_ranges for i in r)

    @property
    def test_indices(self) -> Tuple[int, ...]:
        return tuple(i for r in self.test_ranges for i in r)


def generate_cpcv_folds(n_obs: int, config: CPCVConfig) -> List[CPCVFold]:
    """Materialise the CPCV fold list.

    Partitions ``[0, n_obs)`` into ``n_groups`` near-equal contiguous
    blocks, then enumerates every combination of ``k_test`` groups as
    the test set.  Per-group embargoes are dropped from the train side
    so an observation's outcome can't leak into an adjacent train
    sample.
    """
    from itertools import combinations

    if n_obs < config.n_groups:
        return []  # nothing useful with less than one obs per group

    # Compute group boundaries (start, end) in observation index space.
    base = n_obs // config.n_groups
    rem = n_obs % config.n_groups
    groups: List[range] = []
    cursor = 0
    for g in range(config.n_groups):
        size = base + (1 if g < rem else 0)
        groups.append(range(cursor, cursor + size))
        cursor += size

    folds: List[CPCVFold] = []
    fold_id = 0
    for combo in combinations(range(config.n_groups), config.k_test):
        # Test = union of the chosen groups.
        test_ranges = tuple(groups[g] for g in combo)
        test_set = set(i for r in test_ranges for i in r)

        # Build per-test-group embargo windows.
        embargo_set: set[int] = set()
        embargo_ranges_list: List[range] = []
        for g in combo:
            grp = groups[g]
            left = range(max(0, grp.start - config.embargo_size), grp.start)
            right = range(grp.stop, min(n_obs, grp.stop + config.embargo_size))
            if len(left) > 0:
                embargo_ranges_list.append(left)
                embargo_set.update(left)
            if len(right) > 0:
                embargo_ranges_list.append(right)
                embargo_set.update(right)

        # Train = everything else, minus the embargoes.
        train_idx_set = set(range(n_obs)) - test_set - embargo_set
        # Compact the train indices into contiguous runs for friendly
        # iteration (so downstream code can still loop range objects).
        train_ranges: List[range] = []
        run_start: Optional[int] = None
        prev: Optional[int] = None
        for i in sorted(train_idx_set):
            if run_start is None:
                run_start = i
                prev = i
            elif i == prev + 1:
                prev = i
            else:
                train_ranges.append(range(run_start, prev + 1))
                run_start = i
                prev = i
        if run_start is not None and prev is not None:
            train_ranges.append(range(run_start, prev + 1))

        folds.append(CPCVFold(
            fold_id=fold_id,
            train_ranges=tuple(train_ranges),
            test_ranges=test_ranges,
            embargo_ranges=tuple(embargo_ranges_list),
        ))
        fold_id += 1
    return folds


def walk_forward_cpcv(n_obs: int, config: CPCVConfig) -> List[CPCVFold]:
    """Convenience wrapper around :func:`generate_cpcv_folds` for the API.

    Named to mirror :func:`generate_folds` so callers swapping between
    WF and CPCV only have to flip the function name.  Identity wrapper
    for now; reserved for any future "purge" tweaks that should apply
    in the CPCV variant but not the raw fold generator.
    """
    return generate_cpcv_folds(n_obs, config)
