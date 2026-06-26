"""Time-series-aware splitting. Never shuffles; always preserves chronology.

Provides a simple contiguous train/val/test split and a purged, embargoed
walk-forward iterator for cross-validation that respects label overlap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


def time_series_split(
    n: int, train: float = 0.6, val: float = 0.2,
) -> tuple[slice, slice, slice]:
    """Chronological train/validation/test slices.

    Args:
        n: Number of observations.
        train: Fraction for training.
        val: Fraction for validation (test is the remainder).

    Returns:
        Three :class:`slice` objects over ``range(n)``, in time order.
    """
    if not 0 < train < 1 or not 0 <= val < 1 or train + val >= 1:
        raise ValueError("Require 0 < train, 0 <= val, and train + val < 1.")
    i_train = int(n * train)
    i_val = int(n * (train + val))
    return slice(0, i_train), slice(i_train, i_val), slice(i_val, n)


@dataclass
class PurgedWalkForward:
    """Expanding/rolling walk-forward CV with purge + embargo.

    Each fold trains on data strictly before the test block, with an ``embargo``
    gap of observations removed between train and test so that overlapping
    forward labels cannot leak across the boundary.

    Args:
        n_splits: Number of test folds.
        embargo: Observations to drop between train end and test start
            (set to the label horizon to fully purge overlap).
        expanding: If True, training window grows each fold; else it rolls.
    """

    n_splits: int = 5
    embargo: int = 21
    expanding: bool = True

    def split(self, n: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, test_idx)`` integer-position arrays."""
        if n < self.n_splits + 2:
            return
        bounds = np.linspace(0, n, self.n_splits + 2, dtype=int)
        for k in range(1, self.n_splits + 1):
            train_end = bounds[k]
            test_start, test_end = bounds[k], bounds[k + 1]
            train_start = 0 if self.expanding else bounds[k - 1]
            cut = max(train_start, train_end - self.embargo)
            if cut <= train_start or test_end <= test_start:
                continue
            yield (np.arange(train_start, cut), np.arange(test_start, test_end))
