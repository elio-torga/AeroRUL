"""Target construction and leakage-safe splitting strategies.

Two principles drive every function in this module:

1. **The engine is the unit of independence.** Cycles of one engine are
   strongly autocorrelated, so any split that puts cycles of the same engine
   on both sides leaks information. All splits operate on whole engines.
2. **Validation must look like production.** In service the model sees an
   engine at an arbitrary point of its life, not at failure. Validation
   engines are therefore evaluated at random truncation points, mirroring the
   construction of the official test set.
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd


def add_train_rul(df: pd.DataFrame, max_rul: int) -> pd.DataFrame:
    """Add ``rul_true`` (cycles to failure) and the clipped target ``rul``.

    In run-to-failure data the last recorded cycle is the failure cycle, so
    ``RUL = max_cycle - cycle``. The piece-wise linear target
    ``min(RUL, max_rul)`` encodes the fact that degradation is not observable
    during the early, healthy part of life.
    """
    out = df.copy()
    max_cycle = out.groupby("unit")["cycle"].transform("max")
    out["rul_true"] = (max_cycle - out["cycle"]).astype(int)
    out["rul"] = out["rul_true"].clip(upper=max_rul).astype(float)
    return out


def add_test_rul(df: pd.DataFrame, final_rul: pd.Series, max_rul: int) -> pd.DataFrame:
    """Back-propagate the official end-of-record RUL to every test cycle."""
    out = df.copy()
    last = out.groupby("unit")["cycle"].transform("max")
    out["rul_true"] = (out["unit"].map(final_rul) + last - out["cycle"]).astype(int)
    out["rul"] = out["rul_true"].clip(upper=max_rul).astype(float)
    return out


def last_cycle_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return the most recent observation of every engine."""
    idx = df.groupby("unit")["cycle"].idxmax()
    return df.loc[idx.sort_values()]


def split_engines(df: pd.DataFrame, validation_fraction: float,
                  seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Engine-level random hold-out split."""
    units = np.sort(df["unit"].unique())
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(len(units) * validation_fraction)))
    val_units = rng.choice(units, size=n_val, replace=False)
    is_val = df["unit"].isin(val_units)
    return df.loc[~is_val].copy(), df.loc[is_val].copy()


def sample_truncation_points(df: pd.DataFrame, n_per_engine: int, min_cycles: int,
                             seed: int) -> pd.DataFrame:
    """Pick ``n_per_engine`` random "current time" rows per run-to-failure engine.

    Because every engineered feature is causal (it only uses past cycles), the
    feature row at cycle *t* is exactly what the model would see if the engine
    had been observed up to *t* only. Selecting rows therefore emulates
    truncated test trajectories without recomputing features.
    """
    rng = np.random.default_rng(seed)
    picks = []
    for _, grp in df.groupby("unit", sort=True):
        eligible = grp.index[grp["cycle"] >= min(min_cycles, grp["cycle"].max())]
        size = min(n_per_engine, len(eligible))
        picks.append(rng.choice(eligible, size=size, replace=False))
    return df.loc[np.sort(np.concatenate(picks))]


def subsample_rows(df: pd.DataFrame, stride: int) -> pd.DataFrame:
    """Keep every ``stride``-th cycle of each engine (always keeps the last one)."""
    if stride <= 1:
        return df
    rank_from_end = df.groupby("unit").cumcount(ascending=False)
    return df.loc[rank_from_end % stride == 0]


class EngineGroupKFold:
    """K-fold cross-validation over engines (scikit-learn splitter API).

    Contrary to ``GroupKFold`` it does not need ``groups`` to be passed
    explicitly: when ``groups`` is ``None`` the engine id is read from the
    ``unit`` column of ``X``. This makes it usable inside estimators that do
    not forward groups, such as ``StackingRegressor``.

    Why not ``TimeSeriesSplit``? C-MAPSS engines are independent units; a
    single global time axis does not exist. The temporal-leakage risk is
    *within* an engine (overlapping rolling windows), which engine-grouped
    folds eliminate while every feature remains strictly causal.
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = True,
                 random_state: Optional[int] = None, unit_col: str = "unit"):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.unit_col = unit_col

    def split(self, X, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        units = np.asarray(groups) if groups is not None else X[self.unit_col].to_numpy()
        unique = np.unique(units)
        if self.shuffle:
            unique = np.random.default_rng(self.random_state).permutation(unique)
        for fold in np.array_split(unique, self.n_splits):
            test_mask = np.isin(units, fold)
            yield np.flatnonzero(~test_mask), np.flatnonzero(test_mask)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits
