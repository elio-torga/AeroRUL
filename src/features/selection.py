"""Supervised feature selection as a scikit-learn transformer.

The temporal feature generator produces ~500 heavily redundant columns
(rolling means of neighbouring windows are almost collinear). The selector
applies three successive filters, all fitted on training data only:

1. **Variance filter** - drop (near-)constant columns.
2. **Redundancy filter** - greedy de-duplication: features are ranked by their
   absolute correlation with the target and a feature is kept only if its
   absolute correlation with every already-kept feature is below a threshold.
3. **Relevance filter** - impurity importance of a fast Extra-Trees model
   fitted on a random sub-sample; the ``top_k`` features are retained.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.utils.validation import check_is_fitted


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Variance -> correlation -> importance feature selection.

    Parameters
    ----------
    top_k:
        Number of features kept after the importance ranking.
    correlation_threshold:
        Maximum absolute pairwise correlation tolerated between kept features.
    variance_threshold:
        Minimum variance of a feature.
    passthrough:
        Columns always forwarded untouched (identifiers such as ``unit``).
    """

    def __init__(self, top_k: int = 120, correlation_threshold: float = 0.98,
                 variance_threshold: float = 1e-6, sample_size: int = 15000,
                 n_estimators: int = 80, passthrough: Sequence[str] = ("unit",),
                 random_state: Optional[int] = 42):
        self.top_k = top_k
        self.correlation_threshold = correlation_threshold
        self.variance_threshold = variance_threshold
        self.sample_size = sample_size
        self.n_estimators = n_estimators
        self.passthrough = passthrough
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y=None):
        if y is None:
            raise ValueError("FeatureSelector is supervised and needs y.")
        y = np.asarray(y, dtype=float)
        candidates = [c for c in X.columns if c not in set(self.passthrough)]
        values = X[candidates].to_numpy(dtype=np.float64)

        # 1. variance filter
        variances = values.var(axis=0)
        keep = variances > self.variance_threshold
        candidates = [c for c, k in zip(candidates, keep) if k]
        values = values[:, keep]

        # 2. greedy redundancy filter, most target-relevant first
        standardized = (values - values.mean(0)) / values.std(0)
        y_std = (y - y.mean()) / y.std()
        target_corr = np.abs(standardized.T @ y_std) / len(y)
        order = np.argsort(-target_corr)
        corr = np.abs(np.corrcoef(values, rowvar=False))
        kept_idx: list = []
        for idx in order:
            if not kept_idx or corr[idx, kept_idx].max() < self.correlation_threshold:
                kept_idx.append(idx)
        self.n_after_redundancy_ = len(kept_idx)
        decorrelated = [candidates[i] for i in kept_idx]

        # 3. importance ranking on a sub-sample
        rng = np.random.default_rng(self.random_state)
        n = min(self.sample_size, len(X))
        rows = rng.choice(len(X), size=n, replace=False)
        model = ExtraTreesRegressor(n_estimators=self.n_estimators, max_depth=12,
                                    min_samples_leaf=10, max_features=0.3, n_jobs=-1,
                                    random_state=self.random_state)
        model.fit(X[decorrelated].to_numpy()[rows], y[rows])
        importances = pd.Series(model.feature_importances_, index=decorrelated)
        self.importances_ = importances.sort_values(ascending=False)
        self.selected_features_ = list(self.importances_.index[: self.top_k])
        self.target_correlation_ = pd.Series(target_corr, index=candidates)
        self.n_input_features_ = len(X.columns) - len(set(self.passthrough) & set(X.columns))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "selected_features_")
        keep = [c for c in self.passthrough if c in X.columns] + self.selected_features_
        return X[keep]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(self.passthrough) + self.selected_features_, dtype=object)
