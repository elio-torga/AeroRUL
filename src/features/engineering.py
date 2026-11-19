"""Custom scikit-learn transformers for turbofan degradation features.

All transformers

* accept and return ``pandas.DataFrame`` objects carrying ``unit`` and
  ``cycle`` columns (needed for per-engine grouped operations);
* preserve the input row order and index, so targets stay aligned inside a
  ``Pipeline``;
* are **causal**: the feature value at cycle *t* only uses cycles <= *t* of
  the same engine. The same code therefore serves offline training and
  online inference on partial trajectories.

Rows must be chronologically ordered within each engine (guaranteed by
``src.data.load_data``).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

ID_COLS = ["unit", "cycle"]


def _grouped_rolling(df: pd.DataFrame, cols: Sequence[str], window: int,
                     stat: str, min_periods: int = 1) -> pd.DataFrame:
    """Per-engine trailing-window statistic, aligned back to ``df.index``."""
    rolled = df.groupby("unit", sort=False)[list(cols)].rolling(
        window, min_periods=min_periods)
    out = getattr(rolled, stat)()
    return out.reset_index(level=0, drop=True).reindex(df.index)


def _grouped_ewm(df: pd.DataFrame, cols: Sequence[str], span: int) -> pd.DataFrame:
    """Per-engine exponentially weighted moving average (causal, adjust=False)."""
    out = df.groupby("unit", sort=False)[list(cols)].ewm(span=span, adjust=False).mean()
    return out.reset_index(level=0, drop=True).reindex(df.index)


def _grouped_slope(df: pd.DataFrame, values: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling ordinary-least-squares slope of ``values`` against ``cycle``.

    Uses the closed form ``cov(t, x) / var(t)`` computed from rolling means,
    which is fully vectorised. The slope is the local degradation rate.
    """
    t = df["cycle"].astype(float)
    frame = values.copy()
    frame["unit"] = df["unit"]
    for col in values.columns:
        frame[f"__tx_{col}"] = values[col] * t
    frame["__t"] = t
    frame["__tt"] = t * t
    cols = [c for c in frame.columns if c != "unit"]
    means = _grouped_rolling(frame, cols, window, "mean", min_periods=2)
    var_t = means["__tt"] - means["__t"] ** 2
    slopes = {}
    for col in values.columns:
        cov = means[f"__tx_{col}"] - means["__t"] * means[col]
        slopes[col] = np.where(var_t > 1e-9, cov / var_t.replace(0, np.nan), 0.0)
    return pd.DataFrame(slopes, index=df.index).fillna(0.0)


class OperatingConditionNormalizer(BaseEstimator, TransformerMixin):
    """Remove the operating-regime effect from sensor readings.

    In FD002/FD004 the engines fly in six operating regimes (altitude, Mach,
    throttle). Regime changes move sensor values far more than degradation
    does. The transformer

    1. clusters the three operating settings with K-Means (fitted on train);
    2. z-scores every sensor *within* its regime;
    3. adds a one-hot regime encoding and, per engine, the cumulative share of
       cycles flown in each regime (operating history drives wear).

    Parameters
    ----------
    sensors:
        Sensor columns to normalise.
    n_conditions:
        Number of regimes; ``1`` performs a global standardisation.
    op_cols:
        Operating-setting columns.
    add_history:
        Add cumulative regime-share features.
    """

    def __init__(self, sensors: Sequence[str], n_conditions: int = 1,
                 op_cols: Sequence[str] = ("op_setting_1", "op_setting_2", "op_setting_3"),
                 add_history: bool = True, random_state: int = 42):
        self.sensors = sensors
        self.n_conditions = n_conditions
        self.op_cols = op_cols
        self.add_history = add_history
        self.random_state = random_state

    def _assign(self, X: pd.DataFrame) -> np.ndarray:
        if self.n_conditions == 1:
            return np.zeros(len(X), dtype=int)
        ops = self.scaler_.transform(X[list(self.op_cols)].to_numpy())
        return self.kmeans_.predict(ops)

    def fit(self, X: pd.DataFrame, y=None):
        sensors = list(self.sensors)
        if self.n_conditions > 1:
            ops = X[list(self.op_cols)].to_numpy()
            self.scaler_ = StandardScaler().fit(ops)
            self.kmeans_ = KMeans(n_clusters=self.n_conditions, n_init=10,
                                  random_state=self.random_state).fit(
                self.scaler_.transform(ops))
        labels = self._assign(X)
        grouped = X[sensors].groupby(labels)
        self.means_ = grouped.mean()
        # Sensors that are constant inside a regime get unit scale -> output 0.
        self.stds_ = grouped.std(ddof=0).where(lambda s: s > 1e-8, 1.0)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "means_")
        sensors = list(self.sensors)
        labels = self._assign(X)
        mu = self.means_.reindex(labels).to_numpy()
        sd = self.stds_.reindex(labels).to_numpy()
        out = X[ID_COLS].copy()
        out[sensors] = (X[sensors].to_numpy() - mu) / sd
        if self.n_conditions > 1:
            onehot = np.eye(self.n_conditions)[labels]
            oc_cols = [f"op_condition_{k}" for k in range(self.n_conditions)]
            out[oc_cols] = onehot
            if self.add_history:
                hist = pd.DataFrame(onehot, index=X.index, columns=oc_cols)
                hist["unit"] = X["unit"].to_numpy()
                cum = hist.groupby("unit", sort=False)[oc_cols].cumsum()
                n_seen = hist.groupby("unit", sort=False).cumcount() + 1
                out[[f"hist_share_{c}" for c in oc_cols]] = cum.div(n_seen, axis=0).to_numpy()
        return out

    def get_feature_names_out(self, input_features=None):
        names = ID_COLS + list(self.sensors)
        if self.n_conditions > 1:
            oc = [f"op_condition_{k}" for k in range(self.n_conditions)]
            names += oc + ([f"hist_share_{c}" for c in oc] if self.add_history else [])
        return np.asarray(names, dtype=object)


class TemporalFeatureGenerator(BaseEstimator, TransformerMixin):
    """Causal per-engine temporal features from normalised sensors.

    Generated families (``n_sensors`` columns each):

    * rolling ``mean/std/min/max/skew/kurt`` for every window size;
    * exponentially weighted moving averages for every span;
    * lag features (value ``k`` cycles ago);
    * 1st and 2nd order differences of an EWMA-smoothed signal;
    * rolling OLS slopes (local degradation rate).

    Undefined values at the very beginning of a trajectory (e.g. the skewness
    of two points) are filled with 0, i.e. "no evidence of change" in the
    normalised sensor space.
    """

    def __init__(self, sensors: Sequence[str], windows: Sequence[int] = (5, 10, 20, 30),
                 stats: Sequence[str] = ("mean", "std", "min", "max", "skew", "kurt"),
                 ewm_spans: Sequence[int] = (5, 10, 20), lags: Sequence[int] = (1, 3, 5),
                 diff_smoothing_span: int = 10, slope_windows: Sequence[int] = (10, 30),
                 keep_raw: bool = True):
        self.sensors = sensors
        self.windows = windows
        self.stats = stats
        self.ewm_spans = ewm_spans
        self.lags = lags
        self.diff_smoothing_span = diff_smoothing_span
        self.slope_windows = slope_windows
        self.keep_raw = keep_raw

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        sensors = list(self.sensors)
        blocks: List[pd.DataFrame] = []
        min_periods = {"skew": 3, "kurt": 4, "std": 2}
        for window in self.windows:
            for stat in self.stats:
                block = _grouped_rolling(X, sensors, window, stat, min_periods.get(stat, 1))
                block.columns = [f"{s}_roll{window}_{stat}" for s in sensors]
                blocks.append(block)
        for span in self.ewm_spans:
            block = _grouped_ewm(X, sensors, span)
            block.columns = [f"{s}_ewm{span}" for s in sensors]
            blocks.append(block)
        grouped = X.groupby("unit", sort=False)[sensors]
        for lag in self.lags:
            block = grouped.shift(lag).fillna(X[sensors])
            block.columns = [f"{s}_lag{lag}" for s in sensors]
            blocks.append(block)
        smooth = _grouped_ewm(X, sensors, self.diff_smoothing_span)
        smooth_frame = smooth.assign(unit=X["unit"])
        d1 = smooth_frame.groupby("unit", sort=False)[sensors].diff().fillna(0.0)
        d2 = d1.assign(unit=X["unit"]).groupby("unit", sort=False)[sensors].diff().fillna(0.0)
        blocks.append(d1.set_axis([f"{s}_diff1" for s in sensors], axis=1))
        blocks.append(d2.set_axis([f"{s}_diff2" for s in sensors], axis=1))
        for window in self.slope_windows:
            block = _grouped_slope(X, smooth, window)
            block.columns = [f"{s}_slope{window}" for s in sensors]
            blocks.append(block)
        features = pd.concat(blocks, axis=1).fillna(0.0).astype(np.float32)
        base = X if self.keep_raw else X.drop(columns=sensors)
        return pd.concat([base, features], axis=1)


class SensorInteractionFeatures(BaseEstimator, TransformerMixin):
    """Products and differences of smoothed, normalised sensor pairs.

    Pairs are chosen from engine physics (e.g. T30 x T50 = hot-section thermal
    load) rather than by brute force, keeping the feature space interpretable.
    Smoothing uses the ``ewm{span}`` columns created by
    :class:`TemporalFeatureGenerator`.
    """

    def __init__(self, pairs: Sequence[Tuple[str, str]], span: int = 10):
        self.pairs = pairs
        self.span = span

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        new = {}
        for a, b in self.pairs:
            xa, xb = X[f"{a}_ewm{self.span}"], X[f"{b}_ewm{self.span}"]
            new[f"ix_{a}_x_{b}"] = xa * xb
            new[f"ix_{a}_minus_{b}"] = xa - xb
        return pd.concat([X, pd.DataFrame(new, index=X.index).astype(np.float32)], axis=1)


class HealthIndicatorTransformer(BaseEstimator, TransformerMixin):
    """Data-driven Health Indicator (HI) fused from smoothed sensors.

    A linear model maps the EWMA-smoothed sensors onto the normalised target
    ``HI = min(RUL, max_rul) / max_rul`` (1 = healthy, 0 = failure) - the
    classic "sensor fusion" health index of Wang et al. (2008). It gives tree
    models a single, monotonic degradation axis. Being supervised, it is
    fitted inside the CV loop only on training folds.

    **Row-wise by design.** Everything downstream of the unsupervised stage
    must give the same answer whether it receives a full trajectory or only
    the latest row of each engine (as at inference time). Because the HI is
    linear in the smoothed sensors, its temporal derivatives are obtained
    exactly as the same linear combination of the sensor-level features
    already computed causally upstream::

        HI_ewm{k}   = w . s_ewm{k}       HI_slope{k} = w . s_slope{k}
        HI_diff1    = w . s_diff1

    No grouping over the rows received is ever performed here.
    """

    def __init__(self, sensors: Sequence[str], span: int = 10, max_rul: float = 125.0,
                 ewm_spans: Sequence[int] = (5, 20), slope_windows: Sequence[int] = (10, 30)):
        self.sensors = sensors
        self.span = span
        self.max_rul = max_rul
        self.ewm_spans = ewm_spans
        self.slope_windows = slope_windows

    def _combine(self, X: pd.DataFrame, suffix: str, intercept: bool) -> np.ndarray:
        values = X[[f"{s}_{suffix}" for s in self.sensors]].to_numpy(dtype=float)
        out = values @ self.model_.coef_
        return out + self.model_.intercept_ if intercept else out

    def fit(self, X: pd.DataFrame, y=None):
        if y is None:
            raise ValueError("HealthIndicatorTransformer requires the RUL target in fit().")
        target = np.clip(np.asarray(y, dtype=float), 0, self.max_rul) / self.max_rul
        inputs = X[[f"{s}_ewm{self.span}" for s in self.sensors]].to_numpy(dtype=float)
        self.model_ = LinearRegression().fit(inputs, target)
        self.weights_ = pd.Series(self.model_.coef_, index=list(self.sensors))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "model_")
        new = {"health_indicator": np.clip(self._combine(X, f"ewm{self.span}", True),
                                           -0.5, 1.5)}
        for span in self.ewm_spans:
            if span != self.span:
                new[f"health_indicator_ewm{span}"] = self._combine(X, f"ewm{span}", True)
        for window in self.slope_windows:
            new[f"health_indicator_slope{window}"] = self._combine(X, f"slope{window}", False)
        new["health_indicator_diff1"] = self._combine(X, "diff1", False)
        return pd.concat([X, pd.DataFrame(new, index=X.index).astype(np.float32)], axis=1)


class DropColumns(BaseEstimator, TransformerMixin):
    """Drop identifier columns right before an estimator."""

    def __init__(self, columns: Sequence[str] = ("unit",)):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_out_ = [c for c in X.columns if c not in set(self.columns)]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=[c for c in self.columns if c in X.columns])

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)


def build_feature_pipeline(cfg: dict, n_conditions: int, random_state: Optional[int] = None):
    """Unsupervised feature stage: regime normalisation -> temporal -> interactions."""
    from sklearn.pipeline import Pipeline

    fcfg, sensors = cfg["features"], cfg["data"]["sensors"]
    seed = cfg["project"]["random_seed"] if random_state is None else random_state
    return Pipeline([
        ("condition_normalizer", OperatingConditionNormalizer(
            sensors=sensors, n_conditions=n_conditions,
            op_cols=cfg["data"]["operating_settings"], random_state=seed)),
        ("temporal_features", TemporalFeatureGenerator(
            sensors=sensors, windows=fcfg["rolling_windows"], stats=fcfg["rolling_stats"],
            ewm_spans=fcfg["ewm_spans"], lags=fcfg["lags"],
            diff_smoothing_span=fcfg["diff_smoothing_span"],
            slope_windows=fcfg["slope_windows"])),
        ("interactions", SensorInteractionFeatures(
            pairs=[tuple(p) for p in fcfg["interaction_pairs"]],
            span=fcfg["interaction_span"])),
    ])


_STAT_LABELS = {"mean": "rolling mean", "std": "rolling std", "min": "rolling min",
                "max": "rolling max", "skew": "rolling skewness", "kurt": "rolling kurtosis"}


def describe_feature(name: str) -> str:
    """Human-readable label of an engineered feature name.

    >>> describe_feature("s11_roll30_mean")
    'Ps30 (HPC static pressure) - 30-cycle rolling mean'
    """
    import re

    from src.data.load_data import SENSOR_DESCRIPTIONS

    def sensor(code: str) -> str:
        desc = SENSOR_DESCRIPTIONS.get(code, code)
        short, _, long = desc.partition(" - ")
        long = (long.replace("Total temperature at ", "temp. ")
                .replace("Total pressure at ", "pressure ")
                .replace("Static pressure at ", "static pressure ")
                .split(" (")[0])
        return f"{short} ({long})"

    if name == "cycle":
        return "Engine age (cycles flown)"
    if name.startswith("health_indicator"):
        rest = name.replace("health_indicator", "").strip("_")
        if not rest:
            return "Health indicator (fused sensors)"
        if rest.startswith("slope"):
            return f"Health indicator - {rest[5:]}-cycle degradation rate"
        if rest.startswith("ewm"):
            return f"Health indicator - EWMA span {rest[3:]}"
        return f"Health indicator - {rest}"
    if name.startswith("op_condition_"):
        return f"Operating regime {name.rsplit('_', 1)[1]} (current)"
    if name.startswith("hist_share_op_condition_"):
        return f"Share of life flown in regime {name.rsplit('_', 1)[1]}"
    m = re.match(r"ix_(s\d+)_(x|minus)_(s\d+)$", name)
    if m:
        op = "x" if m.group(2) == "x" else "-"
        return f"{sensor(m.group(1)).split(' (')[0]} {op} {sensor(m.group(3)).split(' (')[0]}"\
               " (smoothed interaction)"
    m = re.match(r"(s\d+)_(.+)$", name)
    if not m:
        return name
    base, kind = sensor(m.group(1)), m.group(2)
    patterns = [
        (r"roll(\d+)_(\w+)", lambda g: f"{g[0]}-cycle {_STAT_LABELS.get(g[1], g[1])}"),
        (r"ewm(\d+)", lambda g: f"EWMA (span {g[0]})"),
        (r"lag(\d+)", lambda g: f"value {g[0]} cycles ago"),
        (r"diff1", lambda g: "1st difference (trend)"),
        (r"diff2", lambda g: "2nd difference (acceleration)"),
        (r"slope(\d+)", lambda g: f"{g[0]}-cycle degradation rate"),
    ]
    for pattern, fmt in patterns:
        mm = re.fullmatch(pattern, kind)
        if mm:
            return f"{base} - {fmt(mm.groups())}"
    return f"{base} - {kind}"
