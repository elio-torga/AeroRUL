"""Model factories, hyper-parameter search, calibration and uncertainty.

Architecture of a fitted AeroRUL model::

    raw sensors
      └─ feature pipeline (unsupervised, fitted once)       src.features.engineering
           OperatingConditionNormalizer -> TemporalFeatureGenerator
           -> SensorInteractionFeatures
      └─ model pipeline (supervised, refitted in every CV fold)
           HealthIndicatorTransformer -> FeatureSelector -> regressor
                                                     (RF | HGB | Stacking ...)
      └─ ConformalQuantileRegressor (80 % interval)  +  optional isotonic map

Splitting the unsupervised stage out is a deliberate trade-off: rolling
features are deterministic and regime normalisation is label-free, so they
can be computed once instead of in every fold of every search iteration,
while everything that sees the target (health indicator, feature
selection, model) is fitted strictly inside the cross-validation loop.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.preprocess import EngineGroupKFold
from src.features.engineering import DropColumns, HealthIndicatorTransformer
from src.features.selection import FeatureSelector
from src.utils import get_logger
from src.utils.metrics import nasa_scorer, rmse

logger = get_logger(__name__)

MODEL_LABELS = {
    "ridge": "Ridge (linear baseline)",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "hgb_default": "HistGradientBoosting (default)",
    "hgb_tuned": "HistGradientBoosting (tuned)",
    "stacking": "Stacking ensemble",
}


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #
def make_regressor(name: str, cfg: dict, seed: int,
                   params: Optional[dict] = None) -> Pipeline:
    """Return ``DropColumns(unit) -> estimator`` for a named model family."""
    mcfg = cfg["models"]
    params = params or {}
    if name == "ridge":
        steps = [("scaler", StandardScaler()),
                 ("reg", Ridge(alpha=mcfg["ridge"]["alpha"]))]
    elif name == "random_forest":
        steps = [("reg", RandomForestRegressor(**mcfg["random_forest"], n_jobs=-1,
                                               random_state=seed))]
    elif name == "extra_trees":
        steps = [("reg", ExtraTreesRegressor(**mcfg["extra_trees"], n_jobs=-1,
                                             random_state=seed))]
    elif name in {"hgb", "hgb_default", "hgb_tuned"}:
        hp = {**mcfg["hist_gradient_boosting"], **params}
        steps = [("reg", HistGradientBoostingRegressor(**hp, early_stopping=False,
                                                       random_state=seed))]
    else:
        raise ValueError(f"Unknown model family {name!r}")
    return Pipeline([("drop_ids", DropColumns(["unit"]))] + steps)


def make_stacking(cfg: dict, seed: int, hgb_params: Optional[dict] = None) -> StackingRegressor:
    """Stacking of diverse learners with a non-negative Ridge meta-learner.

    Out-of-fold base predictions are produced with :class:`EngineGroupKFold`,
    so the meta-learner never sees predictions made on engines a base model
    was trained on. Non-negative weights keep the ensemble interpretable and
    allow exact additive SHAP decomposition (see ``src.models.predict``).
    """
    scfg = cfg["models"]["stacking"]
    estimators = [
        ("ridge", make_regressor("ridge", cfg, seed)),
        ("random_forest", make_regressor("random_forest", cfg, seed)),
        ("extra_trees", make_regressor("extra_trees", cfg, seed)),
        ("hgb", make_regressor("hgb", cfg, seed, hgb_params)),
    ]
    return StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=scfg["final_alpha"], positive=True),
        cv=EngineGroupKFold(n_splits=scfg["cv_folds"], random_state=seed),
        n_jobs=1,
    )


def build_model_pipeline(cfg: dict, estimator, seed: int, memory=None) -> Pipeline:
    """Supervised stage: health indicator -> feature selection -> estimator."""
    scfg = cfg["selection"]
    return Pipeline([
        ("health_indicator", HealthIndicatorTransformer(
            sensors=cfg["data"]["sensors"], span=cfg["features"]["health_indicator_span"],
            max_rul=cfg["target"]["max_rul"], ewm_spans=cfg["features"]["ewm_spans"],
            slope_windows=cfg["features"]["slope_windows"])),
        ("selector", FeatureSelector(
            top_k=scfg["top_k"], correlation_threshold=scfg["correlation_threshold"],
            variance_threshold=scfg["variance_threshold"],
            sample_size=scfg["importance_sample_size"],
            n_estimators=scfg["importance_n_estimators"], random_state=seed)),
        ("model", estimator),
    ], memory=memory)


# --------------------------------------------------------------------------- #
# Hyper-parameter optimisation
# --------------------------------------------------------------------------- #
def _distribution(spec: dict):
    kind = spec["dist"]
    if kind == "loguniform":
        return stats.loguniform(spec["low"], spec["high"])
    if kind == "uniform":
        return stats.uniform(spec["low"], spec["high"] - spec["low"])
    if kind == "randint":
        return stats.randint(spec["low"], spec["high"])
    if kind == "choice":
        return list(spec["values"])
    raise ValueError(f"Unknown distribution {kind!r}")


def tune_hgb(X: pd.DataFrame, y: np.ndarray, cfg: dict, seed: int,
             memory=None) -> Tuple[dict, pd.DataFrame]:
    """Randomised search of HistGradientBoosting hyper-parameters.

    Engine-grouped K-fold CV; the (supervised) health indicator and feature
    selector are refitted in every fold. ``Pipeline(memory=...)`` caches those
    fitted transformers so they are computed once per fold, not once per
    candidate.
    """
    tcfg = cfg["tuning"]
    pipe = build_model_pipeline(cfg, make_regressor("hgb", cfg, seed), seed, memory)
    space = {f"model__reg__{k}": _distribution(v)
             for k, v in tcfg["param_distributions"].items()}
    search = RandomizedSearchCV(
        pipe, space, n_iter=tcfg["n_iter"],
        cv=EngineGroupKFold(n_splits=tcfg["cv_folds"], random_state=seed),
        scoring={"rmse": tcfg["scoring"], "nasa": nasa_scorer},
        refit=False, random_state=seed, n_jobs=1, return_train_score=False,
    )
    search.fit(X, y)
    results = pd.DataFrame(search.cv_results_)
    best = results.loc[results["rank_test_rmse"].idxmin()]
    best_params = {k.replace("model__reg__", ""): v for k, v in best["params"].items()}
    for key in ("max_iter", "max_leaf_nodes", "min_samples_leaf"):
        best_params[key] = int(best_params[key])
    logger.info("Best CV RMSE %.2f with %s", -best["mean_test_rmse"], best_params)
    tidy = results[[c for c in results.columns if c.startswith("param_")]
                   + ["mean_test_rmse", "std_test_rmse", "mean_test_nasa", "mean_fit_time"]]
    tidy = tidy.rename(columns=lambda c: c.replace("param_model__reg__", ""))
    tidy["cv_rmse"] = -tidy.pop("mean_test_rmse")
    tidy["cv_nasa_score"] = -tidy.pop("mean_test_nasa")
    return best_params, tidy.sort_values("cv_rmse").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Uncertainty: Conformalized Quantile Regression
# --------------------------------------------------------------------------- #
class ConformalQuantileRegressor(BaseEstimator, RegressorMixin):
    """Conformalized Quantile Regression (Romano et al., 2019).

    Two quantile gradient-boosting models estimate the ``alpha/2`` and
    ``1 - alpha/2`` conditional quantiles. On a held-out calibration set the
    conformity score ``E = max(q_lo - y, y - q_hi)`` is computed and the
    interval is widened (or shrunk) by its ``(1 - alpha)(1 + 1/n)`` empirical
    quantile, which yields finite-sample marginal coverage under
    exchangeability, regardless of how well the quantile models are specified.
    """

    def __init__(self, alpha: float = 0.2, max_rul: float = 125.0,
                 hgb_params: Optional[dict] = None, max_iter: int = 300,
                 random_state: int = 42):
        self.alpha = alpha
        self.max_rul = max_rul
        self.hgb_params = hgb_params
        self.max_iter = max_iter
        self.random_state = random_state

    def _make(self, quantile: float) -> Pipeline:
        params = dict(self.hgb_params or {})
        params.update(loss="quantile", quantile=quantile, max_iter=self.max_iter,
                      early_stopping=False, random_state=self.random_state)
        return Pipeline([("drop_ids", DropColumns(["unit"])),
                         ("reg", HistGradientBoostingRegressor(**params))])

    def fit(self, X: pd.DataFrame, y) -> "ConformalQuantileRegressor":
        self.lower_model_ = self._make(self.alpha / 2).fit(X, y)
        self.upper_model_ = self._make(1 - self.alpha / 2).fit(X, y)
        self.correction_ = 0.0
        return self

    def _raw(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        lo, hi = self.lower_model_.predict(X), self.upper_model_.predict(X)
        return np.minimum(lo, hi), np.maximum(lo, hi)

    def calibrate(self, X: pd.DataFrame, y) -> "ConformalQuantileRegressor":
        lo, hi = self._raw(X)
        y = np.asarray(y, dtype=float)
        scores = np.maximum(lo - y, y - hi)
        n = len(scores)
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.correction_ = float(np.quantile(scores, level, method="higher"))
        self.raw_coverage_ = float(np.mean((y >= lo) & (y <= hi)))
        logger.info("CQR: raw coverage %.3f -> correction %+.2f cycles",
                    self.raw_coverage_, self.correction_)
        return self

    def predict_interval(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        lo, hi = self._raw(X)
        return (np.clip(lo - self.correction_, 0, self.max_rul),
                np.clip(hi + self.correction_, 0, self.max_rul))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        lo, hi = self.predict_interval(X)
        return (lo + hi) / 2


# --------------------------------------------------------------------------- #
# Point-prediction calibration
# --------------------------------------------------------------------------- #
class IsotonicRULCalibrator(BaseEstimator):
    """Monotone recalibration ``f(pred) -> E[y | pred]``, adopted only if useful.

    Tree ensembles shrink extreme predictions toward the mean (they cannot
    extrapolate), which biases low RUL estimates upwards - exactly where late
    predictions are most expensive. Isotonic regression fitted on validation
    engines corrects this bias without changing the ranking of engines. The
    calibrator is *enabled* only if engine-grouped cross-validation on the
    calibration set shows an RMSE gain larger than ``min_gain``.
    """

    def __init__(self, min_gain: float = 0.01, n_splits: int = 5, random_state: int = 42):
        self.min_gain = min_gain
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, y_pred, y_true, units) -> "IsotonicRULCalibrator":
        y_pred, y_true = np.asarray(y_pred, float), np.asarray(y_true, float)
        oof = np.empty_like(y_pred)
        splitter = EngineGroupKFold(self.n_splits, random_state=self.random_state)
        for tr, te in splitter.split(None, groups=np.asarray(units)):
            iso = IsotonicRegression(out_of_bounds="clip").fit(y_pred[tr], y_true[tr])
            oof[te] = iso.predict(y_pred[te])
        self.rmse_raw_ = rmse(y_true, y_pred)
        self.rmse_calibrated_cv_ = rmse(y_true, oof)
        gain = 1 - self.rmse_calibrated_cv_ / self.rmse_raw_
        self.enabled_ = bool(gain > self.min_gain)
        self.iso_ = IsotonicRegression(out_of_bounds="clip").fit(y_pred, y_true)
        logger.info("Isotonic calibration: RMSE %.2f -> %.2f (CV) -> %s",
                    self.rmse_raw_, self.rmse_calibrated_cv_,
                    "enabled" if self.enabled_ else "disabled")
        return self

    def transform(self, y_pred) -> np.ndarray:
        y_pred = np.asarray(y_pred, float)
        return self.iso_.predict(y_pred) if self.enabled_ else y_pred


# --------------------------------------------------------------------------- #
# Candidate training
# --------------------------------------------------------------------------- #
def fit_candidates(X: pd.DataFrame, y: np.ndarray, cfg: dict, seed: int,
                   hgb_params: Optional[dict], memory=None,
                   names=("ridge", "random_forest", "extra_trees", "hgb_default",
                          "hgb_tuned", "stacking")) -> Dict[str, Pipeline]:
    """Fit every candidate model pipeline on the training engines."""
    fitted = {}
    for name in names:
        if name == "stacking":
            estimator = make_stacking(cfg, seed, hgb_params)
        elif name == "hgb_tuned":
            estimator = make_regressor("hgb", cfg, seed, hgb_params)
        else:
            estimator = make_regressor(name, cfg, seed)
        pipe = build_model_pipeline(cfg, estimator, seed, memory)
        logger.info("Fitting %s on %d rows", MODEL_LABELS[name], len(X))
        fitted[name] = pipe.fit(X, y)
    return fitted


def strip_memory(pipe: Pipeline) -> Pipeline:
    """Detach the joblib cache before serialisation."""
    pipe.memory = None
    return pipe


__all__ = [
    "MODEL_LABELS",
    "ConformalQuantileRegressor",
    "IsotonicRULCalibrator",
    "build_model_pipeline",
    "clone",
    "fit_candidates",
    "make_regressor",
    "make_stacking",
    "strip_memory",
    "tune_hgb",
]
