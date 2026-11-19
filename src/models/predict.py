"""Inference service: the serialisable model bundle and its explainer.

A single ``RULPredictor`` object (persisted with joblib) contains everything
needed in production: the fitted feature pipeline, the point model, the
conformal interval model, the optional isotonic calibrator and metadata. It
consumes *raw* sensor trajectories, exactly as an engine-health-monitoring
feed would deliver them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.utils import get_logger

logger = get_logger(__name__)


class StackedShapExplainer:
    """Exact additive SHAP explanations for (stacked) tree ensembles.

    SHAP values are linear in the model: if ``f = b + sum_i w_i f_i`` (a
    stacking ensemble with a linear meta-learner), then
    ``phi(f) = sum_i w_i phi(f_i)`` and the expected value is
    ``b + sum_i w_i E[f_i]``. Each base learner is explained exactly -
    TreeSHAP for tree ensembles, closed-form linear SHAP for Ridge - and the
    contributions are combined with the meta-learner weights. Local accuracy
    (prediction = base value + sum of contributions) holds exactly.
    """

    def __init__(self, estimator, background: pd.DataFrame):
        import shap  # local import keeps the dependency optional for inference

        if isinstance(estimator, StackingRegressor):
            self.names = list(estimator.named_estimators_.keys())
            self.bases: List[Pipeline] = list(estimator.estimators_)
            self.weights = np.asarray(estimator.final_estimator_.coef_, dtype=float)
            self.intercept = float(estimator.final_estimator_.intercept_)
        else:
            self.names, self.bases = ["model"], [estimator]
            self.weights, self.intercept = np.array([1.0]), 0.0
        self._explainers = []
        for base in self.bases:
            reg = base[-1]
            bg = base[:-1].transform(background)
            if isinstance(reg, Ridge):
                mean = np.asarray(bg, dtype=float).mean(axis=0)
                self._explainers.append(("linear", reg, mean,
                                         float(reg.predict(mean[None, :])[0])))
            else:
                tree = shap.TreeExplainer(reg)
                self._explainers.append(("tree", tree, None,
                                         float(np.ravel(tree.expected_value)[0])))
        self.feature_names = list(self.bases[0][0].transform(background.iloc[:1]).columns)
        self.expected_value = self.intercept + float(
            np.dot(self.weights, [e[3] for e in self._explainers]))

    def shap_values(self, X: pd.DataFrame, per_model: bool = False):
        """Return an ``(n_rows, n_features)`` array of SHAP values."""
        total = np.zeros((len(X), len(self.feature_names)))
        parts: Dict[str, np.ndarray] = {}
        for name, base, weight, (kind, expl, mean, _) in zip(
                self.names, self.bases, self.weights, self._explainers):
            if weight == 0:
                continue
            Xb = np.asarray(base[:-1].transform(X), dtype=float)
            if kind == "linear":
                values = expl.coef_ * (Xb - mean)
            else:
                values = expl.shap_values(Xb, check_additivity=False)
            parts[name] = weight * values
            total += parts[name]
        return (total, parts) if per_model else total

    def explanation(self, X: pd.DataFrame):
        """``shap.Explanation`` object, ready for ``shap.plots.*``."""
        import shap

        values = self.shap_values(X)
        data = self.bases[0][0].transform(X)  # unscaled values for display
        return shap.Explanation(values=values,
                                base_values=np.full(len(X), self.expected_value),
                                data=np.asarray(data, dtype=float),
                                feature_names=self.feature_names)


class RULPredictor:
    """Production bundle: raw sensor trajectories in, RUL + interval + risk out."""

    def __init__(self, dataset: str, feature_pipeline: Pipeline, model_pipeline: Pipeline,
                 interval_model, calibrator, max_rul: float, risk_thresholds: dict,
                 model_name: str, metadata: Optional[dict] = None):
        self.dataset = dataset
        self.feature_pipeline = feature_pipeline
        self.model_pipeline = model_pipeline
        self.interval_model = interval_model
        self.calibrator = calibrator
        self.max_rul = max_rul
        self.risk_thresholds = risk_thresholds
        self.model_name = model_name
        self.metadata = metadata or {}
        self._explainer: Optional[StackedShapExplainer] = None

    # ------------------------------------------------------------------ core
    def compute_features(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Raw C-MAPSS columns -> engineered feature frame."""
        raw = raw.sort_values(["unit", "cycle"]).reset_index(drop=True)
        return self.feature_pipeline.transform(raw)

    def model_inputs(self, features: pd.DataFrame) -> pd.DataFrame:
        """Feature frame -> selected model inputs (health indicator included)."""
        return self.model_pipeline[:-1].transform(features)

    def predict_from_features(self, features: pd.DataFrame) -> pd.DataFrame:
        Z = self.model_inputs(features)
        raw_point = self.model_pipeline[-1].predict(Z)
        point = np.clip(self.calibrator.transform(raw_point), 0, self.max_rul)
        lower, upper = self.interval_model.predict_interval(Z)
        out = features[["unit", "cycle"]].copy()
        out["rul_pred"] = point
        out["rul_lower"] = np.minimum(lower, point)
        out["rul_upper"] = np.maximum(upper, point)
        out["health_indicator"] = Z["health_indicator"].to_numpy() if \
            "health_indicator" in Z else np.nan
        out["risk_level"] = self.risk_level(out["rul_lower"].to_numpy())
        return out

    def predict(self, raw: pd.DataFrame, last_cycle_only: bool = True) -> pd.DataFrame:
        """Predict RUL for every engine (default: at its latest cycle)."""
        preds = self.predict_from_features(self.compute_features(raw))
        if last_cycle_only:
            preds = preds.loc[preds.groupby("unit")["cycle"].idxmax()]
        return preds.reset_index(drop=True)

    def risk_level(self, rul_lower: np.ndarray) -> np.ndarray:
        """Risk band driven by the *lower* interval bound (risk-aware policy)."""
        critical, warning = self.risk_thresholds["critical"], self.risk_thresholds["warning"]
        return np.select([rul_lower <= critical, rul_lower <= warning],
                         ["CRITICAL", "WARNING"], default="HEALTHY")

    # -------------------------------------------------------- explainability
    def explainer(self, background_features: Optional[pd.DataFrame] = None
                  ) -> StackedShapExplainer:
        """Build (and cache) the SHAP explainer of the point model."""
        if self._explainer is None or background_features is not None:
            if background_features is None:
                background_features = self.metadata["background_features"]
            Z = self.model_inputs(background_features)
            self._explainer = StackedShapExplainer(self.model_pipeline[-1], Z)
        return self._explainer

    def explain(self, features: pd.DataFrame):
        """SHAP explanation of the (uncalibrated) point model for given rows."""
        return self.explainer().explanation(self.model_inputs(features))

    # --------------------------------------------------------- persistence
    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._explainer = None
        joblib.dump(self, path, compress=3)
        logger.info("Saved model bundle to %s (%.1f MB)", path, path.stat().st_size / 1e6)
        return path

    @staticmethod
    def load(path: Union[str, Path]) -> "RULPredictor":
        return joblib.load(path)


def load_predictor(dataset: str, models_dir: Union[str, Path] = "models") -> RULPredictor:
    """Load the persisted bundle of one sub-dataset."""
    return RULPredictor.load(Path(models_dir) / f"aerorul_{dataset}.joblib")
