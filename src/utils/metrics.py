"""Prognostics metrics.

Conventions: ``error = y_pred - y_true``. A **positive** error is a *late*
prediction (the model believes the engine has more life than it has), which
is the dangerous direction in maintenance and is penalised harder by the
NASA scoring function.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def nasa_score(y_true, y_pred) -> float:
    """Official PHM08 / C-MAPSS asymmetric scoring function (lower is better).

    ``s = sum(exp(-d/13) - 1)`` for early predictions (d < 0) and
    ``s = sum(exp(d/10) - 1)`` for late predictions (d >= 0), with
    ``d = predicted - true`` (Saxena et al., 2008).
    """
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def phm_accuracy(y_true, y_pred, early: float = 13.0, late: float = 10.0) -> float:
    """Share of predictions inside the PHM08 tolerance window [-early, +late]."""
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.mean((d >= -early) & (d <= late)))


def late_prediction_rate(y_true, y_pred, tolerance: float = 10.0) -> float:
    """Share of predictions that over-estimate RUL by more than ``tolerance``."""
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.mean(d > tolerance))


def concordance_index(y_true, y_pred) -> float:
    """Harrell's C-index between true and predicted time-to-failure.

    Survival-analysis view of ranking quality: the probability that, for two
    random engines, the one failing first also has the lower predicted RUL.
    All test engines are uncensored (true RUL known), so every comparable pair
    with distinct true RUL counts; prediction ties count for one half.
    """
    t = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    dt = t[:, None] - t[None, :]
    dp = p[:, None] - p[None, :]
    comparable = dt < 0
    concordant = (dp < 0) & comparable
    ties = (dp == 0) & comparable
    n = comparable.sum()
    return float((concordant.sum() + 0.5 * ties.sum()) / n) if n else float("nan")


def alert_metrics(y_true, y_pred, horizon: float = 30.0) -> Dict[str, float]:
    """Precision / recall of the binary alert "failure within ``horizon`` cycles"."""
    truth = np.asarray(y_true) <= horizon
    alert = np.asarray(y_pred) <= horizon
    tp = float(np.sum(truth & alert))
    precision = tp / alert.sum() if alert.sum() else float("nan")
    recall = tp / truth.sum() if truth.sum() else float("nan")
    return {"alert_precision": precision, "alert_recall": recall}


def interval_metrics(y_true, lower, upper) -> Dict[str, float]:
    """Empirical coverage and mean width of prediction intervals."""
    y = np.asarray(y_true, dtype=float)
    lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    return {"coverage": float(np.mean((y >= lo) & (y <= hi))),
            "mean_width": float(np.mean(hi - lo))}


def regression_report(y_true, y_pred, lower: Optional[np.ndarray] = None,
                      upper: Optional[np.ndarray] = None, early: float = 13.0,
                      late: float = 10.0, horizon: float = 30.0) -> Dict[str, float]:
    """Full metric set used throughout the project."""
    report = {
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "nasa_score": nasa_score(y_true, y_pred),
        "nasa_score_per_engine": nasa_score(y_true, y_pred) / len(np.asarray(y_true)),
        "phm_accuracy": phm_accuracy(y_true, y_pred, early, late),
        "late_rate": late_prediction_rate(y_true, y_pred, late),
        "c_index": concordance_index(y_true, y_pred),
    }
    report.update(alert_metrics(y_true, y_pred, horizon))
    if lower is not None and upper is not None:
        report.update(interval_metrics(y_true, lower, upper))
    return report


#: scikit-learn scorers (greater is better)
nasa_scorer = make_scorer(nasa_score, greater_is_better=False)
rmse_scorer = make_scorer(rmse, greater_is_better=False)
