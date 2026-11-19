"""Model evaluation and maintenance-policy economics.

Two complementary views:

* **Statistical** - RMSE / MAE / R2, NASA asymmetric score, PHM accuracy
  window, C-index, alert precision/recall, interval coverage.
* **Operational** - what would the fleet cost under different maintenance
  policies driven by the model? This is the number a maintenance planner or
  an airline CFO cares about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from src.utils.metrics import regression_report


def evaluate_predictions(y_true, y_pred, cfg: dict, lower=None, upper=None) -> Dict[str, float]:
    """Metric dictionary using the tolerances defined in ``config.yaml``."""
    ecfg = cfg["evaluation"]
    return regression_report(y_true, y_pred, lower, upper,
                             early=ecfg["phm_early_tolerance"],
                             late=ecfg["phm_late_tolerance"],
                             horizon=ecfg["alert_horizon"])


def evaluate_models(models: Dict[str, object], X: pd.DataFrame, y, cfg: dict,
                    dataset: str, split: str) -> pd.DataFrame:
    """Evaluate several fitted pipelines on the same rows."""
    rows = []
    for name, model in models.items():
        pred = np.clip(model.predict(X), 0, cfg["target"]["max_rul"])
        rows.append({"dataset": dataset, "split": split, "model": name,
                     **evaluate_predictions(y, pred, cfg)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Maintenance policy simulation
# --------------------------------------------------------------------------- #
@dataclass
class CostModel:
    """Per-engine economics of a removal decision (illustrative defaults)."""

    unplanned_failure_cost: float
    planned_maintenance_cost: float
    cost_per_lost_cycle: float
    maintenance_lead_time: int

    @classmethod
    def from_config(cls, cfg: dict) -> "CostModel":
        b = cfg["business"]
        return cls(b["unplanned_failure_cost"], b["planned_maintenance_cost"],
                   b["cost_per_lost_cycle"], b["maintenance_lead_time"])

    def removal_cost(self, rul_at_removal: np.ndarray) -> np.ndarray:
        """Cost when the engine is removed with ``rul_at_removal`` cycles left.

        A non-positive remaining life means the engine failed in service.
        """
        r = np.asarray(rul_at_removal, dtype=float)
        planned = self.planned_maintenance_cost + self.cost_per_lost_cycle * np.maximum(r, 0)
        return np.where(r <= 0, self.unplanned_failure_cost, planned)


def _first_trigger(traj: pd.DataFrame, signal: str, threshold: float) -> pd.Series:
    """True RUL at the first cycle where ``signal <= threshold`` (NaN if never)."""
    hit = traj[traj[signal] <= threshold]
    first = hit.loc[hit.groupby("unit")["cycle"].idxmin()].set_index("unit")["rul_true"]
    return first.reindex(traj["unit"].unique())


def simulate_predictive_policy(traj: pd.DataFrame, signal: str, threshold: float,
                               costs: CostModel) -> pd.DataFrame:
    """Condition-based maintenance driven by a model signal.

    A removal is *ordered* at the first cycle where ``signal <= threshold``;
    it physically happens ``maintenance_lead_time`` cycles later. If the
    engine fails during the lead time (or the alarm never fires) the event is
    an unplanned failure.

    ``traj`` must hold complete run-to-failure trajectories with columns
    ``unit, cycle, rul_true`` and the signal column.
    """
    traj = traj.reset_index(drop=True)  # robust to concatenated frames
    at_trigger = _first_trigger(traj, signal, threshold)
    rul_at_removal = (at_trigger - costs.maintenance_lead_time).fillna(-1).to_numpy()
    return pd.DataFrame({"unit": at_trigger.index, "rul_at_removal": rul_at_removal,
                         "failure": rul_at_removal <= 0,
                         "cost": costs.removal_cost(rul_at_removal)})


def simulate_fixed_interval(lifetimes: np.ndarray, interval: float,
                            costs: CostModel) -> pd.DataFrame:
    """Time-based preventive maintenance: remove every engine at ``interval``."""
    lifetimes = np.asarray(lifetimes, dtype=float)
    rul_at_removal = lifetimes - interval
    return pd.DataFrame({"rul_at_removal": rul_at_removal, "failure": rul_at_removal <= 0,
                         "cost": costs.removal_cost(rul_at_removal)})


def optimal_fixed_interval(train_lifetimes: np.ndarray, costs: CostModel) -> float:
    """Best fixed removal age on *training* engines (no peeking at evaluation data)."""
    grid = np.arange(50, int(np.max(train_lifetimes)) + 1)
    expected = [simulate_fixed_interval(train_lifetimes, t, costs)["cost"].mean() for t in grid]
    return float(grid[int(np.argmin(expected))])


def compare_policies(traj: pd.DataFrame, train_lifetimes: np.ndarray, cfg: dict,
                     point_col: str = "rul_pred", lower_col: str = "rul_lower") -> pd.DataFrame:
    """Expected cost per engine of five maintenance strategies.

    1. *Reactive* - run to failure.
    2. *Fixed interval* - removal age optimised on the training fleet.
    3. *Predictive (point)* - alarm when predicted RUL <= lead time + margin.
    4. *Predictive (risk-aware)* - alarm when the lower 80 % bound <= lead time + margin.
    5. *Oracle* - perfect knowledge (lower bound on achievable cost).
    """
    costs = CostModel.from_config(cfg)
    threshold = costs.maintenance_lead_time + cfg["business"]["safety_margin"]
    lifetimes = traj.groupby("unit")["rul_true"].max().to_numpy() + 1
    n = len(lifetimes)
    interval = optimal_fixed_interval(train_lifetimes, costs)
    results = {
        "Reactive (run-to-failure)": pd.DataFrame(
            {"failure": np.ones(n, bool), "cost": costs.unplanned_failure_cost,
             "rul_at_removal": np.zeros(n)}),
        f"Fixed interval ({interval:.0f} cycles)": simulate_fixed_interval(
            lifetimes, interval, costs),
        "Predictive - point estimate": simulate_predictive_policy(
            traj, point_col, threshold, costs),
        "Predictive - risk-aware (lower bound)": simulate_predictive_policy(
            traj, lower_col, threshold, costs),
        "Oracle (perfect RUL)": pd.DataFrame(
            {"failure": np.zeros(n, bool), "cost": costs.planned_maintenance_cost,
             "rul_at_removal": np.zeros(n)}),
    }
    rows = []
    for name, res in results.items():
        rows.append({"policy": name, "engines": n,
                     "unplanned_failures": int(res["failure"].sum()),
                     "failure_rate": float(res["failure"].mean()),
                     "mean_wasted_cycles": float(np.maximum(res["rul_at_removal"], 0).mean()),
                     "cost_per_engine": float(res["cost"].mean()),
                     "total_cost": float(res["cost"].sum())})
    table = pd.DataFrame(rows)
    reactive = table.loc[0, "cost_per_engine"]
    table["savings_vs_reactive"] = 1 - table["cost_per_engine"] / reactive
    return table


def threshold_sweep(traj: pd.DataFrame, signal: str, cfg: dict,
                    thresholds: Optional[Iterable[float]] = None) -> pd.DataFrame:
    """Cost / failure-rate trade-off as the alarm threshold varies."""
    costs = CostModel.from_config(cfg)
    thresholds = thresholds if thresholds is not None else np.arange(5, 81, 1)
    rows = []
    for th in thresholds:
        res = simulate_predictive_policy(traj, signal, th, costs)
        rows.append({"threshold": th, "signal": signal,
                     "cost_per_engine": res["cost"].mean(),
                     "failure_rate": res["failure"].mean(),
                     "mean_wasted_cycles": np.maximum(res["rul_at_removal"], 0).mean()})
    return pd.DataFrame(rows)
