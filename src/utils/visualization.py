"""Publication-quality plotting helpers with a single, consistent visual system.

Colour roles
------------
* categorical series use :data:`PALETTE` in a fixed order (never cycled);
* risk states use :data:`RISK_COLORS` and are always paired with a text label;
* text and axes stay neutral so the data carries the colour.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401  (PercentFormatter)
import numpy as np
import pandas as pd
import seaborn as sns

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
           "#6250d6", "#e34948"]
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8984"
GRID, SURFACE = "#e6e5e0", "#ffffff"
RISK_COLORS = {"HEALTHY": "#1baf7a", "WARNING": "#eda100", "CRITICAL": "#d03b3b"}
SEQUENTIAL = "Blues"
DIVERGING = sns.diverging_palette(250, 25, s=80, l=55, as_cmap=True)


def set_style() -> None:
    """Apply the project-wide matplotlib / seaborn style."""
    sns.set_theme(context="notebook", style="whitegrid", palette=PALETTE)
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 10, "axes.labelsize": 10.5, "axes.labelcolor": INK_2,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5, "grid.color": GRID, "grid.linewidth": 0.7,
        "legend.frameon": False, "legend.fontsize": 9.5, "lines.linewidth": 2.0,
        "font.family": "DejaVu Sans", "text.color": INK,
    })


def save_figure(fig: plt.Figure, name: str, figures_dir: Optional[Path]) -> Optional[Path]:
    """Save ``fig`` as PNG under ``figures_dir`` (no-op when dir is None)."""
    if figures_dir is None:
        return None
    path = Path(figures_dir) / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    return path


def _subtitle(ax: plt.Axes, text: str) -> None:
    ax.text(0, 1.01, text, transform=ax.transAxes, fontsize=9.5, color=INK_2, va="bottom")


# --------------------------------------------------------------------------- #
# Exploratory plots
# --------------------------------------------------------------------------- #
def plot_sensor_trends(df: pd.DataFrame, sensors: Sequence[str], descriptions: Dict[str, str],
                       units: Optional[Iterable[int]] = None, x: str = "rul_true",
                       ncols: int = 4) -> plt.Figure:
    """Sensor values versus remaining life for a handful of engines."""
    units = list(units) if units is not None else list(df["unit"].unique()[:8])
    nrows = int(np.ceil(len(sensors) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.6 * nrows), sharex=True)
    for ax, sensor in zip(axes.ravel(), sensors):
        for unit in units:
            sub = df[df["unit"] == unit]
            ax.plot(sub[x], sub[sensor], color=PALETTE[0], alpha=0.25, lw=1)
        agg = df.groupby(df[x].clip(upper=300))[sensor].mean()
        ax.plot(agg.index, agg.rolling(9, center=True, min_periods=1).mean(),
                color=PALETTE[1], lw=2)
        ax.set_title(descriptions.get(sensor, sensor).split(" - ")[0] + f"  ({sensor})",
                     fontsize=10)
        ax.invert_xaxis() if x == "rul_true" else None
    for ax in axes.ravel()[len(sensors):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Remaining useful life (cycles)" if x == "rul_true" else x)
    fig.suptitle("Sensor degradation signatures (blue: individual engines, orange: fleet mean)",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Prediction quality
# --------------------------------------------------------------------------- #
def plot_predictions_vs_truth(y_true, y_pred, lower=None, upper=None,
                              title: str = "") -> plt.Figure:
    """Scatter (left) and engines sorted by true RUL with 80 % interval (right)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.6]})
    lim = max(y_true.max(), y_pred.max()) * 1.05
    a1.fill_between([0, lim], [-13, lim - 13], [10, lim + 10], color=PALETTE[2], alpha=0.12,
                    lw=0, label="PHM window [-13, +10]")
    a1.plot([0, lim], [0, lim], color=INK_3, lw=1, ls="--")
    a1.scatter(y_true, y_pred, s=22, color=PALETTE[0], alpha=0.75, edgecolor="white", lw=0.5)
    a1.set(xlim=(0, lim), ylim=(0, lim), xlabel="True RUL (cycles)",
           ylabel="Predicted RUL (cycles)", title="Predicted vs. true")
    a1.legend(loc="upper left")
    order = np.argsort(y_true)
    idx = np.arange(len(order))
    if lower is not None and upper is not None:
        a2.fill_between(idx, np.asarray(lower)[order], np.asarray(upper)[order],
                        color=PALETTE[0], alpha=0.18, lw=0, label="80 % prediction interval")
    a2.plot(idx, y_true[order], color=INK, lw=2, label="True RUL")
    a2.scatter(idx, y_pred[order], s=14, color=PALETTE[1], zorder=3, label="Predicted RUL")
    a2.set(xlabel="Test engines (sorted by true RUL)", ylabel="RUL (cycles)",
           title="Per-engine predictions")
    a2.legend(loc="upper left")
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_error_distribution(errors_by_model: Dict[str, np.ndarray], title: str = "") -> plt.Figure:
    """Error histograms (top) above the asymmetric NASA penalty curve (bottom)."""
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                  gridspec_kw={"height_ratios": [2.2, 1]})
    bins = np.arange(-70, 71, 5)
    for i, (name, err) in enumerate(errors_by_model.items()):
        ax.hist(err, bins=bins, histtype="step", lw=2, color=PALETTE[i], label=name,
                density=True)
    ax.axvline(0, color=INK, lw=1)
    ax.set(ylabel="Density", title=title or "Error distribution and asymmetric cost")
    ax.legend(loc="upper left")
    d = np.linspace(-70, 70, 300)
    ax2.plot(d, np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1), color=INK_2, lw=1.8)
    ax2.set_yscale("log")
    ax2.set_ylim(1e-2, 2e3)
    ax2.axvline(0, color=INK, lw=1)
    ax2.text(-68, 300, "early: safe but wasteful", color=INK_2, fontsize=9)
    ax2.text(22, 300, "late: risk of in-service failure", color=INK_2, fontsize=9)
    ax2.set(xlabel="Prediction error = predicted - true (cycles)",
            ylabel="NASA penalty\nper engine (log)")
    fig.tight_layout()
    return fig


def plot_model_comparison(metrics: pd.DataFrame, metric: str = "rmse",
                          model_order: Optional[Sequence[str]] = None,
                          labels: Optional[Dict[str, str]] = None,
                          title: str = "") -> plt.Figure:
    """Grouped bars: one group per dataset, one bar per model."""
    model_order = list(model_order or metrics["model"].unique())
    datasets = sorted(metrics["dataset"].unique())
    fig, ax = plt.subplots(figsize=(11, 4.8))
    width = 0.8 / len(model_order)
    for i, model in enumerate(model_order):
        vals = [metrics.query("dataset == @d and model == @model")[metric].mean()
                for d in datasets]
        x = np.arange(len(datasets)) + (i - (len(model_order) - 1) / 2) * width
        ax.bar(x, vals, width * 0.9, color=PALETTE[i % len(PALETTE)],
               label=(labels or {}).get(model, model))
    ax.set_xticks(np.arange(len(datasets)), datasets)
    ax.set_ylabel(metric.upper().replace("_", " "))
    ax.set_title(title or f"Test {metric.upper()} by dataset and model")
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -0.1))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def plot_engine_trajectory(traj: pd.DataFrame, unit: int, max_rul: float = 125,
                           ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Predicted RUL over the whole life of one engine, with its interval."""
    sub = traj[traj["unit"] == unit].sort_values("cycle")
    ax = ax or plt.subplots(figsize=(7, 3.8))[1]
    ax.fill_between(sub["cycle"], sub["rul_lower"], sub["rul_upper"], color=PALETTE[0],
                    alpha=0.18, lw=0, label="80 % interval")
    ax.plot(sub["cycle"], sub["rul_true"].clip(upper=max_rul), color=INK, lw=2,
            label="True RUL (clipped)")
    ax.plot(sub["cycle"], sub["rul_pred"], color=PALETTE[1], lw=1.8, label="Predicted")
    ax.set(xlabel="Cycle", ylabel="RUL (cycles)", title=f"Engine {unit}")
    ax.set_ylim(bottom=0)
    return ax


def plot_reliability(y_true, y_pred, y_calibrated=None, n_bins: int = 10,
                     title: str = "Calibration of point predictions") -> plt.Figure:
    """Mean true RUL per bin of predicted RUL (perfect calibration = diagonal)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    fig, ax = plt.subplots(figsize=(5.6, 5))
    edges = np.quantile(y_pred, np.linspace(0, 1, n_bins + 1))
    series = [("Raw model", y_pred, PALETTE[0])]
    if y_calibrated is not None:
        series.append(("Isotonic-calibrated", np.asarray(y_calibrated, float), PALETTE[1]))
    for label, pred, color in series:
        b = np.clip(np.digitize(pred, edges[1:-1]), 0, n_bins - 1)
        frame = pd.DataFrame({"b": b, "p": pred, "t": y_true}).groupby("b").mean()
        ax.plot(frame["p"], frame["t"], marker="o", ms=6, color=color, label=label)
    lim = max(y_true.max(), y_pred.max())
    ax.plot([0, lim], [0, lim], ls="--", color=INK_3, lw=1)
    ax.set(xlabel="Mean predicted RUL (bin)", ylabel="Mean true RUL (bin)", title=title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_coverage_by_bin(y_true, lower, upper, nominal: float = 0.8,
                         title: str = "Interval coverage by true RUL") -> plt.Figure:
    """Empirical coverage of the prediction interval across life stages."""
    y_true = np.asarray(y_true, float)
    bins = pd.cut(y_true, [-1, 25, 50, 75, 100, 126],
                  labels=["0-25", "25-50", "50-75", "75-100", "100-125"])
    covered = (y_true >= np.asarray(lower)) & (y_true <= np.asarray(upper))
    frame = pd.DataFrame({"bin": bins, "covered": covered,
                          "width": np.asarray(upper) - np.asarray(lower)})
    agg = frame.groupby("bin", observed=True).agg(coverage=("covered", "mean"),
                                                  width=("width", "mean"),
                                                  n=("covered", "size"))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(agg.index.astype(str), agg["coverage"], color=PALETTE[0], width=0.6)
    ax.axhline(nominal, color=PALETTE[1], ls="--", lw=1.5, label=f"Nominal {nominal:.0%}")
    for i, (cov, w, n) in enumerate(agg[["coverage", "width", "n"]].to_numpy()):
        ax.text(i, cov - 0.03, f"{cov:.0%}\n±{w / 2:.0f} cyc\nn={n:.0f}", ha="center",
                va="top", fontsize=8.5, color="white", fontweight="bold")
    ax.set(ylim=(0, 1.15), xlabel="True RUL bin (cycles)", ylabel="Empirical coverage",
           title=title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Business plots
# --------------------------------------------------------------------------- #
def plot_policy_costs(table: pd.DataFrame, title: str = "") -> plt.Figure:
    """Horizontal bars of expected cost per engine for each maintenance policy."""
    t = table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    colors = [PALETTE[0] if "Predictive" in p else (INK_3 if "Oracle" in p else "#b9b8b2")
              for p in t["policy"]]
    ax.barh(t["policy"], t["cost_per_engine"] / 1e3, color=colors, height=0.6)
    for i, (cost, fail, sav) in enumerate(t[["cost_per_engine", "failure_rate",
                                             "savings_vs_reactive"]].to_numpy()):
        ax.text(cost / 1e3 + 15, i, f"${cost / 1e3:,.0f}k  |  failures {fail:.0%}  |  "
                f"saving {sav:+.0%}", va="center", fontsize=9, color=INK_2)
    ax.set_xlabel("Expected cost per engine (k$)")
    ax.set_xlim(0, t["cost_per_engine"].max() / 1e3 * 1.75)
    ax.set_title(title or "Maintenance strategy economics")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def plot_threshold_sweep(sweep: pd.DataFrame, chosen: Optional[float] = None,
                         title: str = "Alarm threshold trade-off") -> plt.Figure:
    """Cost per engine (left) and failure rate (right) as the threshold varies."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    for i, (sig, grp) in enumerate(sweep.groupby("signal", sort=False)):
        label = {"rul_pred": "Point estimate", "rul_lower": "Lower 80 % bound"}.get(sig, sig)
        a1.plot(grp["threshold"], grp["cost_per_engine"] / 1e3, color=PALETTE[i], label=label)
        a2.plot(grp["threshold"], grp["failure_rate"], color=PALETTE[i], label=label)
    for ax in (a1, a2):
        if chosen is not None:
            ax.axvline(chosen, color=INK_3, ls="--", lw=1)
        ax.set_xlabel("Alarm threshold (cycles)")
    a1.set(ylabel="Expected cost per engine (k$)", title="Cost")
    a2.set(ylabel="Unplanned failure rate", title="Safety")
    a2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    a1.legend()
    fig.suptitle(title, x=0.01, ha="left", fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Explainability plots
# --------------------------------------------------------------------------- #
def plot_shap_importance(shap_values: np.ndarray, feature_names: Sequence[str],
                         top_n: int = 15, title: str = "Global feature importance (mean |SHAP|)"
                         ) -> plt.Figure:
    """Mean absolute SHAP value per feature (top ``top_n``)."""
    imp = pd.Series(np.abs(shap_values).mean(0), index=feature_names).nlargest(top_n)[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.36 * top_n + 1.2))
    ax.barh(imp.index, imp.values, color=PALETTE[0], height=0.65)
    ax.set_xlabel("Mean |SHAP value| (cycles of RUL)")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def plot_shap_over_life(shap_values: np.ndarray, feature_names: Sequence[str], rul_true,
                        top_n: int = 12,
                        bins: Sequence[int] = (0, 15, 30, 50, 75, 100, 126, 1000)
                        ) -> plt.Figure:
    """Heatmap of mean |SHAP| per feature across life stages."""
    labels = [f"{a}-{b}" if b < 1000 else f">{a}" for a, b in zip(bins[:-1], bins[1:])]
    stage = pd.cut(np.asarray(rul_true), bins=list(bins), labels=labels, right=False)
    frame = pd.DataFrame(np.abs(shap_values), columns=feature_names)
    frame["stage"] = stage
    agg = frame.groupby("stage", observed=True).mean().T
    top = agg.mean(axis=1).nlargest(top_n).index
    agg = agg.loc[top]
    fig, ax = plt.subplots(figsize=(10, 0.42 * top_n + 1.8))
    sns.heatmap(agg, cmap=SEQUENTIAL, ax=ax, annot=True, fmt=".1f", cbar_kws={"label":
                "mean |SHAP| (cycles)"}, linewidths=1, linecolor="white",
                annot_kws={"fontsize": 8})
    ax.set(xlabel="True remaining useful life (cycles)  -  failure on the left", ylabel="",
           title="What drives the prediction at each life stage")
    fig.tight_layout()
    return fig
