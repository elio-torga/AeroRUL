"""End-to-end AeroRUL training pipeline.

Usage::

    python -m src.pipeline                       # all four datasets, full run
    python -m src.pipeline --datasets FD001      # a single dataset
    python -m src.pipeline --quick               # smoke test (minutes)

For every sub-dataset the pipeline

1. loads raw data and builds the clipped RUL target;
2. splits engines into train / validation (engine-level, seeded);
3. fits the unsupervised feature pipeline on training engines;
4. tunes HistGradientBoosting with engine-grouped CV;
5. fits Ridge, Random Forest, Extra Trees, HGB (default / tuned) and the
   stacking ensemble;
6. selects the production model on validation cut points;
7. fits conformal quantile intervals and the isotonic calibrator;
8. evaluates everything on the official NASA test set;
9. simulates maintenance policies on full validation trajectories;
10. computes SHAP explanations and saves the model bundle and figures.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Memory  # noqa: E402

from src.data.load_data import DATASET_INFO, load_dataset  # noqa: E402
from src.data.preprocess import (  # noqa: E402
    add_test_rul,
    add_train_rul,
    last_cycle_rows,
    sample_truncation_points,
    split_engines,
    subsample_rows,
)
from src.features.engineering import build_feature_pipeline  # noqa: E402
from src.models.evaluate import (  # noqa: E402
    compare_policies,
    evaluate_models,
    evaluate_predictions,
    threshold_sweep,
)
from src.models.predict import RULPredictor  # noqa: E402
from src.models.train import (  # noqa: E402
    MODEL_LABELS,
    ConformalQuantileRegressor,
    IsotonicRULCalibrator,
    fit_candidates,
    strip_memory,
    tune_hgb,
)
from src.utils import (  # noqa: E402
    add_file_handler,
    get_logger,
    load_config,
    resolve_path,
    set_seed,
)
from src.utils import visualization as viz  # noqa: E402

logger = get_logger("aerorul.pipeline")


def _quick_config(cfg: dict) -> dict:
    """Shrink the configuration for a fast smoke test."""
    cfg = copy.deepcopy(cfg)
    cfg["tuning"]["n_iter"] = 2
    cfg["tuning"]["cv_folds"] = 2
    for name in ("random_forest", "extra_trees"):
        cfg["models"][name]["n_estimators"] = 30
    cfg["models"]["hist_gradient_boosting"]["max_iter"] = 150
    cfg["models"]["stacking"]["cv_folds"] = 3
    cfg["selection"]["top_k"] = 60
    cfg["uncertainty"]["quantile_max_iter"] = 100
    cfg["explainability"]["n_explain"] = 100
    return cfg


def _check_row_invariance(model, features: pd.DataFrame, rows: pd.Index) -> None:
    """Guard: the supervised stage must be strictly row-wise.

    Predicting on a subset of rows must equal predicting on the full frame and
    then sub-setting. A violation means some step aggregates over the rows it
    receives, which silently breaks inference on latest-cycle data.
    """
    subset = model.predict(features.loc[rows])
    full = pd.Series(model.predict(features), index=features.index).loc[rows].to_numpy()
    gap = float(np.max(np.abs(subset - full)))
    if gap > 1e-6:
        raise RuntimeError(f"Model stage is not row-wise (max gap {gap:.4f} cycles)")
    logger.info("Row-invariance check passed (max gap %.1e)", gap)


def run_dataset(name: str, cfg: dict, tune: bool = True) -> Dict[str, object]:
    """Train, evaluate, explain and persist the model of one sub-dataset."""
    t0 = time.time()
    seed = cfg["project"]["random_seed"]
    set_seed(seed)
    max_rul = cfg["target"]["max_rul"]
    figures = resolve_path(cfg["paths"]["figures_dir"])
    processed = resolve_path(cfg["paths"]["processed_dir"])
    models_dir = resolve_path(cfg["paths"]["models_dir"])

    # 1-2. data, targets and engine-level split -------------------------------
    ds = load_dataset(name, resolve_path(cfg["paths"]["raw_dir"]))
    full_train = add_train_rul(ds.train, max_rul)
    test = add_test_rul(ds.test, ds.test_rul, max_rul)
    train_df, val_df = split_engines(full_train, cfg["split"]["validation_fraction"], seed)
    logger.info("[%s] %d training engines, %d validation engines", name,
                train_df["unit"].nunique(), val_df["unit"].nunique())

    # 3. unsupervised feature stage (fitted on training engines only) --------
    n_cond = cfg["data"]["n_operating_conditions"][name]
    feature_pipe = build_feature_pipeline(cfg, n_cond).fit(train_df)
    F_train, F_val, F_test = (feature_pipe.transform(d) for d in (train_df, val_df, test))
    logger.info("[%s] %d engineered features", name, F_train.shape[1] - 2)

    train_idx = subsample_rows(train_df, cfg["split"]["train_row_stride"]).index
    X_train, y_train = F_train.loc[train_idx], train_df.loc[train_idx, "rul"].to_numpy()
    cut = sample_truncation_points(val_df, cfg["split"]["n_cuts_per_engine"],
                                   cfg["split"]["min_cycles_before_cut"], seed)
    X_cut, y_cut = F_val.loc[cut.index], cut["rul"].to_numpy()
    last_idx = last_cycle_rows(test).index
    target_col = "rul" if cfg["target"]["clip_test_targets"] else "rul_true"
    X_test, y_test = F_test.loc[last_idx], test.loc[last_idx, target_col].to_numpy()

    cache = resolve_path(cfg["paths"]["cache_dir"]) / name
    memory = Memory(str(cache), verbose=0)

    # 4. hyper-parameter search ----------------------------------------------
    cv_table = pd.DataFrame()
    hgb_params = dict(cfg["models"]["hist_gradient_boosting"])
    if tune and cfg["tuning"]["enabled"]:
        best, cv_table = tune_hgb(X_train, y_train, cfg, seed, memory)
        hgb_params.update(best)
        cv_table.insert(0, "dataset", name)

    # 5. candidate models ------------------------------------------------------
    models = fit_candidates(X_train, y_train, cfg, seed, hgb_params, memory)
    for pipe in models.values():
        strip_memory(pipe)

    # 6. model selection on validation cut points ----------------------------
    val_metrics = evaluate_models(models, X_cut, y_cut, cfg, name, "validation")
    production = val_metrics.sort_values("rmse").iloc[0]["model"]
    logger.info("[%s] production model: %s", name, MODEL_LABELS[production])
    test_metrics = evaluate_models(models, X_test, y_test, cfg, name, "test")
    prod_pipe = models[production]
    _check_row_invariance(prod_pipe, F_test, last_idx)

    # 7. uncertainty + calibration -------------------------------------------
    ucfg = cfg["uncertainty"]
    Z_train = prod_pipe[:-1].transform(X_train)
    cqr = ConformalQuantileRegressor(
        alpha=ucfg["alpha"], max_rul=max_rul, random_state=seed,
        hgb_params={k: v for k, v in hgb_params.items() if k != "max_iter"},
        max_iter=ucfg["quantile_max_iter"]).fit(Z_train, y_train)
    cqr.calibrate(prod_pipe[:-1].transform(X_cut), y_cut)
    calibrator = IsotonicRULCalibrator(min_gain=cfg["calibration"]["min_gain"],
                                       random_state=seed)
    calibrator.fit(prod_pipe.predict(X_cut), y_cut, cut["unit"].to_numpy())
    if not cfg["calibration"]["isotonic"]:
        calibrator.enabled_ = False

    selector = prod_pipe.named_steps["selector"]
    bundle = RULPredictor(
        dataset=name, feature_pipeline=feature_pipe, model_pipeline=prod_pipe,
        interval_model=cqr, calibrator=calibrator, max_rul=max_rul,
        risk_thresholds=cfg["business"]["risk_thresholds"], model_name=production,
        metadata={
            "dataset_info": DATASET_INFO[name],
            "hgb_params": hgb_params,
            "n_engineered_features": int(F_train.shape[1] - 2),
            "selected_features": selector.selected_features_,
            "n_after_redundancy": int(selector.n_after_redundancy_),
            "background_features": X_train.sample(
                cfg["explainability"]["n_background"], random_state=seed),
            "interval_alpha": ucfg["alpha"],
            "conformal_correction": cqr.correction_,
            "raw_interval_coverage": cqr.raw_coverage_,
            "isotonic_enabled": calibrator.enabled_,
            "trained_at": time.strftime("%Y-%m-%d %H:%M"),
        })

    # 8. production bundle on the official test set --------------------------
    test_pred = bundle.predict_from_features(X_test)
    test_pred["rul_true"] = y_test
    prod_metrics = evaluate_predictions(y_test, test_pred["rul_pred"], cfg,
                                        test_pred["rul_lower"], test_pred["rul_upper"])
    test_metrics = pd.concat([test_metrics, pd.DataFrame(
        [{"dataset": name, "split": "test", "model": "production", **prod_metrics}])],
        ignore_index=True)
    logger.info("[%s] TEST production  RMSE %.2f | NASA %.0f | coverage %.2f", name,
                prod_metrics["rmse"], prod_metrics["nasa_score"], prod_metrics["coverage"])

    # 9. full-trajectory validation predictions -> business simulation ----
    val_traj = bundle.predict_from_features(F_val)
    val_traj["rul_true"] = val_df["rul_true"].to_numpy()
    train_life = train_df.groupby("unit")["cycle"].max().to_numpy()
    policies = compare_policies(val_traj, train_life, cfg)
    policies.insert(0, "dataset", name)

    # 10. explainability --------------------------------------------------------
    t_shap = time.time()
    explainer = bundle.explainer()
    shap_vals = explainer.shap_values(bundle.model_inputs(X_test))
    shap_frame = pd.DataFrame(shap_vals, columns=explainer.feature_names)
    importance = shap_frame.abs().mean().sort_values(ascending=False)
    logger.info("[%s] SHAP on %d engines in %.1fs", name, len(X_test), time.time() - t_shap)

    # persistence ---------------------------------------------------------------
    bundle.save(models_dir / f"aerorul_{name}.joblib")
    test_pred.to_csv(processed / f"{name}_test_predictions.csv", index=False)
    val_traj.to_parquet(processed / f"{name}_validation_trajectories.parquet", index=False)
    pd.DataFrame({"feature": importance.index, "mean_abs_shap": importance.values}).to_csv(
        processed / f"{name}_shap_importance.csv", index=False)

    # figures ---------------------------------------------------------------------
    viz.set_style()
    fig = viz.plot_predictions_vs_truth(
        y_test, test_pred["rul_pred"], test_pred["rul_lower"], test_pred["rul_upper"],
        title=f"{name} - official test set ({MODEL_LABELS[production]}, "
              f"RMSE {prod_metrics['rmse']:.1f})")
    viz.save_figure(fig, f"{name}_test_predictions", figures)
    errors = {MODEL_LABELS[m]: np.clip(models[m].predict(X_test), 0, max_rul) - y_test
              for m in ("random_forest", "hgb_tuned", "stacking")}
    viz.save_figure(viz.plot_error_distribution(errors, f"{name} - test error distribution"),
                    f"{name}_error_distribution", figures)
    viz.save_figure(viz.plot_coverage_by_bin(
        y_test, test_pred["rul_lower"], test_pred["rul_upper"], 1 - ucfg["alpha"],
        title=f"{name} - 80 % interval coverage on test engines"),
        f"{name}_interval_coverage", figures)
    viz.save_figure(viz.plot_shap_importance(
        shap_vals, explainer.feature_names, cfg["explainability"]["top_n_features"],
        title=f"{name} - global feature importance (mean |SHAP|, test engines)"),
        f"{name}_shap_importance", figures)
    units = val_traj.groupby("unit")["cycle"].max().sort_values().index
    pick = [units[int(q * (len(units) - 1))] for q in (0.1, 0.4, 0.7, 0.95)]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8), sharey=True)
    for ax, unit in zip(axes, pick):
        viz.plot_engine_trajectory(val_traj, unit, max_rul, ax=ax)
    axes[0].legend(loc="lower left")
    fig.suptitle(f"{name} - online RUL tracking on held-out validation engines",
                 x=0.01, ha="left", fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    viz.save_figure(fig, f"{name}_validation_trajectories", figures)
    plt.close("all")

    shutil.rmtree(cache, ignore_errors=True)
    logger.info("[%s] done in %.1f min", name, (time.time() - t0) / 60)
    return {"val_metrics": val_metrics, "test_metrics": test_metrics, "cv_table": cv_table,
            "policies": policies, "val_traj": val_traj.assign(dataset=name),
            "train_lifetimes": train_life, "production": production,
            "importance": importance, "metadata": {
                k: v for k, v in bundle.metadata.items() if k != "background_features"}}


def _fleet_level_outputs(results: Dict[str, dict], cfg: dict) -> pd.DataFrame:
    """Pooled business simulation and cross-dataset comparison figures."""
    figures = resolve_path(cfg["paths"]["figures_dir"])
    viz.set_style()
    traj = pd.concat([r["val_traj"].assign(unit=lambda d, n=n: n + "-" + d["unit"].astype(str))
                      for n, r in results.items()], ignore_index=True)
    lifetimes = np.concatenate([r["train_lifetimes"] for r in results.values()])
    fleet = compare_policies(traj, lifetimes, cfg)
    fleet.insert(0, "dataset", "FLEET")
    viz.save_figure(viz.plot_policy_costs(
        fleet, f"Fleet maintenance economics - {fleet['engines'].iloc[0]} held-out "
               f"run-to-failure engines"), "fleet_policy_costs", figures)
    sweep = pd.concat([threshold_sweep(traj, s, cfg) for s in ("rul_pred", "rul_lower")])
    chosen = cfg["business"]["maintenance_lead_time"] + cfg["business"]["safety_margin"]
    viz.save_figure(viz.plot_threshold_sweep(sweep, chosen), "fleet_threshold_sweep", figures)
    sweep.to_csv(resolve_path(cfg["paths"]["reports_dir"]) / "metrics" / "threshold_sweep.csv",
                 index=False)

    test = pd.concat([r["test_metrics"] for r in results.values()])
    order = ["ridge", "random_forest", "hgb_tuned", "stacking"]
    labels = {k: MODEL_LABELS[k] for k in order}
    if len(results) > 1:
        for metric in ("rmse", "nasa_score"):
            fig = viz.plot_model_comparison(
                test[test["model"].isin(order)], metric, order, labels,
                title=f"Official test set - {metric.upper().replace('_', ' ')} "
                      f"(lower is better)")
            if metric == "nasa_score":
                fig.axes[0].set_yscale("log")
            viz.save_figure(fig, f"model_comparison_{metric}", figures)
    plt.close("all")
    return fleet


def main(argv: List[str] = None) -> None:
    parser = argparse.ArgumentParser(description="AeroRUL training pipeline")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--no-tune", action="store_true")
    parser.add_argument("--quick", action="store_true", help="fast smoke-test settings")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.quick:
        cfg = _quick_config(cfg)
    reports = resolve_path(cfg["paths"]["reports_dir"])
    metrics_dir = resolve_path(str(Path(cfg["paths"]["reports_dir"]) / "metrics"))
    add_file_handler(logger, reports / "pipeline.log")
    datasets = args.datasets or cfg["data"]["datasets"]

    results = {name: run_dataset(name, cfg, tune=not args.no_tune) for name in datasets}

    fleet = _fleet_level_outputs(results, cfg)
    tables = {
        "validation_metrics": pd.concat([r["val_metrics"] for r in results.values()]),
        "test_metrics": pd.concat([r["test_metrics"] for r in results.values()]),
        "tuning_results": pd.concat([r["cv_table"] for r in results.values()]),
        "maintenance_policies": pd.concat(
            [r["policies"] for r in results.values()] + [fleet]),
    }
    for fname, table in tables.items():
        table.to_csv(metrics_dir / f"{fname}.csv", index=False)
    summary = {
        name: {"production_model": r["production"],
               "test": r["test_metrics"].set_index("model").loc["production"].drop(
                   ["dataset", "split"]).to_dict(),
               "top_features": r["importance"].head(10).round(3).to_dict(),
               "metadata": r["metadata"]}
        for name, r in results.items()}
    with open(metrics_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    prod = tables["test_metrics"].query("model == 'production'")
    print("\nOfficial test set - production models")
    print(prod[["dataset", "rmse", "mae", "r2", "nasa_score", "phm_accuracy",
                "coverage", "mean_width"]].round(3).to_string(index=False))
    print("\nFleet maintenance economics")
    print(fleet[["policy", "unplanned_failures", "cost_per_engine",
                 "savings_vs_reactive"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
