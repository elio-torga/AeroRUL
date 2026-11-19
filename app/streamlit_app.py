"""AeroRUL - fleet health monitoring demo.

Run from the project root::

    streamlit run app/streamlit_app.py

Upload raw C-MAPSS sensor data (whitespace-separated .txt or CSV with
canonical column names) and get, for every engine, its Remaining Useful
Life, an 80 % conformal prediction interval, a risk level, a maintenance
recommendation and a SHAP explanation of the prediction.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.load_data import DATASET_INFO, read_sensor_table  # noqa: E402
from src.features.engineering import describe_feature  # noqa: E402
from src.models.predict import RULPredictor  # noqa: E402
from src.utils import load_config  # noqa: E402
from src.utils import visualization as viz  # noqa: E402

st.set_page_config(page_title="AeroRUL - Engine Health Monitoring", page_icon="✈️",
                   layout="wide")
viz.set_style()
CFG = load_config(ROOT / "config.yaml")
MODELS_DIR = ROOT / CFG["paths"]["models_dir"]
RISK_ICON = {"CRITICAL": "🔴", "WARNING": "🟠", "HEALTHY": "🟢"}


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model bundle...")
def load_bundle(dataset: str) -> RULPredictor:
    return RULPredictor.load(MODELS_DIR / f"aerorul_{dataset}.joblib")


@st.cache_data(show_spinner=False)
def load_summary() -> dict:
    path = ROOT / CFG["paths"]["reports_dir"] / "metrics" / "summary.json"
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_data(show_spinner="Computing features and predictions...")
def run_inference(dataset: str, raw: pd.DataFrame):
    bundle = load_bundle(dataset)
    features = bundle.compute_features(raw)
    trajectory = bundle.predict_from_features(features)
    latest = trajectory.loc[trajectory.groupby("unit")["cycle"].idxmax()]
    return features, trajectory, latest.sort_values("rul_lower").reset_index(drop=True)


def available_datasets() -> list:
    return [d for d in DATASET_INFO if (MODELS_DIR / f"aerorul_{d}.joblib").exists()]


def sample_file(dataset: str) -> Path:
    return ROOT / CFG["paths"]["external_dir"] / f"sample_fleet_{dataset}.csv"


def recommendation(row: pd.Series, lead_time: int) -> str:
    if row["risk_level"] == "CRITICAL":
        window = int(row["rul_lower"]) - lead_time
        if window <= 0:
            return (f"Remove from service immediately - the pessimistic RUL bound is already "
                    f"inside the {lead_time}-cycle logistics lead time.")
        return (f"Order removal now - with the {lead_time}-cycle logistics lead time, "
                f"only ~{window} cycles of margin remain at the pessimistic bound.")
    if row["risk_level"] == "WARNING":
        return ("Plan a shop visit at the next convenient slot and increase "
                "monitoring frequency (borescope inspection recommended).")
    return "No action required - continue routine monitoring."


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
datasets = available_datasets()
st.sidebar.title("✈️ AeroRUL")
st.sidebar.caption("Remaining Useful Life prediction for turbofan engines")
if not datasets:
    st.error("No trained model found in `models/`. Run `python -m src.pipeline` first.")
    st.stop()

dataset = st.sidebar.selectbox(
    "Fleet scenario (model)", datasets,
    format_func=lambda d: f"{d} - {DATASET_INFO[d]['conditions']} regime(s), "
                          f"{DATASET_INFO[d]['faults']}")
source = st.sidebar.radio("Sensor data", ["Demo fleet", "Upload file"])
uploaded = None
if source == "Upload file":
    uploaded = st.sidebar.file_uploader(
        "C-MAPSS .txt or CSV", type=["txt", "csv"],
        help="Columns: unit, cycle, op_setting_1..3, s1..s21 (raw C-MAPSS format accepted).")
lead_time = st.sidebar.slider("Maintenance lead time (cycles)", 0, 30,
                              CFG["business"]["maintenance_lead_time"])
st.sidebar.divider()
st.sidebar.markdown(
    "**Risk levels** use the *lower* bound of the 80 % interval:\n\n"
    f"🔴 Critical: ≤ {CFG['business']['risk_thresholds']['critical']} cycles  \n"
    f"🟠 Warning: ≤ {CFG['business']['risk_thresholds']['warning']} cycles  \n"
    "🟢 Healthy: otherwise")

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
try:
    if uploaded is not None:
        raw = read_sensor_table(io.BytesIO(uploaded.getvalue()))
    elif source == "Demo fleet" and sample_file(dataset).exists():
        raw = read_sensor_table(sample_file(dataset))
    else:
        st.info("Upload a sensor file in the sidebar to start, or select *Demo fleet*.")
        st.stop()
except Exception as exc:  # noqa: BLE001 - surface parsing errors to the user
    st.error(f"Could not read the file: {exc}")
    st.stop()

bundle = load_bundle(dataset)
features, trajectory, latest = run_inference(dataset, raw)

# --------------------------------------------------------------------------- #
# Header + KPIs
# --------------------------------------------------------------------------- #
st.title("Engine Health Monitoring")
st.caption(f"Model: **{bundle.model_name.replace('_', ' ').title()}** trained on {dataset} · "
           f"{len(latest)} engines · {len(raw):,} sensor records")
counts = latest["risk_level"].value_counts()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Engines monitored", len(latest))
k2.metric("🔴 Critical", int(counts.get("CRITICAL", 0)))
k3.metric("🟠 Warning", int(counts.get("WARNING", 0)))
k4.metric("Median predicted RUL", f"{latest['rul_pred'].median():.0f} cycles")

tab_fleet, tab_engine, tab_model = st.tabs(["Fleet overview", "Engine drill-down", "Model card"])

# --------------------------------------------------------------------------- #
# Fleet overview
# --------------------------------------------------------------------------- #
with tab_fleet:
    left, right = st.columns([1.25, 1])
    with left:
        fig, ax = plt.subplots(figsize=(7, max(3, 0.28 * len(latest) + 1)))
        ordered = latest.iloc[::-1]
        y = np.arange(len(ordered))
        colors = [viz.RISK_COLORS[r] for r in ordered["risk_level"]]
        ax.hlines(y, ordered["rul_lower"], ordered["rul_upper"], color=viz.GRID, lw=6)
        ax.scatter(ordered["rul_pred"], y, c=colors, s=40, zorder=3, edgecolor="white")
        ax.set_yticks(y, [f"Engine {u}" for u in ordered["unit"]], fontsize=8)
        ax.axvline(lead_time, color=viz.INK_3, ls="--", lw=1)
        ax.set_xlabel("Predicted RUL with 80 % interval (cycles)")
        ax.set_title("Fleet ranked by risk (most urgent on top)")
        ax.grid(axis="y", visible=False)
        st.pyplot(fig, clear_figure=True)
    with right:
        table = latest.assign(
            risk=lambda d: d["risk_level"].map(lambda r: f"{RISK_ICON[r]} {r}"),
            interval=lambda d: d.apply(
                lambda r: f"{r['rul_lower']:.0f} - {r['rul_upper']:.0f}", axis=1))
        st.dataframe(
            table[["unit", "cycle", "rul_pred", "interval", "risk"]].rename(columns={
                "unit": "Engine", "cycle": "Cycles flown", "rul_pred": "RUL (pred.)",
                "interval": "80 % interval", "risk": "Risk"}),
            hide_index=True, width="stretch", height=420,
            column_config={"RUL (pred.)": st.column_config.NumberColumn(format="%.0f")})
        st.download_button("Download predictions (CSV)",
                           latest.to_csv(index=False).encode(),
                           file_name=f"aerorul_predictions_{dataset}.csv", mime="text/csv")

# --------------------------------------------------------------------------- #
# Engine drill-down
# --------------------------------------------------------------------------- #
with tab_engine:
    unit = st.selectbox("Engine", latest["unit"].tolist(),
                        format_func=lambda u: f"Engine {u} - "
                        f"{latest.set_index('unit').loc[u, 'risk_level']}")
    row = latest.set_index("unit").loc[unit]
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted RUL", f"{row['rul_pred']:.0f} cycles")
    c2.metric("80 % interval", f"{row['rul_lower']:.0f} - {row['rul_upper']:.0f}")
    c3.metric("Risk level", f"{RISK_ICON[row['risk_level']]} {row['risk_level']}")
    {"CRITICAL": st.error, "WARNING": st.warning, "HEALTHY": st.success}[row["risk_level"]](
        recommendation(row, lead_time))

    g1, g2 = st.columns(2)
    sub = trajectory[trajectory["unit"] == unit]
    with g1:
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        ax.fill_between(sub["cycle"], sub["rul_lower"], sub["rul_upper"],
                        color=viz.PALETTE[0], alpha=0.18, lw=0, label="80 % interval")
        ax.plot(sub["cycle"], sub["rul_pred"], color=viz.PALETTE[0], label="Predicted RUL")
        ax.axhline(lead_time, color=viz.INK_3, ls="--", lw=1, label="Lead time")
        ax.set(xlabel="Cycle", ylabel="RUL (cycles)", title="RUL estimate over the record")
        ax.set_ylim(bottom=0)
        ax.legend(loc="lower left")
        st.pyplot(fig, clear_figure=True)
    with g2:
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        ax.plot(sub["cycle"], sub["health_indicator"], color=viz.PALETTE[2])
        ax.axhline(0, color=viz.RISK_COLORS["CRITICAL"], ls="--", lw=1)
        ax.set(xlabel="Cycle", ylabel="Health indicator", ylim=(-0.2, 1.2),
               title="Health indicator (1 = healthy, 0 = end of life)")
        st.pyplot(fig, clear_figure=True)

    st.subheader("Why this prediction?")
    with st.spinner("Computing SHAP explanation..."):
        last_row = features.loc[[sub.index[-1]]]
        explanation = bundle.explain(last_row)
    contrib = pd.DataFrame({"feature": explanation.feature_names,
                            "value": explanation.data[0],
                            "shap": explanation.values[0]})
    contrib["abs"] = contrib["shap"].abs()
    top = contrib.nlargest(10, "abs").iloc[::-1]
    base = float(explanation.base_values[0])
    raw_pred = base + contrib["shap"].sum()
    e1, e2 = st.columns([1.3, 1])
    with e1:
        fig, ax = plt.subplots(figsize=(7, 4.6))
        colors = [viz.PALETTE[1] if v < 0 else viz.PALETTE[0] for v in top["shap"]]
        ax.barh([describe_feature(f) for f in top["feature"]], top["shap"], color=colors,
                height=0.6)
        ax.axvline(0, color=viz.INK, lw=1)
        ax.set_xlabel("Contribution to predicted RUL (cycles)")
        ax.set_title(f"Fleet average {base:.0f} → model output {raw_pred:.0f} cycles")
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", visible=False)
        st.pyplot(fig, clear_figure=True)
    with e2:
        st.markdown("**Reading the chart** - orange bars *shorten* the predicted life, "
                    "blue bars *extend* it. Values are in cycles and add up exactly from "
                    "the fleet-average prediction to this engine's prediction.")
        worst = contrib.nsmallest(3, "shap")
        st.markdown("**Main degradation drivers**")
        for _, r in worst.iterrows():
            st.markdown(f"- {describe_feature(r['feature'])}: **{r['shap']:+.1f}** cycles")

# --------------------------------------------------------------------------- #
# Model card
# --------------------------------------------------------------------------- #
with tab_model:
    meta = bundle.metadata
    summary = load_summary().get(dataset, {})
    st.markdown(f"""
**Scenario** - {dataset}: {meta['dataset_info']['conditions']} operating regime(s),
{meta['dataset_info']['faults']}.

**Pipeline** - regime normalisation → {meta['n_engineered_features']} causal temporal
features → health indicator → selection of {len(meta['selected_features'])} features →
**{bundle.model_name.replace('_', ' ')}**. Intervals: conformalized quantile regression
(nominal {1 - meta['interval_alpha']:.0%}, conformal correction
{meta['conformal_correction']:+.1f} cycles). Isotonic recalibration:
{'enabled' if meta['isotonic_enabled'] else 'not adopted (no CV gain)'}.
Trained {meta['trained_at']}.
""")
    if summary:
        test = summary["test"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test RMSE", f"{test['rmse']:.2f}")
        m2.metric("NASA score", f"{test['nasa_score']:.0f}")
        m3.metric("PHM accuracy", f"{test['phm_accuracy']:.0%}")
        m4.metric("Interval coverage", f"{test['coverage']:.0%}")
    st.caption("Limitations: trained on simulated C-MAPSS data; operating regimes and "
               "sensors must match the training scenario. Not for airworthiness decisions.")
