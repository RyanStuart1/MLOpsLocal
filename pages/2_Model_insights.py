import streamlit as st
import os
from shap import Explanation, plots
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from pipeline.metrics import get_model_metrics
from pipeline.chatbot import show_chatbot_sidebar
from pipeline.shap_analysis import compute_shap_native
import json
from pages_components.home_button import render_home_button
from datetime import datetime
import glob
import re
import seaborn as sns

st.set_page_config(layout="wide")
render_home_button()
st.title("Model Insights")

client    = MlflowClient()
model_name = "credit-risk-model"  # must match model version
# pull all model versions, pick the highest one by version number
all_versions   = client.search_model_versions(f"name='{model_name}'")
version_nums = [v.version for v in all_versions]
latest_obj     = max(all_versions, key=lambda v: int(v.version))
latest_version = latest_obj.version
run_id         = latest_obj.run_id
run            = client.get_run(run_id)
metrics        = run.data.metrics


# Adjust column width ratios to give more space
col1, col2, col3, col4= st.columns([1, 1, 1, 1])

with col1:
    st.subheader("Validation Set Matrix")
    st.image("artifacts/confusion_matrices/confusion_matrix_val.png",  use_container_width=True)

with col2:
    st.subheader("Test Set Matrix")
    st.image("artifacts/confusion_matrices/confusion_matrix_test.png",  use_container_width=True)


with col3:
    st.subheader("Latest Metrics")
    metrics = get_model_metrics()

    if metrics:
        # create two sub-columns, 1/1 width
        val_col, test_col = st.columns([1, 1])

        with val_col:
            st.metric("Val Accuracy",   f"{metrics['val_accuracy']:.2f}"  if metrics['val_accuracy']  is not None else "N/A")
            st.metric("Val ROC AUC",    f"{metrics['val_roc_auc']:.2f}"   if metrics['val_roc_auc']   is not None else "N/A")

        with test_col:
            st.metric("Test Accuracy",  f"{metrics['test_accuracy']:.2f}" if metrics['test_accuracy'] is not None else "N/A")
            st.metric("Test ROC AUC",   f"{metrics['test_roc_auc']:.2f}"  if metrics['test_roc_auc']  is not None else "N/A")

    else:
        st.warning("No metrics found in MLflow.")

with col4:
    st.subheader("Performance Chart")
    if metrics:
        labels = ["Val Acc","Val ROC","Test Acc","Test ROC"]
        values = [
            metrics.get("val_accuracy", 0) or 0,
            metrics.get("val_roc_auc", 0)   or 0,
            metrics.get("test_accuracy", 0) or 0,
            metrics.get("test_roc_auc", 0)  or 0,
        ]

        gradient = LinearSegmentedColormap.from_list("orange_red", ["#ff6a00", "#ff0000"])
        colors = [gradient(i / (len(values)-1)) for i in range(len(values))]

        fig, ax = plt.subplots(figsize=(4,3))
        ax.bar(labels, values, color=colors, width=0.6)
        ax.set_ylim(0,1)
        ax.set_ylabel("Score")
        ax.set_title("Latest Model Performance")
        
        for i, v in enumerate(values):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")
        st.pyplot(fig)
    else:
        st.info("Run a training flow first to generate metrics.")

# ————————————————————— New SHAP comparison section —————————————————————
st.markdown("---")
st.header("SHAP Feature‑Importance Comparison")

# Pick two registry versions in the body of the page
colA, colB = st.columns(2)
with colA:
    v1 = st.selectbox("Version 2", version_nums, index=version_nums.index(latest_version))
with colB:
    v2 = st.selectbox("Version 1", version_nums, index=version_nums.index(latest_version))

if st.button("🔍 Compare SHAP for Two Versions"):
    # load & clean data
    df = (
        pd.read_csv("data/synthetic_credit_risk.csv")
          .drop(columns=["target", "loan_default"], errors="ignore")
    )
    FEATURES = [
        "age", "credit_score", "loan_amount", "loan_term_months",
        "employment_length_years", "annual_income",
        "debt_to_income_ratio", "open_accounts", "delinquency_history",
    ]

    # sample & cast exactly to the MLflow signature
    x = (
        df[FEATURES]
         .sample(200, random_state=42)
         .astype({
             "age":                     "int64",
             "credit_score":            "int64",
             "loan_amount":             "float64",
             "loan_term_months":        "int64",
             "employment_length_years": "int64",
             "annual_income":           "float64",
             "debt_to_income_ratio":    "float64",
             "open_accounts":           "int64",
             "delinquency_history":     "int64",
         })
         .reset_index(drop=True)
    )

    # compute SHAP arrays for each version
    expl1, shap1 = compute_shap_native(
        model_name=model_name, version=v1, background=x, data=x
    )
    expl2, shap2 = compute_shap_native(
        model_name=model_name, version=v2, background=x, data=x
    )

    # compute Δ mean(|SHAP|) ordering
    mean1 = np.abs(shap1).mean(axis=0)
    mean2 = np.abs(shap2).mean(axis=0)
    delta = mean2 - mean1
    order = np.argsort(delta)[::-1]
    feats_ord = [FEATURES[i] for i in order]

    # reorder sample & shap arrays
    x_ord  = x[feats_ord]
    s1_ord = shap1[:, order]
    s2_ord = shap2[:, order]

    # wrap into Explanation objects
    exp1 = Explanation(
        values        = s1_ord,
        base_values   = expl1.expected_value,
        data          = x_ord.values,
        feature_names = feats_ord,
    )
    exp2 = Explanation(
        values        = s2_ord,
        base_values   = expl2.expected_value,
        data          = x_ord.values,
        feature_names = feats_ord,
    )

    # overlayed beeswarm
    fig, ax = plt.subplots(figsize=(8, len(feats_ord)*0.4), tight_layout=True)
    plt.sca(ax)
    plots.beeswarm(exp1, show=False, color="#1f77b4", alpha=0.6)
    plots.beeswarm(exp2, show=False, color="#ff7f0e", alpha=0.6)
    ax.set_xlim(-1, 1)
    ax.set_title(f"Overlayed SHAP Beeswarm — v{v1} vs v{v2}", pad=20)
    ax.set_xlabel("SHAP value (impact on model output)")

    # fake legend entries
    ax.scatter([], [], color="#1f77b4", label=f"v{v1}", alpha=0.6)
    ax.scatter([], [], color="#ff7f0e", label=f"v{v2}", alpha=0.6)
    ax.legend(frameon=False)

    st.pyplot(fig)

    # Δ bar chart
    fig2, ax2 = plt.subplots(figsize=(6, 4), tight_layout=True)
    y_pos = np.arange(len(feats_ord))
    bars = ax2.barh(y_pos, delta[order], align="center", color="C1")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(feats_ord)
    ax2.invert_yaxis()
    ax2.axvline(0, color="k", linewidth=0.8)
    ax2.set_xlabel("Δ mean(|SHAP value|) = v2 − v1")
    ax2.set_title(f"Change in Feature Importance (v{v1} vs v{v2})", pad=20)
    ax2.bar_label(bars, fmt="%.2f", padding=4)
    st.pyplot(fig2)

    # export shap values as JSON
    shap_summary = [
        {
            "feature": feats_ord[i],
            "mean_v1": float(mean1[order][i]),
            "mean_v2": float(mean2[order][i]),
            "delta":    float(delta[order][i])
        }
        for i in range(len(feats_ord))
    ]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = "artifacts/shap_summary_manual"
    os.makedirs(archive_dir, exist_ok=True)

    archive_path = os.path.join(archive_dir, f"shap_summary_manual_comparison_{ts}.json")
    with open(archive_path, "w", encoding="utf-8") as f:
        json.dump(shap_summary, f, indent=2)

# ———————————————————— SHAP LINEAGE SECTION ————————————————————
st.markdown("---")
st.header("SHAP Lineage Over Time")

# List all archived summary files
archive_dir = "artifacts/shap_summary_archive"
summary_files = sorted(glob.glob(os.path.join(archive_dir, "*.json")))

if not summary_files:
    st.info("No SHAP summary archive files found.")
else:
    lineage_df = None
    for path in summary_files:
        # Parse version and timestamp from filename
        fname = os.path.basename(path)
        # Parse using regex
        match = re.match(r"shap_summary_v(\d+)_vs_v(\d+)_(\d{8}_\d{6})\.json", fname)
        if match:
            v1, v2, ts = match.groups()
            label = f"{v2} vs {v1} Δ"
        else:
            label = fname

        # Load the summary
        with open(path) as f:
            shap_list = json.load(f)

        df = pd.DataFrame(shap_list)
        df["run"] = label

        # Accumulate into wide format
        pivot = df[["feature", "delta", "run"]].pivot(index="feature", columns="run", values="delta")
        if lineage_df is None:
            lineage_df = pivot
        else:
            # Only keep columns from pivot that aren't already in lineage_df
            new_cols = [col for col in pivot.columns if col not in lineage_df.columns]
            lineage_df = pd.concat([lineage_df, pivot[new_cols]], axis=1)

    # Heatmap with color key
    st.subheader("Delta SHAP Lineage Heatmap")
    fig, ax = plt.subplots(figsize=(len(lineage_df.columns)*1.2, len(lineage_df)*0.6))
    sns.heatmap(
        lineage_df.fillna(0),
        annot=True,
        fmt=".4f",
        cmap="RdBu_r",
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "Δ mean(|SHAP value|)"}
    )
    ax.set_title("SHAP Δ per Feature Across Runs")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    st.pyplot(fig)

    st.subheader("Select Feature to Plot Over Time")
    selected_feat = st.selectbox("Choose a feature", lineage_df.index.tolist())
    series = lineage_df.loc[selected_feat].fillna(0)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(series.index, series.values, marker='o')
    ax.set_xticklabels(series.index, rotation=45, ha='right')
    ax.set_title(f"SHAP Delta Over Time: {selected_feat}")
    ax.set_ylabel("Δ SHAP (v_latest - v_previous)")
    ax.axhline(0, color='gray', linestyle='--')
    st.pyplot(fig)
# ———————————————————— SHAP LINEAGE SECTION ————————————————————
show_chatbot_sidebar()