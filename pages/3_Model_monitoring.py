import streamlit as st
from mlflow.tracking import MlflowClient
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from pipeline.model_drift import calculate_psi, ks_test_pvalue
from pipeline.flow import credit_risk_pipeline
import streamlit.components.v1 as components
import math

st.set_page_config(page_title="Model Comparison", layout="wide")
st.title("Compare ML Models from Registry")

client = MlflowClient()

# Step 1: List all registered models
# Use search_registered_models if list_registered_models() is unavailable
try:
    models = client.search_registered_models()
except AttributeError:
    st.error("Your MLflow version does not support listing models this way.")
    st.stop()

model_names = [m.name for m in models]

selected_model = st.selectbox("📦 Select a model", model_names)

# Step 2: Get versions for the selected model
all_versions = client.search_model_versions(f"name='{selected_model}'")
version_options = sorted([int(v.version) for v in all_versions], reverse=True)

# Step 3: User selects exactly 2 versions to compare
selected_versions = st.multiselect(
    "Select exactly 2 versions to compare",
    version_options,
    default=version_options[:2] if len(version_options) >= 2 else version_options
)

# Step 4: Map version -> (run_id, stage)
version_info = {
    int(v.version): {"run_id": v.run_id, "stage": v.current_stage}
    for v in all_versions if int(v.version) in selected_versions
}

# Step 5: Helper to get metrics
def get_metrics(run_id):
    run = client.get_run(run_id)
    metrics = run.data.metrics
    return {
        "test_accuracy": metrics.get("test_accuracy"),
        "test_roc_auc": metrics.get("test_roc_auc"),
        "val_accuracy": metrics.get("val_accuracy"),
        "val_roc_auc": metrics.get("val_roc_auc"),
    }

# Step 6: Get metrics and stage for each version
if len(selected_versions) == 2:

    v1, v2 = selected_versions
    info1 = version_info[v1]
    info2 = version_info[v2]
    metrics1 = get_metrics(info1["run_id"])
    metrics2 = get_metrics(info2["run_id"])

    # Step 7: Show side-by-side comparison
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Version {v1} (stage: {info1['stage'] or 'None'})")
        st.metric("Test Accuracy", f"{metrics1['test_accuracy']:.2f}" if metrics1['test_accuracy'] else "N/A")
        st.metric("Test ROC AUC", f"{metrics1['test_roc_auc']:.2f}" if metrics1['test_roc_auc'] else "N/A")
        st.metric("Validation Accuracy", f"{metrics1['val_accuracy']:.2f}" if metrics1['val_accuracy'] else "N/A")
        st.metric("Validation ROC AUC", f"{metrics1['val_roc_auc']:.2f}" if metrics1['val_roc_auc'] else "N/A")

    with col2:
        st.subheader(f"Version {v2} (stage: {info2['stage'] or 'None'})")
        st.metric("Test Accuracy", f"{metrics2['test_accuracy']:.2f}" if metrics2['test_accuracy'] else "N/A")
        st.metric("Test ROC AUC", f"{metrics2['test_roc_auc']:.2f}" if metrics2['test_roc_auc'] else "N/A")
        st.metric("Validation Accuracy", f"{metrics2['val_accuracy']:.2f}" if metrics2['val_accuracy'] else "N/A")
        st.metric("Validation ROC AUC", f"{metrics2['val_roc_auc']:.2f}" if metrics2['val_roc_auc'] else "N/A")

st.markdown("---")
st.header("Feature Distribution Drift Check")
st.caption("Compare training data vs. latest inference input to detect feature distribution changes.")

# Load reference data
ref_path = "data/synthetic_credit_risk.csv"
if not os.path.exists(ref_path):
    st.error("Reference training data not found at 'data/synthetic_credit_risk.csv'.")
    st.stop()
df = pd.read_csv(ref_path)
ref_df = df.drop(columns=["target"]) if "target" in df.columns else df.copy()

# Load latest prediction input (or placeholder)
pred_path = "artifacts/prediction_input_sample.csv"
if os.path.exists(pred_path):
    new_df = pd.read_csv(pred_path)
    st.success("Comparing with latest prediction input sample.")
else:
    new_df = ref_df.sample(frac=1.0, random_state=42)
    st.warning("No prediction input sample found. Using shuffled training data as placeholder.")

# Identify numeric features in common
numeric_features = ref_df.select_dtypes(include="number").columns
common_features  = list(numeric_features.intersection(new_df.columns))

if not common_features:
    st.error("No numeric features found in common to compare.")
    st.stop()

# Feature selector
selected_features = st.multiselect(
    "Select features to inspect for drift",
    common_features,
    default=[common_features[0]],
    key="drift_features"
)

if not selected_features:
    st.info("Select at least one feature to visualize.")
    st.stop()

# Compute drift statistics
drift_stats = []
for feature in selected_features:
    psi   = calculate_psi(ref_df[feature], new_df[feature])
    ks_p  = ks_test_pvalue(ref_df[feature], new_df[feature])
    if psi > 0.2 or ks_p < 0.05:
        status = "Drift Detected"
    elif psi > 0.1:
        status = "Moderate Drift"
    else:
        status = "No Drift"
    drift_stats.append({
        "feature":    feature,
        "PSI":        f"{psi:.4f}",
        "KS p-value": f"{ks_p:.4f}",
        "status":     status
    })

# Build summary DataFrame
stats_df = pd.DataFrame(drift_stats).set_index("feature")

# Display summary table at top
st.subheader("Drift Summary")
st.dataframe(stats_df, use_container_width=True)

# Plot all selected features in a grid
n_feats = len(selected_features)
ncols   = 2
nrows   = math.ceil(n_feats / ncols)

fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(ncols * 5, nrows * 3),
    constrained_layout=True
)
axes = axes.flatten()

for ax, feature in zip(axes, selected_features):
    sns.kdeplot(
        ref_df[feature],
        ax=ax,
        label="Training (Ref)",
        color="skyblue",
        fill=False
    )
    sns.kdeplot(
        new_df[feature],
        ax=ax,
        label="Latest Input",
        color="salmon",
        fill=False
    )
    ax.set_title(feature, fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend(fontsize=8)

# Remove any unused subplots
for ax in axes[n_feats:]:
    fig.delaxes(ax)

# Render the grid of plots
st.pyplot(fig)

# html drift
st.title("Data Drift (evidently) Batch processing")

if st.button("Run pipeline & refresh report"):
    # this will execute your Prefect flow locally
    report_path = credit_risk_pipeline()
    st.success(f"Pipeline complete! Report at: {report_path}")

    # load and embed
    with open(report_path, "r", encoding="utf-8") as f:
        drift_html = f.read()
    components.html(drift_html, height=800, scrolling=True)
else:
    st.info("Click the button to run the pipeline and view the latest drift report.")