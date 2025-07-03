import streamlit as st
from mlflow.tracking import MlflowClient
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from pipeline.model_drift import calculate_psi, ks_test_pvalue
from pipeline.flow import credit_risk_pipeline
import streamlit.components.v1 as components

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
        "accuracy": metrics.get("accuracy"),
        "roc_auc": metrics.get("roc_auc")
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
        st.metric("Accuracy", f"{metrics1['accuracy']:.2f}" if metrics1['accuracy'] else "N/A")
        st.metric("ROC AUC", f"{metrics1['roc_auc']:.2f}" if metrics1['roc_auc'] else "N/A")

    with col2:
        st.subheader(f"Version {v2} (stage: {info2['stage'] or 'None'})")
        st.metric("Accuracy", f"{metrics2['accuracy']:.2f}" if metrics2['accuracy'] else "N/A")
        st.metric("ROC AUC", f"{metrics2['roc_auc']:.2f}" if metrics2['roc_auc'] else "N/A")

# Drift Check Section
st.markdown("---")
st.header("Feature Distribution Drift Check")
st.caption("Compare training data vs. latest inference input to detect feature distribution changes.")

ref_path = "data/synthetic_credit_risk.csv"
if not os.path.exists(ref_path):
    st.error("Reference training data not found at 'data/synthetic_credit_risk.csv'.")
else:
    df = pd.read_csv(ref_path)
    ref_df = df.drop(columns=["target"]) if "target" in df.columns else df.copy()

    pred_path = "artifacts/prediction_input_sample.csv"
    if os.path.exists(pred_path):
        new_df = pd.read_csv(pred_path)
        st.success("Comparing with latest prediction input sample.")
    else:
        new_df = ref_df.sample(frac=1.0, random_state=42)
        st.warning("No prediction input sample found. Using shuffled training data as placeholder.")

    numeric_features = ref_df.select_dtypes(include='number').columns
    common_features = list(numeric_features.intersection(new_df.columns))

    if not common_features:
        st.error("No numeric features found in common to compare.")
    else:
        selected_features = st.multiselect(
            "Select features to inspect for drift",
            common_features,
            default=common_features[0],
            key="drift_features"
        )

        if selected_features:
            cols = st.columns(2)  # Create 2 columns for layout

            for i, feature in enumerate(selected_features):
                with cols[i % 2]:  # Alternate features between the two columns
                    st.subheader(f"📈 Distribution for '{feature}'")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.kdeplot(ref_df[feature], ax=ax, label="Training (Ref)", color="skyblue", fill=False, linewidth=2)
                    sns.kdeplot(new_df[feature], ax=ax, label="Latest Input", color="salmon", fill=False, linewidth=2)
                    ax.set_xlabel(feature)
                    ax.set_ylabel("Density")
                    ax.legend()
                    st.pyplot(fig)

                    # Drift stats
                    psi = calculate_psi(ref_df[feature], new_df[feature])
                    ks_p = ks_test_pvalue(ref_df[feature], new_df[feature])

                    st.caption(f"PSI: `{psi:.4f}` | KS p-value: `{ks_p:.4f}`")
                    if psi > 0.2 or ks_p < 0.05:
                        st.error(f"Drift Detected in **'{feature}'**")
                    elif 0.1 < psi <= 0.2:
                        st.warning(f"Moderate Drift in **'{feature}'** (PSI > 0.1)")
                    else:
                        st.success(f"No Drift in **'{feature}'**")
        else:
            st.info("Select at least one feature to visualize.")

st.title("Data Drift (evidently)")

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