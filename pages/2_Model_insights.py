import streamlit as st
import os
from PIL import Image
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
import numpy as np
from pipeline.metrics import get_model_metrics

st.title("Model Insights")
st.set_page_config(layout="wide")

client    = MlflowClient()
model_name = "credit-risk-model"  # must match model version
# pull all model versions, pick the highest one by version number
all_versions   = client.search_model_versions(f"name='{model_name}'")
latest_version = max(all_versions, key=lambda v: int(v.version))
run_id         = latest_version.run_id
run     = client.get_run(run_id)
metrics = run.data.metrics
latest = {
    "val_accuracy":  metrics.get("val_accuracy"),
    "val_roc_auc":   metrics.get("val_roc_auc"),
    "test_accuracy": metrics.get("test_accuracy"),
    "test_roc_auc":  metrics.get("test_roc_auc"),
}



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
        fig, ax = plt.subplots(figsize=(4,3))
        ax.bar(labels, values, width=0.6)
        ax.set_ylim(0,1)
        ax.set_ylabel("Score")
        ax.set_title("Latest Model Performance")
        for i, v in enumerate(values):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")
        st.pyplot(fig)
    else:
        st.info("Run a training flow first to generate metrics.")