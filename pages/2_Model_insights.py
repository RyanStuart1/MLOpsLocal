import streamlit as st
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pipeline.metrics import get_model_metrics

st.title("📊 Model Insights")
st.set_page_config(layout="wide")

# Adjust column width ratios to give more space
col1, col2, col3 = st.columns([3, 2, 3])

with col1:
    st.subheader("Confusion Matrix")
    cm_path = "artifacts/confusion_matrix.png"

    if os.path.exists(cm_path):
        st.image(Image.open(cm_path), caption="Latest confusion matrix from MLflow", use_container_width=True)
    else:
        st.warning("Confusion matrix not found. Run the pipeline first.")

with col2:
    st.subheader("Latest Metrics")
    metrics = get_model_metrics()

    if metrics:
        st.metric("Accuracy", f"{metrics['accuracy']:.2f}")
        st.metric("ROC AUC", f"{metrics['roc_auc']:.2f}")
    else:
        st.warning("No metrics found in MLflow.")

with col3:
    if metrics:
        st.subheader("Score Chart")
        fig, ax = plt.subplots(figsize=(6, 4))  # Bigger chart
        ax.bar(["Accuracy", "ROC AUC"], [metrics["accuracy"], metrics["roc_auc"]],
               color=["#00fe8c", "#8D4FFF"], width=0.5)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_title("Model Performance")
        for i, v in enumerate([metrics["accuracy"], metrics["roc_auc"]]):
            ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
        st.pyplot(fig)
        st.caption("Latest Model Metrics from MLflow")
