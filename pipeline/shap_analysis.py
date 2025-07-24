import mlflow
import shap
import pandas as pd
from mlflow.tracking import MlflowClient
import numpy as np
import os
import json
import shutil
from datetime import datetime 

def compute_shap_native(
    model_name: str,
    version: str,
    background: pd.DataFrame,
    data: pd.DataFrame
):
    """
    Load a Sklearn‑wrapped XGBClassifier from the Model Registry,
    extract its Booster, and run TreeExplainer in approximate mode.
    """
    model_uri = f"models:/{model_name}/{version}"

    # load the sklearn model
    sk_model = mlflow.sklearn.load_model(model_uri)

    # pull out the native Booster
    booster = sk_model.get_booster()

    # TreeExplainer in interventional (fast) mode
    explainer = shap.TreeExplainer(
        booster,
        data=background,
        feature_perturbation="interventional"
    )

    # compute approximate SHAP values
    shap_values = explainer.shap_values(
        data,
        approximate=True,
        check_additivity=False
    )

    return explainer, shap_values

def generate_latest_shap_summary(model_name: str, sample_data: pd.DataFrame) -> str:
    """
    Automatically compares the two latest model versions and saves SHAP summary JSON.
    Returns the file path to the saved summary.
    """
    client = MlflowClient()
    versions = sorted(client.search_model_versions(f"name='{model_name}'"), key=lambda v: int(v.version), reverse=True)

    if len(versions) < 2:
        raise ValueError("Not enough model versions for comparison.")

    v2 = versions[0].version  # latest
    v1 = versions[1].version  # previous

    # Compute SHAP values
    expl1, shap1 = compute_shap_native(model_name, v1, sample_data, sample_data)
    expl2, shap2 = compute_shap_native(model_name, v2, sample_data, sample_data)

    mean1 = np.abs(shap1).mean(axis=0)
    mean2 = np.abs(shap2).mean(axis=0)
    delta = mean2 - mean1
    features = sample_data.columns.tolist()
    order = np.argsort(delta)[::-1]

    # Construct summary list
    shap_summary = [
        {
            "feature": features[i],
            "mean_v1": float(mean1[i]),
            "mean_v2": float(mean2[i]),
            "delta": float(delta[i]),
        }
        for i in order
    ]

    # Save to timestamped JSON
    out_path = "artifacts/shap_summary.json"
    os.makedirs("artifacts", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(shap_summary, f, indent=2)

    # archive shap_summary for each pipeline run
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = "artifacts/shap_summary_archive"
    os.makedirs(archive_dir, exist_ok=True)

    archive_filename = f"shap_summary_v{v1}_vs_v{v2}_{ts}.json"
    archive_path = os.path.join(archive_dir, archive_filename)
    shutil.copy(out_path, archive_path)

    return out_path