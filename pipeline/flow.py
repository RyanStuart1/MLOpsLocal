from prefect import flow, task
from pipeline.train import train_model  # assume you wrap training in a function
from pipeline.model_drift import run_data_drift_check, has_significant_drift
import pandas as pd
import json

@task
def run_training(data_path: str = "data/synthetic_credit_risk.csv"):
    return train_model(data_path)

@flow(name="credit_risk_pipeline")
def credit_risk_pipeline(data_path: str = "data/synthetic_credit_risk.csv"):
    run_training(data_path)

    return "Model trained and saved"


@flow(name="drift_aware_pipeline")
def drift_aware_pipeline(
    data_path: str = "data/synthetic_credit_risk.csv",
    reference_path: str = "data/synthetic_credit_risk.csv",
    current_path:   str = "artifacts/prediction_input_sample.csv",
    drift_html:     str = "artifacts/drift_report.html",
    drift_json:     str = "artifacts/drift_summary.json",
):
    # Run Evidently’s DataDriftPreset (writes JSON + HTML)
    run_data_drift_check(
        reference_path=reference_path,
        current_path=current_path,
        output_path=drift_html,
    )

    # Load the JSON summary
    with open(drift_json, "r") as f:
        summary = json.load(f)

    # Find all features where drift_detected==True
    drifted_features = []
    for metric in summary.get("metrics", []):
        mid = metric.get("metric_id", "")
        if not mid.startswith("ValueDrift"):
            continue

        # pull the payload from "metric", not "value"
        payload = metric.get("metric", {})
        if not isinstance(payload, dict):
            continue

        if payload.get("drift_detected", False):
            feature = mid.split("column=")[1].rstrip(")")
            share   = payload.get("drift_share", 0.0)
            drifted_features.append((feature, share))        

    # Retrain if any drifted features
    if drifted_features:
        print(f"Drift detected in {len(drifted_features)} features, retraining…")
        model_path = credit_risk_pipeline(data_path)
        return {
            "retrained": True,
            "model_path": model_path,
            "drifted_features": drifted_features,
            "drift_report": drift_html,
        }
    else:
        print("No significant drift detected.")
        return {
            "retrained": False,
            "drifted_features": [],
            "drift_report": drift_html,
        }
if __name__ == "__main__":
    credit_risk_pipeline()
