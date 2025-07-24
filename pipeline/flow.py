from prefect import flow, task
from pipeline.train import train_model 
from pipeline.model_drift import run_data_drift_check
from pipeline.shap_analysis import generate_latest_shap_summary
import pandas as pd
import json
import os
from typing import Optional

@task
def run_training(data_path: str = "data/synthetic_credit_risk.csv", seed: Optional[int] = None):
    return train_model(data_path, seed=seed)

# creates pdf in model monitoring page
@flow(name="credit_risk_pipeline")
def credit_risk_pipeline(data_path: str = "data/synthetic_credit_risk.csv", seed: Optional[int] = None):
    run_training(data_path, seed=seed)
    drift_report_path = run_data_drift_check(
    reference_path="data/synthetic_credit_risk.csv",
    current_path="artifacts/prediction_input_sample.csv",
    output_path="reports/drift_report.html"
    )
    return drift_report_path

@flow(name="drift_aware_pipeline")
def drift_aware_pipeline(
    data_path: str = "data/synthetic_credit_risk.csv",
    reference_path: str = "data/synthetic_credit_risk.csv",
    current_path:   str = "artifacts/prediction_input_sample.csv",
    drift_html:     str = "artifacts/drift_report.html",
    drift_json:     str = "artifacts/drift_summary.json",
    drift_share_threshold: float = 0.2,
    model_name: str = "credit-risk-model"
):

    # Generate Evidently’s drift report
    run_data_drift_check(
        reference_path=reference_path,
        current_path=current_path,
        output_path=drift_html,
    )

    # Load the JSON summary
    with open(drift_json, "r") as f:
        summary = json.load(f)

    # Extract the overall drift share from the DriftedColumnsCount metric
    overall_metric = next(
        (m for m in summary["metrics"]
         if m["metric_id"].startswith("DriftedColumnsCount")),
        None
    )
    if overall_metric is None:
        raise ValueError("No DriftedColumnsCount metric found in drift_summary.json")

    overall_share = overall_metric["value"]["share"]

    # Load sample data for SHAP
    sample_data = pd.read_csv(current_path)

    # Always generate SHAP summary with archive
    generate_latest_shap_summary(model_name=model_name, sample_data=sample_data)

    # Load previous seed if it exists
    seed: Optional[int] = None
    seed_path = "artifacts/seed.json"
    if os.path.exists(seed_path):
        with open(seed_path) as f:
            seed_data = json.load(f)
            seed = seed_data.get("random_seed")


    # Compare to threshold and retrain if exceeded
    if overall_share > drift_share_threshold:
        print(f"Overall drift {overall_share:.2%} > {drift_share_threshold:.2%}, retraining…")
        model_path = credit_risk_pipeline(data_path, seed=seed)
        return {
            "retrained": True,
            "model_path": model_path,
            "drift_share": overall_share,
            "drift_report": drift_html,
        }
    else:
        print(f"Overall drift {overall_share:.2%} ≤ {drift_share_threshold:.2%}, skipping retrain.")
        return {
            "retrained": False,
            "drift_share": overall_share,
            "drift_report": drift_html,
        }
    

if __name__ == "__main__":
    drift_aware_pipeline()