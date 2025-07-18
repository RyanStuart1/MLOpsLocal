from prefect import flow, task
from pipeline.train import train_model  # assume you wrap training in a function
from pipeline.model_drift import run_data_drift_check, has_significant_drift
import pandas as pd

@task
def run_training(data_path: str = "data/synthetic_credit_risk.csv"):
    return train_model(data_path)

@flow(name="credit_risk_pipeline")
def credit_risk_pipeline(data_path: str = "data/synthetic_credit_risk.csv"):
    run_training(data_path)
    drift_report_path = run_data_drift_check(
        reference_path="data/synthetic_credit_risk.csv",
        current_path="artifacts/prediction_input_sample.csv",
        output_path="reports/drift_report.html"
    )
    return drift_report_path


@flow(name="drift_aware_pipeline")
def drift_aware_pipeline(data_path: str = "data/synthetic_credit_risk.csv"):
    # Load reference and current inference data
    reference_df = pd.read_csv("data/synthetic_credit_risk.csv")
    current_df = pd.read_csv("artifacts/prediction_input_sample.csv")

    # Check for drifted features
    drifted = has_significant_drift(reference_df, current_df)

    if drifted:
        print("Significant drift detected:")
        for f, psi, ks in drifted:
            print(f" - {f}: PSI={psi:.3f}, KS p={ks:.3f}")
        report_path = credit_risk_pipeline(data_path)
        return {"retrained": True, "report": report_path, "drifted_features": drifted}
    else:
        print("No significant drift detected.")
        return {"retrained": False, "drifted_features": []}

if __name__ == "__main__":
    credit_risk_pipeline()
