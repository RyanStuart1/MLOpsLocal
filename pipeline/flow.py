from prefect import flow, task
from pipeline.train import train_model  # assume you wrap training in a function
from pipeline.model_drift import run_data_drift_check

@task
def run_training():
    return train_model()

@flow(name="credit_risk_pipeline")
def credit_risk_pipeline():
    run_training()
    drift_report_path = run_data_drift_check(
        reference_path="data/synthetic_credit_risk.csv",
        current_path="artifacts/prediction_input_sample.csv",
        output_path="reports/drift_report.html"
    )
    return drift_report_path

if __name__ == "__main__":
    credit_risk_pipeline()
