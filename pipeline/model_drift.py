import numpy as np
from scipy.stats import ks_2samp
from evidently import Report
from evidently.presets import DataDriftPreset
from prefect import task
import pandas as pd
import os
import json

# batch drift
def calculate_psi(expected, actual, buckets=10):
    breakpoints = np.linspace(0, 100, buckets + 1)
    expected_perc = np.percentile(expected, breakpoints)

    psi = 0
    for i in range(buckets):
        e_count = np.sum((expected >= expected_perc[i]) & (expected < expected_perc[i+1])) / len(expected)
        a_count = np.sum((actual >= expected_perc[i]) & (actual < expected_perc[i+1])) / len(actual)

        e_count = 0.0001 if e_count == 0 else e_count
        a_count = 0.0001 if a_count == 0 else a_count

        psi += (e_count - a_count) * np.log(e_count / a_count)
    return psi

def ks_test_pvalue(expected, actual):
    _, p_value = ks_2samp(expected, actual)
    return p_value

@task
def run_data_drift_check(reference_path, current_path, output_path):
    # Load data
    reference_data = pd.read_csv(reference_path)
    current_data   = pd.read_csv(current_path)

    # Drop target columns if present
    for df in (reference_data, current_data):
        for col in ["target", "loan_default"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

    # Build & run Evidently preset
    report = Report(metrics=[DataDriftPreset(drift_share=0.2)])
    result = report.run(reference_data, current_data)

    # Save HTML report
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.save_html(output_path)

    # Load Evidently JSON and augment with our KS tests
    summary = json.loads(result.json())

    # define candidate categoricals & numerics
    cat_cols = (
        list(reference_data.select_dtypes(include=["object", "category"]).columns) +
        [c for c in reference_data.select_dtypes(include="number").columns
         if reference_data[c].nunique() <= 10]
    )
    num_cols = [c for c in reference_data.select_dtypes(include="number").columns
                if c not in cat_cols]

    # Build lookup: column -> test name
    test_mapping = {}
    for col in num_cols:
        test_mapping[col] = f"KolmogorovSmirnovTest(column={col})"
    for col in cat_cols:
        test_mapping[col] = f"ChiSquareTest(column={col})"

    # Attach test name under a new 'test' field in each drift metric
    for m in summary.get("metrics", []):
        mid = m.get("metric_id", "")
        if mid.startswith("ValueDrift"):
            col = mid[mid.find("column=") + len("column=") : -1]
            if col in test_mapping:
                m["test"] = test_mapping[col]
    summary.pop("tests", None)

    # Write augmented JSON to artifacts
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/drift_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Return path to HTML report
    return output_path

def has_significant_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    psi_thresh: float = 0.2,
    ks_thresh: float = 0.05
) -> list:
    """
    Check drift across numeric features using PSI and KS test.

    Returns:
        A list of (feature, psi_value, ks_p_value) tuples where drift is significant.
    """
    numeric_cols = reference_df.select_dtypes(include="number").columns
    drifted_features = []

    for feature in numeric_cols:
        if feature not in current_df.columns:
            continue
        expected = reference_df[feature].dropna()
        actual = current_df[feature].dropna()

        if len(expected) < 10 or len(actual) < 10:
            continue  # skip unstable small samples

        psi = calculate_psi(expected, actual)
        ks_p = ks_test_pvalue(expected, actual)

        if psi > psi_thresh or ks_p < ks_thresh:
            drifted_features.append((feature, psi, ks_p))

    return drifted_features
