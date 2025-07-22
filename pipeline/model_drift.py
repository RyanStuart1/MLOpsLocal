import numpy as np
from scipy.stats import ks_2samp
from evidently import Report
from evidently.presets import DataDriftPreset
from prefect import task
import pandas as pd
import os
import json
import tempfile

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
    report = Report(metrics=[DataDriftPreset()])
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

def has_significant_drift(reference_df, current_df) -> list:
    """
    Uses Evidently under the hood via run_data_drift_check():
    • writes CSVs into a temp dir
    • runs the Evidently preset (which writes artifacts/drift_summary.json + HTML)
    • parses and returns [(col, share, share)] for each drifted feature
    """
    drifted = []
    with tempfile.TemporaryDirectory() as tmpdir:
        # serialize the two snapshots
        ref_path  = os.path.join(tmpdir, "ref.csv")
        curr_path = os.path.join(tmpdir, "curr.csv")
        reference_df.to_csv(ref_path, index=False)
        current_df.to_csv(curr_path, index=False)

        # run the Evidently drift check task
        # you might want to point HTML somewhere permanent,
        # e.g. "artifacts/drift_report.html"
        run_data_drift_check(
            reference_path=ref_path,
            current_path=curr_path,
            output_path="artifacts/drift_report.html"
        )

        # load the resulting JSON summary
        with open("artifacts/drift_summary.json", "r") as r:
            summary = json.load(r)

        # extract any ValueDrift columns where drift_detected==True
        for m in summary.get("metrics", []):
            if not m.get("metric_id", "").startswith("ValueDrift"):
                continue

            col = m["metric_id"].split("column=")[1].rstrip(")")
            val = m.get("value")
            if not isinstance(val, dict):
                continue

            if val.get("drift_detected", False):
                share = val.get("drift_share", 0.0)
                drifted.append((col, share, share))
    # when the context manager exits, tmpdir and its files are auto‑deleted
    return drifted