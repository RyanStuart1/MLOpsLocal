import numpy as np
from scipy.stats import ks_2samp
from evidently import Report
from evidently.presets import DataDriftPreset
from prefect import task
import pandas as pd
import os

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

    # load
    ref_df = pd.read_csv(reference_path)
    cur_df = pd.read_csv(current_path)
    # drop target if present
    for df in (ref_df, cur_df):
        if "target" in df.columns:
            df.drop(columns=["target"], inplace=True)

    # build & run
    report = Report(metrics=[DataDriftPreset()])
    result = report.run(reference_data=ref_df, current_data=cur_df)

    # save locally
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.save_html(output_path)

    return output_path