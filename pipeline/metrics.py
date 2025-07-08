import os
import tempfile
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient
import seaborn as sns
from sklearn.metrics import confusion_matrix
import mlflow

def log_confusion_matrix(
    model,
    x,
    y,
    save_local: bool = True,
    name: str = "confusion_matrix"
):
    """
    Compute & log a confusion matrix for dataset (x, y).

    Parameters:
    - model: The trained model with a .predict method.
    - x: Features DataFrame
    - y: True labels.
    - save_local: Whether to save a local copy under artifacts/.
    - name: Prefix for the saved artifact (e.g. "confusion_matrix_val.png", "confusion_matrix_test.png").
    """
    # Generate predictions and compute the confusion matrix
    preds = model.predict(x)
    cm = confusion_matrix(y, preds)

    # Create the plot
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Default", "Default"],
        yticklabels=["No Default", "Default"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(name.replace('_', ' ').title())

    # Log to MLflow artifact
    artifact_dir = "confusion_matrices"
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = os.path.join(tmpdir, f"{name}.png")
        plt.savefig(tmp_path, bbox_inches="tight")
        mlflow.log_artifact(tmp_path, artifact_path=artifact_dir)

    # Optionally save locally for Streamlit or other use
    if save_local:
        local_dir = os.path.join("artifacts", artifact_dir)
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, f"{name}.png")
        plt.savefig(local_path, dpi=200, bbox_inches="tight")

    plt.close()


def get_model_metrics(registered_model_name: str = "credit-risk-model"):
    """
    Returns a dict with the four metrics for the latest registered version
    of `registered_model_name`.  If nothing is found, returns None.
    """
    client = MlflowClient()
    # find all versions for this registered model
    try:
        all_versions = client.search_model_versions(f"name='{registered_model_name}'")
    except Exception:
        return None

    if not all_versions:
        return None

    # pick the most recent model version / run
    latest = max(all_versions, key=lambda v: int(v.version))
    run_id = latest.run_id

    # fetch the most recent model version / run
    run = client.get_run(run_id)
    metrics   = run.data.metrics

    # 4) return exactly the four metrics from train.py
    return {
        "val_accuracy":  metrics.get("val_accuracy"),
        "val_roc_auc":   metrics.get("val_roc_auc"),
        "test_accuracy": metrics.get("test_accuracy"),
        "test_roc_auc":  metrics.get("test_roc_auc"),
    }
