import os
import tempfile
import matplotlib.pyplot as plt
import mlflow.tracking
import seaborn as sns
from sklearn.metrics import confusion_matrix
import mlflow

def log_confusion_matrix(model, x_test, y_test, save_local=True):
    preds = model.predict(x_test)
    cm = confusion_matrix(y_test, preds)

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Default", "Default"],
                yticklabels=["No Default", "Default"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    # Save and log to MLflow using temp dir
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "confusion_matrix.png")
        plt.savefig(save_path)
        mlflow.log_artifact(save_path)

    # Optional: also save locally for Streamlit
    if save_local:
        os.makedirs("artifacts", exist_ok=True)
        local_path = "artifacts/confusion_matrix.png"
        plt.savefig(local_path, dpi=200)

    plt.close()

def get_model_metrics():
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("credit-risk")

    if experiment is None:
        return None
    
    runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"], max_results=1)
    if not runs:
        return None
    
    runs = runs[0]
    return {
        "accuracy": runs.data.metrics.get("accuracy"),
        "roc_auc": runs.data.metrics.get("roc_auc")
    }