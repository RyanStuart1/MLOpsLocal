import os
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import mlflow

def log_confusion_matrix(model, x_test, y_test):
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
        plt.close()
        mlflow.log_artifact(save_path)
