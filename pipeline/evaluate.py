from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

def evaluate_model(model, x, y):
    """
    Return accuracy and other metrics.
    """
    y_pred = model.predict(x)
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_pred),
        "report": classification_report(y, y_pred, output_dict=True)
    }
    return metrics
