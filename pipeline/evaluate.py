from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

def evaluate_model(model, x_test, y_test):
    """
    Return accuracy and other metrics.
    """
    y_pred = model.predict(x_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True)
    }
    return metrics
