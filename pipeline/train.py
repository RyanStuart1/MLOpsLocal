def train_model():
    import pandas as pd
    import mlflow
    import joblib
    from sklearn.linear_model import LogisticRegression
    from preprocess import preprocess_data
    from evaluate import evaluate_model

    # Load data
    df = pd.read_csv("data/synthetic_credit_risk.csv")

    # Preprocess: split into train/test sets
    x_train, x_test, y_train, y_test = preprocess_data(df)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    # Evaluate model
    metrics = evaluate_model(model, x_test, y_test)

    # Log everything with MLflow
    mlflow.set_experiment("credit-risk")
    with mlflow.start_run():
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("roc_auc", metrics["roc_auc"])
        mlflow.sklearn.log_model(model, "model")

    # Save model locally
    joblib.dump(model, "models/model.pkl")
    print(f"Model trained and saved. Accuracy: {metrics['accuracy']:.4f}")
