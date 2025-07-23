def train_model(data_path: str = "data/synthetic_credit_risk.csv"):
    import os    
    import pandas as pd
    import numpy as np
    import mlflow
    from mlflow.models.signature import infer_signature
    import joblib
    from sklearn.model_selection import GridSearchCV
    from pipeline.metrics import log_confusion_matrix
    from xgboost import XGBClassifier
    from pipeline.preprocess import preprocess_data
    from pipeline.evaluate import evaluate_model
    import random
    import hashlib


    # Creates missing directories if they do not exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("mlruns", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    seed = random.randint(1, 10000)
    random.seed(seed)
    np.random.seed(seed)
    # Load data
    df = pd.read_csv(data_path)

    # Preprocess: split into train/test sets
    x_train, x_val, x_test, y_train, y_val, y_test = preprocess_data(df, random_state=seed)

    # Log a hash of the training features to track data version and ensure reproducibility
    train_hash = hashlib.sha256(pd.util.hash_pandas_object(x_train, index=True).values).hexdigest()

    # Save training sample for drift reference
    x_train.to_csv("artifacts/train_features_sample.csv", index=False)

    # Save test sample as simulated inference input
    x_test.to_csv("artifacts/prediction_input_sample.csv", index=False)

    x_val.to_csv("artifacts/validation_set_sample.csv", index=False)

    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
    }

    # Create base model
    xgb = XGBClassifier(
        eval_metric='logloss',
        random_state=seed,
        n_jobs=-1, 
    )

    # Grid search
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=3,
        verbose=1,
        n_jobs=-1
    )

    fit_params = {"eval_set": [(x_val, y_val)], "verbose": False}
    grid_search.fit(x_train, y_train, **fit_params)
    model = grid_search.best_estimator_

    print("Best hyperparameters from GridSearchCV:", grid_search.best_params_)

    # Evaluate model
    val_metrics = evaluate_model(model, x_val, y_val)
    test_metrics = evaluate_model(model, x_test, y_test)

    # Log everything with MLflow
    mlflow.set_experiment("credit-risk")
    with mlflow.start_run():
        mlflow.log_param("model", "XGBoost")  # Optional: for tagging model name
        mlflow.log_params(grid_search.best_params_)  # Logs best hyperparameters
        mlflow.log_param("random_seed", seed)
        mlflow.log_param("x_train_hash", train_hash)

        mlflow.log_metric("val_accuracy", val_metrics["accuracy"])
        mlflow.log_metric("val_roc_auc", val_metrics["roc_auc"])

        mlflow.log_metric("test_accuracy", test_metrics["accuracy"])
        mlflow.log_metric("test_roc_auc", test_metrics["roc_auc"])

        # New: Log dataset statistics
        mlflow.log_param("n_total_rows", len(df))
        mlflow.log_param("train_size",    len(x_train))
        mlflow.log_param("val_size",      len(x_val))
        mlflow.log_param("test_size",     len(x_test))

        # Register the final model
        input_example = x_test.head(1)
        signature     = infer_signature(x_test, model.predict(x_test))
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name="credit-risk-model",
            input_example=input_example,
            signature=signature
        )


        # Log confusion matrix
        log_confusion_matrix(model, x_test, y_test, save_local=True, name="confusion_matrix_test")
        log_confusion_matrix(model, x_val, y_val, save_local=True, name="confusion_matrix_val")
        
    # Save model locally in pkl file
    joblib.dump(model, "models/model.pkl")
    print(f"Model trained and saved. Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Model trained and saved. Validation Accuracy: {val_metrics['accuracy']:.4f}")

    return model