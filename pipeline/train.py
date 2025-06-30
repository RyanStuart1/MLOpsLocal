def train_model():
    import os    
    import pandas as pd
    import mlflow
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from pipeline.metrics import log_confusion_matrix
    from xgboost import XGBClassifier
    from preprocess import preprocess_data
    from evaluate import evaluate_model


    # Creates missing directories if they do not exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("mlruns", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    # Load data
    df = pd.read_csv("data/synthetic_credit_risk.csv")

    # Preprocess: split into train/test sets
    x_train, x_test, y_train, y_test = preprocess_data(df)

    # Train model
#    model = LogisticRegression(max_iter=20000)
#    model = XGBClassifier(
#    n_estimators=200,         # More boosting rounds (default is 100)
#    max_depth=4,              # Controls tree depth (default is 6)
#    learning_rate=0.1,        # Shrinks contribution of each tree
#    subsample=0.8,            # Randomly sample 80% of training data for each tree
#    colsample_bytree=0.8,     # Use 80% of features per tree
#    use_label_encoder=False,
#    eval_metric='logloss'
#    )
#    model.fit(x_train, y_train)

    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.8],
        'colsample_bytree': [0.8]
    }

    # Create base model
    xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
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
    grid_search.fit(x_train, y_train)
    model = grid_search.best_estimator_

    print("Best hyperparameters from GridSearchCV:")
    print(grid_search.best_params_)

    # Evaluate model
    metrics = evaluate_model(model, x_test, y_test)

    # Log everything with MLflow
    mlflow.set_experiment("credit-risk")
    with mlflow.start_run():
#       mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("model", "XGBoost")  # Optional: for tagging model name
        mlflow.log_params(model.get_params())  # Logs all hyperparameters
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("roc_auc", metrics["roc_auc"])
        mlflow.sklearn.log_model(model, "model")

        # Log confusion matrix
        log_confusion_matrix(model, x_test, y_test)
        
    # Save model locally
    joblib.dump(model, "models/model.pkl")
    print(f"Model trained and saved. Accuracy: {metrics['accuracy']:.4f}")
