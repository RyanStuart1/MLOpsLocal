import mlflow
import shap
import pandas as pd

def compute_shap_native(
    model_name: str,
    version: str,
    background: pd.DataFrame,
    data: pd.DataFrame
):
    """
    Load a Sklearn‑wrapped XGBClassifier from the Model Registry,
    extract its Booster, and run TreeExplainer in approximate mode.
    """
    model_uri = f"models:/{model_name}/{version}"

    # 1) load the sklearn model
    sk_model = mlflow.sklearn.load_model(model_uri)

    # 2) pull out the native Booster
    booster = sk_model.get_booster()

    # 3) TreeExplainer in 'interventional' (fast) mode
    explainer = shap.TreeExplainer(
        booster,
        data=background,
        feature_perturbation="interventional"
    )

    # 4) compute approximate SHAP values
    shap_values = explainer.shap_values(
        data,
        approximate=True,
        check_additivity=False
    )

    return explainer, shap_values