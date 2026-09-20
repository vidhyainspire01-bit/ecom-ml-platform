import os
import mlflow.pyfunc


def load_model():
    model_root = os.getenv("AZUREML_MODEL_DIR")
    for root, _, files in os.walk(model_root):
        if "MLmodel" in files:
            return mlflow.pyfunc.load_model(root)
    return mlflow.pyfunc.load_model(model_root)