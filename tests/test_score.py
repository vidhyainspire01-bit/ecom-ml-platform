import os
import sys
import json
import pandas as pd
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mlops", "deployment", "scoring"))


def _train_test_model(tmp_path):
    X = pd.DataFrame({
        "order_count": [1, 5, 2, 8],
        "avg_order_value": [50.0, 200.0, 75.0, 300.0],
        "days_since_signup": [30, 400, 60, 900],
    })
    y = [1, 0, 1, 0]
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    model_dir = os.path.join(tmp_path, "model")
    mlflow.sklearn.save_model(model, model_dir, serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE)
    return tmp_path


def test_run_returns_predictions(tmp_path):
    os.environ["AZUREML_MODEL_DIR"] = str(_train_test_model(tmp_path))
    import score
    score.init()

    payload = json.dumps({"data": [[3, 100.0, 200]], "columns": ["order_count", "avg_order_value", "days_since_signup"]})
    result = json.loads(score.run(payload))
    assert "predictions" in result


def test_run_returns_error_on_missing_column(tmp_path):
    os.environ["AZUREML_MODEL_DIR"] = str(_train_test_model(tmp_path))
    import score
    score.init()

    payload = json.dumps({"data": [[3, 100.0]], "columns": ["order_count", "avg_order_value"]})
    result = json.loads(score.run(payload))
    assert "error" in result