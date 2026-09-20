"""
Real unit tests against the CURRENT scoring architecture:
mlops/deployment/scoring/common/ (shared) + churn_model/ (per-model).
Replaces the old test_score.py, which tested a flat score.py that no
longer exists after the common/ refactor.
"""
import os
import sys
import json
import pandas as pd
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from mlflow.models.signature import infer_signature

SCORING_ROOT = os.path.join(os.path.dirname(__file__), "..", "mlops", "deployment", "scoring")


def _train_test_model(tmp_path):
    X = pd.DataFrame({
        "order_count": [1, 5, 2, 8],
        "avg_order_value": [50.0, 200.0, 75.0, 300.0],
        "days_since_signup": [30, 400, 60, 900],
    })
    y = [1, 0, 1, 0]
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    signature = infer_signature(X, model.predict(X))
    model_dir = os.path.join(tmp_path, "model")
    mlflow.sklearn.save_model(model, model_dir, signature=signature,
                               serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE)
    return tmp_path


def _fresh_import(module_name, path):
    """Reload from a specific path each time -- tests run in the same
    process, and Python caches modules by name, which would otherwise
    leak state between churn_model and high_frequency_lead imports."""
    for m in list(sys.modules):
        if m in ("main", "preprocessing", "model_loader", "inference_runner") or m.startswith("common"):
            del sys.modules[m]
    if path not in sys.path:
        sys.path.insert(0, path)
    return __import__(module_name)


def test_churn_model_run_returns_predictions(tmp_path):
    os.environ["AZUREML_MODEL_DIR"] = str(_train_test_model(tmp_path))
    main = _fresh_import("main", os.path.join(SCORING_ROOT, "churn_model"))
    main.init()

    payload = json.dumps({"data": [[3, 100.0, 200]], "columns": ["order_count", "avg_order_value", "days_since_signup"]})
    result = json.loads(main.run(payload))
    assert "predictions" in result


def test_churn_model_run_handles_all_integer_payload(tmp_path):
    """The dtype-casting fix -- an all-integer payload must NOT crash
    with a schema enforcement error."""
    os.environ["AZUREML_MODEL_DIR"] = str(_train_test_model(tmp_path))
    main = _fresh_import("main", os.path.join(SCORING_ROOT, "churn_model"))
    main.init()

    payload = json.dumps({"data": [[3, 100, 200]], "columns": ["order_count", "avg_order_value", "days_since_signup"]})
    result = json.loads(main.run(payload))
    assert "predictions" in result


def test_churn_model_run_returns_error_on_missing_column(tmp_path):
    os.environ["AZUREML_MODEL_DIR"] = str(_train_test_model(tmp_path))
    main = _fresh_import("main", os.path.join(SCORING_ROOT, "churn_model"))
    main.init()

    payload = json.dumps({"data": [[3, 100.0]], "columns": ["order_count", "avg_order_value"]})
    result = json.loads(main.run(payload))
    assert "error" in result