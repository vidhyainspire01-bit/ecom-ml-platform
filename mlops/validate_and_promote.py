"""
MLOPS BOUNDARY — STEP 2. Triggered by CI/CD after promotion_config.yml is pushed.

Loads the ALREADY-REGISTERED model DS handed off (by name + version from
config.yml) and re-scores it against a platform-controlled validation
set — DS's own reported metric is never trusted blindly. If it passes,
re-tags the SAME model version to the next stage. No training happens
here at all — this is promotion, not the retrain pipeline.

Also runs a DATA QUALITY CHECK on the validation set itself, at the
promotion boundary -- reuses the exact same run_checks() function
platform/components/data_quality/check.py uses at training time. This
is a genuinely separate check from that one: it validates the data
used to APPROVE a promotion decision, not training input.
"""
import argparse
import json
import os
import sys
import math
import yaml
import pandas as pd
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "platform", "components", "data_quality"))
from check import run_checks  # noqa: E402

METRIC_FNS = {
    "accuracy": accuracy_score,
    "f1": f1_score,
    "roc_auc": roc_auc_score,
    "rmse": lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred)),
}

STAGE_ORDER = ["dev", "staging", "production"]


def load_config(model_name):
    path = f"mlops/models/{model_name}/promotion_config.yml"
    with open(path) as f:
        return yaml.safe_load(f)


def validate_and_promote(model_name, validation_data_path, target_stage):
    config = load_config(model_name)
    registered_name = config["registered_model_name"]
    version = str(config["registered_model_version"])
    metric_name = config["metric_name"]
    min_threshold = float(config["min_threshold"])
    features = config["features"].split(",")
    target_column = config["target_column"]

    client = MlflowClient()
    mv = client.get_model_version(registered_name, version)
    current_stage = mv.tags.get("stage", "dev")

    expected_prev = STAGE_ORDER[STAGE_ORDER.index(target_stage) - 1]
    if current_stage != expected_prev:
        raise SystemExit(
            f"Refusing to promote {registered_name} v{version} to '{target_stage}': "
            f"current stage is '{current_stage}', expected '{expected_prev}'."
        )

    val_df = pd.read_csv(validation_data_path)

    dq_result = run_checks(val_df, features + [target_column], max_null_rate=0.05, min_row_count=20)
    print(f"Validation-set data quality check: {dq_result}")
    if not dq_result["passed"]:
        raise SystemExit(
            f"Refusing to promote {registered_name} v{version} to '{target_stage}': "
            f"validation data failed quality gate: {dq_result['issues']}"
        )

    model = mlflow.pyfunc.load_model(f"models:/{registered_name}/{version}")
    X = val_df[features]
    y_true = val_df[target_column]

    preds = model.predict(X)
    metric_fn = METRIC_FNS[metric_name]
    score = float(metric_fn(y_true, preds))
    passed = score >= min_threshold if metric_name != "rmse" else score <= min_threshold

    result = {
        "registered_model_name": registered_name,
        "version": version,
        "metric_name": metric_name,
        "independent_score": score,
        "ds_reported_score": config.get("observed_metric_at_handoff"),
        "min_threshold": min_threshold,
        "passed": passed,
        "promoted_to": target_stage if passed else None,
    }
    print(json.dumps(result, indent=2))

    if not passed:
        raise SystemExit(f"Independent validation FAILED: {score} vs threshold {min_threshold}. Not promoting.")

    client.set_model_version_tag(registered_name, version, "stage", target_stage)
    print(f"Promoted {registered_name} v{version}: {current_stage} -> {target_stage}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, help="Platform-internal name, e.g. churn_model")
    parser.add_argument("--validation_data", required=True, help="Path to a held-out CSV, platform-controlled, NOT DS's own training data")
    parser.add_argument("--target_stage", required=True, choices=["staging", "production"])
    args = parser.parse_args()
    validate_and_promote(args.model_name, args.validation_data, args.target_stage)


if __name__ == "__main__":
    main()