import argparse
import json
import os
import math
import pandas as pd
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, roc_auc_score


METRIC_FNS = {
    "accuracy": accuracy_score,
    "f1": f1_score,
    "roc_auc": roc_auc_score,
    "rmse": lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred)),
}


def score_and_gate(y_true, y_pred, metric_name: str, min_threshold: float) -> dict:
    """Pure logic, no file I/O, no model loading -- this is what
    tests/test_evaluate.py actually exercises."""
    metric_fn = METRIC_FNS[metric_name]
    score = float(metric_fn(y_true, y_pred))
    passed = score >= min_threshold if metric_name != "rmse" else score <= min_threshold
    return {
        "metric_name": metric_name,
        "score": score,
        "min_threshold": min_threshold,
        "passed": passed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_input", type=str, required=True)
    parser.add_argument("--clean_data", type=str, required=True)
    parser.add_argument("--target_column", type=str, required=True)
    parser.add_argument("--metric_name", type=str, required=True, choices=list(METRIC_FNS))
    parser.add_argument("--min_threshold", type=float, required=True)
    parser.add_argument("--metrics_output", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(args.clean_data, "clean.csv"))
    X = df.drop(columns=[args.target_column])
    y_true = df[args.target_column]

    model = mlflow.sklearn.load_model(args.model_input)

    if args.metric_name == "roc_auc":
        y_pred = model.predict_proba(X)[:, 1]
    else:
        y_pred = model.predict(X)

    result = score_and_gate(y_true, y_pred, args.metric_name, args.min_threshold)
    passed = result["passed"]
    score = result["score"]

    os.makedirs(os.path.dirname(args.metrics_output), exist_ok=True)
    with open(args.metrics_output, "w") as f:
        json.dump(result, f)

    print(json.dumps(result))
    if not passed:
        raise SystemExit(f"Gate failed: {args.metric_name}={score} vs threshold {args.min_threshold}")


if __name__ == "__main__":
    main()