"""
MLOPS BOUNDARY — STEP 1.
Input: a run_id DS hands off ("this one's ready").
Output: mlops/models/<model_name>/promotion_config.yml, including which
registered model NAME + VERSION that run produced — looked up
automatically via MLflow's search_model_versions, not typed in by hand.

This does NOT retrain and does NOT touch the registered model itself.
It only records what DS validated, so validate_and_promote.py knows
exactly which artifact to re-score.
"""
import argparse
import json
import os
import yaml
from mlflow.tracking import MlflowClient


def generate_config(run_id, model_name, registered_model_name, metric_name,
                     min_threshold, features, target_column, model_type, out_dir):
    client = MlflowClient()
    run = client.get_run(run_id)
    params = run.data.params
    metrics = run.data.metrics

    if metric_name not in metrics:
        raise ValueError(f"Run {run_id} has no metric '{metric_name}'. Available: {list(metrics.keys())}")

    # Find which registered model version this run actually produced —
    # not assumed, looked up.
    versions = client.search_model_versions(f"run_id='{run_id}'")
    if not versions:
        raise ValueError(
            f"Run {run_id} has no registered model version. "
            f"Has DS actually called mlflow.register_model() on this run?"
        )
    registered_version = versions[0].version

    hyperparams = {
        k: (int(v) if v.isdigit() else float(v) if _is_float(v) else v)
        for k, v in params.items()
        if k not in ("churn_window_days", "feature_asset")
    }

    config = {
        "model_name": model_name,
        "mlflow_run_id": run_id,
        "registered_model_name": registered_model_name,
        "registered_model_version": registered_version,
        "features": ",".join(features),
        "target_column": target_column,
        "model_type": model_type,
        "hyperparameters": json.dumps(hyperparams),
        "metric_name": metric_name,
        "min_threshold": min_threshold,
        "observed_metric_at_handoff": metrics[metric_name],
    }

    model_dir = os.path.join(out_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    # Deliberately NOT config.yml — that file is Flow 2's retrain input
    # (raw_data_path, hyperparameters). This is Flow 1's promotion
    # record. Same folder, different file, so neither flow's script
    # can clobber the other's config.
    config_path = os.path.join(model_dir, "promotion_config.yml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)

    print(f"Wrote {config_path}")
    print(yaml.dump(config, sort_keys=False))
    return config_path


def _is_float(v):
    try:
        float(v)
        return True
    except ValueError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--model_name", required=True, help="Platform-internal name, e.g. churn_model")
    parser.add_argument("--registered_model_name", required=True, help="What DS registered it as, e.g. ecom-churn_model")
    parser.add_argument("--metric_name", default="roc_auc")
    parser.add_argument("--min_threshold", type=float, required=True)
    parser.add_argument("--features", default="recency_days,order_count,avg_order_value,days_since_signup")
    parser.add_argument("--target_column", default="churned")
    parser.add_argument("--model_type", default="classifier")
    parser.add_argument("--out_dir", default="mlops/models")
    args = parser.parse_args()

    generate_config(
        run_id=args.run_id, model_name=args.model_name,
        registered_model_name=args.registered_model_name,
        metric_name=args.metric_name, min_threshold=args.min_threshold,
        features=args.features.split(","), target_column=args.target_column,
        model_type=args.model_type, out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
