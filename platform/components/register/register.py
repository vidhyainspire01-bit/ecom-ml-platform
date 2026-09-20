import argparse
import json
import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_input", type=str, required=True)
    parser.add_argument("--metrics_input", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--git_sha", type=str, required=True)
    parser.add_argument("--data_version", type=str, required=True)
    parser.add_argument("--registration_info", type=str, required=True)
    args = parser.parse_args()

    with open(args.metrics_input) as f:
        metrics = json.load(f)

    if not metrics.get("passed"):
        raise SystemExit("Refusing to register a model that failed its evaluation gate.")

    registered_name = f"ecom-{args.model_name}"

    model = mlflow.sklearn.load_model(args.model_input)
    model_info = mlflow.sklearn.log_model(
        model, "model",
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
        extra_pip_requirements=["setuptools<81"],
    )

    result = mlflow.register_model(
        model_uri=model_info.model_uri,
        name=registered_name,
    )

    client = MlflowClient()
    client.set_model_version_tag(registered_name, result.version, "git_sha", args.git_sha)
    client.set_model_version_tag(registered_name, result.version, "data_version", args.data_version)
    client.set_model_version_tag(registered_name, result.version, "metric_name", metrics["metric_name"])
    client.set_model_version_tag(registered_name, result.version, "metric_score", str(metrics["score"]))

    info = {
        "registered_name": registered_name,
        "version": result.version,
        "git_sha": args.git_sha,
        "data_version": args.data_version,
        "metric": metrics,
    }
    os.makedirs(os.path.dirname(args.registration_info), exist_ok=True)
    with open(args.registration_info, "w") as f:
        json.dump(info, f)

    print(f"Registered {registered_name} v{result.version} (git_sha={args.git_sha}, data_version={args.data_version})")


if __name__ == "__main__":
    main()