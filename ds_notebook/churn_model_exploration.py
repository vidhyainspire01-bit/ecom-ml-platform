"""
DS-SIDE — simulates a data scientist's Azure ML notebook.

Deliberately does NOT touch raw transactions or curation logic anymore.
DS's job starts at the feature store: pull a named, versioned feature
table, train, register. No RFM math lives here — that's platform code
DS never re-derives. This is what "onboarding a new use case" looks
like once the feature store exists: DS's entire input is one data
asset reference.
"""
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

N_ESTIMATORS = 200
MAX_DEPTH = 6


def train_and_register(features_df, feature_asset_ref, model_name="ecom-churn_model",
                        experiment_name="customer_churn_exploration"):
    mlflow.set_experiment(experiment_name)

    X = features_df[["order_count", "avg_order_value", "days_since_signup"]]
    y = features_df["churned"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    with mlflow.start_run() as run:
        model = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1]
        acc = accuracy_score(y_val, preds)
        auc = roc_auc_score(y_val, probs)

        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("max_depth", MAX_DEPTH)
        mlflow.log_param("feature_asset", feature_asset_ref)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", auc)

        signature = infer_signature(X_train, model.predict(X_train))
        model_info = mlflow.sklearn.log_model(
            model, "model", signature=signature, input_example=X_train.head(3),
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
        )

        result = mlflow.register_model(model_uri=model_info.model_uri, name=model_name)
        client = mlflow.tracking.MlflowClient()
        client.set_model_version_tag(model_name, result.version, "stage", "dev")
        client.set_model_version_tag(model_name, result.version, "feature_asset", feature_asset_ref)

        print(f"\nRUN_ID: {run.info.run_id}")
        print(f"Trained on feature asset: {feature_asset_ref}")
        print(f"Registered: {model_name} v{result.version}  tag=stage:dev")
        print(f"accuracy={acc:.4f}  roc_auc={auc:.4f}")
        return run.info.run_id, result.version


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_data", type=str, required=True)
    parser.add_argument("--feature_asset_ref", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="ecom-churn_model")
    args = parser.parse_args()

    features = pd.read_csv(args.feature_data)
    print(f"Loaded {len(features)} customers from {args.feature_asset_ref}, "
          f"churn rate {features['churned'].mean():.2%}")
    train_and_register(features, args.feature_asset_ref, model_name=args.model_name)


if __name__ == "__main__":
    main()