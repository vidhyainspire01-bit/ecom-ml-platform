import argparse
import json
import os
import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_data", type=str, required=True)
    parser.add_argument("--target_column", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True, choices=["classifier", "regressor"])
    parser.add_argument("--hyperparameters", type=str, required=True)
    parser.add_argument("--model_output", type=str, required=True)
    args = parser.parse_args()

    hp = json.loads(args.hyperparameters)

    df = pd.read_csv(os.path.join(args.clean_data, "clean.csv"))
    X = df.drop(columns=[args.target_column])
    y = df[args.target_column]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    if args.model_type == "classifier":
        model = RandomForestClassifier(**hp, random_state=42)
    else:
        model = RandomForestRegressor(**hp, random_state=42)

    model.fit(X_train, y_train)

    mlflow.log_params(hp)
    mlflow.log_param("model_type", args.model_type)
    mlflow.sklearn.save_model(model, args.model_output, serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE)
    print(f"Trained {args.model_type} on {len(X_train)} rows -> {args.model_output}")


if __name__ == "__main__":
    main()
