import argparse
import os
import pandas as pd


def curate_dataframe(df: pd.DataFrame, feature_cols: list, target_col: str) -> pd.DataFrame:
    """Pure logic, no file I/O -- this is what tests/test_preprocess.py
    actually exercises. main() below is just the CLI/file wrapper."""
    keep_cols = feature_cols + [target_col]
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Config references columns not in data: {missing}")
    return df[keep_cols].dropna()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--target_column", type=str, required=True)
    parser.add_argument("--clean_data", type=str, required=True)
    args = parser.parse_args()

    feature_cols = [f.strip() for f in args.features.split(",")]

    input_files = [f for f in os.listdir(args.input_data) if f.endswith(".csv")]
    if not input_files:
        raise FileNotFoundError(f"No CSV found in {args.input_data}")
    df = pd.read_csv(os.path.join(args.input_data, input_files[0]))

    df = curate_dataframe(df, feature_cols, args.target_column)

    os.makedirs(args.clean_data, exist_ok=True)
    df.to_csv(os.path.join(args.clean_data, "clean.csv"), index=False)
    print(f"Wrote {len(df)} rows, {len(feature_cols)} features -> {args.clean_data}")


if __name__ == "__main__":
    main()