"""
Data quality gate -- runs BEFORE preprocess_step, on every model's
raw input. Model-agnostic: features/target_column come from config,
no churn-specific logic here.

On failure: raises SystemExit, failing this step. preprocess_step
reads THIS step's output, not the raw data directly, so a failure
here genuinely blocks the pipeline rather than just logging a warning.
"""
import argparse
import os
import shutil
import pandas as pd


def run_checks(df: pd.DataFrame, required_columns: list, max_null_rate: float, min_row_count: int) -> dict:
    issues = []

    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")

    if len(df) < min_row_count:
        issues.append(f"Row count {len(df)} below minimum {min_row_count}")

    present_cols = [c for c in required_columns if c in df.columns]
    if present_cols:
        null_rate = df[present_cols].isnull().mean().max()
        if null_rate > max_null_rate:
            issues.append(f"Null rate {null_rate:.2%} exceeds max {max_null_rate:.2%}")

    return {"passed": len(issues) == 0, "issues": issues, "row_count": len(df)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", type=str, required=True)
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--target_column", type=str, required=True)
    parser.add_argument("--max_null_rate", type=float, default=0.05)
    parser.add_argument("--min_row_count", type=int, default=50)
    parser.add_argument("--validated_data", type=str, required=True)
    args = parser.parse_args()

    required_columns = [c.strip() for c in args.features.split(",")] + [args.target_column]

    input_files = [f for f in os.listdir(args.raw_data) if f.endswith(".csv")]
    if not input_files:
        raise SystemExit(f"Data quality gate: no CSV found in {args.raw_data}")
    input_file = input_files[0]
    df = pd.read_csv(os.path.join(args.raw_data, input_file))

    result = run_checks(df, required_columns, args.max_null_rate, args.min_row_count)
    print(f"Data quality check: {result}")

    if not result["passed"]:
        raise SystemExit(f"Data quality gate FAILED: {result['issues']}")

    os.makedirs(args.validated_data, exist_ok=True)
    shutil.copy(os.path.join(args.raw_data, input_file), os.path.join(args.validated_data, input_file))
    print(f"Data quality gate PASSED: {result['row_count']} rows -> {args.validated_data}")


if __name__ == "__main__":
    main()