import pandas as pd

FEATURE_COLUMNS = ["order_count", "avg_order_value", "days_since_signup"]
COLUMN_DTYPES = {"order_count": "int64", "avg_order_value": "float64", "days_since_signup": "int64"}


def preprocess(payload: dict) -> pd.DataFrame:
    data = payload.get("input_data", payload)
    columns = data.get("columns", FEATURE_COLUMNS)
    df = pd.DataFrame(data["data"], columns=columns)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df[FEATURE_COLUMNS]
    for col, dtype in COLUMN_DTYPES.items():
        df[col] = df[col].astype(dtype)
    return df