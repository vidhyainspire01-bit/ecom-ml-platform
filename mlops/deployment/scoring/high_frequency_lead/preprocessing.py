import pandas as pd

FEATURE_COLUMNS = ["recency_days", "avg_order_value", "days_since_signup", "avg_quantity_per_order"]
COLUMN_DTYPES = {"recency_days": "int64", "avg_order_value": "float64", "days_since_signup": "int64", "avg_quantity_per_order": "float64"}


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