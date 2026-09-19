"""
Gold-layer feature build for the churn model. This is the same RFM
logic from ds/generate_features_and_train.py's exploration phase, now
formalized as a versioned, re-runnable step that reads from the shared
silver table instead of raw transactions directly.
"""
import argparse
import os
import pandas as pd


def build_churn_features(orders: pd.DataFrame, churn_window_days: int) -> pd.DataFrame:
    reference_date = orders["InvoiceDate"].max() + pd.Timedelta(days=1)

    per_customer = orders.groupby("CustomerID").agg(
        last_purchase=("InvoiceDate", "max"),
        first_purchase=("InvoiceDate", "min"),
        order_count=("InvoiceNo", "nunique"),
        total_revenue=("LineRevenue", "sum"),
    ).reset_index()

    per_customer["recency_days"] = (reference_date - per_customer["last_purchase"]).dt.days
    per_customer["days_since_signup"] = (reference_date - per_customer["first_purchase"]).dt.days
    per_customer["avg_order_value"] = (per_customer["total_revenue"] / per_customer["order_count"]).round(2)
    per_customer["churned"] = (per_customer["recency_days"] > churn_window_days).astype(int)

    return per_customer[["CustomerID", "recency_days", "order_count",
                          "avg_order_value", "days_since_signup", "churned"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--curated_orders", type=str, required=True)
    parser.add_argument("--churn_window_days", type=int, default=90)
    parser.add_argument("--churn_features", type=str, required=True)
    args = parser.parse_args()

    input_files = [f for f in os.listdir(args.curated_orders) if f.endswith(".csv")]
    orders = pd.read_csv(os.path.join(args.curated_orders, input_files[0]), parse_dates=["InvoiceDate"])

    features = build_churn_features(orders, args.churn_window_days)

    os.makedirs(args.churn_features, exist_ok=True)
    features.to_csv(os.path.join(args.churn_features, "churn_features.csv"), index=False)
    print(f"Built churn features for {len(features)} customers, "
          f"churn rate {features['churned'].mean():.2%} -> {args.churn_features}")


if __name__ == "__main__":
    main()
