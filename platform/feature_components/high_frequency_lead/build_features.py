import argparse
import os
import pandas as pd

HIGH_FREQUENCY_PERCENTILE = 0.75


def build_lead_features(orders: pd.DataFrame) -> pd.DataFrame:
    reference_date = orders["InvoiceDate"].max() + pd.Timedelta(days=1)

    per_customer = orders.groupby("CustomerID").agg(
        last_purchase=("InvoiceDate", "max"),
        first_purchase=("InvoiceDate", "min"),
        order_count=("InvoiceNo", "nunique"),
        total_revenue=("LineRevenue", "sum"),
        total_quantity=("Quantity", "sum"),
    ).reset_index()

    per_customer["recency_days"] = (reference_date - per_customer["last_purchase"]).dt.days
    per_customer["days_since_signup"] = (reference_date - per_customer["first_purchase"]).dt.days
    per_customer["avg_order_value"] = (per_customer["total_revenue"] / per_customer["order_count"]).round(2)
    per_customer["avg_quantity_per_order"] = (per_customer["total_quantity"] / per_customer["order_count"]).round(2)

    threshold = per_customer["order_count"].quantile(HIGH_FREQUENCY_PERCENTILE)
    per_customer["high_frequency"] = (per_customer["order_count"] >= threshold).astype(int)

    return per_customer[["CustomerID", "recency_days", "avg_order_value",
                          "days_since_signup", "avg_quantity_per_order", "high_frequency"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--curated_orders", type=str, required=True)
    parser.add_argument("--lead_features", type=str, required=True)
    args = parser.parse_args()

    input_files = [f for f in os.listdir(args.curated_orders) if f.endswith(".csv")]
    orders = pd.read_csv(os.path.join(args.curated_orders, input_files[0]), parse_dates=["InvoiceDate"])

    features = build_lead_features(orders)

    os.makedirs(args.lead_features, exist_ok=True)
    features.to_csv(os.path.join(args.lead_features, "lead_features.csv"), index=False)
    print(f"Built lead features for {len(features)} customers, "
          f"high-frequency rate {features['high_frequency'].mean():.2%} -> {args.lead_features}")


if __name__ == "__main__":
    main()