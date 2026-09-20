"""
Real unit tests against platform/feature_components/churn/build_features.py's
build_churn_features() -- the actual RFM math, using a small synthetic
orders table instead of real transaction data.
"""
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "platform", "feature_components", "churn"))
from build_features import build_churn_features  # noqa: E402


def _orders(rows):
    """rows: list of (CustomerID, InvoiceNo, InvoiceDate, LineRevenue)"""
    df = pd.DataFrame(rows, columns=["CustomerID", "InvoiceNo", "InvoiceDate", "LineRevenue"])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


def test_recency_is_days_since_last_purchase():
    orders = _orders([
        (1, "INV1", "2026-01-01", 100),
        (1, "INV2", "2026-01-10", 50),
    ])
    features = build_churn_features(orders, churn_window_days=90)
    row = features[features["CustomerID"] == 1].iloc[0]
    assert row["recency_days"] == 1


def test_order_count_counts_distinct_invoices_not_lines():
    orders = _orders([
        (1, "INV1", "2026-01-01", 30),
        (1, "INV1", "2026-01-01", 20),
        (1, "INV2", "2026-01-05", 50),
    ])
    features = build_churn_features(orders, churn_window_days=90)
    row = features[features["CustomerID"] == 1].iloc[0]
    assert row["order_count"] == 2


def test_avg_order_value_divides_by_order_count_not_line_count():
    orders = _orders([
        (1, "INV1", "2026-01-01", 30),
        (1, "INV1", "2026-01-01", 20),
        (1, "INV2", "2026-01-05", 50),
    ])
    features = build_churn_features(orders, churn_window_days=90)
    row = features[features["CustomerID"] == 1].iloc[0]
    assert row["avg_order_value"] == 50.0


def test_churn_label_true_when_recency_exceeds_window():
    orders = _orders([
        (1, "INV1", "2026-01-01", 100),
        (2, "INV2", "2026-06-01", 100),
    ])
    features = build_churn_features(orders, churn_window_days=90)
    cust1 = features[features["CustomerID"] == 1].iloc[0]
    cust2 = features[features["CustomerID"] == 2].iloc[0]
    assert cust1["churned"] == 1
    assert cust2["churned"] == 0


def test_days_since_signup_uses_first_purchase_not_last():
    orders = _orders([
        (1, "INV1", "2026-01-01", 100),
        (1, "INV2", "2026-03-01", 100),
    ])
    features = build_churn_features(orders, churn_window_days=90)
    row = features[features["CustomerID"] == 1].iloc[0]
    assert row["days_since_signup"] == 60