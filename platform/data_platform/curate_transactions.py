"""
Platform-level (silver layer) curation — generic across every model that
consumes order history. No churn logic, no RFM, no target definition
lives here.
"""
import argparse
import os
import pandas as pd


def curate(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df = df[~df["InvoiceNo"].str.startswith("C")]          # drop cancellations
    df = df.dropna(subset=["CustomerID"])
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]  # drop invalid rows
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["LineRevenue"] = df["Quantity"] * df["UnitPrice"]
    df["CustomerID"] = df["CustomerID"].astype(int)
    return df[["InvoiceNo", "CustomerID", "InvoiceDate", "Quantity",
               "UnitPrice", "LineRevenue", "Country"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_transactions", type=str, required=True)
    parser.add_argument("--curated_orders", type=str, required=True)
    args = parser.parse_args()

    input_files = [f for f in os.listdir(args.raw_transactions) if f.endswith(".csv")]
    if not input_files:
        raise FileNotFoundError(f"No CSV found in {args.raw_transactions}")
    raw_df = pd.read_csv(os.path.join(args.raw_transactions, input_files[0]), encoding="ISO-8859-1")

    curated = curate(raw_df)

    os.makedirs(args.curated_orders, exist_ok=True)
    curated.to_csv(os.path.join(args.curated_orders, "curated_orders.csv"), index=False)
    print(f"Curated {len(curated)} order lines for {curated['CustomerID'].nunique()} customers "
          f"-> {args.curated_orders}")


if __name__ == "__main__":
    main()
