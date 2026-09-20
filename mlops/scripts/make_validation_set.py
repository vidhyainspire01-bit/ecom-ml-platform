"""
Carves a held-out validation split from a feature table and uploads it
as its own versioned data asset. DS trains on the full feature table;
this script creates the independent slice validate_and_promote.py
scores against — so promotion checks a set DS never saw, not a
re-read of DS's own reported number.

Deliberately a SEPARATE, deterministic split (fixed random_state) from
whatever split DS's own script does internally, so it's not guaranteed
to overlap with DS's training rows in a way that would make the
"independent" check meaningless.
"""
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target_column", default="churned",
                         help="Model-agnostic: pass the correct label column, e.g. 'high_frequency' for lead scoring")
    parser.add_argument("--validation_fraction", type=float, default=0.25)
    args = parser.parse_args()

    df = pd.read_csv(args.feature_data)
    _, val_df = train_test_split(
        df, test_size=args.validation_fraction, random_state=99,
        stratify=df[args.target_column],
    )
    val_df.to_csv(args.output, index=False)
    print(f"Wrote {len(val_df)} held-out rows -> {args.output}, "
          f"{args.target_column} positive rate {val_df[args.target_column].mean():.2%}")


if __name__ == "__main__":
    main()