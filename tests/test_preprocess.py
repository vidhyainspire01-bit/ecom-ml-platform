"""
Real unit tests against platform/components/preprocess/preprocess.py's
curate_dataframe() -- imports and calls the actual function, doesn't
reimplement its logic.
"""
import sys
import os
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "platform", "components", "preprocess"))
from preprocess import curate_dataframe  # noqa: E402


def test_keeps_only_requested_columns():
    df = pd.DataFrame({
        "a": [1, 2, 3], "b": [4, 5, 6], "target": [0, 1, 0], "extra": ["x", "y", "z"],
    })
    result = curate_dataframe(df, ["a", "b"], "target")
    assert list(result.columns) == ["a", "b", "target"]
    assert "extra" not in result.columns


def test_raises_on_missing_feature_column():
    df = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
    with pytest.raises(ValueError, match="not in data"):
        curate_dataframe(df, ["a", "b"], "target")


def test_raises_on_missing_target_column():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    with pytest.raises(ValueError, match="not in data"):
        curate_dataframe(df, ["a", "b"], "target")


def test_drops_rows_with_nulls():
    df = pd.DataFrame({"a": [1, 2, None], "b": [4, 5, 6], "target": [0, 1, 0]})
    result = curate_dataframe(df, ["a", "b"], "target")
    assert len(result) == 2


def test_empty_dataframe_after_dropna():
    df = pd.DataFrame({"a": [None, None], "b": [1, 2], "target": [0, 1]})
    result = curate_dataframe(df, ["a", "b"], "target")
    assert len(result) == 0