import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "platform", "components", "data_quality"))
from check import run_checks  # noqa: E402


def test_passes_clean_data():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5] * 20, "target": [0, 1] * 50})
    result = run_checks(df, ["a", "target"], max_null_rate=0.05, min_row_count=50)
    assert result["passed"] is True


def test_fails_on_missing_column():
    df = pd.DataFrame({"a": [1, 2, 3] * 20})
    result = run_checks(df, ["a", "target"], max_null_rate=0.05, min_row_count=50)
    assert result["passed"] is False
    assert any("Missing required columns" in i for i in result["issues"])


def test_fails_on_too_few_rows():
    df = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
    result = run_checks(df, ["a", "target"], max_null_rate=0.05, min_row_count=50)
    assert result["passed"] is False
    assert any("Row count" in i for i in result["issues"])


def test_fails_on_high_null_rate():
    df = pd.DataFrame({"a": [1, None, None, None] * 15, "target": [0, 1] * 30})
    result = run_checks(df, ["a", "target"], max_null_rate=0.05, min_row_count=50)
    assert result["passed"] is False
    assert any("Null rate" in i for i in result["issues"])