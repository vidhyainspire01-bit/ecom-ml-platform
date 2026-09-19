"""
Real unit tests against platform/components/evaluate/evaluate.py's
score_and_gate() -- imports and calls the actual function.
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "platform", "components", "evaluate"))
from evaluate import score_and_gate  # noqa: E402


def test_passes_when_score_meets_threshold():
    result = score_and_gate([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2], "roc_auc", 0.80)
    assert result["passed"] is True
    assert result["score"] > 0.80


def test_fails_when_score_below_threshold():
    result = score_and_gate([0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5], "roc_auc", 0.80)
    assert result["passed"] is False


def test_accuracy_metric():
    result = score_and_gate([0, 1, 1, 0], [0, 1, 0, 0], "accuracy", 0.5)
    assert result["metric_name"] == "accuracy"
    assert result["score"] == 0.75
    assert result["passed"] is True


def test_rmse_gate_direction_is_flipped():
    result = score_and_gate([1.0, 2.0, 3.0], [1.1, 2.1, 2.9], "rmse", 0.5)
    assert result["metric_name"] == "rmse"
    assert result["passed"] is True


def test_rmse_gate_fails_on_large_error():
    result = score_and_gate([1.0, 2.0, 3.0], [10.0, 20.0, 30.0], "rmse", 0.5)
    assert result["passed"] is False


def test_boundary_exactly_at_threshold_passes():
    result = score_and_gate([0, 0, 1, 1], [0, 0, 1, 1], "accuracy", 1.0)
    assert result["passed"] is True