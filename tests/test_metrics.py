"""Tests for metrics computation and the recorded results.

Guards the headline numbers: they must match results/metrics.txt exactly,
so the README can never drift from the real experiment output.
"""

import sys
from pathlib import Path

import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fakenews.evaluation import compute_metrics, load_metrics_txt

# Headline numbers from results/metrics.txt (read, not invented).
RECORDED = {
    "accuracy": 0.9057492931196984,
    "precision": 0.7034700315457413,
    "recall": 0.6778115501519757,
    "f1": 0.6904024767801857,
}


def test_compute_metrics_matches_sklearn():
    y_true = [0, 0, 0, 1, 1, 0, 1, 0, 1, 1]
    y_pred = [0, 0, 1, 1, 0, 0, 1, 1, 1, 1]
    got = compute_metrics(y_true, y_pred)
    assert got["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred))
    assert got["precision"] == pytest.approx(precision_score(y_true, y_pred))
    assert got["recall"] == pytest.approx(recall_score(y_true, y_pred))
    assert got["f1"] == pytest.approx(f1_score(y_true, y_pred))


def test_compute_metrics_rejects_empty():
    with pytest.raises(ValueError):
        compute_metrics([], [])


def test_metrics_txt_parses():
    m = load_metrics_txt("results/metrics.txt")
    for key, value in RECORDED.items():
        assert key in m, f"missing key {key} in results/metrics.txt"
        assert m[key] == pytest.approx(value, rel=1e-9), f"{key} drifted: {m[key]}"


def test_metrics_in_valid_range():
    m = load_metrics_txt("results/metrics.txt")
    for key, value in m.items():
        assert 0.0 <= value <= 1.0, f"{key}={value} out of range"
