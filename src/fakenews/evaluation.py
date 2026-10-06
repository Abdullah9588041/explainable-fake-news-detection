"""Classification metrics for the fake-news experiments.

The project's headline numbers live in ``results/metrics.txt`` (produced by
``notebooks/03_model_evaluation.ipynb``). The functions here recompute the same
quantities from raw predictions so results stay verifiable, and parse the
metrics file for regression checks.
"""

from __future__ import annotations

from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

METRIC_KEYS = ("accuracy", "precision", "recall", "f1")


def compute_metrics(y_true, y_pred, positive_label: int = 1) -> dict:
    """Binary classification metrics with the Real-News class as positive.

    Mirrors the convention used in the report and ``results/metrics.txt``:
    label 1 (Real News) is the positive class, label 0 (Fake News) negative.
    """
    y_true = list(y_true)
    y_pred = list(y_pred)
    if len(y_true) == 0:
        raise ValueError("y_true must not be empty")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)),
    }


def load_metrics_txt(path: str | Path) -> dict:
    """Parse ``results/metrics.txt`` (``Key: value`` lines) into a dict."""
    metrics: dict = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower().replace("-", "_").replace(" ", "_")
        if key == "f1_score":
            key = "f1"
        try:
            metrics[key] = float(value.strip())
        except ValueError:
            continue
    return metrics
