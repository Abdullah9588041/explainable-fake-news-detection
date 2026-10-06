"""Explainable fake-news detection: evaluation and explainability helpers.

Thin, dependency-light utilities extracted from the project notebooks so the
metrics and artifacts can be tested without a GPU or the trained BERT weights.
"""

from .evaluation import compute_metrics, load_metrics_txt
from .explain import validate_lime_html

__all__ = ["compute_metrics", "load_metrics_txt", "validate_lime_html"]
