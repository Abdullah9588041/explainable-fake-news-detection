"""Explainability artifact helpers.

Lightweight checks over the explanation outputs produced by
``notebooks/04_explainability_shap.ipynb`` and ``05_explainability_lime.ipynb``.
No model weights or GPU required.
"""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path


class _TagCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        self.tags.append(tag)


def validate_lime_html(path: str | Path) -> dict:
    """Check that a LIME explanation HTML file is non-empty, valid HTML.

    Returns a small report dict; raises ``ValueError`` on failure.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"LIME HTML not found: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text.strip()) < 100:
        raise ValueError(f"LIME HTML suspiciously small ({len(text)} chars): {path}")
    parser = _TagCollector()
    parser.feed(text)
    has_table = "table" in parser.tags
    return {
        "path": str(path),
        "chars": len(text),
        "n_tags": len(parser.tags),
        "has_table": has_table,
    }


def explanation_artifacts(results_dir: str | Path) -> dict:
    """Map of expected explanation artifacts to whether they exist on disk."""
    results_dir = Path(results_dir)
    expected = [
        "shap_explanation.png",
        "lime_explanation.png",
        "lime_explanation.html",
        "confusion_matrix.png",
        "class_distribution.png",
    ]
    return {name: (results_dir / name).is_file() for name in expected}
