"""Tests for repo hygiene: artifacts present, duplicates gone, notebooks stripped."""

import json
from pathlib import Path

import pytest

EXPECTED_ARTIFACTS = [
    "results/confusion_matrix.png",
    "results/class_distribution.png",
    "results/shap_explanation.png",
    "results/lime_explanation.png",
    "results/lime_explanation.html",
    "results/metrics.txt",
    "results/hf_deployment.PNG",
]


@pytest.mark.parametrize("rel", EXPECTED_ARTIFACTS)
def test_result_artifact_exists(rel):
    p = Path(rel)
    assert p.is_file(), f"missing artifact {rel}"
    assert p.stat().st_size > 0, f"empty artifact {rel}"


def test_typo_duplicate_removed():
    assert not Path("results/confusion_metrix.png").exists(), (
        "typo duplicate results/confusion_metrix.png should have been removed"
    )


def test_notebooks_have_no_outputs():
    for nb_path in sorted(Path("notebooks").glob("*.ipynb")):
        nb = json.loads(nb_path.read_text())
        for i, cell in enumerate(nb["cells"]):
            outputs = cell.get("outputs", [])
            assert not outputs, f"{nb_path.name} cell {i} still has outputs"
            assert cell.get("execution_count") in (None, 0), (
                f"{nb_path.name} cell {i} still has execution_count"
            )


def test_large_raw_csv_not_tracked():
    # The 15MB raw file must stay out of git (see data/README.md).
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files", "data/"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "newdatasetwithcoviddata.csv" not in tracked
