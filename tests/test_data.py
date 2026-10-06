"""Tests for the data contract: schema, labels, and basic sanity.

These run on the small preprocessed CSVs only — no model, no GPU.
"""

import pandas as pd

TRAIN = "data/shorttextpreprocessedtrain.csv"
TEST = "data/shorttextpreprocessedtest.csv"


def test_train_schema():
    df = pd.read_csv(TRAIN)
    assert list(df.columns) == ["text", "label"]
    assert set(df["label"].unique()) <= {0, 1}
    assert len(df) > 0


def test_test_schema():
    df = pd.read_csv(TEST)
    assert list(df.columns) == ["text", "label"]
    assert set(df["label"].unique()) <= {0, 1}
    assert len(df) > 0


def test_no_empty_texts_sample():
    df = pd.read_csv(TRAIN, nrows=2000)
    assert df["text"].notna().all()
    assert (df["text"].str.strip() != "").all()


def test_class_imbalance_documented():
    # The README claims the data is imbalanced (~84% fake). Guard the claim.
    df = pd.read_csv(TRAIN)
    fake_frac = (df["label"] == 0).mean()
    assert 0.75 < fake_frac < 0.95, f"unexpected fake fraction {fake_frac}"
