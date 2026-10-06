# Explainable Fake News Detection using BERT

End-to-end fake-news detection: a fine-tuned `bert-base-uncased` classifier with **SHAP** and
**LIME** explanations and a live Hugging Face demo. Built as MS Data Science coursework
(PAF-IAST) and maintained here as a portfolio project.

> **Suggested repo rename** (applied on GitHub by the owner): `explainable-fake-news-detection`

## Problem statement

Short social-media-style news texts need fast, trustworthy fake/real classification — and a
bare accuracy number isn't enough for a system people should act on. This project answers:

1. Can a fine-tuned BERT model separate fake from real short news texts?
2. *Why* did it make each prediction — which words drove the decision (SHAP token attributions, LIME local explanations)?
3. How does it behave under strong class imbalance (~84% fake in training)?

## Methodology

- **Model.** `bert-base-uncased` + sequence-classification head, fine-tuned with cross-entropy
  loss and AdamW (`notebooks/02_bert_training.ipynb`).
- **Evaluation.** Held-out test set (`notebooks/03_model_evaluation.ipynb`); per-class
  precision/recall reported because of the imbalance — not just accuracy.
- **Explainability.** SHAP for token-level Shapley attributions
  (`notebooks/04_explainability_shap.ipynb`); LIME for instance-level explanations rendered as a
  lightweight table for deployment (`notebooks/05_explainability_lime.ipynb`).
- **Deployment.** Interactive Gradio app on Hugging Face Spaces.

## Results

All numbers below are read from `results/metrics.txt` (test set, **Real News as the positive class**)
and the accompanying report — nothing is rounded beyond display.

| Metric | Value |
|---|---|
| Accuracy | 0.9057 |
| Precision (Real) | 0.7035 |
| Recall (Real) | 0.6778 |
| F1-score (Real) | 0.6904 |

Dataset: 25,336 training / 6,525 test short texts (`text`, `label` with 0 = Fake, 1 = Real);
training split is 21,390 fake / 3,946 real.

**Reading the numbers honestly:** accuracy looks high because the majority class (fake) dominates;
the per-class scores show the model is much less certain on real news (F1 0.69). The confusion
matrix (`results/confusion_matrix.png`) and the class-distribution plot tell the same story.

**Live demo:** 👉 https://huggingface.co/spaces/AbdullahBinAjmal/fake-news-bert
(verified reachable 2026-10-06; type any news text, get the prediction plus a LIME explanation)

## Quick start

```bash
pip install -r requirements.txt        # 1. dependencies
pytest -q                               # 2. run the test suite (no GPU needed)
jupyter notebook notebooks/             # 3. run 01 → 05 in order
```

Retraining needs a GPU and ~3 GB free; evaluation/explainability notebooks can run against any
compatible checkpoint placed in `models/bert_fake_news_model/` (see `models/README.md`).

## Project structure

```
├── README.md
├── LICENSE (MIT)
├── requirements.txt          # pinned
├── pyproject.toml
├── .github/workflows/ci.yml  # pytest on 3.10/3.11/3.12
├── src/fakenews/             # evaluation + explainability helpers (tested)
│   ├── evaluation.py         # metrics computation, results/metrics.txt parser
│   └── explain.py            # LIME HTML validation, artifact inventory
├── tests/                    # 18 tests: data schema, metrics, artifacts, hygiene
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_bert_training.ipynb
│   ├── 03_model_evaluation.ipynb
│   ├── 04_explainability_shap.ipynb
│   └── 05_explainability_lime.ipynb   # outputs stripped; re-run to regenerate
├── data/
│   ├── shorttextpreprocessedtrain.csv # 25,336 rows (tracked)
│   ├── shorttextpreprocessedtest.csv  # 6,525 rows (tracked)
│   └── README.md                     # provenance notes (honest: source undocumented)
├── models/
│   └── README.md             # weights not committed; how to retrain
├── results/                  # metrics.txt, figures, LIME HTML, deployment screenshot
└── Report_LaTeX/             # original IEEE-format course report (PDF + source)
```

## Reproducibility

- Dependencies are pinned in `requirements.txt`; CI runs the test suite on Python 3.10–3.12.
- `src/fakenews` + `tests/` verify the data contract and that `results/metrics.txt` matches the
  headline numbers, so the README can never drift from the experiment output.
- Notebook outputs are stripped in git (keeps the repo small); re-run to regenerate figures.
- Random seeds: set inside the training/evaluation notebooks (see notebook 02/03).
- Original training hardware: GPU (see `notebooks/02_bert_training.ipynb` for training arguments).

## Limitations & future work

- **Class imbalance:** with ~84% fake in training, the model is conservative on real news
  (recall 0.68). Class-weighted loss or threshold tuning would likely help.
- **Data provenance:** the preprocessed files were provided as coursework; the upstream source
  is undocumented (see `data/README.md`). Re-validate labels and check train/test leakage before
  strong claims.
- **Short English texts only:** no evidence yet on long articles, other languages, or new topics
  (temporal drift untested).
- **SHAP cost:** exact-ish Shapley attributions on BERT are expensive; the notebook uses
  approximations — explanation fidelity vs. speed is an open trade-off.
- **Future:** calibration analysis, adversarial/robustness tests, multilingual extension,
  comparing against modern baselines (DeBERTa, LLM zero-shot).

## References

- Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (2018)
- Lundberg & Lee, *A Unified Approach to Interpreting Model Predictions* (SHAP, 2017)
- Ribeiro et al., *"Why Should I Trust You?": Explaining the Predictions of Any Classifier* (LIME, 2016)
- Full write-up: `Report_LaTeX/Abdullah_M24F0044DS009.pdf`

## Author

**Abdullah Ajmal** — MS Data Science, PAF-IAST · [GitHub](https://github.com/Abdullah9588041) · [LinkedIn](https://www.linkedin.com/in/abdullah-ajmal-050507183)
