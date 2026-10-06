# Models

`bert_fake_news_model/` currently holds only `config.json` (the `bert-base-uncased`
architecture config). **Model weights are intentionally not committed** — a fine-tuned
BERT checkpoint is ~440 MB.

## To reproduce the trained model

Run the notebooks in order:

1. `notebooks/01_data_exploration.ipynb` — EDA on `data/shorttextpreprocessed*.csv`
2. `notebooks/02_bert_training.ipynb` — fine-tune `bert-base-uncased` for binary classification
   (cross-entropy loss, AdamW). Save the checkpoint into `models/bert_fake_news_model/`.
3. `notebooks/03_model_evaluation.ipynb` — test-set evaluation → `results/metrics.txt`
4. `notebooks/04_explainability_shap.ipynb`, `05_explainability_lime.ipynb` — explanations

Training was originally run on a GPU (see notebook 02 for the exact training arguments).
CPU-only retraining is possible but slow; the evaluation and explainability notebooks can be
re-run against any compatible checkpoint by pointing them at this directory.

A deployed demo of the trained model is live at
https://huggingface.co/spaces/AbdullahBinAjmal/fake-news-bert (verified 2026-10-06).
