# Data

## Files used in the experiments (tracked in git)

| File | Rows | Size | Description |
|---|---|---|---|
| `shorttextpreprocessedtrain.csv` | 25,336 | 2.3 MB | Training set: `text` (cleaned short news text), `label` (0 = Fake, 1 = Real). Class split: 21,390 fake / 3,946 real (~84% fake — imbalanced). |
| `shorttextpreprocessedtest.csv` | 6,525 | 0.6 MB | Held-out test set, same schema. |

**Provenance (honest note):** the original upstream source of these preprocessed files is not
documented — they were provided with the project (course work, PAF-IAST, roll no. M24F0044DS009).
Treat them as *provided* data, not as a cited public benchmark. Any reuse should re-validate the
labels and check for train/test leakage before drawing strong conclusions.

## Files NOT tracked in git

| File | Size | Why |
|---|---|---|
| `newdatasetwithcoviddata.csv` | 15 MB | Raw, unused by the notebook pipeline (no notebook references it). Kept locally only; excluded via `.gitignore`. Schema is the same `text,label` format with COVID-era news items. |

If you need the raw file, ask the repo owner — it is not downloadable from this repository.
