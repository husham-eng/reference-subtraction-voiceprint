# Reference Subtraction for Voiceprint Recognition — reproducibility package

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24192646.svg)](https://doi.org/10.5281/zenodo.24192646)

Code, extracted features, split manifest and sample-level predictions for:

> H. Saied, "Reference Subtraction for Voiceprint Recognition: A Discriminative Approach Based on the Group's Normal Range," ICPIC 2026 (camera-ready version).

## Main result (revised paper)

Reference Subtraction reduces exactly to conventional rules:

| Variant | Equivalent rule | Test check |
|---|---|---|
| Mean-only reference + Euclidean | Nearest centroid | identical predictions, 180/180 |
| Normal-range reference + diagonal Mahalanobis | Class-dependent diagonal Mahalanobis | identical predictions, 180/180 |
| Normal-range reference + Euclidean | Centroid distance weighted by the other speakers' spread | identical predictions, 180/180 |
| Any reference + cosine | no reduction | not better than baseline |

No variant outperformed its conventional counterpart (macro-F1 difference +0.010, 95% bootstrap interval −0.054 to +0.073; exact McNemar 22 vs 22, p = 1.00). Same-session cross-validation (0.95–0.99) far exceeds the separate-session test (0.76–0.84).

## Files

| File | Purpose |
|---|---|
| `01_extract_features.py` | 67-dim features from the `.ogg` recordings → `features.csv` (needs audio) |
| `02_run_experiments.py` | Seven original variants, confusion matrices, McNemar on the named pair |
| `09_reviewer_checks.py` | Direct diagonal Mahalanobis baseline, equivalence checks, exact McNemar, stratified bootstrap intervals, training-only cross-validation |
| `10_make_revision_figures.py` | Figs. 1–3 of the revised paper |
| `00`, `04`, `06`–`08` | Figures of the submitted version (06 needs audio) |
| `03`, `05` | PCA figure of the submitted version — **not used in the revision** (test residuals used true labels) |
| `features.csv` | Features of all 605 recordings (train + test) |
| `split_manifest.csv` | File name, speaker and split of every recording |
| `predictions_all_rules.csv` | Sample-level test predictions of all eight rules |
| `experiments_results.json`, `reviewer_checks.json` | All reported numbers |

## Run

```bash
pip install librosa numpy scipy scikit-learn matplotlib pandas
python 02_run_experiments.py
python 09_reviewer_checks.py      # ~2 min
python 10_make_revision_figures.py
```

## Conventions

- Global standardization: training mean and sample standard deviation (ddof = 1); zero → 1. Applied to all rules, including baselines.
- Reference (other speakers), fingerprint spread and class spread: population standard deviation (ddof = 0) + 1e-6.
- Pitch: probabilistic YIN, 65–2093 Hz (C2–C7); zero if no voiced frame (did not occur).

## Dataset

Five native Sudanese Arabic speakers (3 men, 2 women), one word (ḥammām), WhatsApp voice messages recorded on each speaker's own smartphone; training and test recorded in separate sessions on different days. 425 training / 180 test recordings after removing two exact train–test duplicates. Speaker and device are confounded. Audio is not distributed; it is available from the author on reasonable request.

## Citation

H. Saied, "Reference subtraction for voiceprint recognition: Code, features, split manifest, and predictions," v1.0.0, Zenodo, 2026. doi: 10.5281/zenodo.24192646

## License

Code: MIT. Features and predictions: CC BY 4.0.
