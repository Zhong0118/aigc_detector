# Attribution Pilot Baseline Report

## Dataset

- Raw records: 923
- Clean records used: 780
- Labels: 10
- Semantic prompt groups: 12

## Out-of-fold results

| Method | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Char TF-IDF + LR | 0.3526 | 0.3263 | 0.3342 |
| Word/Char TF-IDF + SVM | 0.3141 | 0.2692 | 0.2762 |
| Frozen E5 + Linear Probe | 0.1731 | 0.1378 | 0.1419 |

## Interpretation boundary

This pilot currently has one model per family. Therefore the reported family classification performance is confounded with the specific routed model version and must not yet be interpreted as generalization to unseen models from the same family.

All scores are out-of-fold. Chinese and English versions of the same semantic prompt are assigned to the same fold to prevent prompt-pair leakage.
