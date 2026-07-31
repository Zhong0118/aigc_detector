# Attribution Pilot Baseline Report

## Dataset

- Raw records: 923
- Clean records used: 780
- Labels: 10
- Semantic prompt groups: 12

## Out-of-fold results

| Method | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Char TF-IDF + LR | 0.3500 | 0.3239 | 0.3314 |
| Word/Char TF-IDF + SVM | 0.3141 | 0.2692 | 0.2762 |

## Interpretation boundary

This pilot currently has one model per family. Therefore the reported family classification performance is confounded with the specific routed model version and must not yet be interpreted as generalization to unseen models from the same family.

All scores are out-of-fold. Chinese and English versions of the same semantic prompt are assigned to the same fold to prevent prompt-pair leakage.
