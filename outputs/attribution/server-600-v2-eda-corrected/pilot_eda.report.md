# Text-attribution pilot EDA report

## Metadata

- Source: `C:\Users\zx\Desktop\zhizengzeng-attribution-server-600-v2.jsonl`
- Format: JSON Lines
- Size: 2,728,248 bytes
- Records: 558
- Families / models / prompts: 10 / 20 / 10

## Quality summary

| Metric | Value |
|---|---:|
| Expected total | 600 |
| Missing | 42 |
| Empty text | 0 |
| Truncated | 9 (1.61%) |
| Content filtered | 0 |
| Language mismatch suspects | 0 |
| Exact duplicate excess | 2 (0.36%) |
| └ Low-temperature-only duplicate excess | 0 |
| └ Other duplicate excess | 2 |
| Route mismatch | 0 |

## Per-family diagnostics

| Family | n | Missing | Truncated | Exact duplicate excess | Latency p50 / p95 (s) | Completion tokens p50 / p95 |
|---|---:|---:|---:|---:|---:|---:|
| claude | 60 | 0 | 1 (1.7%) | 0 | 14.056887 / 51.221644 | 728.5 / 2169.3 |
| deepseek | 30 | 30 | 0 (0.0%) | 0 | 14.419952 / 31.425228 | 993.0 / 2293.85 |
| doubao | 60 | 0 | 0 (0.0%) | 0 | 31.577961 / 72.385559 | 2227.5 / 4029.9 |
| gemini | 60 | 0 | 0 (0.0%) | 0 | 4.765586 / 10.932473 | 616.0 / 1583.35 |
| glm | 59 | 1 | 7 (11.9%) | 0 | 41.722101 / 91.129618 | 1902.0 / 3366.0 |
| kimi | 60 | 0 | 0 (0.0%) | 0 | 12.419632 / 28.347735 | 489.0 / 1086.25 |
| mimo | 60 | 0 | 1 (1.7%) | 0 | 23.368852 / 41.449923 | 802.5 / 1963.35 |
| minimax | 60 | 0 | 0 (0.0%) | 0 | 13.578231 / 34.601891 | 627.0 / 1106.45 |
| openai | 49 | 11 | 0 (0.0%) | 0 | 8.036718 / 27.568089 | 572.0 / 2219.0 |
| qwen | 60 | 0 | 0 (0.0%) | 2 | 10.693421 / 25.187149 | 558.5 / 1200.8 |

## Length distributions

Chinese length is measured as non-whitespace characters; English length is measured as regex-tokenized words.

| Language | n | Min | Median | P95 | Max |
|---|---:|---:|---:|---:|---:|
| en | 278 | 59.0 | 482.0 | 933.15 | 1285.0 |
| zh | 280 | 46.0 | 791.0 | 2226.2 | 2773.0 |

## Key findings

- The collection is incomplete: 42 planned cells are missing.
- 9 rows ended because an output-token limit was reached (`length` or `max_tokens`); these should not enter a formal baseline unchanged.
- 2 rows are exact duplicates beyond the first occurrence. 0 of them occur only at temperature 0 and mainly reflect deterministic repeat generation; the remaining duplicates need closer review.
- Group train/test splits by `prompt_id`; never randomly split individual rows.
- Treat this collection as a pilot until missing cells, truncation, and duplicate handling are resolved.

## Figures

- `01_coverage_quality.png`: per-family coverage and quality failures.
- `02_prompt_family_coverage.png`: prompt-by-family coverage matrix.
- `03_length_distribution.png`: Chinese character and English word distributions.
- `04_latency_tokens.png`: latency and completion-token distributions.
