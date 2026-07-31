# Text-attribution pilot EDA report

## Metadata

- Source: `C:\Users\zx\Desktop\zhizengzeng-attribution-smoke-v2-concurrent.jsonl`
- Format: JSON Lines
- Size: 191,180 bytes
- Records: 36
- Families / models / prompts: 10 / 19 / 2

## Quality summary

| Metric | Value |
|---|---:|
| Expected total | 38 |
| Missing | 2 |
| Empty text | 0 |
| Truncated | 2 (5.56%) |
| Content filtered | 0 |
| Language mismatch suspects | 0 |
| Exact duplicate excess | 0 (0.00%) |
| └ Low-temperature-only duplicate excess | 0 |
| └ Other duplicate excess | 0 |
| Route mismatch | 0 |

## Per-family diagnostics

| Family | n | Missing | Truncated | Exact duplicate excess | Latency p50 / p95 (s) | Completion tokens p50 / p95 |
|---|---:|---:|---:|---:|---:|---:|
| claude | 4 | 0 | 0 (0.0%) | 0 | 17.276826 / 25.661605 | 914.0 / 1052.95 |
| deepseek | 4 | 0 | 0 (0.0%) | 0 | 16.408133 / 42.017462 | 933.0 / 1884.1 |
| doubao | 4 | 0 | 0 (0.0%) | 0 | 31.023446 / 75.497836 | 2586.0 / 3820.7 |
| gemini | 4 | 0 | 2 (50.0%) | 0 | 10.161033 / 16.799958 | 1405.5 / 2836.75 |
| glm | 4 | 0 | 0 (0.0%) | 0 | 41.611503 / 66.886922 | 1967.5 / 2860.55 |
| kimi | 4 | 0 | 0 (0.0%) | 0 | 9.52553 / 32.609002 | 926.5 / 1494.95 |
| mimo | 4 | 0 | 0 (0.0%) | 0 | 19.038026 / 25.766048 | 879.0 / 1351.15 |
| minimax | 1 | 1 | 0 (0.0%) | 0 | 69.183104 / 69.183104 | 2236.0 / 2236.0 |
| openai | 3 | 0 | 0 (0.0%) | 0 | 8.710419 / 9.679776 | 941.0 / 985.1 |
| qwen | 4 | 0 | 0 (0.0%) | 0 | 7.767478 / 22.347848 | 696.0 / 1183.75 |

## Length distributions

Chinese length is measured as non-whitespace characters; English length is measured as regex-tokenized words.

| Language | n | Min | Median | P95 | Max |
|---|---:|---:|---:|---:|---:|
| en | 19 | 86.0 | 766.0 | 988.5 | 1020.0 |
| zh | 17 | 101.0 | 455.0 | 708.8 | 804.0 |

## Key findings

- The collection is incomplete: 2 planned cells are missing.
- 2 rows ended because an output-token limit was reached (`length` or `max_tokens`); these should not enter a formal baseline unchanged.
- Group train/test splits by `prompt_id`; never randomly split individual rows.
- Treat this collection as a pilot until missing cells, truncation, and duplicate handling are resolved.

## Figures

- `01_coverage_quality.png`: per-family coverage and quality failures.
- `02_prompt_family_coverage.png`: prompt-by-family coverage matrix.
- `03_length_distribution.png`: Chinese character and English word distributions.
- `04_latency_tokens.png`: latency and completion-token distributions.
