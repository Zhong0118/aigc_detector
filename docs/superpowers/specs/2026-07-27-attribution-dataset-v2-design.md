# Attribution Dataset V2 Design

## Goal

Build a resumable Zhizengzeng collection run for LLM text attribution with:

- 10 model families and 2 concrete models per family;
- 100 shared prompts;
- one deterministic sample and two diverse samples per model/prompt;
- 6000 planned responses;
- family-level and exact-model labels;
- prompt-level output budgets and complete request provenance.

## Model and prompt design

The model catalog contains 20 unique `model_id` values. Repeated `family_id`
values are required and each family must have one `fast` and one `balanced`
variant.

The prompt manifest contains 100 prompts across 10 domains. Forty semantic
tasks have paired Chinese and English prompts, with another 10 Chinese-native
and 10 English-native tasks. Every prompt records its semantic group, language
origin, task, length, format, system-prompt type, difficulty, output budget,
and validation hints.

## Sampling design

Sampling profiles own their repetition counts:

- `deterministic`: temperature 0.0, one repetition;
- `diverse`: temperature 0.8, two repetitions.

`top_p` is omitted. Output budgets are prompt-specific: 512 tokens for short,
1536 for medium, and 3072 for long prompts. The complete product is
`20 * 100 * (1 + 2) = 6000`.

## Collection behavior

Every request key includes the gateway, concrete model, prompt version,
sampling profile, effective generation parameters, and repetition index.
Successful keys are skipped on resume. Responses retain requested and returned
model IDs. Model-specific request overrides may disable reasoning where the
gateway supports it, but cannot replace core request fields.

The initial smoke profile runs two representative prompts against all 20
models, once each, for 40 requests. A dry run must validate the request count
before any paid call.

## Quality and repair

Raw responses and errors are append-only. Technical failures, truncation,
route mismatch, language mismatch, refusals, duplicates, and format adherence
are audited separately. Repair collection reuses the original request identity
and only reruns missing or rejected cells.

