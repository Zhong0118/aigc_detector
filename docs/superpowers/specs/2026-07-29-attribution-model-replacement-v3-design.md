# Attribution model replacement and V3 recollection design

## Objective

Replace the three unreliable concrete model slots identified by the
600-response V2 audit, validate the complete twenty-model catalog, and produce
a clean 600-response V3 dataset without mixing incompatible model versions.

The family labels remain unchanged:

```text
openai, claude, gemini, deepseek, qwen,
glm, minimax, kimi, doubao, mimo
```

Each family retains two concrete model roles (`fast` and `balanced`), yielding
twenty concrete models in total.

## Confirmed V2 problems

| Slot | V2 model | Evidence | Required action |
|---|---|---|---|
| OpenAI fast | `gpt-4.1-nano-2025-04-14` | 19/30 successes; eleven requests skipped after consecutive empty responses | replace |
| DeepSeek balanced | `deepseek-v4-pro` | 0/30 successes; two exhausted empty-response tasks opened its circuit | replace after candidate probe |
| GLM fast | `glm-4.5-air` | 29/30 successes, seven truncated rows, inconsistent visible-text/token behavior | replace after candidate probe |

The Gemini and MiniMax replacements from the preceding stage remain frozen:

```text
Gemini balanced:  gemini-3.1-flash-lite-preview
MiniMax fast:     abab6.5s-chat
MiniMax balanced: MiniMax-Text-01
```

## Candidate set

OpenAI fast uses the already validated:

```text
gpt-4o-mini-2024-07-18
```

DeepSeek balanced candidates:

```text
deepseek-chat
deepseek-v3.2
deepseek-v3.1-terminus
```

GLM fast candidates:

```text
glm-4.5-flash
glm-4-air-250414
glm-4-flashx-250414
```

These candidates are present in the gateway model listing. Listing presence is
not treated as proof of chat-route availability.

## Candidate probe

The probe uses the same two production smoke prompts:

- Chinese medium response with a 1536-token output budget;
- English long response with a 3072-token output budget.

The six unverified candidates produce twelve paid requests. Candidate requests
use the existing concurrent collector with four global workers and one
in-flight request per concrete model.

OpenAI fast is not repeated in this candidate round because
`gpt-4o-mini-2024-07-18` already passed the same bilingual production-prompt
test.

## Selection gates

A candidate is eligible only if both Chinese and English requests satisfy all
of the following:

1. HTTP/chat route succeeds;
2. returned text is non-empty;
3. returned route matches the requested route;
4. `finish_reason` is `stop`, not `length`, `max_tokens`, or
   `content_filter`;
5. visible output is in the requested language;
6. reasoning tokens do not consume the output budget and leave a short or
   incomplete visible answer;
7. latency is operationally acceptable relative to the alternatives.

If multiple candidates pass, the selection order is:

1. completeness and stability;
2. absence of hidden/reasoning-budget distortion;
3. output-length compliance;
4. latency;
5. lower expected API cost.

No candidate is promoted merely because it is newer or larger.

## Catalog and run versioning

Successful selections are written into a new catalog rather than silently
changing the provenance of the completed V2 run:

```text
configs/text_attribution/zhizengzeng_model_catalog.v3.json
```

New run configurations:

```text
configs/text_attribution/runs/smoke_v3.json
configs/text_attribution/runs/server_600_v3.json
```

Expected request counts:

```text
smoke_v3:     20 models × 2 prompts = 40
server_600_v3: 20 models × 10 prompts × 3 samples = 600
```

Run IDs and output paths use `v3`. They do not append to or overwrite V2
JSONL files.

## Execution sequence

1. Dry-run and execute the twelve-request DeepSeek/GLM candidate probe.
2. Audit candidate outputs and freeze one replacement per affected slot.
3. Generate the V3 catalog and dry-run `smoke_v3`; it must resolve to exactly
   forty requests.
4. Execute the paid forty-request smoke.
5. Require twenty models with two successes each, no empty text, no route
   mismatch, no language mismatch, and no truncation.
6. Dry-run `server_600_v3`; it must resolve to exactly 600 requests and thirty
   requests per model.
7. Execute the clean 600-response V3 collection.
8. Run configuration-aware audit and EDA.
9. Only after the V3 quality gates pass may the 6000-response configuration be
   generated or enabled.

## Treatment of V2 data

The 558 successful V2 records are retained unchanged as pilot and reliability
evidence. They are not deleted, overwritten, or merged into the clean V3
training dataset.

This choice avoids:

- mixing old and replacement concrete versions under one run identity;
- selecting different sample-generation dates for only a subset of families;
- complex supersession rules for truncated and duplicate rows;
- accidentally training on both a bad record and its repair.

V2 may still be used for robustness experiments or later multi-version family
training when its provenance is kept explicit.

## Failure handling

The existing policies remain:

- HTTP 400/403/404 opens a per-model circuit;
- transient network/429/5xx errors receive bounded retries;
- empty completions receive at most two attempts;
- two consecutively exhausted empty tasks open a per-model circuit;
- successful request keys are flushed immediately and skipped on resume;
- API keys, authorization headers, and secret-shaped fields are never written
  to artifacts.

Candidate failure does not modify the official V3 catalog. If no DeepSeek or
GLM candidate passes, that family slot remains unresolved and the forty-request
smoke is not started.

## Verification

Implementation is complete only when:

- candidate and V3 JSON files parse and resolve through `load_run_config`;
- the candidate plan has exactly twelve requests;
- `smoke_v3` has exactly forty requests and two per concrete model;
- `server_600_v3` has exactly 600 requests and thirty per concrete model;
- unit tests cover request counts and selected family/role uniqueness;
- the full repository test suite passes;
- the server bundle contains the collector, audit/EDA code, candidate
  configuration, V3 catalog, V3 run configurations, prompt manifests, and
  updated runbook, but no `.env.local` or API key.
