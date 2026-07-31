# Attribution Dataset V2 API Runbook

## Current runnable stage

The V2 smoke run sends 40 requests:

```text
20 concrete models * (1 deterministic prompt + 1 diverse prompt) = 40
```

It verifies model availability, returned model routes, Chinese and English
generation, temperatures 0.0 and 0.8, medium and long output budgets, latency,
usage metadata, retries, and resume behavior. The paid run must not be started
until its dry run reports exactly 40 requests.

The collector uses four global workers by default, while allowing only one
in-flight request per concrete model. HTTP 400/403/404 opens a circuit for that
model, so its remaining requests are skipped without more paid calls. Empty
completions use a separate two-attempt limit and open a model circuit only
after two consecutive exhausted tasks.

## API key

Create `.env.local` in the project root:

```dotenv
ZZZ_API_KEY=replace-with-your-key
ZZZ_BASE_URL=https://api.zhizengzeng.com/v1
```

Never upload `.env.local` to the server through Git and never include it in a
commit. Create it directly on each machine.

## Local Windows dry run

Run this as one PowerShell line:

```powershell
C:\Users\zx\.conda\envs\dl\python.exe -m scripts.collect_attribution_run --config configs\text_attribution\runs\smoke_v2.json --dry-run
```

Confirm:

```text
request_count: 40
prompt_count: 2
requests_per_model: 2
output_budgets: 1536 -> 20, 3072 -> 20
```

## Linux server dry run

```bash
python -m scripts.collect_attribution_run \
  --config configs/text_attribution/runs/smoke_v2.json \
  --dry-run
```

## Linux server paid smoke run

Run inside the existing `zx` Conda environment and tmux session:

```bash
python -u -m scripts.collect_attribution_run \
  --config configs/text_attribution/runs/smoke_v2.json \
  --env-file .env.local \
  --confirm \
  --timeout-seconds 900 \
  --max-workers 4 \
  --max-attempts 4 \
  --empty-max-attempts 2 \
  --empty-circuit-threshold 2 \
  --retry-base-seconds 5 \
  --sleep-seconds 0.5
```

Outputs:

```text
data/attribution/raw/zhizengzeng-attribution-smoke-v2.jsonl
data/attribution/errors/zhizengzeng-attribution-smoke-v2.errors.jsonl
```

Running the same command again skips successful request keys and only attempts
requests that do not have a successful record.

## Probe replacement model versions

The `/models` response is not proof that a chat route is callable. Before
editing the production catalog, run the bilingual candidate probe:

```bash
python -u -m scripts.probe_attribution_models \
  --config configs/text_attribution/zhizengzeng_model_candidates.v2.json \
  --env-file .env.local \
  --confirm \
  --max-workers 4 \
  --max-tokens 256 \
  --timeout-seconds 120
```

A candidate is eligible only when its Chinese and English probes both return
non-empty, non-truncated text. The 2026-07-29 probes selected:

```text
Kimi fast:        moonshot-v1-8k
MiMo fast:        mimo-v2.5-pro
Gemini balanced:  gemini-3.1-flash-lite-preview
MiniMax fast:     abab6.5s-chat
MiniMax balanced: MiniMax-Text-01
```

The final Gemini and MiniMax candidates were also tested with the full smoke
prompts and output budgets (Chinese medium/1536 and English long/3072). All
six requests returned non-empty text with `finish_reason=stop`, no reasoning
token consumption, and exact returned routes. Rejected candidates either
returned 400/404, remained upstream-saturated, or consumed most of the output
budget as reasoning and ended with `finish_reason=length`.

`gpt-4.1-nano-2025-04-14` remains in the OpenAI fast role: its isolated empty
completion was transient and both targeted prompts succeeded with the bounded
empty-response retry policy.

## Staged 600-response run

After checking the smoke output and replacing any unavailable routes, validate
the first production batch:

```bash
python -m scripts.collect_attribution_run \
  --config configs/text_attribution/runs/server_600_v2.json \
  --dry-run
```

The preview must report:

```text
request_count: 600
prompt_count: 10
requests_per_model: 30
```

Then run:

```bash
python -u -m scripts.collect_attribution_run \
  --config configs/text_attribution/runs/server_600_v2.json \
  --env-file .env.local \
  --confirm \
  --timeout-seconds 900 \
  --max-workers 4 \
  --max-attempts 4 \
  --empty-max-attempts 2 \
  --empty-circuit-threshold 2 \
  --retry-base-seconds 5 \
  --sleep-seconds 0.5
```

Outputs:

```text
data/attribution/raw/zhizengzeng-attribution-server-600-v2.jsonl
data/attribution/errors/zhizengzeng-attribution-server-600-v2.errors.jsonl
```

Generate configuration-aware EDA after the collection:

```bash
python -u -m scripts.analyze_attribution_pilot \
  --config configs/text_attribution/runs/server_600_v2.json \
  --input data/attribution/raw/zhizengzeng-attribution-server-600-v2.jsonl \
  --output-dir outputs/attribution/server-600-v2-eda
```

The `--config` argument is required for production decisions: it preserves
models with zero successful rows in expected totals and family coverage.

## Full 6000-response run

Only start this after the 600-response batch passes route, truncation, language,
and failure-rate checks:

```bash
python -m scripts.collect_attribution_run \
  --config configs/text_attribution/runs/server_6k_v2.json \
  --dry-run
```

The preview must report:

```text
request_count: 6000
prompt_count: 100
requests_per_model: 300
```

Run the full collection in tmux:

```bash
python -u -m scripts.collect_attribution_run \
  --config configs/text_attribution/runs/server_6k_v2.json \
  --env-file .env.local \
  --confirm \
  --timeout-seconds 900 \
  --max-workers 4 \
  --max-attempts 4 \
  --empty-max-attempts 2 \
  --empty-circuit-threshold 2 \
  --retry-base-seconds 5 \
  --sleep-seconds 0.5
```

Outputs:

```text
data/attribution/raw/zhizengzeng-attribution-server-6k-v2.jsonl
data/attribution/errors/zhizengzeng-attribution-server-6k-v2.errors.jsonl
```

## Files required on the server

Upload or synchronize:

```text
scripts/collect_attribution_run.py
scripts/probe_attribution_models.py
research/text_attribution/__init__.py
research/text_attribution/config.py
research/text_attribution/gateway.py
research/text_attribution/prompts.py
research/text_attribution/collection.py
research/text_attribution/run_config.py
research/text_attribution/production.py
research/text_attribution/model_probe.py
configs/text_attribution/zhizengzeng_model_catalog.v2.json
configs/text_attribution/zhizengzeng_model_candidates.v2.json
configs/text_attribution/prompt_manifest.smoke-v2.json
configs/text_attribution/prompt_manifest.v2.json
configs/text_attribution/runs/smoke_v2.json
configs/text_attribution/runs/server_600_v2.json
configs/text_attribution/runs/server_6k_v2.json
```

Create `.env.local` directly on the server after upload.
