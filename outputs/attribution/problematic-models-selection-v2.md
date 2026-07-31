# Problematic model replacement selection (2026-07-29)

## Decision

| Family / role | Selected model | Chinese medium prompt | English long prompt | Decision |
|---|---|---:|---:|---|
| Gemini / balanced | `gemini-3.1-flash-lite-preview` | 472 chars, 3.05 s, stop | 5090 chars / 780 words, 5.49 s, stop | accept |
| MiniMax / fast | `abab6.5s-chat` | 568 chars, 4.60 s, stop | 4047 chars / 630 words, 10.23 s, stop | accept |
| MiniMax / balanced | `MiniMax-Text-01` | 598 chars, 7.69 s, stop | 5350 chars / 769 words, 25.23 s, stop | accept |

All selected routes produced non-empty text, used no reasoning tokens, and
returned the exact requested route for both prompts.

## Rejected or deferred candidates

- `gemini-2.0-flash`: HTTP 400; the provider reports that the model is
  deprecated.
- `gemini-2.0-flash-lite`: upstream 429 saturation in both probes.
- `gemini-3.5-flash`: reasoning consumed most of the output budget and both
  prompts ended with `finish_reason=length`.
- `gemini-2.5-flash-preview-09-2025`: HTTP 404.
- `MiniMax-M2-Stable`, `MiniMax-M2.1-lightning`, and
  `MiniMax-M2.5-highspeed`: reasoning-heavy responses and/or
  `finish_reason=length`.
- `abab6-chat` and `abab6.5-chat`: both routes worked, but
  `abab6.5s-chat` offered the best latency and length compliance for the fast
  role.

## OpenAI transient empty response

`gpt-4.1-nano-2025-04-14` succeeded on both targeted prompts (the Chinese
request succeeded on the second attempt). It remains in the catalog. The
collector limits empty-response retries to two attempts and opens a model
circuit after two consecutively exhausted tasks.

## Next gate

Run the 600-response batch, audit route consistency, empty responses,
truncation, language, and family/model balance, and only then enable the
6000-response run.
