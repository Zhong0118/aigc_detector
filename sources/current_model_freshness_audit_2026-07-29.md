# Current model freshness audit — 2026-07-29

## Scope

This note compares the current V3 attribution catalog with:

- the gateway-advertised inventory in `zhizengzeng-models.json` exported on
  2026-07-27;
- current official vendor documentation where available.

Gateway inventory is evidence of an advertised route, not proof of the
underlying model identity. Every selected route still requires a response-model
and behavior probe.

## Verified current-release facts

- OpenAI documents `gpt-5.4`, `gpt-5.4-mini`, and their pinned March 2026
  snapshots as current API models.  
  https://developers.openai.com/api/docs/models/gpt-5.4  
  https://openai.com/index/introducing-gpt-5-4/
- Anthropic released Claude Sonnet 5 on 2026-06-30 and Claude Opus 4.8 on
  2026-05-28.  
  https://www.anthropic.com/news/claude-sonnet-5  
  https://www.anthropic.com/news/claude-opus-4-8
- Google lists Gemini 3.5 Flash and 3.6 Flash as current models. The
  `gemini-3.1-flash-lite-preview` route was shut down on 2026-05-25.  
  https://ai.google.dev/gemini-api/docs/changelog  
  https://ai.google.dev/gemini-api/docs/deprecations
- DeepSeek announced V4-Pro and V4-Flash on 2026-04-24. Its legacy
  `deepseek-chat` and `deepseek-reasoner` aliases were scheduled for
  discontinuation on 2026-07-24.  
  https://api-docs.deepseek.com/updates/
- Z.AI released GLM-5 in February 2026 and GLM-5.2 in June 2026.  
  https://z.ai/blog/glm-5  
  https://z.ai/blog/glm-5.2
- MiniMax released M2.7 in March 2026 and exposes it through the API platform.  
  https://www.minimax.io/news/minimax-m27-en
- Kimi's official model list states that K2-series routes were scheduled for
  discontinuation on 2026-05-25; Kimi K2.6 is documented as a later model.
  The gateway also advertises Kimi K3, but that route must be independently
  probed.  
  https://platform.kimi.ai/docs/models  
  https://www.kimi.com/ai-models/kimi-k2-6
- Xiaomi documents MiMo-V2.5 as its current series and deprecated V2 routes in
  June 2026.  
  https://mimo.mi.com/docs/en-US/news/v2.5-news
- ByteDance's Seed 2.0 model card identifies `Doubao-Seed-2.0-pro` as an
  official Volcano Engine route. The exported gateway inventory additionally
  advertises Seed 2.1 routes, which require probing.  
  https://yfz.ai/Seed2.0_Model_Card.pdf
- Qwen's official repositories document Qwen3.5 in February 2026. The exported
  gateway inventory advertises Qwen3.6 and Qwen3.7 routes; these must be
  independently probed because their presence in the proxy list alone is not
  sufficient identity evidence.  
  https://github.com/QwenLM/Qwen-Agent

## Consequence for the current V3 catalog

The current catalog was optimized for route success, moderate cost, and speed,
not for release freshness. It should not be used as the main current-generation
benchmark without another selection pass. In particular:

- OpenAI GPT-4o-mini and GPT-4.1-mini are prior generations.
- Claude Sonnet 4.6 has been superseded by Sonnet 5.
- Gemini 2.5 Flash-Lite is old and the selected 3.1 preview route is shut down.
- `deepseek-chat` is a discontinued dynamic alias and is not reproducible.
- Qwen 2025 snapshots predate gateway-advertised Qwen3.5+ routes.
- GLM 4.x routes predate GLM-5/5.2.
- MiniMax abab6.5s and Text-01 predate M2.x.
- Moonshot-v1 is legacy and Kimi K2.5 is no longer a current anchor.
- Doubao Seed 1.6 predates Seed 2.x.
- MiMo V2.5 is current and can remain.

The existing 558-row V2 dataset remains useful as a historical-model cohort.
It should be relabeled with concrete model ID and release cohort rather than
discarded or silently mixed with a current-generation benchmark.
