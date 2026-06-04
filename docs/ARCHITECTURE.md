# AIGC Detector Architecture

## Current Technology Stack

This project is currently a Python-first MVP for AIGC detection and provenance.

| Layer | Technology | Purpose |
| --- | --- | --- |
| Backend API | FastAPI | Receives text/files, runs analysis workflow, returns JSON results |
| Frontend demo | Streamlit | Provides an interactive dashboard/workbench for fast demonstration |
| Database | SQLite + SQLAlchemy | Stores content records, detection results, and provenance records |
| Configuration | YAML | Stores thresholds, ports, provider switches, and future model settings |
| Detection adapters | Python classes | Wrap API-first detection providers and future local models |
| Provenance | Local Python modules | Runs C2PA placeholder parsing, watermark placeholders, attribution hints |
| Report generation | Template generator now, LLM adapter later | Converts structured evidence into user-readable reports |

The frontend is not raw HTML. It is a Streamlit app, which is useful for fast product demos because it can build upload forms, tabs, metrics, charts, and JSON views directly from Python.

## Runtime Ports

| Service | Port | File | Role |
| --- | ---: | --- | --- |
| FastAPI backend | 8000 | `api/main.py` | Main backend API |
| Streamlit dashboard | 8501 | `api/dashboard.py` | Browser-based demo UI |

Typical startup commands:

```bash
powershell -ExecutionPolicy Bypass -File scripts/run_api.ps1
powershell -ExecutionPolicy Bypass -File scripts/run_dashboard.ps1
```

Do not use `uvicorn --reload` during model/data testing. The project stores
Hugging Face caches and trained artifacts under `models/`, so reload mode can
watch dataset downloads and repeatedly restart the backend while the frontend is
submitting requests.

## High-Level Flow

```text
User text/file input
-> Streamlit dashboard
-> FastAPI endpoint
-> AnalysisService
-> ingestion: modality + metadata + fingerprint
-> detection: API-first provider results
-> provenance: threshold-gated C2PA/watermark/attribution
-> reports: explanation report
-> storage: SQLite persistence
-> JSON response
-> Streamlit visualization
```

FastAPI is the system boundary. Streamlit calls FastAPI endpoints and displays the returned result. The detector/provider layer is intentionally separate, so later you can replace the demo provider with Hive, Sightengine, or local small models without rewriting the dashboard.

## Directory Guide

### `api/`

Backend and dashboard entrypoints.

| File | Purpose |
| --- | --- |
| `api/main.py` | Creates the FastAPI app, initializes the database on startup, mounts routes under `/api/v1`, and exposes `/health`. |
| `api/routes.py` | Defines HTTP endpoints such as `/providers`, `/analyze/text`, `/analyze/file`, `/detect`, `/provenance`, and `/report/{content_id}`. |
| `api/analysis.py` | Central workflow service. It connects ingestion, detection, provenance, report generation, and database storage. |
| `api/dashboard.py` | Streamlit UI. It provides text input, file upload, provider status, score display, provenance panel, report panel, and raw JSON view. |
| `api/__init__.py` | Marks `api` as a Python package. |

### `ingestion/`

Input loading, type detection, metadata extraction, and fingerprinting.

| File | Purpose |
| --- | --- |
| `ingestion/loader.py` | Detects content modality from extension and loads text, image, audio, or video into a `ContentItem`. |
| `ingestion/metadata.py` | Extracts file size, extension, timestamps, and basic image metadata/EXIF. |
| `ingestion/fingerprint.py` | Computes modality-aware fingerprints: normalized text hash, image perceptual hash, audio MFCC hash, video frame hash. |
| `ingestion/__init__.py` | Lazy exports ingestion helpers so routes/services can import them simply. |

### `detection/`

AIGC detection layer. This is where current API-first logic and future local models live.

| File | Purpose |
| --- | --- |
| `detection/providers.py` | Main API-first detection adapter. It currently provides a deterministic `demo_api` provider and reserves Hive, Sightengine, and local model interfaces. |
| `detection/fusion.py` | Weighted score fusion utility for combining modality/provider scores. Older local-model code can still use it. |
| `detection/text_detector.py` | Local text detector scaffold based on perplexity and burstiness using `distilgpt2`. This is not the current default MVP path. |
| `detection/image_detector.py` | Local image detector scaffold using torchvision model structure plus pixel artifact features. |
| `detection/audio_detector.py` | Local audio detector scaffold using mel spectrogram features and a small CNN. |
| `detection/video_detector.py` | Local video detector scaffold that samples frames and reuses the image detector. |
| `detection/__init__.py` | Lazy exports detector classes. |

Current MVP behavior:

```text
detection.mode = api_first
demo_provider_enabled = true
local_models.enabled = false
```

This means the app produces full end-to-end results immediately. Later, real API calls or local model weights can be added behind the same interface.

### `provenance/`

Evidence and source-attribution layer.

| File | Purpose |
| --- | --- |
| `provenance/pipeline.py` | Orchestrates provenance checks after detection. It uses the deep provenance threshold to decide whether to run deeper checks. |
| `provenance/c2pa_reader.py` | Local C2PA placeholder parser for supported image metadata boxes. Can later be replaced by `c2patool` for broader support. |
| `provenance/watermark_decoder.py` | Current simple image LSB watermark placeholder. Future Meta Seal/TextSeal/AudioSeal/VideoSeal adapters should fit here. |
| `provenance/attribution.py` | Local model-attribution scaffold using an MLP and known model list. Not the default source attribution path yet. |
| `provenance/fingerprint_registry.py` | Placeholder area for matching fingerprints against known content/model registries. |
| `provenance/__init__.py` | Lazy exports provenance helpers. |

Important distinction:

```text
C2PA: checks signed content credentials / provenance manifests.
Meta Seal family: checks watermark signals when content was watermarked by compatible methods.
Detection API: estimates whether content appears AI-generated.
```

These are complementary evidence sources, not substitutes for each other.

### `reports/`

User-facing report generation.

| File | Purpose |
| --- | --- |
| `reports/generator.py` | Converts structured detection/provenance data into summary, evidence, limitations, and recommendation fields. |
| `reports/__init__.py` | Marks `reports` as a Python package. |

Current report generation is template-based. The config already reserves an LLM provider:

```yaml
report:
  provider: "template"
  llm_provider: "openai"
  api_key_env: "OPENAI_API_KEY"
```

The intended future behavior is: do not ask an LLM to guess whether content is AI. Instead, pass structured detection and provenance results to the LLM and ask it to write a readable explanation.

### `storage/`

Persistence layer.

| File | Purpose |
| --- | --- |
| `storage/database.py` | Loads the SQLite path from config, creates the engine/session factory, and initializes tables. |
| `storage/models.py` | Defines SQLAlchemy models: `Content`, `DetectionResult`, and `ProvenanceRecord`. |
| `storage/__init__.py` | Marks `storage` as a Python package. |

Stored data includes file identity, modality, fingerprint, AI score, label, detection details, C2PA/watermark data, attribution hints, and timestamps.

### `scripts/`

Training entrypoints for later local-model work.

| File | Purpose |
| --- | --- |
| `scripts/train_text.py` | Future text detector training script. |
| `scripts/train_image.py` | Future image detector training script. |
| `scripts/train_audio.py` | Future audio detector training script. |
| `scripts/train_video.py` | Future video detector training script. |

These scripts are not required for the API-first MVP.

### `docs/`

Project planning and explanation.

| File | Purpose |
| --- | --- |
| `docs/SCHEDULE.md` | Implementation plan, ports, API surface, provider plan, and execution schedule. |
| `docs/ARCHITECTURE.md` | This document. Explains the stack, folders, files, and system flow. |

### Root files

| File | Purpose |
| --- | --- |
| `AGENT.md` | Original product and architecture requirement document. |
| `config.yaml` | Runtime configuration: database path, detection mode, thresholds, provider switches, provenance settings, report settings, ports. |
| `requirements.txt` | Python dependencies for backend, dashboard, ML scaffolds, and storage. |
| `.gitignore` | Ignore Python caches, environments, data outputs, logs, and local code index. |

## Current API Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Check backend health |
| `GET` | `/api/v1/providers` | Return provider configuration/status |
| `POST` | `/api/v1/analyze/text` | Analyze text input |
| `POST` | `/api/v1/analyze/file` | Analyze uploaded file |
| `POST` | `/api/v1/detect` | Backward-compatible file analysis endpoint |
| `POST` | `/api/v1/provenance` | Backward-compatible provenance-focused endpoint |
| `GET` | `/api/v1/report/{content_id}` | Load stored database report |

## Extension Plan

### Add a real detection API

1. Add credentials as environment variables, such as `HIVE_API_KEY`.
2. Set the provider to enabled in `config.yaml`.
3. Implement the real HTTP request inside `detection/providers.py`.
4. Normalize the response into `ProviderResult`.

### Add a local model

1. Keep `local_models.enabled` false until the model is ready.
2. Add a model-specific adapter under `detection/`.
3. Return the same `ProviderResult` shape.
4. Enable the adapter in config.

### Add LLM reports

1. Keep detection/provenance as structured evidence.
2. Add an LLM report adapter in `reports/`.
3. Send evidence JSON to the LLM.
4. Require structured output with fields like `summary`, `evidence`, `limitations`, and `recommendation`.

## Current Design Choice

The current project is intentionally:

```text
API-first
local-model-ready
provenance-aware
Streamlit-demonstrable
FastAPI-stable
```

This gives fast visible progress now, while preserving a path toward a more rigorous multi-model detection system later.
