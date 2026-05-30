# AIGC Detection MVP Schedule

## Goal

Build an API-first AIGC detection and provenance framework that can produce a complete demo quickly, while keeping clean extension points for local models and future provenance tools.

## Product Flow

```text
Text input or file upload
-> modality detection
-> metadata + fingerprint
-> API-first detection
-> threshold gate
-> provenance checks
-> LLM-style report generation
-> dashboard + JSON report
```

## Ports

| Service | Port | Command |
| --- | ---: | --- |
| FastAPI backend | 8000 | `uvicorn api.main:app --reload --host 127.0.0.1 --port 8000` |
| Streamlit dashboard | 8501 | `streamlit run api/dashboard.py --server.port 8501` |

## API Surface

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Backend health check |
| `GET /api/v1/providers` | Show configured detection/report providers |
| `POST /api/v1/analyze/text` | Analyze pasted text |
| `POST /api/v1/analyze/file` | Analyze uploaded file |
| `POST /api/v1/detect` | Backward-compatible file analysis endpoint |
| `POST /api/v1/provenance` | Backward-compatible provenance-only endpoint |
| `GET /api/v1/report/{content_id}` | Load stored content report |

## Architecture Layers

| Layer | Responsibility | Current Strategy |
| --- | --- | --- |
| Ingestion | Detect modality, read metadata, compute fingerprint | Extension + metadata + perceptual/hash features |
| Detection | Decide AI vs human | API-first providers, local model adapters reserved |
| Fusion | Combine provider scores | Weighted average / fallback average |
| Threshold Gate | Decide whether to run deep provenance | Run deep checks when score reaches configured threshold |
| Provenance | C2PA, watermark, fingerprint/source hints | Local C2PA parser, Meta Seal interface reserved |
| Report | Human-readable explanation | Template report now, LLM API interface reserved |
| Dashboard | Operable workbench | Text/file input, evidence, provenance, report, history |

## Provider Plan

| Provider | Type | Status |
| --- | --- | --- |
| Demo API provider | Detection | Enabled by default for fast demos |
| Hive | Detection API | Interface reserved via env key |
| Sightengine | Detection API | Interface reserved via env key |
| Local text/image/audio/video detectors | Local models | Disabled by default, adapters reserved |
| C2PA | Local provenance | Enabled when compatible metadata exists |
| Meta Seal / AudioSeal / TextSeal | Local watermark | Interface reserved |
| OpenAI / DeepSeek / Qwen | Report LLM | Interface reserved, template fallback enabled |

## Configuration Memory

```text
Detection threshold: 0.50
Deep provenance threshold: 0.60
Default mode: api_first
Local model default: disabled
Dashboard API base: http://localhost:8000/api/v1
```

## Execution Steps

1. Create this schedule document.
2. Add provider/result schemas and API-first detector adapters.
3. Add provenance orchestration with threshold gate.
4. Add report generator with LLM API interface and local fallback.
5. Update backend routes to expose unified text/file analysis.
6. Redesign Streamlit dashboard around the unified workflow.
7. Run compile and service-level smoke tests.

