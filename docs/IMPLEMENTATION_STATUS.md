# Implementation Status

## Short Answer

The current system is a runnable framework, not a fully wired production detector.

It currently uses:

- Real FastAPI routes.
- Real Streamlit dashboard.
- Real ingestion, metadata extraction, fingerprinting, database writes, and report formatting.
- A deterministic `demo_api` detection provider so the full workflow can run immediately.

It does not yet use:

- A real Hive/Sightengine detection API request.
- A real C2PA library or `c2patool` subprocess.
- A real Meta Seal/TextSeal/AudioSeal/VideoSeal model.
- A real LLM report API.

Those interfaces are reserved so they can be added without changing the dashboard or the overall workflow.

## Actual Current Flow

```text
api/routes.py
-> api/analysis.py
-> ingestion/load_content()
-> ingestion/extract_metadata()
-> ingestion/compute_fingerprint()
-> detection/ApiFirstDetectionEngine.detect()
-> provenance/ProvenancePipeline.analyze()
-> reports/ReportGenerator.generate()
-> storage/SQLAlchemy models
```

## Where File Type Detection Happens

File type detection currently happens in:

```text
ingestion/loader.py
```

The function is:

```python
detect_modality(path: Path) -> str
```

It currently detects modality by file extension:

```text
text: .txt, .md, .csv, .json, .html
image: .jpg, .jpeg, .png, .bmp, .webp, .tiff
audio: .wav, .mp3, .flac, .ogg, .m4a
video: .mp4, .avi, .mov, .mkv, .webm
```

Next improvement: add MIME sniffing with `python-magic` or file signature checks so renamed files are handled more safely.

## Where Fingerprinting Happens

Fingerprinting currently happens in:

```text
ingestion/fingerprint.py
```

The main function is:

```python
compute_fingerprint(path, modality)
```

Current modality-specific behavior:

| Modality | Fingerprint method |
| --- | --- |
| Text | Normalize lowercase whitespace, then SHA-256 |
| Image | Perceptual hash via `imagehash.phash` |
| Audio | MFCC feature mean, then SHA-256 |
| Video | Sample frame perceptual hashes, combine, then SHA-256 |
| Unknown | Raw file SHA-256 |

The fingerprint is used in:

```text
api/analysis.py
```

Specifically, `AnalysisService.analyze_file_path()` calls:

```python
fingerprint = compute_fingerprint(path, item.modality)
```

Then it returns the fingerprint in the API response and stores it into:

```text
Content.file_hash
Content.fingerprint
```

inside `storage/models.py`.

## Where Detection API Logic Lives

FastAPI itself does not contain model logic.

FastAPI only exposes HTTP routes in:

```text
api/routes.py
```

The actual detection provider interface lives in:

```text
detection/providers.py
```

The key class is:

```python
ApiFirstDetectionEngine
```

Current behavior:

- `demo_api` is implemented and enabled by default.
- `hive` is listed but not actually called yet.
- `sightengine` is listed but not actually called yet.
- `local_models` are listed but disabled by default.

So at this exact moment, the project is not calling a real open-source model API or commercial detection API. It is using `demo_api` to make the pipeline demonstrable.

## Why `demo_api` Exists

`demo_api` exists to produce immediate end-to-end output while the real provider credentials and model choices are still undecided.

It gives deterministic results based on file/text content hash. This is useful for:

- UI testing.
- Backend route testing.
- Database testing.
- Report format testing.
- Showing the workflow before real API keys are added.

It should not be treated as a real AIGC detector.

## Where C2PA Is Implemented

Current file:

```text
provenance/c2pa_reader.py
```

Current status:

- It is a lightweight local placeholder parser.
- It tries to find simple C2PA-like boxes/chunks in JPEG/PNG.
- It is not a complete C2PA implementation.
- It does not call the official `c2patool` yet.

Recommended next step:

```text
provenance/c2pa_tool.py
```

Add a wrapper that calls `c2patool` locally and normalizes the result into the same JSON shape used by `ProvenancePipeline`.

## Where Meta Seal Is Implemented

Current status:

- Meta Seal is not actually implemented yet.
- TextSeal, AudioSeal, and VideoSeal are reserved by name in `provenance/pipeline.py`.
- Images currently use only a simple local LSB placeholder in `provenance/watermark_decoder.py`.

Current file:

```text
provenance/watermark_decoder.py
```

Recommended next step:

```text
provenance/metaseal_adapter.py
```

That adapter should wrap Meta Seal/TextSeal/AudioSeal/VideoSeal model calls and return:

```json
{
  "status": "ok",
  "provider": "AudioSeal",
  "result": {
    "detected": true,
    "confidence": 0.91
  }
}
```

## What Is Real vs Reserved

| Module | Current status |
| --- | --- |
| FastAPI backend | Real |
| Streamlit dashboard | Real |
| SQLite storage | Real |
| Metadata extraction | Real |
| Fingerprinting | Real |
| Modality routing | Real, extension-based |
| `demo_api` detection | Real demo logic, not a real detector |
| Hive API | Reserved |
| Sightengine API | Reserved |
| Local text/image/audio/video models | Scaffolded/reserved |
| C2PA | Placeholder parser only |
| Meta Seal | Reserved |
| LLM report generation | Template fallback only |

## Next Implementation Priorities

1. Add one real detection API provider first, preferably Hive or Sightengine.
2. Replace the C2PA placeholder with a `c2patool` wrapper.
3. Add MIME sniffing to ingestion.
4. Add one real watermark adapter, likely AudioSeal or image/video Meta Seal depending on target demo data.
5. Add an LLM report adapter after detection/provenance evidence is stable.

