"""Bilingual connectivity probes for concrete attribution model routes."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Callable


@dataclass(frozen=True)
class ModelCandidate:
    family_id: str
    model_id: str
    replacement_for: str
    variant_role: str


@dataclass(frozen=True)
class CandidateManifest:
    gateway: str
    candidates: tuple[ModelCandidate, ...]


_PROBE_PROMPTS = (
    (
        "zh",
        "请用两句简短中文说明为什么正式采集前需要测试模型接口。只输出正文。",
    ),
    (
        "en",
        "In two short English sentences, explain why a model route should be "
        "tested before a production data run. Output only the answer.",
    ),
)


def load_candidate_manifest(path: str | Path) -> CandidateManifest:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Candidate manifest must be a JSON object")
    gateway = _required_string(payload, "gateway")
    rows = payload.get("candidates")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Candidate manifest needs a non-empty candidates list")
    candidates = tuple(
        ModelCandidate(
            family_id=_required_string(row, "family_id"),
            model_id=_required_string(row, "model_id"),
            replacement_for=_required_string(row, "replacement_for"),
            variant_role=_required_string(row, "variant_role"),
        )
        for row in rows
        if isinstance(row, dict)
    )
    if len(candidates) != len(rows):
        raise ValueError("Every candidate row must be a JSON object")
    model_ids = [candidate.model_id for candidate in candidates]
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("Candidate model_id values must be unique")
    return CandidateManifest(gateway=gateway, candidates=candidates)


def probe_candidates(
    manifest: CandidateManifest,
    *,
    gateway_factory: Callable[[], Any],
    max_workers: int = 4,
    max_tokens: int = 256,
) -> list[dict[str, Any]]:
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    indexed_results: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="model-probe",
    ) as executor:
        futures = {
            executor.submit(
                _probe_candidate,
                candidate,
                gateway_factory=gateway_factory,
                max_tokens=max_tokens,
            ): index
            for index, candidate in enumerate(manifest.candidates)
        }
        for future in as_completed(futures):
            indexed_results[futures[future]] = future.result()
    return [indexed_results[index] for index in range(len(manifest.candidates))]


def _probe_candidate(
    candidate: ModelCandidate,
    *,
    gateway_factory: Callable[[], Any],
    max_tokens: int,
) -> dict[str, Any]:
    gateway = gateway_factory()
    probes = [
        _probe_language(
            candidate,
            gateway=gateway,
            language=language,
            prompt=prompt,
            max_tokens=max_tokens,
        )
        for language, prompt in _PROBE_PROMPTS
    ]
    return {
        "family_id": candidate.family_id,
        "model_id": candidate.model_id,
        "replacement_for": candidate.replacement_for,
        "variant_role": candidate.variant_role,
        "eligible": all(probe["ok"] for probe in probes),
        "probes": probes,
    }


def _probe_language(
    candidate: ModelCandidate,
    *,
    gateway: Any,
    language: str,
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    try:
        result = gateway.generate(
            model=candidate.model_id,
            user_prompt=prompt,
            system_prompt=None,
            temperature=0.0,
            top_p=None,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        secret = getattr(getattr(gateway, "config", None), "api_key", "")
        return {
            "language": language,
            "ok": False,
            "http_status": _status_code(exc),
            "error_type": type(exc).__name__,
            "error_message": _redact(str(exc), secret),
            "response_body_excerpt": _response_body_excerpt(exc, secret),
        }

    record = result.to_record()
    finish_reason = record.get("finish_reason")
    text = str(record.get("text") or "")
    return {
        "language": language,
        "ok": bool(text.strip()) and finish_reason != "length",
        "http_status": None,
        "finish_reason": finish_reason,
        "text": text,
        "elapsed_seconds": record.get("elapsed_seconds"),
        "usage": record.get("usage") or {},
        "response_model": record.get("response_model"),
        "route_mismatch": bool(
            record.get("response_model")
            and record.get("response_model") != candidate.model_id
        ),
    }


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _status_code(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return int(status) if isinstance(status, int) else None


def _redact(value: str, secret: str) -> str:
    if secret:
        value = value.replace(secret, "<redacted>")
    return re.sub(r"sk-[A-Za-z0-9_-]+", "<redacted>", value)


def _response_body_excerpt(
    exc: Exception,
    secret: str,
    *,
    limit: int = 1000,
) -> str | None:
    response = getattr(exc, "response", None)
    if response is None:
        return None
    try:
        value = str(response.text).strip()
    except Exception:
        return None
    if not value:
        return None
    return _redact(value[:limit], secret)
