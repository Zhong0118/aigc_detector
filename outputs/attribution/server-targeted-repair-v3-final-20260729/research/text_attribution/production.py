"""Resumable, configuration-driven attribution data production."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, Sequence

import requests

from .collection import PlannedRequest, write_jsonl_record
from .run_config import ResolvedRunConfig


@dataclass(frozen=True)
class RunSummary:
    planned: int
    written: int
    skipped: int
    failed: int
    circuit_skipped: int = 0
    stopped_reason: str | None = None


@dataclass(frozen=True)
class _RequestOutcome:
    status: str
    item: PlannedRequest
    key: str
    gateway: Any | None = None
    result: Any | None = None
    error: Exception | None = None
    http_status: int | None = None
    attempts: int = 0
    circuit_reason: str | None = None


class _ModelCircuitState:
    def __init__(self, *, empty_threshold: int) -> None:
        self.empty_threshold = empty_threshold
        self._lock = threading.Lock()
        self._open_reasons: dict[str, str] = {}
        self._consecutive_empty: dict[str, int] = {}

    def open_reason(self, model: str) -> str | None:
        with self._lock:
            return self._open_reasons.get(model)

    def record_success(self, model: str) -> None:
        with self._lock:
            self._consecutive_empty[model] = 0

    def record_terminal_failure(
        self,
        model: str,
        *,
        status: int | None,
        empty_completion: bool,
    ) -> str | None:
        with self._lock:
            if status in {400, 403, 404}:
                reason = f"http_{status}"
                self._open_reasons[model] = reason
                return reason
            if not empty_completion:
                self._consecutive_empty[model] = 0
                return None
            count = self._consecutive_empty.get(model, 0) + 1
            self._consecutive_empty[model] = count
            if count >= self.empty_threshold:
                reason = f"consecutive_empty_{count}"
                self._open_reasons[model] = reason
                return reason
            return None


def build_plan_from_config(config: ResolvedRunConfig) -> list[PlannedRequest]:
    plan: list[PlannedRequest] = []
    for model in config.models:
        for prompt in config.prompts:
            for sampling in config.sampling:
                if (
                    sampling.prompt_ids is not None
                    and prompt.prompt_id not in sampling.prompt_ids
                ):
                    continue
                max_tokens = prompt.max_output_tokens or sampling.max_tokens
                if max_tokens is None:
                    raise ValueError("Request has no resolved max_tokens value")
                parameters = sampling.to_collection_parameters(max_tokens=max_tokens)
                for repeat_index in range(sampling.repetitions):
                    plan.append(
                        PlannedRequest(
                            gateway_name=config.gateway,
                            model=model.model_id,
                            prompt=prompt,
                            parameters=parameters,
                            repeat_index=repeat_index,
                        )
                    )
    if len(plan) != config.request_count:
        raise ValueError(
            f"Resolved plan is incomplete: {len(plan)} != {config.request_count}"
        )
    return plan


def request_key(item: PlannedRequest) -> str:
    payload = {
        "gateway": item.gateway_name,
        "model": item.model,
        "prompt_id": item.prompt.prompt_id,
        "prompt_version": item.prompt.version,
        "temperature": item.parameters.temperature,
        "top_p": item.parameters.top_p,
        "max_tokens": item.parameters.max_tokens,
        "sampling_id": item.parameters.sampling_id,
        "repeat_index": item.repeat_index,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def run_collection(
    config: ResolvedRunConfig,
    *,
    gateway: Any,
    output_path: str | Path,
    error_path: str | Path,
    plan: Sequence[PlannedRequest] | None = None,
    sleep_seconds: float = 0.5,
    max_attempts: int = 3,
    empty_max_attempts: int = 2,
    empty_circuit_threshold: int = 2,
    retry_base_seconds: float = 1.0,
    max_workers: int = 1,
    gateway_factory: Callable[[], Any] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> RunSummary:
    if max_attempts <= 0 or empty_max_attempts <= 0:
        raise ValueError("attempt limits must be positive")
    if empty_circuit_threshold <= 0:
        raise ValueError("empty_circuit_threshold must be positive")
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    if sleep_seconds < 0 or retry_base_seconds < 0:
        raise ValueError("sleep values must be non-negative")
    requests_to_run = list(plan) if plan is not None else build_plan_from_config(config)
    output = Path(output_path)
    errors = Path(error_path)
    completed = _load_completed_keys(output)
    family_by_model = {model.model_id: model.family_id for model in config.models}
    variant_by_model = {
        model.model_id: model.variant_role for model in config.models
    }
    written = 0
    skipped = 0
    failed = 0
    circuit_skipped = 0
    stopped_reason: str | None = None

    pending: list[PlannedRequest] = []
    for item in requests_to_run:
        key = request_key(item)
        if key in completed:
            skipped += 1
            continue
        pending.append(item)

    model_locks = {
        model.model_id: threading.Lock()
        for model in config.models
    }
    circuit = _ModelCircuitState(empty_threshold=empty_circuit_threshold)
    global_stop = threading.Event()
    worker_local = threading.local()

    def worker_gateway() -> Any:
        if gateway_factory is None:
            return gateway
        local_gateway = getattr(worker_local, "gateway", None)
        if local_gateway is None:
            local_gateway = gateway_factory()
            worker_local.gateway = local_gateway
        return local_gateway

    def execute(item: PlannedRequest) -> _RequestOutcome:
        key = request_key(item)
        with model_locks[item.model]:
            if global_stop.is_set():
                return _RequestOutcome(status="global_skipped", item=item, key=key)
            circuit_reason = circuit.open_reason(item.model)
            if circuit_reason:
                return _RequestOutcome(
                    status="circuit_skipped",
                    item=item,
                    key=key,
                    circuit_reason=circuit_reason,
                )

            item_gateway = worker_gateway()
            for attempt in range(1, max_attempts + 1):
                try:
                    result = item_gateway.generate(
                        model=item.model,
                        user_prompt=item.prompt.user_prompt,
                        system_prompt=item.prompt.system_prompt,
                        temperature=item.parameters.temperature,
                        top_p=item.parameters.top_p,
                        max_tokens=item.parameters.max_tokens,
                    )
                except Exception as exc:
                    status = _status_code(exc)
                    empty_completion = _is_empty_completion_error(exc)
                    attempt_limit = (
                        min(max_attempts, empty_max_attempts)
                        if empty_completion
                        else max_attempts
                    )
                    if (
                        _is_retryable(exc, status)
                        and attempt < attempt_limit
                    ):
                        sleeper(retry_base_seconds * (2 ** (attempt - 1)))
                        continue
                    if status == 401:
                        global_stop.set()
                    circuit.record_terminal_failure(
                        item.model,
                        status=status,
                        empty_completion=empty_completion,
                    )
                    return _RequestOutcome(
                        status="failed",
                        item=item,
                        key=key,
                        gateway=item_gateway,
                        error=exc,
                        http_status=status,
                        attempts=attempt,
                    )

                circuit.record_success(item.model)
                if sleep_seconds:
                    sleeper(sleep_seconds)
                return _RequestOutcome(
                    status="written",
                    item=item,
                    key=key,
                    gateway=item_gateway,
                    result=result,
                    attempts=attempt,
                )
        raise RuntimeError("request execution ended without an outcome")

    submission_order = (
        _round_robin_by_model(pending)
        if max_workers > 1
        else pending
    )
    futures: list[Future[_RequestOutcome]] = []
    with ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="attribution-collector",
    ) as executor:
        futures.extend(executor.submit(execute, item) for item in submission_order)
        for future in as_completed(futures):
            outcome = future.result()
            item = outcome.item
            if outcome.status == "global_skipped":
                continue
            if outcome.status == "circuit_skipped":
                circuit_skipped += 1
                if progress:
                    progress(
                        {
                            "status": "circuit_skipped",
                            "run_id": config.run_id,
                            "family_id": family_by_model[item.model],
                            "model": item.model,
                            "prompt_id": item.prompt.prompt_id,
                            "reason": outcome.circuit_reason,
                            "circuit_skipped": circuit_skipped,
                            "planned": len(requests_to_run),
                        }
                    )
                continue
            if outcome.status == "written":
                record = _success_record(
                    config,
                    item,
                    outcome.result,
                    outcome.key,
                    family_by_model[item.model],
                    variant_by_model[item.model],
                )
                write_jsonl_record(output, record)
                completed.add(outcome.key)
                written += 1
                if progress:
                    progress(
                        {
                            "status": "written",
                            "run_id": config.run_id,
                            "family_id": family_by_model[item.model],
                            "model": item.model,
                            "prompt_id": item.prompt.prompt_id,
                            "repeat_index": item.repeat_index,
                            "attempts": outcome.attempts,
                            "written": written,
                            "planned": len(requests_to_run),
                        }
                    )
                continue
            if outcome.status != "failed" or outcome.error is None:
                raise RuntimeError(f"Unexpected request outcome: {outcome.status}")
            write_jsonl_record(
                errors,
                _failure_record(
                    config,
                    item,
                    outcome.key,
                    family_by_model[item.model],
                    variant_by_model[item.model],
                    outcome.error,
                    status=outcome.http_status,
                    attempt=outcome.attempts,
                    gateway=outcome.gateway,
                ),
            )
            failed += 1
            if progress:
                progress(
                    {
                        "status": "failed",
                        "run_id": config.run_id,
                        "family_id": family_by_model[item.model],
                        "model": item.model,
                        "prompt_id": item.prompt.prompt_id,
                        "http_status": outcome.http_status,
                        "attempts": outcome.attempts,
                    }
                )
            if outcome.http_status == 401:
                stopped_reason = "authentication"

    return RunSummary(
        planned=len(requests_to_run),
        written=written,
        skipped=skipped,
        failed=failed,
        circuit_skipped=circuit_skipped,
        stopped_reason=stopped_reason,
    )


def _success_record(
    config: ResolvedRunConfig,
    item: PlannedRequest,
    result: Any,
    key: str,
    family_id: str,
    variant_role: str,
) -> dict[str, Any]:
    result_record = result.to_record()
    return {
        "schema_version": 1,
        "run_id": config.run_id,
        "request_key": key,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "family_id": family_id,
        "model_variant_role": variant_role,
        "repeat_index": item.repeat_index,
        "route_mismatch": bool(
            result.response_model and result.response_model != item.model
        ),
        "prompt": {
            "prompt_id": item.prompt.prompt_id,
            "version": item.prompt.version,
            "language": item.prompt.language,
            "domain": item.prompt.domain,
            "task_type": item.prompt.task_type,
            "target_length": item.prompt.target_length,
            "user_prompt": item.prompt.user_prompt,
            "system_prompt": item.prompt.system_prompt,
            "semantic_group_id": item.prompt.semantic_group_id,
            "language_origin": item.prompt.language_origin,
            "target_length_spec": item.prompt.target_length_spec,
            "format_type": item.prompt.format_type,
            "system_prompt_type": item.prompt.system_prompt_type,
            "difficulty": item.prompt.difficulty,
            "max_output_tokens": item.prompt.max_output_tokens,
            "quality_checks": list(item.prompt.quality_checks),
        },
        "parameters": {
            "sampling_id": item.parameters.sampling_id,
            "temperature": item.parameters.temperature,
            "top_p": item.parameters.top_p,
            "max_tokens": item.parameters.max_tokens,
        },
        **result_record,
    }


def _failure_record(
    config: ResolvedRunConfig,
    item: PlannedRequest,
    key: str,
    family_id: str,
    variant_role: str,
    exc: Exception,
    *,
    status: int | None,
    attempt: int,
    gateway: Any,
) -> dict[str, Any]:
    secret = getattr(getattr(gateway, "config", None), "api_key", "")
    return {
        "schema_version": 1,
        "run_id": config.run_id,
        "request_key": key,
        "failed_at": datetime.now(timezone.utc).isoformat(),
        "gateway": config.gateway,
        "family_id": family_id,
        "model_variant_role": variant_role,
        "request_model": item.model,
        "prompt_id": item.prompt.prompt_id,
        "repeat_index": item.repeat_index,
        "parameters": {
            "sampling_id": item.parameters.sampling_id,
            "temperature": item.parameters.temperature,
            "top_p": item.parameters.top_p,
            "max_tokens": item.parameters.max_tokens,
        },
        "error_type": type(exc).__name__,
        "error_message": _redact(str(exc), secret),
        "response_body_excerpt": _response_body_excerpt(exc, secret),
        "http_status": status,
        "attempts": attempt,
    }


def _status_code(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return int(status) if isinstance(status, int) else None


def _is_retryable(exc: Exception, status: int | None) -> bool:
    if status == 429 or (status is not None and status >= 500):
        return True
    if isinstance(exc, ValueError) and str(exc) in {
        "Chat completion returned empty text",
        "Chat completion response must contain at least one choice",
    }:
        return True
    return isinstance(exc, requests.RequestException) and status is None


def _is_empty_completion_error(exc: Exception) -> bool:
    return isinstance(exc, ValueError) and str(exc) in {
        "Chat completion returned empty text",
        "Chat completion response must contain at least one choice",
    }


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


def _load_completed_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("request_key"):
            completed.add(str(row["request_key"]))
    return completed


def _round_robin_by_model(
    requests_to_run: Sequence[PlannedRequest],
) -> list[PlannedRequest]:
    grouped: dict[str, list[PlannedRequest]] = {}
    for item in requests_to_run:
        grouped.setdefault(item.model, []).append(item)
    ordered: list[PlannedRequest] = []
    largest_group = max((len(items) for items in grouped.values()), default=0)
    for index in range(largest_group):
        for items in grouped.values():
            if index < len(items):
                ordered.append(items[index])
    return ordered
