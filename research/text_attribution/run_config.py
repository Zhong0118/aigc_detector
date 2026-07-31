"""Validated model, prompt, and run manifests for attribution collection."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .collection import CollectionParameters
from .prompts import PromptSpec


@dataclass(frozen=True)
class ModelSelection:
    family_id: str
    provider: str
    model_id: str
    enabled: bool
    fixed_route: bool
    reasoning_mode: str
    protocol: str
    roles: tuple[str, ...]
    variant_role: str


@dataclass(frozen=True)
class RunSampling:
    sampling_id: str
    temperature: float
    top_p: float | None
    repetitions: int
    max_tokens: int | None
    prompt_ids: tuple[str, ...] | None

    def to_collection_parameters(self, *, max_tokens: int) -> CollectionParameters:
        return CollectionParameters(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=max_tokens,
            sampling_id=self.sampling_id,
        )


@dataclass(frozen=True)
class ResolvedRunConfig:
    source_path: Path
    run_id: str
    gateway: str
    models: tuple[ModelSelection, ...]
    prompts: tuple[PromptSpec, ...]
    sampling: tuple[RunSampling, ...]
    repetitions: int | None
    max_tokens_by_target_length: dict[str, int]
    max_requests: int

    @property
    def request_count(self) -> int:
        selected_prompt_counts = (
            len(profile.prompt_ids)
            if profile.prompt_ids is not None
            else len(self.prompts)
            for profile in self.sampling
        )
        return len(self.models) * sum(
            count * profile.repetitions
            for count, profile in zip(selected_prompt_counts, self.sampling)
        )

    @property
    def requests_per_model(self) -> int:
        if not self.models:
            return 0
        return self.request_count // len(self.models)


def load_model_catalog(path: str | Path) -> tuple[ModelSelection, ...]:
    payload = _load_json(Path(path))
    rows = payload.get("models")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Model catalog must contain a non-empty `models` list")
    models: list[ModelSelection] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Every model catalog entry must be an object")
        roles = row.get("roles")
        if not isinstance(roles, list) or not roles:
            raise ValueError("Every model catalog entry must have at least one role")
        model = ModelSelection(
            family_id=_required_string(row, "family_id"),
            provider=_required_string(row, "provider"),
            model_id=_required_string(row, "model_id"),
            enabled=bool(row.get("enabled", False)),
            fixed_route=bool(row.get("fixed_route", False)),
            reasoning_mode=str(row.get("reasoning_mode", "unknown")),
            protocol=str(row.get("protocol", "openai_chat_completions")),
            roles=tuple(_required_list_strings(roles, "roles")),
            variant_role=str(row.get("variant_role", "anchor")).strip() or "anchor",
        )
        models.append(model)
    _reject_duplicates([model.model_id for model in models], "model_id")
    _reject_duplicates(
        [f"{model.family_id}\0{model.variant_role}" for model in models],
        "family_id and variant_role combinations",
    )
    return tuple(models)


def load_prompt_manifest(path: str | Path) -> tuple[PromptSpec, ...]:
    payload = _load_json(Path(path))
    rows = payload.get("prompts")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Prompt manifest must contain a non-empty `prompts` list")
    prompts: list[PromptSpec] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Every prompt manifest entry must be an object")
        system_prompt = row.get("system_prompt")
        if system_prompt is not None and not isinstance(system_prompt, str):
            raise ValueError("system_prompt must be a string or null")
        prompts.append(
            PromptSpec(
                prompt_id=_required_string(row, "prompt_id"),
                language=_required_string(row, "language"),
                domain=_required_string(row, "domain"),
                task_type=_required_string(row, "task_type"),
                target_length=_required_string(row, "target_length"),
                user_prompt=_required_string(row, "user_prompt"),
                system_prompt=system_prompt,
                version=_required_string(row, "version"),
                semantic_group_id=str(
                    row.get("semantic_group_id") or row.get("prompt_id")
                ),
                language_origin=str(row.get("language_origin", "unspecified")),
                target_length_spec=(
                    str(row["target_length_spec"])
                    if row.get("target_length_spec") is not None
                    else None
                ),
                format_type=str(row.get("format_type", "natural")),
                system_prompt_type=str(row.get("system_prompt_type", "none")),
                difficulty=str(row.get("difficulty", "core")),
                max_output_tokens=(
                    int(row["max_output_tokens"])
                    if row.get("max_output_tokens") is not None
                    else None
                ),
                quality_checks=tuple(
                    _optional_list_strings(row.get("quality_checks"), "quality_checks")
                ),
            )
        )
    _reject_duplicates([prompt.prompt_id for prompt in prompts], "prompt_id")
    return tuple(prompts)


def load_run_config(
    path: str | Path,
    *,
    allow_disabled: bool = False,
) -> ResolvedRunConfig:
    source_path = Path(path).resolve()
    payload = _load_json(source_path)
    if not bool(payload.get("enabled", False)) and not allow_disabled:
        raise ValueError(f"Run profile is disabled: {source_path}")

    catalog_path = _resolve_manifest_path(
        source_path, _required_string(payload, "model_catalog")
    )
    prompt_path = _resolve_manifest_path(
        source_path, _required_string(payload, "prompt_manifest")
    )
    roles = set(_required_list_strings(payload.get("model_roles"), "model_roles"))
    models = tuple(
        model
        for model in load_model_catalog(catalog_path)
        if model.enabled and roles.intersection(model.roles)
    )
    if not models:
        raise ValueError("Run profile selects no enabled models")

    available_prompts = load_prompt_manifest(prompt_path)
    prompt_ids = _required_list_strings(payload.get("prompt_ids"), "prompt_ids")
    if prompt_ids == ["*"]:
        prompts = available_prompts
    else:
        by_id = {prompt.prompt_id: prompt for prompt in available_prompts}
        unknown = [prompt_id for prompt_id in prompt_ids if prompt_id not in by_id]
        if unknown:
            raise ValueError(f"Unknown prompt_id values: {', '.join(unknown)}")
        prompts = tuple(by_id[prompt_id] for prompt_id in prompt_ids)

    minimum_prompt_count = int(payload.get("minimum_prompt_count", 1))
    if len(prompts) < minimum_prompt_count:
        raise ValueError(
            f"Run profile requires at least {minimum_prompt_count} prompts; "
            f"resolved {len(prompts)}"
        )
    raw_repetitions = payload.get("repetitions")
    repetitions = int(raw_repetitions) if raw_repetitions is not None else None
    if repetitions is not None and repetitions <= 0:
        raise ValueError("repetitions must be positive")
    sampling = _load_sampling(
        payload.get("sampling"),
        default_repetitions=repetitions,
    )
    max_tokens_by_target_length = _load_length_budgets(
        payload.get("max_tokens_by_target_length")
    )
    selected_prompt_ids = {prompt.prompt_id for prompt in prompts}
    unknown_sampling_prompt_ids = sorted(
        {
            prompt_id
            for profile in sampling
            for prompt_id in (profile.prompt_ids or ())
            if prompt_id not in selected_prompt_ids
        }
    )
    if unknown_sampling_prompt_ids:
        raise ValueError(
            "Unknown sampling prompt_id values: "
            + ", ".join(unknown_sampling_prompt_ids)
        )
    max_requests = int(payload.get("max_requests", 0))
    if max_requests <= 0:
        raise ValueError("max_requests must be positive")
    if any(
        prompt.target_length not in max_tokens_by_target_length
        and prompt.max_output_tokens is None
        and profile.max_tokens is None
        for prompt in prompts
        for profile in sampling
    ):
        raise ValueError(
            "Every request needs max_tokens from its prompt or sampling profile"
        )

    resolved = ResolvedRunConfig(
        source_path=source_path,
        run_id=_required_string(payload, "run_id"),
        gateway=_required_string(payload, "gateway"),
        models=models,
        prompts=prompts,
        sampling=sampling,
        repetitions=repetitions,
        max_tokens_by_target_length=max_tokens_by_target_length,
        max_requests=max_requests,
    )
    if resolved.request_count != max_requests:
        raise ValueError(
            "Cartesian product does not match max_requests: "
            f"{resolved.request_count} != {max_requests}"
        )
    return resolved


def _load_length_budgets(value: Any) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("max_tokens_by_target_length must be an object")
    budgets = {str(key).strip(): int(tokens) for key, tokens in value.items()}
    if any(not key or tokens <= 0 for key, tokens in budgets.items()):
        raise ValueError(
            "max_tokens_by_target_length needs non-blank keys and positive values"
        )
    return budgets


def _load_sampling(
    value: Any,
    *,
    default_repetitions: int | None,
) -> tuple[RunSampling, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("sampling must be a non-empty list")
    rows: list[RunSampling] = []
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            raise ValueError("Every sampling entry must be an object")
        temperature = float(row.get("temperature", -1))
        raw_top_p = row.get("top_p")
        top_p = None if raw_top_p is None else float(raw_top_p)
        raw_max_tokens = row.get("max_tokens")
        max_tokens = int(raw_max_tokens) if raw_max_tokens is not None else None
        raw_profile_repetitions = row.get("repetitions", default_repetitions)
        repetitions = (
            int(raw_profile_repetitions)
            if raw_profile_repetitions is not None
            else 0
        )
        sampling_id = str(row.get("sampling_id", f"sampling-{index}")).strip()
        raw_prompt_ids = row.get("prompt_ids")
        prompt_ids = (
            tuple(_required_list_strings(raw_prompt_ids, "sampling.prompt_ids"))
            if raw_prompt_ids is not None
            else None
        )
        if (
            not sampling_id
            or temperature < 0
            or (top_p is not None and not 0 < top_p <= 1)
            or (max_tokens is not None and max_tokens <= 0)
            or repetitions <= 0
        ):
            raise ValueError(
                "Invalid sampling_id, temperature, top_p, max_tokens, or repetitions"
            )
        rows.append(
            RunSampling(
                sampling_id=sampling_id,
                temperature=temperature,
                top_p=top_p,
                repetitions=repetitions,
                max_tokens=max_tokens,
                prompt_ids=prompt_ids,
            )
        )
    _reject_duplicates([row.sampling_id for row in rows], "sampling_id")
    return tuple(rows)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Configuration file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration root must be an object: {path}")
    return payload


def _resolve_manifest_path(run_path: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (run_path.parent / candidate).resolve()


def _required_string(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-blank string")
    return value.strip()


def _required_list_strings(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    rows = [str(item).strip() for item in value]
    if any(not item for item in rows):
        raise ValueError(f"{field} must not contain blank values")
    return rows


def _optional_list_strings(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    rows = [str(item).strip() for item in value]
    if any(not item for item in rows):
        raise ValueError(f"{field} must not contain blank values")
    return rows


def _reject_duplicates(values: list[str], field: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {field} values are not allowed")
