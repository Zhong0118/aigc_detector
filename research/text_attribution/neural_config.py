from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


_EXPERIMENT_IDS = ("N0", "N1", "N2")
_COMMON_FIELDS = frozenset(
    {
        "learning_rate",
        "epochs",
        "weight_decay",
        "warmup_ratio",
        "label_smoothing_factor",
        "gradient_accumulation_steps",
        "early_stopping_patience",
        "save_total_limit",
        "bf16",
    }
)
_EXPERIMENT_REQUIRED_FIELDS = frozenset(
    {"model_id", "max_length", "per_device_train_batch_size"}
)
_EXPERIMENT_FIELDS = _COMMON_FIELDS | _EXPERIMENT_REQUIRED_FIELDS
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "seed",
        "model_catalog",
        "common",
        "experiments",
    }
)


@dataclass(frozen=True)
class NeuralExperimentConfig:
    model_id: str
    max_length: int
    per_device_train_batch_size: int
    learning_rate: float
    epochs: int
    weight_decay: float
    warmup_ratio: float
    label_smoothing_factor: float
    gradient_accumulation_steps: int
    early_stopping_patience: int
    save_total_limit: int
    bf16: bool


@dataclass(frozen=True)
class NeuralScreeningConfig:
    source_path: Path
    schema_version: int
    run_id: str
    seed: int
    model_catalog: Path
    experiments: Mapping[str, NeuralExperimentConfig]


def load_neural_screening_config(
    path: str | Path,
) -> NeuralScreeningConfig:
    source_path = Path(path).resolve()
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Could not load neural screening config {source_path}: {error}"
        ) from error

    if not isinstance(payload, dict):
        raise ValueError("Neural screening config must be a JSON object")
    _require_exact_fields(payload, _TOP_LEVEL_FIELDS, "top-level")

    schema_version = _require_int(payload, "schema_version")
    if schema_version != 1:
        raise ValueError("schema_version must be exactly 1")
    run_id = _require_string(payload, "run_id")
    seed = _require_int(payload, "seed")
    model_catalog_value = _require_string(payload, "model_catalog")

    common = payload["common"]
    if not isinstance(common, dict):
        raise ValueError("common must be an object")
    _require_exact_fields(common, _COMMON_FIELDS, "common")
    _validate_common(common)

    experiments_payload = payload["experiments"]
    if not isinstance(experiments_payload, dict):
        raise ValueError("experiments must be an object")
    actual_ids = set(experiments_payload)
    expected_ids = set(_EXPERIMENT_IDS)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        unknown = sorted(actual_ids - expected_ids)
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unknown:
            details.append(f"unknown {unknown}")
        raise ValueError(
            "experiments must contain exactly N0, N1, and N2"
            f" ({'; '.join(details)})"
        )

    experiments: dict[str, NeuralExperimentConfig] = {}
    for experiment_id in _EXPERIMENT_IDS:
        row = experiments_payload[experiment_id]
        if not isinstance(row, dict):
            raise ValueError(f"experiment {experiment_id} must be an object")
        _reject_unknown_fields(row, _EXPERIMENT_FIELDS, f"experiment {experiment_id}")
        missing_specific = _EXPERIMENT_REQUIRED_FIELDS - row.keys()
        if missing_specific:
            raise ValueError(
                f"Missing experiment {experiment_id} field(s): "
                f"{', '.join(sorted(missing_specific))}"
            )
        merged = {**common, **row}
        experiments[experiment_id] = _parse_experiment(
            experiment_id,
            merged,
        )

    return NeuralScreeningConfig(
        source_path=source_path,
        schema_version=schema_version,
        run_id=run_id,
        seed=seed,
        model_catalog=(source_path.parent / model_catalog_value).resolve(),
        experiments=MappingProxyType(experiments),
    )


def _parse_experiment(
    experiment_id: str,
    values: dict[str, object],
) -> NeuralExperimentConfig:
    prefix = f"experiment {experiment_id}"
    model_id = _require_string(values, "model_id", prefix)
    max_length = _require_positive_int(values, "max_length", prefix)
    batch_size = _require_positive_int(
        values,
        "per_device_train_batch_size",
        prefix,
    )
    learning_rate = _require_float(values, "learning_rate", prefix)
    if learning_rate <= 0:
        raise ValueError(f"{prefix} learning_rate must be positive")
    epochs = _require_positive_int(values, "epochs", prefix)
    weight_decay = _require_float(values, "weight_decay", prefix)
    if weight_decay < 0:
        raise ValueError(f"{prefix} weight_decay must be non-negative")
    warmup_ratio = _require_bounded_ratio(values, "warmup_ratio", prefix)
    smoothing = _require_bounded_ratio(
        values,
        "label_smoothing_factor",
        prefix,
    )
    accumulation = _require_positive_int(
        values,
        "gradient_accumulation_steps",
        prefix,
    )
    patience = _require_positive_int(
        values,
        "early_stopping_patience",
        prefix,
    )
    save_limit = _require_positive_int(values, "save_total_limit", prefix)
    bf16 = values.get("bf16")
    if type(bf16) is not bool:
        raise ValueError(f"{prefix} bf16 must be a bool")

    return NeuralExperimentConfig(
        model_id=model_id,
        max_length=max_length,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        label_smoothing_factor=smoothing,
        gradient_accumulation_steps=accumulation,
        early_stopping_patience=patience,
        save_total_limit=save_limit,
        bf16=bf16,
    )


def _validate_common(values: dict[str, object]) -> None:
    location = "common"
    learning_rate = _require_float(values, "learning_rate", location)
    if learning_rate <= 0:
        raise ValueError("common learning_rate must be positive")
    _require_positive_int(values, "epochs", location)
    weight_decay = _require_float(values, "weight_decay", location)
    if weight_decay < 0:
        raise ValueError("common weight_decay must be non-negative")
    _require_bounded_ratio(values, "warmup_ratio", location)
    _require_bounded_ratio(values, "label_smoothing_factor", location)
    _require_positive_int(values, "gradient_accumulation_steps", location)
    _require_positive_int(values, "early_stopping_patience", location)
    _require_positive_int(values, "save_total_limit", location)
    if type(values.get("bf16")) is not bool:
        raise ValueError("common bf16 must be a bool")


def _require_exact_fields(
    values: dict[str, object],
    expected: frozenset[str],
    location: str,
) -> None:
    _reject_unknown_fields(values, expected, location)
    missing = expected - values.keys()
    if missing:
        raise ValueError(
            f"Missing {location} field(s): {', '.join(sorted(missing))}"
        )


def _reject_unknown_fields(
    values: dict[str, object],
    allowed: frozenset[str],
    location: str,
) -> None:
    unknown = values.keys() - allowed
    if unknown:
        raise ValueError(
            f"Unknown {location} field(s): {', '.join(sorted(unknown))}"
        )


def _require_string(
    values: dict[str, object],
    field: str,
    location: str = "top-level",
) -> str:
    value = values.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} {field} must be a non-empty string")
    return value


def _require_int(
    values: dict[str, object],
    field: str,
    location: str = "top-level",
) -> int:
    value = values.get(field)
    if type(value) is not int:
        raise ValueError(f"{location} {field} must be an int")
    return value


def _require_positive_int(
    values: dict[str, object],
    field: str,
    location: str,
) -> int:
    value = _require_int(values, field, location)
    if value <= 0:
        raise ValueError(f"{location} {field} must be positive")
    return value


def _require_float(
    values: dict[str, object],
    field: str,
    location: str,
) -> float:
    value = values.get(field)
    if type(value) is not float:
        raise ValueError(f"{location} {field} must be a float")
    return value


def _require_bounded_ratio(
    values: dict[str, object],
    field: str,
    location: str,
) -> float:
    value = _require_float(values, field, location)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{location} {field} must be between 0 and 1")
    return value
