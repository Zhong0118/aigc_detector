"""Strict Phase3 configuration and immutable enhancement recipes."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


_BACKBONE = "FacebookAI/xlm-roberta-base"
_CANONICAL_STAGES = ("A0", "A1-alpha01", "A1-alpha03")
_TRANSFORM_STAGES = ("A2", "A3", "A4")
_DATA_VIEWS = ("closed", "fast-to-balanced", "balanced-to-fast")
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "backbone_model_id",
        "model_catalog",
        "max_length",
        "training",
        "canonical_recipes",
        "transforms",
        "screening_seeds",
        "confirmation_seeds",
        "data_views",
        "gates",
    }
)
_TRAINING_FIELDS = frozenset(
    {
        "learning_rate",
        "epochs",
        "weight_decay",
        "warmup_ratio",
        "label_smoothing_factor",
        "early_stopping_patience",
        "save_total_limit",
        "bf16",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "a4_per_device_train_batch_size",
        "a4_gradient_accumulation_steps",
        "a4_low_memory_batch_size",
        "a4_low_memory_gradient_accumulation_steps",
        "projection_dimension",
        "contrastive_temperature",
    }
)
_CANONICAL_RECIPE_FIELDS = frozenset(
    {
        "parent_recipe_id",
        "auxiliary_weight",
        "consistency_weight",
        "contrastive_weight",
        "window_mode",
    }
)
_GATE_FIELDS = frozenset(
    {
        "closed_macro_f1_min_gain",
        "closed_seed_wins_min",
        "worst_family_min_gain",
        "mimo_min_gain",
        "glm_min_gain",
        "holdout_combined_min_gain",
        "holdout_direction_min_gain",
        "forbid_zero_family_f1",
    }
)
_RECIPE_FIELDS = frozenset(
    {
        "stage_id",
        "recipe_id",
        "parent_recipe_id",
        "backbone_model_id",
        "max_length",
        "auxiliary_weight",
        "consistency_weight",
        "contrastive_weight",
        "contrastive_temperature",
        "projection_dimension",
        "window_mode",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "epochs",
        "weight_decay",
        "warmup_ratio",
        "label_smoothing_factor",
        "early_stopping_patience",
        "save_total_limit",
        "bf16",
        "gradient_checkpointing",
        "recipe_hash",
    }
)


@dataclass(frozen=True)
class Phase3TrainingConfig:
    learning_rate: float
    epochs: int
    weight_decay: float
    warmup_ratio: float
    label_smoothing_factor: float
    early_stopping_patience: int
    save_total_limit: int
    bf16: bool
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    a4_per_device_train_batch_size: int
    a4_gradient_accumulation_steps: int
    a4_low_memory_batch_size: int
    a4_low_memory_gradient_accumulation_steps: int
    projection_dimension: int
    contrastive_temperature: float


@dataclass(frozen=True)
class Phase3GateConfig:
    closed_macro_f1_min_gain: float
    closed_seed_wins_min: int
    worst_family_min_gain: float
    mimo_min_gain: float
    glm_min_gain: float
    holdout_combined_min_gain: float
    holdout_direction_min_gain: float
    forbid_zero_family_f1: bool


@dataclass(frozen=True)
class CanonicalRecipeConfig:
    parent_recipe_id: str | None
    auxiliary_weight: float
    consistency_weight: float
    contrastive_weight: float
    window_mode: str


@dataclass(frozen=True)
class Phase3Config:
    source_path: Path
    schema_version: int
    run_id: str
    backbone_model_id: str
    model_catalog: Path
    max_length: int
    training: Phase3TrainingConfig
    canonical_recipes: Mapping[str, CanonicalRecipeConfig]
    transforms: Mapping[str, Mapping[str, object]]
    screening_seeds: tuple[int, ...]
    confirmation_seeds: tuple[int, ...]
    data_views: tuple[str, ...]
    gates: Phase3GateConfig


@dataclass(frozen=True)
class ResolvedPhase3Recipe:
    stage_id: str
    recipe_id: str
    parent_recipe_id: str | None
    backbone_model_id: str
    max_length: int
    auxiliary_weight: float
    consistency_weight: float
    contrastive_weight: float
    contrastive_temperature: float
    projection_dimension: int
    window_mode: str
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    epochs: int
    weight_decay: float
    warmup_ratio: float
    label_smoothing_factor: float
    early_stopping_patience: int
    save_total_limit: int
    bf16: bool
    gradient_checkpointing: bool
    recipe_hash: str

    def to_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "stage_id": self.stage_id,
            "recipe_id": self.recipe_id,
            "parent_recipe_id": self.parent_recipe_id,
            "backbone_model_id": self.backbone_model_id,
            "max_length": self.max_length,
            "auxiliary_weight": self.auxiliary_weight,
            "consistency_weight": self.consistency_weight,
            "contrastive_weight": self.contrastive_weight,
            "contrastive_temperature": self.contrastive_temperature,
            "projection_dimension": self.projection_dimension,
            "window_mode": self.window_mode,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "label_smoothing_factor": self.label_smoothing_factor,
            "early_stopping_patience": self.early_stopping_patience,
            "save_total_limit": self.save_total_limit,
            "bf16": self.bf16,
            "gradient_checkpointing": self.gradient_checkpointing,
        }
        if include_hash:
            payload["recipe_hash"] = self.recipe_hash
        return payload


def load_phase3_config(path: str | Path) -> Phase3Config:
    source_path = Path(path).resolve()
    payload = _read_json_object(source_path, "Phase3 config")
    _require_exact_fields(payload, _TOP_LEVEL_FIELDS, "top-level")

    schema_version = _require_int(payload, "schema_version", "top-level")
    if schema_version != 1:
        raise ValueError("schema_version must be exactly 1")
    run_id = _require_string(payload, "run_id", "top-level")
    backbone = _require_string(payload, "backbone_model_id", "top-level")
    if backbone != _BACKBONE:
        raise ValueError(f"backbone_model_id must be exactly {_BACKBONE}")
    catalog_value = _require_string(payload, "model_catalog", "top-level")
    max_length = _require_positive_int(payload, "max_length", "top-level")
    if max_length != 512:
        raise ValueError("max_length must be exactly 512")

    training = _parse_training(payload["training"])
    canonical = _parse_canonical_recipes(payload["canonical_recipes"])
    transforms = _parse_transforms(payload["transforms"])
    screening_seeds = _require_exact_seed_list(
        payload["screening_seeds"],
        "screening_seeds",
        (42, 52, 62),
    )
    confirmation_seeds = _require_exact_seed_list(
        payload["confirmation_seeds"],
        "confirmation_seeds",
        (12, 22, 32, 72, 82),
    )
    data_views = _require_string_tuple(payload["data_views"], "data_views")
    if data_views != _DATA_VIEWS:
        raise ValueError(f"data_views must be exactly {_DATA_VIEWS}")
    gates = _parse_gates(payload["gates"])

    return Phase3Config(
        source_path=source_path,
        schema_version=schema_version,
        run_id=run_id,
        backbone_model_id=backbone,
        model_catalog=(source_path.parent / catalog_value).resolve(),
        max_length=max_length,
        training=training,
        canonical_recipes=canonical,
        transforms=transforms,
        screening_seeds=screening_seeds,
        confirmation_seeds=confirmation_seeds,
        data_views=data_views,
        gates=gates,
    )


def resolve_phase3_recipe(
    config: Phase3Config,
    stage: str,
    parent_decision: str | Path | None = None,
) -> ResolvedPhase3Recipe:
    if not isinstance(config, Phase3Config):
        raise TypeError("config must be a Phase3Config")
    if stage in _CANONICAL_STAGES:
        if parent_decision is not None:
            raise ValueError(f"canonical stage {stage} must not have a parent decision")
        canonical = config.canonical_recipes[stage]
        return _make_recipe(
            config=config,
            stage_id=stage,
            recipe_id=stage,
            parent_recipe_id=canonical.parent_recipe_id,
            auxiliary_weight=canonical.auxiliary_weight,
            consistency_weight=canonical.consistency_weight,
            contrastive_weight=canonical.contrastive_weight,
            window_mode=canonical.window_mode,
        )
    if stage not in _TRANSFORM_STAGES:
        raise ValueError(
            "stage must be one of " + ", ".join((*_CANONICAL_STAGES, *_TRANSFORM_STAGES))
        )
    if parent_decision is None:
        raise ValueError(f"stage {stage} requires a parent decision")

    decision = _read_json_object(Path(parent_decision), "parent decision")
    parent_stage = _require_string(decision, "stage_id", "parent decision")
    promoted = decision.get("promoted")
    if type(promoted) is not bool:
        raise ValueError("parent decision promoted must be a bool")
    parent_payload = decision.get("promoted_recipe")
    if not isinstance(parent_payload, Mapping):
        raise ValueError("parent decision promoted_recipe must be an object")
    parent = _recipe_from_mapping(parent_payload)

    if stage == "A2":
        if parent_stage != "A1" or not promoted or parent.auxiliary_weight <= 0:
            raise ValueError("A2 requires an A1 decision with promoted=true")
        return _transform_recipe(
            config,
            parent,
            stage_id="A2",
            recipe_id=f"{parent.recipe_id}+A2",
            consistency_weight=float(config.transforms["A2"]["consistency_weight"]),
        )
    if stage == "A3":
        if parent_stage not in {"A1", "A2"}:
            raise ValueError("A3 parent decision must come from A1 or A2")
        return _transform_recipe(
            config,
            parent,
            stage_id="A3",
            recipe_id=f"{parent.recipe_id}+A3",
            contrastive_weight=float(config.transforms["A3"]["contrastive_weight"]),
        )
    if parent_stage != "A3":
        raise ValueError("A4 parent decision must come from A3")
    return _transform_recipe(
        config,
        parent,
        stage_id="A4",
        recipe_id=f"{parent.recipe_id}+A4",
        window_mode=str(config.transforms["A4"]["window_mode"]),
        per_device_train_batch_size=config.training.a4_per_device_train_batch_size,
        gradient_accumulation_steps=config.training.a4_gradient_accumulation_steps,
    )


def write_resolved_recipe(path: str | Path, recipe: ResolvedPhase3Recipe) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(recipe.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def apply_a4_low_memory_profile(
    config: Phase3Config,
    recipe: ResolvedPhase3Recipe,
) -> ResolvedPhase3Recipe:
    """Return the explicit, hash-bound A4 batch-2 memory profile.

    A batch of two cannot simultaneously contain the documented cross-group
    positive and same-group hard negative.  Therefore this profile is valid only
    when A3 was not promoted and the selected A4 parent has no contrastive term.
    Contrastive A4 runs retain batch four and can use gradient checkpointing.
    """

    if recipe.stage_id != "A4":
        raise ValueError("the low-memory profile is defined only for A4")
    if recipe.contrastive_weight > 0:
        raise ValueError(
            "A4 batch-2 low-memory mode is incompatible with inherited A3 "
            "contrastive loss; keep batch 4 and enable gradient checkpointing"
        )
    provisional = replace(
        recipe,
        per_device_train_batch_size=config.training.a4_low_memory_batch_size,
        gradient_accumulation_steps=(
            config.training.a4_low_memory_gradient_accumulation_steps
        ),
        recipe_hash="",
    )
    return replace(provisional, recipe_hash=_recipe_hash(provisional))


def apply_gradient_checkpointing_profile(
    recipe: ResolvedPhase3Recipe,
) -> ResolvedPhase3Recipe:
    """Enable gradient checkpointing as part of the immutable recipe hash."""

    if recipe.gradient_checkpointing:
        return recipe
    provisional = replace(recipe, gradient_checkpointing=True, recipe_hash="")
    return replace(provisional, recipe_hash=_recipe_hash(provisional))


def apply_no_bf16_profile(recipe: ResolvedPhase3Recipe) -> ResolvedPhase3Recipe:
    """Disable BF16 as an explicit, hash-bound compatibility profile."""

    if not recipe.bf16:
        return recipe
    provisional = replace(recipe, bf16=False, recipe_hash="")
    return replace(provisional, recipe_hash=_recipe_hash(provisional))


def load_resolved_recipe(
    path: str | Path, *, verify_recipe_hash: bool = True
) -> ResolvedPhase3Recipe:
    return _recipe_from_mapping(
        _read_json_object(Path(path), "resolved recipe"),
        verify_recipe_hash=verify_recipe_hash,
    )


def _make_recipe(
    *,
    config: Phase3Config,
    stage_id: str,
    recipe_id: str,
    parent_recipe_id: str | None,
    auxiliary_weight: float,
    consistency_weight: float,
    contrastive_weight: float,
    window_mode: str,
) -> ResolvedPhase3Recipe:
    training = config.training
    provisional = ResolvedPhase3Recipe(
        stage_id=stage_id,
        recipe_id=recipe_id,
        parent_recipe_id=parent_recipe_id,
        backbone_model_id=config.backbone_model_id,
        max_length=config.max_length,
        auxiliary_weight=auxiliary_weight,
        consistency_weight=consistency_weight,
        contrastive_weight=contrastive_weight,
        contrastive_temperature=training.contrastive_temperature,
        projection_dimension=training.projection_dimension,
        window_mode=window_mode,
        per_device_train_batch_size=training.per_device_train_batch_size,
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        learning_rate=training.learning_rate,
        epochs=training.epochs,
        weight_decay=training.weight_decay,
        warmup_ratio=training.warmup_ratio,
        label_smoothing_factor=training.label_smoothing_factor,
        early_stopping_patience=training.early_stopping_patience,
        save_total_limit=training.save_total_limit,
        bf16=training.bf16,
        gradient_checkpointing=False,
        recipe_hash="",
    )
    return replace(provisional, recipe_hash=_recipe_hash(provisional))


def _transform_recipe(
    config: Phase3Config,
    parent: ResolvedPhase3Recipe,
    *,
    stage_id: str,
    recipe_id: str,
    consistency_weight: float | None = None,
    contrastive_weight: float | None = None,
    window_mode: str | None = None,
    per_device_train_batch_size: int | None = None,
    gradient_accumulation_steps: int | None = None,
) -> ResolvedPhase3Recipe:
    if parent.backbone_model_id != config.backbone_model_id:
        raise ValueError("parent recipe backbone does not match Phase3 config")
    provisional = replace(
        parent,
        stage_id=stage_id,
        recipe_id=recipe_id,
        parent_recipe_id=parent.recipe_id,
        consistency_weight=(
            parent.consistency_weight
            if consistency_weight is None
            else consistency_weight
        ),
        contrastive_weight=(
            parent.contrastive_weight
            if contrastive_weight is None
            else contrastive_weight
        ),
        window_mode=parent.window_mode if window_mode is None else window_mode,
        per_device_train_batch_size=(
            parent.per_device_train_batch_size
            if per_device_train_batch_size is None
            else per_device_train_batch_size
        ),
        gradient_accumulation_steps=(
            parent.gradient_accumulation_steps
            if gradient_accumulation_steps is None
            else gradient_accumulation_steps
        ),
        recipe_hash="",
    )
    return replace(provisional, recipe_hash=_recipe_hash(provisional))


def _recipe_hash(recipe: ResolvedPhase3Recipe) -> str:
    encoded = json.dumps(
        recipe.to_dict(include_hash=False),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _recipe_from_mapping(
    values: Mapping[str, object], *, verify_recipe_hash: bool = True
) -> ResolvedPhase3Recipe:
    payload = dict(values)
    _require_exact_fields(payload, _RECIPE_FIELDS, "resolved recipe")
    recipe = ResolvedPhase3Recipe(
        stage_id=_require_string(payload, "stage_id", "resolved recipe"),
        recipe_id=_require_string(payload, "recipe_id", "resolved recipe"),
        parent_recipe_id=_optional_string(
            payload.get("parent_recipe_id"), "parent_recipe_id"
        ),
        backbone_model_id=_require_string(
            payload, "backbone_model_id", "resolved recipe"
        ),
        max_length=_require_positive_int(payload, "max_length", "resolved recipe"),
        auxiliary_weight=_require_ratio(payload, "auxiliary_weight", "resolved recipe"),
        consistency_weight=_require_ratio(
            payload, "consistency_weight", "resolved recipe"
        ),
        contrastive_weight=_require_ratio(
            payload, "contrastive_weight", "resolved recipe"
        ),
        contrastive_temperature=_require_positive_float(
            payload, "contrastive_temperature", "resolved recipe"
        ),
        projection_dimension=_require_positive_int(
            payload, "projection_dimension", "resolved recipe"
        ),
        window_mode=_require_window_mode(payload.get("window_mode")),
        per_device_train_batch_size=_require_positive_int(
            payload, "per_device_train_batch_size", "resolved recipe"
        ),
        gradient_accumulation_steps=_require_positive_int(
            payload, "gradient_accumulation_steps", "resolved recipe"
        ),
        learning_rate=_require_positive_float(
            payload, "learning_rate", "resolved recipe"
        ),
        epochs=_require_positive_int(payload, "epochs", "resolved recipe"),
        weight_decay=_require_nonnegative_float(
            payload, "weight_decay", "resolved recipe"
        ),
        warmup_ratio=_require_ratio(payload, "warmup_ratio", "resolved recipe"),
        label_smoothing_factor=_require_ratio(
            payload, "label_smoothing_factor", "resolved recipe"
        ),
        early_stopping_patience=_require_positive_int(
            payload, "early_stopping_patience", "resolved recipe"
        ),
        save_total_limit=_require_positive_int(
            payload, "save_total_limit", "resolved recipe"
        ),
        bf16=_require_bool(payload, "bf16", "resolved recipe"),
        gradient_checkpointing=_require_bool(
            payload, "gradient_checkpointing", "resolved recipe"
        ),
        recipe_hash=_require_hash(payload.get("recipe_hash")),
    )
    if recipe.backbone_model_id != _BACKBONE:
        raise ValueError(f"resolved recipe backbone must be exactly {_BACKBONE}")
    expected_hash = _recipe_hash(replace(recipe, recipe_hash=""))
    if verify_recipe_hash and recipe.recipe_hash != expected_hash:
        raise ValueError("resolved recipe hash does not match recipe content")
    return recipe


def _parse_training(value: object) -> Phase3TrainingConfig:
    if not isinstance(value, Mapping):
        raise ValueError("training must be an object")
    values = dict(value)
    _require_exact_fields(values, _TRAINING_FIELDS, "training")
    training = Phase3TrainingConfig(
        learning_rate=_require_positive_float(values, "learning_rate", "training"),
        epochs=_require_positive_int(values, "epochs", "training"),
        weight_decay=_require_nonnegative_float(values, "weight_decay", "training"),
        warmup_ratio=_require_ratio(values, "warmup_ratio", "training"),
        label_smoothing_factor=_require_ratio(
            values, "label_smoothing_factor", "training"
        ),
        early_stopping_patience=_require_positive_int(
            values, "early_stopping_patience", "training"
        ),
        save_total_limit=_require_positive_int(values, "save_total_limit", "training"),
        bf16=_require_bool(values, "bf16", "training"),
        per_device_train_batch_size=_require_positive_int(
            values, "per_device_train_batch_size", "training"
        ),
        gradient_accumulation_steps=_require_positive_int(
            values, "gradient_accumulation_steps", "training"
        ),
        a4_per_device_train_batch_size=_require_positive_int(
            values, "a4_per_device_train_batch_size", "training"
        ),
        a4_gradient_accumulation_steps=_require_positive_int(
            values, "a4_gradient_accumulation_steps", "training"
        ),
        a4_low_memory_batch_size=_require_positive_int(
            values, "a4_low_memory_batch_size", "training"
        ),
        a4_low_memory_gradient_accumulation_steps=_require_positive_int(
            values, "a4_low_memory_gradient_accumulation_steps", "training"
        ),
        projection_dimension=_require_positive_int(
            values, "projection_dimension", "training"
        ),
        contrastive_temperature=_require_positive_float(
            values, "contrastive_temperature", "training"
        ),
    )
    if (
        training.per_device_train_batch_size
        * training.gradient_accumulation_steps
        != 32
        or training.a4_per_device_train_batch_size
        * training.a4_gradient_accumulation_steps
        != 32
        or training.a4_low_memory_batch_size
        * training.a4_low_memory_gradient_accumulation_steps
        != 32
    ):
        raise ValueError("all Phase3 batch/accumulation pairs must equal effective batch 32")
    return training


def _parse_canonical_recipes(value: object) -> dict[str, CanonicalRecipeConfig]:
    if not isinstance(value, Mapping):
        raise ValueError("canonical_recipes must be an object")
    values = dict(value)
    if tuple(values) != _CANONICAL_STAGES:
        raise ValueError(f"canonical_recipes must be ordered exactly {_CANONICAL_STAGES}")
    parsed: dict[str, CanonicalRecipeConfig] = {}
    for stage in _CANONICAL_STAGES:
        raw = values[stage]
        if not isinstance(raw, Mapping):
            raise ValueError(f"canonical recipe {stage} must be an object")
        row = dict(raw)
        _require_exact_fields(row, _CANONICAL_RECIPE_FIELDS, f"canonical recipe {stage}")
        parsed[stage] = CanonicalRecipeConfig(
            parent_recipe_id=_optional_string(
                row.get("parent_recipe_id"), "parent_recipe_id"
            ),
            auxiliary_weight=_require_ratio(
                row, "auxiliary_weight", f"canonical recipe {stage}"
            ),
            consistency_weight=_require_ratio(
                row, "consistency_weight", f"canonical recipe {stage}"
            ),
            contrastive_weight=_require_ratio(
                row, "contrastive_weight", f"canonical recipe {stage}"
            ),
            window_mode=_require_window_mode(row.get("window_mode")),
        )
    if parsed["A0"] != CanonicalRecipeConfig(None, 0.0, 0.0, 0.0, "single"):
        raise ValueError("canonical A0 recipe must be the exact family-only baseline")
    if parsed["A1-alpha01"].auxiliary_weight != 0.1:
        raise ValueError("A1-alpha01 auxiliary_weight must be exactly 0.1")
    if parsed["A1-alpha03"].auxiliary_weight != 0.3:
        raise ValueError("A1-alpha03 auxiliary_weight must be exactly 0.3")
    return parsed


def _parse_transforms(value: object) -> dict[str, dict[str, object]]:
    if not isinstance(value, Mapping):
        raise ValueError("transforms must be an object")
    values = dict(value)
    if tuple(values) != _TRANSFORM_STAGES:
        raise ValueError(f"transforms must be ordered exactly {_TRANSFORM_STAGES}")
    expected = {
        "A2": ("consistency_weight", 0.1),
        "A3": ("contrastive_weight", 0.1),
        "A4": ("window_mode", "head-center-tail"),
    }
    parsed: dict[str, dict[str, object]] = {}
    for stage, (field, exact_value) in expected.items():
        raw = values[stage]
        if not isinstance(raw, Mapping):
            raise ValueError(f"transform {stage} must be an object")
        row = dict(raw)
        _require_exact_fields(row, frozenset({field}), f"transform {stage}")
        if row[field] != exact_value:
            raise ValueError(f"transform {stage} {field} must be exactly {exact_value}")
        parsed[stage] = row
    return parsed


def _parse_gates(value: object) -> Phase3GateConfig:
    if not isinstance(value, Mapping):
        raise ValueError("gates must be an object")
    values = dict(value)
    _require_exact_fields(values, _GATE_FIELDS, "gates")
    gates = Phase3GateConfig(
        closed_macro_f1_min_gain=_require_float(
            values, "closed_macro_f1_min_gain", "gates"
        ),
        closed_seed_wins_min=_require_positive_int(
            values, "closed_seed_wins_min", "gates"
        ),
        worst_family_min_gain=_require_float(values, "worst_family_min_gain", "gates"),
        mimo_min_gain=_require_float(values, "mimo_min_gain", "gates"),
        glm_min_gain=_require_float(values, "glm_min_gain", "gates"),
        holdout_combined_min_gain=_require_float(
            values, "holdout_combined_min_gain", "gates"
        ),
        holdout_direction_min_gain=_require_float(
            values, "holdout_direction_min_gain", "gates"
        ),
        forbid_zero_family_f1=_require_bool(
            values, "forbid_zero_family_f1", "gates"
        ),
    )
    if gates.closed_seed_wins_min > 3:
        raise ValueError("closed_seed_wins_min cannot exceed three screening seeds")
    return gates


def _read_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not load {label} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _require_exact_fields(
    values: Mapping[str, object],
    expected: frozenset[str],
    location: str,
) -> None:
    actual = set(values)
    unknown = sorted(actual - expected)
    missing = sorted(expected - actual)
    if unknown:
        raise ValueError(f"Unknown {location} field(s): {', '.join(unknown)}")
    if missing:
        raise ValueError(f"Missing {location} field(s): {', '.join(missing)}")


def _require_string(values: Mapping[str, object], field: str, location: str) -> str:
    value = values.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} {field} must be a non-empty string")
    return value


def _optional_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be null or a non-empty string")
    return value


def _require_int(values: Mapping[str, object], field: str, location: str) -> int:
    value = values.get(field)
    if type(value) is not int:
        raise ValueError(f"{location} {field} must be an int")
    return value


def _require_positive_int(
    values: Mapping[str, object], field: str, location: str
) -> int:
    value = _require_int(values, field, location)
    if value <= 0:
        raise ValueError(f"{location} {field} must be positive")
    return value


def _require_float(values: Mapping[str, object], field: str, location: str) -> float:
    value = values.get(field)
    if type(value) not in (int, float):
        raise ValueError(f"{location} {field} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{location} {field} must be a finite number")
    return normalized


def _require_positive_float(
    values: Mapping[str, object], field: str, location: str
) -> float:
    value = _require_float(values, field, location)
    if value <= 0:
        raise ValueError(f"{location} {field} must be positive")
    return value


def _require_nonnegative_float(
    values: Mapping[str, object], field: str, location: str
) -> float:
    value = _require_float(values, field, location)
    if value < 0:
        raise ValueError(f"{location} {field} must be non-negative")
    return value


def _require_ratio(values: Mapping[str, object], field: str, location: str) -> float:
    value = _require_float(values, field, location)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{location} {field} must be between 0 and 1")
    return value


def _require_bool(values: Mapping[str, object], field: str, location: str) -> bool:
    value = values.get(field)
    if type(value) is not bool:
        raise ValueError(f"{location} {field} must be a bool")
    return value


def _require_hash(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("resolved recipe recipe_hash must be a 64-character hex string")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(
            "resolved recipe recipe_hash must be a 64-character hex string"
        ) from error
    return value.lower()


def _require_window_mode(value: object) -> str:
    if value not in {"single", "head-center-tail"}:
        raise ValueError("window_mode must be 'single' or 'head-center-tail'")
    return str(value)


def _require_exact_seed_list(
    value: object,
    field: str,
    expected: tuple[int, ...],
) -> tuple[int, ...]:
    if not isinstance(value, list) or any(type(seed) is not int for seed in value):
        raise ValueError(f"{field} must be a list of integers")
    normalized = tuple(value)
    if normalized != expected:
        raise ValueError(f"{field} must be exactly {expected}")
    return normalized


def _require_string_tuple(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{field} must be a list of non-empty strings")
    return tuple(value)
