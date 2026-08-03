"""Strict configuration contract for the Phase4 attribution Fast Track."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


_EXPECTED_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "a4_recipe_hash",
        "seeds",
        "views",
        "production_seed",
        "branches",
        "cross_fit_folds",
        "fold_seed",
        "bootstrap_resamples",
        "bootstrap_seed",
        "ece_bins",
        "weight_step",
        "max_single_branch_macro_f1_drop",
        "e1_closed_macro_f1_min_gain",
        "holdout_max_drop",
        "accepted_precision_min",
        "coverage_min",
        "ece_max",
    }
)
_VIEWS = ("closed", "fast-to-balanced", "balanced-to-fast")
_BRANCHES = ("A4", "B0", "B1")


@dataclass(frozen=True)
class Phase4FastTrackConfig:
    schema_version: int
    run_id: str
    a4_recipe_hash: str
    seeds: tuple[int, ...]
    views: tuple[str, ...]
    production_seed: int
    branches: tuple[str, ...]
    cross_fit_folds: int
    fold_seed: int
    bootstrap_resamples: int
    bootstrap_seed: int
    ece_bins: int
    weight_step: float
    max_single_branch_macro_f1_drop: float
    e1_closed_macro_f1_min_gain: float
    holdout_max_drop: float
    accepted_precision_min: float
    coverage_min: float
    ece_max: float


def load_phase4_fast_track_config(path: str | Path) -> Phase4FastTrackConfig:
    """Load an exact, versioned Fast Track configuration."""

    config_path = Path(path)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid Phase4 config JSON: {config_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Phase4 config must be a JSON object")
    keys = set(payload)
    unknown = sorted(keys - _EXPECTED_KEYS)
    missing = sorted(_EXPECTED_KEYS - keys)
    if unknown:
        raise ValueError(f"unknown Phase4 config keys: {unknown!r}")
    if missing:
        raise ValueError(f"missing Phase4 config keys: {missing!r}")

    config = Phase4FastTrackConfig(
        schema_version=_integer(payload["schema_version"], "schema_version"),
        run_id=_text(payload["run_id"], "run_id"),
        a4_recipe_hash=_text(payload["a4_recipe_hash"], "a4_recipe_hash"),
        seeds=_integer_tuple(payload["seeds"], "seeds"),
        views=_text_tuple(payload["views"], "views"),
        production_seed=_integer(payload["production_seed"], "production_seed"),
        branches=_text_tuple(payload["branches"], "branches"),
        cross_fit_folds=_integer(payload["cross_fit_folds"], "cross_fit_folds"),
        fold_seed=_integer(payload["fold_seed"], "fold_seed"),
        bootstrap_resamples=_integer(payload["bootstrap_resamples"], "bootstrap_resamples"),
        bootstrap_seed=_integer(payload["bootstrap_seed"], "bootstrap_seed"),
        ece_bins=_integer(payload["ece_bins"], "ece_bins"),
        weight_step=_number(payload["weight_step"], "weight_step"),
        max_single_branch_macro_f1_drop=_number(
            payload["max_single_branch_macro_f1_drop"],
            "max_single_branch_macro_f1_drop",
        ),
        e1_closed_macro_f1_min_gain=_number(
            payload["e1_closed_macro_f1_min_gain"],
            "e1_closed_macro_f1_min_gain",
        ),
        holdout_max_drop=_number(payload["holdout_max_drop"], "holdout_max_drop"),
        accepted_precision_min=_number(
            payload["accepted_precision_min"], "accepted_precision_min"
        ),
        coverage_min=_number(payload["coverage_min"], "coverage_min"),
        ece_max=_number(payload["ece_max"], "ece_max"),
    )
    _validate(config)
    return config


def canonical_config_hash(config: Phase4FastTrackConfig) -> str:
    payload = asdict(config)
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate(config: Phase4FastTrackConfig) -> None:
    if config.schema_version != 1:
        raise ValueError("schema_version must be 1")
    if len(config.a4_recipe_hash) != 64 or any(
        character not in "0123456789abcdef" for character in config.a4_recipe_hash
    ):
        raise ValueError("a4_recipe_hash must be a lowercase SHA-256")
    if not config.seeds or len(config.seeds) != len(set(config.seeds)):
        raise ValueError("seeds must be unique and nonempty")
    if config.production_seed not in config.seeds:
        raise ValueError("production_seed must be present in seeds")
    if config.views != _VIEWS:
        raise ValueError(f"views must be exactly {_VIEWS!r}")
    if config.branches != _BRANCHES:
        raise ValueError(f"branches must be exactly {_BRANCHES!r}")
    if config.cross_fit_folds != 5:
        raise ValueError("cross_fit_folds must be 5")
    if config.bootstrap_resamples <= 0 or config.ece_bins <= 0:
        raise ValueError("bootstrap_resamples and ece_bins must be positive")
    if not 0.0 < config.weight_step <= 1.0 or abs(
        round(1.0 / config.weight_step) - (1.0 / config.weight_step)
    ) > 1e-9:
        raise ValueError("weight_step must divide one exactly")
    for field in (
        "max_single_branch_macro_f1_drop",
        "e1_closed_macro_f1_min_gain",
        "holdout_max_drop",
        "accepted_precision_min",
        "coverage_min",
        "ece_max",
    ):
        value = getattr(config, field)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{field} must be in [0, 1]")


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    return float(value)


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonempty text")
    return value.strip()


def _integer_tuple(value: Any, field: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return tuple(_integer(item, field) for item in value)


def _text_tuple(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    result = tuple(_text(item, field) for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{field} must be unique")
    return result
