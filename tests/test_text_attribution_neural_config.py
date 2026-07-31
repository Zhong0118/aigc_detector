from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pytest

from research.text_attribution.neural_config import load_neural_screening_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs"
    / "text_attribution"
    / "training"
    / "family_neural_screening.v1.json"
)


def _valid_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": "family-attribution-neural-screening-v1",
        "seed": 42,
        "model_catalog": "../zhizengzeng_model_catalog.v5.json",
        "common": {
            "learning_rate": 2e-5,
            "epochs": 6,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "label_smoothing_factor": 0.05,
            "gradient_accumulation_steps": 2,
            "early_stopping_patience": 2,
            "save_total_limit": 2,
            "bf16": True,
        },
        "experiments": {
            "N0": {
                "model_id": "FacebookAI/xlm-roberta-base",
                "max_length": 512,
                "per_device_train_batch_size": 16,
                "gradient_accumulation_steps": 2,
            },
            "N1": {
                "model_id": "microsoft/mdeberta-v3-base",
                "max_length": 512,
                "per_device_train_batch_size": 16,
                "gradient_accumulation_steps": 2,
            },
            "N2": {
                "model_id": "google/canine-c",
                "max_length": 1024,
                "per_device_train_batch_size": 8,
                "gradient_accumulation_steps": 4,
            },
        },
    }


def _write_config(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "training" / "config.json"
    path.parent.mkdir()
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_repository_config_defines_only_the_three_approved_experiments() -> None:
    config = load_neural_screening_config(CONFIG_PATH)

    assert list(config.experiments) == ["N0", "N1", "N2"]
    assert config.experiments["N0"].max_length == 512
    assert config.experiments["N1"].max_length == 512
    assert config.experiments["N2"].max_length == 1024


def test_model_catalog_is_resolved_relative_to_the_config_file(
    tmp_path: Path,
) -> None:
    path = _write_config(tmp_path, _valid_payload())

    config = load_neural_screening_config(path)

    assert config.model_catalog == (
        path.parent / "../zhizengzeng_model_catalog.v5.json"
    ).resolve()
    assert config.model_catalog.is_absolute()


@pytest.mark.parametrize("section", ["top", "common", "experiment"])
def test_unknown_fields_are_rejected(tmp_path: Path, section: str) -> None:
    payload = _valid_payload()
    if section == "top":
        payload["surprise"] = "no"
    elif section == "common":
        common = payload["common"]
        assert isinstance(common, dict)
        common["surprise"] = "no"
    else:
        experiments = payload["experiments"]
        assert isinstance(experiments, dict)
        experiment = experiments["N0"]
        assert isinstance(experiment, dict)
        experiment["surprise"] = "no"

    with pytest.raises(ValueError, match="Unknown .*field"):
        load_neural_screening_config(_write_config(tmp_path, payload))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.update(schema_version=2), "schema_version"),
        (
            lambda payload: payload["experiments"].update(N3={}),  # type: ignore[union-attr]
            "experiments",
        ),
        (
            lambda payload: payload["experiments"].pop("N1"),  # type: ignore[union-attr]
            "experiments",
        ),
        (lambda payload: payload.update(seed=True), "seed"),
        (
            lambda payload: payload["common"].update(epochs=True),  # type: ignore[union-attr]
            "epochs",
        ),
        (
            lambda payload: payload["common"].update(learning_rate=True),  # type: ignore[union-attr]
            "learning_rate",
        ),
        (
            lambda payload: payload["experiments"]["N0"].update(max_length=0),  # type: ignore[index,union-attr]
            "max_length",
        ),
        (
            lambda payload: payload["experiments"]["N0"].update(  # type: ignore[index,union-attr]
                per_device_train_batch_size=-1
            ),
            "per_device_train_batch_size",
        ),
        (
            lambda payload: payload["common"].update(epochs=0),  # type: ignore[union-attr]
            "epochs",
        ),
        (
            lambda payload: payload["common"].update(  # type: ignore[union-attr]
                gradient_accumulation_steps=0
            ),
            "gradient_accumulation_steps",
        ),
        (
            lambda payload: payload["common"].update(warmup_ratio=-0.01),  # type: ignore[union-attr]
            "warmup_ratio",
        ),
        (
            lambda payload: payload["common"].update(warmup_ratio=1.01),  # type: ignore[union-attr]
            "warmup_ratio",
        ),
        (
            lambda payload: payload["common"].update(  # type: ignore[union-attr]
                label_smoothing_factor=-0.01
            ),
            "label_smoothing_factor",
        ),
        (
            lambda payload: payload["common"].update(  # type: ignore[union-attr]
                label_smoothing_factor=1.01
            ),
            "label_smoothing_factor",
        ),
    ],
)
def test_invalid_schema_types_names_and_ranges_are_rejected(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    payload = _valid_payload()
    mutate(payload)

    with pytest.raises(ValueError, match=message):
        load_neural_screening_config(_write_config(tmp_path, payload))


def test_common_values_are_merged_with_per_experiment_overrides(
    tmp_path: Path,
) -> None:
    config = load_neural_screening_config(
        _write_config(tmp_path, _valid_payload())
    )

    n0 = config.experiments["N0"]
    n2 = config.experiments["N2"]
    assert n0.learning_rate == 2e-5
    assert n0.epochs == 6
    assert n0.weight_decay == 0.01
    assert n0.warmup_ratio == 0.1
    assert n0.label_smoothing_factor == 0.05
    assert n0.gradient_accumulation_steps == 2
    assert n0.early_stopping_patience == 2
    assert n0.save_total_limit == 2
    assert n0.bf16 is True
    assert n2.gradient_accumulation_steps == 4


def test_loaded_configuration_is_immutable(tmp_path: Path) -> None:
    config = load_neural_screening_config(
        _write_config(tmp_path, _valid_payload())
    )

    with pytest.raises(FrozenInstanceError):
        config.seed = 7  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        config.experiments["N0"].epochs = 7  # type: ignore[misc]
    with pytest.raises(TypeError):
        config.experiments["N3"] = config.experiments["N0"]  # type: ignore[index]
