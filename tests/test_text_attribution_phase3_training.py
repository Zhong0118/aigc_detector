from __future__ import annotations

from dataclasses import FrozenInstanceError
import importlib
import json
from pathlib import Path

import pytest

from research.text_attribution.phase3_config import (
    load_phase3_config,
    resolve_phase3_recipe,
)


CONFIG_PATH = (
    Path(__file__).parents[1]
    / "configs/text_attribution/training/family_phase3_enhancement.v1.json"
)


def test_training_module_exports_unified_contract() -> None:
    module = importlib.import_module("research.text_attribution.phase3_training")
    assert hasattr(module, "Phase3Trainer")
    assert hasattr(module, "Phase3TrainingRequest")
    assert hasattr(module, "Phase3TrainingResult")
    assert callable(module.train_phase3_experiment)


def test_training_request_is_immutable(tmp_path) -> None:
    module = importlib.import_module("research.text_attribution.phase3_training")
    recipe = resolve_phase3_recipe(load_phase3_config(CONFIG_PATH), "A0")
    request = module.Phase3TrainingRequest(
        recipe=recipe,
        data_view=None,
        output_dir=tmp_path / "run",
        seed=42,
        smoke_test=True,
        require_cuda=False,
    )
    with pytest.raises(FrozenInstanceError):
        request.seed = 52


def test_checkpoint_recipe_hash_mismatch_is_rejected(tmp_path) -> None:
    module = importlib.import_module("research.text_attribution.phase3_training")
    checkpoint = tmp_path / "checkpoint-10"
    checkpoint.mkdir()
    (checkpoint / "phase3_checkpoint.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "recipe_hash": "a" * 64,
                "global_step": 10,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="recipe hash"):
        module.validate_phase3_checkpoint(checkpoint, "b" * 64)


def test_checkpoint_metadata_is_strict(tmp_path) -> None:
    module = importlib.import_module("research.text_attribution.phase3_training")
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    (checkpoint / "phase3_checkpoint.json").write_text(
        '{"schema_version": 1, "recipe_hash": "x", "global_step": 1, "extra": 2}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown|missing"):
        module.validate_phase3_checkpoint(checkpoint, "x")


def test_formal_runtime_policy_rejects_cpu_and_invalid_bf16() -> None:
    module = importlib.import_module("research.text_attribution.phase3_training")
    with pytest.raises(RuntimeError, match="CUDA"):
        module.validate_phase3_runtime_policy(
            require_cuda=True,
            smoke_test=False,
            cuda_available=False,
            bf16_requested=True,
            bf16_supported=False,
        )
    with pytest.raises(RuntimeError, match="BF16"):
        module.validate_phase3_runtime_policy(
            require_cuda=True,
            smoke_test=False,
            cuda_available=True,
            bf16_requested=True,
            bf16_supported=False,
        )
    assert (
        module.validate_phase3_runtime_policy(
            require_cuda=False,
            smoke_test=True,
            cuda_available=False,
            bf16_requested=True,
            bf16_supported=False,
        )
        is False
    )


def test_training_refuses_owned_output_directory_before_runtime(tmp_path) -> None:
    module = importlib.import_module("research.text_attribution.phase3_training")
    recipe = resolve_phase3_recipe(load_phase3_config(CONFIG_PATH), "A0")
    output = tmp_path / "owned"
    output.mkdir()
    request = module.Phase3TrainingRequest(
        recipe=recipe,
        data_view=None,
        output_dir=output,
        seed=42,
        smoke_test=True,
        require_cuda=False,
    )
    with pytest.raises(FileExistsError):
        module.train_phase3_experiment(request)


def test_phase3_trainer_tensor_contract_is_exercised_on_server() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    # The end-to-end server smoke in the runbook executes two optimizer steps and
    # asserts the resulting COMPLETED/loss/model contracts.  This local test keeps
    # dependency discovery explicit without downloading XLM-R during unit tests.
