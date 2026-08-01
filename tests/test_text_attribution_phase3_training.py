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
                "run_fingerprint": "r" * 64,
                "dataset_fingerprint": "d" * 64,
                "split_manifest_hash": "s" * 64,
                "catalog_hash": "c" * 64,
                "seed": 42,
                "view_id": "closed",
                "global_step": 10,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="recipe hash"):
        module.validate_phase3_checkpoint(
            checkpoint,
            expected_recipe_hash="b" * 64,
            expected_run_fingerprint="r" * 64,
            expected_dataset_fingerprint="d" * 64,
            expected_split_manifest_hash="s" * 64,
            expected_catalog_hash="c" * 64,
            expected_seed=42,
            expected_view_id="closed",
        )


def test_checkpoint_run_identity_mismatch_is_rejected(tmp_path) -> None:
    module = importlib.import_module("research.text_attribution.phase3_training")
    checkpoint = tmp_path / "checkpoint-10"
    checkpoint.mkdir()
    (checkpoint / "phase3_checkpoint.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "recipe_hash": "a" * 64,
                "run_fingerprint": "r" * 64,
                "dataset_fingerprint": "d" * 64,
                "split_manifest_hash": "s" * 64,
                "catalog_hash": "c" * 64,
                "seed": 42,
                "view_id": "closed",
                "global_step": 10,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="run fingerprint"):
        module.validate_phase3_checkpoint(
            checkpoint,
            expected_recipe_hash="a" * 64,
            expected_run_fingerprint="x" * 64,
            expected_dataset_fingerprint="d" * 64,
            expected_split_manifest_hash="s" * 64,
            expected_catalog_hash="c" * 64,
            expected_seed=42,
            expected_view_id="closed",
        )


def test_checkpoint_metadata_is_strict(tmp_path) -> None:
    module = importlib.import_module("research.text_attribution.phase3_training")
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    (checkpoint / "phase3_checkpoint.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "recipe_hash": "x",
                "run_fingerprint": "r" * 64,
                "dataset_fingerprint": "d" * 64,
                "split_manifest_hash": "s" * 64,
                "catalog_hash": "c" * 64,
                "seed": 42,
                "view_id": "closed",
                "global_step": 1,
                "extra": 2,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown|missing"):
        module.validate_phase3_checkpoint(
            checkpoint,
            expected_recipe_hash="x",
            expected_run_fingerprint="r" * 64,
            expected_dataset_fingerprint="d" * 64,
            expected_split_manifest_hash="s" * 64,
            expected_catalog_hash="c" * 64,
            expected_seed=42,
            expected_view_id="closed",
        )


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


def test_offline_trainer_lifecycle_saves_best_checkpoint_and_resumes(
    tmp_path, monkeypatch
) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    pytest.importorskip("safetensors")
    from research.text_attribution.phase3_data import (
        CatalogVariant,
        VariantCatalog,
        build_phase3_data_view,
    )
    from research.text_attribution.training_data import AttributionSample
    from research.text_attribution import phase3_training as module

    class TinyTokenizer:
        pad_token_id = 1

        def num_special_tokens_to_add(self, pair=False):
            return 2

        def __call__(self, text, **kwargs):
            return {"input_ids": [3 + ord(char) % 30 for char in text]}

        def build_inputs_with_special_tokens(self, token_ids):
            return [0, *token_ids, 2]

        def save_pretrained(self, destination):
            path = Path(destination)
            path.mkdir(parents=True, exist_ok=True)
            (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    config = transformers.XLMRobertaConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=24,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        pad_token_id=1,
    )
    monkeypatch.setattr(
        module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: TinyTokenizer(),
    )
    monkeypatch.setattr(
        module.AutoModel,
        "from_pretrained",
        lambda *args, **kwargs: transformers.XLMRobertaModel(config),
    )
    monkeypatch.setattr(
        module,
        "validate_phase3_runtime_policy",
        lambda **kwargs: False,
    )

    def sample(index, family, model, split):
        return AttributionSample(
            request_key=f"{split}-{index}",
            text=f"tiny sample {index}",
            family_id=family,
            model_id=model,
            prompt_id=f"en-p{index}",
            semantic_prompt_group=f"p{index}",
            language="en",
            task_type="qa",
            target_length="short",
            temperature=0.7,
            repeat_index=0,
        )

    catalog = VariantCatalog(
        models=(
            CatalogVariant("a", "a-fast", "fast"),
            CatalogVariant("a", "a-balanced", "balanced"),
            CatalogVariant("b", "b-fast", "fast"),
            CatalogVariant("b", "b-balanced", "balanced"),
        )
    )
    train = tuple(
        sample(i, "a" if i % 2 == 0 else "b", "a-fast" if i % 2 == 0 else "b-fast", "train")
        for i in range(8)
    ) + tuple(
        sample(i + 8, "a" if i % 2 == 0 else "b", "a-balanced" if i % 2 == 0 else "b-balanced", "train")
        for i in range(8)
    )
    validation = tuple(
        sample(i + 20, "a" if i % 2 == 0 else "b", "a-fast" if i % 2 == 0 else "b-fast", "validation")
        for i in range(4)
    )
    view = build_phase3_data_view(
        {"train": train, "validation": validation}, catalog, "closed"
    )
    recipe = resolve_phase3_recipe(load_phase3_config(CONFIG_PATH), "A0")
    identity = dict(
        run_fingerprint="r" * 64,
        dataset_fingerprint="d" * 64,
        split_manifest_hash="s" * 64,
        catalog_hash="c" * 64,
    )
    first = module.train_phase3_experiment(
        module.Phase3TrainingRequest(
            recipe=recipe,
            data_view=view,
            output_dir=tmp_path / "first",
            seed=42,
            require_cuda=False,
            use_cpu=True,
            **identity,
        )
    )
    assert first.best_checkpoint is not None
    assert (tmp_path / "first/COMPLETED.json").is_file()
    checkpoint = sorted((tmp_path / "first/checkpoints").glob("checkpoint-*"))[-1]
    assert (checkpoint / "phase3_checkpoint.json").is_file()

    resumed = module.train_phase3_experiment(
        module.Phase3TrainingRequest(
            recipe=recipe,
            data_view=view,
            output_dir=tmp_path / "resumed",
            seed=42,
            require_cuda=False,
            use_cpu=True,
            resume_from_checkpoint=checkpoint,
            **identity,
        )
    )
    assert resumed.validation_logits.shape == (len(validation), 2)
    assert torch.isfinite(torch.from_numpy(resumed.validation_logits)).all()
