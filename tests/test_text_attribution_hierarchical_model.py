from __future__ import annotations

from dataclasses import replace
import importlib
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


def test_model_module_exports_phase3_contract() -> None:
    module = importlib.import_module("research.text_attribution.hierarchical_model")
    assert hasattr(module, "HierarchicalAttributionModel")
    assert hasattr(module, "Phase3ModelOutput")
    assert callable(module.save_phase3_model)
    assert callable(module.load_phase3_model)


def _runtime():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    from research.text_attribution import hierarchical_model

    return torch, transformers, hierarchical_model


def _tiny_backbone(transformers):
    config = transformers.XLMRobertaConfig(
        vocab_size=128,
        hidden_size=24,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        pad_token_id=1,
    )
    return transformers.XLMRobertaModel(config)


def _batch(torch, windows: int = 1):
    input_ids = torch.randint(2, 100, (2, windows, 8))
    attention_mask = torch.ones_like(input_ids)
    window_mask = torch.ones((2, windows), dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "window_mask": window_mask,
    }


def _recipe(stage: str):
    config = load_phase3_config(CONFIG_PATH)
    if stage in {"A0", "A1-alpha01", "A1-alpha03"}:
        return resolve_phase3_recipe(config, stage)
    base = resolve_phase3_recipe(config, "A1-alpha01")
    if stage == "A3":
        return replace(
            base,
            stage_id="A3",
            recipe_id="unit-A3",
            contrastive_weight=0.1,
            recipe_hash="f" * 64,
        )
    if stage == "A4":
        return replace(
            base,
            stage_id="A4",
            recipe_id="unit-A4",
            window_mode="head-center-tail",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            recipe_hash="e" * 64,
        )
    raise AssertionError(stage)


def test_a0_has_only_family_logits() -> None:
    torch, transformers, model_module = _runtime()
    model = model_module.HierarchicalAttributionModel(
        backbone=_tiny_backbone(transformers),
        recipe=_recipe("A0"),
        family_labels=("a", "b", "c"),
        model_labels=(),
    )
    output = model(**_batch(torch))
    assert output.family_logits.shape == (2, 3)
    assert output.model_logits is None
    assert output.projections is None


def test_a1_has_auxiliary_model_logits() -> None:
    torch, transformers, model_module = _runtime()
    model = model_module.HierarchicalAttributionModel(
        backbone=_tiny_backbone(transformers),
        recipe=_recipe("A1-alpha01"),
        family_labels=("a", "b"),
        model_labels=("a-fast", "b-fast"),
    )
    output = model(**_batch(torch))
    assert output.family_logits.shape == (2, 2)
    assert output.model_logits.shape == (2, 2)
    assert output.projections is None


def test_a3_projection_is_normalized() -> None:
    torch, transformers, model_module = _runtime()
    model = model_module.HierarchicalAttributionModel(
        backbone=_tiny_backbone(transformers),
        recipe=_recipe("A3"),
        family_labels=("a", "b"),
        model_labels=("a-fast", "b-fast"),
    ).eval()
    output = model(**_batch(torch))
    torch.testing.assert_close(
        output.projections.norm(dim=-1), torch.ones(2), rtol=1e-5, atol=1e-5
    )


def test_a4_masked_pool_ignores_inactive_windows() -> None:
    torch, transformers, model_module = _runtime()
    model = model_module.HierarchicalAttributionModel(
        backbone=_tiny_backbone(transformers),
        recipe=_recipe("A4"),
        family_labels=("a", "b"),
        model_labels=("a-fast", "b-fast"),
    ).eval()
    batch = _batch(torch, windows=3)
    batch["window_mask"][:, 1:] = 0
    before = model(**batch).family_logits.detach()
    batch["input_ids"][:, 1:] = torch.randint(2, 100, (2, 2, 8))
    after = model(**batch).family_logits.detach()
    torch.testing.assert_close(before, after)


def test_saved_model_reloads_with_identical_logits(tmp_path) -> None:
    torch, transformers, model_module = _runtime()
    recipe = _recipe("A1-alpha01")
    model = model_module.HierarchicalAttributionModel(
        backbone=_tiny_backbone(transformers),
        recipe=recipe,
        family_labels=("a", "b"),
        model_labels=("a-fast", "b-fast"),
    ).eval()
    batch = _batch(torch)
    before = model(**batch).family_logits.detach()
    destination = tmp_path / "model"
    model_module.save_phase3_model(model, destination)
    restored = model_module.load_phase3_model(
        destination,
        expected_recipe_hash=recipe.recipe_hash,
        expected_family_labels=("a", "b"),
        expected_model_labels=("a-fast", "b-fast"),
    ).eval()
    after = restored(**batch).family_logits.detach()
    torch.testing.assert_close(before, after)


def test_reload_rejects_recipe_or_label_mismatch(tmp_path) -> None:
    _, transformers, model_module = _runtime()
    recipe = _recipe("A1-alpha01")
    model = model_module.HierarchicalAttributionModel(
        backbone=_tiny_backbone(transformers),
        recipe=recipe,
        family_labels=("a", "b"),
        model_labels=("a-fast", "b-fast"),
    )
    destination = tmp_path / "model"
    model_module.save_phase3_model(model, destination)
    with pytest.raises(ValueError, match="recipe hash"):
        model_module.load_phase3_model(
            destination, expected_recipe_hash="0" * 64
        )
    with pytest.raises(ValueError, match="family labels"):
        model_module.load_phase3_model(
            destination, expected_family_labels=("wrong", "labels")
        )


def test_model_rejects_metadata_kwargs() -> None:
    torch, transformers, model_module = _runtime()
    model = model_module.HierarchicalAttributionModel(
        backbone=_tiny_backbone(transformers),
        recipe=_recipe("A0"),
        family_labels=("a", "b"),
        model_labels=(),
    )
    with pytest.raises(TypeError):
        model(**_batch(torch), request_key=["secret", "secret-2"])
