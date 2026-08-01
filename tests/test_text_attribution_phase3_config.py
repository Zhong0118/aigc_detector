from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from research.text_attribution.phase3_config import (
    load_phase3_config,
    load_resolved_recipe,
    resolve_phase3_recipe,
    write_resolved_recipe,
)


CONFIG_PATH = Path(
    "configs/text_attribution/training/family_phase3_enhancement.v1.json"
)


def _write_parent_decision(
    path: Path,
    *,
    stage_id: str,
    promoted: bool,
    recipe: object,
) -> Path:
    payload = {
        "schema_version": 1,
        "stage_id": stage_id,
        "promoted": promoted,
        "promoted_recipe": recipe.to_dict(),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_loads_exact_phase3_protocol() -> None:
    config = load_phase3_config(CONFIG_PATH)

    assert config.schema_version == 1
    assert config.backbone_model_id == "FacebookAI/xlm-roberta-base"
    assert config.screening_seeds == (42, 52, 62)
    assert config.confirmation_seeds == (12, 22, 32, 72, 82)
    assert config.data_views == (
        "closed",
        "fast-to-balanced",
        "balanced-to-fast",
    )
    assert config.gates.closed_macro_f1_min_gain == 0.01
    assert config.gates.closed_seed_wins_min == 2
    assert config.gates.forbid_zero_family_f1 is True


def test_resolves_canonical_recipes_with_stable_hashes() -> None:
    config = load_phase3_config(CONFIG_PATH)

    a0 = resolve_phase3_recipe(config, "A0")
    alpha01 = resolve_phase3_recipe(config, "A1-alpha01")
    alpha03 = resolve_phase3_recipe(config, "A1-alpha03")

    assert a0.parent_recipe_id is None
    assert a0.auxiliary_weight == 0.0
    assert a0.consistency_weight == 0.0
    assert a0.contrastive_weight == 0.0
    assert a0.window_mode == "single"
    assert alpha01.parent_recipe_id == "A0"
    assert alpha01.auxiliary_weight == 0.1
    assert alpha03.auxiliary_weight == 0.3
    assert alpha01.recipe_hash == resolve_phase3_recipe(
        config, "A1-alpha01"
    ).recipe_hash
    assert len(alpha01.recipe_hash) == 64
    int(alpha01.recipe_hash, 16)


def test_a2_requires_a_promoted_a1_parent(tmp_path: Path) -> None:
    config = load_phase3_config(CONFIG_PATH)
    a0 = resolve_phase3_recipe(config, "A0")
    failed_a1 = _write_parent_decision(
        tmp_path / "failed-a1.json",
        stage_id="A1",
        promoted=False,
        recipe=a0,
    )

    with pytest.raises(ValueError, match=r"A1.*promoted"):
        resolve_phase3_recipe(config, "A2", failed_a1)


def test_promoted_transforms_add_exactly_one_factor(tmp_path: Path) -> None:
    config = load_phase3_config(CONFIG_PATH)
    a1 = resolve_phase3_recipe(config, "A1-alpha01")
    a1_decision = _write_parent_decision(
        tmp_path / "a1.json",
        stage_id="A1",
        promoted=True,
        recipe=a1,
    )
    a2 = resolve_phase3_recipe(config, "A2", a1_decision)
    assert a2.parent_recipe_id == a1.recipe_id
    assert a2.auxiliary_weight == a1.auxiliary_weight
    assert a2.consistency_weight == 0.1
    assert a2.contrastive_weight == 0.0

    a2_decision = _write_parent_decision(
        tmp_path / "a2.json",
        stage_id="A2",
        promoted=True,
        recipe=a2,
    )
    a3 = resolve_phase3_recipe(config, "A3", a2_decision)
    assert a3.parent_recipe_id == a2.recipe_id
    assert a3.consistency_weight == a2.consistency_weight
    assert a3.contrastive_weight == 0.1
    assert a3.window_mode == "single"

    a3_decision = _write_parent_decision(
        tmp_path / "a3.json",
        stage_id="A3",
        promoted=True,
        recipe=a3,
    )
    a4 = resolve_phase3_recipe(config, "A4", a3_decision)
    assert a4.parent_recipe_id == a3.recipe_id
    assert a4.contrastive_weight == a3.contrastive_weight
    assert a4.window_mode == "head-center-tail"
    assert a4.per_device_train_batch_size == 4
    assert a4.gradient_accumulation_steps == 8


def test_a3_uses_fallback_recipe_after_failed_a1(tmp_path: Path) -> None:
    config = load_phase3_config(CONFIG_PATH)
    a0 = resolve_phase3_recipe(config, "A0")
    decision = _write_parent_decision(
        tmp_path / "a1-fallback.json",
        stage_id="A1",
        promoted=False,
        recipe=a0,
    )

    a3 = resolve_phase3_recipe(config, "A3", decision)

    assert a3.parent_recipe_id == "A0"
    assert a3.auxiliary_weight == 0.0
    assert a3.consistency_weight == 0.0
    assert a3.contrastive_weight == 0.1


def test_recipe_round_trip_rejects_tampered_hash(tmp_path: Path) -> None:
    recipe = resolve_phase3_recipe(load_phase3_config(CONFIG_PATH), "A1-alpha01")
    path = tmp_path / "recipe.json"
    write_resolved_recipe(path, recipe)

    assert load_resolved_recipe(path) == recipe

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["auxiliary_weight"] = 0.3
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        load_resolved_recipe(path)


def test_rejects_unknown_config_fields_and_invalid_fixed_values(
    tmp_path: Path,
) -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["unknown"] = True
    unknown_path = tmp_path / "unknown.json"
    unknown_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown"):
        load_phase3_config(unknown_path)

    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["backbone_model_id"] = "microsoft/mdeberta-v3-base"
    wrong_backbone = tmp_path / "wrong-backbone.json"
    wrong_backbone.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="xlm-roberta-base"):
        load_phase3_config(wrong_backbone)

    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["screening_seeds"] = [42, True, 62]
    invalid_seed = tmp_path / "invalid-seed.json"
    invalid_seed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="screening_seeds"):
        load_phase3_config(invalid_seed)


def test_recipe_dataclass_is_immutable() -> None:
    recipe = resolve_phase3_recipe(load_phase3_config(CONFIG_PATH), "A0")

    with pytest.raises(Exception):
        recipe.stage_id = "A4"  # type: ignore[misc]

    changed = replace(recipe, auxiliary_weight=0.1)
    assert changed.auxiliary_weight == 0.1
    assert recipe.auxiliary_weight == 0.0

