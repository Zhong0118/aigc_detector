from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from scripts.run_family_attribution_phase3 import (
    build_parser,
    compute_phase3_run_fingerprint,
    parse_args,
    resolve_cli_recipe,
)


CONFIG_PATH = (
    Path(__file__).parents[1]
    / "configs/text_attribution/training/family_phase3_enhancement.v1.json"
)
REQUIRED = [
    "--input",
    "data.jsonl",
    "--split-manifest",
    "split.json",
    "--output-dir",
    "run",
    "--view",
    "closed",
    "--seed",
    "42",
]


def test_parser_does_not_expose_test_evaluation() -> None:
    help_text = build_parser().format_help()
    assert "--evaluate-test" not in help_text
    assert "--test" not in help_text


def test_promoted_stage_requires_parent_decision() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--stage", "A3", *REQUIRED])


def test_canonical_promoted_and_replay_modes_are_exclusive(tmp_path) -> None:
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--stage",
                "A0",
                "--recipe",
                str(recipe_path),
                *REQUIRED,
            ]
        )
    with pytest.raises(SystemExit):
        parse_args(["--recipe", str(recipe_path), "--parent-decision", "x", *REQUIRED])


def test_canonical_stage_rejects_parent_decision() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--stage", "A0", "--parent-decision", "x", *REQUIRED])


def test_resolve_canonical_recipe_without_training_dependencies() -> None:
    args = parse_args(["--config", str(CONFIG_PATH), "--stage", "A1-alpha01", *REQUIRED])
    recipe = resolve_cli_recipe(args)
    assert recipe.recipe_id == "A1-alpha01"
    assert recipe.auxiliary_weight == 0.1


def test_low_memory_flag_is_restricted_to_a4() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--stage", "A0", "--a4-low-memory", *REQUIRED])


def test_run_fingerprint_is_canonical_and_sensitive() -> None:
    first = compute_phase3_run_fingerprint(
        recipe_hash="a" * 64,
        dataset_fingerprint="b" * 64,
        split_manifest_hash="c" * 64,
        catalog_hash="d" * 64,
        view_id="closed",
        seed=42,
    )
    same = compute_phase3_run_fingerprint(
        seed=42,
        view_id="closed",
        catalog_hash="d" * 64,
        split_manifest_hash="c" * 64,
        dataset_fingerprint="b" * 64,
        recipe_hash="a" * 64,
    )
    changed = compute_phase3_run_fingerprint(
        recipe_hash="a" * 64,
        dataset_fingerprint="b" * 64,
        split_manifest_hash="c" * 64,
        catalog_hash="d" * 64,
        view_id="fast-to-balanced",
        seed=42,
    )
    assert first == same
    assert first != changed
    assert len(first) == 64


def test_replay_recipe_hash_is_verified(tmp_path) -> None:
    from research.text_attribution.phase3_config import (
        load_phase3_config,
        resolve_phase3_recipe,
        write_resolved_recipe,
    )

    recipe = resolve_phase3_recipe(load_phase3_config(CONFIG_PATH), "A0")
    path = tmp_path / "recipe.json"
    write_resolved_recipe(path, recipe)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["learning_rate"] = 0.1
    path.write_text(json.dumps(payload), encoding="utf-8")
    args = parse_args(["--recipe", str(path), *REQUIRED])
    with pytest.raises(ValueError, match="hash"):
        resolve_cli_recipe(args)


def test_output_directory_must_not_exist(tmp_path) -> None:
    existing = tmp_path / "run"
    existing.mkdir()
    args = parse_args(
        [
            "--config",
            str(CONFIG_PATH),
            "--stage",
            "A0",
            "--input",
            "data.jsonl",
            "--split-manifest",
            "split.json",
            "--output-dir",
            str(existing),
            "--view",
            "closed",
            "--seed",
            "42",
        ]
    )
    from scripts.run_family_attribution_phase3 import validate_output_ownership

    with pytest.raises(FileExistsError):
        validate_output_ownership(Path(args.output_dir))


def test_contrastive_smoke_preserves_a_valid_structured_batch() -> None:
    from research.text_attribution.phase3_data import (
        Phase3DataView,
        StructuredContrastiveBatchSampler,
    )
    from research.text_attribution.training_data import AttributionSample
    from research.text_attribution.phase3_config import (
        load_phase3_config,
        resolve_phase3_recipe,
    )
    from scripts.run_family_attribution_phase3 import _smoke_data_view

    rows = []
    for family_index in range(10):
        family = f"family-{family_index}"
        model = f"model-{family_index}"
        for row_index, group in enumerate(
            (f"unique-{family}-0", f"unique-{family}-1", "shared-0", "shared-1")
        ):
            rows.append(
                AttributionSample(
                    request_key=f"{family}-{row_index}",
                    text="smoke",
                    family_id=family,
                    model_id=model,
                    prompt_id=group,
                    semantic_prompt_group=group,
                    language="en",
                    task_type="qa",
                    target_length="short",
                    temperature=0.7,
                    repeat_index=0,
                )
            )
    families = tuple(f"family-{index}" for index in range(10))
    models = tuple(f"model-{index}" for index in range(10))
    view = Phase3DataView(
        view_id="closed",
        train_samples=tuple(rows),
        validation_samples=tuple(rows[:10]),
        family_to_id={family: index for index, family in enumerate(families)},
        id_to_family={index: family for index, family in enumerate(families)},
        train_visible_model_to_id={model: index for index, model in enumerate(models)},
        id_to_train_visible_model={index: model for index, model in enumerate(models)},
        model_to_family_id={model: index for index, model in enumerate(models)},
    )
    a0 = resolve_phase3_recipe(load_phase3_config(CONFIG_PATH), "A0")
    a3 = replace(
        a0,
        stage_id="A3",
        recipe_id="A0+A3",
        contrastive_weight=0.1,
        recipe_hash="f" * 64,
    )

    smoke = _smoke_data_view(view, a3, seed=42)
    batches = list(
        StructuredContrastiveBatchSampler(
            smoke.train_samples,
            batch_size=16,
            seed=42,
        )
    )

    assert len(smoke.train_samples) == 16
    assert len(batches) == 1
