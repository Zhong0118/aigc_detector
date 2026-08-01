from __future__ import annotations

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
