from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from research.text_attribution.phase3_config import (
    load_phase3_config,
    resolve_phase3_recipe,
    write_resolved_recipe,
)


CONFIG_PATH = (
    Path(__file__).parents[1]
    / "configs/text_attribution/training/family_phase3_enhancement.v1.json"
)
SEEDS = (42, 52, 62)
VIEWS = ("closed", "fast-to-balanced", "balanced-to-fast")
FAMILIES = ("glm", "mimo", "other")


def _write_matrix(
    root: Path,
    recipe,
    *,
    macro_f1: float,
    family_f1: float | None = None,
    zero_mimo_seed: int | None = None,
    zero_mimo_view: str | None = None,
    view_macro_f1: dict[str, float] | None = None,
    nll: float | None = None,
    ece: float = 0.1,
    seeds: tuple[int, ...] = SEEDS,
    dataset_fingerprint: str = "d" * 64,
) -> Path:
    family_value = macro_f1 if family_f1 is None else family_f1
    for seed in seeds:
        for view in VIEWS:
            view_score = (
                macro_f1 if view_macro_f1 is None else view_macro_f1.get(view, macro_f1)
            )
            run = root / f"{recipe.recipe_id}-{view}-seed{seed}"
            run.mkdir(parents=True)
            per_family = {
                family: {
                    "precision": family_value,
                    "recall": family_value,
                    "f1-score": (
                        0.0
                        if family == "mimo"
                        and seed == zero_mimo_seed
                        and (zero_mimo_view is None or view == zero_mimo_view)
                        else family_value
                    ),
                    "support": 10,
                }
                for family in FAMILIES
            }
            metrics = {
                recipe.recipe_id: {
                    "validation": {
                        "overall": {
                            "accuracy": view_score,
                            "macro_f1": view_score,
                            "weighted_f1": view_score,
                            "support": 30,
                        },
                        "per_family": per_family,
                        "calibration": {
                            "negative_log_likelihood": (
                                1.0 - view_score if nll is None else nll
                            ),
                            "expected_calibration_error": ece,
                            "multiclass_brier_score": 0.5,
                        },
                    }
                }
            }
            (run / "metrics.json").write_text(
                json.dumps(metrics), encoding="utf-8"
            )
            (run / "run_config.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "recipe_id": recipe.recipe_id,
                        "recipe_hash": recipe.recipe_hash,
                        "run_fingerprint": f"{seed:02x}{VIEWS.index(view):02x}" + "f" * 60,
                        "selection_scope": "validation_only",
                        "test_evaluated": False,
                        "view_id": view,
                        "training_seed": seed,
                        "split_dataset_fingerprint": dataset_fingerprint,
                        "split_manifest_hash": "e" * 64,
                        "catalog_hash": "c" * 64,
                        "smoke_test": False,
                        "preflight_only": False,
                    }
                ),
                encoding="utf-8",
            )
            write_resolved_recipe(run / "resolved_recipe.json", recipe)
            (run / "COMPLETED.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "complete",
                        "recipe_hash": recipe.recipe_hash,
                        "run_fingerprint": f"{seed:02x}{VIEWS.index(view):02x}" + "f" * 60,
                        "selection_scope": "validation_only",
                        "test_evaluated": False,
                    }
                ),
                encoding="utf-8",
            )
    return root


def _request(tmp_path, parent: Path, candidates: tuple[Path, ...]):
    from research.text_attribution.phase3_selection import Phase3SelectionRequest

    if len(candidates) == 1:
        config = load_phase3_config(CONFIG_PATH)
        existing_recipe = next(candidates[0].rglob("resolved_recipe.json"))
        existing = json.loads(existing_recipe.read_text(encoding="utf-8"))
        missing_id = (
            "A1-alpha03"
            if existing["recipe_id"] == "A1-alpha01"
            else "A1-alpha01"
        )
        missing_root = tmp_path / f"auto-{missing_id}"
        _clone_matrix_as(
            candidates[0],
            missing_root,
            resolve_phase3_recipe(config, missing_id),
        )
        candidates = (*candidates, missing_root)
    return Phase3SelectionRequest(
        config=load_phase3_config(CONFIG_PATH),
        stage_id="A1",
        parent_run_root=parent,
        candidate_run_roots=candidates,
        output_dir=tmp_path / "selection",
    )


def _clone_matrix_as(source: Path, target: Path, recipe) -> None:
    for run in sorted(path.parent for path in source.rglob("COMPLETED.json")):
        destination = target / run.name
        shutil.copytree(run, destination)
        metrics_path = destination / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        old_id = next(iter(metrics))
        metrics[recipe.recipe_id] = metrics.pop(old_id)
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        run_config_path = destination / "run_config.json"
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        run_config["recipe_id"] = recipe.recipe_id
        run_config["recipe_hash"] = recipe.recipe_hash
        run_config_path.write_text(json.dumps(run_config), encoding="utf-8")
        completed_path = destination / "COMPLETED.json"
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
        completed["recipe_hash"] = recipe.recipe_hash
        completed_path.write_text(json.dumps(completed), encoding="utf-8")
        write_resolved_recipe(destination / "resolved_recipe.json", recipe)


def test_promotes_candidate_only_when_all_nine_gates_pass(tmp_path) -> None:
    from research.text_attribution.phase3_selection import select_phase3_stage

    config = load_phase3_config(CONFIG_PATH)
    parent_recipe = resolve_phase3_recipe(config, "A0")
    candidate_recipe = resolve_phase3_recipe(config, "A1-alpha01")
    parent = _write_matrix(tmp_path / "parent", parent_recipe, macro_f1=0.50)
    candidate = _write_matrix(
        tmp_path / "candidate", candidate_recipe, macro_f1=0.52
    )

    decision = select_phase3_stage(_request(tmp_path, parent, (candidate,)))
    assert decision.promoted is True
    assert decision.selected_recipe_id == "A1-alpha01"
    assert len(decision.gates) == 9
    assert all(gate.passed for gate in decision.gates)
    assert decision.summaries["candidate"]["closed"]["macro_f1"]["mean"] == pytest.approx(0.52)


def test_zero_family_f1_forces_fallback_even_with_higher_macro(tmp_path) -> None:
    from research.text_attribution.phase3_selection import select_phase3_stage

    config = load_phase3_config(CONFIG_PATH)
    parent = _write_matrix(
        tmp_path / "parent", resolve_phase3_recipe(config, "A0"), macro_f1=0.50
    )
    candidate = _write_matrix(
        tmp_path / "candidate",
        resolve_phase3_recipe(config, "A1-alpha01"),
        macro_f1=0.60,
        zero_mimo_seed=42,
    )
    decision = select_phase3_stage(_request(tmp_path, parent, (candidate,)))
    assert decision.promoted is False
    assert decision.fallback_recipe_id == "A0"
    assert next(gate for gate in decision.gates if gate.gate_id == "no_zero_family_f1").passed is False


def test_zero_family_f1_gate_only_applies_to_closed_runs(tmp_path) -> None:
    from research.text_attribution.phase3_selection import select_phase3_stage

    config = load_phase3_config(CONFIG_PATH)
    parent = _write_matrix(
        tmp_path / "parent", resolve_phase3_recipe(config, "A0"), macro_f1=0.50
    )
    candidate = _write_matrix(
        tmp_path / "candidate",
        resolve_phase3_recipe(config, "A1-alpha01"),
        macro_f1=0.52,
        zero_mimo_seed=42,
        zero_mimo_view="fast-to-balanced",
    )
    decision = select_phase3_stage(_request(tmp_path, parent, (candidate,)))
    gate = next(gate for gate in decision.gates if gate.gate_id == "no_zero_family_f1")
    assert gate.passed is True


def test_incomplete_matrix_is_rejected(tmp_path) -> None:
    from research.text_attribution.phase3_selection import select_phase3_stage

    config = load_phase3_config(CONFIG_PATH)
    parent = _write_matrix(
        tmp_path / "parent", resolve_phase3_recipe(config, "A0"), macro_f1=0.50
    )
    candidate = _write_matrix(
        tmp_path / "candidate",
        resolve_phase3_recipe(config, "A1-alpha01"),
        macro_f1=0.52,
    )
    shutil.rmtree(candidate / "A1-alpha01-closed-seed62")
    with pytest.raises(ValueError, match="complete.*matrix|matrix.*missing"):
        select_phase3_stage(_request(tmp_path, parent, (candidate,)))


@pytest.mark.parametrize("bad_field", ["smoke_test", "test_evaluated"])
def test_smoke_or_test_evaluated_run_is_rejected(tmp_path, bad_field: str) -> None:
    from research.text_attribution.phase3_selection import select_phase3_stage

    config = load_phase3_config(CONFIG_PATH)
    parent = _write_matrix(
        tmp_path / "parent", resolve_phase3_recipe(config, "A0"), macro_f1=0.50
    )
    candidate = _write_matrix(
        tmp_path / "candidate",
        resolve_phase3_recipe(config, "A1-alpha01"),
        macro_f1=0.52,
    )
    run_config_path = candidate / "A1-alpha01-closed-seed42/run_config.json"
    payload = json.loads(run_config_path.read_text(encoding="utf-8"))
    payload[bad_field] = True
    run_config_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="smoke|test"):
        select_phase3_stage(_request(tmp_path, parent, (candidate,)))


def test_matrix_identity_mismatch_is_rejected(tmp_path) -> None:
    from research.text_attribution.phase3_selection import select_phase3_stage

    config = load_phase3_config(CONFIG_PATH)
    parent = _write_matrix(
        tmp_path / "parent", resolve_phase3_recipe(config, "A0"), macro_f1=0.50
    )
    candidate = _write_matrix(
        tmp_path / "candidate",
        resolve_phase3_recipe(config, "A1-alpha01"),
        macro_f1=0.52,
        dataset_fingerprint="x" * 64,
    )
    with pytest.raises(ValueError, match="fingerprint|identity"):
        select_phase3_stage(_request(tmp_path, parent, (candidate,)))


def test_a1_tie_uses_canonical_alpha_order(tmp_path) -> None:
    from research.text_attribution.phase3_selection import select_phase3_stage

    config = load_phase3_config(CONFIG_PATH)
    parent = _write_matrix(
        tmp_path / "parent", resolve_phase3_recipe(config, "A0"), macro_f1=0.50
    )
    alpha01 = _write_matrix(
        tmp_path / "alpha01",
        resolve_phase3_recipe(config, "A1-alpha01"),
        macro_f1=0.52,
    )
    alpha03 = _write_matrix(
        tmp_path / "alpha03",
        resolve_phase3_recipe(config, "A1-alpha03"),
        macro_f1=0.52,
    )
    decision = select_phase3_stage(_request(tmp_path, parent, (alpha03, alpha01)))
    assert decision.selected_recipe_id == "A1-alpha01"


def test_a1_tie_break_prefers_bidirectional_holdout_before_alpha(tmp_path) -> None:
    from research.text_attribution.phase3_selection import select_phase3_stage

    config = load_phase3_config(CONFIG_PATH)
    parent = _write_matrix(
        tmp_path / "parent", resolve_phase3_recipe(config, "A0"), macro_f1=0.50
    )
    alpha01 = _write_matrix(
        tmp_path / "alpha01",
        resolve_phase3_recipe(config, "A1-alpha01"),
        macro_f1=0.52,
    )
    alpha03 = _write_matrix(
        tmp_path / "alpha03",
        resolve_phase3_recipe(config, "A1-alpha03"),
        macro_f1=0.52,
        view_macro_f1={"closed": 0.52, "fast-to-balanced": 0.55, "balanced-to-fast": 0.55},
    )
    decision = select_phase3_stage(_request(tmp_path, parent, (alpha01, alpha03)))
    assert decision.selected_recipe_id == "A1-alpha03"


def test_a1_requires_both_canonical_candidates(tmp_path) -> None:
    from research.text_attribution.phase3_selection import (
        Phase3SelectionRequest,
        select_phase3_stage,
    )

    config = load_phase3_config(CONFIG_PATH)
    parent = _write_matrix(
        tmp_path / "parent", resolve_phase3_recipe(config, "A0"), macro_f1=0.50
    )
    alpha01 = _write_matrix(
        tmp_path / "alpha01",
        resolve_phase3_recipe(config, "A1-alpha01"),
        macro_f1=0.52,
    )
    request = Phase3SelectionRequest(
        config=config,
        stage_id="A1",
        parent_run_root=parent,
        candidate_run_roots=(alpha01,),
        output_dir=tmp_path / "selection",
    )
    with pytest.raises(ValueError, match="both|two"):
        select_phase3_stage(request)


def test_a2_can_be_skipped_only_after_explicit_a1_fallback(tmp_path) -> None:
    from research.text_attribution.phase3_selection import (
        Phase3SelectionRequest,
        select_phase3_stage,
    )

    upstream = tmp_path / "a1-decision.json"
    upstream.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage_id": "A1",
                "status": "selected",
                "promoted": False,
                "selected_recipe_id": "A0",
                "fallback_recipe_id": "A0",
            }
        ),
        encoding="utf-8",
    )
    request = Phase3SelectionRequest(
        config=load_phase3_config(CONFIG_PATH),
        stage_id="A2",
        parent_run_root=None,
        candidate_run_roots=(),
        output_dir=tmp_path / "selection",
        upstream_decision=upstream,
    )
    decision = select_phase3_stage(request)
    assert decision.status == "skipped_by_gate"
    assert decision.promoted is False
    assert decision.fallback_recipe_id == "A0"


def test_a4_accepts_the_recipe_selected_by_an_a3_fallback(tmp_path) -> None:
    from research.text_attribution.phase3_selection import (
        Phase3SelectionRequest,
        select_phase3_stage,
    )

    config = load_phase3_config(CONFIG_PATH)
    a0 = resolve_phase3_recipe(config, "A0")
    a3_decision = tmp_path / "a3-decision.json"
    a3_decision.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage_id": "A3",
                "promoted": False,
                "promoted_recipe": a0.to_dict(),
            }
        ),
        encoding="utf-8",
    )
    a4 = resolve_phase3_recipe(config, "A4", a3_decision)
    parent = _write_matrix(tmp_path / "parent", a0, macro_f1=0.50)
    candidate = _write_matrix(tmp_path / "candidate", a4, macro_f1=0.52)

    decision = select_phase3_stage(
        Phase3SelectionRequest(
            config=config,
            stage_id="A4",
            parent_run_root=parent,
            candidate_run_roots=(candidate,),
            output_dir=tmp_path / "selection-a4",
        )
    )
    assert decision.promoted is True
    assert decision.selected_recipe_id == a4.recipe_id


def test_decision_serialization_writes_promoted_recipe(tmp_path) -> None:
    from research.text_attribution.phase3_selection import (
        select_phase3_stage,
        write_phase3_stage_decision,
    )

    config = load_phase3_config(CONFIG_PATH)
    parent = _write_matrix(
        tmp_path / "parent", resolve_phase3_recipe(config, "A0"), macro_f1=0.50
    )
    candidate = _write_matrix(
        tmp_path / "candidate",
        resolve_phase3_recipe(config, "A1-alpha01"),
        macro_f1=0.52,
    )
    decision = select_phase3_stage(_request(tmp_path, parent, (candidate,)))
    paths = write_phase3_stage_decision(decision, tmp_path / "selection")
    assert set(paths) == {"decision", "report", "promoted_recipe"}
    promoted = json.loads(paths["promoted_recipe"].read_text(encoding="utf-8"))
    assert promoted["recipe_id"] == "A1-alpha01"
    assert len(promoted["recipe_hash"]) == 64


def test_confirmation_summary_combines_frozen_eight_seeds(tmp_path) -> None:
    from research.text_attribution.phase3_selection import (
        build_phase3_confirmation_summary,
    )

    config = load_phase3_config(CONFIG_PATH)
    winner_recipe = resolve_phase3_recipe(config, "A1-alpha01")
    comparator_recipe = resolve_phase3_recipe(config, "A0")
    winner_screening = _write_matrix(
        tmp_path / "winner-screening", winner_recipe, macro_f1=0.55
    )
    comparator_screening = _write_matrix(
        tmp_path / "comparator-screening", comparator_recipe, macro_f1=0.50
    )
    winner_confirmation = _write_matrix(
        tmp_path / "winner-confirmation",
        winner_recipe,
        macro_f1=0.54,
        seeds=config.confirmation_seeds,
    )
    comparator_confirmation = _write_matrix(
        tmp_path / "comparator-confirmation",
        comparator_recipe,
        macro_f1=0.51,
        seeds=config.confirmation_seeds,
    )

    summary = build_phase3_confirmation_summary(
        config=config,
        winner_screening_root=winner_screening,
        winner_confirmation_root=winner_confirmation,
        comparator_screening_root=comparator_screening,
        comparator_confirmation_root=comparator_confirmation,
    )

    assert summary["seeds"] == [42, 52, 62, 12, 22, 32, 72, 82]
    assert summary["winner"]["closed"]["macro_f1"]["count"] == 8
    assert summary["closed_seed_wins"] == 8
    assert summary["test_evaluated"] is False
