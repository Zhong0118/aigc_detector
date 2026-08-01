"""Complete-matrix validation and promotion gates for Phase3 stages."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from research.text_attribution.phase3_config import (
    Phase3Config,
    ResolvedPhase3Recipe,
    load_resolved_recipe,
    resolve_phase3_recipe,
    write_resolved_recipe,
)


@dataclass(frozen=True)
class Phase3SelectionRequest:
    config: Phase3Config
    stage_id: str
    parent_run_root: Path | None
    candidate_run_roots: tuple[Path, ...]
    output_dir: Path
    upstream_decision: Path | None = None


@dataclass(frozen=True)
class Phase3GateResult:
    gate_id: str
    passed: bool
    actual: Any
    threshold: Any
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "passed": self.passed,
            "actual": self.actual,
            "threshold": self.threshold,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class Phase3StageDecision:
    stage_id: str
    status: str
    promoted: bool
    selected_recipe_id: str
    fallback_recipe_id: str
    promoted_recipe: ResolvedPhase3Recipe
    gates: tuple[Phase3GateResult, ...]
    summaries: Mapping[str, Any]
    candidate_evaluations: tuple[Mapping[str, Any], ...]
    raw_runs: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "selection_scope": "validation_only",
            "test_evaluated": False,
            "stage_id": self.stage_id,
            "status": self.status,
            "promoted": self.promoted,
            "selected_recipe_id": self.selected_recipe_id,
            "fallback_recipe_id": self.fallback_recipe_id,
            "promoted_recipe": self.promoted_recipe.to_dict(),
            "gates": [gate.to_dict() for gate in self.gates],
            "summaries": dict(self.summaries),
            "candidate_evaluations": [dict(row) for row in self.candidate_evaluations],
            "raw_runs": dict(self.raw_runs),
        }


@dataclass(frozen=True)
class _RunMetric:
    seed: int
    view_id: str
    macro_f1: float
    nll: float
    ece: float
    per_family_f1: Mapping[str, float]
    run_dir: Path
    run_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "view_id": self.view_id,
            "macro_f1": self.macro_f1,
            "nll": self.nll,
            "ece": self.ece,
            "per_family_f1": dict(self.per_family_f1),
            "run_dir": str(self.run_dir),
            "run_fingerprint": self.run_fingerprint,
        }


@dataclass(frozen=True)
class _RunMatrix:
    recipe: ResolvedPhase3Recipe
    identity: tuple[str, str, str]
    rows: Mapping[tuple[int, str], _RunMetric]


def select_phase3_stage(request: Phase3SelectionRequest) -> Phase3StageDecision:
    if request.stage_id not in {"A1", "A2", "A3", "A4"}:
        raise ValueError("selection stage_id must be A1, A2, A3, or A4")
    if request.stage_id == "A2" and request.upstream_decision is not None:
        skipped = _maybe_skip_a2(request)
        if skipped is not None:
            return skipped
    if request.parent_run_root is None:
        raise ValueError("selection requires a parent run root")
    if not request.candidate_run_roots:
        raise ValueError("selection requires at least one candidate run root")
    if request.stage_id == "A1" and len(request.candidate_run_roots) != 2:
        raise ValueError("A1 selection requires both canonical candidates")
    if request.stage_id != "A1" and len(request.candidate_run_roots) != 1:
        raise ValueError(f"{request.stage_id} selection requires one candidate")

    parent = _load_run_matrix(request.parent_run_root, request.config)
    candidates = tuple(
        _load_run_matrix(path, request.config)
        for path in request.candidate_run_roots
    )
    _validate_stage_recipes(request.stage_id, parent, candidates)
    for candidate in candidates:
        if candidate.identity != parent.identity:
            raise ValueError(
                "candidate matrix identity/fingerprint does not match parent matrix"
            )
        if tuple(candidate.rows) != tuple(parent.rows):
            raise ValueError("candidate and parent matrix keys do not match")
    if len({candidate.recipe.recipe_hash for candidate in candidates}) != len(
        candidates
    ):
        raise ValueError("duplicate candidate recipe matrices")

    evaluations: list[dict[str, Any]] = []
    evaluated: list[tuple[_RunMatrix, tuple[Phase3GateResult, ...], bool]] = []
    for candidate in candidates:
        gates = _evaluate_gates(parent, candidate, request.config)
        passed = all(gate.passed for gate in gates)
        evaluated.append((candidate, gates, passed))
        evaluations.append(
            {
                "recipe_id": candidate.recipe.recipe_id,
                "recipe_hash": candidate.recipe.recipe_hash,
                "passed": passed,
                "closed_macro_f1_mean": _metric_mean(
                    candidate, "closed", "macro_f1"
                ),
                "gates": [gate.to_dict() for gate in gates],
            }
        )

    selected_candidate, selected_gates, selected_passed = max(
        evaluated,
        key=lambda row: _selection_order_key(request.stage_id, row),
    )
    selected_recipe = (
        selected_candidate.recipe if selected_passed else parent.recipe
    )
    summaries = MappingProxyType(
        {
            "parent": _summarize_matrix(parent),
            "candidate": _summarize_matrix(selected_candidate),
            "all_candidates": {
                candidate.recipe.recipe_id: _summarize_matrix(candidate)
                for candidate in candidates
            },
        }
    )
    raw_runs = MappingProxyType(
        {
            "parent": [
                row.to_dict() for row in parent.rows.values()
            ],
            "candidates": {
                candidate.recipe.recipe_id: [
                    row.to_dict() for row in candidate.rows.values()
                ]
                for candidate in candidates
            },
        }
    )
    return Phase3StageDecision(
        stage_id=request.stage_id,
        status="selected",
        promoted=selected_passed,
        selected_recipe_id=selected_recipe.recipe_id,
        fallback_recipe_id=parent.recipe.recipe_id,
        promoted_recipe=selected_recipe,
        gates=selected_gates,
        summaries=summaries,
        candidate_evaluations=tuple(evaluations),
        raw_runs=raw_runs,
    )


def write_phase3_stage_decision(
    decision: Phase3StageDecision,
    output_dir: str | Path,
) -> Mapping[str, Path]:
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError(f"selection output directory already exists: {root}")
    root.mkdir(parents=True, exist_ok=False)
    decision_path = root / "stage_decision.json"
    report_path = root / "selection_report.md"
    recipe_path = root / "promoted_recipe.json"
    decision_path.write_text(
        json.dumps(decision.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_resolved_recipe(recipe_path, decision.promoted_recipe)
    report_path.write_text(_render_decision_markdown(decision), encoding="utf-8")
    return MappingProxyType(
        {
            "decision": decision_path,
            "report": report_path,
            "promoted_recipe": recipe_path,
        }
    )


def _load_run_matrix(
    root: Path,
    config: Phase3Config,
    seeds: Sequence[int] | None = None,
) -> _RunMatrix:
    root = Path(root)
    expected_seeds = tuple(config.screening_seeds if seeds is None else seeds)
    completed_paths = sorted(root.rglob("COMPLETED.json"))
    if not completed_paths:
        raise ValueError(f"complete run matrix missing under {root}")
    rows: dict[tuple[int, str], _RunMetric] = {}
    recipe: ResolvedPhase3Recipe | None = None
    identity: tuple[str, str, str] | None = None
    fingerprints: set[str] = set()
    for completed_path in completed_paths:
        run_dir = completed_path.parent
        completed = _read_json(completed_path)
        run_config = _read_json(run_dir / "run_config.json")
        if completed.get("status") != "complete":
            raise ValueError(f"smoke/preflight run cannot enter selection: {run_dir}")
        if run_config.get("smoke_test") is not False or run_config.get(
            "preflight_only"
        ) is not False:
            raise ValueError(f"smoke/preflight run cannot enter selection: {run_dir}")
        if completed.get("test_evaluated") is not False or run_config.get(
            "test_evaluated"
        ) is not False:
            raise ValueError(f"test-evaluated run cannot enter selection: {run_dir}")
        if completed.get("selection_scope") != "validation_only" or run_config.get(
            "selection_scope"
        ) != "validation_only":
            raise ValueError(f"run is not validation-only: {run_dir}")

        run_recipe = load_resolved_recipe(run_dir / "resolved_recipe.json")
        if completed.get("recipe_hash") != run_recipe.recipe_hash or run_config.get(
            "recipe_hash"
        ) != run_recipe.recipe_hash:
            raise ValueError(f"recipe hash mismatch in run {run_dir}")
        if recipe is None:
            recipe = run_recipe
        elif recipe.recipe_hash != run_recipe.recipe_hash:
            raise ValueError("one run matrix contains multiple recipe hashes")

        seed = run_config.get("training_seed")
        view_id = run_config.get("view_id")
        if seed not in expected_seeds or view_id not in config.data_views:
            raise ValueError(f"unexpected seed/view in run matrix: {run_dir}")
        key = (seed, view_id)
        if key in rows:
            raise ValueError(f"duplicate seed/view run in matrix: {key!r}")
        fingerprint = run_config.get("run_fingerprint")
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or fingerprint in fingerprints
            or completed.get("run_fingerprint") != fingerprint
        ):
            raise ValueError(f"invalid or duplicate run fingerprint: {run_dir}")
        fingerprints.add(fingerprint)

        run_identity = (
            run_config.get("split_dataset_fingerprint"),
            run_config.get("split_manifest_hash"),
            run_config.get("catalog_hash"),
        )
        if any(not isinstance(value, str) or len(value) != 64 for value in run_identity):
            raise ValueError(f"run identity fields are missing or invalid: {run_dir}")
        if identity is None:
            identity = run_identity
        elif identity != run_identity:
            raise ValueError("run matrix contains mismatched data fingerprints")

        metrics = _read_json(run_dir / "metrics.json")
        try:
            validation = metrics[run_recipe.recipe_id]["validation"]
            macro_f1 = _finite_float(validation["overall"]["macro_f1"])
            nll = _finite_float(
                validation["calibration"]["negative_log_likelihood"]
            )
            ece = _finite_float(
                validation["calibration"]["expected_calibration_error"]
            )
            per_family = {
                family: _finite_float(values["f1-score"])
                for family, values in validation["per_family"].items()
            }
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"invalid metrics contract in {run_dir}") from exc
        if not per_family or any(not 0 <= value <= 1 for value in per_family.values()):
            raise ValueError(f"invalid per-family F1 values in {run_dir}")
        if not 0 <= macro_f1 <= 1 or nll < 0 or not 0 <= ece <= 1:
            raise ValueError(f"invalid aggregate metric values in {run_dir}")
        rows[key] = _RunMetric(
            seed=seed,
            view_id=view_id,
            macro_f1=macro_f1,
            nll=nll,
            ece=ece,
            per_family_f1=MappingProxyType(per_family),
            run_dir=run_dir,
            run_fingerprint=fingerprint,
        )

    expected = {
        (seed, view)
        for seed in expected_seeds
        for view in config.data_views
    }
    missing = expected - set(rows)
    extra = set(rows) - expected
    expected_count = len(expected_seeds) * len(config.data_views)
    if missing or extra or len(rows) != expected_count:
        raise ValueError(
            f"complete {len(expected_seeds)}-seed x {len(config.data_views)}-view "
            "matrix required; "
            f"missing={sorted(missing)!r}, extra={sorted(extra)!r}"
        )
    assert recipe is not None and identity is not None
    family_orders = {tuple(row.per_family_f1) for row in rows.values()}
    if len(family_orders) != 1:
        raise ValueError("run matrix family label order mismatch")
    ordered_rows = {
        (seed, view): rows[(seed, view)]
        for seed in expected_seeds
        for view in config.data_views
    }
    return _RunMatrix(
        recipe=recipe,
        identity=identity,
        rows=MappingProxyType(ordered_rows),
    )


def build_phase3_confirmation_summary(
    *,
    config: Phase3Config,
    winner_screening_root: Path,
    winner_confirmation_root: Path,
    comparator_screening_root: Path,
    comparator_confirmation_root: Path,
) -> dict[str, Any]:
    """Combine frozen 3+5 seed matrices without reopening promotion decisions."""

    winner = _merge_confirmation_matrices(
        _load_run_matrix(winner_screening_root, config, config.screening_seeds),
        _load_run_matrix(
            winner_confirmation_root, config, config.confirmation_seeds
        ),
    )
    comparator = _merge_confirmation_matrices(
        _load_run_matrix(comparator_screening_root, config, config.screening_seeds),
        _load_run_matrix(
            comparator_confirmation_root, config, config.confirmation_seeds
        ),
    )
    if winner.identity != comparator.identity:
        raise ValueError("confirmation recipes use mismatched data fingerprints")
    if winner.recipe.recipe_hash == comparator.recipe.recipe_hash:
        raise ValueError("confirmation winner and comparator must be distinct recipes")
    view_differences = {
        view: _metric_mean(winner, view, "macro_f1")
        - _metric_mean(comparator, view, "macro_f1")
        for view in config.data_views
    }
    return {
        "schema_version": 1,
        "selection_scope": "validation_only",
        "test_evaluated": False,
        "seed_role": "post_selection_confirmation",
        "seeds": [*config.screening_seeds, *config.confirmation_seeds],
        "winner_recipe": winner.recipe.to_dict(),
        "comparator_recipe": comparator.recipe.to_dict(),
        "winner": _summarize_matrix(winner),
        "comparator": _summarize_matrix(comparator),
        "winner_minus_comparator_macro_f1": view_differences,
        "closed_seed_wins": sum(
            winner.rows[(seed, "closed")].macro_f1
            > comparator.rows[(seed, "closed")].macro_f1
            for seed in (*config.screening_seeds, *config.confirmation_seeds)
        ),
    }


def _merge_confirmation_matrices(
    screening: _RunMatrix,
    confirmation: _RunMatrix,
) -> _RunMatrix:
    if screening.recipe.recipe_hash != confirmation.recipe.recipe_hash:
        raise ValueError("screening and confirmation recipe hashes do not match")
    if screening.identity != confirmation.identity:
        raise ValueError("screening and confirmation data fingerprints do not match")
    overlap = set(screening.rows) & set(confirmation.rows)
    if overlap:
        raise ValueError(f"screening and confirmation matrices overlap: {sorted(overlap)!r}")
    return _RunMatrix(
        recipe=screening.recipe,
        identity=screening.identity,
        rows=MappingProxyType({**screening.rows, **confirmation.rows}),
    )


def _evaluate_gates(
    parent: _RunMatrix,
    candidate: _RunMatrix,
    config: Phase3Config,
) -> tuple[Phase3GateResult, ...]:
    gates = config.gates
    closed_gain = _metric_mean(candidate, "closed", "macro_f1") - _metric_mean(
        parent, "closed", "macro_f1"
    )
    seed_wins = sum(
        candidate.rows[(seed, "closed")].macro_f1
        > parent.rows[(seed, "closed")].macro_f1
        for seed in config.screening_seeds
    )
    worst_gain = _mean(
        min(candidate.rows[(seed, "closed")].per_family_f1.values())
        for seed in config.screening_seeds
    ) - _mean(
        min(parent.rows[(seed, "closed")].per_family_f1.values())
        for seed in config.screening_seeds
    )
    mimo_gain = _family_mean(candidate, "closed", "mimo") - _family_mean(
        parent, "closed", "mimo"
    )
    glm_gain = _family_mean(candidate, "closed", "glm") - _family_mean(
        parent, "closed", "glm"
    )
    holdout_views = ("fast-to-balanced", "balanced-to-fast")
    combined_gain = _mean(
        candidate.rows[(seed, view)].macro_f1
        for seed in config.screening_seeds
        for view in holdout_views
    ) - _mean(
        parent.rows[(seed, view)].macro_f1
        for seed in config.screening_seeds
        for view in holdout_views
    )
    direction_gains = {
        view: _metric_mean(candidate, view, "macro_f1")
        - _metric_mean(parent, view, "macro_f1")
        for view in holdout_views
    }
    zero_entries = [
        {
            "seed": row.seed,
            "view_id": row.view_id,
            "family": family,
        }
        for row in candidate.rows.values()
        if row.view_id == "closed"
        for family, value in row.per_family_f1.items()
        if value <= 0.0
    ]
    return (
        _gate("closed_macro_f1_gain", closed_gain, gates.closed_macro_f1_min_gain),
        _gate("closed_seed_wins", seed_wins, gates.closed_seed_wins_min),
        _gate("worst_family_f1_gain", worst_gain, gates.worst_family_min_gain),
        _gate("mimo_f1_gain", mimo_gain, gates.mimo_min_gain),
        _gate("glm_f1_gain", glm_gain, gates.glm_min_gain),
        _gate(
            "holdout_combined_gain",
            combined_gain,
            gates.holdout_combined_min_gain,
        ),
        _gate(
            "fast_to_balanced_gain",
            direction_gains["fast-to-balanced"],
            gates.holdout_direction_min_gain,
        ),
        _gate(
            "balanced_to_fast_gain",
            direction_gains["balanced-to-fast"],
            gates.holdout_direction_min_gain,
        ),
        Phase3GateResult(
            gate_id="no_zero_family_f1",
            passed=not zero_entries if gates.forbid_zero_family_f1 else True,
            actual={"zero_count": len(zero_entries), "examples": zero_entries[:10]},
            threshold={"zero_count": 0},
            detail="No candidate closed seed/family may have F1 equal to zero.",
        ),
    )


def _validate_stage_recipes(
    stage_id: str,
    parent: _RunMatrix,
    candidates: Sequence[_RunMatrix],
) -> None:
    if stage_id == "A1":
        if parent.recipe.stage_id != "A0":
            raise ValueError("A1 selection parent must be A0")
        allowed = {"A1-alpha01", "A1-alpha03"}
        if any(candidate.recipe.recipe_id not in allowed for candidate in candidates):
            raise ValueError("A1 selection candidates must be canonical A1 recipes")
        if {candidate.recipe.recipe_id for candidate in candidates} != allowed:
            raise ValueError("A1 selection requires both canonical A1 recipes")
        return
    if any(candidate.recipe.stage_id != stage_id for candidate in candidates):
        raise ValueError(f"{stage_id} selection received a wrong-stage candidate")
    allowed_parents = {
        "A2": {"A1-alpha01", "A1-alpha03"},
        "A3": {"A0", "A1-alpha01", "A1-alpha03"},
        "A4": set(),
    }
    if stage_id == "A4":
        if parent.recipe.stage_id not in {
            "A0",
            "A1-alpha01",
            "A1-alpha03",
            "A2",
            "A3",
        }:
            raise ValueError("A4 selection parent must be the A3 decision winner")
    elif stage_id == "A3":
        if parent.recipe.stage_id not in {"A0", "A1-alpha01", "A1-alpha03", "A2"}:
            raise ValueError("A3 selection parent must be an A0/A1/A2 winner")
    elif parent.recipe.recipe_id not in allowed_parents[stage_id]:
        raise ValueError("A2 selection parent must be a promoted canonical A1")
    if any(
        candidate.recipe.parent_recipe_id != parent.recipe.recipe_id
        for candidate in candidates
    ):
        raise ValueError(f"{stage_id} candidate parent recipe does not match parent matrix")


def _gate(gate_id: str, actual: float | int, threshold: float | int):
    return Phase3GateResult(
        gate_id=gate_id,
        passed=actual >= threshold,
        actual=actual,
        threshold=threshold,
        detail=f"Require {gate_id} >= {threshold}.",
    )


def _summarize_matrix(matrix: _RunMatrix) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for view_id in ("closed", "fast-to-balanced", "balanced-to-fast"):
        rows = [row for row in matrix.rows.values() if row.view_id == view_id]
        summary[view_id] = {
            "macro_f1": _summary_values([row.macro_f1 for row in rows]),
            "nll": _summary_values([row.nll for row in rows]),
            "ece": _summary_values([row.ece for row in rows]),
            "worst_family_f1": _summary_values(
                [min(row.per_family_f1.values()) for row in rows]
            ),
            "per_family_f1": {
                family: _summary_values(
                    [row.per_family_f1[family] for row in rows]
                )
                for family in rows[0].per_family_f1
            },
        }
    return summary


def _summary_values(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": _mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _metric_mean(matrix: _RunMatrix, view_id: str, metric: str) -> float:
    return _mean(
        getattr(row, metric)
        for row in matrix.rows.values()
        if row.view_id == view_id
    )


def _family_mean(matrix: _RunMatrix, view_id: str, family: str) -> float:
    values = [
        row.per_family_f1[family]
        for row in matrix.rows.values()
        if row.view_id == view_id and family in row.per_family_f1
    ]
    if len(values) != 3:
        raise ValueError(f"formal selection requires family {family!r} in all seeds")
    return _mean(values)


def _mean(values) -> float:
    values = tuple(values)
    if not values:
        raise ValueError("cannot summarize empty metric values")
    return float(sum(values) / len(values))


def _finite_float(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("metric must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError("metric must be finite")
    return normalized


def _a1_recipe_order(recipe_id: str) -> int:
    order = {"A1-alpha01": 0, "A1-alpha03": 1}
    return order.get(recipe_id, 0)


def _selection_order_key(stage_id: str, row) -> tuple[Any, ...]:
    matrix, _gates, passed = row
    if stage_id != "A1":
        return (passed, _metric_mean(matrix, "closed", "macro_f1"))
    holdout_mean = _mean(
        metric.macro_f1
        for metric in matrix.rows.values()
        if metric.view_id in {"fast-to-balanced", "balanced-to-fast"}
    )
    closed_worst = _mean(
        min(metric.per_family_f1.values())
        for metric in matrix.rows.values()
        if metric.view_id == "closed"
    )
    return (
        passed,
        _metric_mean(matrix, "closed", "macro_f1"),
        holdout_mean,
        closed_worst,
        -_metric_mean(matrix, "closed", "nll"),
        -_metric_mean(matrix, "closed", "ece"),
        -_a1_recipe_order(matrix.recipe.recipe_id),
    )


def _maybe_skip_a2(
    request: Phase3SelectionRequest,
) -> Phase3StageDecision | None:
    payload = _read_json(request.upstream_decision)
    if payload.get("stage_id") != "A1":
        raise ValueError("A2 upstream decision must be an A1 decision")
    if payload.get("promoted") is True:
        return None
    if not (
        payload.get("promoted") is False
        and payload.get("fallback_recipe_id") == "A0"
        and payload.get("selected_recipe_id") == "A0"
    ):
        raise ValueError("A2 skip requires an explicit A1 fallback to A0")
    fallback = resolve_phase3_recipe(request.config, "A0")
    return Phase3StageDecision(
        stage_id="A2",
        status="skipped_by_gate",
        promoted=False,
        selected_recipe_id="A0",
        fallback_recipe_id="A0",
        promoted_recipe=fallback,
        gates=(),
        summaries=MappingProxyType({}),
        candidate_evaluations=(),
        raw_runs=MappingProxyType({}),
    )


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {source}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {source}")
    return payload


def _render_decision_markdown(decision: Phase3StageDecision) -> str:
    lines = [
        f"# Phase3 {decision.stage_id} Selection",
        "",
        f"- Status: `{decision.status}`",
        f"- Promoted: `{str(decision.promoted).lower()}`",
        f"- Selected recipe: `{decision.selected_recipe_id}`",
        f"- Fallback recipe: `{decision.fallback_recipe_id}`",
        "- Scope: `validation_only`",
        "- Test evaluated: `false`",
        "",
        "| Gate | Passed | Actual | Threshold |",
        "| --- | --- | --- | --- |",
    ]
    for gate in decision.gates:
        lines.append(
            f"| {gate.gate_id} | {str(gate.passed).lower()} | "
            f"`{gate.actual}` | `{gate.threshold}` |"
        )
    return "\n".join(lines) + "\n"
