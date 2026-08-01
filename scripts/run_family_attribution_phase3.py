#!/usr/bin/env python3
"""Run one leakage-safe Phase3 recipe × seed × validation-view experiment."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Sequence
from uuid import uuid4

from research.text_attribution.phase3_config import (
    ResolvedPhase3Recipe,
    load_phase3_config,
    load_resolved_recipe,
    resolve_phase3_recipe,
)
from research.text_attribution.phase3_data import (
    PHASE3_VIEW_ROLES,
    Phase3DataView,
    build_phase3_data_view,
    load_variant_catalog,
)
from research.text_attribution.split_manifest import (
    apply_split_manifest,
    load_split_manifest,
)
from research.text_attribution.training_data import load_training_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/text_attribution/training/family_phase3_enhancement.v1.json"
)
CANONICAL_STAGES = frozenset({"A0", "A1-alpha01", "A1-alpha03"})
PROMOTED_STAGES = frozenset({"A2", "A3", "A4"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train one Phase3 family-attribution run. This command is strictly "
            "validation-only and has no test-evaluation mode."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--stage", choices=tuple(sorted(CANONICAL_STAGES | PROMOTED_STAGES))
    )
    mode.add_argument("--recipe", help="Replay an exact resolved_recipe.json")
    parser.add_argument("--parent-decision")
    parser.add_argument("--input", required=True)
    parser.add_argument("--catalog")
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--view", choices=tuple(PHASE3_VIEW_ROLES), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--hf-cache")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.stage in PROMOTED_STAGES and not args.parent_decision:
        parser.error(f"--stage {args.stage} requires --parent-decision")
    if args.stage in CANONICAL_STAGES and args.parent_decision:
        parser.error("canonical stages must not receive --parent-decision")
    if args.recipe and args.parent_decision:
        parser.error("--recipe replay must not receive --parent-decision")
    if args.preflight_only and args.smoke_test:
        parser.error("--preflight-only and --smoke-test are distinct modes")
    return args


def resolve_cli_recipe(args: argparse.Namespace) -> ResolvedPhase3Recipe:
    if args.recipe:
        return load_resolved_recipe(args.recipe)
    config = load_phase3_config(args.config)
    return resolve_phase3_recipe(
        config,
        args.stage,
        None if args.parent_decision is None else Path(args.parent_decision),
    )


def compute_phase3_run_fingerprint(
    *,
    recipe_hash: str,
    dataset_fingerprint: str,
    split_manifest_hash: str,
    catalog_hash: str,
    view_id: str,
    seed: int,
) -> str:
    payload = {
        "catalog_hash": catalog_hash,
        "dataset_fingerprint": dataset_fingerprint,
        "recipe_hash": recipe_hash,
        "seed": seed,
        "split_manifest_hash": split_manifest_hash,
        "view_id": view_id,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_output_ownership(output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Phase3 output directory already exists: {output_dir}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_phase3_config(args.config)
    recipe = resolve_cli_recipe(args)
    if recipe.backbone_model_id != config.backbone_model_id:
        raise ValueError("resolved recipe backbone does not match Phase3 config")
    allowed_seeds = set(config.screening_seeds) | set(config.confirmation_seeds)
    if args.seed not in allowed_seeds:
        raise ValueError(
            f"Phase3 seed must be one of the frozen seeds: {sorted(allowed_seeds)!r}"
        )
    if not args.smoke_test and not args.dry_run and args.device != "cuda:0":
        raise ValueError("formal Phase3 training requires --device cuda:0")

    output_dir = Path(args.output_dir).resolve()
    validate_output_ownership(output_dir)
    input_path = Path(args.input).resolve()
    split_path = Path(args.split_manifest).resolve()
    if not split_path.is_file():
        raise FileNotFoundError(f"frozen split manifest not found: {split_path}")
    catalog_path = (
        Path(args.catalog).resolve()
        if args.catalog is not None
        else config.model_catalog
    )
    dataset = load_training_dataset(input_path, catalog_path)
    catalog = load_variant_catalog(catalog_path)
    manifest = load_split_manifest(split_path)
    splits = apply_split_manifest(dataset.samples, manifest)
    test_request_keys = frozenset(row.request_key for row in splits["test"])
    data_view = build_phase3_data_view(
        splits,
        catalog,
        args.view,
        test_request_keys=test_request_keys,
    )
    if args.smoke_test:
        data_view = _smoke_data_view(data_view)

    split_manifest_hash = _sha256_file(split_path)
    catalog_hash = _sha256_file(catalog_path)
    run_fingerprint = compute_phase3_run_fingerprint(
        recipe_hash=recipe.recipe_hash,
        dataset_fingerprint=manifest.dataset_fingerprint,
        split_manifest_hash=split_manifest_hash,
        catalog_hash=catalog_hash,
        view_id=data_view.view_id,
        seed=args.seed,
    )
    summary = _run_summary(
        args=args,
        config=config,
        recipe=recipe,
        data_view=data_view,
        manifest=manifest,
        input_path=input_path,
        catalog_path=catalog_path,
        split_path=split_path,
        run_fingerprint=run_fingerprint,
        split_manifest_hash=split_manifest_hash,
        catalog_hash=catalog_hash,
        test_request_key_count=len(test_request_keys),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(f".{output_dir.name}.run-{uuid4().hex}")
    try:
        from research.text_attribution.phase3_training import (
            Phase3TrainingRequest,
            train_phase3_experiment,
        )

        request = Phase3TrainingRequest(
            recipe=recipe,
            data_view=data_view,
            output_dir=staging,
            seed=args.seed,
            hf_cache=None if args.hf_cache is None else Path(args.hf_cache),
            bf16=not args.no_bf16,
            smoke_test=args.smoke_test,
            require_cuda=not args.smoke_test,
            resume_from_checkpoint=(
                None
                if args.resume_from_checkpoint is None
                else Path(args.resume_from_checkpoint)
            ),
            gradient_checkpointing=args.gradient_checkpointing,
            preflight_only=args.preflight_only,
        )
        result = train_phase3_experiment(request)
        if args.preflight_only:
            _write_json(staging / "run_config.json", summary)
            _write_json(
                staging / "COMPLETED.json",
                {
                    "schema_version": 1,
                    "status": "preflight_complete",
                    "recipe_hash": recipe.recipe_hash,
                    "run_fingerprint": run_fingerprint,
                    "test_evaluated": False,
                },
            )
            os.replace(staging, output_dir)
            return 0

        from research.text_attribution.attribution_metrics import (
            build_prediction_records,
            evaluate_phase3_predictions,
        )
        from research.text_attribution.attribution_reporting import (
            write_phase3_attribution_report,
        )

        family_labels = tuple(
            data_view.id_to_family[index]
            for index in range(len(data_view.id_to_family))
        )
        records = build_prediction_records(
            data_view.validation_samples,
            result.validation_probabilities,
            family_labels,
            experiment_id=recipe.recipe_id,
            split="validation",
        )
        validation_metrics = evaluate_phase3_predictions(
            records,
            family_labels,
            view_id=data_view.view_id,
            train_visible_model_labels=tuple(
                data_view.id_to_train_visible_model[index]
                for index in range(len(data_view.id_to_train_visible_model))
            ),
            window_mode=recipe.window_mode,
        )
        metrics = {
            recipe.recipe_id: {"validation": validation_metrics},
            "selection": {
                "selected_experiment": recipe.recipe_id,
                "selection_split": "validation",
                "selection_metric": "macro_f1",
                "criterion": "single_phase3_validation_view",
            },
        }
        report_dir = staging / "report_bundle"
        report_paths = write_phase3_attribution_report(
            output_dir=report_dir,
            run_config=summary,
            dataset_audit=dataset.audit,
            split_summary={
                "train": len(data_view.train_samples),
                "validation": len(data_view.validation_samples),
            },
            metrics=metrics,
            predictions=records,
        )
        for value in report_paths.values():
            source = Path(value)
            destination = staging / source.name
            if destination.exists():
                raise RuntimeError(
                    f"Phase3 report conflicts with training artifact: {destination}"
                )
            os.replace(source, destination)
        report_dir.rmdir()
        _write_json(
            staging / "view_manifest.json",
            {
                "schema_version": 1,
                "view_id": data_view.view_id,
                "train_sample_count": len(data_view.train_samples),
                "validation_sample_count": len(data_view.validation_samples),
                "train_visible_model_labels": list(
                    data_view.train_visible_model_to_id
                ),
                "test_request_key_count": len(test_request_keys),
                "test_evaluated": False,
            },
        )
        status = "smoke_complete" if args.smoke_test else "complete"
        _write_json(
            staging / "COMPLETED.json",
            {
                "schema_version": 1,
                "status": status,
                "recipe_hash": recipe.recipe_hash,
                "run_fingerprint": run_fingerprint,
                "selection_scope": "validation_only",
                "test_evaluated": False,
            },
        )
        os.replace(staging, output_dir)
    except BaseException as error:
        if staging.exists():
            failed = output_dir.with_name(
                f".{output_dir.name}.failed-{uuid4().hex}"
            )
            try:
                os.replace(staging, failed)
                if hasattr(error, "add_note"):
                    error.add_note(f"failed Phase3 artifacts preserved at {failed}")
            except OSError:
                pass
        raise
    return 0


def _smoke_data_view(view: Phase3DataView) -> Phase3DataView:
    def select(samples, per_model: int):
        selected = []
        counts: dict[str, int] = {}
        for row in sorted(samples, key=lambda item: item.request_key):
            count = counts.get(row.model_id, 0)
            if count < per_model:
                selected.append(row)
                counts[row.model_id] = count + 1
        return tuple(selected)

    return replace(
        view,
        train_samples=select(view.train_samples, 2),
        validation_samples=select(view.validation_samples, 1),
    )


def _run_summary(
    *,
    args,
    config,
    recipe,
    data_view,
    manifest,
    input_path,
    catalog_path,
    split_path,
    run_fingerprint,
    split_manifest_hash,
    catalog_hash,
    test_request_key_count,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": config.run_id,
        "experiments": [recipe.recipe_id],
        "stage_id": recipe.stage_id,
        "recipe_id": recipe.recipe_id,
        "recipe_hash": recipe.recipe_hash,
        "run_fingerprint": run_fingerprint,
        "selection_scope": "validation_only",
        "test_evaluated": False,
        "view_id": data_view.view_id,
        "training_seed": args.seed,
        "input": str(input_path),
        "catalog": str(catalog_path),
        "split_manifest": str(split_path),
        "split_dataset_fingerprint": manifest.dataset_fingerprint,
        "split_manifest_hash": split_manifest_hash,
        "catalog_hash": catalog_hash,
        "split_seed": manifest.seed,
        "train_samples": len(data_view.train_samples),
        "validation_samples": len(data_view.validation_samples),
        "test_request_key_count": test_request_key_count,
        "family_labels": list(data_view.family_to_id),
        "train_visible_model_labels": list(
            data_view.train_visible_model_to_id
        ),
        "backbone_model_id": recipe.backbone_model_id,
        "max_length": recipe.max_length,
        "window_mode": recipe.window_mode,
        "smoke_test": args.smoke_test,
        "preflight_only": args.preflight_only,
        "device": args.device,
        "bf16": recipe.bf16 and not args.no_bf16,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "versions": {"python": platform.python_version()},
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Phase3 run failed: {exc}", file=sys.stderr)
        raise
