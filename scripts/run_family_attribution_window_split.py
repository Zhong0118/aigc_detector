#!/usr/bin/env python3
"""Train A0 on non-overlapping token windows; evaluate on whole documents.

This is not A1–A4. It only changes how A0 consumes long rows: train windows
inherit family_id and related ids, validation stays one row per document so
metrics remain comparable to the truncated A0 baseline.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Sequence
from uuid import uuid4

from research.text_attribution.phase3_config import load_phase3_config, resolve_phase3_recipe
from research.text_attribution.phase3_data import (
    build_phase3_data_view,
    load_variant_catalog,
)
from research.text_attribution.split_manifest import apply_split_manifest, load_split_manifest
from research.text_attribution.training_data import load_training_dataset
from research.text_attribution.window_split import expand_samples_to_windows
from scripts.run_family_attribution_phase3 import (
    _build_validation_only_audit,
    _sha256_file,
    _write_json,
    compute_phase3_run_fingerprint,
    validate_output_ownership,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT / "configs/text_attribution/training/family_phase3_enhancement.v1.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="A0 window-split probe: expand train docs, keep document-level validation."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--input", required=True)
    parser.add_argument("--catalog")
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--view", default="closed", choices=("closed",))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--hf-cache")
    parser.add_argument("--model-source")
    parser.add_argument("--stride", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _local_xlmr_source(hf_cache: str | None, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"model source not found: {path}")
        return path
    cache_root = Path(hf_cache) if hf_cache else PROJECT_ROOT / "models/huggingface"
    refs = cache_root / "models--FacebookAI--xlm-roberta-base" / "refs" / "main"
    if refs.exists():
        snapshot = cache_root / "models--FacebookAI--xlm-roberta-base" / "snapshots" / refs.read_text(encoding="utf-8").strip()
        if snapshot.exists():
            return snapshot
    trained = (
        PROJECT_ROOT
        / "outputs/attribution/family-phase3/A0/A0-closed-seed42/tokenizer"
    )
    if trained.exists():
        return trained
    raise FileNotFoundError("local XLM-R snapshot not found; pass --model-source")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_phase3_config(args.config)
    recipe = resolve_phase3_recipe(config, "A0", None)
    if recipe.window_mode != "single":
        raise ValueError("window-split probe is defined only for A0 single-window recipes")
    if args.seed not in set(config.screening_seeds) | set(config.confirmation_seeds):
        raise ValueError(f"seed must be one of the frozen Phase3 seeds: {args.seed}")
    if not args.smoke_test and not args.dry_run and args.device != "cuda:0":
        raise ValueError("formal window-split training requires --device cuda:0")

    output_dir = Path(args.output_dir).resolve()
    validate_output_ownership(output_dir)
    input_path = Path(args.input).resolve()
    split_path = Path(args.split_manifest).resolve()
    catalog_path = (
        Path(args.catalog).resolve() if args.catalog is not None else config.model_catalog
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

    from transformers import AutoTokenizer

    model_source = _local_xlmr_source(args.hf_cache, args.model_source)
    tokenizer = AutoTokenizer.from_pretrained(str(model_source), local_files_only=True)
    train_windows, window_audit = expand_samples_to_windows(
        data_view.train_samples,
        tokenizer,
        max_length=recipe.max_length,
        stride=args.stride,
    )
    if args.smoke_test:
        train_windows = train_windows[:32]
        data_view = replace(
            data_view,
            train_samples=train_windows,
            validation_samples=data_view.validation_samples[:16],
        )
    else:
        data_view = replace(data_view, train_samples=train_windows)

    validation_only_audit = _build_validation_only_audit(
        (*splits["train"], *splits["validation"])
    )
    split_manifest_hash = _sha256_file(split_path)
    catalog_hash = _sha256_file(catalog_path)
    run_fingerprint = compute_phase3_run_fingerprint(
        recipe_hash=recipe.recipe_hash + ":window-split",
        dataset_fingerprint=manifest.dataset_fingerprint,
        split_manifest_hash=split_manifest_hash,
        catalog_hash=catalog_hash,
        view_id=data_view.view_id,
        seed=args.seed,
    )
    summary = {
        "schema_version": 1,
        "run_id": "family-attribution-a0-window-split-v1",
        "stage_id": "A0-window-split",
        "recipe_id": "A0-window-split",
        "parent_recipe_id": "A0",
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
        "train_source_documents": window_audit["source_documents"],
        "train_window_rows": window_audit["window_rows"]
        if not args.smoke_test
        else len(data_view.train_samples),
        "window_audit": window_audit,
        "validation_documents": len(data_view.validation_samples),
        "family_labels": list(data_view.family_to_id),
        "backbone_model_id": recipe.backbone_model_id,
        "model_source": str(model_source),
        "max_length": recipe.max_length,
        "window_mode": "train_nonoverlap_eval_document",
        "smoke_test": args.smoke_test,
        "device": args.device,
        "versions": {"python": platform.python_version()},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0

    from research.text_attribution.phase3_training import (
        Phase3TrainingRequest,
        train_phase3_experiment,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(f".{output_dir.name}.run-{uuid4().hex}")
    try:
        request = Phase3TrainingRequest(
            recipe=replace(recipe, recipe_id="A0-window-split", stage_id="A0-window-split"),
            data_view=data_view,
            output_dir=staging,
            seed=args.seed,
            run_fingerprint=run_fingerprint,
            dataset_fingerprint=manifest.dataset_fingerprint,
            split_manifest_hash=split_manifest_hash,
            catalog_hash=catalog_hash,
            hf_cache=None if args.hf_cache is None else Path(args.hf_cache),
            model_source=model_source,
            tokenizer_source=model_source,
            bf16=recipe.bf16,
            smoke_test=args.smoke_test,
            require_cuda=not args.smoke_test,
            use_cpu=args.device == "cpu",
        )
        result = train_phase3_experiment(request)
        from research.text_attribution.attribution_metrics import (
            build_prediction_records,
            evaluate_phase3_predictions,
        )
        from research.text_attribution.attribution_reporting import (
            write_phase3_attribution_report,
        )

        family_labels = tuple(
            data_view.id_to_family[index] for index in range(len(data_view.id_to_family))
        )
        records = build_prediction_records(
            data_view.validation_samples,
            result.validation_probabilities,
            family_labels,
            experiment_id="A0-window-split",
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
            window_mode="single",
        )
        metrics = {
            "A0-window-split": {"validation": validation_metrics},
            "selection": {
                "selected_experiment": "A0-window-split",
                "selection_split": "validation",
                "selection_metric": "macro_f1",
                "criterion": "window_split_probe",
            },
        }
        report_dir = staging / "report_bundle"
        report_paths = write_phase3_attribution_report(
            output_dir=report_dir,
            run_config=summary,
            dataset_audit=validation_only_audit,
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
            os.replace(source, destination)
        report_dir.rmdir()
        _write_json(staging / "run_config.json", summary)
        _write_json(staging / "window_audit.json", window_audit)
        _write_json(
            staging / "COMPLETED.json",
            {
                "schema_version": 1,
                "status": "smoke_complete" if args.smoke_test else "complete",
                "recipe_id": "A0-window-split",
                "run_fingerprint": run_fingerprint,
                "selection_scope": "validation_only",
                "test_evaluated": False,
                "baseline_a0_closed_seed42_macro_f1": 0.563941,
            },
        )
        os.replace(staging, output_dir)
    except BaseException as error:
        if staging.exists():
            failed = output_dir.with_name(f".{output_dir.name}.failed-{uuid4().hex}")
            try:
                os.replace(staging, failed)
                if hasattr(error, "add_note"):
                    error.add_note(f"failed artifacts preserved at {failed}")
            except OSError:
                pass
        raise
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Window-split run failed: {exc}", file=sys.stderr)
        raise
