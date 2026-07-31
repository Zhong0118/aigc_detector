"""Run one leakage-safe neural family-attribution experiment."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
import platform
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence
from uuid import uuid4

from research.text_attribution.neural_config import load_neural_screening_config
from research.text_attribution.split_manifest import (
    apply_split_manifest,
    load_split_manifest,
)
from research.text_attribution.training_data import load_training_dataset


DEFAULT_CONFIG_PATH = (
    "configs/text_attribution/training/family_neural_screening.v1.json"
)
EXPERIMENT_IDS = ("N0", "N1", "N2")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one frozen-split neural family-attribution experiment."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--experiment", choices=EXPERIMENT_IDS, required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--catalog")
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hf-cache")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _build_training_request(
    request_type: type[Any],
    *,
    experiment: Any,
    train_samples: tuple[Any, ...],
    validation_samples: tuple[Any, ...],
    family_labels: tuple[str, ...],
    output_dir: Path,
    seed: int,
    hf_cache: Path | None,
    bf16: bool,
    smoke_test: bool,
    resume_from_checkpoint: Path | None,
) -> Any:
    return request_type(
        experiment=experiment,
        train_samples=train_samples,
        validation_samples=validation_samples,
        family_labels=family_labels,
        output_dir=output_dir,
        seed=seed,
        hf_cache=hf_cache,
        bf16=bf16,
        smoke_test=smoke_test,
        resume_from_checkpoint=resume_from_checkpoint,
    )


def _effective_runtime(
    requested_device: str,
    *,
    smoke_test: bool,
    configured_bf16: bool,
) -> tuple[str, bool, int, bool]:
    if requested_device not in {"cpu", "cuda:0"}:
        raise ValueError(
            "device must be 'cpu' or 'cuda:0'; select a physical GPU with "
            "CUDA_VISIBLE_DEVICES before launching the process"
        )
    if smoke_test:
        return "cpu", False, 2, True
    if requested_device == "cpu":
        return "cpu", False, -1, True
    return "cuda:0", configured_bf16, -1, False


def _dependency_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _preserve_failed_run(staging: Path, output_dir: Path, error: BaseException) -> Path:
    failed = output_dir.with_name(f".{output_dir.name}.failed-{uuid4().hex}")
    evidence_path = staging
    try:
        os.replace(staging, failed)
        evidence_path = failed
        try:
            (failed / "FAILED.json").write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass
    except OSError:
        pass
    message = f"Failed run artifacts preserved at: {evidence_path}"
    if hasattr(error, "add_note"):
        error.add_note(message)
    print(message, file=sys.stderr)
    return evidence_path


def _predict_saved_model(
    *,
    model_dir: Path,
    tokenizer_dir: Path,
    samples: tuple[Any, ...],
    family_labels: tuple[str, ...],
    max_length: int,
    device: str,
    hf_cache: Path | None,
) -> Any:
    """Reload final saved artifacts and predict an explicitly requested test split."""

    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        cache_dir=hf_cache,
        local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        cache_dir=hf_cache,
        local_files_only=True,
    )
    expected_id2label = dict(enumerate(family_labels))
    if model.config.id2label != expected_id2label:
        raise ValueError("saved model id2label must exactly match family_labels order")
    target = torch.device(device)
    model.to(target)
    model.eval()
    rows: list[np.ndarray] = []
    batch_size = 16
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            encoded = tokenizer(
                [sample.text for sample in batch],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(target) for key, value in encoded.items()}
            logits = model(**encoded).logits
            rows.append(torch.softmax(logits, dim=-1).detach().cpu().numpy())
    return np.concatenate(rows, axis=0)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.smoke_test and args.evaluate_test:
        raise ValueError("smoke-test cannot evaluate the test split")

    config_path = Path(args.config)
    config = load_neural_screening_config(config_path)
    experiment = config.experiments[args.experiment]
    catalog_path = (
        Path(args.catalog).resolve()
        if args.catalog is not None
        else config.model_catalog
    )
    seed = config.seed if args.seed is None else args.seed
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    dataset = load_training_dataset(args.input, catalog_path)
    split_path = Path(args.split_manifest)
    if not split_path.is_file():
        raise FileNotFoundError(
            f"Existing frozen split manifest is required: {split_path}"
        )
    manifest = load_split_manifest(split_path)
    splits = apply_split_manifest(dataset.samples, manifest)
    configured_bf16 = experiment.bf16 and not args.no_bf16
    effective_device, effective_bf16, max_steps, use_cpu = _effective_runtime(
        args.device,
        smoke_test=args.smoke_test,
        configured_bf16=configured_bf16,
    )
    family_to_id = {family: index for index, family in enumerate(dataset.family_labels)}
    id_to_family = {str(index): family for family, index in family_to_id.items()}

    summary: dict[str, Any] = {
        "schema_version": config.schema_version,
        "run_id": config.run_id,
        "input": str(args.input),
        "clean_samples": len(dataset.samples),
        "families": list(dataset.family_labels),
        "models": list(dataset.model_labels),
        "splits": {name: len(rows) for name, rows in splits.items()},
        "split_manifest": str(split_path),
        "split_dataset_fingerprint": manifest.dataset_fingerprint,
        "split_seed": manifest.seed,
        "training_seed": seed,
        "experiment": args.experiment,
        "model": experiment.model_id,
        "model_name": experiment.model_id,
        "revision": getattr(experiment, "revision", None),
        "max_length": experiment.max_length,
        "family_to_id": family_to_id,
        "id_to_family": id_to_family,
        "training": {
            "max_length": experiment.max_length,
            "per_device_train_batch_size": (experiment.per_device_train_batch_size),
            "learning_rate": experiment.learning_rate,
            "epochs": experiment.epochs,
            "weight_decay": experiment.weight_decay,
            "warmup_ratio": experiment.warmup_ratio,
            "label_smoothing_factor": experiment.label_smoothing_factor,
            "gradient_accumulation_steps": (experiment.gradient_accumulation_steps),
            "early_stopping_patience": experiment.early_stopping_patience,
            "save_total_limit": experiment.save_total_limit,
            "bf16": effective_bf16,
            "max_steps": max_steps,
            "use_cpu": use_cpu,
        },
        "config": str(config.source_path),
        "catalog": str(catalog_path),
        "smoke_test": args.smoke_test,
        "evaluate_test": args.evaluate_test,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "requested_device": args.device,
        "device": effective_device,
        "hf_cache": args.hf_cache,
        "bf16": effective_bf16,
        "versions": {
            "python": platform.python_version(),
            "torch": _dependency_version("torch"),
            "transformers": _dependency_version("transformers"),
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0

    from research.text_attribution.attribution_metrics import (
        build_prediction_records,
        evaluate_family_predictions,
    )
    from research.text_attribution.attribution_reporting import (
        write_attribution_report,
    )

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output-dir already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.run-",
            dir=output_dir.parent,
        )
    )
    try:
        train_samples = splits["train"]
        validation_samples = splits["validation"]
        if args.smoke_test:
            from research.text_attribution.neural_data import select_smoke_samples

            train_samples = select_smoke_samples(
                train_samples,
                per_model=2,
                seed=seed,
            )
            validation_samples = select_smoke_samples(
                validation_samples,
                per_model=1,
                seed=seed,
            )

        from research.text_attribution.neural_training import (
            NeuralTrainingRequest,
            train_neural_experiment,
        )

        request = _build_training_request(
            NeuralTrainingRequest,
            experiment=experiment,
            train_samples=train_samples,
            validation_samples=validation_samples,
            family_labels=dataset.family_labels,
            output_dir=staging,
            seed=seed,
            hf_cache=None if args.hf_cache is None else Path(args.hf_cache),
            bf16=effective_bf16,
            smoke_test=args.smoke_test,
            resume_from_checkpoint=(
                None
                if args.resume_from_checkpoint is None
                else Path(args.resume_from_checkpoint)
            ),
        )
        result = train_neural_experiment(request)
        validation_records = build_prediction_records(
            validation_samples,
            result.validation_probabilities,
            family_labels=dataset.family_labels,
            experiment_id=args.experiment,
            split="validation",
        )
        records = list(validation_records)
        metrics: dict[str, Any] = {
            args.experiment: {
                "validation": evaluate_family_predictions(
                    validation_records,
                    family_labels=dataset.family_labels,
                )
            },
            "selection": {
                "selected_experiment": args.experiment,
                "criterion": "single_neural_experiment_validation_only",
            },
        }

        if args.evaluate_test:
            test_samples = splits["test"]
            test_probabilities = _predict_saved_model(
                model_dir=result.model_dir,
                tokenizer_dir=result.tokenizer_dir,
                samples=test_samples,
                family_labels=dataset.family_labels,
                max_length=experiment.max_length,
                device=effective_device,
                hf_cache=None if args.hf_cache is None else Path(args.hf_cache),
            )
            test_records = build_prediction_records(
                test_samples,
                test_probabilities,
                family_labels=dataset.family_labels,
                experiment_id=args.experiment,
                split="test",
            )
            records.extend(test_records)
            metrics[args.experiment]["test"] = evaluate_family_predictions(
                test_records,
                family_labels=dataset.family_labels,
            )

        report_work = staging / "report_work"
        report_work.mkdir()
        report_paths = write_attribution_report(
            output_dir=report_work,
            run_config=summary,
            dataset_audit=dataset.audit,
            split_summary=summary["splits"],
            metrics=metrics,
            predictions=records,
        )
        for report_path in report_paths.values():
            source = Path(report_path)
            destination = staging / source.name
            if destination.exists():
                raise RuntimeError(
                    f"report artifact conflicts with training artifact: {destination}"
                )
            os.replace(source, destination)
        report_work.rmdir()
        os.replace(staging, output_dir)
    except BaseException as error:
        if staging.exists():
            _preserve_failed_run(staging, output_dir, error)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
