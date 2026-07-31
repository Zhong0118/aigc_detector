from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass
import io
import json
from pathlib import Path
from types import ModuleType
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from research.text_attribution.split_manifest import (
    create_split_manifest,
    save_split_manifest,
)
from research.text_attribution.training_data import load_training_dataset
from tests.text_attribution_test_utils import write_dataset, write_model_catalog


def write_config(path: Path, *, catalog: str = "catalog.json", seed: int = 42) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "test-neural-attribution",
                "seed": seed,
                "model_catalog": catalog,
                "common": {
                    "learning_rate": 2e-5,
                    "epochs": 3,
                    "weight_decay": 0.01,
                    "warmup_ratio": 0.1,
                    "label_smoothing_factor": 0.05,
                    "gradient_accumulation_steps": 2,
                    "early_stopping_patience": 2,
                    "save_total_limit": 1,
                    "bf16": True,
                },
                "experiments": {
                    "N0": {
                        "model_id": "test/n0",
                        "max_length": 128,
                        "per_device_train_batch_size": 8,
                    },
                    "N1": {
                        "model_id": "test/n1",
                        "max_length": 256,
                        "per_device_train_batch_size": 4,
                    },
                    "N2": {
                        "model_id": "test/n2",
                        "max_length": 512,
                        "per_device_train_batch_size": 2,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def prepare_inputs(root: Path, *, groups: int = 20) -> tuple[Path, Path, Path]:
    config_path = write_config(root / "config" / "run.json")
    catalog_path = config_path.parent / "catalog.json"
    write_model_catalog(catalog_path)
    data_path = root / "data.jsonl"
    write_dataset(data_path, groups=groups)
    dataset = load_training_dataset(data_path, catalog_path)
    split_path = root / "split.json"
    save_split_manifest(
        split_path,
        create_split_manifest(dataset.samples, seed=42),
    )
    return config_path, data_path, split_path


def common_args(
    root: Path,
    config_path: Path,
    data_path: Path,
    split_path: Path,
) -> list[str]:
    return [
        "--config",
        str(config_path),
        "--experiment",
        "N0",
        "--input",
        str(data_path),
        "--split-manifest",
        str(split_path),
        "--output-dir",
        str(root / "output"),
    ]


@dataclass
class FakeTrainingResult:
    model_dir: Path
    tokenizer_dir: Path
    validation_logits: np.ndarray
    validation_probabilities: np.ndarray
    trainer_history: tuple[dict[str, object], ...]
    best_checkpoint: str | None


class FakeTrainingRequest:
    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)


def fake_training_module(captured: list[FakeTrainingRequest]) -> ModuleType:
    module = ModuleType("research.text_attribution.neural_training")
    module.NeuralTrainingRequest = FakeTrainingRequest

    def train(request: FakeTrainingRequest) -> FakeTrainingResult:
        captured.append(request)
        request.output_dir.mkdir(parents=True, exist_ok=True)
        model_dir = request.output_dir / "model"
        tokenizer_dir = request.output_dir / "tokenizer"
        model_dir.mkdir()
        tokenizer_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        (tokenizer_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        history = ({"epoch": 1.0, "eval_macro_f1": 1.0},)
        (request.output_dir / "trainer_history.json").write_text(
            json.dumps(list(history)),
            encoding="utf-8",
        )
        probabilities = np.asarray(
            [
                [1.0, 0.0] if sample.family_id == "claude" else [0.0, 1.0]
                for sample in request.validation_samples
            ],
            dtype=float,
        )
        return FakeTrainingResult(
            model_dir=model_dir,
            tokenizer_dir=tokenizer_dir,
            validation_logits=np.log(np.maximum(probabilities, 1e-12)),
            validation_probabilities=probabilities,
            trainer_history=history,
            best_checkpoint=None,
        )

    module.train_neural_experiment = train
    return module


class NeuralFamilyAttributionCliTests(unittest.TestCase):
    def test_dry_run_does_not_import_training_or_create_output(self) -> None:
        from scripts.run_family_attribution_neural import main

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, data_path, split_path = prepare_inputs(root)
            stdout = io.StringIO()
            with (
                patch.dict(
                    sys.modules,
                    {"research.text_attribution.neural_training": None},
                ),
                redirect_stdout(stdout),
            ):
                exit_code = main(
                    [
                        *common_args(root, config_path, data_path, split_path),
                        "--dry-run",
                    ]
                )

            summary = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(summary["experiment"], "N0")
            self.assertEqual(summary["model"], "test/n0")
            self.assertEqual(summary["max_length"], 128)
            self.assertEqual(
                Path(summary["catalog"]),
                (config_path.parent / "catalog.json").resolve(),
            )
            self.assertFalse((root / "output").exists())

    def test_fingerprint_mismatch_fails_before_training(self) -> None:
        from scripts.run_family_attribution_neural import main

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, data_path, split_path = prepare_inputs(root)
            write_dataset(data_path, groups=21)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                main(
                    [
                        *common_args(root, config_path, data_path, split_path),
                        "--dry-run",
                    ]
                )
            self.assertFalse((root / "output").exists())

    def test_smoke_test_rejects_test_evaluation(self) -> None:
        from scripts.run_family_attribution_neural import main

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, data_path, split_path = prepare_inputs(root)
            with self.assertRaisesRegex(ValueError, "smoke.*test"):
                main(
                    [
                        *common_args(root, config_path, data_path, split_path),
                        "--smoke-test",
                        "--evaluate-test",
                    ]
                )
            self.assertFalse((root / "output").exists())

    def test_fake_training_writes_validation_only_report_and_history(self) -> None:
        from scripts.run_family_attribution_neural import main

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, data_path, split_path = prepare_inputs(root)
            captured: list[FakeTrainingRequest] = []
            module = fake_training_module(captured)
            with patch.dict(
                sys.modules,
                {"research.text_attribution.neural_training": module},
            ):
                exit_code = main(common_args(root, config_path, data_path, split_path))

            output = root / "output"
            self.assertEqual(exit_code, 0)
            self.assertEqual(len(captured), 1)
            metrics = json.loads((output / "metrics.json").read_text())
            self.assertEqual(set(metrics["N0"]), {"validation"})
            self.assertEqual(
                metrics["selection"],
                {
                    "selected_experiment": "N0",
                    "criterion": "single_neural_experiment_validation_only",
                },
            )
            predictions = [
                json.loads(line)
                for line in (output / "predictions.jsonl").read_text().splitlines()
            ]
            self.assertEqual({row["split"] for row in predictions}, {"validation"})
            self.assertTrue(
                all(
                    sorted(row["family_probabilities"].values()) == [0.0, 1.0]
                    for row in predictions
                )
            )
            run_config = json.loads((output / "run_config.json").read_text())
            self.assertFalse(run_config["evaluate_test"])
            self.assertEqual(run_config["experiment"], "N0")
            self.assertEqual(
                json.loads((output / "trainer_history.json").read_text()),
                [{"epoch": 1.0, "eval_macro_f1": 1.0}],
            )
            self.assertTrue((output / "overall_metrics.png").is_file())

    def test_existing_output_is_refused_without_mutation(self) -> None:
        from scripts.run_family_attribution_neural import main

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, data_path, split_path = prepare_inputs(root)
            output = root / "output"
            output.mkdir()
            sentinel = output / "sentinel.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "output"):
                main(common_args(root, config_path, data_path, split_path))
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_training_failure_never_publishes_output(self) -> None:
        from scripts.run_family_attribution_neural import main

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, data_path, split_path = prepare_inputs(root)
            module = ModuleType("research.text_attribution.neural_training")
            module.NeuralTrainingRequest = FakeTrainingRequest

            def fail_training(_request: FakeTrainingRequest) -> None:
                raise RuntimeError("injected training failure")

            module.train_neural_experiment = fail_training
            with (
                patch.dict(
                    sys.modules,
                    {"research.text_attribution.neural_training": module},
                ),
                self.assertRaisesRegex(RuntimeError, "training failure"),
            ):
                main(common_args(root, config_path, data_path, split_path))
            self.assertFalse((root / "output").exists())
            self.assertEqual(list(root.glob(".output.run-*")), [])

    def test_cli_overrides_are_passed_to_training_request(self) -> None:
        from scripts.run_family_attribution_neural import main

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, data_path, split_path = prepare_inputs(root)
            override_catalog = root / "override.json"
            write_model_catalog(override_catalog)
            resume = root / "checkpoint"
            resume.mkdir()
            (resume / "trainer_state.json").write_text("{}", encoding="utf-8")
            cache = root / "hf-cache"
            captured: list[FakeTrainingRequest] = []
            module = fake_training_module(captured)
            args = common_args(root, config_path, data_path, split_path)
            args[args.index("N0")] = "N1"
            args.extend(
                [
                    "--catalog",
                    str(override_catalog),
                    "--seed",
                    "99",
                    "--device",
                    "cpu",
                    "--hf-cache",
                    str(cache),
                    "--no-bf16",
                    "--resume-from-checkpoint",
                    str(resume),
                ]
            )
            with patch.dict(
                sys.modules,
                {"research.text_attribution.neural_training": module},
            ):
                main(args)

            request = captured[0]
            self.assertEqual(request.experiment.model_id, "test/n1")
            self.assertEqual(request.seed, 99)
            self.assertEqual(request.device, "cpu")
            self.assertEqual(request.hf_cache, cache)
            self.assertFalse(request.bf16)
            self.assertEqual(request.resume_from_checkpoint, resume)


if __name__ == "__main__":
    unittest.main()
