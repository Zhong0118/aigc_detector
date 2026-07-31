from __future__ import annotations

from contextlib import redirect_stderr
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import subprocess
from types import ModuleType, SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from research.text_attribution.neural_training import NeuralTrainingRequest
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


def fake_training_module(
    captured: list[NeuralTrainingRequest],
    captured_cuda_visible_devices: list[str | None] | None = None,
) -> ModuleType:
    module = ModuleType("research.text_attribution.neural_training")
    module.NeuralTrainingRequest = NeuralTrainingRequest

    def train(request: NeuralTrainingRequest) -> FakeTrainingResult:
        captured.append(request)
        if captured_cuda_visible_devices is not None:
            captured_cuda_visible_devices.append(os.environ.get("CUDA_VISIBLE_DEVICES"))
        request.output_dir.mkdir(parents=True, exist_ok=True)
        model_dir = request.output_dir / "model"
        tokenizer_dir = request.output_dir / "tokenizer"
        model_dir.mkdir()
        tokenizer_dir.mkdir()
        checkpoint_dir = request.output_dir / "checkpoints"
        checkpoint_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        (model_dir / "large-model-marker.bin").write_bytes(b"model-only")
        (tokenizer_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (checkpoint_dir / "checkpoint-marker.bin").write_bytes(b"checkpoint-only")
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
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, data_path, split_path = prepare_inputs(root)
            argv = [
                *common_args(root, config_path, data_path, split_path),
                "--device",
                "cpu",
                "--dry-run",
            ]
            program = f"""
import builtins
import json
import sys
real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    blocked = ("numpy", "sklearn", "matplotlib", "seaborn", "transformers")
    if name.split(".", 1)[0] in blocked:
        raise AssertionError(f"dry-run imported blocked dependency: {{name}}")
    if name == "research.text_attribution.neural_training":
        raise AssertionError("dry-run imported neural_training")
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from scripts.run_family_attribution_neural import main
raise SystemExit(main({argv!r}))
"""
            completed = subprocess.run(
                [sys.executable, "-c", program],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads(completed.stdout)
            self.assertEqual(summary["experiment"], "N0")
            self.assertEqual(summary["device"], "cpu")
            self.assertEqual(summary["model_name"], "test/n0")
            self.assertIsNone(summary["revision"])
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
        from research.text_attribution import attribution_reporting
        from scripts.run_family_attribution_neural import main

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, data_path, split_path = prepare_inputs(root)
            captured: list[NeuralTrainingRequest] = []
            module = fake_training_module(captured)
            real_writer = attribution_reporting.write_attribution_report

            def assert_empty_report_workspace(**kwargs: object) -> dict[str, str]:
                report_root = Path(kwargs["output_dir"])
                self.assertTrue(report_root.is_dir())
                self.assertEqual(list(report_root.iterdir()), [])
                return real_writer(**kwargs)

            with (
                patch.dict(
                    sys.modules,
                    {"research.text_attribution.neural_training": module},
                ),
                patch.object(
                    attribution_reporting,
                    "write_attribution_report",
                    side_effect=assert_empty_report_workspace,
                ) as writer,
            ):
                exit_code = main(common_args(root, config_path, data_path, split_path))

            output = root / "output"
            self.assertEqual(exit_code, 0)
            writer.assert_called_once()
            self.assertEqual(len(captured), 1)
            self.assertTrue((output / "model" / "large-model-marker.bin").is_file())
            self.assertTrue(
                (output / "checkpoints" / "checkpoint-marker.bin").is_file()
            )
            self.assertFalse((output / "report_work").exists())
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
                run_config["family_to_id"],
                {"claude": 0, "openai": 1},
            )
            self.assertEqual(
                run_config["id_to_family"],
                {"0": "claude", "1": "openai"},
            )
            self.assertEqual(
                run_config["training"],
                {
                    "max_length": 128,
                    "per_device_train_batch_size": 8,
                    "learning_rate": 2e-5,
                    "epochs": 3,
                    "weight_decay": 0.01,
                    "warmup_ratio": 0.1,
                    "label_smoothing_factor": 0.05,
                    "gradient_accumulation_steps": 2,
                    "early_stopping_patience": 2,
                    "save_total_limit": 1,
                    "bf16": True,
                    "max_steps": -1,
                    "use_cpu": False,
                },
            )
            self.assertEqual(run_config["device"], "cuda:0")
            self.assertTrue(run_config["bf16"])
            self.assertEqual(
                set(run_config["versions"]),
                {"python", "torch", "transformers"},
            )
            self.assertTrue(all(run_config["versions"].values()))
            self.assertEqual(
                json.loads((output / "trainer_history.json").read_text()),
                [{"epoch": 1.0, "eval_macro_f1": 1.0}],
            )
            self.assertTrue((output / "overall_metrics.png").is_file())

    def test_smoke_run_records_effective_runtime_and_balanced_subsets(self) -> None:
        from scripts.run_family_attribution_neural import main

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, data_path, split_path = prepare_inputs(root)
            captured: list[NeuralTrainingRequest] = []
            module = fake_training_module(captured)
            with patch.dict(
                sys.modules,
                {"research.text_attribution.neural_training": module},
            ):
                main(
                    [
                        *common_args(root, config_path, data_path, split_path),
                        "--device",
                        "cpu",
                        "--smoke-test",
                    ]
                )

            request = captured[0]
            self.assertTrue(request.smoke_test)
            self.assertFalse(request.bf16)
            train_counts = {
                model: sum(sample.model_id == model for sample in request.train_samples)
                for model in {sample.model_id for sample in request.train_samples}
            }
            validation_counts = {
                model: sum(
                    sample.model_id == model for sample in request.validation_samples
                )
                for model in {sample.model_id for sample in request.validation_samples}
            }
            self.assertTrue(all(count <= 2 for count in train_counts.values()))
            self.assertTrue(all(count <= 1 for count in validation_counts.values()))
            run_config = json.loads((root / "output" / "run_config.json").read_text())
            self.assertEqual(run_config["requested_device"], "cpu")
            self.assertEqual(run_config["device"], "cpu")
            self.assertFalse(run_config["bf16"])
            self.assertFalse(run_config["training"]["bf16"])
            self.assertEqual(run_config["training"]["max_steps"], 2)
            self.assertTrue(run_config["training"]["use_cpu"])

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
            module.NeuralTrainingRequest = NeuralTrainingRequest

            def fail_training(request: NeuralTrainingRequest) -> None:
                checkpoint = request.output_dir / "checkpoints" / "checkpoint-1"
                checkpoint.mkdir(parents=True)
                (checkpoint / "trainer_state.json").write_text(
                    "{}",
                    encoding="utf-8",
                )
                (request.output_dir / "training.log").write_text(
                    "last useful log",
                    encoding="utf-8",
                )
                raise RuntimeError("injected training failure")

            module.train_neural_experiment = fail_training
            stderr = io.StringIO()
            with (
                patch.dict(
                    sys.modules,
                    {"research.text_attribution.neural_training": module},
                ),
                redirect_stderr(stderr),
            ):
                with self.assertRaisesRegex(RuntimeError, "training failure"):
                    main(common_args(root, config_path, data_path, split_path))
            self.assertFalse((root / "output").exists())
            failed = list(root.glob(".output.failed-*"))
            self.assertEqual(len(failed), 1)
            self.assertTrue(
                (
                    failed[0] / "checkpoints" / "checkpoint-1" / "trainer_state.json"
                ).is_file()
            )
            self.assertEqual(
                (failed[0] / "training.log").read_text(),
                "last useful log",
            )
            self.assertIn(
                str(failed[0]),
                stderr.getvalue(),
            )
            self.assertEqual(list(root.glob(".output.run-*")), [])

    def test_cli_overrides_are_passed_without_mutating_cuda_environment(self) -> None:
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
            captured: list[NeuralTrainingRequest] = []
            captured_cuda_visible_devices: list[str | None] = []
            module = fake_training_module(
                captured,
                captured_cuda_visible_devices,
            )
            args = common_args(root, config_path, data_path, split_path)
            args[args.index("N0")] = "N1"
            args.extend(
                [
                    "--catalog",
                    str(override_catalog),
                    "--seed",
                    "99",
                    "--device",
                    "cuda:0",
                    "--hf-cache",
                    str(cache),
                    "--no-bf16",
                    "--resume-from-checkpoint",
                    str(resume),
                ]
            )
            with (
                patch.dict(
                    sys.modules,
                    {"research.text_attribution.neural_training": module},
                ),
                patch.dict(
                    os.environ,
                    {"CUDA_VISIBLE_DEVICES": "7"},
                    clear=False,
                ),
            ):
                main(args)
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "7")

            request = captured[0]
            self.assertEqual(request.experiment.model_id, "test/n1")
            self.assertEqual(request.seed, 99)
            self.assertFalse(hasattr(request, "device"))
            self.assertEqual(captured_cuda_visible_devices, ["7"])
            self.assertEqual(request.hf_cache, cache)
            self.assertFalse(request.bf16)
            self.assertEqual(request.resume_from_checkpoint, resume)
            run_config = json.loads((root / "output" / "run_config.json").read_text())
            self.assertEqual(run_config["requested_device"], "cuda:0")
            self.assertEqual(run_config["device"], "cuda:0")

    def test_formal_cpu_run_is_rejected_before_trainer_call(self) -> None:
        from scripts.run_family_attribution_neural import main

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, data_path, split_path = prepare_inputs(root)
            captured: list[NeuralTrainingRequest] = []
            module = fake_training_module(captured)
            with (
                patch.dict(
                    sys.modules,
                    {"research.text_attribution.neural_training": module},
                ),
                self.assertRaisesRegex(
                    ValueError,
                    "formal neural training.*cuda:0.*smoke-test",
                ),
            ):
                main(
                    [
                        *common_args(root, config_path, data_path, split_path),
                        "--device",
                        "cpu",
                    ]
                )

            self.assertEqual(captured, [])
            self.assertFalse((root / "output").exists())
            self.assertEqual(list(root.glob(".output.run-*")), [])

    def test_rejects_nonzero_cuda_device_without_training(self) -> None:
        from scripts.run_family_attribution_neural import main

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, data_path, split_path = prepare_inputs(root)
            captured: list[NeuralTrainingRequest] = []
            module = fake_training_module(captured)
            for invalid_device in ("cuda", "cuda:1", "cuda:7"):
                with self.subTest(device=invalid_device):
                    with (
                        patch.dict(
                            sys.modules,
                            {"research.text_attribution.neural_training": module},
                        ),
                        self.assertRaisesRegex(
                            ValueError,
                            "cpu.*cuda:0.*CUDA_VISIBLE_DEVICES",
                        ),
                    ):
                        main(
                            [
                                *common_args(
                                    root,
                                    config_path,
                                    data_path,
                                    split_path,
                                ),
                                "--device",
                                invalid_device,
                            ]
                        )
            self.assertEqual(captured, [])
            self.assertFalse((root / "output").exists())

    def test_explicit_test_evaluation_adds_test_metrics_and_predictions(self) -> None:
        from scripts import run_family_attribution_neural as cli

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, data_path, split_path = prepare_inputs(root)
            captured: list[NeuralTrainingRequest] = []
            module = fake_training_module(captured)

            def predict_test(**kwargs: object) -> np.ndarray:
                samples = kwargs["samples"]
                return np.asarray(
                    [
                        [1.0, 0.0] if sample.family_id == "claude" else [0.0, 1.0]
                        for sample in samples
                    ]
                )

            with (
                patch.dict(
                    sys.modules,
                    {"research.text_attribution.neural_training": module},
                ),
                patch.object(
                    cli,
                    "_predict_saved_model",
                    side_effect=predict_test,
                ) as predictor,
            ):
                cli.main(
                    [
                        *common_args(root, config_path, data_path, split_path),
                        "--evaluate-test",
                    ]
                )

            output = root / "output"
            metrics = json.loads((output / "metrics.json").read_text())
            self.assertEqual(set(metrics["N0"]), {"validation", "test"})
            predictions = [
                json.loads(line)
                for line in (output / "predictions.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                {row["split"] for row in predictions},
                {"validation", "test"},
            )
            predictor.assert_called_once()
            call = predictor.call_args.kwargs
            self.assertEqual(call["model_dir"], captured[0].output_dir / "model")
            self.assertEqual(
                call["tokenizer_dir"],
                captured[0].output_dir / "tokenizer",
            )

    def test_saved_model_prediction_preserves_ten_family_label_order(self) -> None:
        import torch
        from transformers import (
            BertConfig,
            BertForSequenceClassification,
            BertTokenizerFast,
        )

        from scripts import run_family_attribution_neural as cli

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer_dir = root / "tokenizer"
            model_dir = root / "model"
            vocab_path = root / "vocab.txt"
            vocab_path.write_text(
                "\n".join(
                    [
                        "[PAD]",
                        "[UNK]",
                        "[CLS]",
                        "[SEP]",
                        "[MASK]",
                        "first",
                        "second",
                        "sample",
                        ".",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            tokenizer = BertTokenizerFast(
                vocab_file=str(vocab_path),
                do_lower_case=True,
            )
            tokenizer.save_pretrained(tokenizer_dir)
            family_labels = tuple(f"family-{index}" for index in range(10))
            torch.manual_seed(7)
            model = BertForSequenceClassification(
                BertConfig(
                    vocab_size=len(tokenizer),
                    hidden_size=16,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    intermediate_size=24,
                    max_position_embeddings=64,
                    num_labels=len(family_labels),
                    id2label=dict(enumerate(family_labels)),
                    label2id={
                        label: index for index, label in enumerate(family_labels)
                    },
                )
            )
            model.save_pretrained(model_dir)
            samples = (
                SimpleNamespace(text="first sample."),
                SimpleNamespace(text="second sample."),
            )

            probabilities = cli._predict_saved_model(
                model_dir=model_dir,
                tokenizer_dir=tokenizer_dir,
                samples=samples,
                family_labels=family_labels,
                max_length=32,
                device="cpu",
                hf_cache=None,
            )

            self.assertEqual(probabilities.shape, (2, 10))
            np.testing.assert_allclose(
                probabilities.sum(axis=1),
                np.ones(2),
                rtol=0.0,
                atol=1e-6,
            )
            config_path = model_dir / "config.json"
            config_payload = json.loads(config_path.read_text(encoding="utf-8"))
            config_payload["id2label"] = {
                str(index): family_labels[9 - index] for index in range(10)
            }
            config_path.write_text(
                json.dumps(config_payload),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "id2label.*family_labels",
            ):
                cli._predict_saved_model(
                    model_dir=model_dir,
                    tokenizer_dir=tokenizer_dir,
                    samples=samples,
                    family_labels=family_labels,
                    max_length=32,
                    device="cpu",
                    hf_cache=None,
                )


if __name__ == "__main__":
    unittest.main()
