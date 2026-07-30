from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib

from research.text_attribution.training_data import load_training_dataset
from scripts.run_family_attribution_baselines import (
    _create_or_load_split_manifest,
    main,
)
from tests.text_attribution_test_utils import write_dataset, write_model_catalog


def write_config(
    path: Path,
    *,
    catalog: str = "catalog.json",
    experiments: list[str] | None = None,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "test-family-attribution",
                "experiments": experiments or ["B0"],
                "seed": seed,
                "split": {
                    "train_ratio": ratios[0],
                    "validation_ratio": ratios[1],
                    "test_ratio": ratios[2],
                },
                "model_catalog": catalog,
            }
        ),
        encoding="utf-8",
    )
    return path


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class FamilyAttributionBaselineCliTests(unittest.TestCase):
    def test_dry_run_resolves_config_relative_catalog_and_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_dir = root / "nested" / "config"
            config_path = write_config(config_dir / "run.json")
            write_model_catalog(config_dir / "catalog.json")
            data_path = root / "data.jsonl"
            write_dataset(data_path, groups=20)
            split_path = root / "split.json"
            output_dir = root / "output"
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--config",
                        str(config_path),
                        "--input",
                        str(data_path),
                        "--split-manifest",
                        str(split_path),
                        "--create-split",
                        "--output-dir",
                        str(output_dir),
                        "--dry-run",
                    ]
                )

            summary = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(
                Path(summary["catalog"]),
                (config_dir / "catalog.json").resolve(),
            )
            self.assertFalse(split_path.exists())
            self.assertFalse(output_dir.exists())

    def test_cli_overrides_config_experiments_catalog_and_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = write_config(
                root / "config" / "run.json",
                catalog="missing.json",
                experiments=["B1"],
                seed=7,
            )
            catalog_path = root / "override-catalog.json"
            write_model_catalog(catalog_path)
            data_path = root / "data.jsonl"
            write_dataset(data_path, groups=20)
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--config",
                        str(config_path),
                        "--input",
                        str(data_path),
                        "--catalog",
                        str(catalog_path),
                        "--split-manifest",
                        str(root / "split.json"),
                        "--create-split",
                        "--experiments",
                        "B0",
                        "--seed",
                        "99",
                        "--output-dir",
                        str(root / "output"),
                        "--dry-run",
                    ]
                )

            summary = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(summary["experiments"], ["B0"])
            self.assertEqual(summary["training_seed"], 99)
            self.assertEqual(Path(summary["catalog"]), catalog_path)

    def test_default_run_is_validation_only_and_model_is_reusable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = write_config(root / "run.json")
            write_model_catalog(root / "catalog.json")
            data_path = root / "data.jsonl"
            write_dataset(data_path, groups=20)
            output_dir = root / "output"
            split_path = root / "split.json"

            exit_code = main(
                [
                    "--config",
                    str(config_path),
                    "--input",
                    str(data_path),
                    "--split-manifest",
                    str(split_path),
                    "--create-split",
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(split_path.is_file())
            model_path = output_dir / "models" / "B0.joblib"
            self.assertTrue(model_path.is_file())
            metrics = json.loads((output_dir / "metrics.json").read_text())
            self.assertEqual(set(metrics["B0"]), {"validation"})
            self.assertEqual(metrics["selection"]["selected_experiment"], "B0")
            predictions = read_jsonl(output_dir / "predictions.jsonl")
            self.assertEqual({row["split"] for row in predictions}, {"validation"})
            model = joblib.load(model_path)
            probabilities = model.predict_proba(["A complete held-out example."])
            self.assertEqual(probabilities.shape, (1, 2))

    def test_evaluates_test_only_for_validation_selected_winner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = write_config(
                root / "run.json", experiments=["B0", "B1"]
            )
            write_model_catalog(root / "catalog.json")
            data_path = root / "data.jsonl"
            write_dataset(data_path, groups=20)
            output_dir = root / "output"

            exit_code = main(
                [
                    "--config",
                    str(config_path),
                    "--input",
                    str(data_path),
                    "--split-manifest",
                    str(root / "split.json"),
                    "--create-split",
                    "--output-dir",
                    str(output_dir),
                    "--evaluate-selected-test",
                ]
            )

            self.assertEqual(exit_code, 0)
            metrics = json.loads((output_dir / "metrics.json").read_text())
            winner = metrics["selection"]["selected_experiment"]
            self.assertEqual(
                [
                    experiment
                    for experiment in ("B0", "B1")
                    if "test" in metrics[experiment]
                ],
                [winner],
            )
            predictions = read_jsonl(output_dir / "predictions.jsonl")
            test_experiments = {
                row["experiment_id"]
                for row in predictions
                if row["split"] == "test"
            }
            self.assertEqual(test_experiments, {winner})
            self.assertEqual(metrics["selection"]["selection_split"], "validation")

    def test_reuses_existing_frozen_split_and_rejects_missing_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = write_config(root / "run.json")
            write_model_catalog(root / "catalog.json")
            data_path = root / "data.jsonl"
            write_dataset(data_path, groups=20)
            split_path = root / "split.json"

            with self.assertRaisesRegex(FileNotFoundError, "Frozen split manifest"):
                main(
                    [
                        "--config",
                        str(config_path),
                        "--input",
                        str(data_path),
                        "--split-manifest",
                        str(split_path),
                        "--output-dir",
                        str(root / "output"),
                        "--dry-run",
                    ]
                )

            main(
                [
                    "--config",
                    str(config_path),
                    "--input",
                    str(data_path),
                    "--split-manifest",
                    str(split_path),
                    "--create-split",
                    "--output-dir",
                    str(root / "first"),
                    "--dry-run",
                ]
            )
            self.assertFalse(split_path.exists())

            main(
                [
                    "--config",
                    str(config_path),
                    "--input",
                    str(data_path),
                    "--split-manifest",
                    str(split_path),
                    "--create-split",
                    "--output-dir",
                    str(root / "created"),
                ]
            )
            original_manifest = split_path.read_bytes()
            main(
                [
                    "--config",
                    str(config_path),
                    "--input",
                    str(data_path),
                    "--split-manifest",
                    str(split_path),
                    "--output-dir",
                    str(root / "reused"),
                    "--dry-run",
                ]
            )
            self.assertEqual(split_path.read_bytes(), original_manifest)

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                main(
                    [
                        "--config",
                        str(config_path),
                        "--input",
                        str(data_path),
                        "--split-manifest",
                        str(split_path),
                        "--seed",
                        "99",
                        "--output-dir",
                        str(root / "metadata"),
                        "--dry-run",
                    ]
                )
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["training_seed"], 99)
            self.assertEqual(summary["split_seed"], 42)
            self.assertEqual(
                summary["split_ratios"],
                {"train": 0.7, "validation": 0.15, "test": 0.15},
            )

    def test_create_split_rejects_requested_seed_and_ratio_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = write_config(root / "run.json")
            write_model_catalog(root / "catalog.json")
            data_path = root / "data.jsonl"
            write_dataset(data_path, groups=20)
            split_path = root / "split.json"
            main(
                [
                    "--config",
                    str(config_path),
                    "--input",
                    str(data_path),
                    "--split-manifest",
                    str(split_path),
                    "--create-split",
                    "--output-dir",
                    str(root / "created"),
                ]
            )

            with self.assertRaisesRegex(ValueError, "seed"):
                main(
                    [
                        "--config",
                        str(config_path),
                        "--input",
                        str(data_path),
                        "--split-manifest",
                        str(split_path),
                        "--create-split",
                        "--seed",
                        "7",
                        "--output-dir",
                        str(root / "seed-mismatch"),
                        "--dry-run",
                    ]
                )

            mismatch_config = write_config(
                root / "ratio-run.json", ratios=(0.6, 0.2, 0.2)
            )
            with self.assertRaisesRegex(ValueError, "ratio"):
                main(
                    [
                        "--config",
                        str(mismatch_config),
                        "--input",
                        str(data_path),
                        "--split-manifest",
                        str(split_path),
                        "--create-split",
                        "--output-dir",
                        str(root / "ratio-mismatch"),
                        "--dry-run",
                    ]
                )

    def test_effective_training_seed_must_be_in_sklearn_range(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_model_catalog(root / "catalog.json")
            data_path = root / "data.jsonl"
            write_dataset(data_path, groups=20)
            common = [
                "--input",
                str(data_path),
                "--split-manifest",
                str(root / "split.json"),
                "--create-split",
                "--output-dir",
                str(root / "output"),
                "--dry-run",
            ]

            negative_config = write_config(root / "negative.json", seed=-1)
            with self.assertRaisesRegex(ValueError, "seed"):
                main(["--config", str(negative_config), *common])

            valid_config = write_config(root / "valid.json")
            with self.assertRaisesRegex(ValueError, "seed"):
                main(
                    [
                        "--config",
                        str(valid_config),
                        *common,
                        "--seed",
                        str(2**32),
                    ]
                )

    def test_b3_dry_run_checks_three_fold_training_support(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = write_config(root / "run.json", experiments=["B3"])
            write_model_catalog(root / "catalog.json")
            insufficient_path = root / "insufficient.jsonl"
            sufficient_path = root / "sufficient.jsonl"
            write_dataset(insufficient_path, groups=3)
            write_dataset(sufficient_path, groups=6)

            with self.assertRaisesRegex(ValueError, "B3.*3"):
                main(
                    [
                        "--config",
                        str(config_path),
                        "--input",
                        str(insufficient_path),
                        "--split-manifest",
                        str(root / "insufficient-split.json"),
                        "--create-split",
                        "--output-dir",
                        str(root / "insufficient-output"),
                        "--dry-run",
                    ]
                )

            self.assertEqual(
                main(
                    [
                        "--config",
                        str(config_path),
                        "--input",
                        str(sufficient_path),
                        "--split-manifest",
                        str(root / "sufficient-split.json"),
                        "--create-split",
                        "--output-dir",
                        str(root / "sufficient-output"),
                        "--dry-run",
                    ]
                ),
                0,
            )

    def test_concurrent_split_creation_publishes_one_complete_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_model_catalog(root / "catalog.json")
            data_path = root / "data.jsonl"
            write_dataset(data_path, groups=20)
            dataset = load_training_dataset(data_path, root / "catalog.json")
            split_path = root / "split.json"
            ratios = {
                "train_ratio": 0.7,
                "validation_ratio": 0.15,
                "test_ratio": 0.15,
            }

            with ThreadPoolExecutor(max_workers=4) as executor:
                manifests = list(
                    executor.map(
                        lambda _: _create_or_load_split_manifest(
                            split_path,
                            dataset.samples,
                            seed=42,
                            split_config=ratios,
                            persist=True,
                        ),
                        range(4),
                    )
                )

            payload = json.loads(split_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["seed"], 42)
            self.assertTrue(payload["group_assignments"])
            self.assertEqual(
                {manifest.dataset_fingerprint for manifest in manifests},
                {payload["dataset_fingerprint"]},
            )
            self.assertEqual(
                list(root.glob(f".{split_path.name}.tmp-*")),
                [],
            )

    def test_subset_rerun_removes_stale_model_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = write_config(
                root / "run.json", experiments=["B0", "B1"]
            )
            write_model_catalog(root / "catalog.json")
            data_path = root / "data.jsonl"
            write_dataset(data_path, groups=20)
            split_path = root / "split.json"
            output_dir = root / "output"
            common = [
                "--config",
                str(config_path),
                "--input",
                str(data_path),
                "--split-manifest",
                str(split_path),
                "--output-dir",
                str(output_dir),
            ]
            main([*common, "--create-split"])
            self.assertTrue((output_dir / "models" / "B1.joblib").is_file())

            main([*common, "--experiments", "B0"])

            self.assertTrue((output_dir / "models" / "B0.joblib").is_file())
            self.assertFalse((output_dir / "models" / "B1.joblib").exists())

    def test_training_and_reporting_failure_leave_old_output_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = write_config(
                root / "run.json", experiments=["B0", "B1"]
            )
            write_model_catalog(root / "catalog.json")
            data_path = root / "data.jsonl"
            write_dataset(data_path, groups=20)
            split_path = root / "split.json"
            output_dir = root / "output"
            output_dir.mkdir()
            sentinel = output_dir / "sentinel.txt"
            sentinel.write_text("complete-old-run", encoding="utf-8")
            before = {path.name: path.read_bytes() for path in output_dir.iterdir()}
            common = [
                "--config",
                str(config_path),
                "--input",
                str(data_path),
                "--split-manifest",
                str(split_path),
                "--create-split",
                "--output-dir",
                str(output_dir),
            ]

            from scripts import run_family_attribution_baselines as cli

            real_builder = cli.build_classical_pipeline
            calls = 0

            def fail_second_build(experiment_id: str, seed: int):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("injected training failure")
                return real_builder(experiment_id, seed)

            with patch.object(
                cli, "build_classical_pipeline", side_effect=fail_second_build
            ):
                with self.assertRaisesRegex(RuntimeError, "training failure"):
                    main(common)
            self.assertEqual(
                {path.name: path.read_bytes() for path in output_dir.iterdir()},
                before,
            )

            with patch.object(
                cli,
                "write_attribution_report",
                side_effect=RuntimeError("injected report failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "report failure"):
                    main([*common, "--experiments", "B0"])
            self.assertEqual(
                {path.name: path.read_bytes() for path in output_dir.iterdir()},
                before,
            )
            self.assertEqual(list(root.glob(f".{output_dir.name}.run-*")), [])

    def test_publish_failure_rolls_back_old_output_and_cleans_backup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = write_config(root / "run.json")
            write_model_catalog(root / "catalog.json")
            data_path = root / "data.jsonl"
            write_dataset(data_path, groups=20)
            split_path = root / "split.json"
            output_dir = root / "output"
            output_dir.mkdir()
            sentinel = output_dir / "sentinel.txt"
            sentinel.write_text("complete-old-run", encoding="utf-8")

            from scripts import run_family_attribution_baselines as cli

            real_replace = os.replace
            calls = 0

            def fail_stage_publish(source: str | Path, target: str | Path) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected publish failure")
                real_replace(source, target)

            with patch.object(
                cli, "_replace_path", side_effect=fail_stage_publish, create=True
            ):
                with self.assertRaisesRegex(OSError, "publish failure"):
                    main(
                        [
                            "--config",
                            str(config_path),
                            "--input",
                            str(data_path),
                            "--split-manifest",
                            str(split_path),
                            "--create-split",
                            "--output-dir",
                            str(output_dir),
                        ]
                    )

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "complete-old-run")
            self.assertEqual(list(root.glob(f".{output_dir.name}.backup-*")), [])
            self.assertEqual(list(root.glob(f".{output_dir.name}.run-*")), [])

    def test_malformed_config_and_unknown_config_experiment_fail_early(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = write_config(root / "run.json", experiments=["B9"])
            write_model_catalog(root / "catalog.json")
            data_path = root / "data.jsonl"
            write_dataset(data_path, groups=20)

            with self.assertRaisesRegex(ValueError, "experiment"):
                main(
                    [
                        "--config",
                        str(config_path),
                        "--input",
                        str(data_path),
                        "--split-manifest",
                        str(root / "split.json"),
                        "--create-split",
                        "--output-dir",
                        str(root / "output"),
                        "--dry-run",
                    ]
                )

            config_path.write_text('{"schema_version": true}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "config"):
                main(
                    [
                        "--config",
                        str(config_path),
                        "--input",
                        str(data_path),
                        "--split-manifest",
                        str(root / "split.json"),
                        "--create-split",
                        "--output-dir",
                        str(root / "output"),
                        "--dry-run",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
