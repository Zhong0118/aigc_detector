from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import joblib

from scripts.run_family_attribution_baselines import main
from tests.text_attribution_test_utils import write_dataset, write_model_catalog


def write_config(
    path: Path,
    *,
    catalog: str = "catalog.json",
    experiments: list[str] | None = None,
    seed: int = 42,
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
                    "train_ratio": 0.7,
                    "validation_ratio": 0.15,
                    "test_ratio": 0.15,
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
            self.assertEqual(summary["seed"], 99)
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
