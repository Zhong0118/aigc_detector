from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


DATASET_FINGERPRINT = "a" * 64
FAMILY_TO_ID = {"alpha": 0, "beta": 1}


def metric_row(
    *,
    macro_f1: float = 0.8,
    alpha_f1: float = 0.7,
    beta_f1: float = 0.6,
    nll: float = 0.4,
) -> dict[str, object]:
    return {
        "overall": {
            "accuracy": 0.75,
            "macro_f1": macro_f1,
            "weighted_f1": 0.74,
            "support": 10,
        },
        "calibration": {
            "expected_calibration_error": 0.1,
            "multiclass_brier_score": 0.3,
            "negative_log_likelihood": nll,
        },
        "per_family": {
            "alpha": {
                "precision": 0.8,
                "recall": 0.7,
                "f1-score": alpha_f1,
                "support": 5,
            },
            "beta": {
                "precision": 0.7,
                "recall": 0.6,
                "f1-score": beta_f1,
                "support": 5,
            },
        },
        "confusion_matrix": [[0.8, 0.2], [0.3, 0.7]],
        "per_model": {
            "model-a": {
                "support": 4,
                "accuracy": 0.75,
                "mean_confidence": 0.8,
                "mean_true_family_probability": 0.7,
            },
            "model-b": {
                "support": 6,
                "accuracy": 0.75,
                "mean_confidence": 0.8,
                "mean_true_family_probability": 0.7,
            },
        },
        "subgroups": {
            "language": {
                "en": {
                    "accuracy": 0.75,
                    "macro_f1": macro_f1,
                    "support": 10,
                }
            }
        },
    }


def write_candidate(
    root: Path,
    experiment_id: str,
    *,
    macro_f1: float = 0.8,
    alpha_f1: float = 0.7,
    beta_f1: float = 0.6,
    nll: float = 0.4,
    smoke_test: bool = False,
    evaluate_test: bool = False,
    dataset_fingerprint: str = DATASET_FINGERPRINT,
    split_manifest: str = "frozen-split.json",
    split_summary: dict[str, int] | None = None,
    family_to_id: dict[str, int] | None = None,
    include_validation: bool = True,
    include_test: bool = False,
) -> Path:
    candidate = root / f"candidate-{experiment_id}"
    candidate.mkdir(parents=True)
    (candidate / "run_config.json").write_text(
        json.dumps(
            {
                "experiment": experiment_id,
                "smoke_test": smoke_test,
                "evaluate_test": evaluate_test,
                "split_manifest": split_manifest,
                "split_dataset_fingerprint": dataset_fingerprint,
                "splits": split_summary
                or {"train": 80, "validation": 10, "test": 10},
                "family_to_id": family_to_id or FAMILY_TO_ID,
            }
        ),
        encoding="utf-8",
    )
    splits: dict[str, object] = {}
    if include_validation:
        splits["validation"] = metric_row(
            macro_f1=macro_f1,
            alpha_f1=alpha_f1,
            beta_f1=beta_f1,
            nll=nll,
        )
    if include_test:
        splits["test"] = metric_row()
    (candidate / "metrics.json").write_text(
        json.dumps(
            {
                experiment_id: splits,
                "selection": {
                    "selected_experiment": experiment_id,
                    "criterion": "single_neural_experiment_validation_only",
                },
            }
        ),
        encoding="utf-8",
    )
    return candidate


class NeuralSelectionTests(unittest.TestCase):
    def test_ranking_uses_macro_then_worst_family_then_nll_then_id(self) -> None:
        from research.text_attribution.neural_selection import (
            select_neural_experiments,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            n2 = write_candidate(
                root,
                "N2",
                macro_f1=0.9,
                alpha_f1=0.7,
                beta_f1=0.65,
                nll=0.8,
            )
            n1 = write_candidate(
                root,
                "N1",
                macro_f1=0.9,
                alpha_f1=0.7,
                beta_f1=0.65,
                nll=0.3,
            )
            n0 = write_candidate(
                root,
                "N0",
                macro_f1=0.9,
                alpha_f1=0.7,
                beta_f1=0.65,
                nll=0.3,
            )

            summary = select_neural_experiments([n2, n1, n0])

            self.assertEqual(
                tuple(item.experiment_id for item in summary.ranking),
                ("N0", "N1", "N2"),
            )
            self.assertEqual(summary.top_two, ("N0", "N1"))
            self.assertEqual(summary.selection_scope, "validation_only")
            self.assertEqual(summary.ranking[0].macro_f1, 0.9)
            self.assertEqual(summary.ranking[0].worst_family_f1, 0.65)
            self.assertEqual(summary.ranking[0].negative_log_likelihood, 0.3)
            json.dumps(summary.to_dict(), allow_nan=False)

    def test_worst_family_breaks_a_macro_f1_tie_before_nll(self) -> None:
        from research.text_attribution.neural_selection import (
            select_neural_experiments,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weaker = write_candidate(
                root,
                "N0",
                macro_f1=0.8,
                beta_f1=0.2,
                nll=0.1,
            )
            stronger = write_candidate(
                root,
                "N1",
                macro_f1=0.8,
                beta_f1=0.5,
                nll=0.9,
            )

            summary = select_neural_experiments([weaker, stronger])

            self.assertEqual(summary.top_two, ("N1", "N0"))

    def test_input_order_does_not_change_the_summary(self) -> None:
        from research.text_attribution.neural_selection import (
            select_neural_experiments,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [
                write_candidate(root, "N0", macro_f1=0.4),
                write_candidate(root, "N1", macro_f1=0.8),
                write_candidate(root, "N2", macro_f1=0.6),
            ]

            forward = select_neural_experiments(paths)
            reverse = select_neural_experiments(tuple(reversed(paths)))

            self.assertEqual(forward, reverse)

    def test_rejects_empty_input(self) -> None:
        from research.text_attribution.neural_selection import (
            select_neural_experiments,
        )

        with self.assertRaisesRegex(ValueError, "at least one"):
            select_neural_experiments([])

    def test_rejects_smoke_test_and_test_contamination(self) -> None:
        from research.text_attribution.neural_selection import (
            select_neural_experiments,
        )

        cases = {
            "smoke": {"smoke_test": True},
            "evaluate-test": {"evaluate_test": True},
            "test metrics": {"include_test": True},
        }
        for label, overrides in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                candidate = write_candidate(
                    Path(temporary),
                    "N0",
                    **overrides,
                )
                with self.assertRaisesRegex(ValueError, "smoke|test"):
                    select_neural_experiments([candidate])

    def test_rejects_missing_validation_and_duplicate_experiment_ids(self) -> None:
        from research.text_attribution.neural_selection import (
            select_neural_experiments,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = write_candidate(root, "N0", include_validation=False)
            with self.assertRaisesRegex(ValueError, "validation"):
                select_neural_experiments([missing])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = write_candidate(root / "first", "N0")
            second = write_candidate(root / "second", "N0")
            with self.assertRaisesRegex(ValueError, "duplicate.*N0"):
                select_neural_experiments([first, second])

    def test_rejects_inconsistent_fingerprints_and_label_maps(self) -> None:
        from research.text_attribution.neural_selection import (
            select_neural_experiments,
        )

        cases = {
            "dataset fingerprint": {"dataset_fingerprint": "c" * 64},
            "split manifest": {"split_manifest": "different-split.json"},
            "split summary": {
                "split_summary": {"train": 79, "validation": 11, "test": 10}
            },
            "family label map": {"family_to_id": {"alpha": 1, "beta": 0}},
        }
        for label, overrides in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                reference = write_candidate(root / "reference", "N0")
                changed = write_candidate(root / "changed", "N1", **overrides)
                with self.assertRaisesRegex(ValueError, label):
                    select_neural_experiments([reference, changed])

    def test_rejects_nonfinite_out_of_range_and_incomplete_support(self) -> None:
        from research.text_attribution.neural_selection import (
            select_neural_experiments,
        )

        mutations = {
            "finite": lambda row: row["overall"].__setitem__(
                "macro_f1", float("nan")
            ),
            "within": lambda row: row["per_family"]["alpha"].__setitem__(
                "f1-score", 1.1
            ),
            "support": lambda row: row["per_family"]["alpha"].pop("support"),
        }
        for expected, mutate in mutations.items():
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as temporary:
                candidate = write_candidate(Path(temporary), "N0")
                metrics_path = candidate / "metrics.json"
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                mutate(payload["N0"]["validation"])
                metrics_path.write_text(
                    json.dumps(payload),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, expected):
                    select_neural_experiments([candidate])

    def test_rejects_nested_test_metrics_inside_validation(self) -> None:
        from research.text_attribution.neural_selection import (
            select_neural_experiments,
        )

        with tempfile.TemporaryDirectory() as temporary:
            candidate = write_candidate(Path(temporary), "N0")
            metrics_path = candidate / "metrics.json"
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            payload["N0"]["validation"]["test"] = {"macro_f1": 1.0}
            metrics_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "validation.*keys|test"):
                select_neural_experiments([candidate])

    def test_rejects_test_metrics_inside_selection_metadata(self) -> None:
        from research.text_attribution.neural_selection import (
            select_neural_experiments,
        )

        with tempfile.TemporaryDirectory() as temporary:
            candidate = write_candidate(Path(temporary), "N0")
            metrics_path = candidate / "metrics.json"
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            payload["selection"]["test_metrics"] = {"macro_f1": 1.0}
            metrics_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "selection.*keys|test"):
                select_neural_experiments([candidate])

    def test_rejects_any_casefolded_test_substring_in_nested_keys(self) -> None:
        from research.text_attribution.neural_selection import (
            select_neural_experiments,
        )

        for forbidden_key in (
            "holdout_test_metrics",
            "evaluation_test",
            "testMetrics",
        ):
            with (
                self.subTest(forbidden_key=forbidden_key),
                tempfile.TemporaryDirectory() as temporary,
            ):
                candidate = write_candidate(Path(temporary), "N0")
                metrics_path = candidate / "metrics.json"
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                payload["N0"]["validation"]["subgroups"]["language"]["en"][
                    forbidden_key
                ] = 0.9
                metrics_path.write_text(json.dumps(payload), encoding="utf-8")

                with self.assertRaisesRegex(ValueError, "forbidden test"):
                    select_neural_experiments([candidate])

    def test_allows_test_substrings_in_dynamic_metric_identifiers(self) -> None:
        from research.text_attribution.neural_selection import (
            select_neural_experiments,
        )

        with tempfile.TemporaryDirectory() as temporary:
            candidate = write_candidate(Path(temporary), "N0")
            metrics_path = candidate / "metrics.json"
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            validation = payload["N0"]["validation"]
            validation["per_model"] = {
                "claude-haiku-test": validation["per_model"]["model-a"],
                "gpt-fast-test": validation["per_model"]["model-b"],
            }
            validation["subgroups"] = {
                "test_language": {
                    "latest-test": validation["subgroups"]["language"]["en"]
                }
            }
            metrics_path.write_text(json.dumps(payload), encoding="utf-8")

            summary = select_neural_experiments([candidate])

            self.assertEqual(summary.top_two, ("N0",))

    def test_rejects_nonfinite_numbers_anywhere_in_validation(self) -> None:
        from research.text_attribution.neural_selection import (
            select_neural_experiments,
        )

        with tempfile.TemporaryDirectory() as temporary:
            candidate = write_candidate(Path(temporary), "N0")
            metrics_path = candidate / "metrics.json"
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            payload["N0"]["validation"]["confusion_matrix"][0][0] = float("nan")
            metrics_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "finite"):
                select_neural_experiments([candidate])

    def test_rejects_negative_or_boolean_nested_supports(self) -> None:
        from research.text_attribution.neural_selection import (
            select_neural_experiments,
        )

        cases = (
            ("per_model", -1),
            ("subgroups", -1),
            ("subgroups", True),
        )
        for location, invalid_support in cases:
            with (
                self.subTest(location=location, support=invalid_support),
                tempfile.TemporaryDirectory() as temporary,
            ):
                candidate = write_candidate(Path(temporary), "N0")
                metrics_path = candidate / "metrics.json"
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                validation = payload["N0"]["validation"]
                if location == "per_model":
                    validation["per_model"]["model-a"]["support"] = invalid_support
                else:
                    validation["subgroups"]["language"]["en"][
                        "support"
                    ] = invalid_support
                metrics_path.write_text(json.dumps(payload), encoding="utf-8")

                with self.assertRaisesRegex(ValueError, "support|boolean"):
                    select_neural_experiments([candidate])

    def test_rejects_invalid_confusion_matrix_shape_and_range(self) -> None:
        from research.text_attribution.neural_selection import (
            select_neural_experiments,
        )

        matrices = (
            [[1.0, 0.0]],
            [[1.1, -0.1], [0.3, 0.7]],
        )
        for matrix in matrices:
            with (
                self.subTest(matrix=matrix),
                tempfile.TemporaryDirectory() as temporary,
            ):
                candidate = write_candidate(Path(temporary), "N0")
                metrics_path = candidate / "metrics.json"
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                payload["N0"]["validation"]["confusion_matrix"] = matrix
                metrics_path.write_text(json.dumps(payload), encoding="utf-8")

                with self.assertRaisesRegex(
                    ValueError,
                    "confusion_matrix.*shape|confusion_matrix.*\\[0, 1\\]",
                ):
                    select_neural_experiments([candidate])

    def test_rejects_confusion_matrix_rows_not_summing_to_one(self) -> None:
        from research.text_attribution.neural_selection import (
            select_neural_experiments,
        )

        with tempfile.TemporaryDirectory() as temporary:
            candidate = write_candidate(Path(temporary), "N0")
            metrics_path = candidate / "metrics.json"
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            payload["N0"]["validation"]["confusion_matrix"] = [
                [0.4, 0.4],
                [0.3, 0.7],
            ]
            metrics_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "row.*sum"):
                select_neural_experiments([candidate])

    def test_rejects_per_model_support_total_mismatch(self) -> None:
        from research.text_attribution.neural_selection import (
            select_neural_experiments,
        )

        with tempfile.TemporaryDirectory() as temporary:
            candidate = write_candidate(Path(temporary), "N0")
            metrics_path = candidate / "metrics.json"
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            payload["N0"]["validation"]["per_model"]["model-a"]["support"] = 1000
            metrics_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "per_model.*support.*overall"):
                select_neural_experiments([candidate])

    def test_rejects_subgroup_dimension_support_total_mismatch(self) -> None:
        from research.text_attribution.neural_selection import (
            select_neural_experiments,
        )

        with tempfile.TemporaryDirectory() as temporary:
            candidate = write_candidate(Path(temporary), "N0")
            metrics_path = candidate / "metrics.json"
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            payload["N0"]["validation"]["subgroups"]["language"]["en"][
                "support"
            ] = 999
            metrics_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "subgroups.*support.*overall",
            ):
                select_neural_experiments([candidate])

    def test_rejects_empty_per_model_and_subgroup_mappings(self) -> None:
        from research.text_attribution.neural_selection import (
            select_neural_experiments,
        )

        mutations = (
            lambda validation: validation.__setitem__("per_model", {}),
            lambda validation: validation["subgroups"].__setitem__(
                "language", {}
            ),
        )
        for mutate in mutations:
            with tempfile.TemporaryDirectory() as temporary:
                candidate = write_candidate(Path(temporary), "N0")
                metrics_path = candidate / "metrics.json"
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                mutate(payload["N0"]["validation"])
                metrics_path.write_text(json.dumps(payload), encoding="utf-8")

                with self.assertRaisesRegex(ValueError, "must not be empty"):
                    select_neural_experiments([candidate])

    def test_rejects_missing_required_nested_metrics_and_calibration_keys(
        self,
    ) -> None:
        from research.text_attribution.neural_selection import (
            select_neural_experiments,
        )

        mutations = (
            lambda validation: validation["per_model"]["model-a"].pop(
                "mean_true_family_probability"
            ),
            lambda validation: validation["subgroups"]["language"]["en"].pop(
                "macro_f1"
            ),
            lambda validation: validation["calibration"].__setitem__(
                "unknown_metric", 0.1
            ),
        )
        for mutate in mutations:
            with tempfile.TemporaryDirectory() as temporary:
                candidate = write_candidate(Path(temporary), "N0")
                metrics_path = candidate / "metrics.json"
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                mutate(payload["N0"]["validation"])
                metrics_path.write_text(json.dumps(payload), encoding="utf-8")

                with self.assertRaisesRegex(ValueError, "required|exactly"):
                    select_neural_experiments([candidate])

    def test_rejects_zero_validation_support(self) -> None:
        from research.text_attribution.neural_selection import (
            select_neural_experiments,
        )

        with tempfile.TemporaryDirectory() as temporary:
            candidate = write_candidate(Path(temporary), "N0")
            metrics_path = candidate / "metrics.json"
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            validation = payload["N0"]["validation"]
            validation["overall"]["support"] = 0
            validation["per_family"]["alpha"]["support"] = 0
            validation["per_family"]["beta"]["support"] = 0
            metrics_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "support.*positive"):
                select_neural_experiments([candidate])

    def test_rejects_validation_support_mismatching_run_config_split(self) -> None:
        from research.text_attribution.neural_selection import (
            select_neural_experiments,
        )

        with tempfile.TemporaryDirectory() as temporary:
            candidate = write_candidate(
                Path(temporary),
                "N0",
                split_summary={"train": 80, "validation": 999, "test": 10},
            )

            with self.assertRaisesRegex(ValueError, "validation split.*support"):
                select_neural_experiments([candidate])

    def test_requires_existing_directory_and_both_json_files(self) -> None:
        from research.text_attribution.neural_selection import (
            select_neural_experiments,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                select_neural_experiments([root / "missing"])
            empty = root / "empty"
            empty.mkdir()
            with self.assertRaises(FileNotFoundError):
                select_neural_experiments([empty])

    def test_selection_does_not_modify_or_read_other_candidate_artifacts(self) -> None:
        from research.text_attribution.neural_selection import (
            select_neural_experiments,
        )

        with tempfile.TemporaryDirectory() as temporary:
            candidate = write_candidate(Path(temporary), "N0")
            predictions = candidate / "predictions.jsonl"
            predictions.write_text("not-json and must not be read", encoding="utf-8")
            model = candidate / "model"
            model.mkdir()
            marker = model / "weights.bin"
            marker.write_bytes(b"unchanged")
            before = {
                path: (path.stat().st_mtime_ns, path.read_bytes())
                for path in (
                    candidate / "run_config.json",
                    candidate / "metrics.json",
                    predictions,
                    marker,
                )
            }

            select_neural_experiments([candidate])

            after = {
                path: (path.stat().st_mtime_ns, path.read_bytes()) for path in before
            }
            self.assertEqual(after, before)


class NeuralSelectionCliTests(unittest.TestCase):
    def test_cli_writes_json_and_markdown_without_heavy_imports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidates = [
                write_candidate(root / "inputs", "N0", macro_f1=0.5),
                write_candidate(root / "inputs", "N1", macro_f1=0.9),
            ]
            output = root / "selection"
            program = f"""
import builtins
real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split(".", 1)[0] in ("transformers", "torch", "numpy"):
        raise AssertionError(f"selector imported heavy dependency: {{name}}")
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from scripts.select_family_attribution_neural import main
raise SystemExit(main({[
    "--inputs",
    *(str(path) for path in candidates),
    "--output-dir",
    str(output),
]!r}))
"""
            completed = subprocess.run(
                [sys.executable, "-c", program],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(
                (output / "neural_screening_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            markdown = (output / "neural_screening_summary.md").read_text(
                encoding="utf-8"
            )
            self.assertEqual(payload["top_two"], ["N1", "N0"])
            self.assertEqual(payload["selection_scope"], "validation_only")
            self.assertIn("N1", markdown)
            self.assertIn("validation", markdown.lower())

    def test_cli_refuses_an_existing_output_without_mutation(self) -> None:
        from scripts.select_family_attribution_neural import main

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = write_candidate(root, "N0")
            output = root / "selection"
            output.mkdir()
            sentinel = output / "sentinel.txt"
            sentinel.write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "output"):
                main(
                    [
                        "--inputs",
                        str(candidate),
                        "--output-dir",
                        str(output),
                    ]
                )

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
