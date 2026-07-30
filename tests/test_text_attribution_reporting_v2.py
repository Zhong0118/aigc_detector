from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from research.text_attribution.attribution_metrics import (
    build_prediction_records,
    evaluate_family_predictions,
)
from research.text_attribution.attribution_reporting import write_attribution_report
from research.text_attribution.training_data import AttributionSample


FAMILY_LABELS = (
    "qwen",
    "claude",
    "openai",
    "gemini",
    "deepseek",
    "doubao",
    "glm",
    "kimi",
    "mimo",
    "minimax",
)

EXPECTED_ARTIFACTS = {
    "run_config": "run_config.json",
    "dataset_audit": "dataset_audit.json",
    "metrics": "metrics.json",
    "predictions": "predictions.jsonl",
    "report": "report.md",
    "overall_metrics_png": "overall_metrics.png",
    "confusion_matrices_png": "confusion_matrices.png",
    "per_model_diagnostics_png": "per_model_diagnostics.png",
    "calibration_png": "calibration.png",
    "subgroup_metrics_png": "subgroup_metrics.png",
}


def _records(
    experiment_id: str,
    split: str,
    *,
    repeats: int = 2,
) -> tuple:
    samples: list[AttributionSample] = []
    probabilities: list[list[float]] = []
    for repeat in range(repeats):
        for family_index, family in enumerate(FAMILY_LABELS):
            samples.append(
                AttributionSample(
                    request_key=f"{experiment_id}-{split}-{repeat}-{family}",
                    text=f"complete text from {family}",
                    family_id=family,
                    model_id=f"{family}-model-{repeat % 2}",
                    prompt_id=f"en-task-{repeat:03d}",
                    semantic_prompt_group=f"task-{repeat:03d}",
                    language="en" if family_index % 2 else "zh",
                    task_type="explanation" if repeat == 0 else "argument",
                    target_length="medium" if repeat == 0 else "long",
                    temperature=float(repeat) * 0.8,
                    repeat_index=repeat,
                )
            )
            row = [0.02] * len(FAMILY_LABELS)
            row[family_index] = 0.82
            probabilities.append(row)
    return build_prediction_records(
        tuple(samples),
        np.asarray(probabilities),
        FAMILY_LABELS,
        experiment_id,
        split,
    )


def _metrics(records: tuple) -> dict:
    return evaluate_family_predictions(records, FAMILY_LABELS)


class AttributionReportingTests(unittest.TestCase):
    def test_writes_parseable_report_and_real_figures_for_selection_metadata(self) -> None:
        validation = _records("B0", "validation")
        test = _records("B1", "test", repeats=1)
        metrics = {
            "B0": {"validation": _metrics(validation)},
            "B1": {"test": _metrics(test)},
            "selection": {
                "selected_experiment": "B0",
                "selection_split": "validation",
                "selection_metric": "macro_f1",
                "selected_score": np.float64(0.82),
            },
        }
        mapping_prediction = validation[0].to_dict()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "report"
            artifacts = write_attribution_report(
                output_dir=root,
                run_config={
                    "run_id": "family-v1",
                    "seed": np.int64(42),
                    "experiments": ["B0", "B1"],
                },
                dataset_audit={"kept_records": np.int64(30)},
                split_summary={
                    "train": np.int64(70),
                    "validation": np.int64(20),
                    "test": np.int64(10),
                },
                metrics=metrics,
                predictions=(*validation, mapping_prediction),
            )

            self.assertEqual(set(artifacts), set(EXPECTED_ARTIFACTS))
            for key, filename in EXPECTED_ARTIFACTS.items():
                path = Path(artifacts[key])
                self.assertEqual(path, root / filename)
                self.assertTrue(path.is_file(), key)

            run_config = json.loads((root / "run_config.json").read_text("utf-8"))
            audit = json.loads((root / "dataset_audit.json").read_text("utf-8"))
            written_metrics = json.loads((root / "metrics.json").read_text("utf-8"))
            prediction_rows = [
                json.loads(line)
                for line in (root / "predictions.jsonl").read_text("utf-8").splitlines()
            ]
            self.assertEqual(run_config["seed"], 42)
            self.assertEqual(audit["kept_records"], 30)
            self.assertEqual(
                written_metrics["selection"]["selected_experiment"],
                "B0",
            )
            self.assertEqual(len(prediction_rows), len(validation) + 1)
            self.assertEqual(
                tuple(prediction_rows[0]["family_probabilities"]),
                FAMILY_LABELS,
            )

            report = (root / "report.md").read_text("utf-8")
            self.assertIn("# Family Attribution Baseline Report", report)
            self.assertIn("family-v1", report)
            self.assertIn("Frozen split", report)
            self.assertIn("Accuracy", report)
            self.assertIn("Macro-F1", report)
            self.assertIn("Weighted-F1", report)
            self.assertIn("ECE", report)
            self.assertIn("Brier", report)
            self.assertIn("NLL", report)
            self.assertIn("Selected experiment", report)
            self.assertIn("B0", report)
            self.assertIn("not exact-version attribution", report)

            for key in (
                "overall_metrics_png",
                "confusion_matrices_png",
                "per_model_diagnostics_png",
                "calibration_png",
                "subgroup_metrics_png",
            ):
                payload = Path(artifacts[key]).read_bytes()
                self.assertTrue(payload.startswith(b"\x89PNG\r\n\x1a\n"), key)
                self.assertGreater(len(payload), 2_000, key)

    def test_uses_accuracy_diagnostic_key_and_allows_absent_subgroup_rows(self) -> None:
        records = _records("B2", "validation", repeats=1)
        result = _metrics(records)
        result["per_model"]["qwen-model-0"]["accuracy"] = 0.75
        result["subgroups"] = {}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_attribution_report(
                output_dir=root,
                run_config={"seed": 7, "experiments": ["B2"]},
                dataset_audit={"kept_records": 10},
                split_summary={"train": 0, "validation": 10, "test": 0},
                metrics={"B2": {"validation": result}},
                predictions=records,
            )
            report = (root / "report.md").read_text("utf-8")

        self.assertIn("qwen-model-0", report)
        self.assertIn("0.7500", report)

    def test_rejects_nonfinite_json_values_and_metricless_payloads(self) -> None:
        records = _records("B0", "validation", repeats=1)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "finite"):
                write_attribution_report(
                    output_dir=Path(temporary) / "nonfinite",
                    run_config={"seed": float("nan")},
                    dataset_audit={"kept_records": 10},
                    split_summary={"validation": 10},
                    metrics={"B0": {"validation": _metrics(records)}},
                    predictions=records,
                )
            with self.assertRaisesRegex(ValueError, "metric"):
                write_attribution_report(
                    output_dir=Path(temporary) / "metricless",
                    run_config={"seed": 42},
                    dataset_audit={"kept_records": 10},
                    split_summary={"validation": 10},
                    metrics={"selection": {"selected_experiment": "B0"}},
                    predictions=records,
                )
            with self.assertRaisesRegex(ValueError, "predictions"):
                write_attribution_report(
                    output_dir=Path(temporary) / "predictionless",
                    run_config={"seed": 42},
                    dataset_audit={"kept_records": 10},
                    split_summary={"validation": 10},
                    metrics={"B0": {"validation": _metrics(records)}},
                    predictions=(),
                )


if __name__ == "__main__":
    unittest.main()
