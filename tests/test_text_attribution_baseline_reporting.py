from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from research.text_attribution.baseline_data import BaselineSample
from research.text_attribution.baseline_models import (
    evaluate_embedding_probe,
    evaluate_sparse_method,
)
from research.text_attribution.baseline_reporting import write_baseline_report


def _fixture() -> tuple[list[BaselineSample], list[int]]:
    samples: list[BaselineSample] = []
    folds: list[int] = []
    for group_index in range(4):
        for label, signature in (
            ("openai", "alpha concise structure"),
            ("claude", "beta nuanced explanation"),
        ):
            samples.append(
                BaselineSample(
                    request_key=f"{label}-{group_index}",
                    text=f"{signature} {signature} item {group_index}",
                    label=label,
                    model=f"{label}-model",
                    prompt_id=f"en-task-{group_index}",
                    semantic_prompt_group=f"task-{group_index}",
                    language="en" if group_index % 2 else "zh",
                    task_type="explanation",
                    target_length="short" if group_index % 2 else "medium",
                    temperature=0.0 if group_index < 2 else 0.8,
                    repeat_index=0,
                )
            )
            folds.append(group_index % 2)
    return samples, folds


class BaselineReportingTests(unittest.TestCase):
    def test_report_writes_machine_readable_text_and_visual_artifacts(self) -> None:
        samples, folds = _fixture()
        char_result = evaluate_sparse_method(
            samples,
            folds,
            method="char_lr",
            seed=42,
        )
        svm_result = evaluate_sparse_method(
            samples,
            folds,
            method="tfidf_svm",
            seed=42,
        )
        embeddings = np.asarray(
            [
                [1.0, 0.0] if sample.label == "openai" else [0.0, 1.0]
                for sample in samples
            ],
            dtype=np.float32,
        )
        probe_result = evaluate_embedding_probe(
            samples,
            folds,
            embeddings,
            seed=42,
        )
        audit = {
            "raw_records": len(samples),
            "kept_records": len(samples),
            "excluded_records": 0,
            "excluded_by_reason": {},
            "counts_by_label": {"claude": 4, "openai": 4},
        }

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "baseline-output"
            artifacts = write_baseline_report(
                samples=samples,
                audit=audit,
                results=[char_result, svm_result, probe_result],
                output_dir=output_dir,
                run_config={"seed": 42, "folds": 2},
            )

            for key in (
                "metrics_json",
                "metrics_csv",
                "predictions_jsonl",
                "report_markdown",
                "dataset_audit_json",
                "run_config_json",
            ):
                self.assertTrue(Path(artifacts[key]).is_file(), key)
                self.assertGreater(Path(artifacts[key]).stat().st_size, 0)
            self.assertEqual(len(artifacts["figures"]), 4)
            for figure in artifacts["figures"]:
                self.assertGreater(Path(figure["png"]).stat().st_size, 0)
                self.assertGreater(Path(figure["pdf"]).stat().st_size, 0)

            metrics = json.loads(Path(artifacts["metrics_json"]).read_text("utf-8"))
            self.assertEqual(
                sorted(metrics["methods"]),
                ["char_lr", "frozen_e5", "tfidf_svm"],
            )
            self.assertIn("language", metrics["methods"]["char_lr"]["subgroups"])
            predictions = Path(artifacts["predictions_jsonl"]).read_text("utf-8")
            self.assertEqual(len(predictions.strip().splitlines()), len(samples) * 3)
            report = Path(artifacts["report_markdown"]).read_text("utf-8")
            self.assertIn("one model per family", report.lower())


if __name__ == "__main__":
    unittest.main()
