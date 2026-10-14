"""Regressions for runtime threshold overrides and their metric provenance."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from detection.false_positive_controller import FalsePositiveController


class RuntimeThresholdTests(unittest.TestCase):
    def evaluate(self, entry, **config):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "thresholds.json"
            path.write_text(json.dumps({"artifact_schema_version": 2, "groups": {"global": entry}}))
            controller = FalsePositiveController({"artifact_path": str(path), **config})
            return controller.evaluate(
                "This is a plain passage about daily life. " * 8,
                [SimpleNamespace(status="ok", score=0.7, details={})],
            )

    def test_calibration_search_floor_does_not_become_product_threshold(self):
        for floor in (0, 0.1, 0.4, 0.95, None):
            with self.subTest(floor=floor):
                decision = self.evaluate({"low_threshold": 0.4, "high_threshold": 0.9996,
                                          "threshold_floor": floor})
                self.assertEqual(decision.high_threshold, 0.8)
                self.assertEqual(decision.decision, "uncertain")
                self.assertGreater(decision.high_threshold, decision.low_threshold)

    def test_invalid_domain_candidate_uses_valid_default_or_preserves_calibration(self):
        for candidate in (0, 0.4, float("nan"), float("inf"), 1.1, "invalid", None):
            with self.subTest(candidate=candidate):
                decision = self.evaluate(
                    {"low_threshold": 0.4, "high_threshold": 0.9996},
                    domain_thresholds={"general_qa": {"high_threshold": candidate}},
                )
                self.assertEqual(decision.high_threshold, 0.8)
        decision = self.evaluate({"low_threshold": 0.9, "high_threshold": 0.9996,
                                  "threshold_floor": 0})
        self.assertEqual(decision.high_threshold, 0.9996)
        self.assertGreater(decision.high_threshold, decision.low_threshold)

    def test_changed_cutoffs_do_not_report_old_threshold_dependent_metrics(self):
        cases = (
            {"high_threshold": 0.9996, "threshold_floor": 0},
            {"high_threshold": 0.8, "calibrated_high_threshold": 0.9996},
            {"high_threshold": 0.88, "calibrated_high_threshold": 0.9996,
             "warnings": ["runtime_high_threshold_capped_to_floor"]},
            {"high_threshold": 0.92, "warnings": ["runtime_high_threshold_capped_to_floor"]},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                decision = self.evaluate({
                    "low_threshold": 0.4, "validation_fpr": 0.02, "ai_recall": 0.081,
                    "validation_auroc": 0.75, "human_validation_samples": 100,
                    "ai_validation_samples": 80, "reliability_status": "ok", **overrides,
                })
                self.assertIsNone(decision.threshold_validation_fpr)
                self.assertIsNone(decision.threshold_ai_recall)
                self.assertEqual(decision.threshold_validation_auroc, 0.75)
                self.assertEqual(decision.threshold_reliability_status, "not_evaluated")
                self.assertIn("effective_high_threshold_not_evaluated", decision.threshold_warnings)
                self.assertEqual(decision.threshold_validation_samples, 100)
                self.assertEqual(decision.threshold_ai_validation_samples, 80)

    def test_unchanged_cutoff_preserves_evaluated_metrics(self):
        decision = self.evaluate({
            "low_threshold": 0.4, "high_threshold": 0.93,
            "calibrated_high_threshold": 0.93, "validation_fpr": 0.02,
            "ai_recall": 0.8, "validation_auroc": 0.9, "reliability_status": "ok",
        })
        self.assertEqual(decision.threshold_validation_fpr, 0.02)
        self.assertEqual(decision.threshold_ai_recall, 0.8)
        self.assertEqual(decision.threshold_validation_auroc, 0.9)
        self.assertEqual(decision.threshold_reliability_status, "ok")
        self.assertEqual(decision.threshold_warnings, [])


if __name__ == "__main__":
    unittest.main()
