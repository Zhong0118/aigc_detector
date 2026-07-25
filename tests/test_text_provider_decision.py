from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from detection.providers import ApiFirstDetectionEngine, ProviderResult


class TextProviderDecisionTests(unittest.TestCase):
    def _engine(self, detection: dict) -> tuple[ApiFirstDetectionEngine, Path, tempfile.TemporaryDirectory]:
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        config_path = root / "config.yaml"
        config_path.write_text(yaml.safe_dump({"detection": detection}), encoding="utf-8")
        return ApiFirstDetectionEngine(config_path=str(config_path)), root, tmp

    def test_text_without_usable_provider_is_not_labeled_human(self) -> None:
        engine, root, tmp = self._engine(
            {
                "threshold": 0.8,
                "demo_provider_enabled": False,
                "local_models": {"enabled": False},
                "text_false_positive_control": {"enabled": True, "min_effective_chars": 8},
            }
        )
        self.addCleanup(tmp.cleanup)
        text_path = root / "input.txt"
        text_path.write_text("这是一段足够长的待检测文本。", encoding="utf-8")

        package = engine.detect(text_path, "text", text_path.read_text(encoding="utf-8"))

        self.assertEqual(package.label, "insufficient_evidence")
        self.assertIsNone(package.decision_details["score"])
        self.assertEqual(package.score, 0.0)

    def test_text_middle_score_uses_uncertain_decision(self) -> None:
        engine, root, tmp = self._engine(
            {
                "threshold": 0.8,
                "demo_provider_enabled": True,
                "local_models": {"enabled": False},
                "text_false_positive_control": {
                    "enabled": True,
                    "min_effective_chars": 8,
                    "default_low_threshold": 0.4,
                    "default_high_threshold": 0.8,
                },
            }
        )
        self.addCleanup(tmp.cleanup)
        text_path = root / "input.txt"
        text = "这是一段结构清晰但证据仍然不足的说明文字。"
        text_path.write_text(text, encoding="utf-8")
        demo_result = ProviderResult("demo", "test", "ok", 0.62, "ai")

        with patch.object(engine, "_run_demo_provider", return_value=demo_result):
            package = engine.detect(text_path, "text", text)

        self.assertEqual(package.label, "uncertain")
        self.assertEqual(package.score, 0.62)
        self.assertEqual(package.threshold, 0.8)
        self.assertEqual(package.decision_details["score_kind"], "calibrated_ai_risk")

    def test_provider_status_has_no_online_public_source_branch(self) -> None:
        engine, _, tmp = self._engine(
            {
                "demo_provider_enabled": False,
                "local_models": {"enabled": False},
                "text_false_positive_control": {"enabled": True},
            }
        )
        self.addCleanup(tmp.cleanup)

        names = {item["name"] for item in engine.provider_status()}

        self.assertNotIn("public_source_matcher", names)
        self.assertIn("text_false_positive_controller", names)

    def test_provider_decision_exposes_artifact_v2_reliability_evidence(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        artifact = root / "thresholds.json"
        artifact.write_text(
            json.dumps(
                {
                    "artifact_schema_version": 2,
                    "version": "provider-v2",
                    "groups": {
                        "en:academic_technical:short": {
                            "low_threshold": 0.4,
                            "high_threshold": 0.92,
                            "human_validation_samples": 80,
                            "validation_fpr": 0.0125,
                            "ai_validation_samples": 75,
                            "ai_recall": 0.84,
                            "validation_auroc": 0.9,
                            "reliability_status": "ok",
                            "warnings": [],
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        engine, engine_root, engine_tmp = self._engine(
            {
                "threshold": 0.8,
                "demo_provider_enabled": True,
                "local_models": {"enabled": False},
                "text_false_positive_control": {
                    "enabled": True,
                    "artifact_path": str(artifact),
                    "min_effective_chars": 8,
                },
            }
        )
        self.addCleanup(engine_tmp.cleanup)
        text_path = engine_root / "input.txt"
        text = "This study proposes a neural method and reports experimental results."
        text_path.write_text(text, encoding="utf-8")
        demo_result = ProviderResult("demo", "test", "ok", 0.7, "ai")

        with patch.object(engine, "_run_demo_provider", return_value=demo_result):
            package = engine.detect(text_path, "text", text)

        self.assertEqual(package.decision_details["threshold_ai_recall"], 0.84)
        self.assertEqual(package.decision_details["threshold_ai_validation_samples"], 75)
        self.assertEqual(package.decision_details["threshold_validation_auroc"], 0.9)
        self.assertEqual(package.decision_details["threshold_reliability_status"], "ok")


if __name__ == "__main__":
    unittest.main()
