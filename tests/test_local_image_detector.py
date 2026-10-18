from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from detection.image_source_detector import ImageSourceDetector
from detection.providers import ApiFirstDetectionEngine


class LocalImageDetectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        model_path = Path(self.tmp.name) / "classifier.json"
        self.artifact = {
            "format": "aigc-logistic-regression-v1",
            "classes": ["Real", "generator-a", "generator-b"],
            "coef": [[-2, 0], [2, 0], [0, 2]],
            "intercept": [0, 0, 0],
            "metadata": {"feature_type": "clip"},
            "solver": "lbfgs",
            "multi_class": "multinomial",
        }
        model_path.write_text(json.dumps(self.artifact), encoding="utf-8")
        self.config = {"enabled": True, "model_path": str(model_path), "top_k": 5, "validated": True}
        self.detector = ImageSourceDetector(self.config)

    def test_real_class_mass_is_excluded_from_ai_risk(self):
        with patch.object(self.detector, "extract_features", return_value=[-1, 0]):
            result = self.detector.detect("photo.png")
        expected = np.exp([2, -2, 0])
        expected /= expected.sum()
        self.assertAlmostEqual(result["score"], 1 - expected[0])
        self.assertAlmostEqual(result["class_scores"]["Real"], expected[0])
        self.assertNotIn("Real", result["model_scores"])

    def test_source_candidates_are_conditional_on_ai(self):
        with patch.object(self.detector, "extract_features", return_value=[1, 0]):
            result = self.detector.detect("generated.png")
        candidates = result["attribution"]["top_k"]
        self.assertEqual(candidates[0]["model"], "generator-a")
        self.assertAlmostEqual(sum(c["probability"] for c in candidates), 1.0)
        self.assertNotIn("Real", [c["model"] for c in candidates])

    def test_feature_mismatch_or_nan_is_rejected(self):
        for features in ([0, 0, 0], [float("nan"), 0]):
            with self.subTest(features=features), self.assertRaises(ValueError):
                self.detector.predict_features(features)

    def test_auto_device_is_resolved_before_loading_clip(self):
        import torch
        from PIL import Image
        from provenance.attribution import ProvenanceAttributionEngine

        image_path = Path(self.tmp.name) / "photo.png"
        Image.new("RGB", (2, 2)).save(image_path)
        model = Mock()
        model.to.return_value = model
        model.get_image_features.return_value = torch.ones((1, 2))
        processor = Mock(return_value={"pixel_values": torch.zeros((1, 3, 2, 2))})
        with patch("torch.cuda.is_available", return_value=False), \
                patch("transformers.CLIPModel.from_pretrained", return_value=model), \
                patch("transformers.CLIPImageProcessor.from_pretrained", return_value=processor):
            features = ProvenanceAttributionEngine()._extract_clip_image_features(
                image_path, {}, {"device": "auto", "cache_dir": str(Path(self.tmp.name) / "cache")}
            )
        model.to.assert_called_once_with("cpu")
        self.assertEqual(len(features), 2)

    def test_unvalidated_image_model_abstains_and_hides_source_verdict(self):
        config = {"detection": {
            "demo_provider_enabled": False,
            "local_models": {"enabled": True, "image_source": {**self.config, "validated": False}},
        }}
        with patch.object(ApiFirstDetectionEngine, "_load_config", return_value=config):
            engine = ApiFirstDetectionEngine()
        with patch.object(ImageSourceDetector, "extract_features", return_value=[1, 0]):
            result = engine.detect(Path("image.png"), "image")
        self.assertGreater(result.score, 0.8)
        self.assertEqual(result.label, "uncertain")
        self.assertEqual(result.model_scores, {})
        attribution = result.provider_results[0].details["attribution"]
        self.assertEqual(attribution["status"], "unvalidated")
        self.assertEqual(attribution["top_k"], [])
        self.assertTrue(attribution["experimental_candidates"])

    def test_experimental_image_report_stays_uncertain_and_does_not_call_llm(self):
        from reports.generator import ReportGenerator

        analysis = {
            "filename": "photo.png", "modality": "image",
            "detection": {"score": 0.99, "label": "uncertain", "threshold": 0.8,
                          "providers": [{"details": {"experimental": True}}]},
            "provenance": {"deep_triggered": True},
        }
        reporter = ReportGenerator(config={"report": {"provider": "llm"}})
        with patch.object(reporter, "_generate_llm_report") as external_report:
            report = reporter.generate(analysis)
        external_report.assert_not_called()
        self.assertEqual(report["provider"], "template")
        self.assertIn("不确定", report["summary"])

    def test_local_image_branch_is_available_without_cloud_providers(self):
        config = {"detection": {
            "demo_provider_enabled": False,
            "threshold": 0.8,
            "local_models": {"enabled": True, "image_source": self.config},
        }}
        with patch.object(ApiFirstDetectionEngine, "_load_config", return_value=config):
            engine = ApiFirstDetectionEngine()
        with patch.object(ImageSourceDetector, "extract_features", return_value=[1, 0]):
            result = engine.detect(Path("image.png"), "image")
        self.assertEqual(result.label, "ai")
        self.assertEqual(result.provider_results[0].provider, "local_image_source_classifier")
        self.assertEqual(result.provider_results[0].status, "ok")

        from provenance.pipeline import ProvenancePipeline

        pipeline = object.__new__(ProvenancePipeline)
        pipeline.attribution_config = {"enabled": True, "image": {"enabled": True}}
        pipeline.attributor = Mock()
        attribution = pipeline._run_attribution(Path("image.png"), "image", result, True)
        self.assertEqual(attribution, result.provider_results[0].details["attribution"])
        pipeline.attributor.attribute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
