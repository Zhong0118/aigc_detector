from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from detection.providers import (
    ApiFirstDetectionEngine,
    DetectionUnavailableError,
    ProviderResult,
)


class DetectionAvailabilityTests(unittest.TestCase):
    def _detect(self, results: list[ProviderResult]):
        config = {
            "detection": {
                "threshold": 0.8,
                "demo_provider_enabled": False,
                "local_models": {"enabled": False},
            }
        }
        with patch.object(ApiFirstDetectionEngine, "_load_config", return_value=config):
            engine = ApiFirstDetectionEngine()
        with patch.object(engine, "_run_configured_api_providers", return_value=results):
            return engine.detect(Path("image.png"), "image")

    def test_image_without_any_detector_has_no_human_result(self):
        with self.assertRaises(DetectionUnavailableError):
            self._detect([])

    def test_failed_and_skipped_detectors_do_not_produce_a_score(self):
        results = [
            ProviderResult("cloud", "detection_api", "error", None, "not_run", {}, "secret-token"),
            ProviderResult("local", "local_model", "skipped", None, "not_run", {}),
        ]
        with self.assertRaises(DetectionUnavailableError) as raised:
            self._detect(results)
        self.assertEqual(raised.exception.providers, [
            {"provider": "cloud", "status": "error"},
            {"provider": "local", "status": "skipped"},
        ])
        self.assertNotIn("secret-token", str(raised.exception))

    def test_successful_zero_score_remains_a_valid_detection(self):
        result = self._detect([
            ProviderResult("cloud", "detection_api", "ok", 0.0, "human", {}),
        ])
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.label, "human")

    def test_successful_detector_is_used_when_another_detector_fails(self):
        result = self._detect([
            ProviderResult("failed", "detection_api", "error", None, "not_run", {}),
            ProviderResult("working", "detection_api", "ok", 0.9, "ai", {}),
        ])
        self.assertEqual(result.score, 0.9)
        self.assertEqual(result.label, "ai")

    def test_http_error_is_503_and_does_not_include_credentials_or_a_label(self):
        from api.main import detection_unavailable_handler

        error = DetectionUnavailableError([
            ProviderResult("cloud", "detection_api", "error", None, "not_run", {}, "secret-token"),
        ])
        response = asyncio.run(detection_unavailable_handler(None, error))
        payload = json.loads(response.body)
        self.assertEqual(response.status_code, 503)
        self.assertIn("检测未完成", payload["detail"])
        self.assertNotIn("label", payload)
        self.assertNotIn("secret-token", response.body.decode("utf-8"))

    def test_upload_endpoint_returns_no_verdict_when_detection_is_unavailable(self):
        from fastapi.testclient import TestClient
        from api.main import app

        with patch("api.routes.get_analysis_service") as get_service:
            get_service.return_value.analyze_upload.side_effect = DetectionUnavailableError([])
            response = TestClient(app).post(
                "/api/v1/analyze/file", files={"file": ("image.png", b"image", "image/png")}
            )
        self.assertEqual(response.status_code, 503)
        self.assertIn("检测未完成", response.json()["detail"])
        self.assertNotIn("detection", response.json())


if __name__ == "__main__":
    unittest.main()
