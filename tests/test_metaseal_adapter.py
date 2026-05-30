from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from PIL import Image

from provenance.metaseal_adapter import MetaSealWatermarkDetector


class MetaSealWatermarkDetectorTests(unittest.TestCase):
    def test_missing_dependency_returns_clear_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "audio.wav"
            path.write_bytes(b"fake")
            detector = MetaSealWatermarkDetector(config={"device": "cpu", "audio": {"enabled": True}})

            with patch("provenance.metaseal_adapter.importlib.import_module", side_effect=ImportError("missing")):
                result = detector.detect(path, "audio")

        self.assertEqual(result["status"], "dependency_missing")
        self.assertEqual(result["provider"], "AudioSeal")

    def test_audioseal_detector_is_called_when_available(self) -> None:
        audioseal_module = types.SimpleNamespace()
        detector_instance = Mock()
        detector_instance.to.return_value = detector_instance
        detector_instance.eval.return_value = detector_instance
        detector_instance.detect_watermark.return_value = (torch.tensor(0.82), torch.ones(1, 16))
        audioseal_module.AudioSeal = types.SimpleNamespace(
            load_detector=Mock(return_value=detector_instance)
        )
        librosa_module = types.SimpleNamespace(load=Mock(return_value=([0.0, 0.1], 16000)))

        def fake_import(name: str):
            if name == "audioseal":
                return audioseal_module
            if name == "librosa":
                return librosa_module
            raise ImportError(name)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "audio.wav"
            path.write_bytes(b"fake")
            detector = MetaSealWatermarkDetector(config={"device": "cpu", "audio": {"enabled": True}})

            with patch.object(detector, "_import_module", side_effect=fake_import):
                result = detector.detect(path, "audio")

        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["result"]["detected"])
        self.assertEqual(result["result"]["confidence"], 0.82)
        audioseal_module.AudioSeal.load_detector.assert_called_once()

    def test_videoseal_image_detector_is_called_when_available(self) -> None:
        videoseal_module = types.SimpleNamespace()
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = model
        model.detect.return_value = {"preds": torch.tensor([[0.1, 0.9, -0.2]])}
        videoseal_module.load = Mock(return_value=model)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image.png"
            Image.new("RGB", (8, 8), color="white").save(path)
            detector = MetaSealWatermarkDetector(config={"device": "cpu", "image": {"enabled": True}})

            with patch.object(detector, "_import_module", return_value=videoseal_module):
                result = detector.detect(path, "image")

        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["result"]["detected"])
        self.assertEqual(result["provider"], "VideoSeal")
        videoseal_module.load.assert_called_once()

    def test_videoseal_loader_uses_package_root_when_configured(self) -> None:
        videoseal_module = types.SimpleNamespace()
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = model
        videoseal_module.load = Mock(return_value=model)

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            detector = MetaSealWatermarkDetector(
                config={"device": "cpu", "image": {"enabled": True, "package_root": str(root), "model": "videoseal"}}
            )

            with patch.object(detector, "_import_module", return_value=videoseal_module):
                with patch("provenance.metaseal_adapter.os.chdir") as chdir:
                    loaded = detector._load_videoseal_model({"package_root": str(root), "model": "videoseal"})

        self.assertIs(loaded, model)
        chdir.assert_any_call(root)
        videoseal_module.load.assert_called_once_with("videoseal")

    def test_textseal_can_use_configured_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "input.txt"
            path.write_text("watermarked text", encoding="utf-8")
            detector = MetaSealWatermarkDetector(
                config={"device": "cpu", "text": {"enabled": True, "command": "textseal"}}
            )
            completed = Mock(returncode=0, stdout='{"detected": true, "confidence": 0.77}', stderr="")

            with patch("provenance.metaseal_adapter.subprocess.run", return_value=completed):
                result = detector.detect(path, "text")

        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["result"]["detected"])
        self.assertEqual(result["result"]["confidence"], 0.77)


if __name__ == "__main__":
    unittest.main()
