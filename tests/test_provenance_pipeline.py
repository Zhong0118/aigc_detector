from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from detection.providers import DetectionPackage
from provenance.c2pa_reader import read_c2pa_metadata
from provenance.fingerprint_registry import FingerprintRegistry
from provenance.pipeline import ProvenancePipeline
from storage.database import close_db, get_session, init_db
from storage.models import Content


class C2PAReaderTests(unittest.TestCase):
    def test_read_c2pa_metadata_uses_c2patool_when_available(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            media_path = Path(tmp_dir) / "image.png"
            media_path.write_bytes(b"fake")
            tool_output = {
                "active_manifest": "manifest-1",
                "manifests": {
                    "manifest-1": {
                        "claim_generator": "test-generator",
                        "title": "demo.png",
                        "assertions": [{"label": "c2pa.actions"}],
                        "signature_info": {"cert_serial_number": "123"},
                    }
                },
            }
            completed = Mock(returncode=0, stdout=json.dumps(tool_output), stderr="")

            with patch("provenance.c2pa_reader.shutil.which", return_value="c2patool"):
                with patch("provenance.c2pa_reader.subprocess.run", return_value=completed):
                    metadata = read_c2pa_metadata(media_path)

        self.assertIsNotNone(metadata)
        self.assertEqual(metadata["parser"], "c2patool")
        self.assertTrue(metadata["found"])
        self.assertEqual(metadata["claim_generator"], "test-generator")
        self.assertEqual(metadata["signature_info"]["cert_serial_number"], "123")


class FingerprintRegistryTests(unittest.TestCase):
    def test_lookup_returns_exact_and_near_image_phash_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            init_db(str(Path(tmp_dir) / "test.db"))
            session = get_session()
            try:
                session.add(
                    Content(
                        filename="known.png",
                        modality="image",
                        fingerprint="0000000000000000",
                        file_hash="0000000000000000",
                        source_model="known-source",
                    )
                )
                session.commit()
            finally:
                session.close()

            matches = FingerprintRegistry().lookup(
                "0000000000000001",
                modality="image",
                max_distance=4,
            )
            close_db()

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].match_type, "near_phash")
        self.assertGreater(matches[0].similarity, 0.98)
        self.assertEqual(matches[0].source_model, "known-source")

    def test_pipeline_includes_fingerprint_registry_lookup(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            init_db(str(Path(tmp_dir) / "test.db"))
            session = get_session()
            try:
                session.add(
                    Content(
                        filename="old.txt",
                        modality="text",
                        fingerprint="abc123",
                        file_hash="abc123",
                    )
                )
                session.commit()
            finally:
                session.close()

            media_path = Path(tmp_dir) / "input.txt"
            media_path.write_text("same text", encoding="utf-8")
            detection = DetectionPackage(
                score=0.1,
                label="human",
                threshold=0.5,
                provider_results=[],
                modality_scores={"text": 0.1},
                model_scores={},
            )
            result = ProvenancePipeline(deep_threshold=0.6).analyze(
                media_path,
                "text",
                detection,
                fingerprint="abc123",
            )
            close_db()

        self.assertEqual(result["fingerprint_registry"]["status"], "ok")
        self.assertEqual(result["fingerprint_registry"]["matches"][0]["match_type"], "exact")
        self.assertEqual(result["fingerprint_registry"]["matches"][0]["similarity"], 1.0)


if __name__ == "__main__":
    unittest.main()
