from __future__ import annotations

import json
import os
import joblib
import tempfile
import types
import unittest
import concurrent.futures
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

from detection.providers import DetectionPackage
from provenance.attribution import ProvenanceAttributionEngine
from provenance.pipeline import ProvenancePipeline
from scripts.train_image_attribution import decode_dataset_label
from scripts.train_text_attribution import merge_labeled_samples, normalize_mage_label


class ProvenanceAttributionEngineTests(unittest.TestCase):
    def test_llmdet_text_attributor_normalizes_top_k(self) -> None:
        llmdet_module = types.SimpleNamespace(
            load_probability=Mock(),
            detect=Mock(return_value=[{"GPT-2": 0.7, "Human_write": 0.2, "OPT": 0.1}])
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            text_path = Path(tmp_dir) / "input.txt"
            text_path.write_text("This is a generated-looking paragraph.", encoding="utf-8")
            engine = ProvenanceAttributionEngine(
                {
                    "enabled": True,
                    "text": {"enabled": True, "provider": "llmdet", "top_k": 2},
                }
            )

            with patch.object(engine, "_import_module", return_value=llmdet_module):
                result = engine.attribute(text_path, "text")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["provider"], "fusion")
        self.assertEqual(result["top_k"][0]["model"], "GPT-2")
        self.assertEqual(result["top_k"][0]["probability"], 0.7)
        self.assertEqual(len(result["top_k"]), 2)
        self.assertEqual(result["branches"][0]["provider"], "llmdet")
        llmdet_module.load_probability.assert_called_once()

    def test_llmdet_dependency_error_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            text_path = Path(tmp_dir) / "input.txt"
            text_path.write_text("hello", encoding="utf-8")
            engine = ProvenanceAttributionEngine(
                {"enabled": True, "text": {"enabled": True, "provider": "llmdet"}}
            )

            with patch.object(engine, "_import_module", side_effect=ImportError("No module named 'unilm'")):
                result = engine.attribute(text_path, "text")

        self.assertEqual(result["status"], "dependency_missing")
        self.assertEqual(result["provider"], "fusion")
        self.assertEqual(result["branches"][0]["provider"], "llmdet")
        self.assertIn("unilm", result["branches"][0]["error"])

    def test_unilm_compat_shim_allows_llmdet_import_shape(self) -> None:
        engine = ProvenanceAttributionEngine({"enabled": True})

        engine._install_unilm_compat_shim()

        self.assertIn("unilm", __import__("sys").modules)
        self.assertTrue(hasattr(__import__("sys").modules["unilm"], "UniLMTokenizer"))

    def test_universal_attribution_cli_parses_json_output(self) -> None:
        payload = {
            "top_k": [
                {"model": "stable-diffusion-xl", "probability": 0.62},
                {"model": "midjourney", "probability": 0.24},
            ],
            "confidence": 0.62,
            "unknown_probability": 0.14,
        }
        completed = Mock(returncode=0, stdout=json.dumps(payload), stderr="")

        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "image.png"
            image_path.write_bytes(b"fake")
            engine = ProvenanceAttributionEngine(
                {
                    "enabled": True,
                    "image": {
                        "enabled": True,
                        "provider": "universal_attribution",
                        "command": "ua-detect",
                    },
                }
            )

            with patch("provenance.attribution.subprocess.run", return_value=completed):
                result = engine.attribute(image_path, "image")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["provider"], "fusion")
        self.assertEqual(result["top_k"][0]["model"], "stable-diffusion-xl")
        self.assertEqual(result["confidence"], 0.62)
        self.assertEqual(result["branches"][0]["provider"], "universal_attribution")

    def test_pipeline_runs_configured_attribution_only_when_deep_triggered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            text_path = Path(tmp_dir) / "input.txt"
            text_path.write_text("hello", encoding="utf-8")
            pipeline = ProvenancePipeline(
                deep_threshold=0.6,
                attribution_config={"enabled": True, "text": {"enabled": True, "provider": "llmdet"}},
            )
            pipeline.attributor.attribute = Mock(return_value={"status": "ok", "top_k": [], "confidence": 0.0})
            low_detection = DetectionPackage(
                score=0.2,
                label="human",
                threshold=0.5,
                provider_results=[],
                modality_scores={"text": 0.2},
                model_scores={},
            )
            high_detection = DetectionPackage(
                score=0.9,
                label="ai",
                threshold=0.5,
                provider_results=[],
                modality_scores={"text": 0.9},
                model_scores={},
            )

            low_result = pipeline.analyze(text_path, "text", low_detection, fingerprint="low")
            high_result = pipeline.analyze(text_path, "text", high_detection, fingerprint="high")

        self.assertEqual(low_result["attribution"]["status"], "skipped")
        self.assertEqual(high_result["attribution"]["status"], "ok")
        pipeline.attributor.attribute.assert_called_once_with(text_path, "text")

    def test_pipeline_keeps_provider_hints_and_still_runs_llmdet_attribution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            text_path = Path(tmp_dir) / "input.txt"
            text_path.write_text("hello generated content", encoding="utf-8")
            pipeline = ProvenancePipeline(
                deep_threshold=0.6,
                attribution_config={"enabled": True, "text": {"enabled": True, "provider": "llmdet"}},
            )
            pipeline.attributor.attribute = Mock(
                return_value={
                    "status": "ok",
                    "source": "llmdet",
                    "provider": "llmdet",
                    "top_k": [{"model": "GPT-2", "probability": 0.6}],
                    "confidence": 0.6,
                }
            )
            detection = DetectionPackage(
                score=0.9,
                label="ai",
                threshold=0.5,
                provider_results=[],
                modality_scores={"text": 0.9},
                model_scores={"local-text-zh-long": 0.9997, "chatgpt": 0.9},
            )

            result = pipeline.analyze(text_path, "text", detection, fingerprint="high")

        self.assertEqual(result["provider_hints"]["top_k"][0]["model"], "local-text-zh-long")
        self.assertEqual(result["attribution"]["source"], "llmdet")
        self.assertEqual(result["attribution"]["top_k"][0]["model"], "GPT-2")

    def test_llmdet_runtime_error_includes_user_visible_install_hint(self) -> None:
        llmdet_module = types.SimpleNamespace(
            load_probability=Mock(),
            detect=Mock(side_effect=RuntimeError("model download failed")),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            text_path = Path(tmp_dir) / "input.txt"
            text_path.write_text("This paragraph is long enough to trigger attribution.", encoding="utf-8")
            engine = ProvenanceAttributionEngine(
                {"enabled": True, "text": {"enabled": True, "provider": "llmdet", "top_k": 5}}
            )

            with patch.object(engine, "_import_module", return_value=llmdet_module):
                result = engine.attribute(text_path, "text")

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["provider"], "fusion")
        self.assertEqual(result["branches"][0]["provider"], "llmdet")
        self.assertIn("model download failed", result["branches"][0]["error"])

    def test_llmdet_timeout_returns_error_status(self) -> None:
        llmdet_module = types.SimpleNamespace(load_probability=Mock(), detect=Mock(return_value=[]))
        engine = ProvenanceAttributionEngine(
            {"enabled": True, "text": {"enabled": True, "provider": "llmdet", "timeout_seconds": 1}}
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            text_path = Path(tmp_dir) / "input.txt"
            text_path.write_text("This paragraph is long enough to trigger attribution.", encoding="utf-8")
            with patch.object(engine, "_import_module", return_value=llmdet_module):
                with patch("provenance.attribution.ThreadPoolExecutor") as executor_cls:
                    future = Mock()
                    future.result.side_effect = concurrent.futures.TimeoutError()
                    executor_cls.return_value.__enter__.return_value.submit.return_value = future
                    result = engine.attribute(text_path, "text")

        self.assertEqual(result["status"], "error")
        self.assertIn("timed out", result["branches"][0]["error"])

    def test_llmdet_preflight_reports_data_mismatch_for_missing_or_bad_npz(self) -> None:
        engine = ProvenanceAttributionEngine({"enabled": True})

        with tempfile.TemporaryDirectory() as tmp_dir:
            npz_dir = Path(tmp_dir) / "datasets" / "downloads" / "extracted" / "abc" / "npz"
            npz_dir.mkdir(parents=True)
            for name in ["gpt2", "opt"]:
                with zipfile.ZipFile(npz_dir / f"{name}.npz", "w") as archive:
                    archive.writestr("placeholder.npy", b"placeholder")
            (npz_dir / "bart.npz").write_text("not a zip", encoding="utf-8")

            result = engine._preflight_llmdet_data({"cache_dir": tmp_dir})

        self.assertEqual(result["status"], "data_mismatch")
        self.assertIn("missing_npz", result)
        self.assertIn("bad_npz", result)
        self.assertIn("bart", result["bad_npz"])

    def test_llmdet_sets_project_huggingface_cache(self) -> None:
        engine = ProvenanceAttributionEngine({"enabled": True})

        with patch.dict(os.environ, {}, clear=True):
            engine._configure_huggingface_cache({"cache_dir": "models/huggingface"})

            self.assertEqual(os.environ["HF_HOME"], "models\\huggingface" if os.name == "nt" else "models/huggingface")
            self.assertIn("hub", os.environ["HF_HUB_CACHE"])

    def test_text_prototype_branch_returns_top_k_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            prototypes = Path(tmp_dir) / "text_prototypes.jsonl"
            prototypes.write_text(
                "\n".join(
                    [
                        json.dumps({"model": "qwen", "text": "这是一个中文模型生成的正式说明文本。"}, ensure_ascii=False),
                        json.dumps({"model": "llama", "text": "This is an English assistant style paragraph."}),
                    ]
                ),
                encoding="utf-8",
            )
            text_path = Path(tmp_dir) / "input.txt"
            text_path.write_text("这是一个中文模型生成的说明文本。", encoding="utf-8")
            engine = ProvenanceAttributionEngine(
                {
                    "enabled": True,
                    "text": {
                        "enabled": True,
                        "providers": {
                            "embedding_prototype": {
                                "enabled": True,
                                "provider": "embedding_prototype",
                                "prototypes_path": str(prototypes),
                                "min_chars": 1,
                            }
                        },
                    },
                }
            )

            result = engine.attribute(text_path, "text")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["top_k"][0]["model"], "qwen")

    def test_text_trained_classifier_branch_returns_top_k_candidates(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "text_source_classifier.joblib"
            pipeline = Pipeline(
                [
                    ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))),
                    ("clf", LogisticRegression(max_iter=200)),
                ]
            )
            pipeline.fit(
                [
                    "deepseek 风格的中文总结说明文本",
                    "deepseek 生成的结构化分析内容",
                    "qwen assistant response in english",
                    "qwen model writes concise english answer",
                ],
                ["deepseek", "deepseek", "qwen", "qwen"],
            )
            joblib.dump({"model": pipeline}, model_path)
            text_path = Path(tmp_dir) / "input.txt"
            text_path.write_text("deepseek 风格的中文分析文本", encoding="utf-8")
            engine = ProvenanceAttributionEngine(
                {
                    "enabled": True,
                    "text": {
                        "enabled": True,
                        "providers": {
                            "trained_classifier": {
                                "enabled": True,
                                "provider": "trained_classifier",
                                "model_path": str(model_path),
                            }
                        },
                    },
                }
            )

            result = engine.attribute(text_path, "text")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["top_k"][0]["model"], "deepseek")

    def test_image_prototype_branch_returns_top_k_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            from PIL import Image
            import imagehash

            image_path = Path(tmp_dir) / "input.png"
            Image.new("RGB", (32, 32), color=(255, 0, 0)).save(image_path)
            phash = str(imagehash.phash(Image.open(image_path)))
            prototypes = Path(tmp_dir) / "image_prototypes.jsonl"
            prototypes.write_text(
                json.dumps({"model": "stable-diffusion-xl", "phash": phash}) + "\n",
                encoding="utf-8",
            )
            engine = ProvenanceAttributionEngine(
                {
                    "enabled": True,
                    "image": {
                        "enabled": True,
                        "providers": {
                            "clip_prototype": {
                                "enabled": True,
                                "provider": "clip_prototype",
                                "prototypes_path": str(prototypes),
                            }
                        },
                    },
                }
            )

            result = engine.attribute(image_path, "image")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["top_k"][0]["model"], "stable-diffusion-xl")

    def test_image_trained_classifier_branch_returns_top_k_candidates(self) -> None:
        from PIL import Image
        from sklearn.linear_model import LogisticRegression

        with tempfile.TemporaryDirectory() as tmp_dir:
            red_path = Path(tmp_dir) / "red.png"
            blue_path = Path(tmp_dir) / "blue.png"
            Image.new("RGB", (32, 32), color=(255, 0, 0)).save(red_path)
            Image.new("RGB", (32, 32), color=(0, 0, 255)).save(blue_path)

            engine = ProvenanceAttributionEngine({"enabled": True})
            features = [
                engine._extract_image_features(red_path),
                engine._extract_image_features(blue_path),
                engine._extract_image_features(red_path),
                engine._extract_image_features(blue_path),
            ]
            classifier = LogisticRegression(max_iter=200)
            classifier.fit(features, ["red-generator", "blue-generator", "red-generator", "blue-generator"])

            model_path = Path(tmp_dir) / "image_source_classifier.joblib"
            joblib.dump({"model": classifier, "feature_type": "basic_image_stats"}, model_path)
            query_path = Path(tmp_dir) / "query.png"
            Image.new("RGB", (32, 32), color=(250, 0, 0)).save(query_path)
            engine = ProvenanceAttributionEngine(
                {
                    "enabled": True,
                    "image": {
                        "enabled": True,
                        "providers": {
                            "trained_classifier": {
                                "enabled": True,
                                "provider": "trained_classifier",
                                "model_path": str(model_path),
                            }
                        },
                    },
                }
            )

            result = engine.attribute(query_path, "image")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["top_k"][0]["model"], "red-generator")

    def test_image_training_decodes_classlabel_names(self) -> None:
        class FakeFeature:
            def int2str(self, value: int) -> str:
                return ["Real", "ADM", "BigGAN"][value]

        features = {"generator": FakeFeature()}

        self.assertEqual(decode_dataset_label(features, "generator", 2), "BigGAN")
        self.assertEqual(decode_dataset_label(features, "generator", "ADM"), "ADM")

    def test_text_training_merges_multiple_datasets(self) -> None:
        texts, labels = merge_labeled_samples(
            [
                (["a", "b"], ["model-a", "model-b"]),
                (["c"], ["model-c"]),
            ]
        )

        self.assertEqual(texts, ["a", "b", "c"])
        self.assertEqual(labels, ["model-a", "model-b", "model-c"])

    def test_mage_label_normalization_can_be_enabled(self) -> None:
        self.assertEqual(
            normalize_mage_label("cmv_machine_continuation_gpt-3.5-trubo", enabled=True),
            "gpt-3.5-turbo",
        )
        self.assertEqual(
            normalize_mage_label("cmv_machine_continuation_opt_350m", enabled=True),
            "opt_350m",
        )
        self.assertEqual(
            normalize_mage_label("xsum_human", enabled=True),
            "human",
        )


if __name__ == "__main__":
    unittest.main()
