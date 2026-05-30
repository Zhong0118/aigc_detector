from __future__ import annotations

import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

from detection.providers import ApiFirstDetectionEngine
from reports.generator import ReportGenerator


class ReportGeneratorLlmTests(unittest.TestCase):
    def test_llm_report_uses_openai_compatible_chat_completion(self) -> None:
        config = {
            "report": {
                "provider": "llm",
                "llm_provider": "deepseek",
                "api_key_env": "DEEPSEEK_API_KEY",
                "base_url": "https://api.deepseek.com",
                "model": "deepseek-v4-flash",
            }
        }
        analysis = {
            "filename": "demo.txt",
            "modality": "text",
            "detection": {
                "score": 0.91,
                "label": "ai",
                "providers": [{"provider": "hive", "status": "ok", "score": 0.91}],
                "model_scores": {"gpt-family": 0.83},
            },
            "provenance": {"deep_triggered": True, "c2pa": {}, "watermark": {}},
        }

        response = Mock()
        response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"summary":"疑似 AI 生成文本","evidence":["Hive 高分"],"limitations":["需要人工复核"],"recommendation":"进入溯源复核"}'
                    }
                }
            ]
        }
        response.raise_for_status.return_value = None

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=False):
            with patch("reports.generator.requests.post", return_value=response) as post:
                report = ReportGenerator(config=config).generate(analysis)

        self.assertEqual(report["provider"], "deepseek")
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["summary"], "疑似 AI 生成文本")
        post.assert_called_once()
        self.assertEqual(post.call_args.kwargs["json"]["model"], "deepseek-v4-flash")
        self.assertEqual(
            post.call_args.args[0],
            "https://api.deepseek.com/chat/completions",
        )

    def test_llm_report_accepts_direct_api_key_from_config(self) -> None:
        config = {
            "report": {
                "provider": "llm",
                "llm_provider": "deepseek",
                "api_key": "direct-test-key",
                "base_url": "https://api.deepseek.com",
                "model": "deepseek-v4-flash",
            }
        }
        analysis = {
            "filename": "demo.txt",
            "modality": "text",
            "detection": {
                "score": 0.91,
                "label": "ai",
                "providers": [{"provider": "hive", "status": "ok", "score": 0.91}],
                "model_scores": {},
            },
            "provenance": {"deep_triggered": True, "c2pa": {}, "watermark": {}},
        }

        response = Mock()
        response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"summary":"直接配置 key 可用","evidence":[],"limitations":[],"recommendation":"继续测试"}'
                    }
                }
            ]
        }
        response.raise_for_status.return_value = None

        with patch("reports.generator.requests.post", return_value=response) as post:
            report = ReportGenerator(config=config).generate(analysis)

        self.assertEqual(report["status"], "ok")
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer direct-test-key")

    def test_llm_prompt_uses_provider_neutral_detection_context(self) -> None:
        config = {
            "report": {
                "provider": "llm",
                "llm_provider": "deepseek",
                "api_key": "direct-test-key",
                "base_url": "https://api.deepseek.com",
                "model": "deepseek-v4-flash",
            }
        }
        analysis = {
            "filename": "demo.png",
            "modality": "image",
            "detection": {
                "score": 0.001,
                "label": "human",
                "threshold": 0.5,
                "providers": [
                    {
                        "provider": "hive",
                        "status": "ok",
                        "score": 0.0,
                        "details": {
                            "api_version": "v3_vlm",
                            "model": "hive/vision-language-model",
                            "explanation": "Hive says this looks like a real office photo.",
                        },
                    },
                    {
                        "provider": "sightengine",
                        "status": "ok",
                        "score": 0.001,
                        "details": {"raw": {"type": {"ai_generated": 0.001}}},
                    },
                ],
                "model_scores": {},
            },
            "provenance": {"deep_triggered": False, "c2pa": {}, "watermark": {}},
        }

        response = Mock()
        response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"summary":"多路检测结果显示风险较低","evidence":[],"limitations":[],"recommendation":"保留记录"}'
                    }
                }
            ]
        }
        response.raise_for_status.return_value = None

        with patch("reports.generator.requests.post", return_value=response) as post:
            ReportGenerator(config=config).generate(analysis)

        prompt_text = json.dumps(post.call_args.kwargs["json"]["messages"], ensure_ascii=False)
        self.assertIn("检测分支A", prompt_text)
        self.assertIn("检测分支B", prompt_text)
        self.assertNotIn("hive", prompt_text.lower())
        self.assertNotIn("sightengine", prompt_text.lower())
        self.assertNotIn("v3_vlm", prompt_text.lower())


class ApiProviderTests(unittest.TestCase):
    def test_local_text_detector_branch_uses_project_cache_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "detection": {
                            "threshold": 0.5,
                            "demo_provider_enabled": False,
                            "api_providers": {},
                            "local_models": {
                                "enabled": True,
                                "text_yuchuan": {
                                    "enabled": True,
                                    "cache_dir": "models/huggingface",
                                    "device": "cpu",
                                    "models": {
                                        "zh_long": "zh-long",
                                        "zh_short": "zh-short",
                                        "en_long": "en-long",
                                        "en_short": "en-short",
                                    },
                                },
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            text_path = Path(tmp_dir) / "input.txt"
            text_path.write_text("这是一段用于测试的中文文本。", encoding="utf-8")

            detector = Mock()
            detector.detect.return_value = Mock(
                score=0.73,
                label="ai",
                details={
                    "language": "zh",
                    "length_bucket": "short",
                    "selected_models": ["zh_short"],
                    "model_scores": {"local-text-zh-short": 0.73},
                },
            )

            with patch("detection.providers.RoutedTextAigcDetector", return_value=detector) as cls:
                package = ApiFirstDetectionEngine(config_path=str(config_path)).detect(
                    text_path, "text", "这是一段用于测试的中文文本。"
                )

            self.assertEqual(package.score, 0.73)
            self.assertEqual(package.label, "ai")
            self.assertEqual(package.provider_results[0].provider, "local_text_detector")
            self.assertEqual(package.provider_results[0].status, "ok")
            self.assertEqual(package.model_scores["local-text-zh-short"], 0.73)
            self.assertEqual(cls.call_args.kwargs["cache_dir"], "models/huggingface")

    def test_hive_text_provider_posts_text_data_and_normalizes_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "detection": {
                            "threshold": 0.5,
                            "demo_provider_enabled": False,
                            "api_providers": {
                                "hive": {
                                    "enabled": True,
                                    "api_key_env": "HIVE_API_KEY",
                                    "endpoint": "https://api.thehive.ai/api/v2/task/sync",
                                }
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            text_path = Path(tmp_dir) / "input.txt"
            text_path.write_text("hello aigc", encoding="utf-8")

            response = Mock()
            response.json.return_value = {
                "status": [
                    {
                        "response": {
                            "output": [
                                {
                                    "classes": [
                                        {"class": "human", "score": 0.08},
                                        {"class": "ai_generated", "score": 0.92},
                                    ]
                                }
                            ]
                        }
                    }
                ]
            }
            response.raise_for_status.return_value = None

            with patch.dict(os.environ, {"HIVE_API_KEY": "hive-key"}, clear=False):
                with patch("detection.providers.requests.post", return_value=response) as post:
                    package = ApiFirstDetectionEngine(config_path=str(config_path)).detect(
                        text_path, "text", "hello aigc"
                    )

            self.assertEqual(package.score, 0.92)
            self.assertEqual(package.label, "ai")
            self.assertEqual(package.provider_results[0].status, "ok")
            self.assertEqual(post.call_args.kwargs["json"]["models"], ["ai_generated_text"])
            self.assertEqual(post.call_args.kwargs["json"]["text_data"], "hello aigc")

    def test_hive_accepts_direct_api_key_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "detection": {
                            "threshold": 0.5,
                            "demo_provider_enabled": False,
                            "api_providers": {
                                "hive": {
                                    "enabled": True,
                                    "api_key": "direct-hive-key",
                                }
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            text_path = Path(tmp_dir) / "input.txt"
            text_path.write_text("hello aigc", encoding="utf-8")

            response = Mock()
            response.json.return_value = {"type": {"ai_generated": 0.66}}
            response.raise_for_status.return_value = None

            with patch("detection.providers.requests.post", return_value=response) as post:
                package = ApiFirstDetectionEngine(config_path=str(config_path)).detect(
                    text_path, "text", "hello aigc"
                )

            self.assertEqual(package.score, 0.66)
            self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "token direct-hive-key")

    def test_hive_v3_vlm_text_provider_uses_chat_completions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "detection": {
                            "threshold": 0.5,
                            "demo_provider_enabled": False,
                            "api_providers": {
                                "hive": {
                                    "enabled": True,
                                    "api_version": "v3_vlm",
                                    "api_key": "v3-secret",
                                }
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            text_path = Path(tmp_dir) / "input.txt"
            text_path.write_text("hello aigc", encoding="utf-8")

            response = Mock()
            response.json.return_value = {
                "choices": [
                    {
                        "message": {
                            "content": '{"ai_generated_score":0.81,"label":"ai_generated","explanation":"LLM-like phrasing","model_scores":{"gpt-family":0.7}}'
                        }
                    }
                ]
            }
            response.raise_for_status.return_value = None

            with patch("detection.providers.requests.post", return_value=response) as post:
                package = ApiFirstDetectionEngine(config_path=str(config_path)).detect(
                    text_path, "text", "hello aigc"
                )

            self.assertEqual(package.score, 0.81)
            self.assertEqual(package.label, "ai")
            self.assertEqual(package.model_scores["gpt-family"], 0.7)
            self.assertEqual(
                post.call_args.args[0],
                "https://api.thehive.ai/api/v3/chat/completions",
            )
            self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer v3-secret")
            self.assertEqual(post.call_args.kwargs["json"]["model"], "hive/vision-language-model")

    def test_sightengine_image_provider_uploads_media_and_normalizes_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "detection": {
                            "threshold": 0.5,
                            "demo_provider_enabled": False,
                            "api_providers": {
                                "sightengine": {
                                    "enabled": True,
                                    "api_user_env": "SIGHTENGINE_API_USER",
                                    "api_secret_env": "SIGHTENGINE_API_SECRET",
                                }
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            image_path = Path(tmp_dir) / "image.jpg"
            image_path.write_bytes(b"fake image bytes")

            response = Mock()
            response.json.return_value = {
                "status": "success",
                "type": {"ai_generated": 0.87, "midjourney": 0.71},
            }
            response.raise_for_status.return_value = None

            env = {
                "SIGHTENGINE_API_USER": "user",
                "SIGHTENGINE_API_SECRET": "secret",
            }
            with patch.dict(os.environ, env, clear=False):
                with patch("detection.providers.requests.post", return_value=response) as post:
                    package = ApiFirstDetectionEngine(config_path=str(config_path)).detect(
                        image_path, "image"
                    )

            self.assertEqual(package.score, 0.87)
            self.assertEqual(package.model_scores["midjourney"], 0.71)
            self.assertEqual(package.provider_results[0].status, "ok")
            self.assertEqual(post.call_args.args[0], "https://api.sightengine.com/1.0/check.json")
            self.assertEqual(post.call_args.kwargs["data"]["models"], "genai")

    def test_sightengine_accepts_direct_credentials_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "detection": {
                            "threshold": 0.5,
                            "demo_provider_enabled": False,
                            "api_providers": {
                                "sightengine": {
                                    "enabled": True,
                                    "api_user": "direct-user",
                                    "api_secret": "direct-secret",
                                }
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            image_path = Path(tmp_dir) / "image.jpg"
            image_path.write_bytes(b"fake image bytes")

            response = Mock()
            response.json.return_value = {"type": {"ai_generated": 0.74}}
            response.raise_for_status.return_value = None

            with patch("detection.providers.requests.post", return_value=response) as post:
                package = ApiFirstDetectionEngine(config_path=str(config_path)).detect(
                    image_path, "image"
                )

            self.assertEqual(package.score, 0.74)
            self.assertEqual(post.call_args.kwargs["data"]["api_user"], "direct-user")
            self.assertEqual(post.call_args.kwargs["data"]["api_secret"], "direct-secret")


if __name__ == "__main__":
    unittest.main()
