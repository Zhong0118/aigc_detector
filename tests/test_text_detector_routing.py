from __future__ import annotations

import unittest

from detection.text_detector import RoutedTextAigcDetector


class RoutedTextAigcDetectorTests(unittest.TestCase):
    def test_routes_short_and_long_text_by_language(self) -> None:
        detector = RoutedTextAigcDetector(
            models={
                "zh_long": "zh-long",
                "zh_short": "zh-short",
                "en_long": "en-long",
                "en_short": "en-short",
            },
            short_text_chars=10,
            cache_dir="models/huggingface",
            device="cpu",
        )

        self.assertEqual(detector.select_model_keys("这是一段中文"), ["zh_short"])
        self.assertEqual(detector.select_model_keys("这是一段比较长的中文文本"), ["zh_long"])
        self.assertEqual(detector.select_model_keys("short en"), ["en_short"])
        self.assertEqual(detector.select_model_keys("this is a longer English text"), ["en_long"])

    def test_routes_mixed_text_to_both_language_models(self) -> None:
        detector = RoutedTextAigcDetector(
            models={
                "zh_long": "zh-long",
                "zh_short": "zh-short",
                "en_long": "en-long",
                "en_short": "en-short",
            },
            short_text_chars=20,
            cache_dir="models/huggingface",
            device="cpu",
        )

        self.assertEqual(detector.select_model_keys("中文 mixed English"), ["zh_short", "en_short"])


if __name__ == "__main__":
    unittest.main()
