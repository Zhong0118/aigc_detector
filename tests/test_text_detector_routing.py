from __future__ import annotations

import unittest
from unittest.mock import patch

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

    def test_detect_records_sentence_aware_segment_scores(self) -> None:
        detector = RoutedTextAigcDetector(
            models={
                "zh_long": "zh-long",
                "zh_short": "zh-short",
                "en_long": "en-long",
                "en_short": "en-short",
            },
            short_text_chars=10,
            segment_target_chars=8,
            segment_max_chars=12,
            segment_overlap_sentences=0,
            cache_dir="models/huggingface",
            device="cpu",
        )
        text = "第一段有完整内容。第二段继续说明问题。第三段给出最终结论。"

        with patch.object(detector, "_run_model", side_effect=[0.9, 0.3, 0.6]) as run_model:
            result = detector.detect(text)

        self.assertEqual(run_model.call_count, 3)
        self.assertEqual(len(result.details["segments"]), 3)
        weights = [segment["end"] - segment["start"] for segment in result.details["segments"]]
        expected = sum(score * weight for score, weight in zip([0.9, 0.3, 0.6], weights)) / sum(weights)
        self.assertAlmostEqual(result.score, expected, places=3)
        for segment in result.details["segments"]:
            self.assertEqual(text[segment["start"] : segment["end"]], segment["text"])


if __name__ == "__main__":
    unittest.main()
