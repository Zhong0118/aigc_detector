from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import Mock
from tempfile import TemporaryDirectory

from detection.providers import DetectionPackage, ProviderResult
from provenance.high_score_blocks import (
    collect_detection_segments,
    select_attribution_units,
    should_run_text_attribution,
    unit_weight,
)
from provenance.pipeline import ProvenancePipeline
from provenance.window_a0_fusion import (
    decide_family_status,
    load_branch_weights,
    mix_equal_probabilities,
)
from scripts.tune_window_a0_fusion_weights import _score_weights, _simplex
import numpy as np


def _seg(index: int, start: int, end: int, score: float, text: str) -> dict:
    return {
        "index": index,
        "start": start,
        "end": end,
        "score": score,
        "text": text,
    }


class HighScoreBlockSelectionTests(unittest.TestCase):
    def test_prefers_high_threshold_blocks_and_fills_from_low(self) -> None:
        segments = [
            _seg(0, 0, 10, 0.20, "humanish"),
            _seg(1, 10, 20, 0.90, "high-a"),
            _seg(2, 20, 30, 0.40, "mid"),
            _seg(3, 40, 50, 0.95, "high-b"),
        ]
        units = select_attribution_units(
            segments,
            low_threshold=0.35,
            high_threshold=0.88,
            min_blocks=3,
            max_blocks=4,
        )
        indices = [idx for unit in units for idx in unit["block_indices"]]
        self.assertEqual(indices, [1, 2, 3])
        self.assertEqual(len(units), 2)
        self.assertEqual(units[0]["block_indices"], [1, 2])
        self.assertEqual(units[1]["block_indices"], [3])

    def test_does_not_glue_non_adjacent_high_blocks(self) -> None:
        segments = [
            _seg(0, 0, 10, 0.91, "aaa"),
            _seg(1, 20, 30, 0.92, "bbb"),
        ]
        units = select_attribution_units(segments, min_blocks=1, max_blocks=8)
        self.assertEqual([unit["block_indices"] for unit in units], [[0], [1]])
        self.assertEqual(units[0]["text"], "aaa")
        self.assertEqual(units[1]["text"], "bbb")

    def test_merges_only_overlapping_or_touching_windows(self) -> None:
        segments = [
            _seg(0, 0, 12, 0.91, "abcdefghijkl"),
            _seg(1, 8, 20, 0.93, "ijklxxxxxxxx"),
        ]
        units = select_attribution_units(segments, min_blocks=1, max_blocks=8)
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0]["start"], 0)
        self.assertEqual(units[0]["end"], 20)
        self.assertEqual(units[0]["text"], "abcdefghijklxxxxxxxx")

    def test_weight_is_length_times_detection_score(self) -> None:
        unit = {"char_len": 10, "detection_score": 0.5}
        self.assertEqual(unit_weight(unit), 5.0)

    def test_gate_allows_uncertain_document_with_enough_high_blocks(self) -> None:
        segments = [
            _seg(0, 0, 10, 0.2, "a"),
            _seg(1, 10, 20, 0.9, "b"),
            _seg(2, 20, 30, 0.91, "c"),
        ]
        self.assertTrue(
            should_run_text_attribution(
                document_score=0.32,
                segments=segments,
                high_threshold=0.88,
            )
        )
        self.assertFalse(
            should_run_text_attribution(
                document_score=0.32,
                segments=[
                    _seg(0, 0, 40, 0.40, "mid-a"),
                    _seg(1, 40, 80, 0.41, "mid-b"),
                    _seg(2, 80, 120, 0.20, "low"),
                ],
                high_threshold=0.88,
            )
        )

    def test_short_single_block_uses_low_threshold_not_high_risk_label(self) -> None:
        self.assertTrue(
            should_run_text_attribution(
                document_score=0.4,
                segments=[_seg(0, 0, 80, 0.4, "short")],
                high_threshold=0.88,
            )
        )


class PipelineBlockGateTests(unittest.TestCase):
    def test_pipeline_runs_text_attribution_on_high_blocks_below_deep_score(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            text_path = Path(tmp_dir) / "input.txt"
            text_path.write_text("x" * 40, encoding="utf-8")
            pipeline = ProvenancePipeline(
                deep_threshold=0.6,
                attribution_config={"enabled": True, "text": {"enabled": True}},
            )
            pipeline.attributor.attribute = Mock(
                return_value={"status": "ok", "top_k": [{"model": "qwen", "probability": 0.4}]}
            )
            detection = DetectionPackage(
                score=0.32,
                label="uncertain",
                threshold=0.88,
                provider_results=[
                    ProviderResult(
                        provider="local-text",
                        provider_type="local",
                        status="ok",
                        score=0.32,
                        label="uncertain",
                        details={
                            "segments": [
                                _seg(0, 0, 20, 0.91, "a" * 20),
                                _seg(1, 20, 40, 0.90, "b" * 20),
                            ]
                        },
                    )
                ],
                modality_scores={"text": 0.32},
                model_scores={},
                decision_details={
                    "low_threshold": 0.35,
                    "high_threshold": 0.88,
                    "segments": [
                        _seg(0, 0, 20, 0.91, "a" * 20),
                        _seg(1, 20, 40, 0.90, "b" * 20),
                    ],
                },
            )
            result = pipeline.analyze(text_path, "text", detection, fingerprint="x")

        self.assertFalse(result["deep_triggered"])
        self.assertTrue(result["attribution_triggered"])
        self.assertEqual(result["attribution"]["status"], "ok")
        pipeline.attributor.attribute.assert_called_once()
        args, kwargs = pipeline.attributor.attribute.call_args
        self.assertEqual(args[0], text_path)
        self.assertEqual(args[1], "text")
        self.assertIs(kwargs["detection"], detection)

    def test_collect_segments_from_package(self) -> None:
        detection = DetectionPackage(
            score=0.1,
            label="human",
            threshold=0.8,
            provider_results=[],
            modality_scores={"text": 0.1},
            model_scores={},
            decision_details={"segments": [_seg(0, 0, 4, 0.9, "abcd")]},
        )
        self.assertEqual(len(collect_detection_segments(detection)), 1)


class EqualWeightMixTests(unittest.TestCase):
    def test_four_branch_equal_mix_matches_offline_protocol(self) -> None:
        labels = ("openai", "qwen")
        mixed = mix_equal_probabilities(
            {
                "A0": {"openai": 0.8, "qwen": 0.2},
                "B0": {"openai": 0.4, "qwen": 0.6},
                "B1": {"openai": 0.2, "qwen": 0.8},
                "B3": {"openai": 0.6, "qwen": 0.4},
            },
            labels,
            {"A0": 0.25, "B0": 0.25, "B1": 0.25, "B3": 0.25},
        )
        self.assertAlmostEqual(mixed["openai"], 0.5)
        self.assertAlmostEqual(mixed["qwen"], 0.5)

    def test_unknown_only_when_top1_is_below_floor(self) -> None:
        labels = ("openai", "qwen", "glm")
        mid = decide_family_status(np.array([0.3211, 0.2287, 0.0822]), labels)
        self.assertEqual(mid["status"], "family_only")
        self.assertEqual(mid["predicted_family"], "openai")
        self.assertEqual(mid["top_k"][0]["model"], "openai")
        self.assertNotEqual(mid["top_k"][0]["model"], "unknown")
        weak = decide_family_status(np.array([0.18, 0.17, 0.16]), labels)
        self.assertEqual(weak["status"], "unknown")
        self.assertEqual(weak["predicted_family"], "unknown")
        self.assertEqual(weak["top_k"][0]["model"], "openai")
        clear = decide_family_status(np.array([0.70, 0.20, 0.10]), labels)
        self.assertEqual(clear["status"], "family_only")
        self.assertEqual(clear["predicted_family"], "openai")

    def test_explicit_weights_normalize_and_simplex_contains_equal(self) -> None:
        weights = load_branch_weights({"A0": 2, "B0": 1, "B1": 1, "B3": 0})
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertAlmostEqual(weights["A0"], 0.5)
        grid = _simplex(0.25)
        self.assertIn({"A0": 0.25, "B0": 0.25, "B1": 0.25, "B3": 0.25}, grid)
        labels = ("openai", "qwen")
        branches = {
            "A0": np.array([[0.9, 0.1], [0.2, 0.8]]),
            "B0": np.array([[0.1, 0.9], [0.2, 0.8]]),
            "B1": np.array([[0.1, 0.9], [0.2, 0.8]]),
            "B3": np.array([[0.1, 0.9], [0.2, 0.8]]),
        }
        true = ["openai", "qwen"]
        a0_only = _score_weights(branches, true, labels, {"A0": 1, "B0": 0, "B1": 0, "B3": 0})
        b_only = _score_weights(branches, true, labels, {"A0": 0, "B0": 1, "B1": 0, "B3": 0})
        self.assertGreater(a0_only["macro_f1"], b_only["macro_f1"])


if __name__ == "__main__":
    unittest.main()
