from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError, dataclass, field
from pathlib import Path

from detection.false_positive_controller import FalsePositiveController
from detection.text_domain import DomainDecision, TextDomainRouter
from detection.text_segmentation import SentenceAwareSegmenter


@dataclass
class FakeProviderResult:
    provider: str = "local_text_yuchuan"
    status: str = "ok"
    score: float | None = None
    details: dict = field(default_factory=dict)


class SentenceAwareSegmenterTests(unittest.TestCase):
    def test_segments_keep_source_coordinates_and_sentence_boundaries(self) -> None:
        text = "第一句用于测试。第二句继续补充内容！第三句给出结论？第四句收尾。"
        segmenter = SentenceAwareSegmenter(target_chars=18, max_chars=28, overlap_sentences=1)

        segments = segmenter.split(text)

        self.assertGreaterEqual(len(segments), 2)
        for segment in segments:
            self.assertEqual(text[segment.start : segment.end], segment.text)
            self.assertLessEqual(len(segment.text), 28)
        self.assertTrue(segments[0].text.endswith(("。", "！", "？")))
        self.assertLess(segments[1].start, segments[0].end)

    def test_short_text_is_returned_as_one_segment(self) -> None:
        text = "这是一段较短但完整的文本。"
        segments = SentenceAwareSegmenter(target_chars=100, max_chars=120).split(text)

        self.assertEqual(len(segments), 1)
        self.assertEqual((segments[0].start, segments[0].end, segments[0].text), (0, len(text), text))

    def test_oversized_next_sentence_does_not_drop_the_document_tail(self) -> None:
        text = ("甲" * 39 + "。") + ("乙" * 39 + "。") + ("丙" * 39 + "。")
        segmenter = SentenceAwareSegmenter(target_chars=60, max_chars=70, overlap_sentences=1)

        segments = segmenter.split(text)
        covered = set()
        for segment in segments:
            covered.update(range(segment.start, segment.end))

        self.assertEqual(covered, set(range(len(text))))
        self.assertEqual(segments[-1].end, len(text))


class TextDomainRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.router = TextDomainRouter()

    def test_domain_decision_is_immutable(self) -> None:
        decision = DomainDecision("general_qa", 0.25, ())

        with self.assertRaises(FrozenInstanceError):
            decision.domain = "literature"  # type: ignore[misc]

    def test_multiple_academic_markers_outweigh_novel_overlap(self) -> None:
        decision = self.router.classify("We propose a novel method for neural text analysis.")

        self.assertEqual(decision.domain, "academic_technical")
        self.assertIn("method", decision.evidence)
        self.assertIn("neural", decision.evidence)

    def test_novel_method_is_academic_without_literary_context(self) -> None:
        decision = self.router.classify("A novel method.")

        self.assertEqual(decision.domain, "academic_technical")
        self.assertIn("novel method", decision.evidence)

    def test_novel_with_literary_evidence_remains_literature(self) -> None:
        decision = self.router.classify(
            "The novel follows its protagonist through the narrator's recollections."
        )

        self.assertEqual(decision.domain, "literature")
        self.assertIn("novel", decision.evidence)

    def test_equal_cross_domain_evidence_uses_explicit_priority_before_confidence(self) -> None:
        decision = self.router.classify("The protagonist describes a method.")

        self.assertEqual(decision.domain, "academic_technical")
        self.assertEqual(decision.confidence, 0.8)

    def test_routes_english_academic_study(self) -> None:
        decision = self.router.classify(
            "The study evaluates a neural algorithm on a public dataset and reports the results."
        )

        self.assertEqual(decision.domain, "academic_technical")
        self.assertGreater(decision.confidence, 0.5)
        self.assertIn("study", decision.evidence)

    def test_routes_english_government_announcement(self) -> None:
        decision = self.router.classify(
            "The government announced the new public policy in an official statement."
        )

        self.assertEqual(decision.domain, "official_news")
        self.assertIn("government", decision.evidence)

    def test_preserves_existing_chinese_domain_examples(self) -> None:
        examples = {
            "宝玉听了，忙问缘故，黛玉只低头拭泪。": "literature",
            "本文研究一种神经网络算法，实验结果表明该方法有效。": "academic_technical",
            "根据有关规定，政府现将年度工作安排公告如下。": "official_news",
        }

        self.assertEqual(
            {text: self.router.classify(text).domain for text in examples},
            examples,
        )

    def test_precedence_is_deterministic_when_markers_overlap(self) -> None:
        decision = self.router.classify("政府公告：研究部门将发布新的数据集管理办法。")

        self.assertEqual(decision.domain, "academic_technical")
        self.assertIn("研究", decision.evidence)
        self.assertEqual(decision.confidence, 0.8)

    def test_defaults_to_general_qa_with_rule_based_fallback_evidence(self) -> None:
        decision = self.router.classify("How can I make a quick dinner tonight?")

        self.assertEqual(decision.domain, "general_qa")
        self.assertLess(decision.confidence, 0.5)
        self.assertEqual(decision.evidence, ())


class FalsePositiveControllerTests(unittest.TestCase):
    def test_missing_detector_scores_returns_insufficient_evidence(self) -> None:
        controller = FalsePositiveController({"min_effective_chars": 8})

        decision = controller.evaluate("这是一段足够长的待检测文本。", [])

        self.assertEqual(decision.decision, "insufficient_evidence")
        self.assertIsNone(decision.score)
        self.assertIn("no_usable_detector_score", decision.uncertainty_reasons)

    def test_middle_score_is_uncertain_instead_of_human(self) -> None:
        controller = FalsePositiveController(
            {"min_effective_chars": 8, "default_low_threshold": 0.4, "default_high_threshold": 0.8}
        )

        decision = controller.evaluate(
            "这是一段结构清晰但证据并不充分的普通说明文本。",
            [FakeProviderResult(score=0.62)],
        )

        self.assertEqual(decision.decision, "uncertain")
        self.assertAlmostEqual(decision.score or 0.0, 0.62)
        self.assertEqual(decision.low_threshold, 0.4)
        self.assertEqual(decision.high_threshold, 0.8)

    def test_calibrated_literature_threshold_prevents_global_false_positive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "thresholds.json"
            artifact.write_text(
                json.dumps(
                    {
                        "version": "test-v1",
                        "groups": {
                            "zh:literature:short": {
                                "low_threshold": 0.35,
                                "high_threshold": 0.93,
                                "human_samples": 300,
                                "validation_samples": 100,
                                "validation_fpr": 0.01,
                                "reliability_status": "warning",
                                "warnings": ["calibrated_threshold_near_score_ceiling"],
                            },
                            "global": {
                                "low_threshold": 0.4,
                                "high_threshold": 0.8,
                                "human_samples": 1000,
                            },
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            controller = FalsePositiveController(
                {
                    "artifact_path": str(artifact),
                    "min_effective_chars": 8,
                    "short_text_chars": 200,
                }
            )

            decision = controller.evaluate(
                "宝玉听了，忙问缘故，黛玉只低头拭泪，不肯言语。",
                [FakeProviderResult(score=0.88)],
            )

        self.assertEqual(decision.domain, "literature")
        self.assertEqual(decision.threshold_group, "zh:literature:short")
        self.assertEqual(decision.decision, "uncertain")
        self.assertEqual(decision.artifact_version, "test-v1")
        self.assertEqual(decision.threshold_validation_samples, 100)
        self.assertEqual(decision.threshold_validation_fpr, 0.01)
        self.assertEqual(decision.threshold_reliability_status, "warning")
        self.assertEqual(
            decision.threshold_warnings,
            ["calibrated_threshold_near_score_ceiling"],
        )

    def test_uncalibrated_domain_uses_its_configured_fallback_not_global_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "thresholds.json"
            artifact.write_text(
                json.dumps(
                    {
                        "version": "test-v1",
                        "groups": {
                            "zh": {"low_threshold": 0.4, "high_threshold": 0.97},
                            "global": {"low_threshold": 0.4, "high_threshold": 0.98},
                        },
                    }
                ),
                encoding="utf-8",
            )
            controller = FalsePositiveController(
                {
                    "artifact_path": str(artifact),
                    "min_effective_chars": 8,
                    "domain_thresholds": {
                        "general_qa": {"low_threshold": 0.4, "high_threshold": 0.8}
                    },
                }
            )

            decision = controller.evaluate(
                "这是一段普通写作内容，用于检查领域阈值回退是否正确。",
                [FakeProviderResult(score=0.9)],
            )

        self.assertEqual(decision.threshold_group, "built-in:general_qa:short")
        self.assertEqual(decision.high_threshold, 0.8)
        self.assertEqual(decision.decision, "high_risk_ai")

    def test_long_document_requires_distributed_segment_evidence(self) -> None:
        controller = FalsePositiveController(
            {
                "min_effective_chars": 8,
                "short_text_chars": 20,
                "default_high_threshold": 0.75,
                "min_high_coverage": 0.3,
                "min_adjacent_high_segments": 2,
            }
        )
        provider = FakeProviderResult(
            score=0.86,
            details={
                "segments": [
                    {"index": 0, "start": 0, "end": 20, "score": 0.95},
                    {"index": 1, "start": 20, "end": 40, "score": 0.42},
                    {"index": 2, "start": 40, "end": 60, "score": 0.41},
                ]
            },
        )

        decision = controller.evaluate("甲" * 60, [provider])

        self.assertEqual(decision.decision, "uncertain")
        self.assertIn("isolated_high_segment", decision.uncertainty_reasons)

    def test_two_adjacent_high_segments_can_produce_high_risk(self) -> None:
        controller = FalsePositiveController(
            {
                "min_effective_chars": 8,
                "short_text_chars": 20,
                "default_high_threshold": 0.75,
                "min_high_coverage": 0.3,
                "min_adjacent_high_segments": 2,
            }
        )
        provider = FakeProviderResult(
            score=0.9,
            details={
                "segments": [
                    {"index": 0, "start": 0, "end": 20, "score": 0.91},
                    {"index": 1, "start": 20, "end": 40, "score": 0.89},
                    {"index": 2, "start": 40, "end": 60, "score": 0.35},
                ]
            },
        )

        decision = controller.evaluate("这是普通说明文字。" * 8, [provider])

        self.assertEqual(decision.decision, "high_risk_ai")
        self.assertGreaterEqual(decision.high_risk_coverage, 0.3)

    def test_single_full_coverage_high_segment_can_produce_high_risk(self) -> None:
        controller = FalsePositiveController(
            {
                "min_effective_chars": 8,
                "short_text_chars": 20,
                "default_high_threshold": 0.75,
                "min_analyzed_coverage": 0.7,
                "min_high_coverage": 0.3,
                "min_adjacent_high_segments": 2,
            }
        )
        provider = FakeProviderResult(
            score=0.99,
            details={
                "segments": [
                    {"index": 0, "start": 0, "end": 60, "score": 0.99},
                ]
            },
        )

        decision = controller.evaluate("甲" * 60, [provider])

        self.assertEqual(decision.decision, "high_risk_ai")
        self.assertEqual(decision.adjacent_high_segments, 1)
        self.assertEqual(decision.analyzed_coverage, 1.0)


if __name__ == "__main__":
    unittest.main()
