from __future__ import annotations

import json
import unittest

from reports.generator import ReportGenerator


def analysis_for(label: str, score: float, reasons: list[str]) -> dict:
    return {
        "filename": "demo.txt",
        "modality": "text",
        "detection": {
            "score": score,
            "label": label,
            "threshold": 0.9,
            "providers": [],
            "model_scores": {},
            "decision": {
                "score": None if label == "insufficient_evidence" else score,
                "decision": label,
                "low_threshold": 0.4,
                "high_threshold": 0.9,
                "threshold_group": "zh:literature:short",
                "domain": "literature",
                "analyzed_coverage": 1.0,
                "uncertainty_reasons": reasons,
            },
        },
        "provenance": {
            "deep_triggered": False,
            "c2pa": {},
            "watermark": {},
            "fingerprint_registry": {"status": "ok", "match_count": 0, "matches": []},
            "attribution": {},
        },
    }


class FalsePositiveReportTests(unittest.TestCase):
    def test_long_chinese_review_requires_document_or_window_evidence(self):
        config = {"report": {"provider": "template", "long_text_review": {
            "scoring_protocol": "review-fixture", "min_chars": 1500,
            "document_threshold": 0.9995, "segment_threshold": 0.9997,
            "min_high_segment_share": 0.5,
        }}}
        reporter = ReportGenerator(config=config)
        for domain, chars, protocol, score, scores, expected in (
            ("official_news", 3000, "review-fixture", 0.98, [0.9997, 0.96, 0.98], False),
            ("official_news", 3000, "review-fixture", 0.98, [0.9997, 0.9998, 0.95, 0.94], True),
            ("official_news", 3000, "review-fixture", 0.9995, [0.9995, 0.9995], True),
            ("academic_technical", 3531, "review-fixture", 0.878, [0.5235, 0.9994], True),
            ("official_news", 800, "review-fixture", 0.98, [0.97, 0.99], True),
            ("official_news", 3000, "different-model", 0.98, [0.97, 0.99], True),
        ):
            with self.subTest(domain=domain, chars=chars, protocol=protocol, score=score, scores=scores):
                analysis = analysis_for("uncertain", score, ["score_in_uncertain_interval"])
                detection = analysis["detection"]
                detection["decision"]["domain"] = domain
                detection["providers"] = [{"status": "ok", "details": {
                    "selected_models": ["zh_long"], "screening_threshold": 0.8,
                    "scoring_protocol": protocol, "effective_chars": chars,
                    "segments": [{"score": value} for value in scores],
                }}]
                report = reporter.generate(analysis)
                self.assertEqual(bool(report["review_label"]), expected)

    def test_uncertain_report_keeps_high_scoring_segment_clues_visible(self):
        analysis = analysis_for("uncertain", 0.878, ["score_in_uncertain_interval"])
        analysis["detection"]["providers"] = [{"status": "ok", "details": {
            "screening_threshold": 0.8,
            "segments": [{"score": 0.95}, {"score": 0.2}, {"score": 0.99}],
        }}]
        report = ReportGenerator(config={"report": {"provider": "template"}}).generate(analysis)
        evidence = " ".join(report["evidence"])
        self.assertIn("2/3", evidence)
        self.assertIn("80.00%", evidence)
        self.assertIn("复核线索", evidence)
        self.assertIn("不确定", report["summary"])
        self.assertEqual(report["review_label"], "疑似 AI，需复核")

    def test_uncertain_report_explains_calibrated_interval(self) -> None:
        report = ReportGenerator(config={"report": {"provider": "template"}}).generate(
            analysis_for("uncertain", 0.72, ["score_in_uncertain_interval"])
        )
        joined = json.dumps(report, ensure_ascii=False)

        self.assertIn("不确定", report["summary"])
        self.assertIn("40.00%", joined)
        self.assertIn("90.00%", joined)
        self.assertIn("经典文学", joined)
        self.assertIn("AI 特征风险分数", joined)
        self.assertIn("不是文本由 AI 生成的概率", joined)
        self.assertNotIn("校准 AI 风险分数", joined)
        self.assertNotIn("检测分数属于概率证据", joined)
        self.assertIn("人工复核", report["recommendation"])

    def test_insufficient_report_does_not_call_zero_a_human_score(self) -> None:
        report = ReportGenerator(config={"report": {"provider": "template"}}).generate(
            analysis_for("insufficient_evidence", 0.0, ["no_usable_detector_score"])
        )
        joined = json.dumps(report, ensure_ascii=False)

        self.assertIn("证据不足", report["summary"])
        self.assertIn("没有可用的检测模型分数", joined)
        self.assertNotIn("未明显检出 AIGC", report["summary"])

    def test_report_explains_human_fpr_ai_recall_and_reliability_warnings(self) -> None:
        analysis = analysis_for("uncertain", 0.72, ["score_in_uncertain_interval"])
        analysis["detection"]["decision"].update(
            {
                "threshold_validation_fpr": 0.012,
                "threshold_validation_samples": 1000,
                "threshold_ai_recall": 0.84,
                "threshold_ai_validation_samples": 900,
                "threshold_validation_auroc": 0.91,
                "threshold_reliability_status": "warning",
                "threshold_warnings": [
                    "ai_recall_below_minimum",
                    "mixed_language_threshold_fallback",
                ],
            }
        )

        report = ReportGenerator(config={"report": {"provider": "template"}}).generate(
            analysis
        )
        joined = json.dumps(report, ensure_ascii=False)

        self.assertIn("真人验证集误报率为 1.20%", joined)
        self.assertIn("1000", joined)
        self.assertIn("AI 验证集召回率为 84.00%", joined)
        self.assertIn("900", joined)
        self.assertIn("AUROC 为 0.910", joined)
        self.assertIn("AI 验证召回率低于最低要求", joined)
        self.assertIn("混合语种未独立校准", joined)
        self.assertNotIn("校准 AI 风险分数", joined)
        self.assertNotIn("检测分数属于概率证据", joined)

    def test_report_explains_runtime_cutoff_has_no_measured_fpr(self) -> None:
        analysis = analysis_for("uncertain", 0.72, ["score_in_uncertain_interval"])
        analysis["detection"]["decision"].update(
            {
                "threshold_validation_fpr": None,
                "threshold_validation_samples": 1000,
                "threshold_ai_recall": None,
                "threshold_ai_validation_samples": 900,
                "threshold_validation_auroc": 0.91,
                "threshold_warnings": ["effective_high_threshold_not_evaluated"],
            }
        )
        report = ReportGenerator(config={"report": {"provider": "template"}}).generate(
            analysis
        )
        text = json.dumps(report, ensure_ascii=False)
        self.assertIn("当前运行阈值未完成独立验证", text)
        self.assertNotIn("真人验证集误报率为", text)
        self.assertNotIn("AI 验证集召回率为", text)
        self.assertIn("AUROC 为 0.910", text)

    def test_report_keeps_builtin_fallback_readable_without_validation_metrics(self) -> None:
        analysis = analysis_for("uncertain", 0.72, ["score_in_uncertain_interval"])
        analysis["detection"]["decision"]["threshold_group"] = (
            "built-in:general_qa:short"
        )

        report = ReportGenerator(config={"report": {"provider": "template"}}).generate(
            analysis
        )
        joined = json.dumps(report, ensure_ascii=False)

        self.assertIn("内置保守阈值", joined)
        self.assertNotIn("真人验证集误报率为 None", joined)
        self.assertNotIn("AI 验证集召回率为 None", joined)


if __name__ == "__main__":
    unittest.main()
