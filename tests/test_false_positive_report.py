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
    def test_calibrated_review_cutoff_applies_across_old_length_and_domain_boundaries(self):
        reporter = ReportGenerator(config={"report": {"provider": "template"}})
        for branch, domain, chars, cutoff, score, expected in (
            ("zh_short", "general_qa", 499, 0.94, 0.92, False),
            ("zh_long", "official_news", 1499, 0.85, 0.86, True),
            ("zh_long", "official_news", 1500, 0.85, 0.86, True),
            ("zh_long", "academic_technical", 3531, 0.85, 0.84, False),
            ("zh_long", "academic_technical", 3531, 0.85, 0.85, True),
        ):
            with self.subTest(branch=branch, domain=domain, chars=chars, score=score):
                analysis = analysis_for("uncertain", score, ["score_in_uncertain_interval"])
                detection = analysis["detection"]
                detection["decision"].update(domain=domain, threshold_group=branch, screening_threshold=cutoff)
                detection["providers"] = [{"status": "ok", "details": {
                    "selected_models": [branch], "screening_threshold": 0.8, "effective_chars": chars,
                    "segments": [{"score": score}],
                }}]
                report = reporter.generate(analysis)
                self.assertEqual(bool(report["review_label"]), expected)
                self.assertIn(f"{cutoff:.2%}", " ".join(report["evidence"]))

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

    def test_uncalibrated_scores_are_hidden_in_template_and_llm_reports(self):
        for reason in ("model_branch_not_calibrated", "scoring_protocol_not_evaluated"):
            for provider in ("template", "llm"):
                with self.subTest(reason=reason, provider=provider):
                    analysis = analysis_for("insufficient_evidence", 0.999, [reason])
                    reporter = ReportGenerator(config={"report": {"provider": provider}})
                    report = reporter.generate(analysis)
                    text = json.dumps(report, ensure_ascii=False)
                    self.assertIn("未校准 / 证据不足", report["summary"])
                    self.assertIn("不输出可比风险分", report["summary"])
                    self.assertNotIn("99.90%", text)
                    self.assertIsNone(report["review_label"])
                    self.assertEqual(analysis["detection"]["score"], 0.999)

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
