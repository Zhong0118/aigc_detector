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
    def test_uncertain_report_explains_calibrated_interval(self) -> None:
        report = ReportGenerator(config={"report": {"provider": "template"}}).generate(
            analysis_for("uncertain", 0.72, ["score_in_uncertain_interval"])
        )
        joined = json.dumps(report, ensure_ascii=False)

        self.assertIn("不确定", report["summary"])
        self.assertIn("40.00%", joined)
        self.assertIn("90.00%", joined)
        self.assertIn("经典文学", joined)
        self.assertIn("人工复核", report["recommendation"])

    def test_insufficient_report_does_not_call_zero_a_human_score(self) -> None:
        report = ReportGenerator(config={"report": {"provider": "template"}}).generate(
            analysis_for("insufficient_evidence", 0.0, ["no_usable_detector_score"])
        )
        joined = json.dumps(report, ensure_ascii=False)

        self.assertIn("证据不足", report["summary"])
        self.assertIn("没有可用的检测模型分数", joined)
        self.assertNotIn("未明显检出 AIGC", report["summary"])


if __name__ == "__main__":
    unittest.main()
