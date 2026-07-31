from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from research.text_attribution.eda import analyze_jsonl, write_eda_artifacts


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _row(
    *,
    family: str,
    model: str,
    prompt_id: str,
    language: str,
    target_length: str,
    text: str,
    finish_reason: str = "stop",
    elapsed_seconds: float = 1.0,
    completion_tokens: int = 10,
    reasoning_tokens: int = 0,
) -> dict:
    return {
        "family_id": family,
        "request_model": model,
        "text": text,
        "finish_reason": finish_reason,
        "elapsed_seconds": elapsed_seconds,
        "prompt": {
            "prompt_id": prompt_id,
            "language": language,
            "target_length": target_length,
            "task_type": "explanation",
        },
        "parameters": {"temperature": 0.0},
        "usage": {
            "completion_tokens": completion_tokens,
            "completion_tokens_details": {
                "reasoning_tokens": reasoning_tokens,
            },
        },
    }


class AttributionEdaTests(unittest.TestCase):
    def test_cli_config_resolves_full_expected_model_universe(self) -> None:
        from scripts.analyze_attribution_pilot import _resolve_expected_universe

        expected_per_model, expected_models = _resolve_expected_universe(
            PROJECT_ROOT
            / "configs"
            / "text_attribution"
            / "runs"
            / "server_600_v2.json",
            None,
        )

        self.assertEqual(expected_per_model, 30)
        self.assertEqual(len(expected_models), 20)
        self.assertEqual(expected_models["deepseek-v4-pro"], "deepseek")

    def test_expected_model_universe_counts_models_with_zero_successes(self) -> None:
        rows = [
            _row(
                family="alpha",
                model="alpha-1",
                prompt_id="zh-1",
                language="zh",
                target_length="short",
                text="示例文本",
            ),
            _row(
                family="beta",
                model="beta-1",
                prompt_id="en-1",
                language="en",
                target_length="medium",
                text="Example text.",
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "missing-model.jsonl"
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )

            analysis = analyze_jsonl(
                path,
                expected_per_model=2,
                expected_models={
                    "alpha-1": "alpha",
                    "alpha-2": "alpha",
                    "beta-1": "beta",
                },
            )

        self.assertEqual(analysis["summary"]["models"], 3)
        self.assertEqual(analysis["summary"]["expected_total"], 6)
        self.assertEqual(analysis["summary"]["missing_to_expected"], 4)
        self.assertEqual(analysis["by_family"]["alpha"]["records"], 1)
        self.assertEqual(analysis["by_family"]["alpha"]["missing_to_expected"], 3)
        self.assertEqual(analysis["by_family"]["alpha"]["coverage_rate"], 0.25)
        self.assertEqual(analysis["by_family"]["beta"]["missing_to_expected"], 1)
        self.assertEqual(analysis["by_family"]["beta"]["coverage_rate"], 0.5)

    def test_analyze_jsonl_reports_quality_length_and_latency(self) -> None:
        rows = [
            _row(
                family="alpha",
                model="alpha-1",
                prompt_id="zh-1",
                language="zh",
                target_length="short",
                text="天空很蓝",
                finish_reason="length",
                elapsed_seconds=2.0,
                completion_tokens=12,
            ),
            _row(
                family="alpha",
                model="alpha-1",
                prompt_id="zh-1",
                language="zh",
                target_length="short",
                text="天空很蓝",
                elapsed_seconds=4.0,
                completion_tokens=14,
            ),
            _row(
                family="beta",
                model="beta-1",
                prompt_id="en-1",
                language="en",
                target_length="medium",
                text="The sky is blue.",
                elapsed_seconds=6.0,
                completion_tokens=20,
                reasoning_tokens=5,
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pilot.jsonl"
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )

            analysis = analyze_jsonl(path, expected_per_model=4)

        self.assertEqual(analysis["summary"]["records"], 3)
        self.assertEqual(analysis["summary"]["duplicate_excess_count"], 1)
        self.assertEqual(
            analysis["summary"]["low_temperature_only_duplicate_excess_count"],
            1,
        )
        self.assertEqual(analysis["summary"]["truncated_count"], 1)
        self.assertEqual(analysis["summary"]["collection_duration_seconds"], None)
        self.assertEqual(analysis["by_family"]["alpha"]["records"], 2)
        self.assertEqual(analysis["by_family"]["alpha"]["missing_to_expected"], 2)
        self.assertEqual(analysis["by_family"]["alpha"]["truncated_rate"], 0.5)
        self.assertEqual(analysis["by_family"]["alpha"]["latency_p50_seconds"], 3.0)
        self.assertEqual(analysis["by_family"]["beta"]["reasoning_token_rows"], 1)
        self.assertEqual(analysis["by_prompt_family"]["zh-1"]["alpha"], 2)
        self.assertEqual(analysis["length_by_language"]["zh"]["median"], 4.0)
        self.assertEqual(analysis["length_by_language"]["en"]["median"], 4.0)

    def test_analyze_jsonl_rejects_invalid_rows_with_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "broken.jsonl"
            path.write_text('{"text": "ok"}\nnot-json\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, r"broken\.jsonl:2"):
                analyze_jsonl(path)

    def test_analyze_jsonl_flags_language_mismatch_and_terminal_failures(self) -> None:
        rows = [
            _row(
                family="alpha",
                model="alpha-1",
                prompt_id="en-1",
                language="en",
                target_length="long",
                text="这是一段完全使用中文生成的回答，没有按照英文提示词作答。",
                finish_reason="max_tokens",
            ),
            _row(
                family="beta",
                model="beta-1",
                prompt_id="zh-1",
                language="zh",
                target_length="medium",
                text="无法提供相关内容",
                finish_reason="content_filter",
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "quality.jsonl"
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )

            analysis = analyze_jsonl(path)

        self.assertEqual(analysis["summary"]["truncated_count"], 1)
        self.assertEqual(analysis["summary"]["content_filtered_count"], 1)
        self.assertEqual(analysis["summary"]["language_mismatch_count"], 1)
        self.assertEqual(analysis["by_family"]["alpha"]["language_mismatch_count"], 1)

    def test_write_eda_artifacts_creates_report_and_figures(self) -> None:
        rows = [
            _row(
                family="alpha",
                model="alpha-1",
                prompt_id="zh-1",
                language="zh",
                target_length="short",
                text="示例文本",
            ),
            _row(
                family="beta",
                model="beta-1",
                prompt_id="en-1",
                language="en",
                target_length="medium",
                text="Example text for analysis.",
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "pilot.jsonl"
            output_dir = root / "report"
            input_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )

            artifacts = write_eda_artifacts(
                input_path,
                output_dir,
                expected_per_model=4,
            )

            self.assertTrue(Path(artifacts["analysis_json"]).is_file())
            self.assertTrue(Path(artifacts["report_markdown"]).is_file())
            self.assertEqual(len(artifacts["figures"]), 4)
            for figure in artifacts["figures"]:
                self.assertTrue(Path(figure).is_file())
                self.assertGreater(Path(figure).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
