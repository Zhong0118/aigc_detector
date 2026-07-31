from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from research.text_attribution.baseline_data import (
    BaselineSample,
    assign_group_folds,
    load_clean_samples,
    semantic_prompt_group,
)


def _row(
    request_key: str,
    *,
    text: str,
    family: str = "openai",
    prompt_id: str = "zh-science-explain-001",
    language: str = "zh",
    finish_reason: str = "stop",
    route_mismatch: bool = False,
    temperature: float = 0.0,
) -> dict:
    return {
        "request_key": request_key,
        "family_id": family,
        "request_model": f"{family}-model",
        "text": text,
        "finish_reason": finish_reason,
        "route_mismatch": route_mismatch,
        "repeat_index": 0,
        "prompt": {
            "prompt_id": prompt_id,
            "language": language,
            "task_type": "explanation",
            "target_length": "medium",
        },
        "parameters": {"temperature": temperature},
    }


def _sample(label: str, group_index: int) -> BaselineSample:
    prompt_id = f"zh-task-{group_index:03d}"
    return BaselineSample(
        request_key=f"{label}-{group_index}",
        text=f"{label} 的测试文本 {group_index}",
        label=label,
        model=f"{label}-model",
        prompt_id=prompt_id,
        semantic_prompt_group=f"task-{group_index:03d}",
        language="zh",
        task_type="explanation",
        target_length="medium",
        temperature=0.0,
        repeat_index=0,
    )


class BaselineDataTests(unittest.TestCase):
    def test_cleaning_reports_every_exclusion_reason(self) -> None:
        kept_zh = _row("keep-zh", text="这是一段完整且符合要求的中文回答。")
        kept_en = _row(
            "keep-en",
            text="This is a complete English answer with enough Latin words.",
            family="claude",
            prompt_id="en-science-explain-001",
            language="en",
        )
        rows = [
            kept_zh,
            kept_en,
            _row("truncated", text="未完成", finish_reason="length"),
            _row("filtered", text="无法回答", finish_reason="content_filter"),
            _row("route", text="正常文本", route_mismatch=True),
            _row(
                "wrong-language",
                text="这是一段明显使用中文回答的文本，因此不符合英文提示词的语言要求。",
                prompt_id="en-science-explain-001",
                language="en",
            ),
            _row("duplicate", text=kept_zh["text"]),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.jsonl"
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )

            samples, audit = load_clean_samples(path)

        self.assertEqual(audit["raw_records"], 7)
        self.assertEqual(audit["kept_records"], 2)
        self.assertEqual(audit["excluded_by_reason"]["truncated"], 1)
        self.assertEqual(audit["excluded_by_reason"]["content_filtered"], 1)
        self.assertEqual(audit["excluded_by_reason"]["route_mismatch"], 1)
        self.assertEqual(audit["excluded_by_reason"]["language_mismatch"], 1)
        self.assertEqual(audit["excluded_by_reason"]["exact_duplicate"], 1)
        self.assertEqual([sample.request_key for sample in samples], ["keep-zh", "keep-en"])

    def test_semantic_prompt_group_pairs_languages(self) -> None:
        self.assertEqual(
            semantic_prompt_group("zh-science-explain-001"),
            "science-explain-001",
        )
        self.assertEqual(
            semantic_prompt_group("en-science-explain-001"),
            "science-explain-001",
        )
        self.assertEqual(semantic_prompt_group("neutral-001"), "neutral-001")

    def test_group_folds_are_deterministic_and_never_split_a_group(self) -> None:
        samples = [
            _sample(label, group_index)
            for group_index in range(12)
            for label in ("openai", "claude")
        ]

        first = assign_group_folds(samples, n_splits=6, seed=42)
        second = assign_group_folds(samples, n_splits=6, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(sorted(set(first)), list(range(6)))
        group_to_folds: dict[str, set[int]] = {}
        for sample, fold in zip(samples, first, strict=True):
            group_to_folds.setdefault(sample.semantic_prompt_group, set()).add(fold)
        self.assertTrue(all(len(folds) == 1 for folds in group_to_folds.values()))

    def test_group_folds_reject_impossible_split_counts(self) -> None:
        samples = [_sample("openai", 0), _sample("claude", 0)]
        with self.assertRaisesRegex(ValueError, "n_splits"):
            assign_group_folds(samples, n_splits=6, seed=42)

    def test_invalid_json_reports_path_and_line(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "broken.jsonl"
            path.write_text("{}\nnot-json\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, r"broken\.jsonl:2"):
                load_clean_samples(path)


if __name__ == "__main__":
    unittest.main()
