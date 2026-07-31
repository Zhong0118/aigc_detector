from __future__ import annotations

import unittest

from provenance.text_attribution_policy import HierarchicalAttributionPolicy
from provenance.text_attribution_schema import AttributionSample, assign_prompt_split


class AttributionSampleTests(unittest.TestCase):
    def test_direct_sample_has_stable_identity_and_contains_no_secret_field(self) -> None:
        kwargs = {
            "prompt_group_id": "prompt-zh-0001",
            "prompt_template_id": "general-qa-v1",
            "text": "这是一个用于来源归因数据契约测试的完整回答。",
            "language": "zh",
            "domain": "general_qa",
            "task_type": "question_answering",
            "provider": "teacher-relay",
            "model_family": "qwen",
            "model_id": "qwen-example-version",
            "deployment_snapshot": "2026-07-26",
            "collection_time": "2026-07-26T15:00:00+08:00",
            "split": "train",
            "temperature": 0.7,
            "top_p": 0.9,
        }

        first = AttributionSample(**kwargs)
        second = AttributionSample(**kwargs)
        record = first.to_record()

        self.assertEqual(first.sample_id, second.sample_id)
        self.assertEqual(record["original_generator"], "qwen-example-version")
        self.assertEqual(record["last_transformer"], "qwen-example-version")
        self.assertEqual(record["generation_chain"], ["qwen-example-version"])
        self.assertNotIn("api_key", record)
        self.assertNotIn("authorization", record)

    def test_rewritten_sample_preserves_original_last_and_generation_chain(self) -> None:
        sample = AttributionSample(
            prompt_group_id="rewrite-0001",
            prompt_template_id="polish-v1",
            text="A polished answer that preserves the original meaning.",
            language="en",
            domain="general_qa",
            task_type="rewrite",
            provider="teacher-relay",
            model_family="claude",
            model_id="claude-polisher-version",
            deployment_snapshot="2026-07-26",
            collection_time="2026-07-26T15:05:00+08:00",
            split="test",
            original_generator="qwen-original-version",
            last_transformer="claude-polisher-version",
            generation_chain=("qwen-original-version", "claude-polisher-version"),
            transformation_type="polished",
        )

        record = sample.to_record()

        self.assertEqual(record["original_generator"], "qwen-original-version")
        self.assertEqual(record["last_transformer"], "claude-polisher-version")
        self.assertEqual(
            record["generation_chain"],
            ["qwen-original-version", "claude-polisher-version"],
        )
        self.assertEqual(record["transformation_type"], "polished")

    def test_blank_required_fields_are_rejected(self) -> None:
        valid = {
            "prompt_group_id": "prompt-1",
            "prompt_template_id": "qa-v1",
            "text": "A sufficiently clear answer.",
            "language": "en",
            "domain": "general_qa",
            "task_type": "question_answering",
            "provider": "teacher-relay",
            "model_family": "qwen",
            "model_id": "qwen-example-version",
            "deployment_snapshot": "2026-07-26",
            "collection_time": "2026-07-26T15:00:00+08:00",
            "split": "train",
        }
        for field in ("prompt_group_id", "text", "model_family", "model_id"):
            with self.subTest(field=field):
                invalid = {**valid, field: ""}
                with self.assertRaisesRegex(ValueError, field):
                    AttributionSample(**invalid)


class PromptSplitTests(unittest.TestCase):
    def test_same_prompt_group_always_receives_same_split(self) -> None:
        assignments = {
            assign_prompt_split("shared-prompt-42", seed=20260726)
            for _ in range(20)
        }

        self.assertEqual(len(assignments), 1)

    def test_different_seed_changes_at_least_one_group_assignment(self) -> None:
        groups = [f"prompt-{index:04d}" for index in range(200)]
        first = [assign_prompt_split(group, seed=1) for group in groups]
        second = [assign_prompt_split(group, seed=2) for group in groups]

        self.assertNotEqual(first, second)

    def test_invalid_split_ratios_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            assign_prompt_split(
                "prompt-1",
                seed=1,
                ratios={"train": 0.7, "calibration": 0.2, "test": 0.2},
            )


class HierarchicalAttributionPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = HierarchicalAttributionPolicy(
            min_chars=20,
            min_known_probability=0.65,
            max_prototype_distance=0.75,
            min_exact_probability=0.80,
            min_family_probability=0.70,
            min_exact_margin=0.15,
        )
        self.model_to_family = {
            "qwen-a": "qwen",
            "qwen-b": "qwen",
            "llama-a": "llama",
        }
        self.text = "This passage is long enough to support a source decision."

    def test_returns_exact_when_known_exact_and_family_evidence_agree(self) -> None:
        decision = self.policy.decide(
            text=self.text,
            exact_probabilities={"qwen-a": 0.90, "llama-a": 0.10},
            family_probabilities={"qwen": 0.94, "llama": 0.06},
            model_to_family=self.model_to_family,
            known_probability=0.95,
            prototype_distance=0.20,
        )

        self.assertEqual(decision.status, "exact")
        self.assertEqual(decision.model_family, "qwen")
        self.assertEqual(decision.model_id, "qwen-a")

    def test_returns_family_only_when_exact_candidates_share_one_family(self) -> None:
        decision = self.policy.decide(
            text=self.text,
            exact_probabilities={"qwen-a": 0.48, "qwen-b": 0.44, "llama-a": 0.08},
            family_probabilities={"qwen": 0.93, "llama": 0.07},
            model_to_family=self.model_to_family,
            known_probability=0.91,
            prototype_distance=0.25,
        )

        self.assertEqual(decision.status, "family_only")
        self.assertEqual(decision.model_family, "qwen")
        self.assertIsNone(decision.model_id)

    def test_returns_unknown_for_low_known_probability_or_far_prototype(self) -> None:
        low_known = self.policy.decide(
            text=self.text,
            exact_probabilities={"qwen-a": 0.90, "llama-a": 0.10},
            family_probabilities={"qwen": 0.90, "llama": 0.10},
            model_to_family=self.model_to_family,
            known_probability=0.40,
            prototype_distance=0.20,
        )
        far_prototype = self.policy.decide(
            text=self.text,
            exact_probabilities={"qwen-a": 0.90, "llama-a": 0.10},
            family_probabilities={"qwen": 0.90, "llama": 0.10},
            model_to_family=self.model_to_family,
            known_probability=0.90,
            prototype_distance=0.90,
        )

        self.assertEqual(low_known.status, "unknown")
        self.assertEqual(far_prototype.status, "unknown")

    def test_returns_abstain_when_exact_and_family_heads_conflict(self) -> None:
        decision = self.policy.decide(
            text=self.text,
            exact_probabilities={"qwen-a": 0.91, "llama-a": 0.09},
            family_probabilities={"qwen": 0.08, "llama": 0.92},
            model_to_family=self.model_to_family,
            known_probability=0.95,
            prototype_distance=0.20,
        )

        self.assertEqual(decision.status, "abstain")
        self.assertIn("family_exact_conflict", decision.warnings)

    def test_returns_abstain_for_short_text(self) -> None:
        decision = self.policy.decide(
            text="too short",
            exact_probabilities={"qwen-a": 0.99},
            family_probabilities={"qwen": 0.99},
            model_to_family=self.model_to_family,
            known_probability=0.99,
            prototype_distance=0.01,
        )

        self.assertEqual(decision.status, "abstain")
        self.assertIn("insufficient_text", decision.warnings)


if __name__ == "__main__":
    unittest.main()
