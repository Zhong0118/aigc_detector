from __future__ import annotations

from collections import UserDict
import unittest

from research.text_attribution.neural_data import (
    NeuralAttributionDataset,
    build_family_label_maps,
    model_balanced_weights,
    select_smoke_samples,
)
from research.text_attribution.training_data import AttributionSample


def make_sample(
    request_key: str,
    *,
    family_id: str = "openai",
    model_id: str = "gpt-fast-test",
) -> AttributionSample:
    return AttributionSample(
        request_key=request_key,
        text=f"text for {request_key}",
        family_id=family_id,
        model_id=model_id,
        prompt_id=f"prompt-{request_key}",
        semantic_prompt_group=f"group-{request_key}",
        language="en",
        task_type="explanation",
        target_length="short",
        temperature=0.3,
        repeat_index=0,
    )


class RecordingTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __call__(self, text: str, **kwargs: object) -> dict[str, list[int]]:
        self.calls.append((text, kwargs))
        return {"input_ids": [11, 12], "attention_mask": [1, 1]}


class StaticTokenizer:
    def __init__(self, output: object) -> None:
        self.output = output

    def __call__(self, text: str, **kwargs: object) -> object:
        return self.output


class NeuralAttributionDatasetTests(unittest.TestCase):
    def test_tokenizes_with_exact_parameters_and_returns_integer_label(self) -> None:
        sample = make_sample("sample-1")
        tokenizer = RecordingTokenizer()
        dataset = NeuralAttributionDataset(
            (sample,),
            tokenizer,
            {"openai": 7},
            max_length=128,
        )

        item = dataset[0]

        self.assertEqual(len(dataset), 1)
        self.assertEqual(
            tokenizer.calls,
            [
                (
                    sample.text,
                    {
                        "truncation": True,
                        "max_length": 128,
                        "padding": False,
                    },
                )
            ],
        )
        self.assertEqual(
            item,
            {"input_ids": [11, 12], "attention_mask": [1, 1], "labels": 7},
        )
        self.assertIs(type(item["labels"]), int)

    def test_accepts_custom_mapping_and_returns_a_shallow_copy(self) -> None:
        tokenizer_output = UserDict(
            {"input_ids": [21, 22], "attention_mask": [1, 1]}
        )
        dataset = NeuralAttributionDataset(
            (make_sample("sample-1"),),
            StaticTokenizer(tokenizer_output),
            {"openai": 2},
            max_length=32,
        )

        item = dataset[0]

        self.assertIsInstance(item, dict)
        self.assertIsNot(item, tokenizer_output)
        self.assertIs(item["input_ids"], tokenizer_output["input_ids"])
        self.assertNotIn("labels", tokenizer_output)
        self.assertEqual(item["labels"], 2)

    def test_rejects_non_mapping_tokenizer_outputs_with_clear_context(self) -> None:
        invalid_outputs = ([("input_ids", [31, 32])], None)

        for output in invalid_outputs:
            with self.subTest(output=output):
                dataset = NeuralAttributionDataset(
                    (make_sample("sample-1"),),
                    StaticTokenizer(output),
                    {"openai": 0},
                    max_length=32,
                )
                with self.assertRaisesRegex(TypeError, r"tokenizer.*Mapping"):
                    dataset[0]

    def test_model_input_does_not_leak_sample_metadata(self) -> None:
        sample = make_sample("secret-request-key")
        dataset = NeuralAttributionDataset(
            (sample,),
            RecordingTokenizer(),
            {"openai": 0},
            max_length=16,
        )

        item = dataset[0]

        metadata_fields = {
            "request_key",
            "text",
            "family_id",
            "model_id",
            "prompt_id",
            "semantic_prompt_group",
            "language",
            "task_type",
            "target_length",
            "temperature",
            "repeat_index",
        }
        self.assertTrue(metadata_fields.isdisjoint(item))

    def test_rejects_empty_samples_unknown_family_and_invalid_max_length(self) -> None:
        tokenizer = RecordingTokenizer()
        sample = make_sample("sample-1")

        with self.assertRaisesRegex(ValueError, "samples"):
            NeuralAttributionDataset((), tokenizer, {"openai": 0}, max_length=16)
        with self.assertRaisesRegex(ValueError, "family"):
            NeuralAttributionDataset(
                (sample,), tokenizer, {"claude": 0}, max_length=16
            )
        for invalid in (0, -1, True, 1.5, "16"):
            with self.subTest(max_length=invalid):
                with self.assertRaisesRegex(ValueError, "max_length"):
                    NeuralAttributionDataset(
                        (sample,), tokenizer, {"openai": 0}, max_length=invalid
                    )

    def test_rejects_non_integer_family_label_ids_including_bool(self) -> None:
        tokenizer = RecordingTokenizer()
        sample = make_sample("sample-1")

        for invalid in ("0", 1.5, True):
            with self.subTest(label_id=invalid):
                with self.assertRaisesRegex(ValueError, "family_to_id"):
                    NeuralAttributionDataset(
                        (sample,),
                        tokenizer,
                        {"openai": invalid},
                        max_length=16,
                    )


class FamilyLabelMapTests(unittest.TestCase):
    def test_preserves_input_order_and_builds_inverse(self) -> None:
        family_to_id, id_to_family = build_family_label_maps(
            ("claude", "openai", "gemini")
        )

        self.assertEqual(
            family_to_id,
            {"claude": 0, "openai": 1, "gemini": 2},
        )
        self.assertEqual(
            id_to_family,
            {0: "claude", 1: "openai", 2: "gemini"},
        )

    def test_rejects_empty_duplicate_blank_and_non_string_labels(self) -> None:
        invalid_cases = (
            (),
            ("openai", "openai"),
            ("openai", ""),
            ("openai", "   "),
            ("openai", 3),
        )
        for labels in invalid_cases:
            with self.subTest(labels=labels):
                with self.assertRaises(ValueError):
                    build_family_label_maps(labels)


class ModelBalancedWeightsTests(unittest.TestCase):
    def test_weights_each_sample_by_inverse_model_frequency(self) -> None:
        samples = (
            make_sample("gpt-1"),
            make_sample("gpt-2"),
            make_sample("claude-1", family_id="claude", model_id="claude-test"),
        )

        weights = model_balanced_weights(samples)

        self.assertEqual(weights, (0.5, 0.5, 1.0))
        self.assertIsInstance(weights, tuple)

    def test_rejects_empty_samples(self) -> None:
        with self.assertRaisesRegex(ValueError, "samples"):
            model_balanced_weights(())


class SmokeSampleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.samples = tuple(
            [
                *(
                    make_sample(f"gpt-{index}")
                    for index in range(6)
                ),
                *(
                    make_sample(
                        f"claude-{index}",
                        family_id="claude",
                        model_id="claude-test",
                    )
                    for index in range(6)
                ),
                make_sample(
                    "gemini-0",
                    family_id="gemini",
                    model_id="gemini-test",
                ),
            ]
        )

    def test_is_order_independent_limited_and_covers_models_and_families(self) -> None:
        selected = select_smoke_samples(self.samples, per_model=2, seed=19)
        reordered = select_smoke_samples(
            tuple(reversed(self.samples)),
            per_model=2,
            seed=19,
        )

        self.assertEqual(selected, reordered)
        self.assertIsInstance(selected, tuple)
        self.assertEqual(
            {sample.model_id for sample in selected},
            {"gpt-fast-test", "claude-test", "gemini-test"},
        )
        self.assertEqual(
            {sample.family_id for sample in selected},
            {"openai", "claude", "gemini"},
        )
        for model_id in {sample.model_id for sample in selected}:
            self.assertLessEqual(
                sum(sample.model_id == model_id for sample in selected),
                2,
            )

    def test_different_seeds_can_change_the_selection(self) -> None:
        first = select_smoke_samples(self.samples, per_model=2, seed=3)
        second = select_smoke_samples(self.samples, per_model=2, seed=97)

        self.assertNotEqual(first, second)

    def test_rejects_invalid_per_model_and_seed_including_bool(self) -> None:
        for invalid in (0, -1, True, 1.5, "2"):
            with self.subTest(per_model=invalid):
                with self.assertRaisesRegex(ValueError, "per_model"):
                    select_smoke_samples(
                        self.samples,
                        per_model=invalid,
                        seed=1,
                    )

        for invalid in (True, 1.5, "1", None):
            with self.subTest(seed=invalid):
                with self.assertRaisesRegex(ValueError, "seed"):
                    select_smoke_samples(
                        self.samples,
                        per_model=1,
                        seed=invalid,
                    )

    def test_requires_per_model_and_seed_as_keyword_arguments(self) -> None:
        with self.assertRaises(TypeError):
            select_smoke_samples(self.samples, 1, 7)


if __name__ == "__main__":
    unittest.main()
