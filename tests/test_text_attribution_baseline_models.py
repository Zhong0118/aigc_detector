from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.run_attribution_pilot_baselines import main
from research.text_attribution.baseline_data import BaselineSample
from research.text_attribution.baseline_models import (
    build_sparse_pipeline,
    embedding_cache_fingerprint,
    evaluate_embedding_probe,
    evaluate_sparse_method,
    load_embedding_cache,
    save_embedding_cache,
)


def _classification_fixture() -> tuple[list[BaselineSample], list[int]]:
    samples: list[BaselineSample] = []
    folds: list[int] = []
    for group_index in range(6):
        fold = group_index % 2
        for label, signature in (
            ("openai", "alpha concise structured response"),
            ("claude", "beta nuanced explanatory answer"),
        ):
            samples.append(
                BaselineSample(
                    request_key=f"{label}-{group_index}",
                    text=f"{signature} repeated signature {signature} {group_index}",
                    label=label,
                    model=f"{label}-model",
                    prompt_id=f"en-task-{group_index}",
                    semantic_prompt_group=f"task-{group_index}",
                    language="en",
                    task_type="explanation",
                    target_length="medium",
                    temperature=0.0,
                    repeat_index=0,
                )
            )
            folds.append(fold)
    return samples, folds


class BaselineModelTests(unittest.TestCase):
    def test_sparse_methods_produce_one_oof_prediction_per_sample(self) -> None:
        samples, folds = _classification_fixture()

        for method in ("char_lr", "tfidf_svm"):
            with self.subTest(method=method):
                result = evaluate_sparse_method(
                    samples,
                    folds,
                    method=method,
                    seed=42,
                )
                self.assertEqual(result.method, method)
                self.assertEqual(len(result.predictions), len(samples))
                self.assertEqual(
                    sorted({row.fold for row in result.predictions}),
                    [0, 1],
                )
                self.assertGreaterEqual(result.oof_metrics["macro_f1"], 0.0)
                self.assertLessEqual(result.oof_metrics["macro_f1"], 1.0)
                self.assertEqual(result.labels, ["claude", "openai"])
                self.assertEqual(len(result.confusion_matrix), 2)

    def test_unknown_sparse_method_fails_explicitly(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported sparse method"):
            build_sparse_pipeline("not-a-method", seed=42)

    def test_embedding_fingerprint_is_stable_and_sensitive_to_content(self) -> None:
        samples, _ = _classification_fixture()
        first = embedding_cache_fingerprint(
            samples,
            model_id="intfloat/multilingual-e5-base",
            max_length=512,
        )
        second = embedding_cache_fingerprint(
            samples,
            model_id="intfloat/multilingual-e5-base",
            max_length=512,
        )
        changed = embedding_cache_fingerprint(
            [*samples[:-1], _replace_text(samples[-1], "changed text")],
            model_id="intfloat/multilingual-e5-base",
            max_length=512,
        )

        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_embedding_cache_round_trip_rejects_wrong_row_order(self) -> None:
        samples, _ = _classification_fixture()
        embeddings = np.arange(len(samples) * 4, dtype=np.float32).reshape(
            len(samples), 4
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "embeddings.npz"
            save_embedding_cache(
                path,
                embeddings,
                samples,
                model_id="intfloat/multilingual-e5-base",
                max_length=512,
            )
            loaded = load_embedding_cache(
                path,
                samples,
                model_id="intfloat/multilingual-e5-base",
                max_length=512,
            )
            np.testing.assert_array_equal(loaded, embeddings)

            with self.assertRaisesRegex(ValueError, "fingerprint|row order"):
                load_embedding_cache(
                    path,
                    list(reversed(samples)),
                    model_id="intfloat/multilingual-e5-base",
                    max_length=512,
                )

    def test_embedding_probe_uses_the_same_oof_contract(self) -> None:
        samples, folds = _classification_fixture()
        embeddings = np.asarray(
            [
                [1.0, 0.0, index / 100.0]
                if sample.label == "openai"
                else [0.0, 1.0, index / 100.0]
                for index, sample in enumerate(samples)
            ],
            dtype=np.float32,
        )

        result = evaluate_embedding_probe(samples, folds, embeddings, seed=42)

        self.assertEqual(result.method, "frozen_e5")
        self.assertEqual(len(result.predictions), len(samples))
        self.assertEqual(result.oof_metrics["accuracy"], 1.0)

    def test_cli_dry_run_validates_data_without_creating_output(self) -> None:
        samples, _ = _classification_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "fixture.jsonl"
            output_dir = root / "output"
            input_path.write_text(
                "".join(
                    json.dumps(_sample_to_raw_row(sample)) + "\n"
                    for sample in samples
                ),
                encoding="utf-8",
            )

            exit_code = main(
                [
                    "--input",
                    str(input_path),
                    "--output-dir",
                    str(output_dir),
                    "--methods",
                    "char_lr",
                    "tfidf_svm",
                    "--folds",
                    "2",
                    "--dry-run",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertFalse(output_dir.exists())


def _replace_text(sample: BaselineSample, text: str) -> BaselineSample:
    values = sample.to_dict()
    values["text"] = text
    return BaselineSample(**values)


def _sample_to_raw_row(sample: BaselineSample) -> dict:
    return {
        "request_key": sample.request_key,
        "family_id": sample.label,
        "request_model": sample.model,
        "text": sample.text,
        "finish_reason": "stop",
        "route_mismatch": False,
        "repeat_index": sample.repeat_index,
        "prompt": {
            "prompt_id": sample.prompt_id,
            "language": sample.language,
            "task_type": sample.task_type,
            "target_length": sample.target_length,
        },
        "parameters": {"temperature": sample.temperature},
    }


if __name__ == "__main__":
    unittest.main()
