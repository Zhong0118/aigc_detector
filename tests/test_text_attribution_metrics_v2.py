"""Contract tests for probability-based family attribution metrics."""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from research.text_attribution.attribution_metrics import (
    build_prediction_records,
    evaluate_family_predictions,
    expected_calibration_error,
)
from research.text_attribution.attribution_types import FamilyPrediction
from research.text_attribution.training_data import AttributionSample


FAMILIES = ("alpha", "beta")


def make_sample(index: int, family_id: str = "alpha") -> AttributionSample:
    return AttributionSample(
        request_key=f"request-{index}",
        text="sample text",
        family_id=family_id,
        model_id=f"{family_id}-model",
        prompt_id="prompt-1",
        semantic_prompt_group="general",
        language="en",
        task_type="summary",
        target_length="short",
        temperature=0.7,
        repeat_index=index,
    )


class AttributionMetricsTests(unittest.TestCase):
    def test_perfect_predictions_produce_expected_metrics(self) -> None:
        samples = (
            make_sample(1, "alpha"),
            make_sample(2, "beta"),
            make_sample(3, "alpha"),
            make_sample(4, "beta"),
        )
        records = build_prediction_records(
            samples,
            np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]]),
            FAMILIES,
            experiment_id="exp-1",
            split="test",
        )

        metrics = evaluate_family_predictions(records, FAMILIES)

        self.assertEqual(metrics["overall"]["accuracy"], 1.0)
        self.assertEqual(metrics["overall"]["macro_f1"], 1.0)
        self.assertEqual(metrics["overall"]["weighted_f1"], 1.0)
        self.assertEqual(metrics["confusion_matrix"], [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(metrics["per_model"]["alpha-model"]["support"], 2)
        self.assertEqual(metrics["per_model"]["alpha-model"]["accuracy"], 1.0)
        self.assertNotIn("family_accuracy", metrics["per_model"]["alpha-model"])
        self.assertLess(metrics["calibration"]["negative_log_likelihood"], 0.3)
        self.assertEqual(metrics["subgroups"]["language"]["en"]["support"], 4)

    def test_build_records_rejects_probability_rows_not_summing_to_one(self) -> None:
        with self.assertRaisesRegex(ValueError, "sum to one"):
            build_prediction_records(
                (make_sample(1),),
                np.array([[0.6, 0.3]]),
                FAMILIES,
                experiment_id="exp-1",
                split="test",
            )

    def test_build_records_rejects_nonfinite_and_negative_probabilities(self) -> None:
        for probabilities in (np.array([[np.nan, 1.0]]), np.array([[-0.1, 1.1]])):
            with self.subTest(probabilities=probabilities):
                with self.assertRaises(ValueError):
                    build_prediction_records(
                        (make_sample(1),), probabilities, FAMILIES, "exp-1", "test"
                    )

    def test_build_records_rejects_duplicate_and_unknown_family_labels(self) -> None:
        with self.assertRaises(ValueError):
            build_prediction_records(
                (make_sample(1),), np.array([[0.5, 0.5]]), ("alpha", "alpha"), "e", "t"
            )
        with self.assertRaises(ValueError):
            build_prediction_records(
                (make_sample(1, "unknown"),), np.array([[0.5, 0.5]]), FAMILIES, "e", "t"
            )

    def test_evaluation_rejects_empty_records(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_family_predictions((), FAMILIES)

    def test_ece_includes_confidence_one_in_final_bin(self) -> None:
        records = build_prediction_records(
            (make_sample(1),), np.array([[1.0, 0.0]]), FAMILIES, "e", "t"
        )

        self.assertEqual(expected_calibration_error(records, bins=10), 0.0)

    def test_metrics_reject_invalid_external_prediction_confidence(self) -> None:
        record = FamilyPrediction(
            request_key="request-1",
            experiment_id="exp-1",
            split="test",
            true_family="alpha",
            predicted_family="alpha",
            family_probabilities={"alpha": 0.8, "beta": 0.2},
            confidence=0.8,
            model_id="alpha-model",
            prompt_id="prompt-1",
            semantic_prompt_group="general",
            language="en",
            task_type="summary",
            target_length="short",
            temperature=0.7,
        )
        invalid_records = {
            "nan": replace(record, confidence=float("nan")),
            "negative": replace(record, confidence=-0.1),
            "above_one": replace(record, confidence=1.1),
            "inconsistent": replace(record, confidence=0.7),
        }

        for name, invalid_record in invalid_records.items():
            with self.subTest(method="evaluate", confidence=name):
                with self.assertRaisesRegex(ValueError, "confidence"):
                    evaluate_family_predictions((invalid_record,), FAMILIES)
            with self.subTest(method="ece", confidence=name):
                with self.assertRaisesRegex(ValueError, "confidence"):
                    expected_calibration_error((invalid_record,))


if __name__ == "__main__":
    unittest.main()
