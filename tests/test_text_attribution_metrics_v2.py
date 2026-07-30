"""Contract tests for probability-based family attribution metrics."""

from __future__ import annotations

import json
import math
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
        for probabilities in (
            np.array([[np.nan, 1.0]]),
            np.array([[-0.1, 1.1]]),
            np.array([[1.0000005, 0.0]]),
        ):
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

    def test_metrics_reject_non_argmax_predictions_and_ece_requires_top_confidence(self) -> None:
        record = FamilyPrediction(
            request_key="request-1", experiment_id="exp-1", split="test", true_family="alpha",
            predicted_family="beta", family_probabilities={"alpha": 0.8, "beta": 0.2},
            confidence=0.2, model_id="alpha-model", prompt_id="prompt-1",
            semantic_prompt_group="general", language="en", task_type="summary",
            target_length="short", temperature=0.7,
        )

        with self.assertRaisesRegex(ValueError, "argmax"):
            evaluate_family_predictions((record,), FAMILIES)
        with self.assertRaisesRegex(ValueError, "confidence"):
            expected_calibration_error((record,))

    def test_tied_probabilities_select_the_first_family_label(self) -> None:
        record = build_prediction_records(
            (make_sample(1, "beta"),), np.array([[0.5, 0.5]]), ("beta", "alpha"), "e", "t"
        )[0]

        self.assertEqual(record.predicted_family, "beta")
        evaluate_family_predictions((record,), ("beta", "alpha"))
        with self.assertRaisesRegex(ValueError, "argmax"):
            evaluate_family_predictions((replace(record, predicted_family="alpha"),), ("beta", "alpha"))

    def test_family_prediction_normalizes_numeric_values_and_is_json_isolated(self) -> None:
        source_probabilities = {"alpha": np.float64(0.8), "beta": np.float32(0.2)}
        record = FamilyPrediction(
            request_key="request-1", experiment_id="exp-1", split="test", true_family="alpha",
            predicted_family="alpha", family_probabilities=source_probabilities,
            confidence=np.float32(0.8), model_id="alpha-model", prompt_id="prompt-1",
            semantic_prompt_group="general", language="en", task_type="summary",
            target_length="short", temperature=np.float64(0.7),
        )
        source_probabilities["alpha"] = 0.1
        result = record.to_dict()
        result["family_probabilities"]["alpha"] = 0.1
        result["confidence"] = 0.1

        self.assertIsInstance(record.confidence, float)
        self.assertIsInstance(record.temperature, float)
        self.assertTrue(all(isinstance(value, float) for value in record.family_probabilities.values()))
        self.assertEqual(record.family_probabilities["alpha"], 0.8)
        self.assertEqual(record.confidence, float(np.float32(0.8)))
        self.assertEqual(record.to_dict()["family_probabilities"]["alpha"], 0.8)
        json.dumps(result)

    def test_family_prediction_rejects_numeric_strings_and_booleans(self) -> None:
        values = (
            {"confidence": "0.8"},
            {"temperature": "0.7"},
            {"family_probabilities": {"alpha": "0.8", "beta": 0.2}},
            {"confidence": True},
        )
        base = {
            "request_key": "request-1", "experiment_id": "exp-1", "split": "test",
            "true_family": "alpha", "predicted_family": "alpha",
            "family_probabilities": {"alpha": 0.8, "beta": 0.2}, "confidence": 0.8,
            "model_id": "alpha-model", "prompt_id": "prompt-1",
            "semantic_prompt_group": "general", "language": "en", "task_type": "summary",
            "target_length": "short", "temperature": 0.7,
        }
        for override in values:
            with self.subTest(override=override):
                with self.assertRaisesRegex(ValueError, "real number"):
                    FamilyPrediction(**(base | override))

    def test_calibration_brier_and_nll_match_known_values(self) -> None:
        records = (
            FamilyPrediction(
                request_key="request-1", experiment_id="exp-1", split="test", true_family="alpha",
                predicted_family="alpha", family_probabilities={"alpha": 0.8, "beta": 0.2},
                confidence=0.8, model_id="model", prompt_id="prompt-1", semantic_prompt_group="general",
                language="en", task_type="summary", target_length="short", temperature=0.7,
            ),
            FamilyPrediction(
                request_key="request-2", experiment_id="exp-1", split="test", true_family="beta",
                predicted_family="beta", family_probabilities={"alpha": 0.3, "beta": 0.7},
                confidence=0.7, model_id="model", prompt_id="prompt-1", semantic_prompt_group="general",
                language="en", task_type="summary", target_length="short", temperature=0.7,
            ),
        )
        calibration = evaluate_family_predictions(records, FAMILIES)["calibration"]

        self.assertEqual(calibration["expected_calibration_error"], 0.25)
        self.assertAlmostEqual(calibration["multiclass_brier_score"], 0.13, places=15)
        self.assertAlmostEqual(calibration["negative_log_likelihood"], 0.2899092476264711, places=15)

    def test_nll_uses_fixed_epsilon_for_zero_true_probability(self) -> None:
        record = FamilyPrediction(
            request_key="request-1", experiment_id="exp-1", split="test", true_family="alpha",
            predicted_family="beta", family_probabilities={"alpha": 0.0, "beta": 1.0},
            confidence=1.0, model_id="model", prompt_id="prompt-1", semantic_prompt_group="general",
            language="en", task_type="summary", target_length="short", temperature=0.7,
        )

        metrics = evaluate_family_predictions((record,), FAMILIES)

        self.assertEqual(metrics["calibration"]["negative_log_likelihood"], -math.log(1e-12))

    def test_label_order_subgroup_metrics_and_invalid_ece_bins(self) -> None:
        record = FamilyPrediction(
            request_key="request-1", experiment_id="exp-1", split="test", true_family="alpha",
            predicted_family="alpha", family_probabilities={"alpha": 0.8, "beta": 0.2},
            confidence=0.8, model_id="model", prompt_id="prompt-1", semantic_prompt_group="general",
            language="en", task_type="summary", target_length="short", temperature=0.7,
        )

        metrics = evaluate_family_predictions((record,), ("beta", "alpha"))

        self.assertEqual(metrics["overall"]["accuracy"], 1.0)
        self.assertEqual(metrics["subgroups"]["language"]["en"]["macro_f1"], 0.5)
        for bins in (0, -1, True, 1.5):
            with self.subTest(bins=bins):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    evaluate_family_predictions((record,), FAMILIES, ece_bins=bins)


if __name__ == "__main__":
    unittest.main()
