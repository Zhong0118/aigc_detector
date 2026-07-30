from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import FeatureUnion

from research.text_attribution.classical_models import (
    EXPERIMENT_IDS,
    build_classical_pipeline,
    predict_family_probabilities,
)
from research.text_attribution.split_manifest import apply_split_manifest, create_split_manifest
from research.text_attribution.training_data import load_training_dataset
from tests.text_attribution_test_utils import write_dataset, write_model_catalog


class ClassicalModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        root = Path(cls._temporary.name)
        data_path = root / "samples.jsonl"
        catalog_path = root / "catalog.json"
        write_dataset(data_path, groups=20)
        write_model_catalog(catalog_path)
        dataset = load_training_dataset(data_path, catalog_path)
        cls.family_labels = dataset.family_labels
        cls.splits = apply_split_manifest(dataset.samples, create_split_manifest(dataset.samples, seed=42))
        cls.train_texts = [sample.text for sample in cls.splits["train"]]
        cls.train_labels = [sample.family_id for sample in cls.splits["train"]]
        cls.validation_texts = [sample.text for sample in cls.splits["validation"]]
        cls.validation_labels = [sample.family_id for sample in cls.splits["validation"]]

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_declares_all_classical_experiments(self) -> None:
        self.assertEqual(EXPERIMENT_IDS, ("B0", "B1", "B2", "B3"))

    def test_each_experiment_fits_and_returns_aligned_validation_probabilities(self) -> None:
        for experiment_id in EXPERIMENT_IDS:
            with self.subTest(experiment_id=experiment_id):
                model = build_classical_pipeline(experiment_id, seed=42)
                model.fit(self.train_texts, self.train_labels)

                probabilities = predict_family_probabilities(
                    model, self.validation_texts, tuple(reversed(self.family_labels))
                )

                self.assertEqual(probabilities.shape, (len(self.validation_texts), 2))
                self.assertEqual(probabilities.dtype, np.float64)
                self.assertTrue(np.isfinite(probabilities).all())
                np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
                predicted = np.asarray(tuple(reversed(self.family_labels)))[np.argmax(probabilities, axis=1)]
                np.testing.assert_array_equal(predicted, model.predict(self.validation_texts))

    def test_requested_label_order_changes_probability_columns_semantically(self) -> None:
        model = build_classical_pipeline("B0", seed=42).fit(self.train_texts, self.train_labels)

        normal = predict_family_probabilities(model, self.validation_texts, self.family_labels)
        reversed_order = predict_family_probabilities(model, self.validation_texts, tuple(reversed(self.family_labels)))

        np.testing.assert_allclose(normal, reversed_order[:, ::-1])

    def test_experiment_components_are_distinct(self) -> None:
        b0 = build_classical_pipeline("B0", seed=42)
        b1 = build_classical_pipeline("B1", seed=42)
        b2 = build_classical_pipeline("B2", seed=42)
        b3 = build_classical_pipeline("B3", seed=42)

        self.assertIn("char", b0.named_steps)
        self.assertIn("style", b1.named_steps)
        for pipeline in (b0, b1, b2):
            with self.subTest(pipeline=pipeline):
                classifier = pipeline.named_steps["classifier"]
                self.assertEqual(classifier.solver, "liblinear")
                self.assertEqual(classifier.class_weight, "balanced")
                self.assertEqual(classifier.max_iter, 5000)
                self.assertEqual(classifier.random_state, 42)
        self.assertIsInstance(b2.named_steps["features"], FeatureUnion)
        self.assertEqual(
            {name for name, _ in b2.named_steps["features"].transformer_list}, {"char", "style"}
        )
        self.assertIsInstance(b3.named_steps["classifier"], CalibratedClassifierCV)
        self.assertEqual(
            {name for name, _ in b3.named_steps["features"].transformer_list}, {"word", "char"}
        )

    def test_joblib_round_trip_preserves_probabilities(self) -> None:
        model = build_classical_pipeline("B2", seed=42).fit(self.train_texts, self.train_labels)
        expected = predict_family_probabilities(model, self.validation_texts, self.family_labels)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.joblib"
            joblib.dump(model, path)
            restored = joblib.load(path)
        np.testing.assert_allclose(
            expected,
            predict_family_probabilities(restored, self.validation_texts, self.family_labels),
        )

    def test_unknown_experiment_fails_explicitly(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported classical experiment"):
            build_classical_pipeline("unknown", seed=42)

    def test_probability_contract_rejects_bad_labels_and_empty_texts(self) -> None:
        model = build_classical_pipeline("B0", seed=42).fit(self.train_texts, self.train_labels)

        with self.assertRaisesRegex(ValueError, "nonempty"):
            predict_family_probabilities(model, self.validation_texts, ())
        with self.assertRaisesRegex(ValueError, "unique"):
            predict_family_probabilities(model, self.validation_texts, ("claude", "claude"))
        with self.assertRaisesRegex(ValueError, "match model classes"):
            predict_family_probabilities(model, self.validation_texts, ("claude", "missing"))
        empty = predict_family_probabilities(model, [], self.family_labels)
        self.assertEqual(empty.shape, (0, 2))
        self.assertEqual(empty.dtype, np.float64)

    def test_probability_contract_rejects_scalar_and_non_string_texts(self) -> None:
        model = build_classical_pipeline("B0", seed=42).fit(self.train_texts, self.train_labels)

        with self.assertRaisesRegex(ValueError, "sequence"):
            predict_family_probabilities(model, "one text", self.family_labels)
        with self.assertRaisesRegex(TypeError, "strings"):
            predict_family_probabilities(model, ["valid", 1], self.family_labels)


if __name__ == "__main__":
    unittest.main()
