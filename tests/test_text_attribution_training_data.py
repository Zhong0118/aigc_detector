from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from research.text_attribution.training_data import (
    load_model_catalog,
    load_training_dataset,
)
from tests.text_attribution_test_utils import raw_row, write_dataset, write_model_catalog


class TrainingDataTests(unittest.TestCase):
    def test_loads_family_target_and_model_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            catalog_path = directory / "catalog.json"
            data_path = directory / "data.jsonl"
            write_model_catalog(catalog_path)
            catalog_payload = json.loads(catalog_path.read_text(encoding="utf-8"))
            catalog_payload["models"].extend(
                [
                    {
                        "enabled": True,
                        "family_id": "openai",
                        "model_id": "gpt-enabled-absent-test",
                        "variant_role": "experimental",
                    },
                    {
                        "enabled": False,
                        "model_id": "disabled-malformed-absent-test",
                    },
                    "legacy-malformed-absent-test",
                ]
            )
            catalog_path.write_text(
                json.dumps(catalog_payload),
                encoding="utf-8",
            )
            write_dataset(data_path, groups=3)

            dataset = load_training_dataset(data_path, catalog_path)

        self.assertEqual(len(dataset.samples), 12)
        self.assertEqual(dataset.family_labels, ("claude", "openai"))
        self.assertEqual(
            dataset.model_labels,
            (
                "claude-haiku-test",
                "claude-sonnet-test",
                "gpt-balanced-test",
                "gpt-fast-test",
            ),
        )
        self.assertEqual(dataset.samples[0].family_id, "claude")
        self.assertEqual(dataset.samples[0].model_id, "claude-haiku-test")
        self.assertEqual(dataset.audit["kept_records"], 12)
        self.assertEqual(
            dataset.model_to_family,
            {
                "claude-haiku-test": "claude",
                "claude-sonnet-test": "claude",
                "gpt-balanced-test": "openai",
                "gpt-enabled-absent-test": "openai",
                "gpt-fast-test": "openai",
            },
        )
        self.assertEqual(
            dataset.audit["catalog_models_absent_from_clean_data"],
            ["gpt-enabled-absent-test"],
        )
        self.assertNotIn("disabled-malformed-absent-test", dataset.model_to_family)
        self.assertNotIn("legacy-malformed-absent-test", dataset.model_to_family)

    def test_rejects_family_catalog_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            catalog_path = directory / "catalog.json"
            data_path = directory / "data.jsonl"
            write_model_catalog(catalog_path)
            data_path.write_text(
                json.dumps(raw_row("gpt-fast-test", 0, family_id="claude")) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "catalog family mismatch"):
                load_training_dataset(data_path, catalog_path)

    def test_rejects_enabled_catalog_model_without_variant_role(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            catalog_path = Path(temporary) / "catalog.json"
            catalog_path.write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "family_id": "openai",
                                "model_id": "gpt-fast-test",
                                "enabled": True,
                                "variant_role": " ",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "variant_role"):
                load_model_catalog(catalog_path)

    def test_rejects_malformed_catalog_rows(self) -> None:
        malformed_rows = (
            ("enabled", {"enabled": "true"}, "enabled"),
            ("enabled", {"enabled": 1}, "enabled"),
            ("enabled", {"enabled": None}, "enabled"),
            (
                "family_id",
                {
                    "enabled": True,
                    "family_id": ["openai"],
                    "model_id": "gpt-fast-test",
                    "variant_role": "fast",
                },
                "family_id",
            ),
            (
                "model_id",
                {
                    "enabled": True,
                    "family_id": "openai",
                    "model_id": 123,
                    "variant_role": "fast",
                },
                "model_id",
            ),
            (
                "variant_role",
                {
                    "enabled": True,
                    "family_id": "openai",
                    "model_id": "gpt-fast-test",
                    "variant_role": {"role": "fast"},
                },
                "variant_role",
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            catalog_path = Path(temporary) / "catalog.json"
            for name, row, message in malformed_rows:
                with self.subTest(name=name):
                    catalog_path.write_text(
                        json.dumps({"models": [row]}), encoding="utf-8"
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        load_model_catalog(catalog_path)

    def test_ignores_non_object_catalog_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            catalog_path = Path(temporary) / "catalog.json"
            catalog_path.write_text(
                json.dumps(
                    {
                        "models": [
                            "legacy-note",
                            {
                                "enabled": True,
                                "family_id": "openai",
                                "model_id": "gpt-fast-test",
                                "variant_role": "fast",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            catalog = load_model_catalog(catalog_path)

        self.assertEqual(len(catalog), 1)
        self.assertEqual(catalog[0].model_id, "gpt-fast-test")

    def test_rejects_malformed_catalog_before_training_data_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            catalog_path = directory / "catalog.json"
            data_path = directory / "data.jsonl"
            catalog_path.write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "enabled": "true",
                                "family_id": "openai",
                                "model_id": "gpt-fast-test",
                                "variant_role": "fast",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            data_path.write_text(
                json.dumps(raw_row("gpt-fast-test", 0)) + "\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "enabled"):
                load_training_dataset(data_path, catalog_path)


if __name__ == "__main__":
    unittest.main()
