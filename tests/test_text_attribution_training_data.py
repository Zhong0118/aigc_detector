from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from research.text_attribution.training_data import load_training_dataset
from tests.text_attribution_test_utils import raw_row, write_dataset, write_model_catalog


class TrainingDataTests(unittest.TestCase):
    def test_loads_family_target_and_model_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            catalog_path = directory / "catalog.json"
            data_path = directory / "data.jsonl"
            write_model_catalog(catalog_path)
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


if __name__ == "__main__":
    unittest.main()
