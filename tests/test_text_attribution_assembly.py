from pathlib import Path
import unittest

from research.text_attribution.assembly import assemble_v3_records
from research.text_attribution.production import (
    build_plan_from_config,
    request_key,
)
from research.text_attribution.repair import load_repair_manifest
from research.text_attribution.run_config import load_run_config
from scripts.assemble_attribution_v3 import parse_args


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "text_attribution"
V3_CONFIG = CONFIG_ROOT / "runs" / "server_600_v3.json"
V4_CONFIG = CONFIG_ROOT / "runs" / "server_600_v4.json"
V2_CONFIG = CONFIG_ROOT / "runs" / "server_600_v2.json"
V4_INCREMENTAL_CONFIG = (
    CONFIG_ROOT / "runs" / "incremental_batch_480_v4.json"
)
REPAIR_MANIFEST = (
    CONFIG_ROOT / "repairs" / "server_600_v2_quality_repairs.v3.json"
)
V4_REPAIR_MANIFEST = (
    CONFIG_ROOT / "repairs" / "server_600_v4_quality_repairs.json"
)
REPLACEMENT_MODELS = {
    "gpt-4o-mini-2024-07-18",
    "deepseek-chat",
    "glm-4-flashx-250414",
}
RETIRED_MODELS = {
    "gpt-4.1-nano-2025-04-14",
    "deepseek-v4-pro",
    "glm-4.5-air",
}


def _record(key: str, model: str, *, run_id: str = "source") -> dict:
    return {
        "request_key": key,
        "request_model": model,
        "run_id": run_id,
        "family_id": "placeholder",
        "model_variant_role": "placeholder",
        "finish_reason": "stop",
        "route_mismatch": False,
        "text": f"valid generated text {key}",
    }


class AttributionAssemblyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_run_config(V3_CONFIG)
        self.manifest = load_repair_manifest(REPAIR_MANIFEST)
        self.expected_plan = build_plan_from_config(self.config)
        superseded = {
            cell.supersedes_request_key for cell in self.manifest.cells
        }
        self.base = [
            _record(request_key(item), item.model, run_id="server-600-v2")
            for item in self.expected_plan
            if item.model not in REPLACEMENT_MODELS
            and request_key(item) not in superseded
        ]
        self.base.extend(
            _record(f"retired-{index}", model, run_id="server-600-v2")
            for index, model in enumerate(sorted(RETIRED_MODELS))
        )
        self.replacements = [
            _record(request_key(item), item.model, run_id="replacement-90-v3")
            for item in self.expected_plan
            if item.model in REPLACEMENT_MODELS
        ]
        self.repairs = [
            _record(request_key(item), item.model, run_id="quality-repairs-v3")
            for item in self.manifest.plan
        ]

    def test_assembly_produces_exactly_the_v3_logical_key_space(self) -> None:
        result = assemble_v3_records(
            config=self.config,
            repair_manifest=self.manifest,
            base_records=self.base,
            replacement_records=self.replacements,
            repair_records=self.repairs,
            retired_models=RETIRED_MODELS,
        )

        expected_keys = {request_key(item) for item in self.expected_plan}
        self.assertEqual(len(result.records), 600)
        self.assertEqual(
            {row["request_key"] for row in result.records},
            expected_keys,
        )
        self.assertEqual(result.report["kept_from_base"], 506)
        self.assertEqual(result.report["added_replacements"], 90)
        self.assertEqual(result.report["added_repairs"], 4)

    def test_repair_rows_keep_both_logical_and_physical_provenance(self) -> None:
        result = assemble_v3_records(
            config=self.config,
            repair_manifest=self.manifest,
            base_records=self.base,
            replacement_records=self.replacements,
            repair_records=self.repairs,
            retired_models=RETIRED_MODELS,
        )
        repaired = [row for row in result.records if "repair" in row]

        self.assertEqual(len(repaired), 4)
        self.assertTrue(
            all(
                row["request_key"] == row["repair"]["supersedes_request_key"]
                and row["request_key"]
                != row["repair"]["collection_request_key"]
                and row["source_run_id"] == "quality-repairs-v3"
                for row in repaired
            )
        )

    def test_assembly_rejects_an_incomplete_replacement_batch(self) -> None:
        with self.assertRaisesRegex(ValueError, "replacement.*missing"):
            assemble_v3_records(
                config=self.config,
                repair_manifest=self.manifest,
                base_records=self.base,
                replacement_records=self.replacements[:-1],
                repair_records=self.repairs,
                retired_models=RETIRED_MODELS,
            )

    def test_assembly_rejects_truncated_final_rows(self) -> None:
        bad_repairs = [dict(row) for row in self.repairs]
        bad_repairs[0]["finish_reason"] = "length"

        with self.assertRaisesRegex(ValueError, "truncated"):
            assemble_v3_records(
                config=self.config,
                repair_manifest=self.manifest,
                base_records=self.base,
                replacement_records=self.replacements,
                repair_records=bad_repairs,
                retired_models=RETIRED_MODELS,
            )

    def test_assembly_rejects_exact_duplicate_final_text(self) -> None:
        duplicate_replacements = [dict(row) for row in self.replacements]
        duplicate_replacements[-1]["text"] = duplicate_replacements[0]["text"]

        with self.assertRaisesRegex(ValueError, "exact duplicate"):
            assemble_v3_records(
                config=self.config,
                repair_manifest=self.manifest,
                base_records=self.base,
                replacement_records=duplicate_replacements,
                repair_records=self.repairs,
                retired_models=RETIRED_MODELS,
            )

    def test_cli_has_the_three_frozen_retired_routes(self) -> None:
        args = parse_args(
            [
                "--config",
                str(V3_CONFIG),
                "--repair-manifest",
                str(REPAIR_MANIFEST),
                "--base",
                "base.jsonl",
                "--replacements",
                "replacements.jsonl",
                "--repairs",
                "repairs.jsonl",
                "--output",
                "assembled.jsonl",
            ]
        )

        self.assertEqual(set(args.retired_models), RETIRED_MODELS)


class AttributionV4AssemblyTests(unittest.TestCase):
    def test_selected_model_assembly_reuses_118_and_adds_480_plus_2(self) -> None:
        config = load_run_config(V4_CONFIG)
        base_config = load_run_config(V2_CONFIG)
        incremental = load_run_config(V4_INCREMENTAL_CONFIG)
        manifest = load_repair_manifest(V4_REPAIR_MANIFEST)
        expected_plan = build_plan_from_config(config)
        replacement_models = {model.model_id for model in incremental.models}
        selected_models = {model.model_id for model in config.models}
        retired_models = {
            model.model_id
            for model in base_config.models
            if model.model_id not in selected_models
        }
        superseded = {
            cell.supersedes_request_key for cell in manifest.cells
        }
        base = [
            _record(request_key(item), item.model, run_id="server-600-v2")
            for item in expected_plan
            if item.model not in replacement_models
            and request_key(item) not in superseded
        ]
        replacements = [
            _record(request_key(item), item.model, run_id="incremental-480-v4")
            for item in expected_plan
            if item.model in replacement_models
        ]
        repairs = [
            _record(request_key(item), item.model, run_id="quality-repairs-v4")
            for item in manifest.plan
        ]

        result = assemble_v3_records(
            config=config,
            repair_manifest=manifest,
            base_records=base,
            replacement_records=replacements,
            repair_records=repairs,
            retired_models=retired_models,
            replacement_models=replacement_models,
        )

        self.assertEqual(len(result.records), 600)
        self.assertEqual(result.report["kept_from_base"], 118)
        self.assertEqual(result.report["added_replacements"], 480)
        self.assertEqual(result.report["added_repairs"], 2)

    def test_cli_accepts_base_and_replacement_configs_for_v4(self) -> None:
        args = parse_args(
            [
                "--config",
                str(V4_CONFIG),
                "--base-config",
                str(V2_CONFIG),
                "--replacement-config",
                str(V4_INCREMENTAL_CONFIG),
                "--repair-manifest",
                str(V4_REPAIR_MANIFEST),
                "--base",
                "base.jsonl",
                "--replacements",
                "replacements.jsonl",
                "--repairs",
                "repairs.jsonl",
                "--output",
                "assembled.jsonl",
            ]
        )

        self.assertEqual(args.base_config, V2_CONFIG)
        self.assertEqual(args.replacement_config, V4_INCREMENTAL_CONFIG)


if __name__ == "__main__":
    unittest.main()
