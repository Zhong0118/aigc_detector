from pathlib import Path
import unittest

from research.text_attribution.production import request_key
from research.text_attribution.repair import load_repair_manifest
from scripts.collect_attribution_repairs import build_preview, parse_args


ROOT = Path(__file__).resolve().parents[1]
REPAIR_MANIFEST = (
    ROOT / "configs/text_attribution/repairs/server_600_v2_quality_repairs.v3.json"
)
V4_REPAIR_MANIFEST = (
    ROOT / "configs/text_attribution/repairs/server_600_v4_quality_repairs.json"
)


class RepairManifestTests(unittest.TestCase):
    def test_v4_manifest_repairs_only_the_two_reused_truncated_rows(self) -> None:
        manifest = load_repair_manifest(V4_REPAIR_MANIFEST)

        self.assertEqual(manifest.run_id, "zhizengzeng-attribution-quality-repairs-v4")
        self.assertEqual(manifest.sampling_id, "repair-v4")
        self.assertEqual(len(manifest.cells), 2)
        self.assertEqual(len(manifest.plan), 2)
        self.assertEqual(
            {cell.model_id for cell in manifest.cells},
            {"claude-sonnet-4-6-20260217", "mimo-v2.5"},
        )
        self.assertEqual(
            {item.parameters.sampling_id for item in manifest.plan},
            {"repair-v4"},
        )

    def test_quality_manifest_builds_exactly_four_explicit_requests(self) -> None:
        manifest = load_repair_manifest(REPAIR_MANIFEST)

        self.assertEqual(manifest.run_id, "zhizengzeng-attribution-quality-repairs-v3")
        self.assertEqual(len(manifest.cells), 4)
        self.assertEqual(len(manifest.plan), 4)
        self.assertEqual(
            {cell.reason for cell in manifest.cells},
            {"truncated", "exact_duplicate"},
        )
        self.assertEqual(
            {item.parameters.sampling_id for item in manifest.plan},
            {"repair-v3"},
        )
        self.assertEqual(
            {item.parameters.max_tokens for item in manifest.plan},
            {1024, 1536},
        )

    def test_physical_request_keys_do_not_overwrite_superseded_rows(self) -> None:
        manifest = load_repair_manifest(REPAIR_MANIFEST)
        physical_keys = {request_key(item) for item in manifest.plan}
        superseded_keys = {
            cell.supersedes_request_key for cell in manifest.cells
        }

        self.assertEqual(len(physical_keys), 4)
        self.assertEqual(len(superseded_keys), 4)
        self.assertTrue(physical_keys.isdisjoint(superseded_keys))

    def test_manifest_exposes_model_metadata_for_collection(self) -> None:
        manifest = load_repair_manifest(REPAIR_MANIFEST)

        self.assertEqual(
            {model.model_id for model in manifest.collection_config.models},
            {
                "claude-sonnet-4-6-20260217",
                "mimo-v2.5",
                "qwen-plus-2025-07-28",
            },
        )
        self.assertEqual(manifest.collection_config.max_requests, 4)


class RepairCliTests(unittest.TestCase):
    def test_cli_defaults_to_safe_dry_run_controls(self) -> None:
        args = parse_args(["--manifest", str(REPAIR_MANIFEST), "--dry-run"])

        self.assertEqual(args.max_workers, 4)
        self.assertEqual(args.empty_max_attempts, 2)
        self.assertFalse(args.confirm)

    def test_preview_lists_every_superseded_logical_key(self) -> None:
        manifest = load_repair_manifest(REPAIR_MANIFEST)
        preview = build_preview(manifest)

        self.assertEqual(preview["request_count"], 4)
        self.assertEqual(len(preview["repairs"]), 4)
        self.assertEqual(
            {row["reason"] for row in preview["repairs"]},
            {"truncated", "exact_duplicate"},
        )


if __name__ == "__main__":
    unittest.main()
