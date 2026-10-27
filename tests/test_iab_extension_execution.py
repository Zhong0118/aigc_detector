"""Offline metadata/process regressions; no network, GPU, model or image inference."""
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.image_attribution.extension_split import group_and_split_extension_v2
from research.image_attribution.iab_protocol import group_and_split
from research.image_attribution.protocol import sha256


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def expansion_rows(n):
    rng = random.Random(716)
    return [dict(sample_id=f"image-{i:03d}", source_label="model", is_ai=1,
        semantic_class="cat", prompt_id=str(i), pixel_sha256=f"pixels-{i}",
        sha256=f"bytes-{i}", phash=f"{rng.getrandbits(64):016x}", relative_path=f"image-{i}.png") for i in range(n)]


class ExtensionExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, IAB_MONITOR_STATE_DIR="monitor")
        self.env.start(); self.addCleanup(self.env.stop)
        self.p = module("prepare_iab_extension_manifest")
        self.m = module("monitor_iab_extension")
        self.folder = self.root / "monitor"
        self.m.old.write(self.folder / "config.json", dict(dataset="data", suffix="test-v7", per_cell=200,
            workers=4, route="direct", epochs=30, gpu_index=1, parent_manifest="parent.jsonl",
            research_config="research.json", reuse_from="old", prefix_selection=True))
        (self.root / "data").mkdir()
        (self.root / "parent.jsonl").write_text("parent")
        self.required_files = self.m.RUNTIME_CODE + self.m.BASELINE_ASSETS + ("assets/test.bin", "data/selection.json",
            "monitor/iab-public-v3.2-metadata.json", "research/extra.py")
        for name in self.required_files:
            path = self.root / name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("registered fixture")
        (self.root / "research.json").write_text(json.dumps(dict(output="features", gpu_index=1,
            dataset=dict(path="data", selection_sha256=sha256(self.root / "data/selection.json")),
            parent_manifest=dict(path="parent.jsonl", sha256=sha256(self.root / "parent.jsonl")),
            assets=dict(test=dict(path="assets/test.bin", sha256=sha256(self.root / "assets/test.bin"))),
            implementation_sha256={"research/extra.py": sha256(self.root / "research/extra.py")})))
        self.register()

    def register(self):
        self.m.old.write(self.folder / "registration.json", dict(files_sha256={
            "monitor/config.json": sha256(self.folder / "config.json"),
            "parent.jsonl": sha256(self.root / "parent.jsonl"),
            "research.json": sha256(self.root / "research.json"),
            **{name: sha256(self.root / name) for name in self.required_files}}))

    def parent_rows(self):
        parent, _ = group_and_split(expansion_rows(20))
        rows, _ = group_and_split_extension_v2(expansion_rows(80), frozen_rows=parent)
        return parent, rows

    def test_v2_recheck_accepts_hash_assignment_old_floor_rejects(self):
        from scripts.prepare_iab_manifest import validate_existing_split as old_check
        parent, rows = self.parent_rows()
        self.assertEqual(self.p.validate_existing_split(rows, parent)["protocol"], "future-extension-split-v2")
        with self.assertRaisesRegex(ValueError, "split/group mismatch"):
            old_check(rows, parent)

    def test_changed_parent_identity_rejected(self):
        for field in ("sha256", "pixel_sha256", "phash", "source_label", "semantic_class", "prompt_id", "is_ai"):
            with self.subTest(field=field):
                parent, rows = self.parent_rows()
                rows[0][field] = "changed"
                with self.assertRaisesRegex(ValueError, "frozen sample changed"):
                    self.p.validate_existing_split(rows, parent)

    def test_frozen_group_and_partition_bridges_fail(self):
        for same_split in (False, True):
            with self.subTest(same_split=same_split):
                parent, _ = self.parent_rows()
                a = parent[0]
                b = next(r for r in parent if r["group_id"] != a["group_id"] and (r["split"] == a["split"]) == same_split)
                bridge = dict(expansion_rows(21)[-1], sha256=a["sha256"], pixel_sha256=b["pixel_sha256"])
                with self.assertRaisesRegex(ValueError, "frozen (split|group) conflict"):
                    self.p.validate_existing_split(parent + [bridge], parent)

    def test_metadata_only_cli_recheck_and_tampered_split(self):
        parent, rows = self.parent_rows()
        dataset = self.root / "data"
        manifest, parent_path = dataset / "manifest.jsonl", self.root / "parent.jsonl"
        parent_path.write_text("".join(json.dumps(r) + "\n" for r in parent))
        manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
        (dataset / "selection.json").write_text("{}")
        audit = dict(protocol="future-extension-split-v2", seed=42, parent_manifest_sha256=sha256(parent_path),
            manifest_sha256=sha256(manifest), selection_sha256=sha256(dataset / "selection.json"))
        self.m.old.write(dataset / "audit.json", audit)
        self.assertEqual(self.p.prepare(dataset, parent_path, check_existing=True)["samples"], 80)
        self.assertFalse(any(dataset.glob("*.png")))  # No images opened by metadata recheck.
        rows[0]["split"] = "dev" if rows[0]["split"] != "dev" else "train"
        manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
        audit["manifest_sha256"] = sha256(manifest)
        self.m.old.write(dataset / "audit.json", audit)
        with self.assertRaisesRegex(ValueError, "split/group mismatch"):
            self.p.prepare(dataset, parent_path, check_existing=True)

    def test_freeze_preserves_existing_output(self):
        for name in ("manifest.jsonl", "audit.json"):
            with self.subTest(name=name):
                data = self.root / name.replace(".", "-"); data.mkdir()
                path = data / name; path.write_text("keep exactly\n")
                with self.assertRaises(FileExistsError):
                    self.p.prepare(data, self.root / "parent.jsonl")
                self.assertEqual(path.read_text(), "keep exactly\n")

    def test_receipt_identity_and_member_must_match_selection(self):
        import hashlib
        entry = dict(filename="a.png", crc32=7)
        sid = "iab-" + hashlib.sha256(b"model/cat/a.png").hexdigest()[:24]
        selection = dict(entries={"model/cat": [entry]}, audit=dict(selected_by_cell={"model/cat": 1}))
        status = dict(selected=1, selected_by_cell={"model/cat": 1})
        row = dict(sample_id=sid, source_label="model", semantic_class="cat", source_evidence=dict(zip_member="a.png", zip_crc32=7))
        receipts = [dict(source="model", semantic="cat", selected=1, rows=[row])]
        self.assertEqual(self.p.selected_receipts(selection, status, receipts), [row])
        row["sample_id"] = "different"
        with self.assertRaisesRegex(ValueError, "sample identity"):
            self.p.selected_receipts(selection, status, receipts)
        row["sample_id"] = sid; row["source_evidence"]["zip_member"] = "different"
        with self.assertRaisesRegex(ValueError, "ZIP member identity"):
            self.p.selected_receipts(selection, status, receipts)

    def test_registered_sha_mutation_rejected(self):
        self.m.verify_registration(self.root)
        (self.root / "research.json").write_text("changed")
        with self.assertRaisesRegex(ValueError, "registered file changed"):
            self.m.verify_registration(self.root)

    def test_registered_shared_asset_directory_and_hash_check(self):
        with tempfile.TemporaryDirectory() as external:
            asset_dir = self.root / "outputs/image_experiments/assets"
            name = "outputs/image_experiments/assets/dinov2/model.safetensors"
            extra = self.root / name; extra.parent.mkdir(); extra.write_bytes(b"registered DINO fixture")
            research = self.m.old.read(self.root / "research.json")
            research["assets"]["test"] = dict(path=name, sha256=sha256(extra))
            self.m.old.write(self.root / "research.json", research)
            self.required_files += (name,); self.register()
            target = Path(external) / "shared-assets"
            asset_dir.rename(target); asset_dir.symlink_to(target, target_is_directory=True)
            self.assertFalse((self.root / self.m.BASELINE_ASSETS[0]).is_symlink())
            self.m.verify_registration(self.root)
            (target / "resnet50-0676ba61.pth").write_bytes(b"changed weight")
            with self.assertRaisesRegex(ValueError, "registered file changed"):
                self.m.verify_registration(self.root)

    def test_source_and_data_symlink_escapes_are_rejected(self):
        for name in (self.m.RUNTIME_CODE[0], "data/selection.json", "research.json"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as external:
                path = self.root / name; target = Path(external) / "outside"
                path.rename(target); path.symlink_to(target)
                try:
                    with self.assertRaisesRegex(ValueError, "registered file changed"):
                        self.m.verify_registration(self.root)
                finally:
                    path.unlink(); target.rename(path)

    def test_asset_escape_requires_explicit_approved_asset_binding(self):
        with tempfile.TemporaryDirectory() as external:
            target = Path(external) / "weight.bin"; target.write_bytes(b"not an approved asset")
            name = "outputs/image_experiments/assets/unapproved.bin"
            (self.root / name).symlink_to(target)
            registry = self.m.old.read(self.folder / "registration.json")
            registry["files_sha256"][name] = sha256(target)
            self.m.old.write(self.folder / "registration.json", registry)
            with self.assertRaisesRegex(ValueError, "registered file changed"):
                self.m.verify_registration(self.root)

    def test_registration_paths_must_be_relative_without_parent_traversal(self):
        original = self.m.old.read(self.folder / "registration.json")
        name = self.m.BASELINE_ASSETS[0]
        for key in (str(self.root / name), name.replace("/assets/", "/assets/../assets/")):
            with self.subTest(key=key):
                registry = dict(files_sha256=dict(original["files_sha256"], **{key: sha256(self.root / name)}))
                self.m.old.write(self.folder / "registration.json", registry)
                with self.assertRaisesRegex(ValueError, "registered file changed"):
                    self.m.verify_registration(self.root)

    def test_runtime_registration_omission_fails(self):
        registry = self.m.old.read(self.folder / "registration.json")
        # Existing draft verifies only these three paths, leaving all executable code unbound.
        registry["files_sha256"] = {k: v for k, v in registry["files_sha256"].items()
            if k in ("monitor/config.json", "parent.jsonl", "research.json")}
        self.m.old.write(self.folder / "registration.json", registry)
        with self.assertRaisesRegex(ValueError, "missing extension file/hash registration"):
            self.m.verify_registration(self.root)

    def test_research_dataset_and_parent_bindings_cannot_be_omitted(self):
        original = self.m.old.read(self.root / "research.json")
        for key in ("dataset", "parent_manifest"):
            with self.subTest(key=key):
                changed = dict(original); changed.pop(key)
                self.m.old.write(self.root / "research.json", changed); self.register()
                with self.assertRaisesRegex(ValueError, "research dataset/parent registration mismatch"):
                    self.m.verify_registration(self.root)

    def test_research_selection_path_and_hash_must_match_registry(self):
        original = self.m.old.read(self.root / "research.json")
        for changes in (dict(path="old-data"), dict(selection_sha256="wrong"), dict(selection_sha256=None)):
            with self.subTest(changes=changes):
                changed = dict(original, dataset=dict(original["dataset"], **changes))
                self.m.old.write(self.root / "research.json", changed); self.register()
                with self.assertRaisesRegex(ValueError, "research dataset/parent registration mismatch"):
                    self.m.verify_registration(self.root)

    def test_research_parent_path_and_hash_must_match_registry(self):
        original = self.m.old.read(self.root / "research.json")
        for changes in (dict(path="old-parent.jsonl"), dict(sha256="wrong"), dict(sha256=None)):
            with self.subTest(changes=changes):
                changed = dict(original, parent_manifest=dict(original["parent_manifest"], **changes))
                self.m.old.write(self.root / "research.json", changed); self.register()
                with self.assertRaisesRegex(ValueError, "research dataset/parent registration mismatch"):
                    self.m.verify_registration(self.root)

    def test_runtime_settings_are_exact(self):
        original = self.m.old.read(self.folder / "config.json")
        for key, value in (("route", "auto"), ("workers", 8), ("epochs", 20), ("gpu_index", 0), ("per_cell", 100), ("parent_manifest", "")):
            with self.subTest(key=key):
                self.m.old.write(self.folder / "config.json", dict(original, **{key: value}))
                with self.assertRaises(ValueError):
                    self.m.settings(self.root)

    def test_pipeline_command_never_uses_old_shell(self):
        args = self.m.command(self.root, "pipeline")
        self.assertEqual(args[1], "scripts/run_iab_extension_m2.sh")
        self.assertNotIn("scripts/run_iab_m2.sh", args)
        self.assertEqual(args[-2:], ["30", "1"])

    def test_busy_gpu_blocks_pipeline_and_feature(self):
        self.m.old.write(self.folder / "download.json", dict(status="completed", exit_code=0))
        for feature in (False, True):
            with self.subTest(feature=feature), patch.object(self.m, "verify_download"), patch.object(
                self.m, "certify_baseline", return_value=feature), patch.object(self.m.old, "gpu", return_value=dict(index=1, used_mib=2048)), patch.object(
                self.m, "launch", side_effect=AssertionError("busy GPU pipeline launch")), patch.object(
                self.m.subprocess, "run", side_effect=AssertionError("busy GPU feature launch")):
                report = {}; self.m.advance(self.root, report)
                self.assertIn("waiting", report)
                self.assertFalse((self.folder / "pipeline.json").exists())
                self.assertFalse((self.root / "features/launch.json").exists())

    def test_failed_unknown_and_nonzero_completed_states_are_preserved(self):
        path = self.folder / "download.json"
        for status in ("failed", "running", "launch_intent", "completed"):
            with self.subTest(status=status), patch.object(self.m, "launch", side_effect=AssertionError("state restarted")):
                self.m.old.write(path, dict(status=status, exit_code=7))
                before = path.read_bytes()
                report = {}; self.m.advance(self.root, report)
                self.assertIn("needs_attention", report)
                self.assertEqual(path.read_bytes(), before)

    def test_corrupt_worker_state_cannot_be_treated_as_absent(self):
        path = self.folder / "download.json"; path.write_text("{partial")
        with self.assertRaisesRegex(ValueError, "unreadable state"):
            self.m.advance(self.root, {})
        self.assertEqual(path.read_text(), "{partial")

    def test_empty_worker_state_is_preserved_for_attention(self):
        path = self.folder / "download.json"; path.write_text("{}")
        with patch.object(self.m, "verify_download"), patch.object(self.m, "launch", side_effect=AssertionError("empty state restarted")):
            report = {}; self.m.advance(self.root, report)
        self.assertIn("needs_attention", report)
        self.assertEqual(path.read_text(), "{}")

    def test_empty_feature_launch_is_preserved_for_attention(self):
        self.m.old.write(self.folder / "download.json", dict(status="completed", exit_code=0))
        path = self.root / "features/launch.json"
        self.m.old.write(path, {})
        with patch.object(self.m, "verify_download"), patch.object(self.m, "certify_baseline", return_value=True):
            report = {}; self.m.advance(self.root, report)
        self.assertIn("needs_attention", report)
        self.assertEqual(self.m.old.read(path), {})

    def test_empty_existing_certificate_is_rejected(self):
        self.m.old.write(self.folder / "completion.json", {})
        with self.assertRaisesRegex(ValueError, "completed exit-0 pipeline"):
            self.m.certify_baseline(self.root)

    def test_live_adoption_pid_reuse_and_exclusive_lock(self):
        with patch.object(self.m.old, "identity", side_effect=lambda pid: ["boot", "start"] if pid == os.getpid() else None), patch.object(
            self.m, "launch", side_effect=AssertionError("duplicate")):
            state = dict(status="running", pid=os.getpid(), identity=["boot", "start"])
            self.m.old.write(self.folder / "download.json", state)
            report = {}; self.m.advance(self.root, report)
            self.assertIn("waiting", report)
            state["identity"] = ["different boot", "reused pid"]
            self.m.old.write(self.folder / "download.json", state)
            report = {}; self.m.advance(self.root, report)
            self.assertIn("needs_attention", report)
        with self.m.old.lock(self.folder / "supervisor.lock") as first:
            with self.m.old.lock(self.folder / "supervisor.lock") as second:
                self.assertIsNotNone(first); self.assertIsNone(second)

    def test_unregistered_worker_does_not_run(self):
        with self.assertRaises(PermissionError), patch.object(self.m.old, "worker", side_effect=AssertionError("unregistered")):
            self.m.worker(self.root, "pipeline")

    def test_worker_records_real_subprocess_failure(self):
        state = dict(status="running", pid=os.getpid(), identity=["boot", "start"])
        self.m.old.write(self.folder / "pipeline.json", state)
        with patch.object(self.m.old, "identity", return_value=state["identity"]), patch.object(
            self.m, "verify_download"), patch.object(self.m, "command", return_value=[sys.executable, "-c", "raise SystemExit(7)"]):
            original = self.m.old.command
            try:
                self.assertEqual(self.m.worker(self.root, "pipeline"), 7)
            finally:
                self.m.old.command = original
        self.assertEqual(self.m.old.read(self.folder / "pipeline.json")["exit_code"], 7)
        self.assertEqual(self.m.old.read(self.folder / "pipeline.json")["status"], "failed")
        self.assertFalse((self.folder / "completion.json").exists())

    def baseline_certificate(self):
        manifest = self.root / "data/manifest.jsonl"; manifest.write_text("frozen manifest")
        summary = "outputs/image_experiments/iab-m2-test-v7-summary.json"
        record = dict(status="completed", manifest_sha256=sha256(manifest))
        self.m.old.write(self.root / summary, record)
        self.m.old.write(self.folder / "validated-summary.json", record)
        self.m.old.write(self.folder / "completion.json", dict(record, summary=summary))
        self.m.old.write(self.folder / "pipeline.json", dict(status="completed", exit_code=0))

    def test_completion_requires_matching_summary_hashes_and_exit_zero(self):
        self.baseline_certificate()
        self.assertTrue(self.m.certify_baseline(self.root, seal=True))
        self.assertTrue(self.m.certify_baseline(self.root))
        certificate = self.m.old.read(self.folder / "completion.json")
        self.assertFalse(certificate["final_project_completed"])
        self.assertEqual(certificate["completion_scope"], "extension_baseline_only")
        self.m.old.write(self.folder / "validated-summary.json", dict(status="completed", manifest_sha256="changed"))
        with self.assertRaisesRegex(ValueError, "manifest mismatch"):
            self.m.certify_baseline(self.root)

    def test_equal_manifest_with_different_summary_metrics_is_rejected(self):
        self.baseline_certificate()
        path = self.folder / "validated-summary.json"
        validated = self.m.old.read(path)
        validated["metrics"] = {"accuracy": .999}
        self.m.old.write(path, validated)
        with self.assertRaisesRegex(ValueError, "summary content mismatch"):
            self.m.certify_baseline(self.root, seal=True)

    def test_completed_feature_stops_at_awaiting_analysis(self):
        self.m.old.write(self.folder / "download.json", dict(status="completed", exit_code=0))
        digest = sha256(self.root / "research.json")
        self.m.old.write(self.root / "features/state.json", dict(status="completed", exit_code=0, config_sha256=digest))
        self.m.old.write(self.root / "features/summary.json", dict(status="completed", config_sha256=digest))
        with patch.object(self.m, "verify_download"), patch.object(self.m, "certify_baseline", return_value=True):
            report = {}; self.m.advance(self.root, report)
        self.assertEqual(report["status"], "awaiting_analysis")
        self.assertFalse(report["final_project_completed"])

    def test_failed_feature_does_not_restart(self):
        self.m.old.write(self.folder / "download.json", dict(status="completed", exit_code=0))
        path = self.root / "features/state.json"
        self.m.old.write(path, dict(status="failed", exit_code=2))
        before = path.read_bytes()
        with patch.object(self.m, "verify_download"), patch.object(self.m, "certify_baseline", return_value=True), patch.object(
            self.m.subprocess, "run", side_effect=AssertionError("failed feature restarted")):
            report = {}; self.m.advance(self.root, report)
        self.assertIn("needs_attention", report)
        self.assertEqual(path.read_bytes(), before)

    def test_snapshot_has_no_network_probe(self):
        with patch.object(self.m.old, "gpu", return_value=dict(index=1, used_mib=0)), patch.object(
            self.m.old, "probe_transports", side_effect=AssertionError("network probe")):
            report = self.m.snapshot(self.root)
        self.assertIn("no probes", report["transport_validation"])

    def test_pipeline_checks_extension_before_training(self):
        scripts = self.root / "scripts"; scripts.mkdir(exist_ok=True)
        (scripts / "run_iab_extension_m2.sh").write_bytes((ROOT / "scripts/run_iab_extension_m2.sh").read_bytes())
        runtime = self.root / ".venv-image/bin"; runtime.mkdir(parents=True)
        calls = self.root / "calls.jsonl"
        py = runtime / "python"
        py.write_text("#!" + sys.executable + "\nimport sys,json\nfrom pathlib import Path\n" +
            f"with Path({str(calls)!r}).open('a') as f:f.write(json.dumps(sys.argv[1:])+'\\n')\n" +
            "if '--check-existing' in sys.argv:raise SystemExit(7)\n")
        py.chmod(0o755)
        # flock is Linux-only; this test isolates pipeline routing, not OS lock semantics.
        flock = runtime / "flock"; flock.write_text("#!/bin/sh\nexit 0\n"); flock.chmod(0o755)
        (self.root / "data/manifest.jsonl").write_text("frozen")
        env = dict(os.environ, PATH=str(runtime) + os.pathsep + os.environ["PATH"])
        result = subprocess.run(["bash", "scripts/run_iab_extension_m2.sh", "data", "test", "summary", "parent.jsonl", "30", "1"], cwd=self.root, env=env, capture_output=True)
        self.assertEqual(result.returncode, 7, result.stderr)
        recorded = [json.loads(line) for line in calls.read_text().splitlines()]
        self.assertTrue(any("scripts/prepare_iab_extension_manifest.py" in c and "--check-existing" in c for c in recorded))
        self.assertFalse(any("scripts/run_image_experiment.py" in c for c in recorded))


if __name__ == "__main__":
    unittest.main()
