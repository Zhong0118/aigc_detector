"""CPU fixtures guard provenance and paired development comparisons."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/compare_image_scale_epochs.py"
spec = importlib.util.spec_from_file_location("scale_comparison", SCRIPT)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def row(sid, label, split="dev"):
    return dict(sample_id=sid, split=split, group_id="group-" + sid[-1],
                is_ai=int(label != "real"), source_label=None if label == "real" else label,
                sha256="bytes-" + sid, pixel_sha256="pixels-" + sid,
                semantic_class="scene", prompt_id=None)


def prediction(r, classes, scores, task="A0"):
    return dict(sample_id=r["sample_id"], split="dev", status="ok",
                true_label=("ai" if r["is_ai"] else "real") if task == "D0" else r["source_label"],
                classes=classes, **{("match_scores" if task == "OFA" else "scores"): scores})


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.rows = {r["sample_id"]: r for r in [row("a0", "a"), row("b0", "b")]}
        self.preds = [prediction(self.rows["a0"], ["a", "b"], [.9, .1]),
                      prediction(self.rows["b0"], ["a", "b"], [.2, .8])]

    def test_align_uses_ids_not_file_order(self):
        ids, y, a, b, classes = comparison.align(self.rows, self.preds, self.preds[::-1], "A0")
        self.assertEqual(ids, ["a0", "b0"])
        self.assertEqual(classes, ["a", "b"])
        np.testing.assert_array_equal(y, [0, 1])
        np.testing.assert_array_equal(a, b)

    def test_wrong_truth_duplicate_missing_or_foreign_id_is_rejected(self):
        mutations = []
        bad = copy.deepcopy(self.preds); bad[0]["true_label"] = "b"; mutations.append(bad)
        mutations.extend([self.preds[:1], self.preds + self.preds[:1]])
        bad = copy.deepcopy(self.preds); bad[0]["sample_id"] = "foreign"; mutations.append(bad)
        for records in mutations:
            with self.subTest(records=records), self.assertRaises(ValueError):
                comparison.align(self.rows, self.preds, records, "A0")

    def test_vocabulary_order_nondev_and_bad_scores_are_rejected(self):
        for field, value in [("classes", ["b", "a"]), ("split", "holdout"),
                             ("scores", [.2, .2]), ("scores", [float("nan"), .1])]:
            bad = copy.deepcopy(self.preds); bad[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                comparison.align(self.rows, self.preds, bad, "A0")

    def test_manifest_extension_preserves_all_identity_fields(self):
        for field in ["split", "group_id", "is_ai", "source_label", "sha256", "pixel_sha256"]:
            changed = copy.deepcopy(self.rows); changed["a0"][field] = "changed"
            with self.subTest(field=field), self.assertRaises(ValueError):
                comparison.verify_extension(self.rows, changed)
        extended = dict(self.rows, extra=row("extra", "a"))
        self.assertEqual(comparison.verify_extension(self.rows, extended), 2)

    def test_detection_auc_and_threshold_use_named_ai_column(self):
        actual = comparison.stats(np.array([1, 1, 0, 0]),
                                  np.array([[.1, .9], [.8, .2], [.7, .3], [.9, .1]]),
                                  "D0", ["ai", "real"])
        self.assertAlmostEqual(actual["auroc"], .75)
        self.assertAlmostEqual(actual["accuracy"], .75)
        self.assertAlmostEqual(actual["fpr_0_5"], .5)
        self.assertAlmostEqual(actual["tpr_0_5"], 1.)

    def test_ofa_accepts_independent_match_scores_and_keeps_auc_separate(self):
        preds = [prediction(self.rows["a0"], ["a", "b"], [.9, .8], "OFA"),
                 prediction(self.rows["b0"], ["a", "b"], [.7, .9], "OFA")]
        _, y, p, _, classes = comparison.align(self.rows, preds, preds, "OFA")
        actual = comparison.stats(y, p, "OFA", classes)
        self.assertEqual(actual, {"n": 2, "macro_auroc": 1.0})

    def test_group_bootstrap_identical_predictions_have_zero_paired_delta(self):
        y = np.array([0, 1, 0, 1]); p = np.array([[.9,.1],[.2,.8],[.8,.2],[.7,.3]])
        actual = comparison.group_bootstrap(y, p, p, ["g1", "g1", "g2", "g2"])
        self.assertEqual(actual["repeats"], 1000)
        self.assertEqual(actual["group_n"], 2)
        self.assertEqual(actual["delta_95_ci"], {"top1": [0.,0.], "top3": [0.,0.], "macro_f1": [0.,0.]})

    def test_new_scale_requires_doubled_source_semantic_cells(self):
        with self.assertRaises(ValueError):
            comparison.verify_cell_scale(self.rows, self.rows)
        extra = {r["sample_id"]:r for r in [row("a1","a"), row("b1","b")]}
        self.assertEqual(comparison.verify_cell_scale(self.rows, dict(self.rows, **extra)), 2)

    def test_auc_counts_equal_scores_as_half_a_win(self):
        self.assertEqual(comparison.auc([False, True], [.5, .5]), .5)

    def test_cli_pending_missing_new_run_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "inputs"; base.mkdir()
            records = [row("r0", "real"), row("r1", "real"), row("a0", "a"),
                       row("a1", "a"), row("b0", "b"), row("b1", "b"),
                       row("ta", "a", "train"), row("tb", "b", "train")]
            manifest = base / "iab-balanced-v5-50/manifest.jsonl"
            manifest.parent.mkdir(); manifest.write_text("".join(json.dumps(r)+"\n" for r in records))
            manifest_sha = comparison.sha256(manifest)
            for suffix, epochs in [("v5-50", 12), ("v5-50-e20", 20)]:
                for task, prefix in comparison.RUNS.items():
                    folder = base / f"{prefix}-iab-{suffix}"; folder.mkdir()
                    selected = [r for r in records if r["split"] == "dev" and (task == "D0" or r["is_ai"])]
                    classes = ["real", "ai"] if task == "D0" else ["a", "b"]
                    preds = []
                    for r in selected:
                        label = ("ai" if r["is_ai"] else "real") if task == "D0" else r["source_label"]
                        preds.append(prediction(r, classes, [.9,.1] if label==classes[0] else [.1,.9], task))
                    (folder/"predictions.jsonl").write_text("".join(json.dumps(p)+"\n" for p in preds))
                    config = dict(manifest_sha256=manifest_sha, epochs=epochs, seed=42, batch_size=32,
                                  pretrained_sha256="same-"+task, training_protocol="fixed", ofa_denoiser_microbatch=1)
                    metrics = dict(holdout_evaluated=False, selected_epoch=1, classes=classes)
                    history = [dict(epoch=i, lr=.001, seconds=1, train={"n":2},
                                    dev={"n":len(selected), "auroc":1., "macro_f1":1.}) for i in range(1,epochs+1)]
                    for name, data in [("resolved_config.json", config), ("metrics.json", metrics),
                                       ("status.json", dict(status="completed",exit_code=0)), ("history.json",history)]:
                        (folder/name).write_text(json.dumps(data))
                    if task == "OFA":
                        (folder/"reference_manifest.json").write_text(json.dumps({"a":["ta"],"b":["tb"]}))
                    assets = {p.name:{"sha256":comparison.sha256(p)} for p in folder.iterdir()}
                    (folder/"asset_manifest.json").write_text(json.dumps(assets))
            output = Path(tmp) / "report"
            args = ["--base", str(base), "--output", str(output), "--expected-anchor-sha", manifest_sha,
                    "--expected-d0", "6", "--expected-attribution", "4", "--expected-sources", "2"]
            self.assertEqual(comparison.main(args), 2)
            summary = json.loads((output/"summary.json").read_text())
            self.assertEqual(summary["status"], "pending")
            self.assertFalse(summary["holdout_evaluated"])
            self.assertEqual(summary["conditions"]["100/20"]["status"], "pending")
            self.assertEqual(summary["paired"]["50/12→50/20"]["A0"]["delta"]["top1"], 0.)
            self.assertTrue((output/"epochs.csv").exists())
            self.assertFalse((output/"completion.json").exists())
            self.assertEqual(comparison.main(args), 1)

            # All three conditions now exist. New dev must remain distinct from
            # the paired anchor dev, even when predictions happen to be equal.
            extra = []
            for original in records:
                r = copy.deepcopy(original)
                r.update(sample_id="new-"+r["sample_id"], group_id="new-"+r["group_id"],
                         sha256="new-"+r["sha256"], pixel_sha256="new-"+r["pixel_sha256"])
                extra.append(r)
            manifest100 = base / "iab-balanced-v6-100/manifest.jsonl"
            manifest100.parent.mkdir()
            manifest100.write_text("".join(json.dumps(r)+"\n" for r in records+extra))
            for task, prefix in comparison.RUNS.items():
                folder = base/f"{prefix}-iab-v6-100-e20"
                shutil.copytree(base/f"{prefix}-iab-v5-50-e20", folder)
                config = json.loads((folder/"resolved_config.json").read_text())
                config["manifest_sha256"] = comparison.sha256(manifest100)
                (folder/"resolved_config.json").write_text(json.dumps(config))
                old_preds = [json.loads(line) for line in (folder/"predictions.jsonl").read_text().splitlines()]
                more = copy.deepcopy(old_preds)
                for p in more:
                    p["sample_id"] = "new-" + p["sample_id"]
                (folder/"predictions.jsonl").write_text("".join(json.dumps(p)+"\n" for p in old_preds+more))
                history = json.loads((folder/"history.json").read_text())
                for entry in history:
                    entry["dev"]["n"] *= 2
                (folder/"history.json").write_text(json.dumps(history))
                assets = {p.name:{"sha256":comparison.sha256(p)} for p in folder.iterdir() if p.name != "asset_manifest.json"}
                (folder/"asset_manifest.json").write_text(json.dumps(assets))
            complete_output = Path(tmp)/"complete-report"
            complete_args = list(args); complete_args[3] = str(complete_output)
            self.assertEqual(comparison.main(complete_args), 0)
            final = json.loads((complete_output/"summary.json").read_text())
            self.assertEqual(final["same_dev"]["100/20"]["A0"]["n"], 4)
            self.assertEqual(final["full_dev"]["100/20"]["A0"]["n"], 8)
            self.assertEqual(len(final["paired"]), 3)
            self.assertTrue((complete_output/"completion.json").exists())
            # Tampering with saved evidence cannot be downgraded to pending.
            target = base/"a0-resnet50-iab-v6-100-e20/predictions.jsonl"
            target.write_text(target.read_text()+"\n")
            invalid_args = list(args); invalid_args[3] = str(Path(tmp)/"invalid-report")
            self.assertEqual(comparison.main(invalid_args), 1)
            self.assertEqual(json.loads((Path(tmp)/"invalid-report/summary.json").read_text())["status"], "invalid")


if __name__ == "__main__":
    unittest.main()
