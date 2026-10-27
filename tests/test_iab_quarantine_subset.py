"""Offline curation tests: no network, GPU, image decoding, or model inference."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.image_attribution.protocol import sha256
from scripts.prepare_iab_extension_manifest import selected_receipts


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


class QuarantineSubsetTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / "scripts/curate_iab_quarantine_subset.py"
        self.assertTrue(path.exists(), "approved subset curation entry is not implemented")
        spec = importlib.util.spec_from_file_location("curate_iab_quarantine_subset", path)
        self.m = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.m)
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); raw = self.root / "raw"
        hashes = [0, 255, 15, 0xf0f0f0f0f0f0f0f0, 0x5555555555555555, 0xaaaaaaaaaaaaaaaa]
        self.rows = []; entries = {}; self.receipts = []
        for i in range(6):
            source = "model" if i < 5 else "real"; name = f"{source}/cat/{i}.png"
            sid = "iab-" + hashlib.sha256(f"{source}/cat/{name}".encode()).hexdigest()[:24]
            rel = f"images/{source}/cat/{i}.png"; image = raw / rel
            image.parent.mkdir(parents=True, exist_ok=True); image.write_bytes(f"fixture-{i}".encode())
            row = dict(sample_id=sid, relative_path=rel, sha256=sha256(image), pixel_sha256=f"pixel-{i}",
                phash=f"{hashes[i]:016x}", is_ai=int(source != "real"), source_label=source if source != "real" else None,
                semantic_class="cat", prompt_id=None, split=None, group_id=None,
                source_evidence=dict(zip_member=name, zip_crc32=i))
            write(image.with_suffix(".json"), row); self.rows.append(row)
            entries.setdefault(f"{source}/cat", []).append(dict(filename=name, crc32=i, size=image.stat().st_size,
                compressed_size=image.stat().st_size))
        for source in ("model", "real"):
            rows = [r for r in self.rows if (r['source_label'] or 'real') == source]
            receipt = dict(source=source, semantic="cat", selected=len(rows), rows=rows)
            write(raw / f"receipts/{source}.json", receipt); self.receipts.append(receipt)
        counts = {k:len(v) for k,v in entries.items()}
        write(raw / "selection.json", dict(per_cell=200, entries=entries, audit=dict(selected_by_cell=counts,
            requested_ai_per_source_semantic=200, ai_selected=5, real_selected=1)))
        write(raw / "download_status.json", dict(status="completed", selected=6, selected_by_cell=counts))
        parent = [dict(self.rows[i], split=split, group_id=self.rows[i]['sample_id']) for i,split in
            [(0,"train"),(1,"dev"),(5,"holdout")]]
        self.parent = self.root / "parent.jsonl"
        self.parent.write_text("".join(json.dumps(r)+"\n" for r in parent))
        write(self.root / "excluded.json", [self.rows[2]['sample_id']])
        self.cfg = dict(source=dict(path="raw", selection_sha256=sha256(raw/'selection.json'),
            status_sha256=sha256(raw/'download_status.json')), parent_manifest=dict(path="parent.jsonl", sha256=sha256(self.parent)),
            quarantine=dict(path="excluded.json", sha256=sha256(self.root/'excluded.json'), count=1),
            expected_source_samples=6, expected_output_samples=5, expected_cells=2, seed=42, output="derived",
            authorization="Explicit human approval to quarantine only registered new samples")
        self.config = self.root / "curation.json"; write(self.config, self.cfg)

    def run_curation(self):
        write(self.config, self.cfg)
        return self.m.curate(self.root, self.config)

    def test_only_new_registered_sample_excluded_all_parent_files_hardlinked(self):
        before = {p:sha256(p) for p in (self.root/'raw').rglob('*') if p.is_file()}
        report = self.run_curation(); out = self.root/'derived'
        self.assertEqual(report['selected'], 5); self.assertEqual(report['quarantined'], 1)
        self.assertEqual(report['network_requests'], 0); self.assertFalse(report['holdout_scored'])
        selection = json.loads((out/'selection.json').read_text()); status = json.loads((out/'download_status.json').read_text())
        receipts = [json.loads(p.read_text()) for p in (out/'receipts').glob('*.json')]
        rows = selected_receipts(selection,status,receipts)
        self.assertEqual({r['sample_id'] for r in rows}, {r['sample_id'] for r in self.rows} - {self.rows[2]['sample_id']})
        for row in rows:
            for path in [Path(row['relative_path']), Path(row['relative_path']).with_suffix('.json')]:
                self.assertTrue((out/path).samefile(self.root/'raw'/path))
        self.assertFalse((out/self.rows[2]['relative_path']).exists())
        self.assertTrue((self.root/'raw'/self.rows[2]['relative_path']).exists())
        self.assertTrue(all(sha256(p)==digest for p,digest in before.items()))
        self.assertFalse((out/'manifest.jsonl').exists())  # Existing anchored preparer owns the integrity/freeze gate.

    def test_parent_exclusion_rejected_before_output_creation(self):
        write(self.root/'excluded.json',[self.rows[0]['sample_id']]);self.cfg['quarantine']['sha256']=sha256(self.root/'excluded.json')
        with self.assertRaisesRegex(ValueError,'parent'):self.run_curation()
        self.assertFalse((self.root/'derived').exists())

    def test_unregistered_or_duplicate_exclusion_rejected(self):
        for ids in [['unknown'],[self.rows[2]['sample_id']]*2]:
            with self.subTest(ids=ids):
                write(self.root/'excluded.json',ids);self.cfg['quarantine']['sha256']=sha256(self.root/'excluded.json')
                with self.assertRaises(ValueError):self.run_curation()
                self.assertFalse((self.root/'derived').exists())

    def test_still_bridging_subset_fails_before_output_creation(self):
        write(self.root/'excluded.json',[self.rows[3]['sample_id']]);self.cfg['quarantine']['sha256']=sha256(self.root/'excluded.json')
        with self.assertRaisesRegex(ValueError,'frozen split conflict'):self.run_curation()
        self.assertFalse((self.root/'derived').exists())

    def test_tampered_bound_input_rejected(self):
        (self.root/'raw/selection.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'hash'):self.run_curation()
        self.assertFalse((self.root/'derived').exists())

    def test_existing_output_is_never_overwritten(self):
        out=self.root/'derived';out.mkdir();(out/'keep').write_text('unchanged')
        with self.assertRaises(FileExistsError):self.run_curation()
        self.assertEqual((out/'keep').read_text(),'unchanged')

    def test_missing_sidecar_cannot_produce_completed_status(self):
        (self.root/'raw'/self.rows[4]['relative_path']).with_suffix('.json').unlink()
        with self.assertRaises(FileNotFoundError):self.run_curation()
        self.assertFalse((self.root/'derived/download_status.json').exists())

    def test_source_or_output_escape_is_rejected(self):
        self.cfg['output']='../outside'
        with self.assertRaisesRegex(ValueError,'workspace'):self.run_curation()

    def test_symlink_alias_cannot_place_output_inside_raw_source(self):
        (self.root/'alias').symlink_to(self.root/'raw',target_is_directory=True)
        self.cfg['output']='alias/nested'
        with self.assertRaisesRegex(ValueError,'independent'):self.run_curation()
        self.assertFalse((self.root/'raw/nested').exists())
