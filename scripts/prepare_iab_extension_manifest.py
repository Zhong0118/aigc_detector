#!/usr/bin/env python3
"""Freeze an anchored IAB expansion using extension split protocol v2."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.image_attribution.extension_split import group_and_split_extension_v2
from research.image_attribution.protocol import sha256, validate_manifest
from scripts.prepare_iab_manifest import validate_receipts


def selected_receipts(selection, status, receipts):
    rows = validate_receipts(selection, status, receipts)
    expected = {}
    for cell, entries in selection["entries"].items():
        source, semantic = cell.split("/", 1)
        for entry in entries:
            sid = "iab-" + hashlib.sha256(f"{source}/{semantic}/{entry['filename']}".encode()).hexdigest()[:24]
            if sid in expected:
                raise ValueError("duplicate selection identity")
            expected[sid] = entry
    if set(expected) != {r["sample_id"] for r in rows}:
        raise ValueError("receipt sample identity differs from selection")
    for row in rows:
        entry = expected[row["sample_id"]]
        evidence = row["source_evidence"]
        if evidence.get("zip_member") != entry["filename"] or evidence.get("zip_crc32") != entry["crc32"]:
            raise ValueError("receipt ZIP member identity differs from selection")
    return rows


def validate_existing_split(rows, frozen, seed=42):
    expected, audit = group_and_split_extension_v2(rows, seed, frozen_rows=frozen)
    actual = {r["sample_id"]: r for r in rows}
    if len(actual) != len(rows):
        raise ValueError("duplicate manifest sample identity")
    for row in expected:
        before = actual[row["sample_id"]]
        if (before["split"], before["group_id"]) != (row["split"], row["group_id"]):
            raise ValueError("existing manifest split/group mismatch: " + row["sample_id"])
    return dict(status="verified", protocol=audit["protocol"], samples=len(rows), frozen_samples=len(frozen))


def integrity(rows, root):
    """Decode and rehash every image, including sealed holdout integrity metadata."""
    from PIL import Image
    import imagehash
    validate_manifest(rows, root)
    for row in rows:
        with Image.open(root / row["relative_path"]) as image:
            image.load()
            rgb = image.convert("RGB")
            pixels = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            if pixels != row["pixel_sha256"] or str(imagehash.phash(rgb)) != row["phash"]:
                raise ValueError("image integrity metadata changed: " + row["sample_id"])


def prepare(root, parent, seed=42, check_existing=False):
    root, parent = Path(root), Path(parent)
    manifest, audit_path = root / "manifest.jsonl", root / "audit.json"
    if not check_existing and (manifest.exists() or audit_path.exists()):
        raise FileExistsError("manifest or audit already frozen; preserve existing outputs")
    frozen = [json.loads(line) for line in parent.read_text().splitlines() if line.strip()]
    validate_manifest(frozen, parent.parent, verify_files=False)
    if check_existing:
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        validate_manifest(rows, root, verify_files=False)
        audit = json.loads(audit_path.read_text())
        if (audit.get("protocol") != "future-extension-split-v2" or
            audit.get("parent_manifest_sha256") != sha256(parent) or
            audit.get("manifest_sha256") != sha256(manifest) or
            audit.get("selection_sha256") != sha256(root / "selection.json") or audit.get("seed") != seed):
            raise ValueError("frozen extension audit/hash mismatch")
        return validate_existing_split(rows, frozen, seed)
    status = json.loads((root / "download_status.json").read_text())
    if status.get("status") != "completed":
        raise ValueError("acquisition incomplete")
    selection = json.loads((root / "selection.json").read_text())
    receipts = [json.loads(p.read_text()) for p in sorted((root / "receipts").glob("*.json"))]
    if len(receipts) != 230:
        raise ValueError("missing source/semantic cells")
    rows = selected_receipts(selection, status, receipts)
    rows, audit = group_and_split_extension_v2(rows, seed, frozen_rows=frozen)
    integrity(rows, root)
    with manifest.open("x") as stream:
        stream.write("".join(json.dumps(row) + "\n" for row in rows))
    audit.update(dataset="ImageAttributionBench", version="3.2", seed=seed,
        parent_manifest=str(parent.resolve()), parent_manifest_sha256=sha256(parent),
        manifest_sha256=sha256(manifest), selection_sha256=sha256(root / "selection.json"),
        selected=len(rows), requested_per_cell=selection["audit"]["requested_ai_per_source_semantic"],
        selected_by_cell=selection["audit"]["selected_by_cell"], real_selected=selection["audit"]["real_selected"],
        sources=sorted({r["source_label"] for r in rows if r["is_ai"]}),
        split_counts=dict(Counter(r["split"] for r in rows)),
        source_semantic_split_counts={f"{s}/{c}/{split}": n for (s,c,split), n in sorted(Counter(
            (r["source_label"] or "real", r["semantic_class"], r["split"]) for r in rows).items())},
        license="CC BY 4.0 (Dataverse V3.2 metadata); underlying dataset/model conditions retained",
        official_test_used=False, holdout_scored=False, whole_volume_md5_verified_for_all_inputs=False,
        rows_with_whole_archive_md5_verification=sum(bool(r["source_evidence"]["whole_volume_md5_verified"]) for r in rows),
        unique_byte_hashes=len({r["sha256"] for r in rows}), unique_pixel_hashes=len({r["pixel_sha256"] for r in rows}),
        limitations=["Development subset; single seed; no independent test claim", audit["ratio_note"],
            "Publisher prompt indices assumed aligned across generator folders", "Real filenames do not establish prompt correspondence",
            "Member CRC32 and retained image hashes; not whole archive MD5 verification", "Holdout integrity checked but never scored"])
    with audit_path.open("x") as stream:
        stream.write(json.dumps(audit, indent=2) + "\n")
    return {k:v for k,v in audit.items() if k not in ("near_duplicate_links", "source_semantic_split_counts")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-existing", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.dataset, args.parent_manifest, args.seed, args.check_existing)), flush=True)


if __name__ == "__main__":
    main()
