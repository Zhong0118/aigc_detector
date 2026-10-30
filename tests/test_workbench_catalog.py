"""Experiment catalog reads only the hash-checked sealed overview and keeps the key facts."""
import json
import shutil
from pathlib import Path

import pytest

from tests.test_workbench_image_engine import OVERLAY
from workbench.catalog import build_catalog

ROOT = Path(__file__).resolve().parents[1]
CATALOG = OVERLAY["catalog"]


def catalog(root=ROOT):
    return build_catalog(dict(research_root=str(root), catalog=CATALOG))


@pytest.mark.skipif(not (ROOT / CATALOG["overview"]["path"]).is_file(), reason="sealed overview not present")
def test_catalog_quotes_registered_counts_and_paired_intervals():
    c = catalog()
    assert c["status"] == "ok" and not c["errors"]
    groups = {g["id"]: g for g in c["groups"]}
    clean = {r["method"].split("（")[0]: r for r in groups["detection-clean"]["rows"]}
    assert (clean["D-OpenCLIP"]["fp"], clean["D-OpenCLIP"]["real_n"]) == (19, 979)
    ffhq = {r["method"]: r["fp"] for r in groups["detection-ffhq"]["rows"]}
    assert ffhq == {"ResNet-heldFFHQ": 56, "OpenCLIP-heldFFHQ": 15, "ConvNeXt-heldFFHQ": 56}
    cal = [r for r in groups["detection-ffhq"]["calibrated_rows"] if r["method"] == "OpenCLIP-heldFFHQ" and r["target_fpr"] == .01]
    assert cal[0]["fp"] == 7 and cal[0]["real_n"] == 93
    unknown = {r["method"]: r for r in groups["unknown-1275"]["rows"]}
    ofa = unknown["OFA20+1（已知质量）"]
    assert (ofa["known_retention_count"], ofa["known_correct_and_retained_count"], ofa["unknown_false_acceptance_count"]) \
        == (58, 28, 0) and ofa["known_n"] == 1212 and ofa["unknown_n"] == 63
    dna = next(p for p in groups["source-1331"]["paired"] if p["comparison"].startswith("DNA-Det 线性 − OpenCLIP"))
    assert round(dna["deltas"]["top1"]["delta"] * 100, 2) == 12.55
    assert [round(x * 100, 2) for x in dna["deltas"]["top1"]["ci95"]] == [9.20, 15.92]
    assert any("7/93" in fact for fact in c["key_facts"]) and any("58/1212" in fact for fact in c["key_facts"])
    assert "A3_unreviewed_family_mapping" in c["not_exposed"]
    assert "/home/" not in json.dumps(c, ensure_ascii=False)


def test_catalog_fails_closed_on_hash_mismatch(tmp_path):
    target = tmp_path / CATALOG["overview"]["path"]
    target.parent.mkdir(parents=True)
    target.write_text("{}")
    c = catalog(tmp_path)
    assert c["status"] == "unavailable" and c["groups"] == []
    assert {e["code"] for e in c["errors"]} == {"artifact_mismatch", "model_unavailable"}


@pytest.mark.skipif(not (ROOT / CATALOG["overview"]["path"]).is_file(), reason="sealed overview not present")
def test_catalog_does_not_require_per_image_predictions(tmp_path):
    for spec in CATALOG.values():
        target = tmp_path / spec["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / spec["path"], target)
    assert catalog(tmp_path)["status"] == "ok"
