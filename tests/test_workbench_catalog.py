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
    c = catalog(tmp_path)
    # Aggregate results still load; without the method evidence files the metadata is "not registered",
    # reported as a partial catalog, and never filled in with a guess.
    assert c["groups"] and c["status"] == "partial" and {e["code"] for e in c["errors"]} == {"artifact_mismatch"}
    a0 = next(m for m in c["methods"] if m["id"] == "attr-resnet50-q179-v1")
    assert a0["train_n"] is None and a0["selection_n"] is None


def _rows(c):
    return {(g["id"], r["method"]): r for g in c["groups"] for r in g["rows"]}


@pytest.mark.skipif(not (ROOT / CATALOG["overview"]["path"]).is_file(), reason="sealed overview not present")
def test_training_metadata_comes_from_sealed_evidence_without_guessing():
    c = catalog()
    methods = {m["id"]: m for m in c["methods"]}
    ep = lambda e: (e["value"], e["status"])  # noqa: E731
    a0 = methods["attr-resnet50-q179-v1"]
    assert ep(a0["training_epochs"]) == (30, "registered") and ep(a0["selected_epoch"]) == (29, "registered")
    assert a0["train_n"] == 23348 and any("parent dev" in n for n in a0["notes"])
    det = methods["det-openclip-q179-v1"]
    assert (det["train_n"], det["selection_n"], ep(det["selected_epoch"])) == (37338, 2311, (30, "registered"))
    assert methods["research-openclip-ffhq"]["train_n"] == 33577 and "held-FFHQ" in methods["research-openclip-ffhq"]["protocol"]
    assert methods["research-openclip-open20"]["train_n"] == 21236 and "20 已知类" in methods["research-openclip-open20"]["protocol"]
    assert ep(methods["research-ofa-tplus1"]["selected_epoch"]) == (18, "registered")
    # best epoch registered but the run length is not: shown as not registered, not as 30
    for mid, best in (("research-rgb22", 28), ("research-fingerprint22", 22), ("research-feature-fusion22", 29),
                      ("research-dino-detection", 30)):
        assert ep(methods[mid]["training_epochs"]) == (None, "not_registered")
        assert ep(methods[mid]["selected_epoch"]) == (best, "registered")
    for mid in ("research-resnet-control", "research-resnet-ffhq", "research-resnet-open20", "research-ofa-pair"):
        assert ep(methods[mid]["training_epochs"]) == (None, "not_registered")
    for mid in ("research-openclip-knn22", "research-omnidfa-k10", "research-omniaid-v2",
                "research-score-fusion-uniform", "research-score-fusion-calibrated"):
        for key in ("training_epochs", "selected_epoch"):
            assert methods[mid][key]["value"] is None and methods[mid][key]["status"] == "not_applicable"
        assert "只训练头" not in methods[mid]["backbone"]
    values = [m[k]["value"] for m in c["methods"] for k in ("training_epochs", "selected_epoch")]
    assert 0 not in values and all(v is None or v > 0 for v in values)


@pytest.mark.skipif(not (ROOT / CATALOG["overview"]["path"]).is_file(), reason="sealed overview not present")
def test_result_rows_carry_metadata_and_keep_protocols_apart():
    c = catalog()
    rows = _rows(c)
    groups = {g["id"]: g for g in c["groups"]}
    dino = next(r for r in groups["detection-clean"]["rows"] if r["method"].startswith("DINOv2"))
    assert dino["training_epochs"]["status"] == "not_registered" and dino["selected_epoch"]["value"] == 30
    knn = next(r for r in groups["source-1331"]["rows"] if r["method"].startswith("OpenCLIP KNN"))
    assert knn["training_epochs"]["status"] == "not_applicable"
    proto = next(r for r in groups["unknown-1275"]["rows"] if "原型" in r["method"])
    assert any("原型拒识分数本身无训练轮次" in n for n in proto["meta_notes"])
    assert {r["n"] for r in groups["detection-clean"]["rows"]} == {2310}
    assert {r["n"] for r in groups["detection-ffhq"]["rows"]} == {225}
    assert {r["n"] for r in groups["unknown-1275"]["rows"]} == {1275}
    assert "完整 AI dev 4197" in groups["source-full-dev"]["rows"][0]["protocol"]
    assert "父 dev" in next(r for r in groups["source-1331"]["rows"] if r["role"] == "default")["protocol"]
    assert all("training_epochs" in r for r in rows.values())
    assert c["key_facts_primary"] == 4 and any("56" in f and "15" in f for f in c["key_facts"][:4])
    assert any("58/1212" in f for f in c["key_facts"][:4]) and any("父 dev" in f for f in c["key_facts"][:4])
