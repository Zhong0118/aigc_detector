"""Experiment catalog reads only hash-registered results and keeps denominators explicit."""
import shutil
from pathlib import Path

import pytest
import yaml

from workbench.catalog import build_catalog

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "configs/workbench/preview.overlay.example.yaml"
RESEARCH = ROOT.parent / "attribution-pilot-baseline"


def settings(research_root):
    text = OVERLAY.read_text().replace("${RESEARCH_ROOT}", str(research_root))
    return yaml.safe_load(text)["workbench"]


needs_research = pytest.mark.skipif(not (RESEARCH / "outputs/image_experiments").is_dir(),
                                    reason="registered research outputs are only on the research server")


@needs_research
def test_registered_numbers_and_denominators():
    catalog = build_catalog(settings(RESEARCH))
    assert catalog["status"] == "ok" and not catalog["errors"]
    views = {v["id"]: v for v in catalog["views"]}
    main = views["detection"]["groups"][0]["rows"][0]
    assert (main["n"], main["negative_n"], main["fp"]) == (2310, 979, 19)
    assert round(main["auroc"] * 100, 4) == 99.8908 and round(main["fpr"] * 100, 4) == 1.9408
    ffhq = {r["model"]: r for r in views["detection"]["groups"][1]["rows"]}
    assert ffhq["D-OpenCLIP-FFHQ"]["fp"] == 15 and ffhq["ResNet50 D2-semantic"]["fp"] == 56
    a0 = views["attribution"]["groups"][0]["rows"][0]
    assert a0["n"] == 1331 and round(a0["top1"] * 100, 4) == 77.16 and round(a0["top3"] * 100, 4) == 92.9376
    resnet = views["rejection"]["groups"][0]["rows"][0]
    assert (resnet["known_retention_count"], resnet["known_n"]) == (576, 1212)
    assert (resnet["known_correct_and_retained_count"], resnet["unknown_rejection_count"], resnet["unknown_n"]) == (539, 59, 63)
    routes = {r["id"]: r for r in catalog["routes"]}
    assert set(routes) == {"ConvNeXt", "DNA-Det", "OmniDFA", "OFA T+1", "A3"}
    assert all(r["upload_inference"] is False for r in routes.values())
    assert "/home/" not in str(catalog)


@needs_research
def test_changed_result_file_is_rejected(tmp_path):
    config = settings(RESEARCH)
    spec = config["catalog"]["sources"]["fusion_summary"]
    copy = tmp_path / spec["path"]
    copy.parent.mkdir(parents=True)
    shutil.copy(RESEARCH / spec["path"], copy)
    copy.write_text(copy.read_text().replace("0.75", "0.5", 1))
    config["research_root"] = str(tmp_path)
    catalog = build_catalog(config)
    assert {e["code"] for e in catalog["errors"]} == {"artifact_mismatch", "model_unavailable"}
    assert all(v["id"] != "attribution" for v in catalog["views"])
