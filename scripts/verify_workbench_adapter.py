#!/usr/bin/env python3
"""Adapter consistency on the registered selection images (never holdout).

Checks, with tolerance 1e-5 for D-OpenCLIP and A0: single vs batch, two independent full
reloads, the adapter's end-to-end service result, and the offline references (D-OpenCLIP
feature cache + head, A0 saved dev predictions). OFA research scores are compared with the
saved OFA predictions and reported against OFA's own registered 1e-4 tolerance.
Writes a redacted JSON (sample IDs, errors, hashes; no paths, no image data).
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from workbench.image_engine import ImageModels, ImageService, decode_image, sha256_file
from workbench.settings import load_config, workbench_settings

TOLERANCE, OFA_TOLERANCE = 1e-5, 1e-4
MANIFEST = "outputs/image_experiments/iab-balanced-v7-200-q179/manifest.jsonl"
FEATURE_CACHE = "outputs/image_experiments/openclip-q179-20261029/features.pt"
A0_PREDICTIONS = "outputs/image_experiments/a0-resnet50-iab-v7-200-q179-e30/predictions.jsonl"
OFA_PREDICTIONS = "outputs/image_experiments/ofa-iab-v7-200-q179-e30/predictions.jsonl"
ROLES = "outputs/image_experiments/openclip-q179-20261029/D-OpenCLIP/roles.json"


def saved_rows(path, ids):
    rows = {}
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            if row["sample_id"] in ids:
                rows[row["sample_id"]] = row
    return rows


def max_error(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="prepared preview config.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # OFA is loaded here even when the preview disables it, so its CPU error stays on record.
    settings = dict(workbench_settings(load_config(args.config)), research_ofa=True)
    root = Path(settings["research_root"])
    provenance = json.loads((root / settings["artifacts"]["dopenclip_provenance"]["path"]).read_text())
    registered = provenance["actual_image_verification"]["selection_sample_ids"]
    selection = json.loads((root / ROLES).read_text())["selection"]
    a0_saved = saved_rows(root / A0_PREDICTIONS, set(selection))
    ofa_saved = saved_rows(root / OFA_PREDICTIONS, set(selection))
    # Saved A0/OFA predictions only cover AI images; add the first two AI selection images.
    ai_ids = [i for i in selection if i in a0_saved and i in ofa_saved][:2]
    ids = registered + [i for i in ai_ids if i not in registered]
    if len(ai_ids) < 2:
        raise SystemExit("need two AI selection images with saved A0/OFA predictions")
    rows = {r["sample_id"]: r for r in map(json.loads, (root / MANIFEST).read_text().splitlines()) if r["sample_id"] in ids}
    data_root = (root / MANIFEST).parent.resolve()
    payloads = []
    for sample_id in ids:
        row = rows[sample_id]
        if row["split"] != "dev":
            raise SystemExit("refusing non-dev image")  # holdout is never opened
        path = (data_root / row["relative_path"]).resolve()
        if not path.is_relative_to(data_root) or sha256_file(path) != row["sha256"]:
            raise SystemExit("image path or SHA mismatch")
        payloads.append(path.read_bytes())
    images = [decode_image(p)[0] for p in payloads]

    started = time.monotonic()
    first = ImageModels(settings)
    first.load()
    load_seconds = time.monotonic() - started
    single = dict(det=[first.detect([i])[0] for i in images], a0=[first.attribute([i])[0] for i in images],
                  ofa=[first.ofa_scores([i])[0] for i in images])
    batch = dict(det=first.detect(images), a0=first.attribute(images), ofa=first.ofa_scores(images))
    head = first.head
    del first
    second = ImageModels(settings)
    second.load()
    reload = dict(det=second.detect(images), a0=second.attribute(images), ofa=second.ofa_scores(images))

    service = ImageService(settings, models=second)
    service_results = [service.analyze(p, "x.png", research=True, request_id=f"verify-{i:04d}") for i, p in enumerate(payloads)]
    service_det = [r["detection"]["score"] for r in service_results]
    service_a0 = []
    for r in service_results:
        top = r["attribution"]["top_k"] or next(x["top_k"] for x in r["research"] if x["id"] == "a0_closed_set_research")
        lookup = {x["label"]: x["score"] for x in top}
        service_a0.append([lookup[c] for c in second.a0_classes])

    cache_path = root / FEATURE_CACHE
    if sha256_file(cache_path) != provenance["feature_cache_sha256"]:
        raise SystemExit("feature cache hash differs from provenance")
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    index = {sample_id: i for i, sample_id in enumerate(cache["identity"]["ids"])}
    with torch.inference_mode():
        offline_det = head(cache["features"][[index[i] for i in ids]].float()).softmax(1)[:, 1].tolist()
    if any(a0_saved[i]["classes"] != second.a0_classes for i in ai_ids):
        raise SystemExit("A0 saved class order differs from checkpoint")
    if any(ofa_saved[i]["classes"] != second.ofa_classes for i in ai_ids):
        raise SystemExit("OFA saved class order differs from reference bank")
    pos = [ids.index(i) for i in ai_ids]

    checks = {
        "d_openclip_single_vs_batch": max_error(single["det"], batch["det"]),
        "d_openclip_reload": max_error(batch["det"], reload["det"]),
        "d_openclip_service_vs_direct": max_error(service_det, reload["det"]),
        "d_openclip_vs_offline_cache_head": max_error(reload["det"], offline_det),
        "a0_single_vs_batch": max_error(single["a0"], batch["a0"]),
        "a0_reload": max_error(batch["a0"], reload["a0"]),
        "a0_service_vs_direct": max_error(service_a0, reload["a0"]),
        "a0_vs_saved_predictions": max_error([reload["a0"][k] for k in pos], [a0_saved[i]["scores"] for i in ai_ids]),
    }
    ofa_checks = {
        "ofa_single_vs_batch": max_error(single["ofa"], batch["ofa"]),
        "ofa_reload": max_error(batch["ofa"], reload["ofa"]),
        "ofa_vs_saved_predictions": max_error([reload["ofa"][k] for k in pos], [ofa_saved[i]["match_scores"] for i in ai_ids]),
    }
    evidence = dict(
        schema="workbench-adapter-consistency.v1", created_unix=int(time.time()),
        selection_sample_ids=ids, registered_detection_ids=registered, ai_reference_ids=ai_ids,
        roles_sha256=sha256_file(root / ROLES), images_opened=len(ids), holdout_opened=False, device="cpu",
        torch=torch.__version__, python=platform.python_version(), model_load_seconds=round(load_seconds, 1),
        tolerance=TOLERANCE, checks=checks, passed=all(v <= TOLERANCE for v in checks.values()),
        ofa_tolerance=OFA_TOLERANCE, ofa_checks=ofa_checks, ofa_passed=all(v <= OFA_TOLERANCE for v in ofa_checks.values()),
        unknown=[r["unknown"] for r in service_results], detection_labels=[r["detection"]["label"] for r in service_results],
        model_bundle=service_results[0]["model_bundle"],
        offline_references=dict(feature_cache_sha256=provenance["feature_cache_sha256"],
                                a0_predictions_sha256=sha256_file(root / A0_PREDICTIONS),
                                ofa_predictions_sha256=sha256_file(root / OFA_PREDICTIONS)),
        note="Offline references were produced on GPU0 by the registered runs; adapter runs on CPU.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(dict(passed=evidence["passed"], ofa_passed=evidence["ofa_passed"], checks=checks, ofa_checks=ofa_checks), indent=2))
    sys.exit(0 if evidence["passed"] else 1)


if __name__ == "__main__":
    main()
