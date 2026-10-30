#!/usr/bin/env python3
"""Real-model numerical acceptance for the workbench adapters (dev selection images only).

For each served component (det-openclip-q179-v1, attr-resnet50-q179-v1, research-dna-lp22)
the adapter is checked directly - not through the 0.5 gate - against:
  * the registered offline reference (feature cache + head, or saved dev predictions),
  * an independent reference built from the registered research functions on this device,
  * itself: single vs batch and a second full reload.
Tolerance 1e-5 (max absolute score error). The service end-to-end scores must equal the
direct adapter. Never opens holdout, never re-runs dataset evaluation, never refits.
Writes redacted evidence (sample IDs, errors, hashes, timings; no paths or image data).
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

from workbench.image_engine import ImageModels, ImageService, decode_image, digest_json, sha256_file
from workbench.settings import load_config, workbench_settings

TOLERANCE = 1e-5
DATASET = "outputs/image_experiments/iab-balanced-v7-200-q179"
MANIFEST_SHA256 = "9dd85636e2a19dbe833d9ebe2bee0c2a6b6eaf43015699e4fd9f27db3694b4c3"
INVENTORY = "docs/IMAGE_MODEL_INVENTORY_2026-10-30.json"
OPENCLIP_RUN = "outputs/image_experiments/openclip-q179-20261029"
DNA_RUN = "outputs/image_experiments/dna-wild-q179-20261029"
A0_PREDICTIONS = "outputs/image_experiments/a0-resnet50-iab-v7-200-q179-e30/predictions.jsonl"


def max_error(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def cached_features(root: Path, run: str, provenance: dict, ids: list[str]) -> torch.Tensor:
    path = root / run / "features.pt"
    if sha256_file(path) != provenance["feature_cache_sha256"]:
        raise SystemExit(f"{run} feature cache differs from its provenance")
    cache = torch.load(path, map_location="cpu", weights_only=True)
    index = {sample_id: i for i, sample_id in enumerate(cache["identity"]["ids"])}
    return cache["features"][[index[i] for i in ids]].float()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="prepared preview config.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    settings = workbench_settings(load_config(args.config))
    root, registry = Path(settings["research_root"]), settings["registry"]
    torch.set_num_threads(int(settings.get("torch_threads", 8)))

    # -- images: inventory probe (AI source-selection IDs) + registered detection selection IDs
    inventory = json.loads((ROOT / INVENTORY).read_text())["validation_probe"]
    if inventory["holdout_allowed"] or inventory["source_manifest_sha256"] != MANIFEST_SHA256:
        raise SystemExit("unexpected validation probe")
    det_prov = json.loads((root / registry["detection"]["evidence"]["provenance"]["path"]).read_text())
    det_roles = json.loads((root / OPENCLIP_RUN / "D-OpenCLIP/roles.json").read_text())
    dna_spec = next(m for m in registry["research_methods"] if m["id"] == "research-dna-lp22")
    dna_prov = json.loads((root / dna_spec["evidence"]["provenance"]["path"]).read_text())
    dna_roles = json.loads((root / DNA_RUN / "A-DNA-Wild-LP22/roles.json").read_text())
    source_ids = list(inventory["source_selection_ids"])
    det_ids = list(det_prov["actual_image_verification"]["selection_sample_ids"])
    ids = list(dict.fromkeys(source_ids + det_ids))
    if sha256_file(root / DATASET / "manifest.jsonl") != MANIFEST_SHA256:
        raise SystemExit("manifest anchor mismatch")
    rows = {r["sample_id"]: r for r in map(json.loads, (root / DATASET / "manifest.jsonl").read_text().splitlines())
            if r["sample_id"] in ids}
    if any(rows[i]["split"] != "dev" for i in ids):
        raise SystemExit("refusing non-dev image")  # holdout is never opened
    if not set(det_ids) <= set(det_roles["selection"]) or not set(source_ids) <= set(dna_roles["selection"]):
        raise SystemExit("probe images must come from the registered selection roles")
    if not set(source_ids) <= set(det_roles["selection"]):
        raise SystemExit("source probe images must also be detection selection images")
    a0_saved = {}
    with open(root / A0_PREDICTIONS) as handle:
        for line in handle:
            row = json.loads(line)
            if row["sample_id"] in source_ids:
                a0_saved[row["sample_id"]] = row
    data_root = (root / DATASET).resolve()
    payloads = []
    for sample_id in ids:
        path = (data_root / rows[sample_id]["relative_path"]).resolve()
        if not path.is_relative_to(data_root) or sha256_file(path) != rows[sample_id]["sha256"]:
            raise SystemExit("image path or SHA mismatch")
        payloads.append(path.read_bytes())
    # The research loaders use PIL convert('RGB'); the service decodes through image-upload-rgb-v1.
    decoded = [decode_image(p)[0] for p in payloads]
    src = [ids.index(i) for i in source_ids]

    # -- full load #1 (cold, includes hashing every asset)
    started = time.monotonic()
    first = ImageModels(settings)
    first.load()
    cold_load = time.monotonic() - started
    if first.research.get("research-dna-lp22") != "ready":
        raise SystemExit(f"DNA adapter failed to load: {first.research}")
    single = dict(det=[first.detect([i])[0] for i in decoded], a0=[first.attribute([decoded[k]])[0] for k in src],
                  dna=[first.dna_scores([decoded[k]])[0] for k in src])
    batch = dict(det=first.detect(decoded), a0=first.attribute([decoded[k] for k in src]),
                 dna=first.dna_scores([decoded[k] for k in src]))
    del first
    started = time.monotonic()
    second = ImageModels(settings)
    second.load()
    warm_load = time.monotonic() - started
    reload = dict(det=second.detect(decoded), a0=second.attribute([decoded[k] for k in src]),
                  dna=second.dna_scores([decoded[k] for k in src]))

    # -- registered offline references
    from research.image_attribution.dna_route import DIMENSION as DNA_DIM
    from research.image_attribution.dna_route import load_backbone as dna_backbone
    from research.image_attribution.models import load_checkpoint, resnet_transform
    from research.image_attribution.openclip_route import DIMENSION as CLIP_DIM
    from research.image_attribution.openclip_route import load_backbone as clip_backbone
    from research.image_attribution.openclip_route import normalized
    from scripts.run_image_dna import load_head as dna_head
    from scripts.run_image_openclip import load_head as clip_head
    from scripts.run_image_openclip import probabilities
    det_head = clip_head(root / registry["detection"]["assets"]["head"]["path"], ["real", "ai"], "cpu")
    dna_classes = dna_prov["classes"]
    head_dna = dna_head(root / dna_spec["assets"]["head"]["path"], dna_classes, "cpu")
    offline = dict(
        det=probabilities(det_head, normalized(cached_features(root, OPENCLIP_RUN, det_prov, ids), CLIP_DIM), "cpu")[:, 1],
        a0=[a0_saved[i]["scores"] for i in source_ids],
        dna=probabilities(head_dna, normalized(cached_features(root, DNA_RUN, dna_prov, source_ids), DNA_DIM), "cpu"))
    if any(a0_saved[i]["classes"] != second.a0_classes for i in source_ids) or dna_classes != second.a0_classes:
        raise SystemExit("saved class order differs from the served class map")

    # -- independent reference built from the registered research functions (PIL convert path)
    from PIL import Image
    import io
    research_images = [Image.open(io.BytesIO(p)).convert("RGB") for p in payloads]
    det_config = json.loads((root / registry["detection"]["evidence"]["config"]["path"]).read_text())
    dna_config = json.loads((root / dna_spec["evidence"]["config"]["path"]).read_text())
    with torch.inference_mode():
        visual, clip_tf = clip_backbone(root, det_config, "cpu")
        ref_det = probabilities(det_head, normalized(visual(torch.stack([clip_tf(i) for i in research_images])).float(),
                                                     CLIP_DIM), "cpu")[:, 1]
        del visual
        a0_model, _ = load_checkpoint(root / registry["attribution"]["assets"]["model"]["path"], "cpu")
        a0_tf = resnet_transform()
        ref_a0 = a0_model.eval()(torch.stack([a0_tf(research_images[k]) for k in src])).softmax(1).numpy()
        dna_model, dna_tf = dna_backbone(root, dna_config, "cpu")
        ref_dna = probabilities(head_dna, normalized(dna_model(torch.stack([dna_tf(research_images[k]) for k in src])),
                                                     DNA_DIM), "cpu")

    # -- service end to end (gate + research), using the second load
    def assets(spec):
        return {k: v["sha256"] for k, v in spec.get("assets", {}).items()}
    provisional = dict(device="cpu", components={c["model_id"] if "model_id" in c else c["id"]: dict(
        passed=True, assets_sha256=assets(c)) for c in (registry["detection"], registry["attribution"], dna_spec)})
    service = ImageService(settings, models=second, acceptance=provisional)
    persist = lambda r: r.__setitem__("persistence", dict(status="ok", content_id=0))  # noqa: E731 - no DB here
    standard = [service.analyze(p, persist=persist)[0] for p in payloads]
    assumed = [service.analyze(payloads[k], mode="research", method_id="research-a0-assumed-known-ai",
                               persist=persist)[0] for k in src]
    dna_service = [service.analyze(payloads[k], mode="research", method_id="research-dna-lp22",
                                   persist=persist)[0] for k in src]

    def vector(top_k):
        lookup = {t["source"]: t["score"] for t in top_k}
        return [lookup[c] for c in second.a0_classes]
    timing = dict(
        standard_total_ms=[r["runtime"]["timing_ms"]["total_ms"] for r in standard],
        standard_detection_ms=[r["runtime"]["timing_ms"]["detection_ms"] for r in standard],
        standard_attribution_ms=[r["runtime"]["timing_ms"]["attribution_ms"] for r in standard],
        research_dna_ms=[r["runtime"]["timing_ms"]["research_ms"] for r in dna_service])
    gated_service_a0 = [vector(r["attribution"]["top_k"]) for k, r in zip(src, [standard[k] for k in src])
                        if r["attribution"]["status"] == "ok"]
    components = {
        registry["detection"]["model_id"]: dict(checks=dict(
            single_vs_batch=max_error(single["det"], batch["det"]), full_reload=max_error(batch["det"], reload["det"]),
            vs_registered_feature_cache_and_head=max_error(reload["det"], offline["det"]),
            vs_research_reference_on_device=max_error(reload["det"], ref_det),
            service_vs_direct=max_error([r["detection"]["score"] for r in standard], reload["det"])),
            spec=registry["detection"], images=ids),
        registry["attribution"]["model_id"]: dict(checks=dict(
            single_vs_batch=max_error(single["a0"], batch["a0"]), full_reload=max_error(batch["a0"], reload["a0"]),
            vs_registered_saved_predictions=max_error(reload["a0"], offline["a0"]),
            vs_research_reference_on_device=max_error(reload["a0"], ref_a0),
            service_research_vs_direct=max_error([vector(r["research"][0]["outputs"]["top_k"]) for r in assumed], reload["a0"]),
            **({"service_gated_vs_direct": max_error(gated_service_a0, [reload["a0"][j] for j, k in enumerate(src)
                                                                           if standard[k]["attribution"]["status"] == "ok"])}
               if gated_service_a0 else {})),
            spec=registry["attribution"], images=source_ids),
        "research-dna-lp22": dict(checks=dict(
            single_vs_batch=max_error(single["dna"], batch["dna"]), full_reload=max_error(batch["dna"], reload["dna"]),
            vs_registered_feature_cache_and_head=max_error(reload["dna"], offline["dna"]),
            vs_research_reference_on_device=max_error(reload["dna"], ref_dna),
            service_vs_direct=max_error([vector(r["research"][0]["outputs"]["top_k"]) for r in dna_service], reload["dna"])),
            spec=dna_spec, images=source_ids),
    }
    report = {}
    for name, c in components.items():
        worst = max(c["checks"].values())
        report[name] = dict(passed=bool(np.isfinite(worst) and worst <= TOLERANCE), max_error=worst, checks=c["checks"],
                            assets_sha256=assets(c["spec"]), selection_sample_ids=c["images"])
    evidence = dict(
        schema="workbench-adapter-acceptance-v1", created_unix=int(time.time()), device="cpu", precision="fp32",
        tf32=False, torch_threads=torch.get_num_threads(), torch=torch.__version__, python=platform.python_version(),
        cpu=platform.processor() or platform.machine(), tolerance=TOLERANCE, registry_sha256=digest_json(registry),
        bundle_id=registry["bundle_id"], manifest_sha256=MANIFEST_SHA256, holdout_opened=False, images_opened=len(ids),
        images=[dict(sample_id=i, split=rows[i]["split"], is_ai=rows[i]["is_ai"]) for i in ids],
        components=report,
        detection_labels={i: r["detection"]["label"] for i, r in zip(ids, standard)},
        gate={i: r["attribution"]["status"] for i, r in zip(ids, standard)},
        unknown=[r["unknown"] for r in standard],
        load_seconds=dict(cold_first_load=round(cold_load, 1), second_full_reload=round(warm_load, 1)),
        warm_inference_ms=timing,
        offline_reference_note="Registered caches/predictions were produced by the registered GPU0 runs; "
                               "the adapter and the on-device research reference run on CPU.",
        a0_predictions_sha256=sha256_file(root / A0_PREDICTIONS))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: dict(passed=v["passed"], max_error=v["max_error"]) for k, v in report.items()}, indent=2))
    print(json.dumps(dict(load_seconds=evidence["load_seconds"], warm_inference_ms=timing), indent=2))
    sys.exit(0 if all(v["passed"] for v in report.values()) else 1)


if __name__ == "__main__":
    main()
