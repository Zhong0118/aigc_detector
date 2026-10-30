#!/usr/bin/env python3
"""Live acceptance against a running isolated preview API (dev selection images only).

Checks the HTTP contract end to end: live scores vs the registered offline references
(<= 1e-5), the 0.5 gate, research DNA, dedupe/conflict, typed errors, bounded concurrency,
history snapshots, the experiment catalog, the text flow and preview isolation. Output is
redacted: sample IDs, status codes, errors and timings only.
"""
import argparse
import concurrent.futures
import io
import json
import sqlite3
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import requests
import torch
from PIL import Image

from workbench.image_engine import sha256_file
from workbench.settings import load_config, workbench_settings

TOLERANCE = 1e-5
DATASET = "outputs/image_experiments/iab-balanced-v7-200-q179"
EVIDENCE = "docs/workbench_preview/evidence/adapter_acceptance.json"
TEXT = ("人工智能技术的快速发展正在深刻改变内容生产方式。研究者需要在检测准确率、误报控制与可解释性之间取得平衡，"
        "同时明确模型结论的适用范围与局限，避免把检测分数误读为生成概率。") * 3


def encoded(mode="RGB", fmt="PNG", size=(64, 48), **save):
    buffer = io.BytesIO()
    Image.new(mode, size, "gray" if mode in ("RGB", "L") else None).save(buffer, format=fmt, **save)
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--api", default="http://127.0.0.1:8611/api/v1")
    parser.add_argument("--main-db", required=True, help="online DB whose mtime must not change")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    settings = workbench_settings(config)
    root, registry, api = Path(settings["research_root"]), settings["registry"], args.api
    evidence = json.loads((ROOT / EVIDENCE).read_text())
    ids = [i["sample_id"] for i in evidence["images"]]
    rows = {r["sample_id"]: r for r in map(json.loads, (root / DATASET / "manifest.jsonl").read_text().splitlines())
            if r["sample_id"] in ids}
    assert all(rows[i]["split"] == "dev" for i in ids)  # holdout is never opened
    images = {}
    for i in ids:
        path = root / DATASET / rows[i]["relative_path"]
        assert sha256_file(path) == rows[i]["sha256"]
        images[i] = path.read_bytes()
    main_db = Path(args.main_db)
    main_mtime = main_db.stat().st_mtime
    preview_db = Path(config["database"]["path"])
    checks = {}

    def post(name, data, timeout=300, **form):
        return requests.post(f"{api}/analyze/file", files={"file": (name, data)}, data=form, timeout=timeout)

    # -- capabilities
    started = time.monotonic()
    caps = requests.get(f"{api}/capabilities", timeout=30).json()
    image = caps["image"]
    caps_text = json.dumps(caps, ensure_ascii=False)
    checks["capabilities"] = dict(
        ok=caps["preview"] and image["state"] == "ready" and image["device"] == "cpu" and image["unknown"]["score"] is None
        and {m["id"] for m in image["research_methods"] if m["run_allowed"]} == {
            "research-dna-lp22", "research-a0-assumed-known-ai", "research-openclip-calibrated-thresholds"}
        and not any(m["run_allowed"] for m in image["offline_only_methods"]) and "/home/" not in caps_text
        and image["registry_sha256"] == evidence["registry_sha256"],
        load_seconds=image["load_seconds"], bundle_id=image["bundle_id"], registry_sha256=image["registry_sha256"])
    before = requests.get(f"{api}/reports", params=dict(limit=1), timeout=30).json()["total"]

    # -- offline references (same as the device acceptance)
    from research.image_attribution.dna_route import DIMENSION as DNA_DIM
    from research.image_attribution.openclip_route import DIMENSION as CLIP_DIM
    from research.image_attribution.openclip_route import normalized
    from scripts.run_image_dna import load_head as dna_head
    from scripts.run_image_openclip import load_head as clip_head
    from scripts.run_image_openclip import probabilities
    dna_spec = next(m for m in registry["research_methods"] if m["id"] == "research-dna-lp22")

    def cache(run, provenance_path, sample_ids):
        prov = json.loads((root / provenance_path).read_text())
        data = torch.load(root / run / "features.pt", map_location="cpu", weights_only=True)
        index = {s: k for k, s in enumerate(data["identity"]["ids"])}
        return data["features"][[index[s] for s in sample_ids]].float(), prov
    det_feat, _ = cache("outputs/image_experiments/openclip-q179-20261029", registry["detection"]["evidence"]["provenance"]["path"], ids)
    det_ref = probabilities(clip_head(root / registry["detection"]["assets"]["head"]["path"], ["real", "ai"], "cpu"),
                            normalized(det_feat, CLIP_DIM), "cpu")[:, 1]
    ai_ids = [i for i in ids if rows[i]["is_ai"]]
    dna_feat, dna_prov = cache("outputs/image_experiments/dna-wild-q179-20261029", dna_spec["evidence"]["provenance"]["path"], ai_ids)
    dna_ref = probabilities(dna_head(root / dna_spec["assets"]["head"]["path"], dna_prov["classes"], "cpu"),
                            normalized(dna_feat, DNA_DIM), "cpu")
    a0_ref = {}
    with open(root / "outputs/image_experiments/a0-resnet50-iab-v7-200-q179-e30/predictions.jsonl") as handle:
        for line in handle:
            row = json.loads(line)
            if row["sample_id"] in ai_ids:
                a0_ref[row["sample_id"]] = row["scores"]
    classes = registry["attribution"]["classes"]

    def vector(top_k):
        lookup = {t["source"]: t["score"] for t in top_k}
        return [lookup[c] for c in classes]

    # -- standard live inference
    live, saved = {}, {}
    for k, sample_id in enumerate(ids):
        rid = str(uuid.uuid4())
        t = time.monotonic()
        response = post(f"selection-{k}.png", images[sample_id], request_id=rid)
        seconds = time.monotonic() - t
        body = response.json()
        r = body["image_result"]
        saved[sample_id] = (rid, body)
        a = r["attribution"]
        live[sample_id] = dict(
            status=response.status_code, is_ai=rows[sample_id]["is_ai"], label=r["detection"]["label"],
            detection_error=abs(r["detection"]["score"] - float(det_ref[k])), attribution=a["status"],
            attribution_error=(float(np.max(np.abs(np.array(vector(a["top_k"])) - np.array(a0_ref[sample_id]))))
                               if a["status"] == "ok" else None),
            gate_ok=(a["status"] == "ok") == (r["detection"]["score"] >= 0.5),
            unknown_null=r["unknown"]["score"] is None, persisted=r["persistence"]["status"] == "ok",
            legacy_keys=all(x in body for x in ("content_id", "filename", "modality", "metadata", "fingerprint",
                                                "detection", "provenance", "report")),
            client_seconds=round(seconds, 3), server_total_ms=r["runtime"]["timing_ms"]["total_ms"])
    checks["standard_live_vs_offline"] = dict(
        ok=all(v["status"] == 200 and v["detection_error"] <= TOLERANCE and v["gate_ok"] and v["unknown_null"]
               and v["persisted"] and v["legacy_keys"] and (v["attribution_error"] is None or v["attribution_error"] <= TOLERANCE)
               for v in live.values()) and any(v["attribution"] == "skipped" for v in live.values()),
        per_image=live)

    # -- research uploads
    research = {}
    for k, sample_id in enumerate(ai_ids):
        r = post("r.png", images[sample_id], mode="research", research_method_id="research-dna-lp22").json()["image_result"]
        research[sample_id] = dict(dna_error=float(np.max(np.abs(np.array(vector(r["research"][0]["outputs"]["top_k"]))
                                                                 - dna_ref[k]))),
                                   standard_unchanged=r["detection"] == saved[sample_id][1]["image_result"]["detection"])
    real_id = next(i for i in ids if not rows[i]["is_ai"])
    assumed = post("r.png", images[real_id], mode="research",
                   research_method_id="research-a0-assumed-known-ai").json()["image_result"]
    cal = post("r.png", images[real_id], mode="research",
               research_method_id="research-openclip-calibrated-thresholds").json()["image_result"]
    checks["research"] = dict(
        ok=all(v["dna_error"] <= TOLERANCE and v["standard_unchanged"] for v in research.values())
        and assumed["attribution"]["status"] == "skipped" and len(assumed["research"][0]["outputs"]["top_k"]) == 22
        and [t["threshold_id"] for t in cal["research"][0]["outputs"]["thresholds"]] == ["cal-fpr-0.01-v1", "cal-fpr-0.05-v1"],
        dna=research, assumed_status=assumed["attribution"]["status"])

    # -- dedupe and conflict
    rid, first = saved[ids[0]]
    again = post("again.png", images[ids[0]], request_id=rid).json()
    conflict = post("other.png", images[ids[1]], request_id=rid)
    checks["dedupe"] = dict(ok=again["deduplicated"] is True and again["image_result"] == first["image_result"]
                            and conflict.status_code == 409 and conflict.json()["error"]["code"] == "request_id_conflict")

    # -- typed errors (synthetic fixtures only)
    exif = Image.Exif()
    exif[0x0112] = 6
    cases = {
        "empty": (b"", "a.png", {}, 400, "empty_file"),
        "garbage_named_png": (b"not an image", "a.png", {}, 400, "decode_failed"),
        "rgba": (encoded("RGBA"), "a.png", {}, 400, "unsupported_image_policy"),
        "cmyk_jpeg": (encoded("CMYK", "JPEG"), "a.jpg", {}, 400, "unsupported_image_policy"),
        "exif_orientation_6": (encoded("RGB", "JPEG", exif=exif.tobytes()), "a.jpg", {}, 400, "unsupported_image_policy"),
        "gif": (encoded("RGB", "GIF"), "a.bin", {}, 415, "unsupported_media_type"),
        "over_10mib": (b"\xff\xd8\xff" + b"\0" * (10 * 1024 * 1024 + 1), "big.jpg", {}, 413, "too_large"),
        "over_pixels": (encoded("L", size=(4001, 4001)), "big.png", {}, 413, "too_large"),
        "bad_mode": (encoded(), "a.png", {"mode": "default"}, 422, "unsupported_mode"),
        "offline_only_method": (encoded(), "a.png", {"mode": "research", "research_method_id": "research-ofa-pair"},
                                422, "unknown_method_id"),
        "bad_request_id": (encoded(), "a.png", {"request_id": "x"}, 422, "invalid_request_id"),
        "text_research": (b"hello", "a.txt", {"mode": "research", "research_method_id": "research-dna-lp22"},
                          422, "unsupported_mode"),
    }
    errors = {}
    for name, (data, filename, form, status, code) in cases.items():
        response = post(filename, data, **form)
        body = response.json()
        errors[name] = dict(status=response.status_code, code=body.get("error", {}).get("code"),
                            ok=response.status_code == status and body.get("error", {}).get("code") == code
                            and "detection" not in body and "/home/" not in response.text)
    checks["typed_errors"] = dict(ok=all(v["ok"] for v in errors.values()), cases=errors)

    # -- bounded concurrency: distinct requests, never a fake success
    with concurrent.futures.ThreadPoolExecutor(6) as pool:
        responses = list(pool.map(lambda k: post(f"c{k}.png", images[ids[k % len(ids)]], request_id=str(uuid.uuid4())),
                                  range(6)))
    codes = sorted(r.status_code for r in responses)
    busy = [r.json()["error"] for r in responses if r.status_code != 200]
    checks["concurrency"] = dict(ok=set(codes) <= {200, 503} and all(e["code"] == "busy" and e["retryable"] for e in busy)
                                 and 200 in codes, status_codes=codes)

    # -- history snapshots
    after = requests.get(f"{api}/reports", params=dict(limit=100, modality="image"), timeout=30).json()
    content_id = saved[ids[0]][1]["content_id"]
    detail = requests.get(f"{api}/report/{content_id}", timeout=30).json()
    missing = requests.get(f"{api}/report/999999", timeout=30)
    checks["history"] = dict(ok=detail["image_result"] == saved[ids[0]][1]["image_result"] and after["total"] > 0
                             and missing.status_code == 404 and missing.json()["error"]["code"] == "report_not_found",
                             image_reports=after["total"], reports_before=before)

    # -- experiments catalog
    catalog = requests.get(f"{api}/experiments", timeout=60).json()
    checks["experiments"] = dict(ok=catalog["status"] == "ok" and len(catalog["groups"]) >= 9 and
                                 any("7/93" in f for f in catalog["key_facts"]), groups=len(catalog["groups"]))

    # -- text flow unchanged and stored with a snapshot
    t = time.monotonic()
    text = requests.post(f"{api}/analyze/text", json=dict(text=TEXT, filename="note.txt"), timeout=600)
    body = text.json()
    text_detail = requests.get(f"{api}/report/{body['content_id']}", timeout=30).json()
    checks["text_flow"] = dict(ok=text.status_code == 200 and body["modality"] == "text" and "report" in body
                               and "image_result" not in body and text_detail["report_snapshot"]["analysis"]["detection"]
                               == body["detection"], seconds=round(time.monotonic() - t, 2))

    # -- isolation
    tmp_dir = preview_db.parent / "tmp"
    leftovers = [p for p in tmp_dir.rglob("*") if p.is_file() and any(sha256_file(p) == rows[i]["sha256"] for i in ids)]
    with sqlite3.connect(preview_db) as conn:
        dump = "\n".join(conn.iterdump())
    checks["isolation"] = dict(
        ok=main_db.stat().st_mtime == main_mtime and preview_db.resolve() != main_db.resolve() and not leftovers
        and images[ids[0]][100:132].hex() not in dump and "/home/" not in json.dumps(catalog, ensure_ascii=False),
        main_db_mtime_unchanged=main_db.stat().st_mtime == main_mtime, upload_files_left=len(leftovers))

    result = dict(schema="workbench-live-acceptance-v1", created_unix=int(time.time()), api="loopback preview",
                  device=image["device"], holdout_opened=False, images=ids, tolerance=TOLERANCE,
                  passed=all(c["ok"] for c in checks.values()), checks=checks,
                  wall_seconds=round(time.monotonic() - started, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v["ok"] for k, v in checks.items()}, indent=2))
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
