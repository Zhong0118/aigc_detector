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
    parser.add_argument("--expect-commit", help="inference_code_commit every new result must report (clean, no +dirty)")
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

    # -- capacity recovers after busy bursts and failed decodes (admission happens before decoding)
    broken = post("broken.png", b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    recovered = post("recover.png", images[ids[-1]], request_id=str(uuid.uuid4()))
    checks["recovery"] = dict(ok=broken.status_code == 400 and broken.json()["error"]["code"] == "decode_failed"
                              and recovered.status_code == 200, after_decode_failure=recovered.status_code)

    # -- every new result names the exact running code
    commits = {b["image_result"]["inference_code_commit"] for _, b in saved.values()}
    commits.add(recovered.json()["image_result"]["inference_code_commit"])
    checks["code_version"] = dict(ok=(args.expect_commit is None or commits == {args.expect_commit})
                                  and not any(c.endswith("+dirty") for c in commits), commits=sorted(commits))

    # -- history snapshots
    after = requests.get(f"{api}/reports", params=dict(limit=100, modality="image"), timeout=30).json()
    content_id = saved[ids[0]][1]["content_id"]
    detail = requests.get(f"{api}/report/{content_id}", timeout=30).json()
    missing = requests.get(f"{api}/report/999999", timeout=30)
    checks["history"] = dict(ok=detail["image_result"] == saved[ids[0]][1]["image_result"] and after["total"] > 0
                             and missing.status_code == 404 and missing.json()["error"]["code"] == "report_not_found",
                             image_reports=after["total"], reports_before=before)

    # -- history UTC time range: [created_from, created_to), combined with other filters, same totals for paging
    def listing(**params):
        response = requests.get(f"{api}/reports", params=params, timeout=30)
        return response.status_code, response.json()
    anchor = next(i for i in after["items"] if i["content_id"] == content_id)["created_at"]  # e.g. 2026-10-30T07:00:00Z
    stamp = anchor.replace("Z", "+00:00")
    from datetime import datetime, timedelta
    moment = datetime.fromisoformat(stamp)
    later = (moment + timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
    _, inclusive = listing(created_from=anchor, created_to=later, modality="image", limit=100)
    _, exclusive = listing(created_to=anchor, modality="image", limit=100)
    _, combined = listing(created_from=anchor, created_to=later, modality="image", status="ok", q="selection-0", limit=100)
    _, empty = listing(created_from="2099-01-01")
    _, day = listing(created_from=moment.date().isoformat(), limit=5, offset=0)
    pages = [listing(created_from=moment.date().isoformat(), limit=5, offset=o)[1] for o in range(0, day["total"], 5)]
    bad_status, bad = listing(created_from="2026-10-31", created_to="2026-10-30")
    junk_status, junk = listing(created_from="yesterday")
    _, legacy = listing(limit=5)
    # UTC overflow must be a typed 422 for either end (requests URL-encodes "+")
    overflow = [listing(**{field: value}) for field, value in (("created_from", "0001-01-01T00:00:00+14:00"),
                                                               ("created_to", "9999-12-31T23:59:59-14:00"))]
    _, offset_range = listing(created_from=(moment + timedelta(hours=8)).strftime("%Y-%m-%dT%H:%M:%S+08:00"),
                              created_to=(moment + timedelta(hours=8, seconds=1)).strftime("%Y-%m-%dT%H:%M:%S+08:00"),
                              modality="image", limit=100)
    checks["history_time_filter"] = dict(
        ok=content_id in [i["content_id"] for i in inclusive["items"]]
        and content_id not in [i["content_id"] for i in exclusive["items"]]
        and [i["content_id"] for i in combined["items"]] == [content_id] and empty["total"] == 0 and empty["items"] == []
        and sum(len(p["items"]) for p in pages) == day["total"] and all(p["total"] == day["total"] for p in pages)
        and bad_status == 422 and bad["error"]["code"] == "invalid_time_range"
        and junk_status == 422 and junk["error"]["code"] == "invalid_time_range" and legacy["total"] >= after["total"]
        and all(code == 422 and body["error"]["code"] == "invalid_time_range" and body["error"]["retryable"] is False
                for code, body in overflow)
        and [i["content_id"] for i in offset_range["items"]] == [i["content_id"] for i in inclusive["items"]],
        anchor=anchor, day_total=day["total"], pages=len(pages), overflow_status=[code for code, _ in overflow])

    # -- experiments catalog with training metadata from sealed evidence
    catalog = requests.get(f"{api}/experiments", timeout=60).json()
    methods = {m["id"]: m for m in catalog.get("methods", [])}
    epoch = lambda mid, key: (methods[mid][key]["value"], methods[mid][key]["status"])  # noqa: E731
    unknown = {r["method"]: r for g in catalog["groups"] if g["id"] == "unknown-1275" for r in g["rows"]}
    proto, msp = unknown["OpenCLIP open20 原型余弦"], unknown["OpenCLIP open20 MSP"]
    checks["experiments"] = dict(
        ok=catalog["status"] == "ok" and catalog["schema_version"] == "experiments-v2" and len(catalog["groups"]) >= 9
        and any("7/93" in f for f in catalog["key_facts"]) and catalog["key_facts_primary"] == 4
        and epoch("attr-resnet50-q179-v1", "training_epochs") == (30, "registered")
        and epoch("attr-resnet50-q179-v1", "selected_epoch") == (29, "registered")
        and epoch("research-rgb22", "training_epochs") == (None, "not_registered")
        and epoch("research-openclip-knn22", "training_epochs") == (None, "not_applicable")
        and all("training_epochs" in r for g in catalog["groups"] for r in g["rows"])
        and (proto["training_epochs"]["status"], proto["selected_epoch"]["status"]) == ("not_applicable",) * 2
        and (proto["support_head"]["training_epochs"]["value"], proto["support_head"]["selected_epoch"]["value"]) == (30, 29)
        and (msp["training_epochs"]["value"], msp["selected_epoch"]["value"]) == (30, 29) and "support_head" not in msp,
        groups=len(catalog["groups"]), methods=len(methods))

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

    result = dict(schema="workbench-live-acceptance-v2", created_unix=int(time.time()), api="loopback preview",
                  device=image["device"], holdout_opened=False, images=ids, tolerance=TOLERANCE,
                  passed=all(c["ok"] for c in checks.values()), checks=checks,
                  wall_seconds=round(time.monotonic() - started, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v["ok"] for k, v in checks.items()}, indent=2))
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
