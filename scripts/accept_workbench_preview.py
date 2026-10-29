#!/usr/bin/env python3
"""Live acceptance against a running isolated preview API (no holdout, redacted output).

Uses only the registered dev selection images, compares the live D-OpenCLIP score with the
offline feature cache + head (<= 1e-5), and exercises dedupe, research mode, typed errors,
busy, history, the experiment catalog, the text flow and preview isolation.
"""
import argparse
import concurrent.futures
import hashlib
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import requests
import torch
from PIL import Image

from workbench.image_engine import sha256_file
from workbench.settings import load_config, workbench_settings

MANIFEST = "outputs/image_experiments/iab-balanced-v7-200-q179/manifest.jsonl"
FEATURE_CACHE = "outputs/image_experiments/openclip-q179-20261029/features.pt"
TEXT = ("人工智能技术的快速发展正在深刻改变内容生产方式。研究者需要在检测准确率、误报控制与可解释性之间取得平衡，"
        "同时明确模型结论的适用范围与局限，避免把检测分数误读为生成概率。") * 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--api", default="http://127.0.0.1:8611/api/v1")
    parser.add_argument("--main-db", required=True, help="online DB whose mtime must not change")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    settings = workbench_settings(config)
    root = Path(settings["research_root"])
    provenance = json.loads((root / settings["artifacts"]["dopenclip_provenance"]["path"]).read_text())
    ids = provenance["actual_image_verification"]["selection_sample_ids"]
    rows = {r["sample_id"]: r for r in map(json.loads, (root / MANIFEST).read_text().splitlines()) if r["sample_id"] in ids}
    assert all(rows[i]["split"] == "dev" for i in ids)
    data_root = (root / MANIFEST).parent
    images = {i: (data_root / rows[i]["relative_path"]).read_bytes() for i in ids}
    assert all(sha256_file(data_root / rows[i]["relative_path"]) == rows[i]["sha256"] for i in ids)
    main_db_mtime = Path(args.main_db).stat().st_mtime
    tmp_dir = Path(config["database"]["path"]).parent / "tmp"
    checks, api = {}, args.api

    def post(name, data, **form):
        return requests.post(f"{api}/analyze/file", files={"file": (name, data)}, data=form, timeout=300)

    caps = requests.get(f"{api}/capabilities", timeout=30).json()
    checks["capabilities"] = dict(ok=caps["preview"] and caps["image"]["state"] == "ready" and caps["image"]["device"] == "cpu"
                                  and caps["image"]["unknown"] == {"status": "unavailable", "score": None},
                                  ofa=caps["image"]["research"]["ofa_pair_verification"])
    before = requests.get(f"{api}/history", timeout=30).json()["total"]

    # Real images: live score vs offline cache + head.
    from scripts.run_image_openclip import load_head
    head = load_head(root / settings["artifacts"]["dopenclip_head"]["path"], ["real", "ai"], "cpu")
    cache = torch.load(root / FEATURE_CACHE, map_location="cpu", weights_only=True)
    index = {s: k for k, s in enumerate(cache["identity"]["ids"])}
    live = {}
    for k, sample_id in enumerate(ids):
        start = time.monotonic()
        response = post(f"sample-{k}.png", images[sample_id], request_id=f"accept-{k:04d}-{int(time.time())}")
        body = response.json()["image_result"]
        with torch.inference_mode():
            offline = float(head(cache["features"][[index[sample_id]]].float()).softmax(1)[0, 1])
        live[sample_id] = dict(status=response.status_code, error=abs(body["detection"]["score"] - offline),
                               label=body["detection"]["label"], attribution=body["attribution"]["status"],
                               top1_shown=bool(body["attribution"]["top_k"]), seconds=round(time.monotonic() - start, 2),
                               legacy_keys=all(x in response.json() for x in ("filename", "modality", "detection", "provenance", "report")))
    checks["live_vs_offline"] = dict(ok=all(v["status"] == 200 and v["error"] <= 1e-5 and v["legacy_keys"] for v in live.values()),
                                     per_image=live)

    rid = f"dedupe-{int(time.time())}"
    first = post("dup.png", images[ids[0]], request_id=rid).json()["image_result"]
    second = post("dup.png", images[ids[0]], request_id=rid).json()["image_result"]
    checks["dedupe"] = dict(ok=second.get("deduplicated") is True and second["analysis_id"] == first["analysis_id"])

    research = post("research.png", images[ids[0]], mode="research").json()["image_result"]
    items = {i["id"]: i for i in research["research"]}
    checks["research_mode"] = dict(
        ok=research["mode"] == "research" and len(items["d_openclip_calibrated_thresholds"]["rows"]) == 2
        and items["ofa_pair_verification"]["status"] == "unavailable" and research["unknown"]["score"] is None,
        ofa_reason=items["ofa_pair_verification"].get("reason"))

    def png(**kw):
        buffer = io.BytesIO()
        Image.new("RGB", kw.pop("size", (64, 64))).save(buffer, format=kw.pop("format", "PNG"), **kw)
        return buffer.getvalue()

    cases = {"empty": (b"", "invalid_input"), "garbage.png": (b"\x89PNG\r\n\x1a\n" + b"x" * 64, None),
             "truncated.png": (png()[:70], "decode_failed"), "anim.gif": (png(format="GIF"), "invalid_input"),
             "tiny.png": (png(size=(8, 8)), "invalid_input"), "wide.png": (png(size=(9000, 40)), "too_large")}
    errors = {}
    for name, (data, expected) in cases.items():
        r = post(name if name != "empty" else "empty.png", data)
        e = r.json().get("error") or {}
        errors[name] = dict(status=r.status_code, code=e.get("code"), has_request_id=bool(e.get("request_id")),
                            retryable=e.get("retryable"), image_result="image_result" in r.json())
        assert expected is None or e.get("code") == expected, (name, e)
    checks["typed_errors"] = dict(ok=all(v["status"] >= 400 and v["has_request_id"] and v["retryable"] is False
                                         and not v["image_result"] for v in errors.values()), cases=errors)

    with concurrent.futures.ThreadPoolExecutor(6) as pool:
        statuses = list(pool.map(lambda k: post(f"c{k}.png", images[ids[k % 2]]).status_code, range(6)))
    checks["busy_under_concurrency"] = dict(ok=set(statuses) <= {200, 429} and 429 in statuses and 200 in statuses,
                                            statuses=sorted(statuses))

    page = requests.get(f"{api}/history", params=dict(kind="image", page_size=5), timeout=30).json()
    item = requests.get(f"{api}/history/{page['items'][0]['analysis_id']}", timeout=30).json()
    missing = requests.get(f"{api}/history/not-a-real-id", timeout=30)
    snapshot_text = json.dumps(item)
    checks["history"] = dict(ok=page["total"] >= 1 and "model_bundle" in item["snapshot"] and missing.status_code == 404
                             and missing.json()["error"]["code"] == "report_not_found"
                             and "/home/" not in snapshot_text and "base64" not in snapshot_text,
                             snapshot_bytes=len(snapshot_text))

    catalog = requests.get(f"{api}/experiments", timeout=60).json()
    views = {v["id"]: v for v in catalog["views"]}
    main_row = views["detection"]["groups"][0]["rows"][0]
    checks["experiments"] = dict(ok=catalog["status"] == "ok" and main_row["fp"] == 19 and main_row["negative_n"] == 979
                                 and views["rejection"]["groups"][0]["rows"][0]["known_retention_count"] == 576,
                                 routes={r["id"]: r["run_status"] for r in catalog["routes"]})

    start = time.monotonic()
    text = requests.post(f"{api}/analyze/text", json=dict(text=TEXT, filename="sample.txt"), timeout=900)
    body = text.json() if text.ok else {}
    checks["text_flow"] = dict(ok=text.status_code == 200 and body.get("analysis_id") is not None and "detection" in body,
                               label=body.get("detection", {}).get("label"), report_provider=body.get("report", {}).get("provider"),
                               providers=[(p.get("name"), p.get("status")) for p in body.get("detection", {}).get("providers", [])],
                               seconds=round(time.monotonic() - start, 1))

    after = requests.get(f"{api}/history", timeout=30).json()["total"]
    # Libraries create cache dirs in the private TMPDIR; what matters is that no upload is persisted.
    uploaded = {hashlib.sha256(b).hexdigest() for b in images.values()}
    persisted = [p for base in (tmp_dir, Path(config["database"]["path"]).parent)
                 for p in base.rglob("*") if p.is_file() and p.stat().st_size < 50_000_000 and sha256_file(p) in uploaded]
    db_bytes = Path(config["database"]["path"]).read_bytes()
    unchanged = Path(args.main_db).stat().st_mtime == main_db_mtime
    raw_in_db = any(b[len(b) // 2:len(b) // 2 + 64] in db_bytes for b in images.values())
    hashes_in_db = all(h.encode() in db_bytes for h in uploaded)
    checks["isolation"] = dict(ok=unchanged and not persisted and hashes_in_db and not raw_in_db,
                               main_db_unchanged=unchanged, uploaded_images_persisted=len(persisted),
                               raw_image_bytes_in_preview_db=raw_in_db, hashes_in_preview_db=hashes_in_db,
                               history_added=after - before)
    evidence = dict(schema="workbench-live-acceptance.v1", created_unix=int(time.time()), selection_sample_ids=ids,
                    holdout_opened=False, passed=all(c["ok"] for c in checks.values()), checks=checks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v["ok"] for k, v in checks.items()}, indent=2), "\npassed:", evidence["passed"])
    sys.exit(0 if evidence["passed"] else 1)


if __name__ == "__main__":
    main()
