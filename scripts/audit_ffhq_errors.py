"""CPU-only audit of saved q179 FFHQ predictions; never train or run inference."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXPECTED_MANIFEST = "9dd85636e2a19dbe833d9ebe2bee0c2a6b6eaf43015699e4fd9f27db3694b4c3"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_saved_run(base):
    base = Path(base)
    status = json.loads((base/"status.json").read_text())
    if status.get("status") != "completed" or status.get("exit_code") != 0:
        raise ValueError("saved run is not completed successfully")
    metrics = json.loads((base/"metrics.json").read_text())
    checkpoint = Path(metrics["checkpoint"])
    actual = digest(checkpoint)
    if actual != metrics["checkpoint_sha256"]:
        raise ValueError("saved checkpoint hash mismatch")
    return dict(status=status, checkpoint=str(checkpoint), checkpoint_sha256=actual)


def validate_predictions(predictions, manifest, roles):
    """Fail closed on holdout, role, identity, metadata, or score mismatches."""
    owners = {}
    for role, ids in roles.items():
        for sid in ids:
            if sid not in manifest:
                raise ValueError("role contains unknown sample ID")
            if manifest[sid]["split"] == "holdout":
                raise ValueError("holdout ID in roles")
            if sid in owners:
                raise ValueError("sample ID repeated across roles")
            owners[sid] = role
    selected = {"calibration": {}, "evaluation": {}}
    for pred in predictions:
        sid = pred["sample_id"]
        if sid not in manifest:
            raise ValueError("prediction contains unknown sample ID")
        row = manifest[sid]
        if row["split"] == "holdout":
            raise ValueError("holdout prediction rejected")
        role = pred["role"]
        if role not in selected or owners.get(sid) != role or row["split"] != "dev":
            raise ValueError("prediction role/ID mismatch")
        for field in ("group_id", "is_ai", "semantic_class", "source_label"):
            if pred[field] != row[field]:
                raise ValueError("prediction/manifest metadata mismatch")
        scores = pred["scores"]
        if (pred["classes"] != ["real", "ai"] or len(scores) != 2
                or not all(math.isfinite(x) and 0 <= x <= 1 for x in scores)
                or not math.isclose(sum(scores), 1, abs_tol=1e-5)):
            raise ValueError("invalid probability/class ordering")
        if pred["condition"] != "clean":
            continue
        if sid in selected[role]:
            raise ValueError("duplicate clean prediction")
        selected[role][sid] = pred
    for role, predictions_by_id in selected.items():
        if set(predictions_by_id) != set(roles[role]):
            raise ValueError("clean prediction IDs do not cover declared role")
    return selected


def inspect_image(row, root):
    """Inspect nonholdout real dev bytes/metadata; PIL may decode PNG to read EXIF."""
    if row["split"] == "holdout":
        raise ValueError("holdout image read rejected")
    if row["split"] != "dev" or row["is_ai"]:
        raise ValueError("header audit accepts real dev images only")
    root = Path(root).resolve()
    path = (root / row["relative_path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("image path escapes dataset")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != row["sha256"]:
        raise ValueError("image hash mismatch")
    with Image.open(io.BytesIO(data)) as image:
        width, height = image.size
        if (width, height, image.format) != (row["width"], row["height"], row["format"]):
            raise ValueError("manifest/header dimensions or format mismatch")
        short, long = sorted((width, height))
        resized_long = int(256 * long / short)
        quantization = getattr(image, "quantization", None)
        result = dict(sample_id=row["sample_id"], semantic_class=row["semantic_class"],
            width=width, height=height, resolution=f"{width}x{height}",
            format=image.format, mode=image.mode, file_bytes=len(data),
            bytes_per_pixel=len(data)/(width*height), aspect_ratio=long/short,
            resize_short_edge_factor=256/short,
            crop_retained_fraction=(224*224)/(256*resized_long),
            info_keys=",".join(sorted(image.info)), exif_tag_count=len(image.getexif()),
            icc_bytes=len(image.info.get("icc_profile", b"")),
            jpeg_quantization_sha256=(hashlib.sha256(json.dumps(quantization, sort_keys=True).encode()).hexdigest()
                                      if quantization else None),
            sha256_verified=True, png_chunks=None, png_bit_depth=None,
            png_color_type=None, png_zlib_headers=None)
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        offset, chunks, zlib_headers = 8, [], []
        while offset + 12 <= len(data):
            length = struct.unpack(">I", data[offset:offset+4])[0]
            kind = data[offset+4:offset+8].decode("ascii")
            payload = data[offset+8:offset+8+length]
            if offset+12+length > len(data):
                raise ValueError("truncated PNG chunk")
            if kind not in chunks:
                chunks.append(kind)
            if kind == "IHDR":
                result["png_bit_depth"], result["png_color_type"] = payload[8:10]
            if kind == "IDAT" and not zlib_headers:
                zlib_headers.append(payload[:2].hex())
            offset += length+12
            if kind == "IEND":
                break
        result["png_chunks"] = ",".join(chunks)
        result["png_zlib_headers"] = ",".join(zlib_headers)
    return result


def score_summary(predictions):
    from research.image_attribution.followups import wilson
    scores = np.array([p["scores"][1] for p in predictions], dtype=float)
    if not len(scores):
        return {"n": 0}
    fp = int((scores >= .5).sum())
    return dict(n=len(scores), predicted_ai=fp, predicted_ai_rate=fp/len(scores),
        rate_wilson95=wilson(fp, len(scores)), score_mean=float(scores.mean()),
        score_quantiles=dict(zip(("min", "p25", "median", "p75", "max"),
                                np.quantile(scores, [0, .25, .5, .75, 1]).tolist())))


def paired_summary(primary, reference):
    if set(primary) != set(reference):
        raise ValueError("paired sample IDs differ")
    if any(primary[s]["is_ai"] or reference[s]["is_ai"] for s in primary):
        raise ValueError("paired false positives require real images")
    a = {s for s, p in primary.items() if p["scores"][1] >= .5}
    b = {s for s, p in reference.items() if p["scores"][1] >= .5}
    delta = [primary[s]["scores"][1]-reference[s]["scores"][1] for s in sorted(primary)]
    return dict(n=len(primary), primary_fp=len(a), reference_fp=len(b),
        both_fp=len(a & b), primary_only_fp=len(a-b), reference_only_fp=len(b-a),
        neither_fp=len(primary)-len(a|b), mean_score_delta=float(np.mean(delta)))


def feature_summary(rows):
    categorical = ("format", "resolution", "mode", "info_keys", "exif_tag_count",
                   "icc_bytes", "png_chunks", "png_bit_depth", "png_color_type",
                   "png_zlib_headers", "jpeg_quantization_sha256")
    numeric = ("file_bytes", "bytes_per_pixel", "aspect_ratio",
               "resize_short_edge_factor", "crop_retained_fraction")
    return dict(n=len(rows),
        categorical={key:dict(Counter(str(row[key]) for row in rows)) for key in categorical},
        numeric={key:dict(zip(("min", "p25", "median", "p75", "max"),
            np.quantile([row[key] for row in rows], [0, .25, .5, .75, 1]).tolist()))
                 for key in numeric} if rows else {})


def run(output):
    from research.image_attribution.followups import partition_rows
    config_path = ROOT/"configs/image_experiments/iab_followups_v7_200_q179_e30.json"
    config = json.loads(config_path.read_text())
    dataset = ROOT/config["dataset"]
    manifest_path = dataset/"manifest.jsonl"
    if digest(manifest_path) != EXPECTED_MANIFEST:
        raise ValueError("frozen manifest hash mismatch")
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    manifest = {row["sample_id"]: row for row in rows}
    if len(manifest) != len(rows):
        raise ValueError("duplicate manifest IDs")
    sources = [manifest_path, config_path, Path(__file__), ROOT/"tests/test_ffhq_audit.py"]
    predictions, role_counts, verified_sources, completion = {}, {}, {}, {}
    for name in ("D2-semantic", "D1-control", "D3"):
        base = ROOT/config["output_root"]/name/"evaluate"
        completion[name] = verify_saved_run(base)
        role_path, pred_path = base/"roles.json", base/"predictions.jsonl"
        resolved_path = base/"resolved_config.json"
        roles = json.loads(role_path.read_text())
        expected = partition_rows(rows, dict(config["experiments"][name], seed=config["seed"]))
        if roles != {role:[r["sample_id"] for r in values] for role, values in expected.items()}:
            raise ValueError(f"{name}: saved roles do not match registered partition")
        resolved = json.loads(resolved_path.read_text())
        if resolved["manifest_sha256"] != EXPECTED_MANIFEST or resolved["roles_sha256"] != digest(role_path):
            raise ValueError("saved manifest/roles identity mismatch")
        if resolved["config_sha256"] != digest(config_path):
            raise ValueError("saved configuration hash mismatch")
        for rel, expected_sha in resolved["source_hashes"].items():
            if digest(ROOT/rel) != expected_sha:
                raise ValueError(f"saved/current source mismatch: {rel}")
            verified_sources[rel] = expected_sha
        predictions[name] = validate_predictions(
            [json.loads(line) for line in pred_path.read_text().splitlines()], manifest, roles)
        role_counts[name] = {role:dict(n=len(values),
            ffhq=sum(r["semantic_class"] == "FFHQ" for r in values),
            real_ffhq=sum(r["semantic_class"] == "FFHQ" and not r["is_ai"] for r in values))
            for role, values in expected.items()}
        sources.extend([role_path, pred_path, resolved_path, base/"metrics.json", base/"status.json"])
    primary = {sid:p for sid,p in predictions["D2-semantic"]["evaluation"].items() if not p["is_ai"]}
    if len(primary) != 93 or sum(p["scores"][1] >= .5 for p in primary.values()) != 56:
        raise ValueError("registered FFHQ 56/93 result not reproduced")
    controls = {}
    for name in ("D1-control", "D3"):
        controls[name] = {sid:predictions[name]["evaluation"][sid] for sid in primary}
    real_evaluation_ids = sorted(sid for sid,p in predictions["D1-control"]["evaluation"].items() if not p["is_ai"])
    features = {sid:inspect_image(manifest[sid], dataset) for sid in real_evaluation_ids}
    evaluation_by_semantic = {}
    for name, roles in predictions.items():
        evaluation_by_semantic[name] = {}
        for role, selected in roles.items():
            semantics = sorted({p["semantic_class"] for p in selected.values() if not p["is_ai"]})
            evaluation_by_semantic[name][role] = {semantic:score_summary(
                [p for p in selected.values() if not p["is_ai"] and p["semantic_class"] == semantic])
                for semantic in semantics}
    groups = defaultdict(list)
    for sid, values in features.items():
        groups[values["semantic_class"]].append(values)
    for sid in primary:
        groups["FFHQ_D2_FP" if primary[sid]["scores"][1] >= .5 else "FFHQ_D2_TN"].append(features[sid])
    conditional = {}
    for key in ("resolution", "format", "mode", "png_chunks", "png_bit_depth",
                "png_color_type", "png_zlib_headers", "info_keys", "exif_tag_count"):
        buckets = defaultdict(list)
        for sid, pred in primary.items():
            buckets[str(features[sid][key])].append(pred)
        conditional[key] = {value:score_summary(ps) for value,ps in buckets.items()}
    for key, cuts in (("bytes_per_pixel", (1., 1.5, 2.)), ("file_bytes", (65536, 98304, 131072))):
        buckets = defaultdict(list)
        for sid, pred in primary.items():
            value = features[sid][key]
            lo = max([float("-inf")]+[c for c in cuts if c <= value])
            hi = min([float("inf")]+[c for c in cuts if c > value])
            buckets[f"[{lo},{hi})"].append(pred)
        conditional[key] = {value:score_summary(ps) for value,ps in sorted(buckets.items())}
    nonholdout_metadata = Counter((r["split"],r["semantic_class"],r["is_ai"],
        r["format"],r["width"],r["height"]) for r in rows if r["split"] != "holdout")
    summary = dict(created_at=datetime.now(timezone.utc).isoformat(),
        source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        scope="Exploratory saved-score/header audit; associations, not causal diagnosis",
        manifest_sha256=EXPECTED_MANIFEST, manifest_n=len(rows),
        split_counts=dict(Counter(r["split"] for r in rows)),
        threshold=.5, threshold_fitted=False, training=False, inference=False,
        holdout_image_reads=0, inspected_real_dev_image_n=len(features),
        primary=score_summary(list(primary.values())),
        matched_comparisons={name:paired_summary(primary,p) for name,p in controls.items()},
        role_counts=role_counts, verified_completion=completion,
        control_d3_identical_score_vectors=sum(
            predictions["D1-control"]["evaluation"][sid]["scores"] ==
            predictions["D3"]["evaluation"][sid]["scores"] for sid in real_evaluation_ids),
        real_saved_scores_by_role_semantic=evaluation_by_semantic,
        image_feature_groups={name:feature_summary(values) for name,values in sorted(groups.items())},
        ffhq_conditional_fpr=conditional,
        preprocessing=dict(rgb_conversion=True, resize_short_edge=256, center_crop=224,
            normalization_mean=[.485,.456,.406], normalization_std=[.229,.224,.225],
            same_clean_train_eval_path=True, held_ffhq_excluded_from=["train","selection","calibration"],
            geometry_note="Retained fraction is geometric 224x224/(resized width*height), not face coverage",
            source_hashes_verified=verified_sources),
        provenance={str(p.relative_to(ROOT)):digest(p) for p in sources},
        limitations=[
            "All scores are existing development-role scores; holdout remains sealed.",
            "D2-semantic has no FFHQ training/selection/calibration; D1-control trains with FFHQ.",
            "D3 checkpoint selection previously saw parent dev; its favorable paired score is exploratory.",
            "PNG metadata/file size can proxy image content and cannot establish encoding causality.",
            "Headers cannot reveal upstream JPEG history, original resolution, face crop/alignment, or resampling.",
            "No image rendering, image-content annotation, re-encoding intervention, training, or inference performed.",
            "Conditional strata are descriptive; no evaluation threshold fitting or model selection.",
            "Wilson intervals treat images as independent; group correlation may reduce effective sample size.",
        ])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    with (output/"image_features.jsonl").open("w") as stream:
        for sid in real_evaluation_ids:
            stream.write(json.dumps(dict(features[sid],
                d1_control_score=predictions["D1-control"]["evaluation"][sid]["scores"][1],
                d3_score=predictions["D3"]["evaluation"][sid]["scores"][1],
                d2_semantic_score=primary[sid]["scores"][1] if sid in primary else None))+"\n")
    with (output/"nonholdout_metadata_counts.jsonl").open("w") as stream:
        for values,n in sorted(nonholdout_metadata.items()):
            stream.write(json.dumps(dict(zip(("split","semantic_class","is_ai","format","width","height"),values)) | {"n": n}, sort_keys=True)+"\n")
    report = [
        "# FFHQ saved-prediction audit",
        "",
        f"- D2-semantic FFHQ false positives: {summary['primary']['predicted_ai']}/93 at the unchanged 0.5 threshold.",
        f"- Same IDs, D1-control: {summary['matched_comparisons']['D1-control']['reference_fp']}/93; D3: {summary['matched_comparisons']['D3']['reference_fp']}/93.",
        f"- {len(features)} real evaluation-role image headers/byte hashes inspected; zero holdout image reads.",
        "- Full manifest read only as metadata to verify original splits and frozen roles.",
        "",
        "See summary.json for conditional rates, paired disagreements, role counts, source SHA256 provenance and limitations.",
        "image_features.jsonl contains per-image header measurements and aligned saved scores.",
        "nonholdout_metadata_counts.jsonl contains aggregate train/dev geometry and formats only.",
        "",
        "The observed association is not a causal explanation. D1-control and D2 differ in FFHQ training exposure,",
        "and D3 inherits parent-dev checkpoint-selection bias. PNG size also reflects content.",
        "Header equality does not exclude different face framing, alignment, texture or upstream encoding.",
        "",
        "Next hypotheses: distribution-specific face alignment/texture or resampling histories;",
        "content-controlled re-encoding and crop interventions on development samples would test these.",
        "Such interventions require separate inference and are not part of this audit.",
        "",
        "Reproduce into a NEW output directory:",
        "",
        "```sh",
        "OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-image/bin/python scripts/audit_ffhq_errors.py --output NEW_DIRECTORY",
        "```",
    ]
    (output/"README.md").write_text("\n".join(report)+"\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({key:result[key] for key in ("primary","matched_comparisons","inspected_real_dev_image_n","holdout_image_reads")},indent=2))
