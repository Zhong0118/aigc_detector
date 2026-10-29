#!/usr/bin/env python3
"""Registered OpenCLIP project adaptation; explicit execution, offline assets only."""
import argparse
import fcntl
import importlib.metadata
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.metrics import f1_score

from research.image_attribution.feature_ablation import fit_head
from research.image_attribution.openclip_route import (
    DIMENSION, PREPROCESSING, cache_identity, digest, encode_rows, fit_rejection_threshold,
    job_parts, knn_probabilities, knn_neighbor_labels, load_backbone, normalized, prototypes, read_cache,
    rejection_report, write_cache,
)
from research.image_attribution.protocol import sha256, validate_manifest, write_json
from scripts.run_image_followup import class_report, evaluate_scores

DEFAULT_CONFIG = "configs/image_experiments/iab_openclip_q179_20261029.json"
ROLES = ("train", "selection", "calibration", "evaluation")


def prepare(config):
    data = ROOT / config["dataset"]
    receipt = config["environment_receipt"]
    if sha256(ROOT / receipt["path"]) != receipt["sha256"]:
        raise ValueError("environment receipt changed")
    for filename, key in [("manifest.jsonl", "manifest_sha256"), ("selection.json", "selection_sha256")]:
        if sha256(data / filename) != config[key]:
            raise ValueError(f"dataset identity mismatch: {filename}")
    rows = [json.loads(line) for line in (data / "manifest.jsonl").read_text().splitlines() if line.strip()]
    validate_manifest(rows, data, verify_files=False)
    for relative, expected in config["implementation_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"implementation changed: {relative}")
    cache_rows = sorted([r for r in rows if r["split"] != "holdout"], key=lambda r: r["sample_id"])
    identity = cache_identity(cache_rows, config["asset"], config["implementation_sha256"])
    jobs = {}
    for name, experiment in config["experiments"].items():
        parts, classes = job_parts(rows, experiment, config["seed"])
        if len(classes) != experiment["class_count"]:
            raise ValueError(f"class count mismatch: {name}")
        jobs[name] = dict(classes=classes, role_counts={r: len(v) for r, v in parts.items()},
                          role_ids_sha256={r: digest([x["sample_id"] for x in v]) for r, v in parts.items()})
    return data, rows, cache_rows, identity, jobs


def require_disk_space(path):
    if shutil.disk_usage(path).free < 20 * 1024**3:
        raise RuntimeError("at least 20 GiB free disk space is required before OpenCLIP execution")


def gpu_idle(device):
    if not str(device).startswith("cuda"):
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or device not in ("cuda", "cuda:0"):
        raise RuntimeError("set CUDA_VISIBLE_DEVICES=0 and use cuda or cuda:0; GPU1 is reserved")
    used = int(subprocess.check_output([
        "nvidia-smi", "--id=0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True).strip())
    if used > 1024:
        raise RuntimeError(f"physical GPU0 memory is busy: {used} MiB > 1024 MiB")


def role_features(features, index, parts):
    return {role: features[[index[row["sample_id"]] for row in selected]]
            for role, selected in parts.items()}


def probabilities(head, features, device):
    result = []
    with torch.inference_mode():
        for batch in features.split(128):
            result.append(head(batch.to(device)).softmax(1).cpu().numpy())
    return np.concatenate(result)


def load_head(path, classes, device):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["classes"] != classes or saved["input_dimensions"] != DIMENSION:
        raise ValueError("head vocabulary or input dimensions changed")
    model = torch.nn.Linear(DIMENSION, len(classes)).to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    return model.eval()


def score_jsonl(path, rows, classes, scores, extra=None):
    with path.open("x") as handle:
        for i, (row, p) in enumerate(zip(rows, scores)):
            value = dict(sample_id=row["sample_id"], group_id=row["group_id"], source_label=row["source_label"],
                         is_ai=bool(row["is_ai"]), semantic_class=row["semantic_class"],
                         classes=classes, scores=[float(x) for x in p], predicted_class=classes[int(p.argmax())], role=path.stem, condition="clean")
            if extra:
                value.update({name: float(values[i]) for name, values in extra.items()})
            handle.write(json.dumps(value, allow_nan=False) + "\n")


def verify_images(config, data, rows, cached, score_fn, device):
    if len(rows) < 2:
        raise ValueError("two actual selection images required")
    model, transform = load_backbone(ROOT, config, device)
    batch = encode_rows(model, transform, data, rows[:2], device, batch_size=2)
    single = encode_rows(model, transform, data, rows[:2], device, batch_size=1)
    checks = dict(image_single_batch_feature_max_error=float((batch - single).abs().max()),
                  image_cache_feature_max_error=float((batch - cached[:2]).abs().max()),
                  image_single_batch_score_max_error=float(np.max(np.abs(score_fn(batch) - score_fn(single)))),
                  image_cache_score_max_error=float(np.max(np.abs(score_fn(batch) - score_fn(cached[:2])))),
                  selection_sample_ids=[r["sample_id"] for r in rows[:2]],
                  full_offline_backbone_reload=True, images_opened=2)
    if any(value > 1e-5 for name, value in checks.items() if name.endswith("max_error")):
        raise ValueError(f"actual-image reproducibility failed: {checks}")
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return checks


def run_job(config, name, rows, features, cache_rows, data, out, device, cache_sha256=None):
    experiment = config["experiments"][name]
    parts, classes = job_parts(rows, experiment, config["seed"])
    index = {r["sample_id"]: i for i, r in enumerate(cache_rows)}
    x = role_features(features, index, parts)
    provenance = dict(model="ViT-L-14-quickgelu", asset=config["asset"], dependencies=config["dependencies"],
        manifest_sha256=config["manifest_sha256"], selection_sha256=config["selection_sha256"],
        code=config["implementation_sha256"], preprocessing=PREPROCESSING, classes=classes,
        feature_cache_sha256=cache_sha256,
        feature_cache_identity_sha256=digest(cache_identity(cache_rows, config["asset"], config["implementation_sha256"])),
        experiment=experiment, feature_dimensions=DIMENSION, backbone_frozen=True, holdout_scored=False,
        role_counts={r: len(v) for r, v in parts.items()},
        role_ids_sha256={r: digest([z["sample_id"] for z in v]) for r, v in parts.items()},
        limitations=config["limitations"])
    write_json(out / "provenance.json", provenance)
    write_json(out / "roles.json", {r: [z["sample_id"] for z in v] for r, v in parts.items()})
    write_json(out / "status.json", dict(status="running", stage="fit", experiment=name))
    scores = {}
    if experiment.get("estimator") == "knn":
        labels = torch.tensor([classes.index(r["source_label"]) for r in parts["train"]])
        history = []
        best_score, best_k = -1., None
        truth = [classes.index(r["source_label"]) for r in parts["selection"]]
        neighbors = knn_neighbor_labels(x["selection"], x["train"], labels, len(classes), 20)
        for k in range(1, 21):
            p = torch.nn.functional.one_hot(neighbors[:, :k], num_classes=len(classes)).float().mean(1).numpy()
            value = float(f1_score(truth, p.argmax(1), labels=list(range(len(classes))),
                                   average="macro", zero_division=0))
            history.append(dict(k=k, selection_macro_f1=value))
            if value > best_score:  # Ascending k chooses the smallest k when tied.
                best_score, best_k = value, k
        write_json(out / "knn_selection.json", history)
        torch.save(dict(schema="openclip-knn-v1", reference=x["train"], labels=labels, classes=classes,
                        k=best_k, reference_sample_ids=[r["sample_id"] for r in parts["train"]]),
                   out / "knn.pth")
        saved = torch.load(out / "knn.pth", map_location="cpu", weights_only=True)
        if saved["classes"] != classes or saved["reference_sample_ids"] != [r["sample_id"] for r in parts["train"]]:
            raise ValueError("KNN reload identity mismatch")
        score_fn = lambda q: knn_probabilities(q, saved["reference"], saved["labels"], len(classes), saved["k"])
        scores = {role: score_fn(x[role]) for role in ("calibration", "evaluation")}
        result = class_report(parts["evaluation"], classes, scores["evaluation"])
        result.update(best_k=best_k, selection_macro_f1=best_score, epoch=None, training_epochs=None)
        asset_path = out / "knn.pth"
    else:
        linear = out / "linear"
        linear.mkdir()
        training_parts = parts
        training_x = x
        if experiment["mode"] == "open_set":
            # fit_head's closed-set evaluator only receives known rows. Unknown scores
            # are produced after the selected checkpoint is fixed.
            training_parts = {role: [r for r in values if r["source_label"] in classes]
                              for role, values in parts.items()}
            training_x = role_features(features, index, training_parts)
        roles = ROLES if experiment["task"] == "detection" else ("train", "selection", "evaluation")
        head_config = dict(config, task=experiment["task"], dimensions={"openclip": DIMENSION})
        result = fit_head(torch.cat([training_x[r] for r in roles]), training_parts, classes,
                          head_config, linear, device, "openclip", provenance)
        # A job's root status stays running while the nested linear fitting stage ends.
        asset_path = linear / "head.pth"
        head = load_head(asset_path, classes, device)
        score_fn = lambda q: probabilities(head, q, device)
        scores = {role: score_fn(x[role]) for role in ("calibration", "evaluation")}
        records = [json.loads(line) for line in (linear / "history.jsonl").read_text().splitlines()]
        if [record["epoch"] for record in records] != list(range(1, 31)):
            raise ValueError("expected exactly epochs 1 through 30")
        result["training_epochs"] = 30
        result["epoch"] = result["best_epoch"]
    extra = {}
    if experiment["mode"] == "open_set":
        proto = prototypes(x["train"], parts["train"], classes)
        torch.save(dict(classes=classes, prototypes=proto,
                        train_ids=[r["sample_id"] for r in parts["train"]]), out / "prototypes.pth")
        saved_proto = torch.load(out / "prototypes.pth", map_location="cpu", weights_only=True)
        if saved_proto["classes"] != classes or not torch.equal(saved_proto["prototypes"], proto):
            raise ValueError("prototype reload mismatch")
        confidence = {role: dict(msp=scores[role].max(1),
                       prototype_cosine=(normalized(x[role]) @ proto.T).max(1).values.numpy())
                      for role in ("calibration", "evaluation")}
        result = dict(closed_known_head=result, standard_msp=evaluate_scores(
            dict(experiment, seed=config["seed"]), parts, classes, scores), rejection={})
        for method in ("msp", "prototype_cosine"):
            threshold = fit_rejection_threshold(parts["calibration"], classes,
                confidence["calibration"][method], experiment["unknown_false_acceptance_target"])
            result["rejection"][method] = dict(
                threshold_fitted_role="calibration", threshold_unknown_sources=experiment["unknown_calibration"],
                rejection_score="1 - " + method, acceptance_threshold=threshold,
                distance_threshold=1 - threshold,
                calibration=rejection_report(parts["calibration"], classes, confidence["calibration"][method],
                                             scores["calibration"].argmax(1), threshold),
                evaluation=rejection_report(parts["evaluation"], classes, confidence["evaluation"][method],
                                            scores["evaluation"].argmax(1), threshold))
        extra = confidence
        result["prototype_sha256"] = sha256(out / "prototypes.pth")
        result["training_epochs"] = 30
        result["epoch"] = result["closed_known_head"]["best_epoch"]
    for role in ("calibration", "evaluation"):
        score_jsonl(out / (role + ".jsonl"), parts[role], classes, scores[role], extra.get(role))
    with (out / "predictions.jsonl").open("x") as combined:
        for role in ("calibration", "evaluation"):
            combined.write((out / (role + ".jsonl")).read_text())
    write_json(out / "status.json", dict(status="running", stage="actual_image_reload", experiment=name))
    checks = verify_images(config, data, parts["selection"], x["selection"], score_fn, device)
    result.update(actual_image_verification=checks, head_asset_sha256=sha256(asset_path),
                  classes=classes, holdout_scored=False, limitations=config["limitations"])
    write_json(out / "metrics.json", result)
    provenance.update(head_asset_sha256=sha256(asset_path), actual_image_verification=checks)
    write_json(out / "provenance.json", provenance)
    write_json(out / "status.json", dict(status="completed", experiment=name, exit_code=0,
                                        metrics=str((out / "metrics.json").relative_to(ROOT))))
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--action", choices=["all", "cache", "job"], default="all")
    parser.add_argument("--experiment")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    data, rows, cache_rows, identity, jobs = prepare(config)
    if not args.execute:
        print(json.dumps(dict(status="prepared", images_opened=False, jobs=jobs,
                              cache_rows=len(cache_rows), output_root=config["output_root"]), indent=2))
        return
    if config.get("execution_enabled") is not True:
        raise PermissionError("configuration execution disabled")
    if args.action == "job" and args.experiment not in jobs:
        raise ValueError("--action job requires a registered --experiment")
    out = ROOT / config["output_root"]
    with (ROOT / "outputs/image_experiments/gpu0-research.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu_idle(args.device)
        require_disk_space(ROOT)
        out.mkdir(parents=True, exist_ok=True)
        state = out / "status.json"
        if state.exists() and json.loads(state.read_text()).get("status") == "failed":
            raise FileExistsError("failed run requires explicit investigation; automatic retry is forbidden")
        asset = config["asset"]
        asset_file = ROOT / asset["path"]
        if asset_file.stat().st_size != asset["size"] or sha256(asset_file) != asset["sha256"]:
            raise ValueError("official asset changed or incomplete")
        for package, version in config["dependencies"].items():
            if importlib.metadata.version(package) != version:
                raise ValueError(f"runtime dependency changed: {package}")
        registration = out / "registration.json"
        binding = dict(config_sha256=digest(config), config=config)
        if registration.exists():
            if json.loads(registration.read_text()) != binding:
                raise ValueError("output root belongs to a different experiment registration")
        else:
            write_json(registration, binding)
        launch = dict(pid=os.getpid(), boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                      process_start_ticks=Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19],
                      command_line=Path("/proc/self/cmdline").read_bytes().decode().split(chr(0))[:-1],
                      started_unix=time.time(), config_sha256=digest(config))
        write_json(out / "launch.json", launch)
        def write_state(state):
            write_json(out / "status.json", dict(launch=launch, **state))
        torch.set_num_threads(4)
        if str(args.device).startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        write_state(dict(status="running", stage="cache", pid=os.getpid()))
        cache = out / "features.pt"
        try:
            if cache.exists():
                receipt = out / "cache.json"
                if not receipt.exists() or json.loads(receipt.read_text())["sha256"] != sha256(cache):
                    raise ValueError("cache receipt missing or cache file modified")
                features = read_cache(cache, identity)
            else:
                model, transform = load_backbone(ROOT, config, args.device)
                features = encode_rows(model, transform, data, cache_rows, args.device,
                                       batch_size=config["training"]["batch_size"])
                write_cache(cache, features, identity)
                del model
                if str(args.device).startswith("cuda"):
                    torch.cuda.empty_cache()
            cache_sha = sha256(cache)
            write_json(out / "cache.json", dict(identity=identity, sha256=cache_sha,
                       n=len(features), dimensions=DIMENSION, holdout_read=False))
            if args.action == "cache":
                write_state(dict(status="cache_ready"))
                return
            selected = [args.experiment] if args.action == "job" else list(jobs)
            for name in selected:
                job_out = out / name
                if job_out.exists():
                    status = job_out / "status.json"
                    if status.exists() and json.loads(status.read_text()).get("status") == "completed" and json.loads(status.read_text()).get("exit_code") == 0:
                        if not (job_out / "metrics.json").exists():
                            raise ValueError(f"completed job lacks metrics: {name}")
                        result = json.loads((job_out / "metrics.json").read_text())
                        checkpoint = job_out / ("knn.pth" if config["experiments"][name].get("estimator") == "knn" else "linear/head.pth")
                        if sha256(checkpoint) != result["head_asset_sha256"]:
                            raise ValueError(f"completed job checkpoint changed: {name}")
                        continue
                    raise FileExistsError(f"existing incomplete job is never automatically retrained: {name}")
                job_out.mkdir()
                write_state(dict(status="running", experiment=name, pid=os.getpid()))
                try:
                    run_job(config, name, rows, features, cache_rows, data, job_out, args.device, cache_sha256=cache_sha)
                except Exception as error:
                    write_json(job_out / "status.json", dict(status="failed", error=repr(error)))
                    raise
            write_state(dict(status="completed" if args.action == "all" else "job_completed",
                                                experiments=selected, exit_code=0))
        except Exception as error:
            write_state(dict(status="failed", error=repr(error)))
            raise


if __name__ == "__main__":
    main()
