"""CPU integration: real queue/head/cache/prediction protocols, synthetic encoders.

No external models, dataset images or GPUs are needed. Registration guards have
their own tests; only that check and the heavyweight frozen encoders are replaced.
"""
import builtins
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from research.image_attribution import feature_ablation, models, vision_backbones
from scripts import predict_image_research, run_image_research_queue
from scripts.run_image_experiment import write_predictions


def _json(path):
    return json.loads(path.read_text())


def _jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_run_jobs_full_cpu_pipeline(tmp_path, monkeypatch):
    root = tmp_path
    data = root / "synthetic-images"
    data.mkdir()
    roles = ("train", "selection", "calibration", "evaluation")
    parts = {"detection": {}, "attribution": {}}
    colors = {}
    manifest = []
    for role_index, role in enumerate(roles):
        rows = []
        for index, source in enumerate((None, None, "s0", "s1")):
            sid = f"{role}-{index}"
            color = (20 + role_index*40 + index*3, 30 + index*35, 210 - role_index*20 - index*7)
            path = data / f"{sid}.png"
            Image.new("RGB", (12, 10), color).save(path)
            colors[sid] = np.array(color, dtype=np.float32) / 255
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            row = dict(sample_id=sid, relative_path=path.name, sha256=digest,
                       pixel_sha256="pixels-"+sid, group_id="group-"+sid,
                       split="train" if role == "train" else "dev", is_ai=source is not None,
                       source_label=source, semantic_class="fixture-scene", prompt_id=None)
            rows.append(row)
        parts["detection"][role] = rows
        parts["attribution"][role] = [row for row in rows if row["is_ai"]]
        manifest.extend(rows)
    holdout = data / "holdout-never-read.png"
    Image.new("RGB", (12, 10), (255, 0, 255)).save(holdout)
    manifest.append(dict(sample_id="holdout-sentinel", relative_path=holdout.name,
                         split="holdout", group_id="holdout-group", is_ai=True, source_label="s0"))
    (data/"manifest.jsonl").write_text("".join(json.dumps(row)+"\n" for row in manifest))
    manifest_hash = hashlib.sha256((data/"manifest.jsonl").read_bytes()).hexdigest()
    jobs = [dict(id="D-DINO", task="detection", variant="rgb"),
            dict(id="A-RGB", task="attribution", variant="rgb"),
            dict(id="A-FP", task="attribution", variant="fingerprint"),
            dict(id="A-Fusion", task="attribution", variant="fusion")]
    cfg = dict(seed=42, gpu_index=0, dataset=dict(path=data.name), classes=["s0", "s1"],
               dimensions=dict(rgb=4, fingerprint=3, fusion=7), jobs=jobs,
               training=dict(epochs=2, batch_size=2, learning_rate=.01,
                             scheduler_step=1, scheduler_gamma=.9),
               target_fprs=[.01, .05], bootstrap_repeats=10,
               extraction_batch_size=dict(rgb=3, fingerprint=2), dinov2_directory="unused-assets/dino",
               assets={name:dict(path=f"unused-assets/{name}", sha256="fixture-"+name)
                       for name in ("dinov2_model", "dinov2_config", "dinov2_processor", "fingerprint", "denoiser")},
               preprocessing=dict(rgb="synthetic-rgb", fingerprint="synthetic-fingerprint"),
               implementation_sha256={"fixture": "memory-source"},
               baseline_predictions=dict(detection="baseline-D0/predictions.jsonl",
                                         attribution="baseline-A0/predictions.jsonl"))
    # Real serializer; only parent-dev rows, deliberately reversed for ID alignment.
    for task, folder, classes in (("detection", "baseline-D0", ["real", "ai"]),
                                  ("attribution", "baseline-A0", ["s0", "s1"])):
        out = root/folder; out.mkdir()
        rows = [row for role in roles[1:] for row in parts[task][role]][::-1]
        labels = [int(row["is_ai"]) if task == "detection" else classes.index(row["source_label"]) for row in rows]
        probabilities = np.array([[.8,.2] if label == 0 else [.2,.8] for label in labels])
        write_predictions(out, rows, labels, classes, probabilities, task)

    def transform(image):
        return torch.from_numpy(np.array(image, dtype=np.float32).transpose(2,0,1).copy()/255)

    class RGBEncoder(torch.nn.Module):
        def forward(self, images):
            means = images.mean(dim=(2,3))
            return torch.cat((means, torch.ones((len(images),1), device=images.device)), dim=1)

    class FingerprintEncoder(torch.nn.Module):
        def forward(self, images):
            means = images.mean(dim=(2,3))
            return torch.stack((means[:,0]+.25*means[:,1], means[:,2]+.5, torch.ones_like(means[:,0])), dim=1)

    monkeypatch.setattr(vision_backbones, "load_dinov2", lambda directory, device: (RGBEncoder().eval(), transform, {}))
    monkeypatch.setattr(models, "load_ofa", lambda weights, denoiser, device: (FingerprintEncoder().eval(), None, None))
    monkeypatch.setattr(models, "ofa_features", lambda images, encoder, denoiser: encoder(images))
    monkeypatch.setattr(models, "ofa_transform", transform)
    monkeypatch.setattr(predict_image_research, "ROOT", root)
    monkeypatch.setattr(run_image_research_queue, "verify_registered_files", lambda *args: None)

    for obj, name in ((torch.backends.cudnn, "deterministic"), (torch.backends.cudnn, "benchmark"),
                      (torch.backends.cudnn, "allow_tf32"), (torch.backends.cuda.matmul, "allow_tf32")):
        monkeypatch.setattr(obj, name, getattr(obj, name))

    # A forbidden image open fails immediately, including hash reads before PIL.
    real_open, real_path_open = builtins.open, Path.open
    image_reads = []
    def audit(path, mode):
        if isinstance(path, (str, Path)) and "r" in mode:
            candidate = Path(path)
            assert candidate != holdout, "holdout image was opened"
            if candidate.suffix == ".png":
                image_reads.append(candidate.name)
    def guarded_open(path, mode="r", *args, **kwargs):
        audit(path, mode)
        return real_open(path, mode, *args, **kwargs)
    def guarded_path_open(path, mode="r", *args, **kwargs):
        audit(path, mode)
        return real_path_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(Path, "open", guarded_path_open)

    # Observe the real linear forwards, not a substitute trainer. This catches
    # accidental source-calibration inclusion even when output IDs look correct.
    active = {}; source_forward_inputs = {job["variant"]:[] for job in jobs[1:]}
    real_linear, real_fit = torch.nn.Linear, feature_ablation.fit_head
    class AuditedLinear(real_linear):
        def forward(self, value):
            if active.get("task") == "attribution":
                source_forward_inputs[active["variant"]].append(value.detach().cpu().numpy().copy())
            return super().forward(value)
    def traced_fit(*args, **kwargs):
        active.update(task=args[3]["task"], variant=args[6])
        try:
            return real_fit(*args, **kwargs)
        finally:
            active.clear()
    monkeypatch.setattr(torch.nn, "Linear", AuditedLinear)
    monkeypatch.setattr(feature_ablation, "fit_head", traced_fit)
    base = root/"research-output"; base.mkdir()
    plan = dict(status="ready", manifest_sha256=manifest_hash, selection_sha256="synthetic-selection", holdout_scored=False)
    threads = torch.get_num_threads()
    try:
        run_image_research_queue.run_jobs(root, cfg, "synthetic-config-hash", plan, parts, base, device="cpu")
    finally:
        torch.set_num_threads(threads)

    summary = _json(base/"summary.json")
    assert summary["status"] == "completed" and summary["holdout_scored"] is False
    assert set(summary["results"]) == {job["id"] for job in jobs}
    assert image_reads and holdout.name not in image_reads
    for job in jobs:
        out = base/job["id"]; task = job["task"]
        status = _json(out/"status.json")
        assert status["status"] == "completed" and status["exit_code"] == 0
        result = _json(out/"metrics.json")
        assert result["image_inference_verified"] is True
        assert result["reload_max_error"] <= 1e-6 and result["single_batch_max_error"] <= 1e-5
        check = _json(out/"image-inference-check.json")
        assert check["cached_vs_image_max_error"] <= 1e-5
        assert check["image_single_batch_max_error"] <= 1e-5
        assert check["role"] == "selection" and check["holdout_scored"] is False
        assert check["sample_ids"] == [row["sample_id"] for row in parts[task]["selection"][:2]]
        assert len(check["batch"]) == len(check["single"]) == 2
        saved = torch.load(out/"head.pth", map_location="cpu", weights_only=True)
        assert saved["task"] == task and saved["input_dimensions"] == cfg["dimensions"][job["variant"]]
        assert all(value.device.type == "cpu" for value in saved["state_dict"].values())
        history = _jsonl(out/"history.jsonl")
        assert len(history) == 2
        assert all(entry["train"]["n"] == len(parts[task]["train"]) for entry in history)
        predictions = _jsonl(out/"predictions.jsonl")
        assert [row["sample_id"] for row in predictions] == [row["sample_id"] for row in parts[task]["evaluation"]]
        assert all(row["split"] == "dev" and row["status"] == "ok" for row in predictions)
        assert all(np.isclose(sum(row["scores"]), 1) for row in predictions)
        if task == "attribution":
            assert result["calibration_used"] is False and "calibrated" not in result
            assert not (out/"calibration").exists()
            assert all(row["classes"] == ["s0", "s1"] and row["true_label"] in {"s0", "s1"} for row in predictions)
        else:
            assert {row["sample_id"] for row in _jsonl(out/"calibration/predictions.jsonl")} == {
                row["sample_id"] for row in parts[task]["calibration"]}

    def normalized(value):
        return value / np.linalg.norm(value)
    for variant, batches in source_forward_inputs.items():
        observed = np.concatenate(batches)
        assert len(observed) > 0
        for row in parts["attribution"]["calibration"]:
            r, g, b = colors[row["sample_id"]]
            rgb = normalized(np.array([r,g,b,1.]))
            fp = normalized(np.array([r+.25*g,b+.5,1.]))
            reserved = {"rgb":rgb, "fingerprint":fp, "fusion":normalized(np.r_[rgb,fp])}[variant]
            assert not np.isclose(observed, reserved, atol=1e-6).all(axis=1).any(), "source calibration reached a head"
    fp_cache = torch.load(base/"feature-cache/fingerprint.pt", map_location="cpu", weights_only=True)
    reserved_ids = {row["sample_id"] for row in parts["attribution"]["calibration"]}
    assert not reserved_ids.intersection(fp_cache["identity"]["sample_ids"])
    comparisons = _json(base/"paired-comparison.json")
    assert len(comparisons) == 4 and summary["comparisons"] == comparisons
    d0 = comparisons["D0-ResNet50_vs_D-DINO"]
    assert set(d0["first_calibrated"]["calibrated"]) == set(d0["second_calibrated"]) == {"0.01", "0.05"}
    for target in ("0.01", "0.05"):
        assert d0["first_calibrated"]["calibrated"][target]["calibration_negative_n"] == 2
        assert d0["second_calibrated"][target]["calibration_negative_n"] == 2
    for name, record in comparisons.items():
        if name != "D0-ResNet50_vs_D-DINO":
            assert record["paired_group_bootstrap"]["repeats"] == 1000
