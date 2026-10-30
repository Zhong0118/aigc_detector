"""Image service contract with fake models: input policy, gating, failures, research, dedupe.

Fakes only exercise orchestration; real-model numbers are verified separately by
scripts/verify_workbench_adapter.py on the research server.
"""
import io
import json
import re
import threading
from pathlib import Path

import pytest
import yaml
from PIL import Image

from workbench.image_engine import (CONDITIONAL_STATEMENT, ImageModels, ImageService, WorkbenchError, decode_image,
                                    digest_json, verify_asset)

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = yaml.safe_load(re.sub(r"\$\{\w+\}", "/nonexistent",
                                (ROOT / "configs/workbench/preview.overlay.example.yaml").read_text()))["workbench"]
CLASSES = OVERLAY["registry"]["attribution"]["classes"]


def settings(**extra):
    return dict(OVERLAY, research_root="/nonexistent", code_commit="test", **extra)


def image_bytes(size=(64, 48), mode="RGB", fmt="PNG", **save):
    buffer = io.BytesIO()
    Image.new(mode, size).save(buffer, format=fmt, **save)
    return buffer.getvalue()


png = image_bytes


def acceptance(*ids):
    reg = OVERLAY["registry"]
    specs = {reg["detection"]["model_id"]: reg["detection"], reg["attribution"]["model_id"]: reg["attribution"],
             **{m["id"]: m for m in reg["research_methods"]}}
    return dict(device="cpu", components={i: dict(passed=True, assets_sha256={k: v["sha256"] for k, v in
                                                                             specs[i].get("assets", {}).items()})
                                          for i in ids})


class FakeModels:
    def __init__(self, score=0.9, attribution_fails=False, detect_fails=False, dna_fails=False):
        self.score, self.attribution_fails, self.detect_fails, self.dna_fails = score, attribution_fails, detect_fails, dna_fails
        self.fixed_metrics = dict(n=2310, negative_count=979, positive_count=1331, fpr=0.0194, tpr=0.9895)
        self.calibrated = {"0.01": dict(calibration_negative_n=1014, calibration_fpr=0.0099,
                                        evaluation=dict(threshold=0.74, n=2310, negative_count=979, fpr=0.0102, tpr=0.97)),
                           "0.05": dict(calibration_negative_n=1014, calibration_fpr=0.049,
                                        evaluation=dict(threshold=0.27, n=2310, negative_count=979, fpr=0.041, tpr=0.995))}
        self.a0_classes = CLASSES
        self.research = {m["id"]: "ready" for m in OVERLAY["registry"]["research_methods"]}
        self.calls = dict(detect=0, attribute=0, dna=0)

    def load(self):
        pass

    def detect(self, images):
        self.calls["detect"] += 1
        if self.detect_fails:
            raise RuntimeError("boom")
        if self.score != self.score:
            raise ValueError("model returned a non-finite or out-of-range score")
        return [self.score] * len(images)

    def attribute(self, images):
        self.calls["attribute"] += 1
        if self.attribution_fails:
            raise RuntimeError("boom")
        row = [0.01] * 22
        row[5], row[2], row[9] = 0.6, 0.2, 0.0
        return [row] * len(images)

    def dna_scores(self, images):
        self.calls["dna"] += 1
        if self.dna_fails:
            raise RuntimeError("boom")
        row = [0.02] * 22
        row[7], row[2] = 0.5, 0.08
        return [row] * len(images)


def service(models=None, accepted=("research-dna-lp22",), **extra):
    return ImageService(settings(**extra), models=models or FakeModels(), acceptance=acceptance(*accepted))


def stored(result):
    result["persistence"] = dict(status="ok", content_id=7)
    return 7


def run(svc, data=None, **kwargs):
    kwargs.setdefault("persist", stored)
    return svc.analyze(data if data is not None else png(), **kwargs)[0]


# ------------------------------------------------------------------ input policy
def _with_byte(data, index, value):
    return data[:index] + bytes([value]) + data[index + 1:]


def _exif(orientation):
    exif = Image.Exif()
    exif[0x0112] = orientation
    return image_bytes(fmt="JPEG", exif=exif.tobytes())


def _animated_png():
    buffer = io.BytesIO()
    frames = [Image.new("RGB", (32, 32), c) for c in ("red", "blue")]
    frames[0].save(buffer, format="PNG", save_all=True, append_images=frames[1:])
    return buffer.getvalue()


def _transparent_rgb_png():
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32)).save(buffer, format="PNG", transparency=(0, 0, 0))
    return buffer.getvalue()


@pytest.mark.parametrize("data,code", [
    (b"", "empty_file"),
    (b"definitely not an image", "decode_failed"),
    (png()[:60], "decode_failed"),
    (b"\xff\xd8\xff" + b"\0" * (10 * 1024 * 1024), "too_large"),
    (png(size=(9000, 4)), "too_large"),
    (png(size=(4001, 4001), mode="L"), "too_large"),
    (png(mode="RGBA"), "unsupported_image_policy"),
    (png(mode="LA"), "unsupported_image_policy"),
    (png(mode="P"), "unsupported_image_policy"),
    (png(mode="I;16"), "unsupported_image_policy"),
    (_with_byte(png(), 24, 16), "unsupported_image_policy"),
    (_transparent_rgb_png(), "unsupported_image_policy"),
    (image_bytes(mode="CMYK", fmt="JPEG"), "unsupported_image_policy"),
    (_exif(6), "unsupported_image_policy"),
    (_animated_png(), "unsupported_image_policy"),
    (image_bytes(fmt="GIF"), "unsupported_media_type"),
    (image_bytes(fmt="WEBP"), "unsupported_media_type"),
    (image_bytes(fmt="BMP"), "unsupported_media_type"),
    (image_bytes(fmt="TIFF"), "unsupported_media_type"),
])
def test_input_policy_rejections(data, code):
    with pytest.raises(WorkbenchError) as error:
        decode_image(data)
    assert error.value.code == code
    assert "/" not in error.value.message.replace("JPEG/PNG", "")  # no paths in messages


@pytest.mark.parametrize("data,mode", [(png(mode="L"), "L"), (image_bytes(fmt="JPEG"), "RGB"), (_exif(1), "RGB"),
                                       (png(size=(4000, 4000), mode="L"), "L")])
def test_input_policy_accepts_static_8bit_rgb_or_l(data, mode):
    rgb, info = decode_image(data)
    assert rgb.mode == "RGB" and info["color_mode"] == mode and info["decode_policy"] == "image-upload-rgb-v1"
    assert info["media_type"] in ("image/png", "image/jpeg") and len(info["sha256"]) == 64


# ------------------------------------------------------------------ standard path
def test_above_threshold_runs_gated_source_and_never_estimates_unknown():
    models = FakeModels(score=0.9)
    r = run(service(models), request_id="00000000-0000-4000-8000-000000000001")
    assert r["schema_version"] == "image-result-v1" and r["status"] == "ok" and r["mode"] == "standard"
    assert r["bundle_id"] == "image-standard-q179-v1" and len(r["registry_sha256"]) == 64
    d, a = r["detection"], r["attribution"]
    assert d == dict(d, status="ok", model_id="det-openclip-q179-v1", score=0.9, score_kind="binary_softmax_p_ai",
                     label="ai", threshold=0.5, threshold_id="fixed-0.5-v1",
                     class_map_sha256=digest_json(["real", "ai"]))
    assert a["status"] == "ok" and a["label_space"] == "known_ai_22" and a["score_kind"] == "known_ai_source_softmax"
    assert a["conditional_statement"] == CONDITIONAL_STATEMENT and a["class_map_sha256"] == digest_json(CLASSES)
    assert [t["rank"] for t in a["top_k"]] == list(range(1, 23))
    assert a["top_k"][0] == dict(rank=1, source=CLASSES[5], score=0.6)  # raw 22-class softmax, not renormalised
    assert a["top_k"][1]["source"] == CLASSES[2]
    assert [t["source"] for t in a["top_k"][2:5]] == [CLASSES[0], CLASSES[1], CLASSES[3]]  # ties: class index
    assert r["unknown"]["status"] == "unavailable" and r["unknown"]["score"] is None
    assert r["research"] == [] and r["persistence"] == dict(status="ok", content_id=7)
    assert r["runtime"]["device"] == "cpu" and r["runtime"]["precision"] == "fp32"
    assert r["runtime"]["timing_ms"]["attribution_ms"] is not None and r["runtime"]["external_calls"] is False


@pytest.mark.parametrize("score,label,attribution", [(0.5, "ai", "ok"), (0.4999, "real", "skipped"), (0.0, "real", "skipped")])
def test_threshold_is_inclusive_and_low_scores_skip_source(score, label, attribution):
    models = FakeModels(score=score)
    r = run(service(models))
    assert r["detection"]["label"] == label and r["attribution"]["status"] == attribution
    if attribution == "skipped":
        assert models.calls["attribute"] == 0 and r["attribution"]["top_k"] == []
        assert r["attribution"]["reason_code"] == "below_detection_threshold"
        assert r["status"] == "ok" and r["runtime"]["timing_ms"]["attribution_ms"] is None


def test_source_failure_is_partial_and_keeps_real_detection():
    r = run(service(FakeModels(score=0.95, attribution_fails=True)))
    assert r["status"] == "partial" and r["detection"]["score"] == 0.95 and r["detection"]["label"] == "ai"
    assert r["attribution"]["status"] == "error" and r["attribution"]["top_k"] == []


@pytest.mark.parametrize("models", [FakeModels(detect_fails=True), FakeModels(score=float("nan"))])
def test_detection_failure_is_an_error_never_zero_or_human(models):
    with pytest.raises(WorkbenchError) as error:
        run(service(models))
    assert error.value.code == "inference_failed" and error.value.status == 500
    assert models.calls["attribute"] == 0


def test_not_ready_states_fail_closed():
    with pytest.raises(WorkbenchError) as error:
        ImageService({}).analyze(png())
    assert error.value.code == "model_unavailable"
    loading = ImageService(settings())
    loading.state = "loading"
    with pytest.raises(WorkbenchError) as error:
        loading.analyze(png())
    assert error.value.code == "model_unavailable" and error.value.retryable


def test_models_require_device_acceptance_for_exact_assets():
    evidence = OVERLAY["acceptance_evidence"]
    for spec in (dict(evidence, path="docs/workbench_preview/evidence/missing.json"), dict(evidence, sha256="0" * 64)):
        svc = ImageService(settings(acceptance_evidence=spec))  # missing file / hash-pin mismatch
        svc.models, svc.acceptance = FakeModels(), None
        svc._load()
        assert svc.state == "unavailable" and svc.error.code == "model_unavailable"
    svc = ImageService(settings(), acceptance=acceptance("det-openclip-q179-v1", "attr-resnet50-q179-v1"))
    svc.models = FakeModels()
    svc._load()
    assert svc.state == "ready" and svc.method_state("research-dna-lp22") == "unvalidated"
    stale = acceptance("det-openclip-q179-v1", "attr-resnet50-q179-v1")
    stale["components"]["attr-resnet50-q179-v1"]["assets_sha256"]["model"] = "0" * 64
    svc = ImageService(settings(), acceptance=stale)
    svc.models = FakeModels()
    svc._load()
    assert svc.state == "unavailable"


def test_busy_when_the_bounded_queue_is_full():
    svc = service()
    for _ in range(2):
        svc._gate.acquire()
    with pytest.raises(WorkbenchError) as error:
        run(svc)
    assert error.value.code == "busy" and error.value.status == 503 and error.value.retryable


def test_lock_timeout_reports_busy():
    svc = service(queue_timeout_seconds=0.05)
    svc._lock.acquire()
    try:
        with pytest.raises(WorkbenchError) as error:
            run(svc)
        assert error.value.code == "busy"
    finally:
        svc._lock.release()


def free_slots(svc):
    return svc._gate._value  # read-only peek; probing with acquire() would race the workers under test


def counting_decode(monkeypatch, fail=False):
    import workbench.image_engine as engine
    calls, original = [], engine.decode_image

    def wrapper(*args, **kwargs):
        calls.append(1)
        if fail:
            raise RuntimeError("decode_was_called")
        return original(*args, **kwargs)
    monkeypatch.setattr(engine, "decode_image", wrapper)
    return calls


def test_full_queue_is_rejected_before_any_decoding(monkeypatch):
    calls = counting_decode(monkeypatch, fail=True)  # reaching decode would raise RuntimeError, not busy
    models = FakeModels()
    svc = service(models)
    while svc._gate.acquire(blocking=False):
        pass
    with pytest.raises(WorkbenchError) as error:
        svc.analyze(png(), request_id="00000000-0000-4000-8000-0000000000c1", persist=stored)
    assert error.value.code == "busy" and calls == [] and models.calls["detect"] == 0 and not svc._inflight
    svc._gate.release()
    svc._gate.release()
    monkeypatch.undo()
    assert run(svc, request_id="00000000-0000-4000-8000-0000000000c1")["status"] == "ok"  # same id retries cleanly


@pytest.mark.parametrize("failure", ["decode", "lock_timeout", "detection", "persist"])
def test_capacity_and_state_recover_after_every_failure_path(failure):
    models = FakeModels(detect_fails=failure == "detection")
    svc = service(models, queue_timeout_seconds=0.05)
    data = b"\x89PNG\r\n\x1a\n" + b"0" * 64 if failure == "decode" else png()
    persist = (lambda r: (_ for _ in ()).throw(OSError("disk"))) if failure == "persist" else stored
    if failure == "lock_timeout":
        svc._lock.acquire()
    try:
        if failure == "persist":
            assert svc.analyze(data, persist=persist)[0]["persistence"]["status"] == "error"
        else:
            with pytest.raises(WorkbenchError) as error:
                svc.analyze(data, persist=persist)
            assert error.value.code == dict(decode="decode_failed", lock_timeout="busy", detection="inference_failed")[failure]
    finally:
        if failure == "lock_timeout":
            svc._lock.release()
    assert free_slots(svc) == 2 and not svc._inflight
    assert svc._lock.acquire(blocking=False)
    svc._lock.release()
    models.detect_fails = False
    assert run(svc, png(size=(70, 40)))["detection"]["status"] == "ok"  # capacity is usable again


def test_concurrent_admission_never_exceeds_max_pending(monkeypatch):
    calls = counting_decode(monkeypatch)
    release, entered = threading.Event(), threading.Event()
    models = FakeModels()
    original = models.detect
    models.detect = lambda images: (entered.set(), release.wait(5), original(images))[2]
    svc = service(models)
    results = []
    workers = [threading.Thread(target=lambda i=i: results.append(run(svc, png(size=(60 + i, 40))))) for i in range(2)]
    for worker in workers:
        worker.start()
    entered.wait(5)
    while free_slots(svc) != 0:  # second request admitted and waiting on the inference lock
        pass
    with pytest.raises(WorkbenchError) as error:
        svc.analyze(png(size=(90, 40)), persist=stored)
    assert error.value.code == "busy" and len(calls) == 2
    release.set()
    for worker in workers:
        worker.join()
    assert len(results) == 2 and models.calls["detect"] == 2 and free_slots(svc) == 2


def test_completed_request_is_reused_even_when_the_queue_is_full():
    models = FakeModels()
    svc = service(models)
    rid = "00000000-0000-4000-8000-0000000000c2"
    first = run(svc, request_id=rid)
    while svc._gate.acquire(blocking=False):
        pass
    again, dedup = svc.analyze(png(), request_id=rid, persist=stored)
    assert dedup and again is first and models.calls["detect"] == 1
    with pytest.raises(WorkbenchError) as error:
        svc.analyze(png(size=(65, 48)), request_id=rid)
    assert error.value.code == "request_id_conflict"


# ------------------------------------------------------------------ request id, modes, methods
def test_request_id_dedupe_and_conflict():
    models = FakeModels()
    svc = service(models)
    rid = "00000000-0000-4000-8000-0000000000aa"
    first = run(svc, request_id=rid)
    again, dedup = svc.analyze(png(), request_id=rid, persist=stored)
    assert dedup and again is first and models.calls["detect"] == 1
    with pytest.raises(WorkbenchError) as error:
        svc.analyze(png(size=(65, 48)), request_id=rid)
    assert error.value.code == "request_id_conflict" and error.value.status == 409
    with pytest.raises(WorkbenchError) as error:
        svc.analyze(png(), mode="research", method_id="research-a0-assumed-known-ai", request_id=rid)
    assert error.value.code == "request_id_conflict"


def test_in_flight_duplicate_is_busy_not_rerun():
    gate = threading.Event()
    models = FakeModels()
    original = models.detect
    models.detect = lambda images: (gate.wait(5), original(images))[1]
    svc = service(models)
    rid = "00000000-0000-4000-8000-0000000000bb"
    worker = threading.Thread(target=lambda: run(svc, request_id=rid))
    worker.start()
    while rid not in svc._inflight:
        pass
    with pytest.raises(WorkbenchError) as error:
        svc.analyze(png(), request_id=rid)
    assert error.value.code == "busy"
    while free_slots(svc) != 1:  # worker admitted and blocked in detection
        pass
    assert free_slots(svc) == 1  # the duplicate did not take a second slot
    gate.set()
    worker.join()
    assert models.calls["detect"] == 1 and free_slots(svc) == 2


@pytest.mark.parametrize("kwargs,code", [
    (dict(request_id="not-a-uuid"), "invalid_request_id"),
    (dict(mode="default"), "unsupported_mode"),
    (dict(mode="standard", method_id="research-dna-lp22"), "unknown_method_id"),
    (dict(mode="research"), "unknown_method_id"),
    (dict(mode="research", method_id="research-ofa-pair"), "unknown_method_id"),  # offline only, not allowlisted
    (dict(mode="research", method_id="../../etc/passwd"), "unknown_method_id"),
])
def test_invalid_parameters_are_rejected(kwargs, code):
    with pytest.raises(WorkbenchError) as error:
        service().analyze(png(), **kwargs)
    assert error.value.code == code and error.value.status == 422


def test_unvalidated_research_method_is_not_runnable():
    svc = service(accepted=())
    assert svc.method_state("research-dna-lp22") == "unvalidated"
    with pytest.raises(WorkbenchError) as error:
        svc.analyze(png(), mode="research", method_id="research-dna-lp22")
    assert error.value.code == "model_unavailable"


def test_research_methods_are_separate_and_never_rewrite_standard():
    standard = run(service(FakeModels(score=0.3)))
    for method in ("research-dna-lp22", "research-a0-assumed-known-ai", "research-openclip-calibrated-thresholds"):
        models = FakeModels(score=0.3)
        r = run(service(models), mode="research", method_id=method)
        assert r["mode"] == "research" and len(r["research"]) == 1
        assert r["detection"] == standard["detection"] and r["attribution"]["status"] == "skipped"
        assert r["unknown"] == standard["unknown"]
        item = r["research"][0]
        assert item["method_id"] == method and item["status"] == "ok" and item["limitations"]
    dna = run(service(FakeModels(score=0.3)), mode="research", method_id="research-dna-lp22")["research"][0]
    assert dna["outputs"]["top_k"][0] == dict(rank=1, source=CLASSES[7], score=0.5)
    assumed = run(service(FakeModels(score=0.3)), mode="research", method_id="research-a0-assumed-known-ai")["research"][0]
    assert assumed["outputs"]["top_k"][0]["source"] == CLASSES[5]
    cal = run(service(FakeModels(score=0.3)), mode="research",
              method_id="research-openclip-calibrated-thresholds")["research"][0]["outputs"]
    assert cal["p_ai"] == 0.3 and [t["label"] for t in cal["thresholds"]] == ["real", "ai"]
    assert [t["threshold_id"] for t in cal["thresholds"]] == ["cal-fpr-0.01-v1", "cal-fpr-0.05-v1"]


def test_research_failure_is_partial():
    r = run(service(FakeModels(dna_fails=True)), mode="research", method_id="research-dna-lp22")
    assert r["status"] == "partial" and r["research"][0]["status"] == "error" and r["detection"]["status"] == "ok"


# ------------------------------------------------------------------ persistence
def test_storage_failure_is_partial_and_never_claims_saved():
    def broken(result):
        result["persistence"] = dict(status="ok", content_id=3)  # set before commit, then the commit fails
        raise RuntimeError("disk full")
    r = service().analyze(png(), persist=broken)[0]
    assert r["status"] == "partial" and r["persistence"] == dict(status="error", content_id=None,
                                                                 reason_code="storage_failed")
    assert r["detection"]["status"] == "ok"


# ------------------------------------------------------------------ assets and capabilities
def test_verify_asset_rejects_escape_size_and_hash(tmp_path):
    data = tmp_path / "w.bin"
    data.write_bytes(b"weights")
    import hashlib
    good = dict(path="w.bin", bytes=7, sha256=hashlib.sha256(b"weights").hexdigest())
    assert verify_asset(tmp_path, good, "w") == tmp_path / "w.bin"
    for spec, code in ((dict(good, path="../w.bin"), "artifact_mismatch"), (dict(good, path=str(data)), "artifact_mismatch"),
                       (dict(good, bytes=8), "artifact_mismatch"), (dict(good, sha256="0" * 64), "artifact_mismatch"),
                       (dict(good, path="missing.bin"), "model_unavailable")):
        with pytest.raises(WorkbenchError) as error:
            verify_asset(tmp_path, spec, "w")
        assert error.value.code == code
    (tmp_path / "link.bin").symlink_to("/etc/hosts")
    with pytest.raises(WorkbenchError) as error:
        verify_asset(tmp_path, dict(good, path="link.bin"), "w")
    assert error.value.code == "artifact_mismatch"


def test_real_loader_fails_closed_on_registry_mismatch(tmp_path):
    bad = settings()
    bad["research_root"] = str(tmp_path)  # assets missing entirely
    with pytest.raises(WorkbenchError) as error:
        ImageModels(bad).load()
    assert error.value.code in ("model_unavailable", "artifact_mismatch")
    bad["registry"] = json.loads(json.dumps(bad["registry"]))
    bad["registry"]["code_sha256"]["scripts/run_image_dna.py"] = "0" * 64
    with pytest.raises(WorkbenchError) as error:
        ImageModels(bad).load()
    assert error.value.code == "artifact_mismatch"


def test_capabilities_expose_policy_and_states_without_paths():
    caps = service().capabilities()
    text = json.dumps(caps, ensure_ascii=False)
    assert "/home/" not in text and "outputs/" not in text and "nonexistent" not in text
    assert caps["input_policy"]["max_bytes"] == 10 * 1024 * 1024 and caps["input_policy"]["max_pixels"] == 16_000_000
    assert caps["unknown"]["score"] is None and caps["external_calls"] is False
    states = {m["id"]: m["run_allowed"] for m in caps["research_methods"]}
    assert states == {"research-dna-lp22": True, "research-a0-assumed-known-ai": True,
                      "research-openclip-calibrated-thresholds": True}
    assert all(not m["run_allowed"] for m in caps["offline_only_methods"])
    assert {m["id"] for m in caps["offline_only_methods"]} >= {"research-ofa-pair", "research-ofa-tplus1",
                                                               "research-score-fusion-uniform"}
