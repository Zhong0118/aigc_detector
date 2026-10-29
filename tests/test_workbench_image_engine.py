"""Image adapter semantics with fake models: validation, gating, failures, no fake success."""
import io
import json
import threading

import pytest
from PIL import Image

from workbench.image_engine import (CONDITIONAL_STATEMENT, ImageService, WorkbenchError, decode_image)

CLASSES = sorted(f"src{i:02d}" for i in range(22))


def png(size=(64, 48), mode="RGB", **save):
    buffer = io.BytesIO()
    Image.new(mode, size).save(buffer, format=save.pop("format", "PNG"), **save)
    return buffer.getvalue()


class FakeModels:
    def __init__(self, score=0.9, attribution_fails=False, detect_fails=False):
        self.score, self.attribution_fails, self.detect_fails = score, attribution_fails, detect_fails
        self.bundle = {"dopenclip_head": "a" * 64, "a0_model": "b" * 64}
        self.fixed_metrics = dict(n=2310, negative_count=979, positive_count=1331, fpr=0.0194, tpr=0.9895)
        self.calibrated = {"0.01": dict(calibration_negative_n=1014, calibration_fpr=0.0099,
                                        evaluation=dict(threshold=0.74, n=2310, negative_count=979, fpr=0.0102, tpr=0.97)),
                           "0.05": dict(calibration_negative_n=1014, calibration_fpr=0.049,
                                        evaluation=dict(threshold=0.27, n=2310, negative_count=979, fpr=0.041, tpr=0.995))}
        self.a0_classes, self.a0_label_space = CLASSES, "iab-22:test"
        self.preprocessing = dict(size=224)
        self.ofa_ready, self.ofa_error = False, None
        self.calls = 0

    def detect(self, images):
        self.calls += 1
        if self.detect_fails:
            raise RuntimeError("boom")
        if self.score != self.score:  # NaN must never escape
            raise ValueError("model returned a non-finite or out-of-range score")
        return [self.score] * len(images)

    def attribute(self, images):
        if self.attribution_fails:
            raise RuntimeError("boom")
        row = [0.01] * 22
        row[5], row[2], row[9] = 0.6, 0.2, 0.0
        return [row] * len(images)


def service(models=None, **settings):
    return ImageService(dict(dict(research_root="/nonexistent", research_ofa=False,
                                  research_ofa_note="cpu_reproduction_exceeds_registered_tolerance"), **settings),
                        models=models or FakeModels())


@pytest.mark.parametrize("data,code", [
    (b"", "invalid_input"), (b"not an image at all", "invalid_input"),
    (png()[:60], "decode_failed"), (png(size=(9000, 40)), "too_large"),
    (png(size=(16, 16)), "invalid_input"), (png(mode="I;16"), "invalid_input"),
    (png(format="TIFF"), "invalid_input"), (png(format="GIF"), "invalid_input"),
])
def test_server_side_validation_rejects_with_typed_errors(data, code):
    with pytest.raises(WorkbenchError) as error:
        decode_image(data)
    assert error.value.code == code


def test_byte_limit_and_multi_frame_are_rejected():
    with pytest.raises(WorkbenchError) as error:
        decode_image(png(), dict(max_bytes=10))
    assert error.value.code == "too_large"
    buffer = io.BytesIO()
    frames = [Image.new("RGB", (64, 64), c) for c in ("red", "blue")]
    frames[0].save(buffer, format="PNG", save_all=True, append_images=frames[1:])
    with pytest.raises(WorkbenchError) as error:
        decode_image(buffer.getvalue())
    assert error.value.code == "invalid_input"


def test_alpha_and_exif_follow_research_loader_without_silent_rotation():
    rgb, info = decode_image(png(mode="RGBA"))
    assert rgb.mode == "RGB" and info["has_alpha"] and info["decode_policy"]["alpha"] == "discarded_by_convert"
    exif = Image.Exif()
    exif[274] = 6
    rgb, info = decode_image(png(size=(80, 40), format="JPEG", exif=exif))
    assert rgb.size == (80, 40) and info["exif_orientation"] == 6
    assert info["decode_policy"]["exif_orientation"] == "not_applied"


def test_above_threshold_gives_conditional_top_k_and_unknown_stays_null():
    result = service().analyze(png(), "a.png")
    assert result["status"] == "ok" and result["detection"]["label"] == "at_or_above_threshold"
    assert result["detection"]["calibrated_probability"] is False
    assert result["attribution"]["status"] == "ok"
    assert result["attribution"]["conditional_statement"] == CONDITIONAL_STATEMENT
    top = result["attribution"]["top_k"]
    assert len(top) == 22 and [t["label"] for t in top[:2]] == [CLASSES[5], CLASSES[2]]
    assert result["unknown"] == dict(status="unavailable", score=None,
                                     reason="main_22_class_interface_has_no_accepted_unknown_rejection")
    json.dumps(result, allow_nan=False)


def test_below_threshold_main_conclusion_carries_no_source():
    models = FakeModels(score=0.2)
    result = service(models).analyze(png(), "a.png")
    assert result["detection"]["label"] == "below_threshold"
    assert result["attribution"]["status"] == "not_applicable" and result["attribution"]["top_k"] == []
    research = service(FakeModels(score=0.2)).analyze(png(), "a.png", research=True)
    ids = [item["id"] for item in research["research"]]
    assert ids == ["d_openclip_calibrated_thresholds", "a0_closed_set_research", "ofa_pair_verification", "registered_only"]
    assert research["attribution"]["top_k"] == []
    ofa = research["research"][2]
    assert ofa["status"] == "unavailable" and ofa["scores"] == []
    assert ofa["reason"] == "cpu_reproduction_exceeds_registered_tolerance"
    assert [row["reached"] for row in research["research"][0]["rows"]] == [False, False]


def test_partial_and_failed_inference_never_fake_success():
    result = service(FakeModels(attribution_fails=True)).analyze(png(), "a.png")
    assert result["status"] == "partial" and result["attribution"]["status"] == "error"
    assert result["attribution"]["top_k"] == []
    for models in (FakeModels(detect_fails=True), FakeModels(score=float("nan"))):
        with pytest.raises(WorkbenchError) as error:
            service(models).analyze(png(), "a.png")
        assert error.value.code == "inference_failed" and error.value.retryable


def test_unconfigured_or_loading_service_is_unavailable_not_zero():
    with pytest.raises(WorkbenchError) as error:
        ImageService({}).analyze(png(), "a.png")
    assert error.value.code == "model_unavailable" and not error.value.retryable
    loading = ImageService(dict(research_root="/nonexistent"))
    loading.state = "loading"
    with pytest.raises(WorkbenchError) as error:
        loading.analyze(png(), "a.png")
    assert error.value.retryable


def test_request_id_dedupes_and_busy_is_retryable():
    models = FakeModels()
    svc = service(models)
    first = svc.analyze(png(), "a.png", request_id="req-00000001")
    again = svc.analyze(png(), "a.png", request_id="req-00000001")
    assert again["analysis_id"] == first["analysis_id"] and again["deduplicated"] and models.calls == 1
    busy = service(max_pending=1)
    assert busy._gate.acquire(blocking=False)
    with pytest.raises(WorkbenchError) as error:
        busy.analyze(png(), "a.png")
    assert error.value.code == "busy" and error.value.retryable
    busy._gate.release()


def test_inference_is_serialized():
    active, peak, lock = [0], [0], threading.Lock()

    class Slow(FakeModels):
        def detect(self, images):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            threading.Event().wait(0.05)
            with lock:
                active[0] -= 1
            return super().detect(images)

    svc = service(Slow(), max_pending=4)
    threads = [threading.Thread(target=svc.analyze, args=(png(), "a.png")) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak[0] == 1
