import hashlib
import importlib
import json

import pytest
from PIL import Image


def audit():
    return importlib.import_module("scripts.audit_ffhq_errors")


def row(name="one", split="dev"):
    return dict(sample_id=name, group_id=name, split=split, is_ai=0,
                semantic_class="FFHQ", source_label=None,
                width=256, height=256, format="PNG", relative_path="one.png")


def prediction(name="one", role="evaluation", score=0.7):
    return dict(sample_id=name, group_id=name, role=role, condition="clean",
                is_ai=0, semantic_class="FFHQ", source_label=None,
                classes=["real", "ai"], scores=[1-score, score])


def test_holdout_prediction_is_rejected():
    with pytest.raises(ValueError, match="holdout"):
        audit().validate_predictions([prediction()], {"one": row(split="holdout")},
                                    {"evaluation": ["one"], "calibration": []})


def test_image_inspection_rejects_holdout_before_file_access(tmp_path):
    with pytest.raises(ValueError, match="holdout"):
        audit().inspect_image(row(split="holdout"), tmp_path)


@pytest.mark.parametrize("change", ["role", "id", "missing", "duplicate", "metadata", "nonfinite"])
def test_role_and_id_mismatch_fail_closed(change):
    predictions = [prediction()]
    if change == "role":
        predictions[0]["role"] = "calibration"
    elif change == "id":
        predictions[0]["sample_id"] = "unknown"
    elif change == "missing":
        predictions = []
    elif change == "duplicate":
        predictions *= 2
    elif change == "metadata":
        predictions[0]["is_ai"] = 1
    else:
        predictions[0]["scores"] = [float("nan"), 0.7]
    with pytest.raises(ValueError):
        audit().validate_predictions(predictions, {"one": row()},
                                    {"evaluation": ["one"], "calibration": []})


def test_clean_predictions_are_keyed_by_role_and_id():
    actual = audit().validate_predictions(
        [prediction(), dict(prediction(), condition="jpeg75")],
        {"one": row()}, {"evaluation": ["one"], "calibration": []})
    assert list(actual["evaluation"]) == ["one"]
    assert actual["evaluation"]["one"]["scores"][1] == 0.7


def test_matched_scores_compare_identical_ids_and_fixed_threshold():
    base = {"one": prediction(score=.7), "two": prediction("two", score=.2)}
    reference = {"one": prediction(score=.1), "two": prediction("two", score=.8)}
    actual = audit().paired_summary(base, reference)
    assert actual["n"] == 2
    assert actual["primary_fp"] == 1
    assert actual["reference_fp"] == 1
    assert actual["primary_only_fp"] == 1
    assert actual["reference_only_fp"] == 1
    with pytest.raises(ValueError):
        audit().paired_summary(base, {"one": reference["one"]})


def test_header_inspection_records_real_png_geometry(tmp_path):
    path = tmp_path / "one.png"
    Image.new("RGB", (256, 256), "white").save(path)
    record = dict(row(), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    actual = audit().inspect_image(record, tmp_path)
    assert actual["format"] == "PNG"
    assert actual["resolution"] == "256x256"
    assert actual["crop_retained_fraction"] == pytest.approx(0.765625)
    assert actual["png_chunks"] == "IHDR,IDAT,IEND"
    assert actual["jpeg_quantization_sha256"] is None
    assert actual["sha256_verified"] is True


def test_header_inspection_rejects_wrong_file_bytes(tmp_path):
    Image.new("RGB", (256, 256)).save(tmp_path / "one.png")
    with pytest.raises(ValueError, match="hash"):
        audit().inspect_image(dict(row(), sha256="0"*64), tmp_path)



@pytest.mark.parametrize("status", [
    {"status": "running", "exit_code": 0},
    {"status": "completed", "exit_code": 1},
    {"status": "completed"},
])
def test_saved_run_requires_successful_completion(tmp_path, status):
    (tmp_path / "status.json").write_text(json.dumps(status))
    with pytest.raises(ValueError, match="completed"):
        audit().verify_saved_run(tmp_path)


def test_saved_checkpoint_must_match_recorded_hash(tmp_path):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint fixture")
    (tmp_path / "status.json").write_text(json.dumps({"status": "completed", "exit_code": 0}))
    (tmp_path / "metrics.json").write_text(json.dumps({
        "checkpoint": str(checkpoint), "checkpoint_sha256": "0"*64,
    }))
    with pytest.raises(ValueError, match="checkpoint"):
        audit().verify_saved_run(tmp_path)


def test_successful_saved_run_binds_checkpoint_file_bytes(tmp_path):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint fixture")
    expected = hashlib.sha256(b"checkpoint fixture").hexdigest()
    (tmp_path / "status.json").write_text(json.dumps({"status": "completed", "exit_code": 0}))
    (tmp_path / "metrics.json").write_text(json.dumps({
        "checkpoint": str(checkpoint), "checkpoint_sha256": expected,
    }))
    actual = audit().verify_saved_run(tmp_path)
    assert actual["checkpoint_sha256"] == expected
    assert actual["status"]["exit_code"] == 0
