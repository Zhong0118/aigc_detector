from scripts.validate_text_release import build_calibration_artifact


def test_published_metrics_use_test_instead_of_checkpoint_selection_validation():
    rows = [dict(split="calibration", label=0, result={"score": 0.5}) for _ in range(100)]
    rows += [dict(split="validation", label=0, result={"score": 0.9}),
             dict(split="validation", label=1, result={"score": 0.1}),
             dict(split="test", label=0, result={"score": 0.1}),
             dict(split="test", label=1, result={"score": 0.9})]
    artifact = build_calibration_artifact(rows, "fixed-model-v1", "release")
    assert artifact["metric_partition"] == "test"
    assert artifact["groups"]["zh"]["validation_fpr"] == 0
    assert artifact["groups"]["zh"]["ai_recall"] == 1


def test_mixed_language_has_its_own_calibration_evidence():
    chinese = "中文技术说明。" * 30
    mixed = chinese + " English terms and technical details " * 15
    rows = [dict(text=chinese, split="calibration", label=0, result={"score": 0.5}) for _ in range(100)]
    rows += [dict(text=mixed, split="calibration", label=0, result={"score": 0.7}) for _ in range(100)]
    artifact = build_calibration_artifact(rows, "fixed-model-v1", "release")
    assert artifact["groups"]["zh"]["high_threshold"] == 0.5
    assert artifact["groups"]["mixed"]["high_threshold"] == 0.7
    assert artifact["groups"]["mixed"]["human_calibration_samples"] == 100
