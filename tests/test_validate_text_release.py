import pytest
from types import SimpleNamespace

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


def test_temperature_fit_reduces_nll_and_rejects_nonvalidation_rows():
    from scripts.validate_text_release import fit_temperature

    rows = [{"split": "validation", "label": label, "result": {"details": {"raw_margin": margin}}}
            for label, margin in [(0, -8), (0, -8), (1, 8), (1, 8), (0, 8), (1, -8)]]
    fit = fit_temperature(rows)
    assert fit["temperature"] == pytest.approx(8 / 0.6931471805599453, rel=1e-5)
    assert fit["validation_nll_after"] < fit["validation_nll_before"]
    for split in ("test", "calibration", "external", "regression"):
        with pytest.raises(ValueError, match="validation"):
            fit_temperature([*rows, {**rows[0], "split": split}])


def test_screening_cutoff_uses_validation_only_and_retains_long_ai():
    from scripts.validate_text_release import fit_screening_threshold, screening_positive

    rows = [dict(split="validation", label=1, text="人" * chars, result={"score": score})
            for chars, score in [(550, 0.85), (550, 0.9), (550, 0.9), (550, 0.95), (1500, 0.65), (1520, 0.7)]]
    fit = fit_screening_threshold(rows, target_recall=0.8)
    assert fit["threshold"] == 0.65
    assert fit["partitions"]["long_ai"]["samples"] == 2
    assert screening_positive("uncertain", 0.65, fit["threshold"])
    assert not screening_positive("insufficient_evidence", 0.9, fit["threshold"])
    assert not screening_positive("uncertain", 0.64999, fit["threshold"])
    assert screening_positive("high_risk_ai", 0.5, fit["threshold"])
    with pytest.raises(ValueError, match="validation"):
        fit_screening_threshold([*rows, dict(rows[0], split="test")], target_recall=0.8)


def test_chinese_model_with_english_acronyms_uses_its_own_calibration(tmp_path):
    from detection.false_positive_controller import FalsePositiveController
    from detection.text_threshold_artifact import model_calibration_group
    from scripts.validate_text_release import build_branch_artifact, fit_branch_temperatures
    import json

    text = "中文密码算法比较附录讨论 RSA ECDSA PQC 的基本概念。" * 12
    aliases = {"zh_long": "zh", "zh_short": "zh", "en_long": "en_long"}

    def row(split, label, score, model):
        return dict(text=text, split=split, label=label, source="fixture", result=dict(
            score=score, details=dict(selected_models=[model], raw_margin=(2 * label - 1) * 3.0)))

    rows = [row("calibration", 0, 0.60, "zh_long") for _ in range(50)]
    rows += [row("calibration", 0, 0.97, "en_long") for _ in range(50)]
    rows += [row("validation", label, 0.8, model) for label in (0, 1) for model in ("zh_long", "en_long")]
    temperatures = fit_branch_temperatures(rows, aliases)
    assert {group: entry["samples"] for group, entry in temperatures["groups"].items()} == {"zh": 2, "en_long": 2}
    artifact = build_branch_artifact(rows, "token-logit-v1:fixture", "fixture", aliases, "pooled")
    assert artifact["groups"]["zh"]["high_threshold"] == 0.60
    artifact["groups"]["zh"]["screening_threshold"] = 0.5
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps(artifact))
    ctl = FalsePositiveController({"artifact_path": str(path)})
    assert ctl._language(text) == "mixed"
    details = dict(selected_models=["zh_long"], scoring_protocol=artifact["scoring_protocol"])
    assert model_calibration_group(details, aliases) == "zh"
    decision = ctl.evaluate(text, [SimpleNamespace(status="ok", score=0.8, details=details)])
    assert (decision.threshold_group, decision.decision) == ("zh", "high_risk_ai")
    assert decision.to_dict()["screening_threshold"] == 0.5
    details["selected_models"] = ["zh_long", "en_long"]
    decision = ctl.evaluate(text, [SimpleNamespace(status="ok", score=0.8, details=details)])
    assert decision.decision == "insufficient_evidence"


def test_branch_screening_uses_human_budget_and_stays_below_strict_cutoff():
    from scripts.validate_text_release import fit_branch_screening

    scores = [i / 100 for i in range(1, 101)]
    fit = fit_branch_screening(scores, high_threshold=0.99, target_fpr=0.05)
    assert 0.95 < fit["threshold"] < 0.99
    assert sum(score >= fit["threshold"] for score in scores) == 5
    with pytest.raises(ValueError, match="strict"):
        fit_branch_screening(scores, high_threshold=0.95, target_fpr=0.05)


def test_conservative_cutoff_uses_only_human_calibration_and_ignores_sparse_sources():
    from scripts.validate_text_release import build_branch_artifact

    def row(source, split, label, score):
        return dict(source=source, split=split, label=label,
                    result=dict(score=score, details=dict(selected_models=["zh_long"])))

    rows = [row("news", "calibration", 0, 0.2) for _ in range(1000)]
    rows += [row("hc3", "calibration", 0, score) for score in [0.3] * 98 + [0.8, 0.85]]
    rows += [row("few", "calibration", 0, 0.97), row("hc3", "test", 0, 0.99), row("hc3", "validation", 0, 0.99)]
    args = (rows, "token-logit-v1:fixture", "fixture", {"zh_long": "zh"})
    pooled = build_branch_artifact(*args, "pooled")["groups"]["zh"]
    conservative = build_branch_artifact(*args, "source_max")["groups"]["zh"]
    assert pooled["high_threshold"] == 0.3
    assert conservative["high_threshold"] == 0.8


def test_strict_error_budget_is_independent_for_each_model_branch():
    from detection.text_threshold_artifact import validate_text_threshold_artifact
    from scripts.validate_text_release import build_branch_artifact

    rows = [dict(source="fixture", split="calibration", label=0,
                 result=dict(score=i / 200, details={"selected_models": [branch]}))
            for branch in ("zh_short", "zh_long") for i in range(1, 121)]
    artifact = build_branch_artifact(rows, "token-logit-v1:fixture", "fixture",
                                     {"zh_short": "zh_short", "zh_long": "zh_long"}, "pooled",
                                     target_fpr_by_group={"zh_short": 0.01})
    assert artifact["groups"]["zh_short"]["high_threshold"] == 0.6
    assert artifact["groups"]["zh_long"]["high_threshold"] == 0.595
    assert validate_text_threshold_artifact(artifact, mode="readiness", expected_dataset_profile="binary-chinese-v1").valid
