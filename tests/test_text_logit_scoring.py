import json
import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import yaml
from transformers import BertTokenizerFast

from detection.false_positive_controller import FalsePositiveController
from detection.providers import ApiFirstDetectionEngine
from detection.text_detector import RoutedTextAigcDetector
from scripts.finetune_text_detector import score_documents
from tests.test_finetune_text_detector import tiny_bert


def test_batched_logit_validation_matches_runtime(tiny_bert):
    tokenizer, model = tiny_bert
    model.eval()
    rows = [{"text": "人" * 12 + "智" * 2}, {"text": "人。智。" * 135}]
    detector = RoutedTextAigcDetector(models={"zh_short": "fixture", "zh_long": "fixture"}, device="cpu",
                                    max_length=8, aggregation_mode="logit_mean")
    detector._loaded = {key: (tokenizer, model) for key in detector.models}

    actual = score_documents(model, tokenizer, rows, device="cpu", batch_size=24, max_length=8, aggregation_mode="logit_mean")

    assert actual == pytest.approx([detector.detect(row["text"]).score for row in rows], abs=1e-7)
    assert model.training is False


def test_logit_windows_weight_only_content_tokens_and_use_label_order(tmp_path):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n人\n智\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab))

    class Classifier:
        config = SimpleNamespace(id2label={0: "ai", 1: "human"}, max_position_embeddings=8)

        def __call__(self, input_ids, **kwargs):
            margin = 6.0 if (input_ids == 6).any() else -2.0
            return SimpleNamespace(logits=torch.tensor([[margin + 3, 3.0]]))

    detector = RoutedTextAigcDetector({}, device="cpu", aggregation_mode="logit_mean")
    with patch.object(detector, "_load_model", return_value=(tokenizer, Classifier())):
        assert detector._run_model("zh_long", "unused", "人" * 6 + "智" * 2) == 0.0


def test_document_margin_weights_only_previously_uncovered_characters():
    text = "中" * 20
    segments = [SimpleNamespace(index=i, start=start, end=end, text=text[start:end])
                for i, (start, end) in enumerate(((0, 10), (7, 20)))]
    detector = RoutedTextAigcDetector({"zh_short": "fake"}, device="cpu",
                                    aggregation_mode="logit_mean", temperature=2.0)
    with patch.object(detector.zh_segmenter, "split", return_value=segments), \
         patch.object(detector, "_run_model", side_effect=[2.0, 6.0]):
        result = detector.detect(text)
    assert result.details["raw_margin"] == 4.0
    assert result.details["temperature"] == 2.0
    assert [x["raw_margin"] for x in result.details["segments"]] == [2.0, 6.0]
    assert result.score == pytest.approx(0.8807970779778823, abs=1e-12)


def test_temperature_preserves_order_and_fingerprints_configuration():
    legacy = RoutedTextAigcDetector({}, device="cpu")
    assert legacy.scoring_protocol == RoutedTextAigcDetector({}, device="cpu", aggregation_mode="probability_mean").scoring_protocol
    t1 = RoutedTextAigcDetector({}, device="cpu", aggregation_mode="logit_mean")
    t2 = RoutedTextAigcDetector({}, device="cpu", aggregation_mode="logit_mean", temperature=2.0)
    assert len({legacy.scoring_protocol, t1.scoring_protocol, t2.scoring_protocol}) == 3
    for detector in (t1, t2):
        scores = [detector._as_probability(m) for m in (-1000, -2, 0, 2, 1000)]
        assert 0 < scores[0] < scores[1] < scores[2] == 0.5 < scores[3] < scores[4] < 1
    assert t2._as_probability(4.0) < t1._as_probability(4.0)
    for invalid in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="temperature"):
            RoutedTextAigcDetector({}, device="cpu", aggregation_mode="logit_mean", temperature=invalid)


def test_logit_protocol_rejects_an_old_cutoff_instead_of_only_hiding_metrics():
    legacy = RoutedTextAigcDetector({}, device="cpu")
    candidate = RoutedTextAigcDetector({}, device="cpu", aggregation_mode="logit_mean", temperature=2.0)
    controller = FalsePositiveController()
    controller.artifact = {"scoring_protocol": legacy.scoring_protocol, "groups": {
        "zh": {"low_threshold": 0.4, "high_threshold": 0.91, "validation_fpr": 0.02}}}
    result = SimpleNamespace(status="ok", score=0.94, details={"scoring_protocol": candidate.scoring_protocol,
                                                              "aggregation_mode": "logit_mean"})
    decision = controller.evaluate("中文文本测试。" * 40, [result])
    assert decision.decision == "insufficient_evidence"
    assert decision.high_threshold != 0.91
    assert decision.threshold_validation_fpr is None
    assert "scoring_protocol_not_evaluated" in decision.threshold_warnings


def test_text_engine_does_not_round_away_a_valid_strict_threshold_crossing(tmp_path):
    config = tmp_path / "config.yaml"
    artifact = tmp_path / "thresholds.json"
    config.write_text(yaml.safe_dump({"detection": {
        "demo_provider_enabled": False,
        "local_models": {"enabled": True, "text_yuchuan": {"enabled": True, "device": "cpu",
            "models": {"zh_short": "test"}, "aggregation_mode": "logit_mean", "temperature": 2.0}},
        "text_false_positive_control": {"enabled": True, "artifact_path": str(artifact)},
    }}))
    engine = ApiFirstDetectionEngine(config_path=str(config))
    detector = engine._get_local_text_detector(engine.local_cfg["text_yuchuan"])
    artifact.write_text(json.dumps({"scoring_protocol": detector.scoring_protocol, "groups": {
        "zh": {"low_threshold": 0.4, "high_threshold": 0.800040}}}))
    engine = ApiFirstDetectionEngine(config_path=str(config))
    detector = engine._get_local_text_detector(engine.local_cfg["text_yuchuan"])
    margin = 2 * math.log(0.800043 / (1 - 0.800043))
    with patch.object(detector, "_run_model", return_value=margin):
        result = engine.detect(tmp_path / "example.txt", "text", "中" * 200)
    assert detector.temperature == 2.0
    assert result.score == pytest.approx(0.800043, abs=1e-8)
    assert result.threshold == 0.800040
    assert result.label == "high_risk_ai"
