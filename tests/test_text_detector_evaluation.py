from dataclasses import replace

import pytest

from detection.text_resource_adapters import TextSample
from scripts.evaluate_text_detector import select_samples, binary_metrics


def test_selection_is_deduplicated_grouped_and_order_independent():
    samples = []
    for i in range(40):
        for label in ("human", "ai"):
            samples.append(TextSample(
                text=f"{label} answer {i}", label=label, language="zh", domain="general_qa",
                source="fixture", source_id=f"{i}-{label}", partition_id=f"question-{i}",
            ))
    duplicate = replace(samples[0], source_id="duplicate")
    selected = select_samples(samples + [duplicate], limit=100, seed=17)
    assert selected == select_samples(list(reversed(samples + [duplicate])), limit=100, seed=17)
    calibration = {s.partition_key for role, s in selected if role == "calibration"}
    test = {s.partition_key for role, s in selected if role == "test"}
    assert calibration and test and calibration.isdisjoint(test)
    assert len({s.text for _, s in selected}) == len(selected)
    assert all(s.label == "human" for role, s in selected if role == "calibration")


def test_metrics_count_misses_and_false_positives_at_strict_cutoff():
    result = binary_metrics(["human", "human", "ai", "ai"], [0.1, 0.8, 0.8, 0.9], 0.8)
    assert result["tp"] == 1
    assert result["fn"] == 1
    assert result["fp"] == 0
    assert result["tn"] == 2
    assert result["recall"] == 0.5
    assert result["fpr"] == 0
    assert binary_metrics(["ai"], [0.1], 0.8)["fpr"] is None
    assert binary_metrics(["human"], [0.1], 0.8)["auroc"] is None
    with pytest.raises(ValueError):
        binary_metrics(["ai"], [float("nan")], 0.8)
    with pytest.raises(ValueError):
        binary_metrics(["ai"], [], 0.8)


def test_duplicate_aliases_cannot_leak_a_related_question_into_test():
    samples = [
        TextSample(text="same human response", label="human", language="zh", domain="general_qa",
                   source="fixture", source_id=group, partition_id=group)
        for group in ("q2", "q0")
    ]
    samples.append(TextSample(text="unique AI answer B", label="ai", language="zh", domain="general_qa",
                              source="fixture", source_id="q0-ai", partition_id="q0"))
    selected = select_samples(samples, limit=100, seed=17)
    assert len({role for role, _ in selected}) == 1
