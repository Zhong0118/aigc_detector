from scripts.prepare_text_detection_data import prepare, digest


def test_aliases_propagate_holdout_and_excluded_text_never_reaches_training():
    text = "a complete human response " * 10
    rows = [dict(text=text, label=0, group="a", split="train", source="fixture"),
            dict(text=text, label=0, group="b", split="test", source="fixture"),
            dict(text="a generated answer " * 10, label=1, group="a", split="train", source="fixture")]
    selected = prepare(rows)
    assert len(selected) == 2
    assert {r["split"] for r in selected} == {"test"}
    assert prepare(rows, excluded=[digest(" ".join(text.split()))]) == []
