from scripts.prepare_text_detection_v2 import SplitGuard, fingerprints


def test_group_and_normalized_lead_never_cross_fit_partitions():
    intro = "开头重复" * 40
    heldout = {"text": intro + "原结尾", "group": "test-group"}
    row = {"text": "，".join(intro) + "新结尾", "group": "alternate", "source": "fixture", "label": 0}
    assert fingerprints(heldout["text"])[0] != fingerprints(row["text"])[0]
    assert fingerprints(heldout["text"])[1] == fingerprints(row["text"])[1]
    guard = SplitGuard([heldout])
    assert not guard.accept([row], "calibration")
    assert not guard.accept([dict(row, group="test-group", text="完全不同")], "validation")
    independent = dict(row, text="干净真人正文", group="new")
    assert not guard.accept([independent, dict(row, label=1)], "train")
    assert guard.accept([independent], "validation")
    assert not guard.accept([dict(independent, text="同源的新版本")], "train")
