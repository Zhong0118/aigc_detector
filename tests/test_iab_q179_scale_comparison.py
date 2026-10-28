"""CPU fixtures for the 100/30 versus 200-q179/30 same-sample comparison helpers."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/compare_iab_v6_100_vs_v7_200_q179.py"
spec = importlib.util.spec_from_file_location("q179_comparison", SCRIPT)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def row(sid, group, is_ai=1, source="a"):
    return dict(sample_id=sid, split="dev", group_id=group, is_ai=is_ai, source_label=source if is_ai else None)


def parts():
    roles = {role: [row(f"{role}-{i}", f"g-{role}-{i}") for i in range(2)] for role in comparison.ROLES}
    return {"detection": roles, "attribution": copy.deepcopy(roles)}


def test_preserved_roles_accept_extension_with_new_ids():
    old = parts(); new = copy.deepcopy(old)
    new["detection"]["train"].append(row("extra", "g-extra"))
    counts = comparison.verify_preserved_roles(old, new)
    assert counts["detection"]["evaluation"] == 2


@pytest.mark.parametrize("change", ["moved", "group", "label", "missing"])
def test_preserved_roles_reject_any_original_change(change):
    old = parts(); new = copy.deepcopy(old)
    item = new["detection"]["evaluation"][0]
    if change == "moved":
        new["detection"]["selection"].append(new["detection"]["evaluation"].pop(0))
    elif change == "group":
        item["group_id"] = "other"
    elif change == "label":
        item["source_label"] = "b"
    else:
        new["detection"]["evaluation"].pop(0)
    with pytest.raises(ValueError):
        comparison.verify_preserved_roles(old, new)


def test_reference_diff_counts_shared_and_replaced_ids():
    diff = comparison.reference_diff({"a": ["1", "2"], "b": ["3"]}, {"a": ["1", "4"], "b": ["3"]})
    assert diff == [dict(source="a", old_n=2, new_n=2, shared=1, removed=1, added=1),
                    dict(source="b", old_n=1, new_n=1, shared=1, removed=0, added=0)]
    with pytest.raises(ValueError):
        comparison.reference_diff({"a": ["1"]}, {"b": ["1"]})


def test_threshold_bootstrap_is_zero_for_identical_models_and_signed_for_lower_threshold():
    binary = np.tile([0, 0, 1, 1, 0, 1], 10); scores = np.tile([.1, .6, .7, .9, .3, .4], 10)
    groups = [f"g{i // 2}" for i in range(60)]
    same = comparison.threshold_bootstrap(binary, scores, scores, .5, .5, groups, repeats=50)
    assert same["delta_95_ci"] == dict(fpr=[0.0, 0.0], tpr=[0.0, 0.0])
    looser = comparison.threshold_bootstrap(binary, scores, scores, .5, .2, groups, repeats=50)
    assert looser["delta_95_ci"]["fpr"][1] > 0 and looser["delta_95_ci"]["tpr"][0] >= 0


def test_threshold_bootstrap_requires_both_classes_and_aligned_rows():
    with pytest.raises(ValueError):
        comparison.threshold_bootstrap([1, 1], [.2, .3], [.2, .3], .5, .5, ["a", "b"], repeats=5)
    with pytest.raises(ValueError):
        comparison.threshold_bootstrap([0, 1], [.2, .3], [.2], .5, .5, ["a", "b"], repeats=5)


def test_detection_bootstrap_identical_predictions_have_zero_intervals():
    y = np.tile([0, 1, 0, 1], 10); p = np.tile([[.8, .2], [.3, .7], [.4, .6], [.1, .9]], (10, 1))
    result = comparison.detection_bootstrap(y, p, p.copy(), [f"g{i}" for i in range(40)], ["real", "ai"], repeats=30)
    assert all(ci == [0.0, 0.0] for ci in result["delta_95_ci"].values())


def test_source_rows_report_signed_per_source_change_and_skip_empty_class():
    y = np.array([0, 0, 1]); p = np.array([[.9, .1, 0], [.2, .8, 0], [.3, .7, 0]]); q = np.array([[.9, .1, 0], [.9, .1, 0], [.3, .7, 0]])
    rows = comparison.source_rows("A0", y, p, q, ["a", "b", "c"], ["g1", "g2", "g3"])
    assert rows[0]["delta_top1"] == pytest.approx(.5) and rows[0]["group_n"] == 2
    assert rows[2]["n"] == 0 and "delta_top1" not in rows[2]


def test_worker_alive_rejects_recycled_pid(tmp_path):
    assert comparison.worker_alive(dict(pid=999999999, identity=["boot", "1"])) is False
