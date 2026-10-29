import importlib
import json

import numpy as np
import pytest


def mod():
    return importlib.import_module("scripts.compare_image_next_routes")


def recipe(**kwargs):
    return dict(task="detection", mode="clean", seed=42, **kwargs)


def row(sid, ai, source, group=None, semantic="FFHQ"):
    return dict(sample_id=sid, is_ai=ai, source_label=source, split="dev",
                group_id=group or sid, semantic_class=semantic, sha256=sid)


def pred(r, scores, role="evaluation", classes=None):
    return dict(sample_id=r["sample_id"], group_id=r["group_id"],
                is_ai=r["is_ai"], source_label=r["source_label"], semantic_class=r["semantic_class"],
                role=role, condition="clean", classes=classes or ["real","ai"], scores=scores)


def inputs():
    # seed 42 assigns g0/g3 to evaluation and g2 to calibration.
    rows=[row("r1",False,None,"g0"),row("r2",False,None,"g3"),
          row("a1",True,"a","g0"),row("a2",True,"a","g3")]
    p=[pred(r,s) for r,s in zip(rows,[[.8,.2],[.7,.3],[.1,.9],[.4,.6]])]
    return {r["sample_id"]:r for r in rows},p


def comparison(manifest, before, after, **kwargs):
    return mod().compare(manifest,before,after,"detection",repeats=20,
                         before_recipe=recipe(),after_recipe=recipe(),**kwargs)


def test_holdout_cannot_enter_comparison():
    manifest,p=inputs();manifest["r1"]["split"]="holdout"
    with pytest.raises(ValueError,match="dev|split"):
        comparison(manifest,p,p)


def test_valid_calibration_is_not_mixed_into_evaluation():
    manifest,p=inputs()
    cal=row("cal",False,None,"g2")
    manifest["cal"]=cal
    result=comparison(manifest,p+[pred(cal,[0.,1.],"calibration")],p)
    assert result["n"]==4 and result["before"]["fpr_0_5"]==0


def test_missing_reference_eval_sample_is_not_silently_dropped():
    manifest,p=inputs()
    with pytest.raises(ValueError,match="IDs"):
        comparison(manifest,p[:-1],p)


def test_candidate_truncation_is_rejected_against_frozen_recipe():
    manifest,p=inputs()
    with pytest.raises(ValueError,match="IDs"):
        comparison(manifest,p,[p[0],p[2]])


def test_calibration_ids_relabelled_evaluation_are_rejected():
    manifest,p=inputs()
    cal=row("cal",False,None,"g2")
    manifest["cal"]=cal
    with pytest.raises(ValueError,match="role|IDs"):
        comparison(manifest,p+[pred(cal,[1.,0.])],p+[pred(cal,[1.,0.])])


def test_partial_declared_calibration_role_is_rejected():
    manifest,p=inputs()
    c1,c2=row("c1",False,None,"g2"),row("c2",True,"a","g2")
    manifest.update(c1=c1,c2=c2)
    with pytest.raises(ValueError,match="IDs"):
        comparison(manifest,p+[pred(c1,[1.,0.],"calibration")],p)


def test_group_or_label_tamper_rejected():
    manifest,p=inputs();q=[dict(x) for x in p];q[0]["group_id"]="wrong"
    with pytest.raises(ValueError,match="metadata"):
        comparison(manifest,p,q)


def test_vocab_change_rejected():
    manifest,p=inputs();q=[dict(x,classes=["ai","real"]) for x in p]
    with pytest.raises(ValueError,match="vocabulary"):
        comparison(manifest,p,q)


def test_known_pair_fixed_point_and_direction():
    manifest,p=inputs();q=[dict(x) for x in p];q[0]["scores"]=[.1,.9]
    result=comparison(manifest,p,q)
    assert result["delta_after_minus_before"]["fpr_0_5"]==.5
    assert result["same_ids_labels_groups_verified"]


def test_nonfinite_probabilities_rejected():
    manifest,p=inputs();q=[dict(x) for x in p];q[0]["scores"]=[np.nan,np.nan]
    with pytest.raises(ValueError,match="probabil"):
        comparison(manifest,p,q)


def test_explicit_held_semantic_recipe_allows_complete_subset():
    manifest,p=inputs()
    extra=[row("cat_real",False,None,"g6","cat"),row("cat_ai",True,"a","g6","cat")]
    manifest.update({r["sample_id"]:r for r in extra})
    before=p+[pred(extra[0],[.9,.1]),pred(extra[1],[.1,.9])]
    result=mod().compare(manifest,before,p,"detection",repeats=20,
        before_recipe=recipe(),
        after_recipe=dict(task="detection",mode="cross_semantic",seed=42,held_semantics=["FFHQ"]))
    assert result["n"]==4
    assert result["reference_evaluation_n"]==6
    assert result["candidate_evaluation_n"]==4


def legacy(records):
    return [dict(sample_id=p["sample_id"], split="dev", status="ok", task="detection",
                 true_label="ai" if p["is_ai"] else "real", classes=p["classes"],scores=p["scores"])
            for p in records]


def test_legacy_evaluation_uses_derived_ids_without_roles_file():
    manifest,p=inputs()
    result=mod().normalize_legacy(legacy(p),manifest,recipe())
    assert {x["sample_id"] for x in result}=={"r1","r2","a1","a2"}
    assert all(x["role"]=="evaluation" and x["group_id"]==manifest[x["sample_id"]]["group_id"] for x in result)


def test_modified_legacy_roles_cannot_authorize_truncated_data():
    manifest,p=inputs()
    roles={"train":[],"selection":[],"calibration":[],"evaluation":["r1","r2","a1"]}
    with pytest.raises(ValueError,match="roles"):
        mod().normalize_legacy(legacy(p[:-1]),manifest,recipe(),legacy_roles=roles)


def test_legacy_calibration_cannot_masquerade_as_evaluation():
    manifest,p=inputs()
    c=row("cal",False,None,"g2");manifest["cal"]=c
    with pytest.raises(ValueError,match="IDs|role"):
        mod().normalize_legacy(legacy(p+[pred(c,[1.,0.])]),manifest,recipe())


def test_legacy_wrong_true_label_is_not_overwritten():
    manifest,p=inputs();records=legacy(p);records[0]["true_label"]="ai"
    with pytest.raises(ValueError,match="label"):
        mod().normalize_legacy(records,manifest,recipe())


def test_legacy_full_parent_dev_requires_explicit_scope():
    manifest,p=inputs()
    c=row("cal",False,None,"g2");manifest["cal"]=c
    records=legacy(p+[pred(c,[1.,0.])])
    result=mod().normalize_legacy(records,manifest,recipe(),source_scope="parent_dev")
    assert {x["sample_id"] for x in result}=={"r1","r2","a1","a2"}
