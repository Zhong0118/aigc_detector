"""Read-only same-role saved-score comparison for new visual routes.

Never reads images, trains, selects thresholds, or scores holdout. Call compare
with role-tagged records, or use --config with explicit completed source runs.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_image_scale_epochs import indexed, stats, group_bootstrap, sha256
from scripts.compare_iab_v6_100_vs_v7_200_q179 import detection_bootstrap, source_rows
from research.image_attribution.followups import partition_rows, validate_probabilities

MANIFEST_SHA = "9dd85636e2a19dbe833d9ebe2bee0c2a6b6eaf43015699e4fd9f27db3694b4c3"

def derived_parts(manifest, recipe):
    if recipe.get("seed") != 42 or recipe.get("task") not in ("detection", "attribution"):
        raise ValueError("comparison requires the frozen task and seed 42 recipe")
    return partition_rows(list(manifest.values()), recipe)


def validate_saved_roles(manifest, recipe, roles):
    expected = {role: sorted(r["sample_id"] for r in rows)
                for role, rows in derived_parts(manifest, recipe).items()}
    if set(roles) != set(expected) or any(sorted(roles[role]) != ids for role, ids in expected.items()):
        raise ValueError("saved roles differ from full recipe-derived roles")


def normalize_legacy(records, manifest, recipe, legacy_roles=None, source_scope="evaluation"):
    """Attach manifest identities only after validating the complete legacy input."""
    parts = derived_parts(manifest, recipe)
    if legacy_roles is not None:
        validate_saved_roles(manifest, recipe, legacy_roles)
    evaluation = {row["sample_id"] for row in parts["evaluation"]}
    if source_scope == "evaluation":
        expected = evaluation
    elif source_scope == "parent_dev":
        expected = {sid for sid,row in manifest.items() if row["split"] == "dev"
                    and (recipe["task"] == "detection" or row["is_ai"])}
    else:
        raise ValueError("unsupported legacy source scope")
    by_id = indexed(records)
    if set(by_id) != expected:
        raise ValueError("legacy input IDs differ from complete declared scope")
    converted = []
    for sid, record in by_id.items():
        row = manifest[sid]
        if (record.get("status") != "ok" or record.get("split") != "dev"
                or record.get("task") != recipe["task"]):
            raise ValueError("invalid legacy prediction task/status/dev split")
        label = ("ai" if row["is_ai"] else "real") if recipe["task"] == "detection" else row["source_label"]
        if record.get("true_label") != label:
            raise ValueError("legacy label mismatch")
        for field in ("group_id", "is_ai", "source_label", "semantic_class"):
            if field in record and record[field] != row[field]:
                raise ValueError("legacy metadata differs from frozen manifest")
        if sid in evaluation:
            converted.append(dict(record, role="evaluation", condition="clean",
                **{field:row[field] for field in ("group_id","is_ai","source_label","semantic_class")}))
    return converted


def selected(records, manifest, recipe):
    parts = derived_parts(manifest, recipe)
    expected = {role:{r["sample_id"] for r in rows} for role,rows in parts.items()}
    selected_by_role = {"evaluation": []}
    for record in records:
        if record.get("condition", "clean") != "clean":
            continue
        role, sid = record.get("role"), record.get("sample_id")
        if role not in ("calibration", "evaluation"):
            raise ValueError("prediction has invalid declared role")
        if sid not in manifest or manifest[sid]["split"] != "dev":
            raise ValueError("only frozen dev IDs may enter comparison")
        if sid not in expected[role]:
            raise ValueError("prediction role/IDs differ from frozen recipe")
        for field in ("group_id", "is_ai", "source_label", "semantic_class"):
            if record.get(field) != manifest[sid][field]:
                raise ValueError("prediction metadata differs from frozen manifest")
        selected_by_role.setdefault(role, []).append(record)
    indexed_roles = {}
    for role, values in selected_by_role.items():
        indexed_roles[role] = indexed(values)
        if set(indexed_roles[role]) != expected[role]:
            raise ValueError(f"missing or unexpected {role} IDs; require full frozen role")
    return indexed_roles["evaluation"]

def compare(manifest, before, after, task, repeats=1000, *, before_recipe, after_recipe):
    if task not in ("detection", "attribution"):
        raise ValueError("unsupported comparison task")
    if before_recipe.get("task") != task or after_recipe.get("task") != task:
        raise ValueError("recipe task differs from comparison")
    a, b = selected(before, manifest, before_recipe), selected(after, manifest, after_recipe)
    if not b or not set(b) <= set(a):
        raise ValueError("missing evaluation IDs; no intersection fallback")
    if set(a) != set(b) and after_recipe.get("mode") not in ("cross_semantic", "cross_source"):
        raise ValueError("evaluation subset requires an explicit held-domain recipe")
    ids = sorted(b)
    classes = b[ids[0]]["classes"]
    if (not isinstance(classes, list) or len(classes) < 2 or len(set(classes)) != len(classes)
            or (task == "detection" and classes != ["real", "ai"])):
        raise ValueError("invalid vocabulary")
    p,q,y,groups = [],[],[],[]
    for sid in ids:
        row = manifest[sid]
        if row["split"] != "dev":
            raise ValueError("only frozen dev IDs may enter comparison")
        label = ("ai" if row["is_ai"] else "real") if task == "detection" else row["source_label"]
        if label not in classes:
            raise ValueError("unknown label in closed-set comparison")
        for rec in (a[sid],b[sid]):
            if rec["classes"] != classes:
                raise ValueError("comparison vocabulary differs")
            for field in ("group_id", "is_ai", "source_label", "semantic_class"):
                if rec.get(field) != row[field]:
                    raise ValueError("prediction metadata differs from frozen manifest")
        p.append(a[sid]["scores"]);q.append(b[sid]["scores"])
        y.append(classes.index(label));groups.append(row["group_id"])
    p=validate_probabilities(p,len(ids),len(classes));q=validate_probabilities(q,len(ids),len(classes))
    y=np.asarray(y)
    metric_task="D0" if task=="detection" else "A0"
    first,second=stats(y,p,metric_task,classes),stats(y,q,metric_task,classes)
    boot=(detection_bootstrap(y,p,q,groups,classes,repeats=repeats) if task=="detection"
          else group_bootstrap(y,p,q,groups,repeats=repeats))
    return dict(n=len(ids),same_ids_labels_groups_verified=True,classes=classes,before=first,after=second,
                reference_evaluation_n=len(a),candidate_evaluation_n=len(b),
                complete_recipe_roles_verified=True,before_recipe=before_recipe,after_recipe=after_recipe,
                delta_after_minus_before={k:second[k]-first[k] for k in second if k!="n"},
                group_bootstrap=boot,
                per_source=[] if task=="detection" else source_rows("new minus reference",y,p,q,classes,groups),
                holdout_scored=False,
                note="Exploratory existing developer roles, conditional on selected models; not independent test or causal estimate.")

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve existing comparison")
    cfg=json.loads(args.config.read_text())
    inputs={str(args.config):sha256(args.config),str(Path(__file__).relative_to(ROOT)):sha256(__file__)}
    mp=ROOT/cfg["manifest"]
    if sha256(mp)!=MANIFEST_SHA or cfg["manifest_sha256"]!=MANIFEST_SHA:
        raise ValueError("frozen manifest mismatch")
    inputs[str(mp.relative_to(ROOT))]=MANIFEST_SHA
    manifest=indexed([json.loads(line) for line in mp.read_text().splitlines()])
    results={}
    def read_run(spec):
        folder=ROOT/spec["run"]
        sp=folder/"status.json"
        status=json.loads(sp.read_text())
        if status.get("status")!="completed" or status.get("exit_code")!=0:
            raise ValueError("input run not completed")
        pp=folder/spec.get("predictions","predictions.jsonl")
        for path in (sp,pp):
            inputs[str(path.relative_to(ROOT))]=sha256(path)
        records=[json.loads(line) for line in pp.read_text().splitlines()]
        recipe = spec["recipe"]
        roles = None
        if spec.get("roles"):
            rp=ROOT/spec["roles"];roles=json.loads(rp.read_text())
            inputs[str(rp.relative_to(ROOT))]=sha256(rp)
            validate_saved_roles(manifest,recipe,roles)
        if records and all("role" not in record for record in records):
            records = normalize_legacy(records,manifest,recipe,legacy_roles=roles,
                                       source_scope=spec.get("legacy_scope","evaluation"))
        return records
    for pair in cfg["pairs"]:
        if pair["id"] in results:
            raise ValueError("duplicate comparison ID")
        results[pair["id"]]=compare(manifest,read_run(pair["before"]),read_run(pair["after"]),pair["task"],
            before_recipe=pair["before"]["recipe"],after_recipe=pair["after"]["recipe"])
    # Inputs must remain unchanged across analysis.
    for path,digest in inputs.items():
        if sha256(ROOT/path)!=digest:
            raise ValueError("input changed during comparison")
    args.output.mkdir(parents=True,exist_ok=False)
    for name,value in (("summary.json",results),("provenance.json",dict(inputs=inputs,holdout_scored=False,images_read=False))):
        (args.output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    (args.output/"completion.json").write_text(json.dumps(dict(status="completed",exit_code=0,
        summary_sha256=sha256(args.output/"summary.json"),provenance_sha256=sha256(args.output/"provenance.json"),
        holdout_scored=False),indent=2)+"\n")
if __name__=="__main__":
    main()
