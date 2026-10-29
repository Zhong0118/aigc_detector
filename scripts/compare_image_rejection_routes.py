"""Read-only paired open-set comparison; thresholds use kling calibration only."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.image_attribution.followups import partition_rows,choose_threshold,validate_probabilities,wilson
from scripts.compare_image_scale_epochs import indexed,sha256
MANIFEST="outputs/image_experiments/iab-balanced-v7-200-q179/manifest.jsonl"
MANIFEST_SHA="9dd85636e2a19dbe833d9ebe2bee0c2a6b6eaf43015699e4fd9f27db3694b4c3"
RECIPE=dict(task="attribution",mode="open_set",seed=42,unknown_calibration=["kling"],unknown_evaluation=["grok3"])
FIELDS=("known_retention","known_correct_and_retained","unknown_false_acceptance","unknown_rejection","unknown_auroc")

def report(cal,ev,classes,cp,ep,pred):
    unknown=np.array([r["source_label"] not in classes for r in cal])
    if not unknown.any() or {r["source_label"] for r in cal if r["source_label"] not in classes}!={"kling"}:
        raise ValueError("unknown calibration must contain kling only")
    if {r["source_label"] for r in ev if r["source_label"] not in classes}!={"grok3"}:
        raise ValueError("unknown evaluation must contain grok3 only")
    cp=np.asarray(cp,float);ep=np.asarray(ep,float);pred=np.asarray(pred,int)
    if cp.shape!=(len(cal),) or ep.shape!=(len(ev),) or pred.shape!=(len(ev),) or not np.isfinite(cp).all() or not np.isfinite(ep).all():
        raise ValueError("invalid confidence/prediction shape")
    if (pred<0).any() or (pred>=len(classes)).any():
        raise ValueError("predicted class index invalid")
    known=np.array([r["source_label"] in classes for r in ev])
    if not known.any():
        raise ValueError("known evaluation samples required")
    threshold=choose_threshold(cp[unknown],.05)
    accept=ep>=threshold
    correct=np.array([classes[i]==r["source_label"] for i,r in zip(pred,ev)])
    result=dict(n=len(ev),known_n=int(known.sum()),unknown_n=int((~known).sum()),
        threshold=threshold,calibration_unknown_n=int(unknown.sum()),
        calibration_unknown_false_acceptance=float((cp[unknown]>=threshold).mean()),
        unknown_auroc=float(roc_auc_score(~known,-ep)),
        known_closed_set_accuracy=float(correct[known].mean()),threshold_role="calibration",
        score_note="Acceptance evidence, not calibrated probability of unknown origin")
    for name,mask,value in [("known_retention",known,accept),("known_correct_and_retained",known,accept&correct),
                            ("unknown_false_acceptance",~known,accept),("unknown_rejection",~known,~accept)]:
        n=int(mask.sum());hit=int(value[mask].sum())
        result[name]=hit/n;result[name+"_count"]=hit;result[name+"_wilson95"]=wilson(hit,n)
    return result

def bootstrap(cal,ev,classes,methods,repeats=1000):
    known=np.array([r["source_label"] in classes for r in ev])
    unknowncal=np.array([r["source_label"] not in classes for r in cal])
    def group_indices(rows):
        names=sorted({r["group_id"] for r in rows})
        return names,np.array([names.index(r["group_id"]) for r in rows])
    eg,ei=group_indices(ev)
    cu=[r for r in cal if r["source_label"] not in classes]
    cg,ci=group_indices(cu)
    rng=np.random.default_rng(42);samples={name:[] for name in methods};thresholds={name:[] for name in methods}
    skipped=0
    for _ in range(repeats):
        ew=np.bincount(rng.integers(0,len(eg),len(eg)),minlength=len(eg))[ei]
        if ew[known].sum()==0 or ew[~known].sum()==0:
            skipped+=1;continue
        cw=np.bincount(rng.integers(0,len(cg),len(cg)),minlength=len(cg))[ci]
        for name,(cp,ep,pred) in methods.items():
            t=choose_threshold(np.repeat(cp[unknowncal],cw),.05);thresholds[name].append(t)
            accepted=ep>=t
            correct=np.array([classes[i]==r["source_label"] for i,r in zip(pred,ev)])
            samples[name].append([
                ew[known&accepted].sum()/ew[known].sum(),
                ew[known&accepted&correct].sum()/ew[known].sum(),
                ew[~known&accepted].sum()/ew[~known].sum(),
                ew[~known&~accepted].sum()/ew[~known].sum(),
                roc_auc_score(~known,-ep,sample_weight=ew)])
    if repeats-skipped<.95*repeats:
        raise ValueError("insufficient valid bootstrap replicates")
    base="resnet_msp";out={}
    for name,values in samples.items():
        if name==base:continue
        delta=np.asarray(values)-np.asarray(samples[base])
        out[name+"_minus_"+base]={k:v for k,v in zip(FIELDS,np.quantile(delta,[.025,.975],axis=0).T.tolist())}
    delta=np.asarray(samples["openclip_prototype"])-np.asarray(samples["openclip_msp"])
    out["openclip_prototype_minus_openclip_msp"]={k:v for k,v in zip(FIELDS,np.quantile(delta,[.025,.975],axis=0).T.tolist())}
    return dict(repeats=repeats,valid_repeats=repeats-skipped,seed=42,evaluation_groups=len(eg),unknown_calibration_groups=len(cg),
        method="Paired original-group percentile bootstrap; refit each method only on resampled kling calibration; shared calibration/evaluation draws",
        delta_95_ci=out,threshold_95_ci={k:np.quantile(v,[.025,.975]).tolist() for k,v in thresholds.items()},
        limitations="Conditional on frozen representation/checkpoint; source variation limited to kling and grok3")

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError("preserve existing report")
    inputs={}
    def read(path,lines=False):
        path=ROOT/path;inputs[str(path.relative_to(ROOT))]=sha256(path)
        return [json.loads(s) for s in path.read_text().splitlines()] if lines else json.loads(path.read_text())
    rows=read(MANIFEST,True)
    if inputs[MANIFEST]!=MANIFEST_SHA:raise ValueError("manifest changed")
    manifest=indexed(rows);parts=partition_rows(rows,RECIPE)
    classes=sorted({r["source_label"] for r in parts["train"]})
    if len(classes)!=20:raise ValueError("20 known classes required")
    cal,ev=parts["calibration"],parts["evaluation"]
    routes={}
    for name,folder in [("resnet","outputs/image_experiments/followups-v7-200-q179-e30/A1/evaluate"),
                        ("openclip","outputs/image_experiments/openclip-q179-20261029/A-OpenCLIP-open20")]:
        folder=Path(folder);status=read(folder/"status.json")
        if status.get("status")!="completed" or status.get("exit_code")!=0:raise ValueError("route not completed")
        roles=read(folder/"roles.json")
        if roles!={k:[r["sample_id"] for r in v] for k,v in parts.items()}:raise ValueError("saved roles differ from frozen recipe")
        metric=read(folder/"metrics.json");rec=read(folder/"predictions.jsonl",True)
        route={}
        for role in ("calibration","evaluation"):
            selected=indexed([r for r in rec if r.get("role")==role and r.get("condition","clean")=="clean"])
            if set(selected)!={r["sample_id"] for r in parts[role]}:raise ValueError("missing or extra role predictions")
            scores=[];proto=[]
            for row in parts[role]:
                r=selected[row["sample_id"]]
                if row["split"]!="dev" or r["classes"]!=classes:raise ValueError("nondev or changed vocabulary")
                if any(r[k]!=row[k] for k in ("group_id","source_label","is_ai","semantic_class")):raise ValueError("metadata mismatch")
                scores.append(r["scores"])
                if name=="openclip":proto.append(r["prototype_cosine"])
            scores=validate_probabilities(scores,len(parts[role]),len(classes))
            route[role]=dict(msp=scores.max(1),pred=scores.argmax(1),prototype=np.asarray(proto))
        route["metrics"]=metric;routes[name]=route
    methods={name+"_msp":(route["calibration"]["msp"],route["evaluation"]["msp"],route["evaluation"]["pred"]) for name,route in routes.items()}
    methods["openclip_prototype"]=(routes["openclip"]["calibration"]["prototype"],routes["openclip"]["evaluation"]["prototype"],routes["openclip"]["evaluation"]["pred"])
    reports={name:report(cal,ev,classes,*value) for name,value in methods.items()}
    # Recompute published metrics rather than trusting aggregate JSON alone.
    old=routes["resnet"]["metrics"]["conditions"]["clean"]
    for k,alias in [("threshold","acceptance_threshold"),("known_retention","known_acceptance"),
                    ("known_correct_and_retained","known_correct_and_accepted_rate"),("unknown_false_acceptance","unknown_false_acceptance"),("unknown_auroc","unknown_auroc")]:
        if not np.isclose(reports["resnet_msp"][k],old[alias],rtol=0,atol=1e-10):raise ValueError("saved ResNet metric differs")
    for name,method in [("openclip_msp","msp"),("openclip_prototype","prototype_cosine")]:
        saved=routes["openclip"]["metrics"]["rejection"][method]["evaluation"]
        for k in FIELDS:
            if not np.isclose(reports[name][k],saved[k],rtol=0,atol=1e-10):raise ValueError("saved OpenCLIP metric differs")
    intervals=bootstrap(cal,ev,classes,methods)
    inputs[str(Path(__file__).relative_to(ROOT))]=sha256(__file__)
    for path,digest in inputs.items():
        if sha256(ROOT/path)!=digest:raise ValueError("input changed during comparison")
    args.output.mkdir(parents=True,exist_ok=False)
    for filename,value in [("summary.json",dict(methods=reports,paired_uncertainty=intervals,holdout_scored=False,
            caveat="Exploratory developer comparison, single seed, limited unknown source pair; no deployment calibration guarantee")),
                           ("provenance.json",dict(inputs=inputs,recipe=RECIPE,manifest_sha256=MANIFEST_SHA,images_read=False))]:
        (args.output/filename).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    (args.output/"completion.json").write_text(json.dumps(dict(status="completed",exit_code=0,
        summary_sha256=sha256(args.output/"summary.json"),provenance_sha256=sha256(args.output/"provenance.json"),
        holdout_scored=False),indent=2)+"\n")
if __name__=="__main__":main()
