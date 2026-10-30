#!/usr/bin/env python3
"""One-shot frozen OmniDFA part1 linear and train-only 10-shot prototype adaptations."""
import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from research.image_attribution.omnidfa_route import (
    DIMENSION,MODEL,PREPROCESSING,ALLOWED_GLOBALS,cache_identity,encode_rows,load_backbone,
    verify_asset,select_support,support_manifest,prototype_vectors,prototype_scores,
)
from research.image_attribution.feature_ablation import fit_head
from research.image_attribution.followups import partition_rows
from research.image_attribution.openclip_route import digest,job_parts,normalized,read_cache,write_cache
from research.image_attribution.protocol import sha256,validate_manifest,write_json
from scripts.run_image_openclip import gpu_idle,probabilities,require_disk_space,role_features,score_jsonl
from scripts.run_image_followup import class_report

DEFAULT_CONFIG="configs/image_experiments/iab_omnidfa_q179_20261029.json"
LINEAR="A-OmniDFA-LP22"
PROTO="A-OmniDFA-prototype22-K10"
RECIPES={
    LINEAR:dict(task="attribution",mode="clean",class_count=22,estimator="linear"),
    PROTO:dict(task="attribution",mode="clean",class_count=22,estimator="prototype",k=10,score_temperature=1.0),
}
CODE_FILES=(
    "scripts/run_image_omnidfa.py","research/image_attribution/omnidfa_route.py",
    "research/image_attribution/feature_ablation.py","research/image_attribution/openclip_route.py",
    "research/image_attribution/followups.py","research/image_attribution/protocol.py",
    "research/image_attribution/models.py","research/image_attribution/iab_protocol.py",
    "scripts/run_image_openclip.py","scripts/run_image_followup.py","scripts/run_image_experiment.py",
)
DEPENDENCIES=("torch","torchvision","timm","numpy","Pillow","scikit-learn")


def validate_recipe(config):
    expected=dict(epochs=30,batch_size=32,learning_rate=.001,scheduler_step=1,scheduler_gamma=.9,
        optimizer="Adam",backbone_frozen=True,loss="unweighted_CrossEntropyLoss")
    if config["seed"] != 42 or config["training"] != expected or config["experiments"] != RECIPES:
        raise ValueError("only registered seed42 linear30 and fixed10-shot recipes allowed")


def registered_rows(rows):
    return sorted([r for r in rows if r["split"]!="holdout" and r["is_ai"]],key=lambda r:r["sample_id"])


def prepare(config):
    validate_recipe(config)
    if config["preprocessing"]!=PREPROCESSING:
        raise ValueError("preprocessing changed")
    data=ROOT/config["dataset"]
    for filename,key in (("manifest.jsonl","manifest_sha256"),("selection.json","selection_sha256")):
        if sha256(data/filename)!=config[key]:
            raise ValueError(f"dataset identity mismatch: {filename}")
    if set(config["implementation_sha256"])!=set(CODE_FILES) or set(config["dependencies"])!=set(DEPENDENCIES):
        raise ValueError("incomplete code/dependency binding")
    for filename,expected in config["implementation_sha256"].items():
        if sha256(ROOT/filename)!=expected:
            raise ValueError(f"implementation changed: {filename}")
    for package,expected in config["dependencies"].items():
        if importlib.metadata.version(package)!=expected:
            raise ValueError(f"dependency changed: {package}")
    verify_asset(ROOT,config["asset"])
    imported=json.loads(verify_asset(ROOT,config["asset"]["import_receipt"]).read_text())
    if (imported.get("status")!="completed" or imported.get("exit_code")!=0 or imported.get("sha256")!=config["asset"]["sha256"]
            or imported.get("bytes")!=config["asset"]["size"]):
        raise ValueError("asset import receipt incomplete or changed")
    static=json.loads(verify_asset(ROOT,config["asset"]["static_inspection"]).read_text())
    if (static.get("status")!="static_inspection_completed" or static.get("asset_sha256")!=config["asset"]["sha256"]
            or static.get("unsafe_globals")!=ALLOWED_GLOBALS
            or static.get("asset_size")!=config["asset"]["size"]):
        raise ValueError("checkpoint static inspection mismatch")
    receipt=json.loads(verify_asset(ROOT,config["asset_cpu_receipt"]).read_text())
    expected=dict(status="verified",asset_sha256=config["asset"]["sha256"],asset_size=config["asset"]["size"],
        strict_full_model_load=True,feature_dimensions=128,weights_only=True,allowed_checkpoint_globals=ALLOWED_GLOBALS,
        implementation_sha256=config["implementation_sha256"],dependencies=config["dependencies"])
    if any(receipt.get(k)!=v for k,v in expected.items()):
        raise ValueError("CPU verification receipt is absent or stale")
    rows=[json.loads(line) for line in (data/"manifest.jsonl").read_text().splitlines()]
    validate_manifest(rows,data,verify_files=False)
    parts,classes=job_parts(rows,RECIPES[LINEAR],config["seed"])
    job=dict(classes=classes,role_counts={k:len(v) for k,v in parts.items()},
        role_ids_sha256={k:digest([r["sample_id"] for r in v]) for k,v in parts.items()})
    if len(classes)!=22 or job!=config["expected_job"]:
        raise ValueError("registered class vocabulary or role bounds changed")
    support=support_manifest(parts["train"],classes,config["seed"])
    if digest(support)!=config["expected_support_sha256"]:
        raise ValueError("registered train-only support changed")
    cache_rows=registered_rows(rows)
    return data,rows,cache_rows,cache_identity(cache_rows,config),job


def new_output(path):
    Path(path).mkdir(parents=True,exist_ok=False)


def load_head(path,classes,device):
    saved=torch.load(path,map_location="cpu",weights_only=True)
    if saved["classes"]!=classes or saved["input_dimensions"]!=DIMENSION or saved["variant"]!="omnidfa":
        raise ValueError("head vocabulary or feature dimensions changed")
    head=torch.nn.Linear(DIMENSION,len(classes)).to(device)
    head.load_state_dict(saved["state_dict"],strict=True)
    return head.eval()


def verify_images(config,data,rows,cached,score_fn,device):
    if len(rows)<2:
        raise ValueError("two selection images required")
    allowed={r["sample_id"] for r in partition_rows(rows,dict(task="attribution",mode="clean",seed=config["seed"]))["selection"]}
    if any(r["sample_id"] not in allowed for r in rows[:2]):
        raise ValueError("reload probes must use selection images")
    model=load_backbone(ROOT,config,device)
    batch=encode_rows(model,data,rows[:2],device,config["seed"],batch_size=2)
    single=encode_rows(model,data,rows[:2],device,config["seed"],batch_size=1)
    checks=dict(image_cache_feature_max_error=float((batch-cached[:2]).abs().max()),
        image_single_batch_feature_max_error=float((batch-single).abs().max()),
        image_cache_score_max_error=float(np.abs(score_fn(batch)-score_fn(cached[:2])).max()),
        image_single_batch_score_max_error=float(np.abs(score_fn(batch)-score_fn(single)).max()),
        images_opened=2,full_offline_backbone_reload=True,selection_sample_ids=[r["sample_id"] for r in rows[:2]])
    if any(not np.isfinite(v) or v>1e-5 for k,v in checks.items() if k.endswith("max_error")):
        raise ValueError(f"actual-image reproducibility failed: {checks}")
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return checks


def run_job(config,name,rows,cache_rows,features,data,out,device,cache_sha):
    if cache_rows!=registered_rows(rows) or len(features)!=len(cache_rows):
        raise ValueError("feature cache rows/order/count mismatch")
    features=normalized(features,DIMENSION)
    experiment=config["experiments"][name]
    parts,classes=job_parts(rows,experiment,config["seed"])
    if len(classes)!=experiment["class_count"]:
        raise ValueError("class vocabulary size changed")
    index={r["sample_id"]:i for i,r in enumerate(cache_rows)}
    x=role_features(features,index,parts)
    provenance=dict(model=MODEL,asset=config["asset"],dependencies=config["dependencies"],
        manifest_sha256=config["manifest_sha256"],selection_sha256=config["selection_sha256"],
        code=config["implementation_sha256"],preprocessing=PREPROCESSING,feature_dimensions=DIMENSION,
        feature_cache_sha256=cache_sha,feature_cache_identity_sha256=digest(cache_identity(cache_rows,config)),
        backbone_frozen=True,classes=classes,holdout_scored=False,experiment=experiment,
        role_counts={k:len(v) for k,v in parts.items()},
        role_ids_sha256={k:digest([r["sample_id"] for r in v]) for k,v in parts.items()},
        limitations=config["limitations"],source_overlap=config.get("source_overlap",[]))
    write_json(out/"provenance.json",provenance)
    write_json(out/"roles.json",{k:[r["sample_id"] for r in v] for k,v in parts.items()})
    write_json(out/"status.json",dict(status="running",stage="fit",experiment=name))
    if experiment["estimator"]=="linear":
        linear=out/"linear";linear.mkdir()
        head_config=dict(config,task="attribution",dimensions={"omnidfa":DIMENSION})
        result=fit_head(torch.cat([x[k] for k in ("train","selection","evaluation")]),parts,classes,
            head_config,linear,device,"omnidfa",provenance)
        history=[json.loads(line) for line in (linear/"history.jsonl").read_text().splitlines()]
        if [v["epoch"] for v in history]!=list(range(1,31)):
            raise ValueError("incomplete 30epoch history")
        asset_path=linear/"head.pth"
        head=load_head(asset_path,classes,device)
        score_fn=lambda q:probabilities(head,q,device)
        verify_score=score_fn
        scores={role:score_fn(x[role]) for role in ("calibration","evaluation")}
        result.update(class_report(parts["evaluation"],classes,scores["evaluation"]))
        result.update(training_epochs=30,epoch=result["best_epoch"],verification_score_kind="linear-softmax")
    elif experiment["estimator"]=="prototype":
        selected=select_support(parts["train"],classes,config["seed"])
        support=support_manifest(parts["train"],classes,config["seed"])
        centers=prototype_vectors(features[[index[r["sample_id"]] for r in selected]],selected,classes)
        write_json(out/"support.json",support)
        asset_path=out/"prototypes.pth"
        torch.save(dict(prototypes=centers,classes=classes,support=support,provenance=provenance),asset_path)
        saved=torch.load(asset_path,map_location="cpu",weights_only=True)
        if (saved["classes"]!=classes or saved["support"]!=support or saved["provenance"]!=provenance
                or not torch.equal(saved["prototypes"],centers)):
            raise ValueError("prototype reload identity or tensor mismatch")
        score_fn=lambda q:prototype_scores(q,saved["prototypes"])[1]
        verify_score=lambda q:prototype_scores(q,saved["prototypes"])[0]
        scores,raw_files={},{}
        for role in ("calibration","evaluation"):
            raw,scores[role]=prototype_scores(x[role],saved["prototypes"])
            if not np.array_equal(raw.argmax(1),scores[role].argmax(1)):
                raise ValueError("cosine/softmax class ordering differs")
            path=out/(role+"-cosine.npy");np.save(path,raw)
            raw_files[role]=dict(path=path.name,sha256=sha256(path),
                sample_ids_sha256=digest([r["sample_id"] for r in parts[role]]),classes=classes)
        write_json(out/"cosine-score-files.json",raw_files)
        result=class_report(parts["evaluation"],classes,scores["evaluation"])
        result.update(training_epochs=None,epoch=None,k=10,score_temperature=1.0,score_conversion_dtype="float64",support_sha256=digest(support),
            raw_cosine_files=raw_files,verification_score_kind="raw-cosine",
            score_semantics="Uncalibrated temperature-1 softmax of cosine for closed-class ranking only; not an unknown probability.")
    else:
        raise ValueError("unregistered estimator")
    for role in ("calibration","evaluation"):
        score_jsonl(out/(role+".jsonl"),parts[role],classes,scores[role])
    with (out/"predictions.jsonl").open("x") as handle:
        for role in ("calibration","evaluation"):
            handle.write((out/(role+".jsonl")).read_text())
    write_json(out/"status.json",dict(status="running",stage="actual_image_reload",experiment=name))
    checks=verify_images(config,data,parts["selection"],x["selection"],verify_score,device)
    result.update(actual_image_verification=checks,head_asset_sha256=sha256(asset_path),
        failed_samples=[],holdout_scored=False,limitations=config["limitations"],classes=classes)
    write_json(out/"metrics.json",result)
    write_json(out/"provenance.json",dict(provenance,actual_image_verification=checks,
        head_asset_sha256=result["head_asset_sha256"],score_semantics=result.get("score_semantics")))
    write_json(out/"status.json",dict(status="completed",exit_code=0,experiment=name))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default=DEFAULT_CONFIG)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--device",default="cuda")
    args=parser.parse_args()
    config=json.loads((ROOT/args.config).read_text())
    if args.execute and config.get("execution_enabled") is not True:
        raise PermissionError("execution disabled")
    data,rows,cache_rows,identity,job=prepare(config)
    if not args.execute:
        print(json.dumps(dict(status="prepared",images_opened=False,cache_rows=len(cache_rows),job=job),indent=2))
        return
    out=ROOT/config["output_root"]
    with (ROOT/"outputs/image_experiments/gpu0-research.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        gpu_idle(args.device);require_disk_space(ROOT);new_output(out)
        launch=dict(pid=os.getpid(),boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            process_start_ticks=Path("/proc/self/stat").read_text().rsplit(")",1)[1].split()[19],
            command_line=Path("/proc/self/cmdline").read_bytes().decode().split(chr(0))[:-1],started_unix=time.time())
        write_json(out/"launch.json",launch)
        write_json(out/"registration.json",dict(config=config,config_sha256=digest(config),job=job))
        def state(**kwargs):
            write_json(out/"status.json",dict(launch=launch,updated_unix=time.time(),**kwargs))
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        try:
            state(status="running",stage="cache",completed_images=0,total_images=len(cache_rows))
            model=load_backbone(ROOT,config,args.device)
            features=encode_rows(model,data,cache_rows,args.device,config["seed"],config["training"]["batch_size"],
                progress=lambda n,total:state(status="running",stage="cache",completed_images=n,total_images=total))
            write_cache(out/"features.pt",features,identity);cache_sha=sha256(out/"features.pt")
            features=read_cache(out/"features.pt",identity,DIMENSION)
            write_json(out/"cache.json",dict(identity=identity,sha256=cache_sha,n=len(features),dimensions=DIMENSION,holdout_read=False))
            del model
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()
            for name in (LINEAR,PROTO):
                job_out=out/name;job_out.mkdir()
                try:
                    run_job(config,name,rows,cache_rows,features,data,job_out,args.device,cache_sha)
                except Exception as error:
                    write_json(job_out/"status.json",dict(status="failed",exit_code=1,error=repr(error)))
                    raise
            state(status="completed",exit_code=0,experiments=[LINEAR,PROTO])
        except Exception as error:
            state(status="failed",exit_code=1,error=repr(error))
            raise

if __name__=="__main__":
    main()
