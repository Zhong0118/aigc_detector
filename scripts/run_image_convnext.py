#!/usr/bin/env python3
"""Frozen official ConvNeXt feature baseline; independent, one-shot q179 run."""
import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from research.image_attribution.feature_ablation import fit_head
from research.image_attribution.openclip_route import (
    digest, image_tensor, job_parts, normalized, write_cache,
)
from research.image_attribution.protocol import sha256, validate_manifest, write_json
from scripts.run_image_openclip import (
    gpu_idle, load_head, probabilities, require_disk_space, role_features, score_jsonl,
)

MODEL = "convnext_tiny.fb_in22k_ft_in1k"
DIMENSION = 768
PREPROCESSING = dict(input_size=[3, 288, 288], interpolation="bicubic", crop_pct=1.0,
    crop_mode="center", mean=[.485,.456,.406], std=[.229,.224,.225], rgb=True,
    feature_normalization="l2", test_size=True)
DEFAULT_CONFIG = "configs/image_experiments/iab_convnext_q179_20261029.json"
RECIPES = {
    "D-ConvNeXt": dict(task="detection", mode="clean", class_count=2),
    "D-ConvNeXt-FFHQ": dict(task="detection", mode="cross_semantic", held_semantics=["FFHQ"], class_count=2),
    "A-ConvNeXt22": dict(task="attribution", mode="clean", class_count=22),
}


def validate_recipe(config):
    expected = dict(epochs=30,batch_size=32,learning_rate=.001,scheduler_step=1,
        scheduler_gamma=.9,optimizer="Adam",backbone_frozen=True,loss="unweighted_CrossEntropyLoss")
    if config["seed"] != 42 or config["training"] != expected:
        raise ValueError("only registered seed42/30epoch frozen-head recipe allowed")
    if not config["experiments"] or any(RECIPES.get(k) != v for k,v in config["experiments"].items()):
        raise ValueError("unregistered ConvNeXt experiment recipe")


def cache_identity(rows, config):
    if any(r["split"] not in ("train","dev") for r in rows):
        raise ValueError("holdout forbidden in feature cache")
    ids=[r["sample_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate cache IDs")
    return dict(schema="convnext-frozen-cache-v1", model=MODEL, preprocessing=PREPROCESSING,
        dimensions=DIMENSION, ids=ids, rows_sha256=digest([{k:r[k] for k in (
            "sample_id","group_id","relative_path","sha256","split","is_ai","source_label","semantic_class")}
            for r in rows]),asset=config["asset"],code=config["implementation_sha256"])


def verify_asset(root, asset):
    path=Path(root)/asset["path"]
    if path.stat().st_size != asset["size"] or sha256(path) != asset["sha256"]:
        raise ValueError("official asset incomplete or identity changed")
    return path


def load_backbone(root, config, device):
    import timm
    from safetensors.torch import load_file
    path=verify_asset(root,config["asset"])
    source=json.loads(verify_asset(root,config["model_config"]).read_text())
    pcfg=source["pretrained_cfg"]
    if (source["architecture"]!="convnext_tiny" or source["num_features"]!=DIMENSION
            or pcfg["test_input_size"]!=PREPROCESSING["input_size"]
            or pcfg["test_crop_pct"]!=1.0 or pcfg["mean"]!=PREPROCESSING["mean"]
            or pcfg["std"]!=PREPROCESSING["std"] or pcfg["interpolation"]!="bicubic"):
        raise ValueError("official transform/config differs from registered test preprocessing")
    model=timm.create_model(MODEL,pretrained=False)
    model.load_state_dict(load_file(str(path),device="cpu"),strict=True)
    if model.num_features != DIMENSION or model.get_classifier().out_features != 1000:
        raise ValueError("strict full classifier verification failed")
    model.reset_classifier(0)
    model.eval().requires_grad_(False).to(device)
    transform=timm.data.create_transform(input_size=tuple(PREPROCESSING["input_size"]),
        interpolation="bicubic",mean=PREPROCESSING["mean"],std=PREPROCESSING["std"],
        crop_pct=1.0,crop_mode="center",is_training=False)
    return model,transform


def encode_rows(model, transform, data, rows, device, batch_size=32, progress=None):
    batches=[]
    with torch.inference_mode():
        for start in range(0,len(rows),batch_size):
            batch=torch.stack([image_tensor(data,r,transform) for r in rows[start:start+batch_size]])
            batches.append(normalized(model(batch.to(device)),DIMENSION).cpu())
            if progress and (start % (batch_size*32)==0 or start+batch_size>=len(rows)):
                progress(min(start+batch_size,len(rows)),len(rows))
    return torch.cat(batches)


def verify_images(config,data,rows,cached,score_fn,device):
    if len(rows)<2: raise ValueError("two selection images required")
    model,transform=load_backbone(ROOT,config,device)
    batch=encode_rows(model,transform,data,rows[:2],device,batch_size=2)
    single=encode_rows(model,transform,data,rows[:2],device,batch_size=1)
    checks=dict(image_cache_feature_max_error=float((batch-cached[:2]).abs().max()),
        image_single_batch_feature_max_error=float((batch-single).abs().max()),
        image_cache_score_max_error=float(np.abs(score_fn(batch)-score_fn(cached[:2])).max()),
        image_single_batch_score_max_error=float(np.abs(score_fn(batch)-score_fn(single)).max()),
        images_opened=2,full_offline_backbone_reload=True,
        selection_sample_ids=[r["sample_id"] for r in rows[:2]])
    if any(v>1e-5 for k,v in checks.items() if k.endswith("max_error")):
        raise ValueError(f"actual-image reproducibility failed: {checks}")
    del model
    if str(device).startswith("cuda"):torch.cuda.empty_cache()
    return checks


def prepare(config):
    validate_recipe(config)
    if config["preprocessing"]!=PREPROCESSING:raise ValueError("preprocessing changed")
    data=ROOT/config["dataset"]
    for name,key in (("manifest.jsonl","manifest_sha256"),("selection.json","selection_sha256")):
        if sha256(data/name)!=config[key]:raise ValueError(f"dataset changed: {name}")
    for path,expected in config["implementation_sha256"].items():
        if sha256(ROOT/path)!=expected:raise ValueError(f"implementation changed: {path}")
    for package,expected in config["dependencies"].items():
        if importlib.metadata.version(package)!=expected:raise ValueError(f"dependency changed: {package}")
    verify_asset(ROOT,config["asset"]);verify_asset(ROOT,config["model_config"])
    rows=[json.loads(line) for line in (data/"manifest.jsonl").read_text().splitlines()]
    validate_manifest(rows,data,verify_files=False)
    cache_rows=sorted([r for r in rows if r["split"]!="holdout"],key=lambda r:r["sample_id"])
    jobs={}
    for name,experiment in config["experiments"].items():
        parts,classes=job_parts(rows,experiment,config["seed"])
        if len(classes)!=experiment["class_count"]:raise ValueError("class count mismatch")
        jobs[name]=dict(classes=classes,role_counts={k:len(v) for k,v in parts.items()},
            role_ids_sha256={k:digest([r["sample_id"] for r in v]) for k,v in parts.items()})
    return data,rows,cache_rows,cache_identity(cache_rows,config),jobs


def new_output(path):
    Path(path).mkdir(parents=True,exist_ok=False)


def run_job(config,name,rows,cache_rows,features,data,out,device,cache_sha):
    experiment=config["experiments"][name]
    parts,classes=job_parts(rows,experiment,config["seed"])
    index={r["sample_id"]:i for i,r in enumerate(cache_rows)}
    x=role_features(features,index,parts)
    provenance=dict(model=MODEL,asset=config["asset"],model_config=config["model_config"],
        dependencies=config["dependencies"],manifest_sha256=config["manifest_sha256"],
        selection_sha256=config["selection_sha256"],code=config["implementation_sha256"],
        preprocessing=PREPROCESSING,feature_dimensions=DIMENSION,backbone_frozen=True,
        feature_cache_sha256=cache_sha,experiment=experiment,classes=classes,holdout_scored=False,
        role_counts={k:len(v) for k,v in parts.items()},
        role_ids_sha256={k:digest([r["sample_id"] for r in v]) for k,v in parts.items()},
        limitations=config["limitations"])
    write_json(out/"provenance.json",provenance)
    write_json(out/"roles.json",{k:[r["sample_id"] for r in v] for k,v in parts.items()})
    write_json(out/"status.json",dict(status="running",stage="head_training"))
    linear=out/"linear";linear.mkdir()
    roles=("train","selection","calibration","evaluation") if experiment["task"]=="detection" else ("train","selection","evaluation")
    head_config=dict(config,task=experiment["task"],dimensions={"convnext":DIMENSION})
    result=fit_head(torch.cat([x[k] for k in roles]),parts,classes,head_config,linear,device,"convnext",provenance)
    history=[json.loads(line) for line in (linear/"history.jsonl").read_text().splitlines()]
    if [r["epoch"] for r in history]!=list(range(1,31)):raise ValueError("incomplete 30epoch history")
    head=load_head(linear/"head.pth",classes,device)
    score_fn=lambda batch:probabilities(head,batch,device)
    for role in ("calibration","evaluation"):
        score_jsonl(out/(role+".jsonl"),parts[role],classes,score_fn(x[role]))
    with (out/"predictions.jsonl").open("x") as f:
        for role in ("calibration","evaluation"):f.write((out/(role+".jsonl")).read_text())
    write_json(out/"status.json",dict(status="running",stage="actual_image_reload"))
    checks=verify_images(config,data,parts["selection"],x["selection"],score_fn,device)
    result.update(actual_image_verification=checks,head_asset_sha256=sha256(linear/"head.pth"),
        training_epochs=30,epoch=result["best_epoch"],classes=classes,failed_samples=[],limitations=config["limitations"])
    write_json(out/"metrics.json",result)
    write_json(out/"provenance.json",dict(provenance,actual_image_verification=checks,head_asset_sha256=result["head_asset_sha256"]))
    write_json(out/"status.json",dict(status="completed",exit_code=0,experiment=name))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default=DEFAULT_CONFIG)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--device",default="cuda")
    args=parser.parse_args()
    config=json.loads((ROOT/args.config).read_text())
    data,rows,cache_rows,identity,jobs=prepare(config)
    if not args.execute:
        print(json.dumps(dict(status="prepared",images_opened=False,cache_rows=len(cache_rows),jobs=jobs),indent=2));return
    if config.get("execution_enabled") is not True:raise PermissionError("execution disabled")
    out=ROOT/config["output_root"]
    with (ROOT/"outputs/image_experiments/gpu0-research.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        gpu_idle(args.device);require_disk_space(ROOT);new_output(out)
        launch=dict(pid=os.getpid(),boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            process_start_ticks=Path("/proc/self/stat").read_text().rsplit(")",1)[1].split()[19],
            command_line=Path("/proc/self/cmdline").read_bytes().decode().split(chr(0))[:-1],started_unix=time.time())
        write_json(out/"launch.json",launch)
        write_json(out/"registration.json",dict(config=config,config_sha256=digest(config),jobs=jobs))
        def state(**kwargs):write_json(out/"status.json",dict(launch=launch,updated_unix=time.time(),**kwargs))
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        try:
            state(status="running",stage="cache",completed_images=0,total_images=len(cache_rows))
            model,transform=load_backbone(ROOT,config,args.device)
            features=encode_rows(model,transform,data,cache_rows,args.device,config["training"]["batch_size"],
                progress=lambda n,total:state(status="running",stage="cache",completed_images=n,total_images=total))
            write_cache(out/"features.pt",features,identity);cache_sha=sha256(out/"features.pt")
            write_json(out/"cache.json",dict(identity=identity,sha256=cache_sha,n=len(features),dimensions=DIMENSION,holdout_read=False))
            del model
            if str(args.device).startswith("cuda"):torch.cuda.empty_cache()
            for name in jobs:
                state(status="running",stage="head_training",experiment=name)
                job=out/name;job.mkdir()
                try:run_job(config,name,rows,cache_rows,features,data,job,args.device,cache_sha)
                except Exception as error:
                    write_json(job/"status.json",dict(status="failed",exit_code=1,error=repr(error)));raise
            state(status="completed",exit_code=0,experiments=list(jobs))
        except Exception as error:
            state(status="failed",exit_code=1,error=repr(error));raise

if __name__=="__main__":main()
