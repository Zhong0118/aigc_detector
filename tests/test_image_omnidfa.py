"""CPU safety, deterministic views and train-only few-shot protocol tests."""
import argparse
import importlib.util
import json
import random

import numpy as np
from PIL import Image
import pytest
import torch

from research.image_attribution.openclip_route import normalized, job_parts, read_cache, write_cache
from research.image_attribution.protocol import sha256


def modules():
    assert importlib.util.find_spec("research.image_attribution.omnidfa_route") is not None, "OmniDFA route not implemented"
    from research.image_attribution import omnidfa_route
    from scripts import run_image_omnidfa
    return omnidfa_route, run_image_omnidfa


def row(i, source="a", split="train", group=None):
    return dict(sample_id=str(i),group_id=str(i) if group is None else group,sha256=str(i),
        pixel_sha256=str(i),relative_path=f"{i}.png",source_label=source,split=split,
        is_ai=1,semantic_class="test")


def cfg(asset=None):
    m,r=modules()
    return dict(seed=42,asset=asset or {},preprocessing=m.PREPROCESSING,dependencies={},
        implementation_sha256={},manifest_sha256="manifest",selection_sha256="selection",
        experiments={k:dict(v) for k,v in r.RECIPES.items()},
        training=dict(epochs=30,batch_size=32,learning_rate=.001,scheduler_step=1,
            scheduler_gamma=.9,optimizer="Adam",backbone_frozen=True,loss="unweighted_CrossEntropyLoss"),
        limitations=["CPU fixture"])


class TinyBackbone(torch.nn.Module):
    num_features=768
    def __init__(self):
        super().__init__()
        self.layers=torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten(),torch.nn.Linear(3,768))
    def forward(self,x):
        return self.layers(x)


class UnreviewedMetadata:
    pass


@pytest.fixture
def tiny_asset(tmp_path,monkeypatch):
    m,_=modules()
    torch.set_num_threads(1)
    # Replace only the expensive timm constructor; strict loader, MLP and image path stay real.
    def build(name,pretrained,num_classes):
        assert name=="convnext_small" and pretrained is False and num_classes==0
        return TinyBackbone()
    monkeypatch.setattr(m,"create_model",build)
    torch.manual_seed(42)
    model=m.TwinFingerprint()
    path=tmp_path/"model.pth"
    torch.save(dict(model=model.state_dict(),args=argparse.Namespace(epoch=20)),path)
    return dict(path=path.name,size=path.stat().st_size,sha256=sha256(path))


def test_seeded_views_are_repeatable_restore_rng_and_upscale_small_images():
    m,_=modules()
    image=Image.fromarray(np.random.default_rng(3).integers(0,256,(90,170,3),dtype=np.uint8))
    py_state=random.getstate();torch_state=torch.random.get_rng_state().clone()
    a=m.preprocess(image,"sample-A",42)
    assert random.getstate()==py_state and torch.equal(torch.random.get_rng_state(),torch_state)
    random.random();torch.rand(7)
    b=m.preprocess(image,"sample-A",42)
    assert all(torch.equal(x,y) for x,y in zip(a,b))
    assert [tuple(x.shape) for x in a]==[(3,224,224),(3,224,224)]
    assert all(float(x.min())>=0 and float(x.max())<=1 for x in a)
    c=m.preprocess(image,"sample-B",42)
    assert not torch.equal(a[0],c[0])


def test_seeded_views_match_fixed_local_and_global_geometry():
    m,_=modules()
    from torchvision import transforms
    image=Image.fromarray(np.random.default_rng(9).integers(0,256,(300,450,3),dtype=np.uint8))
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(m.sample_seed("geometry",42))
        tensor=transforms.ToTensor()(image)
        local=transforms.RandomCrop(224)(tensor)
        global_view=transforms.RandomCrop(224)(transforms.Resize(224)(tensor))
    actual=m.preprocess(image,"geometry",42)
    assert torch.equal(actual[0],local) and torch.equal(actual[1],global_view)


def test_safe_checkpoint_strict_load_and_allowlist_scope(tmp_path,tiny_asset):
    m,_=modules()
    before=torch.serialization.get_safe_globals()
    model=m.load_backbone(tmp_path,cfg(tiny_asset),"cpu")
    assert torch.serialization.get_safe_globals()==before
    assert not any(p.requires_grad for p in model.parameters())
    with torch.inference_mode():
        out=model(torch.ones(2,3,8,8),torch.zeros(2,3,8,8))
    assert out.shape==(2,128)
    assert model.mlp.fc1.weight.shape==(512,1536)
    path=tmp_path/tiny_asset["path"]
    with torch.serialization.safe_globals([argparse.Namespace]):
        value=torch.load(path,weights_only=True)
    value["model"].pop("mlp.fc2.bias")
    bad=tmp_path/"missing.pth";torch.save(value,bad)
    with pytest.raises(RuntimeError,match="Missing key"):
        m.load_backbone(tmp_path,cfg(dict(path=bad.name,size=bad.stat().st_size,sha256=sha256(bad))),"cpu")


def test_unreviewed_checkpoint_global_is_rejected(tmp_path,tiny_asset):
    m,_=modules()
    bad=tmp_path/"unsafe.pth"
    torch.save(dict(model={},args=UnreviewedMetadata()),bad)
    with pytest.raises(ValueError,match="unreviewed"):
        m.load_backbone(tmp_path,cfg(dict(path=bad.name,size=bad.stat().st_size,sha256=sha256(bad))),"cpu")
    with pytest.raises(ValueError,match="identity"):
        m.load_backbone(tmp_path,cfg(dict(tiny_asset,sha256="0"*64)),"cpu")


def test_support_selection_is_train_only_unique_groups_and_order_independent():
    m,_=modules()
    rows=[row(i,"a" if i<15 else "b") for i in range(30)]
    chosen=m.select_support(rows,["a","b"],42)
    assert len(chosen)==20
    assert [r["sample_id"] for r in chosen]==[r["sample_id"] for r in m.select_support(rows[::-1],["a","b"],42)]
    assert all(len({r["group_id"] for r in chosen if r["source_label"]==source})==10 for source in ["a","b"])
    with pytest.raises(ValueError,match="train"):
        m.select_support(rows+[row(100,"a","dev")],["a","b"],42)
    with pytest.raises(ValueError,match="10"):
        m.select_support([dict(r,group_id="one") if r["source_label"]=="a" else r for r in rows],["a","b"],42)


def test_prototype_is_mean_of_unit_features_then_unit_mean():
    m,_=modules()
    rows=[row(i,"a" if i<10 else "b") for i in range(20)]
    f=torch.zeros(20,128)
    f[:5,0]=2;f[5:10,1]=3;f[10:,2]=4
    prototypes=m.prototype_vectors(f,rows,["a","b"])
    expected=torch.zeros(2,128);expected[0,:2]=2**-.5;expected[1,2]=1
    assert torch.allclose(prototypes,expected,atol=1e-7)
    raw,scores=m.prototype_scores(expected,prototypes)
    assert np.allclose(raw,np.eye(2),atol=1e-6)
    assert np.allclose(scores,[[.7310586,.2689414],[.2689414,.7310586]],atol=1e-6)
    assert np.array_equal(raw.argmax(1),scores.argmax(1))


def test_cache_and_image_guard_holdout_corruption_and_identity(tmp_path):
    m,_=modules()
    rows=[row(1),row(2,"b","dev")]
    identity=m.cache_identity(rows,cfg())
    path=tmp_path/"cache.pt";write_cache(path,torch.eye(2,128),identity)
    assert read_cache(path,identity,128).shape==(2,128)
    for changed in ([dict(rows[0],source_label="b"),rows[1]],rows[::-1]):
        with pytest.raises(ValueError,match="identity"):
            read_cache(path,m.cache_identity(changed,cfg()),128)
    for bad in ([row(1,split="holdout")],[row(1),row(1)],[dict(row(1),is_ai=0)]):
        with pytest.raises(ValueError):m.cache_identity(bad,cfg())
    with pytest.raises(ValueError,match="holdout"):
        m.encode_rows(torch.nn.Identity(),tmp_path,[row(1,split="holdout")],"cpu",42)
    (tmp_path/"1.png").write_bytes(b"bad bytes")
    with pytest.raises(ValueError,match="SHA"):
        m.encode_rows(torch.nn.Identity(),tmp_path,[row(1)],"cpu",42)
    with pytest.raises(OSError):
        m.encode_rows(torch.nn.Identity(),tmp_path,[dict(row(1),sha256=sha256(tmp_path/"1.png"))],"cpu",42)


def test_budget_recipe_and_output_reuse_are_rejected(tmp_path,monkeypatch):
    _,r=modules()
    config=cfg();r.validate_recipe(config)
    with pytest.raises(ValueError):r.validate_recipe(dict(config,seed=43))
    changed=cfg();changed["experiments"][r.PROTO]["k"]=9
    with pytest.raises(ValueError):r.validate_recipe(changed)
    out=tmp_path/"run";r.new_output(out);(out/"status.json").write_text('{"status":"failed"}')
    with pytest.raises(FileExistsError):r.new_output(out)
    (tmp_path/"disabled.json").write_text('{"execution_enabled":false}')
    monkeypatch.setattr(r,"ROOT",tmp_path);monkeypatch.setattr("sys.argv",["omnidfa","--config","disabled.json","--execute"])
    with pytest.raises(PermissionError,match="disabled"):r.main()


def toy_rows(tmp_path):
    rows=[row(i,"a" if i%2 else "b","train" if i<40 else "dev") for i in range(180)]
    rows.append(row(999,"a","holdout"))
    parts,_=job_parts(rows,dict(task="attribution",mode="clean"),42)
    replacements={}
    for i,r in enumerate(parts["selection"][:2]):
        path=tmp_path/r["relative_path"]
        Image.new("RGB",(90,160),[(20,80,180),(180,80,20)][i]).save(path)
        replacements[r["sample_id"]]=dict(r,sha256=sha256(path))
    return [replacements.get(r["sample_id"],r) for r in rows]


def test_actual_selection_cache_reload_and_mismatch(tmp_path,tiny_asset,monkeypatch):
    m,r=modules();monkeypatch.setattr(r,"ROOT",tmp_path)
    config=cfg(tiny_asset)
    rows=toy_rows(tmp_path);parts,_=job_parts(rows,dict(task="attribution",mode="clean"),42)
    chosen=parts["selection"][:2]
    model=m.load_backbone(tmp_path,config,"cpu")
    cached=m.encode_rows(model,tmp_path,chosen,"cpu",42)
    score=lambda x:x[:,:2].softmax(1).numpy()
    checks=r.verify_images(config,tmp_path,chosen,cached,score,"cpu")
    assert checks["images_opened"]==2 and checks["full_offline_backbone_reload"]
    with pytest.raises(ValueError,match="reproducibility"):
        r.verify_images(config,tmp_path,chosen,cached.flip(0),score,"cpu")
    with pytest.raises(ValueError,match="selection"):
        r.verify_images(config,tmp_path,[dict(chosen[0],split="train"),chosen[1]],cached,score,"cpu")


@pytest.mark.parametrize("estimator",["linear","prototype"])
def test_toy_full_job_30epochs_or_fixed_prototype(tmp_path,tiny_asset,monkeypatch,estimator):
    m,r=modules();monkeypatch.setattr(r,"ROOT",tmp_path)
    config=cfg(tiny_asset)
    name=r.LINEAR if estimator=="linear" else r.PROTO
    config["experiments"][name]["class_count"]=2
    rows=toy_rows(tmp_path);cache_rows=r.registered_rows(rows)
    parts,classes=job_parts(rows,config["experiments"][name],42)
    model=m.load_backbone(tmp_path,config,"cpu")
    chosen=parts["selection"][:2];actual=m.encode_rows(model,tmp_path,chosen,"cpu",42)
    features=normalized(torch.randn(len(cache_rows),128,generator=torch.Generator().manual_seed(42)))
    index={v["sample_id"]:i for i,v in enumerate(cache_rows)}
    for v,f in zip(chosen,actual):features[index[v["sample_id"]]]=f
    bad=tmp_path/"bad";bad.mkdir()
    with pytest.raises(ValueError,match="cache"):
        r.run_job(config,name,rows,cache_rows[::-1],features,tmp_path,bad,"cpu","cache")
    out=tmp_path/"job";out.mkdir()
    r.run_job(config,name,rows,cache_rows,features,tmp_path,out,"cpu","cache")
    assert json.loads((out/"status.json").read_text())["status"]=="completed"
    result=json.loads((out/"metrics.json").read_text())
    assert result["failed_samples"]==[] and result["holdout_scored"] is False
    predictions=[json.loads(x) for x in (out/"predictions.jsonl").read_text().splitlines()]
    assert {v["sample_id"] for v in predictions if v["role"]=="evaluation"}=={v["sample_id"] for v in parts["evaluation"]}
    assert not any(v["sample_id"]=="999" for v in predictions)
    if estimator=="linear":
        history=[json.loads(x) for x in (out/"linear/history.jsonl").read_text().splitlines()]
        assert [v["epoch"] for v in history]==list(range(1,31))
        with pytest.raises(ValueError,match="vocabulary"):
            r.load_head(out/"linear/head.pth",classes[::-1],"cpu")
    else:
        assert result["training_epochs"] is None and result["epoch"] is None and result["k"]==10
        support=json.loads((out/"support.json").read_text())
        assert len(support["samples"])==20
        assert all(x["split"]=="train" for x in support["samples"])
        raw=np.load(out/"evaluation-cosine.npy")
        scored=np.asarray([v["scores"] for v in predictions if v["role"]=="evaluation"])
        assert np.array_equal(raw.argmax(1),scored.argmax(1))

def test_preflight_binds_receipts_support_roles_and_sources(tmp_path,tiny_asset,monkeypatch):
    import importlib.metadata
    m,r=modules()
    original_root=r.ROOT
    for name in r.CODE_FILES:
        target=tmp_path/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((original_root/name).read_bytes())
    config=cfg(tiny_asset)
    config["dependencies"]={name:importlib.metadata.version(name) for name in r.DEPENDENCIES}
    config["implementation_sha256"]={name:sha256(tmp_path/name) for name in r.CODE_FILES}
    def receipt_file(name,body):
        path=tmp_path/name;path.write_text(json.dumps(body))
        return dict(path=name,size=path.stat().st_size,sha256=sha256(path))
    config["asset"]=dict(tiny_asset,
        import_receipt=receipt_file("import.json",dict(status="completed",bytes=tiny_asset["size"],sha256=tiny_asset["sha256"],exit_code=0)),
        static_inspection=receipt_file("static.json",dict(status="static_inspection_completed",asset_size=tiny_asset["size"],
            asset_sha256=tiny_asset["sha256"],unsafe_globals=["argparse.Namespace"])))
    cpu=dict(status="verified",asset_sha256=tiny_asset["sha256"],asset_size=tiny_asset["size"],
        strict_full_model_load=True,feature_dimensions=128,weights_only=True,allowed_checkpoint_globals=["argparse.Namespace"],
        implementation_sha256=config["implementation_sha256"],dependencies=config["dependencies"])
    config["asset_cpu_receipt"]=receipt_file("cpu.json",cpu)
    data=tmp_path/"data";data.mkdir();config["dataset"]="data"
    rows=[row(i,f"class-{i%22:02d}","train" if i<220 else "dev") for i in range(440)]
    rows.append(row(999,"class-00","holdout"))
    (data/"manifest.jsonl").write_text("".join(json.dumps(v)+"\n" for v in rows))
    (data/"selection.json").write_text("{}")
    config["manifest_sha256"]=sha256(data/"manifest.jsonl");config["selection_sha256"]=sha256(data/"selection.json")
    parts,classes=job_parts(rows,r.RECIPES[r.LINEAR],42)
    config["expected_job"]=dict(classes=classes,role_counts={k:len(v) for k,v in parts.items()},
        role_ids_sha256={k:r.digest([v["sample_id"] for v in values]) for k,values in parts.items()})
    config["expected_support_sha256"]=r.digest(m.support_manifest(parts["train"],classes,42))
    monkeypatch.setattr(r,"ROOT",tmp_path)
    prepared=r.prepare(config)
    assert len(prepared[2])==440 and all(v["split"]!="holdout" for v in prepared[2])
    with pytest.raises(ValueError,match="support"):
        r.prepare(dict(config,expected_support_sha256="changed"))
    with pytest.raises(ValueError,match="role bounds"):
        r.prepare(dict(config,expected_job={}))
    with pytest.raises(ValueError,match="implementation"):
        r.prepare(dict(config,implementation_sha256={**config["implementation_sha256"],r.CODE_FILES[0]:"changed"}))
    bad=cfg(tiny_asset);bad.update(config)
    bad["asset_cpu_receipt"]=receipt_file("bad-cpu.json",dict(cpu,weights_only=False))
    with pytest.raises(ValueError,match="CPU verification"):
        r.prepare(bad)
    bad=dict(config,asset=dict(config["asset"],import_receipt=receipt_file("bad-import.json",
        dict(status="running",bytes=tiny_asset["size"],sha256=tiny_asset["sha256"]))))
    with pytest.raises(ValueError,match="import receipt"):
        r.prepare(bad)
    bad=dict(config,asset=dict(config["asset"],static_inspection=receipt_file("bad-static.json",
        dict(status="failed",asset_size=tiny_asset["size"],asset_sha256=tiny_asset["sha256"],
             unsafe_globals=["argparse.Namespace"]))))
    with pytest.raises(ValueError,match="static inspection"):
        r.prepare(bad)


def test_nonfinite_cache_and_changed_dependencies_fail(tmp_path):
    m,_=modules()
    rows=[row(1)];identity=m.cache_identity(rows,cfg())
    path=tmp_path/"bad.pt"
    torch.save(dict(identity=identity,features=torch.full((1,128),float("nan"))),path)
    with pytest.raises(ValueError,match="nonfinite"):read_cache(path,identity,128)
    path2=tmp_path/"ok.pt";write_cache(path2,torch.eye(1,128),identity)
    changed=cfg();changed["dependencies"]={"timm":"changed"}
    with pytest.raises(ValueError,match="identity"):read_cache(path2,m.cache_identity(rows,changed),128)

def test_rank_only_softmax_keeps_adjacent_float32_cosine_order():
    m,_=modules()
    query=torch.zeros(1,128);query[0,0]=1
    centers=torch.zeros(22,128);centers[:,0]=.1;centers[:,1]=(1-.1**2)**.5
    centers[1,0]=torch.nextafter(centers[1,0],torch.tensor(float("inf")))
    centers[1,1]=(1-centers[1,0]**2)**.5
    raw,scores=m.prototype_scores(query,centers)
    assert raw[0,1]>raw[0,0]
    assert scores[0,1]>scores[0,0]
    assert scores.argmax(1).tolist()==[1]
