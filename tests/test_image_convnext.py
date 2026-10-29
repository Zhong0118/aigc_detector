import copy
import json
import pytest
import torch
from scripts import run_image_convnext as route


def row(i, split="train"):
    return dict(sample_id=str(i), group_id=str(i), relative_path=f"{i}.png",
                sha256=str(i), split=split, is_ai=1, source_label="a", semantic_class="imagenet")


def config():
    return dict(asset={"sha256": "asset"}, implementation_sha256={"code": "hash"})


def test_convnext_cache_has_distinct_identity_and_binds_image_labels():
    rows = [row(1)]
    identity = route.cache_identity(rows, config())
    assert identity["model"] == "convnext_tiny.fb_in22k_ft_in1k"
    assert identity["preprocessing"]["input_size"] == [3, 288, 288]
    changed = [dict(rows[0], source_label="different")]
    assert identity != route.cache_identity(changed, config())
    with pytest.raises(ValueError):
        route.cache_identity([row(2, "holdout")], config())
    with pytest.raises(ValueError):
        route.cache_identity([row(1),row(1)], config())


def test_existing_output_never_restarted(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "status.json").write_text('{"status":"failed"}')
    with pytest.raises(FileExistsError):
        route.new_output(output)
    assert json.loads((output / "status.json").read_text())["status"] == "failed"


def test_unknown_and_changed_training_protocol_rejected():
    cfg = dict(training=dict(epochs=30,batch_size=32,learning_rate=.001,scheduler_step=1,scheduler_gamma=.9,optimizer="Adam",backbone_frozen=True,loss="unweighted_CrossEntropyLoss"),seed=42,
        experiments={"D-ConvNeXt":dict(task="detection",mode="clean",class_count=2)})
    route.validate_recipe(cfg)
    for key,value in [("epochs",20),("batch_size",16),("backbone_frozen",False)]:
        bad=copy.deepcopy(cfg);bad["training"][key]=value
        with pytest.raises(ValueError):route.validate_recipe(bad)
    bad=copy.deepcopy(cfg);bad["experiments"]["A-unknown"]={"task":"attribution","mode":"open_set","class_count":20}
    with pytest.raises(ValueError):route.validate_recipe(bad)


def test_actual_image_cache_disagreement_blocks_completion(tmp_path, monkeypatch):
    import hashlib
    import numpy as np
    from PIL import Image
    class Encoder(torch.nn.Module):
        def forward(self,x):return torch.cat([x.mean((2,3))]*256,dim=1)
    transform=lambda im:torch.from_numpy(np.array(im,dtype=np.float32).transpose(2,0,1).copy())
    rows=[]
    for i,color in enumerate([(20,80,180),(180,80,20)]):
        p=tmp_path/f"{i}.png";Image.new("RGB",(4,4),color).save(p)
        rows.append(dict(row(i,"dev"),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    monkeypatch.setattr(route,"load_backbone",lambda *args:(Encoder(),transform))
    cached=route.encode_rows(Encoder(),transform,tmp_path,rows,"cpu")
    score=lambda x:x[:,:2].softmax(1).numpy()
    assert route.verify_images({},tmp_path,rows,cached,score,"cpu")["images_opened"]==2
    with pytest.raises(ValueError,match="reproducibility"):
        route.verify_images({},tmp_path,rows,cached.flip(0),score,"cpu")


def test_asset_hash_and_size_must_both_match(tmp_path):
    import hashlib
    p=tmp_path/'model';p.write_bytes(b'official')
    asset=dict(path='model',size=8,sha256=hashlib.sha256(b'official').hexdigest())
    assert route.verify_asset(tmp_path,asset)==p
    p.write_bytes(b'modified')
    with pytest.raises(ValueError,match='identity'):route.verify_asset(tmp_path,asset)


@pytest.mark.parametrize('task,name', [('detection','D-ConvNeXt'),('attribution','A-ConvNeXt22')])
def test_cpu_job_real_history_roles_and_provenance(tmp_path, monkeypatch, task, name):
    monkeypatch.setattr(route,'ROOT',tmp_path)
    monkeypatch.setattr(route,'verify_images',lambda *args:{'images_opened':2})
    torch.set_num_threads(1)
    rows=[]
    for i in range(150):
        r=dict(row(i,'train' if i<40 else 'dev'),source_label=['a','b'][i%2])
        if task=='detection' and i%2:r.update(is_ai=0,source_label=None)
        rows.append(r)
    cfg=dict(seed=42,experiments={name:dict(task=task,mode='clean')},asset={},model_config={},dependencies={},
        manifest_sha256='m',selection_sha256='s',implementation_sha256={},limitations=[],bootstrap_repeats=2,
        training=dict(epochs=30,batch_size=32,learning_rate=.001,scheduler_step=1,scheduler_gamma=.9))
    features=route.normalized(torch.randn(len(rows),768,generator=torch.Generator().manual_seed(42)))
    out=tmp_path/'job';out.mkdir()
    route.run_job(cfg,name,rows,rows,features,tmp_path,out,'cpu','cache')
    assert json.loads((out/'status.json').read_text())['exit_code']==0
    assert [json.loads(x)['epoch'] for x in (out/'linear/history.jsonl').read_text().splitlines()]==list(range(1,31))
    records=[json.loads(x) for x in (out/'predictions.jsonl').read_text().splitlines()]
    assert {r['role'] for r in records}=={'calibration','evaluation'}
    assert all(int(r['sample_id'])>=40 for r in records)
    p=json.loads((out/'provenance.json').read_text())
    assert p['model']==route.MODEL and p['preprocessing']==route.PREPROCESSING
    assert not p['holdout_scored']
