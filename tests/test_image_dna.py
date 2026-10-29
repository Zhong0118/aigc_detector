"""CPU boundaries for the new, offline DNA-Det adaptation."""
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from research.image_attribution.openclip_route import job_parts, normalized, read_cache, write_cache
from research.image_attribution.protocol import sha256


def modules():
    assert importlib.util.find_spec("research.image_attribution.dna_route") is not None, "DNA route not implemented"
    from research.image_attribution import dna_route
    from scripts import run_image_dna
    return dna_route, run_image_dna


def row(i, source="a", split="train"):
    return dict(sample_id=str(i), group_id=str(i), sha256=str(i), pixel_sha256=str(i),
                relative_path=f"{i}.png", source_label=source, split=split,
                is_ai=1, semantic_class="test")


@pytest.fixture
def asset(tmp_path):
    route, _ = modules()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = route.DNAEncoder()
    path = tmp_path / "model.pth"
    torch.save(model.state_dict(), path)
    return dict(path=path.name, size=path.stat().st_size, sha256=sha256(path))


def config(asset=None):
    route, _ = modules()
    return dict(seed=42, asset=asset or {}, dependencies={}, implementation_sha256={},
        preprocessing=route.PREPROCESSING, manifest_sha256="manifest", selection_sha256="selection",
        training=dict(epochs=30, batch_size=32, learning_rate=.001, scheduler_step=1,
                      scheduler_gamma=.9, optimizer="Adam", backbone_frozen=True,
                      loss="unweighted_CrossEntropyLoss"),
        experiments={"A-DNA-Wild-LP22":dict(task="attribution", mode="clean", class_count=22)},
        limitations=["CPU fixture; no paper performance claim"])


def test_complete_official_state_mapping_and_frozen_features(tmp_path, asset):
    route, _ = modules()
    state = torch.load(tmp_path / asset["path"], weights_only=True)
    assert state["main.0.main.0.weight"].shape == (64, 3, 4, 4)
    assert state["main.7.main.0.weight"].shape == (512, 512, 3, 3)
    assert state["classification_head.1.weight"].shape == (11, 512)
    model, transform = route.load_backbone(tmp_path, config(asset), "cpu")
    assert not any(p.requires_grad for p in model.parameters())
    with torch.inference_mode():
        z = model(transform(Image.new("RGB", (128, 128), "red")).unsqueeze(0))
    assert z.shape == (1, 512)
    assert torch.isfinite(z).all()
    # A missing original classifier must fail before it can be discarded.
    state.pop("classification_head.1.weight")
    bad = tmp_path / "missing.pth"
    torch.save(state, bad)
    bad_asset = dict(path=bad.name, size=bad.stat().st_size, sha256=sha256(bad))
    with pytest.raises(RuntimeError, match="Missing key"):
        route.load_backbone(tmp_path, config(bad_asset), "cpu")


@pytest.mark.parametrize("size", [(80, 180), (200, 200)])
def test_preprocessing_matches_original_full_image_pipeline(size):
    route, _ = modules()
    from torchvision.transforms import CenterCrop, ToTensor, Normalize
    rng = np.random.default_rng(3)
    image = Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8))
    expected = image
    if size[0] != size[1]:
        expected = CenterCrop((128, 128))(expected)
    expected = expected.resize((128, 128)).resize((512, 512))
    expected = Normalize([.5]*3, [.5]*3)(ToTensor()(expected))
    assert torch.equal(route.preprocess(image), expected)


def test_holdout_and_bad_image_bytes_fail_before_inference(tmp_path):
    route, _ = modules()
    encoder = torch.nn.Identity()
    with pytest.raises(ValueError, match="holdout"):
        route.encode_rows(encoder, route.preprocess, tmp_path, [row(1, split="holdout")], "cpu")
    Image.new("RGB", (10, 10)).save(tmp_path / "1.png")
    with pytest.raises(ValueError, match="SHA"):
        route.encode_rows(encoder, route.preprocess, tmp_path, [row(1)], "cpu")


def test_asset_integrity_gate(tmp_path, asset):
    route, _ = modules()
    with pytest.raises(ValueError, match="identity|incomplete"):
        route.verify_asset(tmp_path, dict(asset, sha256="0"*64))
    with pytest.raises(ValueError, match="identity|incomplete"):
        route.verify_asset(tmp_path, dict(asset, size=asset["size"]+1))
    with pytest.raises(ValueError, match="unverified"):
        route.verify_asset(tmp_path, dict(asset, sha256=None))


def test_cache_binds_roles_labels_preprocess_and_dependencies(tmp_path):
    route, _ = modules()
    rows = [row(1), row(2, "b", "dev")]
    cfg = config()
    identity = route.cache_identity(rows, cfg)
    path = tmp_path / "features.pt"
    write_cache(path, torch.eye(2, 512), identity)
    assert read_cache(path, identity, 512).shape == (2, 512)
    for altered in ([dict(rows[0], source_label="b"), rows[1]],
                    [dict(rows[0], group_id="other"), rows[1]], rows[::-1]):
        with pytest.raises(ValueError, match="identity"):
            read_cache(path, route.cache_identity(altered, cfg), 512)
    for key, value in [("dependencies", {"torch":"changed"}), ("asset", {"sha256":"changed"}),
                       ("manifest_sha256", "changed")]:
        with pytest.raises(ValueError, match="identity"):
            read_cache(path, route.cache_identity(rows, dict(cfg, **{key:value})), 512)
    for bad in ([row(1, split="holdout")], [row(1), row(1)], [dict(row(1), is_ai=0)]):
        with pytest.raises(ValueError):
            route.cache_identity(bad, cfg)


def test_registered_recipe_rejects_silent_budget_and_task_changes():
    _, runner = modules()
    cfg = config()
    runner.validate_recipe(cfg)
    for bad in (dict(cfg, seed=43), dict(cfg, training=dict(cfg["training"], epochs=29)),
                dict(cfg, experiments={"other":dict(task="attribution",mode="clean",class_count=22)})):
        with pytest.raises(ValueError):
            runner.validate_recipe(bad)


def test_failed_or_completed_output_is_never_reused(tmp_path):
    _, runner = modules()
    out = tmp_path / "run"
    runner.new_output(out)
    (out / "status.json").write_text('{"status":"failed"}')
    with pytest.raises(FileExistsError):
        runner.new_output(out)
    assert json.loads((out / "status.json").read_text())["status"] == "failed"


def image_rows(tmp_path, chosen):
    result = []
    for i, r in enumerate(chosen):
        path = tmp_path / r["relative_path"]
        Image.new("RGB", (130, 170), [(20,80,180),(180,80,20)][i]).save(path)
        result.append(dict(r, sha256=sha256(path)))
    return result


def test_actual_selection_images_strict_reload_and_cache_mismatch(tmp_path, asset, monkeypatch):
    route, runner = modules()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    cfg = config(asset)
    rows = [row(i, "a" if i%2 else "b", "dev") for i in range(30)]
    parts, _ = job_parts([row(100),row(101,"b")]+rows, cfg["experiments"]["A-DNA-Wild-LP22"],42)
    selected = image_rows(tmp_path, parts["selection"][:2])
    model, transform = route.load_backbone(tmp_path, cfg, "cpu")
    cached = route.encode_rows(model,transform,tmp_path,selected,"cpu")
    score = lambda x:x[:,:2].softmax(1).numpy()
    checks = runner.verify_images(cfg,tmp_path,selected,cached,score,"cpu")
    assert checks["images_opened"] == 2 and checks["full_offline_backbone_reload"]
    with pytest.raises(ValueError, match="reproducibility"):
        runner.verify_images(cfg,tmp_path,selected,cached.flip(0),score,"cpu")
    with pytest.raises(ValueError, match="selection"):
        runner.verify_images(cfg,tmp_path,[dict(selected[0],split="train"),selected[1]],cached,score,"cpu")


def test_toy_thirty_epoch_job_and_vocabulary_cache_bounds(tmp_path, asset, monkeypatch):
    route, runner = modules()
    monkeypatch.setattr(runner,"ROOT",tmp_path)
    cfg = config(asset)
    cfg["experiments"]["A-DNA-Wild-LP22"]["class_count"] = 2
    rows = [row(i,"a" if i%2 else "b","train" if i<40 else "dev") for i in range(180)]
    rows.append(row(999,"a","holdout"))
    parts, classes = job_parts(rows,cfg["experiments"]["A-DNA-Wild-LP22"],42)
    selected = image_rows(tmp_path,parts["selection"][:2])
    updates = {r["sample_id"]:r for r in selected}
    rows = [updates.get(r["sample_id"],r) for r in rows]
    cache_rows = sorted([r for r in rows if r["split"]!="holdout"], key=lambda r:r["sample_id"])
    model, transform = route.load_backbone(tmp_path,cfg,"cpu")
    actual = route.encode_rows(model,transform,tmp_path,selected,"cpu")
    features = normalized(torch.randn(len(cache_rows),512,generator=torch.Generator().manual_seed(42)))
    index = {r["sample_id"]:i for i,r in enumerate(cache_rows)}
    for r,f in zip(selected,actual):features[index[r["sample_id"]]]=f
    badout = tmp_path / "bad";badout.mkdir()
    with pytest.raises(ValueError,match="cache"):
        runner.run_job(cfg,rows,cache_rows[::-1],features,tmp_path,badout,"cpu","cachehash")
    out=tmp_path/"job";out.mkdir()
    runner.run_job(cfg,rows,cache_rows,features,tmp_path,out,"cpu","cachehash")
    assert json.loads((out/"status.json").read_text())["status"] == "completed"
    history=[json.loads(x) for x in (out/"linear/history.jsonl").read_text().splitlines()]
    assert [x["epoch"] for x in history] == list(range(1,31))
    predictions=[json.loads(x) for x in (out/"predictions.jsonl").read_text().splitlines()]
    assert {p["role"] for p in predictions} == {"calibration","evaluation"}
    assert not any(p["sample_id"]=="999" for p in predictions)
    assert {p["sample_id"] for p in predictions if p["role"]=="evaluation"} == {r["sample_id"] for r in parts["evaluation"]}
    assert json.loads((out/"metrics.json").read_text())["failed_samples"] == []
    head=runner.load_head(out/"linear/head.pth",classes,"cpu")
    assert head.in_features == 512
    with pytest.raises(ValueError,match="vocabulary"):
        runner.load_head(out/"linear/head.pth",classes[::-1],"cpu")

def test_asset_supports_existing_shared_outputs_symlink(tmp_path, asset):
    route, _ = modules()
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared/"model.pth").write_bytes((tmp_path/asset["path"]).read_bytes())
    workspace = tmp_path/"workspace";workspace.mkdir()
    (workspace/"outputs").symlink_to(shared, target_is_directory=True)
    expected = dict(asset,path="outputs/model.pth")
    assert route.verify_asset(workspace,expected).read_bytes() == (shared/"model.pth").read_bytes()
    with pytest.raises(ValueError,match="path"):
        route.verify_asset(workspace,dict(expected,path="../shared/model.pth"))


def test_disabled_execution_stops_before_preflight(tmp_path,monkeypatch):
    _, runner=modules()
    (tmp_path/"disabled.json").write_text('{"execution_enabled":false}')
    monkeypatch.setattr(runner,"ROOT",tmp_path)
    monkeypatch.setattr("sys.argv",["dna","--config","disabled.json","--execute"])
    with pytest.raises(PermissionError,match="disabled"):
        runner.main()
    assert sorted(x.name for x in tmp_path.iterdir()) == ["disabled.json"]

def test_preflight_binds_cpu_receipt_dependencies_code_and_role_ids(tmp_path,asset,monkeypatch):
    import importlib.metadata
    route,runner=modules()
    original_root=runner.ROOT
    for name in runner.CODE_FILES:
        target=tmp_path/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((original_root/name).read_bytes())
    cfg=config(asset)
    cfg.update(dataset="data",expected_job={})
    cfg["dependencies"]={name:importlib.metadata.version(name) for name in runner.DEPENDENCIES}
    cfg["implementation_sha256"]={name:sha256(tmp_path/name) for name in runner.CODE_FILES}
    data=tmp_path/"data";data.mkdir()
    rows=[row(i,f"class-{i%22:02d}","train" if i<44 else "dev") for i in range(220)]
    rows.append(row(999,"class-00","holdout"))
    (data/"manifest.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    (data/"selection.json").write_text("{}")
    cfg["manifest_sha256"]=sha256(data/"manifest.jsonl")
    cfg["selection_sha256"]=sha256(data/"selection.json")
    parts,classes=job_parts(rows,runner.RECIPE,42)
    cfg["expected_job"]=dict(classes=classes,role_counts={k:len(v) for k,v in parts.items()},
        role_ids_sha256={k:runner.digest([r["sample_id"] for r in v]) for k,v in parts.items()})
    receipt=dict(status="verified",asset_sha256=asset["sha256"],asset_size=asset["size"],
        strict_full_model_load=True,original_classes=11,feature_dimensions=512,weights_only=True,
        implementation_sha256=cfg["implementation_sha256"],dependencies=cfg["dependencies"])
    p=tmp_path/"cpu.json";p.write_text(json.dumps(receipt))
    cfg["asset_cpu_receipt"]=dict(path=p.name,sha256=sha256(p),size=p.stat().st_size)
    monkeypatch.setattr(runner,"ROOT",tmp_path)
    prepared=runner.prepare(cfg)
    assert len(prepared[2])==220 and all(r["split"]!="holdout" for r in prepared[2])
    bad=dict(cfg,expected_job=dict(cfg["expected_job"],role_ids_sha256={}))
    with pytest.raises(ValueError,match="role bounds"):
        runner.prepare(bad)
    bad=dict(cfg,dependencies=dict(cfg["dependencies"],numpy="unregistered"))
    with pytest.raises(ValueError,match="dependency"):
        runner.prepare(bad)
    (tmp_path/runner.CODE_FILES[0]).write_text("changed")
    with pytest.raises(ValueError,match="implementation"):
        runner.prepare(cfg)
    (tmp_path/runner.CODE_FILES[0]).write_bytes((original_root/runner.CODE_FILES[0]).read_bytes())
    p.write_text(json.dumps(dict(receipt,strict_full_model_load=False)))
    cfg["asset_cpu_receipt"]=dict(path=p.name,sha256=sha256(p),size=p.stat().st_size)
    with pytest.raises(ValueError,match="CPU verification"):
        runner.prepare(cfg)

def test_corrupt_image_with_matching_hash_is_not_skipped(tmp_path):
    route,_=modules()
    path=tmp_path/"7.png";path.write_bytes(b"not an image")
    bad=dict(row(7),sha256=sha256(path))
    with pytest.raises(OSError):
        route.encode_rows(torch.nn.Identity(),route.preprocess,tmp_path,[bad],"cpu")


def test_cache_rejects_nonfinite_tensor_payload(tmp_path):
    route,_=modules()
    identity=route.cache_identity([row(1)],config())
    path=tmp_path/"bad.pt"
    torch.save(dict(identity=identity,features=torch.full((1,512),float("nan"))),path)
    with pytest.raises(ValueError,match="nonfinite"):
        read_cache(path,identity,512)
