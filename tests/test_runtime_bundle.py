"""Portable bundle selection, integrity, and process ownership contracts."""
import hashlib
import io
import json
import tarfile

import pytest

from scripts import runtime_bundle


def tiny_bundle(tmp_path, monkeypatch):
    deployment = tmp_path / "deployment"
    files = runtime_bundle.assets(tmp_path, deployment)
    routes = {"zh_short":"zh_short","zh_long":"zh_long","en_short":"en","en_long":"en"}
    for name,path in files.items():
        path.parent.mkdir(parents=True,exist_ok=True)
        if name.startswith("calibration/"):
            path.write_text(json.dumps({"scoring_protocol":runtime_bundle.SOURCE_PROTOCOL,"model_routes":routes}))
        else:
            path.write_bytes(name.encode())
    monkeypatch.setattr(runtime_bundle,"WEIGHT_HASHES",{
        name:hashlib.sha256(f"models/text_binary/{name}/model.safetensors".encode()).hexdigest()
        for name in runtime_bundle.WEIGHT_HASHES
    })
    (tmp_path/"preview.db").write_bytes(b"private database")
    (tmp_path/"checkpoints").mkdir()
    (tmp_path/"checkpoints/old.pt").write_bytes(b"old checkpoint")
    archive = tmp_path / "runtime.tar.gz"
    runtime_bundle.pack(tmp_path,deployment,archive)
    return archive, files


def test_bundle_uses_explicit_files_and_checks_installed_bytes(tmp_path,monkeypatch):
    bundle,files=tiny_bundle(tmp_path,monkeypatch)
    target=tmp_path/"runtime_assets"
    with tarfile.open(bundle) as tar:
        assert set(tar.getnames())==set(files)|set(runtime_bundle.META)
        assert all(item.isfile() for item in tar)
    runtime_bundle.unpack(bundle,target)
    runtime_bundle.verify_installed(bundle,target)
    defaced=target/next(iter(files))
    defaced.write_bytes(b"unexpected bytes")
    with pytest.raises(ValueError,match="checksum mismatch"):
        runtime_bundle.verify_installed(bundle,target)


def test_unexpected_archive_member_cannot_escape_extraction(tmp_path,monkeypatch):
    bundle,_=tiny_bundle(tmp_path,monkeypatch)
    poisoned=tmp_path/"unexpected.tar.gz"
    with tarfile.open(bundle) as original, tarfile.open(poisoned,"w:gz") as archive:
        for member in original:
            archive.addfile(member,original.extractfile(member))
        bad=tarfile.TarInfo("../escape")
        bad.size=1
        archive.addfile(bad,io.BytesIO(b"x"))
    target=tmp_path/"runtime_assets"
    with pytest.raises(ValueError,match="Unexpected archive entry"):
        runtime_bundle.unpack(poisoned,target)
    assert not target.exists() and not (tmp_path/"escape").exists()


def test_database_config_uses_explicit_runtime_path(tmp_path,monkeypatch):
    from storage.database import _load_db_path

    config=tmp_path/"config.yaml"
    config.write_text("database:\n  path: /dedicated/runtime/history.db\n")
    monkeypatch.setenv("AIGC_CONFIG",str(config))
    assert _load_db_path()=="/dedicated/runtime/history.db"
