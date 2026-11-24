"""Package and verify the precise text inference asset set."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile

RELEASE = "text-binary-20261124-en-v2a"
MODEL_NAMES = {
    "zh_short": "20261117-mdeberta-v1", "zh_long": "20261123-mdeberta-v2-six-layer",
    "en_short": "20261124-en-v2", "en_long": "20261124-en-v2",
}
WEIGHT_HASHES = {
    "20261117-mdeberta-v1": "044d4c22c9c659c11b8f6872f4d660cea48e73ac6ac4d7c9df77cbff003c77d9",
    "20261123-mdeberta-v2-six-layer": "2339b66fcd3da0481e70608142760d9aac6f503142054b2aecc05fc35b82e34a",
    "20261124-en-v2": "11e172d0550e884f3ebfe34a7fe6db08c4c66b758736714943e305dbc382789b",
}
BINARY_FILES = (
    "added_tokens.json", "config.json", "model.safetensors", "special_tokens_map.json",
    "spm.model", "tokenizer.json", "tokenizer_config.json",
)
A0_SOURCE = "outputs/attribution/family-window-split/A0-canonical-seed42-20261117"
A0_FILES = (
    "model/model.safetensors", "model/backbone_config/config.json", "model/phase3_model_config.json",
    "model/resolved_recipe.json", "tokenizer/sentencepiece.bpe.model", "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json",
    "resolved_recipe.json", "label_maps.json", "window_audit.json",
)
CLASSICAL = {
    "B0": "outputs/attribution/family-phase4/fast-track-v1/classical/closed-seed42/B0/model.joblib",
    "B1": "outputs/attribution/family-phase4/fast-track-v1/classical/closed-seed42/B1/model.joblib",
    "B3": "outputs/attribution/family-phase1/final/models/B3.joblib",
}
META = ("bundle-manifest.json", "SHA256SUMS")
SOURCE_PROTOCOL = "token-logit-v1:2e084c9491123921"


def assets(root: Path, deployment: Path) -> dict[str, Path]:
    files = {f"models/text_binary/{name}/{file}": root / "models/text_binary" / name / file
             for name in sorted(set(MODEL_NAMES.values())) for file in BINARY_FILES}
    files.update({f"attribution/A0/{file}": root / A0_SOURCE / file for file in A0_FILES})
    files.update({f"attribution/{name}/model.joblib": root / path for name, path in CLASSICAL.items()})
    files.update({f"calibration/{file}": deployment / file for file in ("temperature.json", "thresholds.json")})
    return files


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4*1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def checksum_lines(files: dict) -> str:
    return "".join(f"{meta['sha256']}  {name}\n" for name,meta in sorted(files.items()))


def _metadata(file: Path) -> dict:
    return {"bytes": file.stat().st_size, "sha256": sha256(file)}


def pack(root: Path, deployment: Path, output: Path) -> None:
    paths = assets(root, deployment)
    missing = [str(p) for p in paths.values() if not p.is_file() or p.is_symlink() or not p.stat().st_size]
    if missing:
        raise ValueError("Missing required regular assets:\n" + "\n".join(missing))
    calibration = [json.loads(paths[f"calibration/{file}"].read_text(encoding="utf-8"))
                   for file in ("temperature.json", "thresholds.json")]
    if any(a["scoring_protocol"] != SOURCE_PROTOCOL for a in calibration):
        raise ValueError("Not the frozen bilingual en-v2 A release")
    if calibration[0]["model_routes"] != calibration[1]["model_routes"]:
        raise ValueError("Temperature and threshold route tables differ")
    files = {name: _metadata(file) for name,file in paths.items()}
    for name,expected in WEIGHT_HASHES.items():
        if files[f"models/text_binary/{name}/model.safetensors"]["sha256"] != expected:
            raise ValueError(f"Unexpected detector checkpoint: {name}")

    manifest = dict(schema=1, release=RELEASE, source_scoring_protocol=SOURCE_PROTOCOL,
                    files=files, payload_bytes=sum(r["bytes"] for r in files.values()))
    payload = (json.dumps(manifest,indent=2,sort_keys=True)+"\n").encode()
    checksummed = dict(files, **{"bundle-manifest.json":{"sha256":hashlib.sha256(payload).hexdigest()}})
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with tempfile.NamedTemporaryFile(prefix=f".{output.name}.", dir=output.parent, delete=False) as temporary:
        tmp = Path(temporary.name)
    try:
        with tarfile.open(tmp, "w:gz", compresslevel=1, dereference=True) as archive:
            for name,content in zip(META, (payload,checksum_lines(checksummed).encode()), strict=True):
                item = tarfile.TarInfo(name)
                item.size = len(content)
                item.mode = 0o600
                archive.addfile(item,io.BytesIO(content))
            for name,path in sorted(paths.items()):
                item = archive.gettarinfo(str(path),arcname=name)
                if not item.isfile() or item.size != files[name]["bytes"]:
                    raise ValueError(f"Asset changed while packing: {path}")
                item.uid=item.gid=0
                item.uname=item.gname=""
                item.mode=0o600
                with path.open("rb") as handle:
                    archive.addfile(item,handle)
        # link publishes atomically without replacing an existing bundle.
        os.link(tmp,output)
    finally:
        tmp.unlink(missing_ok=True)
    print(f"Bundle {output}\nCompressed bytes {output.stat().st_size}\nPayload bytes {manifest['payload_bytes']}")
    for name in (*META,*sorted(files)):
        print(name)


def _read_metadata(archive):
    metadata = []
    for name in META:
        item = archive.next()
        if not item or item.name != name or not item.isfile() or item.size > 128*1024:
            raise ValueError(f"Invalid bundle metadata: {name}")
        with archive.extractfile(item) as handle:
            metadata.append(handle.read())
    payload,sums=metadata
    manifest=json.loads(payload)
    files=manifest["files"]
    if manifest["schema"] != 1 or manifest["release"] != RELEASE or set(files)!=set(assets(Path("."),Path("."))):
        raise ValueError("Unexpected bundle release or asset allowlist")
    if manifest["source_scoring_protocol"] != SOURCE_PROTOCOL:
        raise ValueError("Unexpected calibration protocol")
    checksummed=dict(files, **{"bundle-manifest.json":{"sha256":hashlib.sha256(payload).hexdigest()}})
    if sums != checksum_lines(checksummed).encode():
        raise ValueError("SHA256SUMS does not match the manifest")
    for name,digest in WEIGHT_HASHES.items():
        if files[f"models/text_binary/{name}/model.safetensors"]["sha256"] != digest:
            raise ValueError(f"Unexpected detector checkpoint: {name}")
    return payload,sums,manifest


def verify_installed(bundle: Path, target: Path) -> None:
    if not target.is_dir() or target.is_symlink():
        raise ValueError("runtime_assets must be a regular local directory")
    with tarfile.open(bundle,"r|gz") as archive:
        payload,sums,manifest=_read_metadata(archive)
    if (target/META[0]).read_bytes()!=payload or (target/META[1]).read_bytes()!=sums:
        raise ValueError("Existing assets came from a different bundle")
    for name,meta in manifest["files"].items():
        path=target/name
        if not path.is_file() or path.resolve()!=path or _metadata(path)!=meta:
            raise ValueError(f"Installed asset checksum mismatch: {name}")


def unpack(bundle: Path, target: Path) -> dict:
    """Stream and verify the pinned allowlist. Extraction never follows archive links."""
    expected = set(assets(Path("."),Path(".")))
    with tarfile.open(bundle,"r|gz") as archive:
        payload,sums,manifest=_read_metadata(archive)
        files=manifest["files"]
        seen=set()
        with tempfile.TemporaryDirectory(prefix=".unpack-",dir=target.parent) as directory:
            temp=Path(directory)
            while (item := archive.next()) is not None:
                if item.name not in expected or item.name in seen or not item.isfile():
                    raise ValueError(f"Unexpected archive entry: {item.name}")
                if item.size != files[item.name]["bytes"]:
                    raise ValueError(f"Asset size mismatch: {item.name}")
                file=temp/item.name
                file.parent.mkdir(parents=True,exist_ok=True)
                with archive.extractfile(item) as source, file.open("xb") as destination:
                    shutil.copyfileobj(source,destination,4*1024*1024)
                if sha256(file) != files[item.name]["sha256"]:
                    raise ValueError(f"SHA256 mismatch: {item.name}")
                seen.add(item.name)
            if seen != expected:
                raise ValueError("Missing archive assets")
            (temp/META[0]).write_bytes(payload)
            (temp/META[1]).write_bytes(sums)
            temp.rename(target)
    return manifest


def main() -> None:
    default = Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=default)
    parser.add_argument("--deployment", type=Path, help=f"Defaults to deployments/{RELEASE} under source-root")
    parser.add_argument("--output", type=Path, help="Defaults to dist/aigc-text-runtime-20261124.tar.gz")
    opts=parser.parse_args()
    root=opts.source_root.resolve()
    deployment=(opts.deployment or root/"deployments"/RELEASE).resolve()
    output=(opts.output or root/"dist/aigc-text-runtime-20261124.tar.gz").resolve()
    pack(root,deployment,output)


if __name__=="__main__":
    main()
