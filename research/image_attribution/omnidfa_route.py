"""Independent fixed-view adapter for official OmniDFA part1 fingerprints.

Architecture and validation transforms:
https://github.com/teheperinko541/OmniDFA/tree/35b9052e83e05436682095818693493f79da9458
(model/model.py, datasets/omnifake_episode.py, datasets/custom_transforms.py).
The checkpoint model card is Apache-2.0; upstream source licensing is unestablished.
"""
import argparse
import hashlib
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image
import torch
from torch import nn
from timm import create_model
from timm.layers import Mlp
from torchvision import transforms

from research.image_attribution.openclip_route import digest, image_tensor, normalized, prototypes
from research.image_attribution.protocol import sha256

DIMENSION = 128
MODEL = "OmniDFA-part1-TwinNeXt-frozen"
ALLOWED_GLOBALS = ["argparse.Namespace"]
PREPROCESSING = dict(rgb=True, min_short_edge=256,
    minimum_resize_interpolations=["PIL-BILINEAR","PIL-BICUBIC","PIL-LANCZOS"],
    local_crop=224, global_short_edge=224, global_interpolation="tensor-bilinear",
    global_antialias=True, global_crop=224, mean=None, std=None,
    view_seed="sha256(omnidfa-fixed-view-v1:seed:sample_id) first8bytes modulo 2**63-1",
    feature_normalization="l2", original_episode_sampling=False)


class TwinFingerprint(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone1 = create_model("convnext_small", pretrained=False, num_classes=0)
        self.backbone2 = create_model("convnext_small", pretrained=False, num_classes=0)
        if self.backbone1.num_features != 768 or self.backbone2.num_features != 768:
            raise ValueError("unexpected ConvNeXt-Small feature width")
        self.mlp = Mlp(in_features=1536, hidden_features=512, out_features=DIMENSION)

    def forward(self, local, global_view):
        return self.mlp(torch.cat((self.backbone1(local), self.backbone2(global_view)), dim=1))


def sample_seed(sample_id, seed):
    raw = hashlib.sha256(f"omnidfa-fixed-view-v1:{seed}:{sample_id}".encode()).digest()
    return int.from_bytes(raw[:8], "big") % (2**63-1)


def preprocess(image, sample_id, seed):
    """Retain validation geometry, fixing one view per sample without changing caller RNG."""
    image = image.convert("RGB")
    fixed_seed = sample_seed(sample_id, seed)
    rng = random.Random(fixed_seed)
    scale = max(1., 256/image.width, 256/image.height)
    if scale != 1.:
        mode = rng.choice((Image.Resampling.BILINEAR, Image.Resampling.BICUBIC, Image.Resampling.LANCZOS))
        image = image.resize((round(image.width*scale), round(image.height*scale)), mode)
    tensor = transforms.functional.to_tensor(image)
    # Seed only the CPU generator used by RandomCrop; never initialize or reseed CUDA.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(fixed_seed)
        local = transforms.RandomCrop(224)(tensor)
        global_view = transforms.RandomCrop(224)(transforms.Resize(224, antialias=True)(tensor))
    return local, global_view


def verify_asset(root, asset):
    expected = asset.get("sha256")
    if not isinstance(expected, str) or re.fullmatch("[0-9a-f]{64}", expected) is None:
        raise ValueError("unverified asset SHA")
    root = Path(root).resolve()
    relative = Path(asset["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("asset path escapes workspace")
    path = (root/relative).resolve()
    shared = (root/"outputs/image_experiments/assets").resolve()
    if not path.is_relative_to(root) and not (
            relative.parts[:3] == ("outputs","image_experiments","assets") and path.is_relative_to(shared)):
        raise ValueError("asset path escapes workspace")
    if path.stat().st_size != asset["size"] or sha256(path) != expected:
        raise ValueError("official asset identity mismatch")
    return path


def load_backbone(root, config, device):
    path = verify_asset(root, config["asset"])
    unsafe = sorted(torch.serialization.get_unsafe_globals_in_checkpoint(path))
    if unsafe != ALLOWED_GLOBALS:
        raise ValueError(f"unreviewed checkpoint globals: {unsafe}")
    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError("checkpoint model state_dict missing")
    state = checkpoint.pop("model")
    del checkpoint  # Training optimizer tensors are not needed for frozen inference.
    if not all(isinstance(x, torch.Tensor) for x in state.values()):
        raise ValueError("model state must contain tensors only")
    model = TwinFingerprint()
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False).to(device)


def cache_identity(rows, config):
    if any(r["split"] not in ("train","dev") for r in rows):
        raise ValueError("holdout forbidden in feature cache")
    if any(r["is_ai"] != 1 for r in rows):
        raise ValueError("OmniDFA attribution cache requires AI rows")
    ids = [r["sample_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate cache IDs")
    if config["preprocessing"] != PREPROCESSING:
        raise ValueError("preprocessing changed")
    return dict(schema="omnidfa-fixed-view-cache-v1", model=MODEL, dimensions=DIMENSION,
        preprocessing=PREPROCESSING, ids=ids,
        rows_sha256=digest([{k:r[k] for k in ("sample_id","group_id","relative_path","sha256",
            "pixel_sha256","split","is_ai","source_label","semantic_class")} for r in rows]),
        asset=config["asset"], code=config["implementation_sha256"], dependencies=config["dependencies"],
        seed=config["seed"], manifest_sha256=config["manifest_sha256"], selection_sha256=config["selection_sha256"])


def encode_rows(model, data, rows, device, seed, batch_size=32, progress=None):
    if not rows or batch_size < 1:
        raise ValueError("nonempty rows and positive batch size required")
    if any(r["split"] not in ("train","dev") for r in rows):
        raise ValueError("holdout image access forbidden")
    if any(r["is_ai"] != 1 for r in rows):
        raise ValueError("AI rows required")
    batches = []
    with torch.inference_mode():
        for start in range(0,len(rows),batch_size):
            views = [image_tensor(data, r, lambda image:preprocess(image,r["sample_id"],seed))
                     for r in rows[start:start+batch_size]]
            local, global_view = (torch.stack(v).to(device) for v in zip(*views))
            batches.append(normalized(model(local,global_view),DIMENSION).cpu())
            if progress and (start % (batch_size*32)==0 or start+batch_size >= len(rows)):
                progress(min(start+batch_size,len(rows)),len(rows))
    return torch.cat(batches)


def select_support(rows, classes, seed):
    if any(r["split"] != "train" or r["is_ai"] != 1 or r["source_label"] not in classes for r in rows):
        raise ValueError("support requires known train-only AI rows")
    selected = []
    for source in classes:
        groups = {}
        for r in rows:
            if r["source_label"] == source:
                groups.setdefault(r["group_id"], []).append(r)
        if len(groups) < 10:
            raise ValueError(f"10 unique train groups required for {source}")
        order = sorted(groups, key=lambda g:(digest([seed,"support-group",source,g]),g))[:10]
        selected.extend(min(groups[g], key=lambda r:(digest([seed,"support-image",r["sample_id"]]),r["sample_id"]))
                        for g in order)
    return selected


def support_manifest(rows, classes, seed):
    selected = select_support(rows,classes,seed)
    return dict(seed=seed,k=10,policy="seed-hash train groups then seed-hash representative; no selection tuning",
        samples=[{k:r[k] for k in ("sample_id","group_id","source_label","split","sha256","pixel_sha256")} for r in selected])


def prototype_vectors(features, support_rows, classes):
    if len(features) != len(support_rows) or len(support_rows) != 10*len(classes):
        raise ValueError("10 supports per class required")
    for source in classes:
        chosen = [r for r in support_rows if r["source_label"]==source]
        if len(chosen) != 10 or len({r["group_id"] for r in chosen}) != 10:
            raise ValueError("10 unique support groups per class required")
    return prototypes(normalized(features,DIMENSION),support_rows,classes)


def prototype_scores(features, centers):
    raw = normalized(features,DIMENSION) @ normalized(centers,DIMENSION).T
    # A fixed monotone conversion only for the shared closed-class score schema.
    return raw.cpu().numpy(), raw.double().softmax(1).cpu().numpy()
