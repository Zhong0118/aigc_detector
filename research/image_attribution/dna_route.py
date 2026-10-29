"""Independent frozen-feature adapter for the published DNA-Det architecture.

Architecture/checkpoint names and image protocol are documented at:
https://github.com/ICTMCG/DNA-Det/tree/d811395b92fb22d5f82f06ee98dbde77e53cea69
(models/models.py, data/dataset.py, configs/in_the_wild.py).
This implements standard PyTorch blocks; upstream code/weight licensing is unestablished.
It does not implement DNA-Det contrastive training or reuse its 11-class predictions.
"""
from collections import OrderedDict
from pathlib import Path
import re

from PIL import Image
import torch
from torch import nn
from torchvision import transforms

from research.image_attribution.openclip_route import digest, image_tensor, normalized
from research.image_attribution.protocol import sha256

DIMENSION = 512
MODEL = "DNA-Det-in-the-wild-frozen-GAP"
PREPROCESSING = dict(rgb=True, nonsquare_center_crop=[128, 128],
    resize_sizes=[[128, 128], [512, 512]], interpolation="PIL-BICUBIC",
    mean=[.5, .5, .5], std=[.5, .5, .5], whole_image=True,
    feature_layer="GAP before classifier", feature_normalization="l2")


class DNAEncoder(nn.Module):
    """Construct the complete original classifier so loading remains strict."""
    def __init__(self):
        super().__init__()
        blocks, channels = [], 3
        for i, width in enumerate((64, 64, 128, 128, 256, 256, 512, 512)):
            downsample = i % 2 == 0
            block = nn.Sequential(nn.Conv2d(channels, width, 4 if downsample else 3,
                                            2 if downsample else 1, 1),
                                  nn.BatchNorm2d(width), nn.LeakyReLU(.2))
            blocks.append(nn.Sequential(OrderedDict([("main", block)])))
            channels = width
        self.main = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classification_head = nn.Sequential(nn.Dropout(.2, inplace=True), nn.Linear(512, 11))

    def forward(self, images):
        return self.pool(self.main(images)).flatten(1)


def preprocess(image):
    image = image.convert("RGB")
    if image.width != image.height:
        image = transforms.functional.center_crop(image, [128, 128])
    image = image.resize((128, 128), Image.Resampling.BICUBIC)
    image = image.resize((512, 512), Image.Resampling.BICUBIC)
    return transforms.functional.normalize(transforms.functional.to_tensor(image), [.5]*3, [.5]*3)


def verify_asset(root, asset):
    expected = asset.get("sha256")
    if not isinstance(expected, str) or re.fullmatch("[0-9a-f]{64}", expected) is None:
        raise ValueError("unverified asset SHA; import and CPU verification required")
    root = Path(root).resolve()
    relative = Path(asset["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("asset path escapes workspace")
    path = (root / relative).resolve()
    # The existing asset directory is shared across research worktrees.
    shared = ((root/"outputs").resolve(), (root/"outputs/image_experiments/assets").resolve())
    if not path.is_relative_to(root) and not (
            relative.parts[0] == "outputs" and any(path.is_relative_to(p) for p in shared)):
        raise ValueError("asset path escapes workspace")
    if path.stat().st_size != asset["size"] or sha256(path) != expected:
        raise ValueError("official asset incomplete or identity changed")
    return path


def load_backbone(root, config, device):
    path = verify_asset(root, config["asset"])
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(x, torch.Tensor) for x in state.values()):
        raise ValueError("plain tensor state_dict required")
    model = DNAEncoder()
    model.load_state_dict(state, strict=True)
    # The original 11-class head is checked above and never used for q179 scores.
    model.classification_head = nn.Identity()
    return model.eval().requires_grad_(False).to(device), preprocess


def cache_identity(rows, config):
    if any(r["split"] not in ("train", "dev") for r in rows):
        raise ValueError("holdout forbidden in feature cache")
    if any(r["is_ai"] != 1 for r in rows):
        raise ValueError("DNA attribution cache requires AI rows only")
    ids = [r["sample_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate feature cache IDs")
    if config["preprocessing"] != PREPROCESSING:
        raise ValueError("preprocessing changed")
    return dict(schema="dna-frozen-cache-v1", model=MODEL, dimensions=DIMENSION,
        preprocessing=PREPROCESSING, ids=ids,
        rows_sha256=digest([{k:r[k] for k in ("sample_id", "group_id", "relative_path",
            "sha256", "pixel_sha256", "split", "is_ai", "source_label", "semantic_class")} for r in rows]),
        asset=config["asset"], dependencies=config["dependencies"],
        code=config["implementation_sha256"], manifest_sha256=config["manifest_sha256"],
        selection_sha256=config["selection_sha256"], seed=config["seed"])


def encode_rows(model, transform, data, rows, device, batch_size=32, progress=None):
    if not rows or batch_size < 1:
        raise ValueError("nonempty rows and positive batch size required")
    if any(r["split"] not in ("train", "dev") for r in rows):
        raise ValueError("holdout image access forbidden")
    if any(r["is_ai"] != 1 for r in rows):
        raise ValueError("DNA attribution requires AI rows")
    batches = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            images = torch.stack([image_tensor(data, r, transform) for r in rows[start:start+batch_size]])
            batches.append(normalized(model(images.to(device)), DIMENSION).cpu())
            if progress and (start % (batch_size*32) == 0 or start+batch_size >= len(rows)):
                progress(min(start+batch_size, len(rows)), len(rows))
    return torch.cat(batches)
