"""Offline frozen OpenCLIP features and train-only nonparametric estimators."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score

from research.image_attribution.followups import partition_rows, choose_threshold, wilson
from research.image_attribution.protocol import sha256

DIMENSION = 768
PREPROCESSING = dict(size=224, resize_mode="shortest", interpolation="bicubic",
                     mean=[0.48145466, 0.4578275, 0.40821073],
                     std=[0.26862954, 0.26130258, 0.27577711], crop="center", rgb=True)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def normalized(value, dimensions=None):
    if (not isinstance(value, torch.Tensor) or value.ndim != 2 or
            not torch.is_floating_point(value) or not torch.isfinite(value).all() or
            (value.norm(dim=1) <= 1e-12).any() or
            (dimensions is not None and value.shape[1] != dimensions)):
        raise ValueError("invalid, zero or nonfinite feature vectors")
    return F.normalize(value.float(), dim=1)


def cache_identity(rows, assets, code):
    if any(row["split"] == "holdout" for row in rows):
        raise ValueError("holdout forbidden in feature cache")
    ids = [row["sample_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate feature IDs")
    return dict(schema="openclip-frozen-cache-v1", rows_sha256=digest([
        {k: row[k] for k in ("sample_id", "relative_path", "sha256", "split")} for row in rows]),
        ids=ids, assets=assets, code=code, preprocessing=PREPROCESSING,
        model="ViT-L-14-quickgelu", projected_dimensions=DIMENSION)


def write_cache(path, features, identity):
    features = normalized(features)
    if len(features) != len(identity["ids"]):
        raise ValueError("feature count mismatch")
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    torch.save(dict(identity=identity, features=features.cpu()), temporary)
    temporary.rename(path)


def read_cache(path, identity, dimensions=DIMENSION):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.get("identity") != identity:
        raise ValueError("feature cache identity mismatch")
    features = normalized(value["features"], dimensions)
    if len(features) != len(identity["ids"]):
        raise ValueError("feature count mismatch")
    return features


def job_parts(rows, experiment, seed):
    parts = partition_rows(rows, dict(experiment, seed=seed))
    classes = (["real", "ai"] if experiment["task"] == "detection"
               else sorted({r["source_label"] for r in parts["train"]}))
    if any(not values for values in parts.values()):
        raise ValueError("all development roles must be populated")
    if experiment["task"] == "attribution":
        for role in ("train", "selection"):
            if any(r["source_label"] not in classes for r in parts[role]):
                raise ValueError("unknown label in training or selection")
    return parts, classes


def prototypes(features, rows, classes):
    if any(r["split"] != "train" or r["source_label"] not in classes for r in rows):
        raise ValueError("prototypes require known train rows only")
    x = normalized(features)
    means = []
    for name in classes:
        indices = [i for i, row in enumerate(rows) if row["source_label"] == name]
        if not indices:
            raise ValueError("missing train prototype class")
        means.append(x[indices].mean(0))
    return normalized(torch.stack(means))


def knn_neighbor_labels(query, reference, labels, n_classes, k, chunk=256):
    query, reference = normalized(query), normalized(reference)
    if not 1 <= k <= len(reference) or len(reference) != len(labels):
        raise ValueError("invalid KNN reference or k")
    if labels.dtype != torch.long or (labels < 0).any() or (labels >= n_classes).any():
        raise ValueError("invalid KNN labels")
    outputs = []
    for x in query.split(chunk):
        # Stable reference-order tie breaking; reference rows are sorted sample IDs.
        idx = torch.argsort(x @ reference.T, dim=1, descending=True, stable=True)[:, :k]
        outputs.append(labels[idx])
    return torch.cat(outputs)


def knn_probabilities(query, reference, labels, n_classes, k):
    neighbors = knn_neighbor_labels(query, reference, labels, n_classes, k)
    return F.one_hot(neighbors, num_classes=n_classes).float().mean(1).numpy()


def fit_rejection_threshold(calibration_rows, classes, confidence, target):
    confidence = np.asarray(confidence, dtype=np.float64)
    if confidence.shape != (len(calibration_rows),) or not np.isfinite(confidence).all():
        raise ValueError("invalid calibration confidence")
    unknown = np.array([r["source_label"] not in classes for r in calibration_rows])
    return choose_threshold(confidence[unknown], target)


def rejection_report(rows, classes, confidence, predictions, threshold):
    confidence = np.asarray(confidence, dtype=np.float64)
    known = np.array([r["source_label"] in classes for r in rows])
    if known.all() or not known.any() or not np.isfinite(confidence).all():
        raise ValueError("finite scores and both known/unknown truth required")
    accepted = confidence >= threshold
    correct = np.array([classes[int(p)] == r["source_label"] for p, r in zip(predictions, rows)])
    result = dict(n=len(rows), known_n=int(known.sum()), unknown_n=int((~known).sum()),
                  acceptance_threshold=float(threshold), unknown_auroc=float(roc_auc_score(~known, -confidence)),
                  known_closed_set_accuracy=float(correct[known].mean()),
                  score_note="Rejection evidence, not a calibrated probability of unknown origin",
                  uncertainty_note="Wilson intervals treat images as independent; groups may be correlated")
    for name, mask, successes in [
        ("known_retention", known, accepted),
        ("known_correct_and_retained", known, accepted & correct),
        ("unknown_false_acceptance", ~known, accepted),
        ("unknown_rejection", ~known, ~accepted),
    ]:
        n, count = int(mask.sum()), int(successes[mask].sum())
        result[name] = count / n
        result[name + "_wilson95"] = wilson(count, n)
    result["per_source"] = {source: dict(n=sum(r["source_label"] == source for r in rows),
        acceptance_rate=float(accepted[[r["source_label"] == source for r in rows]].mean()))
        for source in sorted({r["source_label"] for r in rows})}
    return result


def load_backbone(root, config, device):
    # Import only after explicit execution; never call hub/pretrained download functions.
    import open_clip
    from safetensors.torch import load_file
    for package, expected in config["dependencies"].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError(f"dependency version mismatch: {package}")
    asset = config["asset"]
    path = Path(root) / asset["path"]
    if path.stat().st_size != asset["size"] or sha256(path) != asset["sha256"]:
        raise ValueError("official OpenCLIP asset identity mismatch")
    model = open_clip.create_model("ViT-L-14-quickgelu", pretrained=None, device="cpu")
    model.load_state_dict(load_file(str(path), device="cpu"), strict=True)
    if model.visual.output_dim != DIMENSION:
        raise ValueError("unexpected projected feature dimensions")
    visual = model.visual.eval().requires_grad_(False).to(device)
    transform = open_clip.image_transform(
        image_size=224, is_train=False, mean=PREPROCESSING["mean"],
        std=PREPROCESSING["std"], interpolation="bicubic", resize_mode="shortest")
    return visual, transform


def image_tensor(data, row, transform):
    if row["split"] == "holdout":
        raise ValueError("holdout image access forbidden")
    data = Path(data).resolve()
    path = (data / row["relative_path"]).resolve()
    if not path.is_relative_to(data) or sha256(path) != row["sha256"]:
        raise ValueError("image path or SHA mismatch")
    with Image.open(path) as image:
        return transform(image.convert("RGB"))


def encode_rows(model, transform, data, rows, device, batch_size=32):
    batches = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            x = torch.stack([image_tensor(data, row, transform) for row in rows[start:start + batch_size]])
            batches.append(normalized(model(x.to(device)), DIMENSION).cpu())
    return torch.cat(batches)
