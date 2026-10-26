"""Local-only frozen DINOv2-S/14 features for an independently trained head."""

from pathlib import Path

import torch
from transformers import AutoImageProcessor, AutoModel


class _FrozenDinov2(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # A parent training loop must not activate dropout in the frozen backbone.
        return super().train(False)

    @torch.no_grad()
    def forward(self, pixel_values):
        # no_grad produces ordinary tensors that a trainable head can save.
        return self.model(pixel_values=pixel_values).last_hidden_state[:, 0, :]


class _PILTransform:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, image):
        return self.processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"][0]


def load_dinov2(local_dir, device):
    """Return ``(encoder, PIL_transform, metadata)`` from an existing local checkpoint.

    Requires the official DINOv2-small config, processor config and safetensors.
    The caller owns downloading/pinning weights and moving image batches to device.
    Outputs are unnormalized 384-dimensional CLS features; no detector is pretrained.
    Missing, incompatible or corrupt files raise without network/random fallback.
    """
    directory = Path(local_dir).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Local DINOv2 directory does not exist: {directory}")

    model, loading_info = AutoModel.from_pretrained(
        str(directory),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        output_loading_info=True,
    )
    if any(loading_info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Incompatible or incomplete DINOv2 checkpoint: {loading_info}")
    if model.config.model_type != "dinov2" or model.config.hidden_size != 384:
        raise ValueError("Expected DINOv2-small with 384-dimensional CLS features")
    processor = AutoImageProcessor.from_pretrained(
        str(directory),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=False,
    )
    encoder = _FrozenDinov2(model).to(device)
    metadata = {
        "name": "dinov2_small_frozen",
        "local_dir": str(directory),
        "feature_dim": 384,
        "feature_type": "cls",
        "l2_normalized": False,
        "preprocessing": processor.to_dict(),
    }
    return encoder, _PILTransform(processor), metadata
