"""CPU tests: mock checkpoint loading; never download or run a dataset."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from PIL import Image

from research.image_attribution import vision_backbones


class TinyBackbone(torch.nn.Module):
    def __init__(self, hidden_size=384):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(3.0))
        self.dropout = torch.nn.Dropout(0.9)
        self.config = SimpleNamespace(model_type="dinov2", hidden_size=hidden_size)
        self.grad_enabled = None

    def forward(self, *, pixel_values):
        self.grad_enabled = torch.is_grad_enabled()
        cls = self.scale * torch.ones(pixel_values.shape[0], 1, self.config.hidden_size)
        patches = torch.full_like(cls, 9.0)
        return SimpleNamespace(last_hidden_state=torch.cat((cls, patches), dim=1))


@pytest.fixture
def loaders(tmp_path):
    backbone = TinyBackbone()
    processor = Mock()
    processor.to_dict.return_value = {"crop_size": {"height": 224, "width": 224}}
    processor.return_value = {"pixel_values": torch.full((1, 3, 224, 224), 0.25)}
    loading_info = {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []}
    with patch.object(vision_backbones.AutoModel, "from_pretrained", return_value=(backbone, loading_info)) as model_loader:
        with patch.object(vision_backbones.AutoImageProcessor, "from_pretrained", return_value=processor) as processor_loader:
            yield tmp_path, backbone, processor, model_loader, processor_loader


def test_loader_is_local_only_and_requires_safetensors(loaders):
    directory, _, _, model_loader, processor_loader = loaders
    _, _, metadata = vision_backbones.load_dinov2(directory, "cpu")

    model_loader.assert_called_once_with(
        str(directory), local_files_only=True, trust_remote_code=False,
        use_safetensors=True, output_loading_info=True
    )
    processor_loader.assert_called_once_with(
        str(directory), local_files_only=True, trust_remote_code=False, use_fast=False
    )
    assert metadata["feature_dim"] == 384
    assert metadata["l2_normalized"] is False


def test_encoder_stays_frozen_and_returns_raw_cls_usable_by_a_trainable_head(loaders):
    directory, backbone, _, _, _ = loaders
    encoder, _, _ = vision_backbones.load_dinov2(directory, "cpu")
    encoder.train()
    assert not encoder.training
    assert not backbone.training
    assert not any(parameter.requires_grad for parameter in encoder.parameters())

    features = encoder(torch.zeros(2, 3, 224, 224, requires_grad=True))
    assert features.shape == (2, 384)
    torch.testing.assert_close(features, torch.full((2, 384), 3.0))
    assert features.requires_grad is False
    assert backbone.grad_enabled is False
    head = torch.nn.Linear(384, 1)
    head(features).sum().backward()
    assert head.weight.grad is not None


def test_transform_converts_rgb_and_removes_only_batch_axis(loaders):
    directory, _, processor, _, _ = loaders
    _, transform, metadata = vision_backbones.load_dinov2(directory, "cpu")
    pixels = transform(Image.new("L", (400, 240)))

    assert pixels.shape == (3, 224, 224)
    torch.testing.assert_close(pixels, torch.full((3, 224, 224), 0.25))
    assert processor.call_args.kwargs["images"].mode == "RGB"
    assert processor.call_args.kwargs["return_tensors"] == "pt"
    assert metadata["preprocessing"]["crop_size"] == {"height": 224, "width": 224}


def test_missing_local_directory_does_not_call_loaders(loaders):
    directory, _, _, model_loader, processor_loader = loaders
    with pytest.raises(FileNotFoundError):
        vision_backbones.load_dinov2(directory / "missing", "cpu")
    model_loader.assert_not_called()
    processor_loader.assert_not_called()


def test_missing_checkpoint_error_propagates_without_fallback(loaders):
    directory, _, _, model_loader, _ = loaders
    model_loader.side_effect = OSError("no local safetensors checkpoint")
    with pytest.raises(OSError, match="safetensors"):
        vision_backbones.load_dinov2(directory, "cpu")
    assert model_loader.call_count == 1


def test_non_small_backbone_is_rejected(loaders):
    directory, _, _, model_loader, _ = loaders
    model_loader.return_value = (TinyBackbone(hidden_size=768), {})
    with pytest.raises(ValueError, match="384"):
        vision_backbones.load_dinov2(directory, "cpu")


def test_incomplete_checkpoint_cannot_leave_randomly_initialized_parameters(loaders):
    directory, backbone, _, model_loader, _ = loaders
    model_loader.return_value = (backbone, {"missing_keys": ["encoder.layer.0.weight"]})
    with pytest.raises(ValueError, match="checkpoint"):
        vision_backbones.load_dinov2(directory, "cpu")
