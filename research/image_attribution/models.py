"""Paper architectures; no fallback to random weights or missing pair heads.

OFA architecture adapted from jumpycat/OFAAttribution (MIT), commit 3780cbf.
ResNet50 configuration follows mttry/ImageAttributionBench (MIT), 19bb7d8.
See docs/IMAGE_METHOD_AUDIT.md for preprocessing and protocol differences.
"""
from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms


def resnet_transform():
    return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                               transforms.ToTensor(), transforms.Normalize(
                                   [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class ResNetClassifier(nn.Module):
    def __init__(self, num_classes, pretrained_path=None):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        if pretrained_path:
            self.backbone.load_state_dict(torch.load(pretrained_path, map_location='cpu', weights_only=True))
        self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.fc(self.backbone(x))


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    classes = checkpoint['classes']
    if (checkpoint['task'] == 'detection' and classes != ['real', 'ai']) or (
            checkpoint['task'] == 'attribution' and classes != sorted(set(classes))):
        raise ValueError('invalid class order')
    if checkpoint['task'] not in {'detection', 'attribution'}:
        raise ValueError('unsupported checkpoint task')
    if checkpoint.get('preprocessing') != 'iab-resnet50-v1':
        raise ValueError('unsupported preprocessing identity')
    model = ResNetClassifier(len(classes))
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.to(device).eval(), checkpoint


def load_feature_cache(path, identity):
    cache = torch.load(path, map_location='cpu', weights_only=True)
    if cache['identity'] != identity:
        raise ValueError('feature cache identity mismatch')
    if len(cache['features']) != len(identity['sample_ids']):
        raise ValueError('feature cache sample count mismatch')
    return cache['features']


class PairHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(256, 128), nn.LeakyReLU(.2, inplace=True),
                                nn.Linear(128, 128), nn.LeakyReLU(.2, inplace=True), nn.Linear(128, 1))

    def forward(self, first, second):
        return self.fc(torch.cat([first, second], dim=1)).squeeze(1)


class FrequencyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.resnet50(weights=None)
        self.base.fc = nn.Linear(2048, 128)

    def forward(self, residual):
        return self.base(torch.log1p(torch.abs(torch.fft.fft2(residual))))


def reference_score(pair, support, query):
    """Author uses S_features first, D_features second; concatenation is asymmetric."""
    return pair(support, query.expand(len(support), -1)).sigmoid().mean()


def load_ofa(fingerprint_path, denoiser_path, device):
    checkpoint = torch.load(fingerprint_path, map_location='cpu', weights_only=True)
    if 'pair_classifier' not in checkpoint:
        raise ValueError('complete OFA requires pair_classifier weights')
    encoder, pair = FrequencyEncoder(), PairHead()
    encoder.load_state_dict(checkpoint['model'], strict=True)
    pair.load_state_dict(checkpoint['pair_classifier'], strict=True)
    # Exact author's 17 valid 3x3 convolution layers, BN only in middle layers.
    layers = []
    for i in range(17):
        bn = 0 < i < 16
        layers.append(nn.Conv2d(3 if i == 0 else 64, 3 if i == 16 else 64,
                                3, padding=0, bias=not bn))
        if bn:
            layers.append(nn.BatchNorm2d(64, momentum=.1))
        if i < 16:
            layers.append(nn.ReLU(inplace=True))
    denoiser = nn.Sequential(*layers)
    denoiser.load_state_dict(torch.load(denoiser_path, map_location='cpu', weights_only=True)['network'], strict=True)
    for model in (encoder, pair, denoiser):
        model.to(device).eval().requires_grad_(False)
    return encoder, pair, denoiser


def ofa_transform(image):
    if min(image.size) < 256:
        image = transforms.functional.resize(image, 256)
    image = transforms.functional.center_crop(image, 256)
    # Author normalizes to [-1,1], then reverses it in img2noise; retain both steps.
    return transforms.functional.normalize(transforms.functional.to_tensor(image), [.5]*3, [.5]*3)


def ofa_features(tensor, encoder, denoiser):
    pixels = (tensor.clamp(-1, 1) + 1) / 2
    return encoder(denoiser(pixels) / 256.)
