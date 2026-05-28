from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


@dataclass
class DetectionResult:
    score: float
    label: str
    details: dict


class ImageDetector:
    def __init__(self, model_name: str = "efficientnet_v2_s", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = self._build_model()
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _build_model(self) -> nn.Module:
        if self.model_name == "efficientnet_v2_s":
            model = models.efficientnet_v2_s(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        else:
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 2)
        return model.to(self.device)

    def detect(self, image: Image.Image) -> DetectionResult:
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            ai_prob = probs[0, 1].item()

        pixel_stats = self._analyze_pixel_artifacts(image)

        label = "ai" if ai_prob > 0.5 else "human"
        return DetectionResult(
            score=ai_prob,
            label=label,
            details={"model_prob": ai_prob, **pixel_stats},
        )

    def _analyze_pixel_artifacts(self, image: Image.Image) -> dict:
        arr = np.array(image).astype(np.float32)
        return {
            "mean_intensity": float(arr.mean()),
            "std_intensity": float(arr.std()),
            "high_freq_energy": self._high_freq_energy(arr),
        }

    def _high_freq_energy(self, arr: np.ndarray) -> float:
        gray = np.mean(arr, axis=2) if arr.ndim == 3 else arr
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        r = min(h, w) // 4
        mask = np.ones_like(gray, dtype=bool)
        y, x = np.ogrid[:h, :w]
        mask[(y - cy) ** 2 + (x - cx) ** 2 <= r ** 2] = False
        high_freq = np.abs(fft_shift[mask])
        return float(np.mean(high_freq))

    def load_weights(self, path: str):
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
