from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class AttributionResult:
    top_k: List[Dict[str, float]]
    confidence: float
    features_used: List[str]


class AttributionMLP(nn.Module):
    def __init__(self, input_dim: int = 64, num_models: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_models),
        )

    def forward(self, x):
        return self.net(x)


KNOWN_MODELS = [
    "chatgpt-4", "chatgpt-3.5", "claude-3", "gemini-pro",
    "stable-diffusion-xl", "midjourney-v6", "dall-e-3",
    "elevenlabs", "bark", "suno",
]


class ModelAttribution:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AttributionMLP(
            input_dim=64, num_models=len(KNOWN_MODELS)
        ).to(self.device)
        self.model.eval()

    def attribute(self, features: Dict[str, float], top_k: int = 3) -> AttributionResult:
        feature_vec = self._prepare_features(features)
        tensor = torch.FloatTensor(feature_vec).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        top_indices = torch.topk(probs, min(top_k, len(KNOWN_MODELS))).indices
        results = []
        for idx in top_indices:
            results.append({
                "model": KNOWN_MODELS[idx.item()],
                "probability": probs[idx.item()].item(),
            })

        return AttributionResult(
            top_k=results,
            confidence=results[0]["probability"] if results else 0.0,
            features_used=list(features.keys()),
        )

    def _prepare_features(self, features: Dict[str, float]) -> np.ndarray:
        vec = np.zeros(64, dtype=np.float32)
        for i, (_, val) in enumerate(sorted(features.items())):
            if i >= 64:
                break
            vec[i] = val
        return vec

    def load_weights(self, path: str):
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
