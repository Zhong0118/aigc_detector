from dataclasses import dataclass
from typing import Dict, Optional

import yaml


@dataclass
class FusionResult:
    score: float
    label: str
    modality_scores: Dict[str, float]


class FusionClassifier:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        fusion_cfg = cfg.get("detection", {}).get("fusion", {})
        self.method = fusion_cfg.get("method", "weighted_average")
        self.weights = fusion_cfg.get("weights", {
            "text": 0.3, "image": 0.3, "audio": 0.2, "video": 0.2,
        })

    def fuse(self, scores: Dict[str, float], threshold: float = 0.5) -> FusionResult:
        if self.method == "weighted_average":
            final_score = self._weighted_average(scores)
        elif self.method == "max":
            final_score = max(scores.values()) if scores else 0.0
        else:
            final_score = self._weighted_average(scores)

        label = "ai" if final_score > threshold else "human"
        return FusionResult(score=final_score, label=label, modality_scores=scores)

    def _weighted_average(self, scores: Dict[str, float]) -> float:
        total_weight = 0.0
        weighted_sum = 0.0

        for modality, score in scores.items():
            w = self.weights.get(modality, 0.0)
            weighted_sum += w * score
            total_weight += w

        if total_weight == 0:
            return 0.0
        return weighted_sum / total_weight
