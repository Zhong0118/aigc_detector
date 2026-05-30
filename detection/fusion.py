"""检测分数融合工具。

这个文件用于把多个模态或多个检测器的分数融合成一个最终 AI 概率。
当前 API-first 主流程主要在 `detection/providers.py` 内做简单平均，
但这个 FusionClassifier 仍然保留给后续多模型/多模态组合使用。

后续可能用法：
- text_api + local_text_model 融合
- image_api + local_image_model 融合
- video frame scores 融合
- 多模态文件或混合内容融合

detection 后续还需要补充：
- provider 级权重，而不只是 modality 权重
- 分数校准，例如 Platt scaling / isotonic calibration
- 置信度计算，把 provider 分歧程度也纳入 confidence
- explainability 字段，说明最终分数主要来自哪些 provider
"""

from dataclasses import dataclass
from typing import Dict, Optional

import yaml


@dataclass
class FusionResult:
    """融合后的检测结果。"""

    score: float
    label: str
    modality_scores: Dict[str, float]


class FusionClassifier:
    """分数融合器。

    当前支持：
    - weighted_average: 按 config.yaml 中的权重加权平均
    - max: 取最高分
    """

    def __init__(self, config_path: str = "config.yaml"):
        """读取融合配置。

        config 路径默认是项目根目录下的 config.yaml。
        如果 detection.fusion 没有配置，会使用默认权重。
        """
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        fusion_cfg = cfg.get("detection", {}).get("fusion", {})
        self.method = fusion_cfg.get("method", "weighted_average")
        self.weights = fusion_cfg.get("weights", {
            "text": 0.3, "image": 0.3, "audio": 0.2, "video": 0.2,
        })

    def fuse(self, scores: Dict[str, float], threshold: float = 0.5) -> FusionResult:
        """融合多个分数并输出最终标签。

        scores 形如：
        {"text": 0.72, "image": 0.31}
        """
        if self.method == "weighted_average":
            final_score = self._weighted_average(scores)
        elif self.method == "max":
            final_score = max(scores.values()) if scores else 0.0
        else:
            final_score = self._weighted_average(scores)

        label = "ai" if final_score > threshold else "human"
        return FusionResult(score=final_score, label=label, modality_scores=scores)

    def _weighted_average(self, scores: Dict[str, float]) -> float:
        """按配置权重计算加权平均分。"""
        total_weight = 0.0
        weighted_sum = 0.0

        for modality, score in scores.items():
            w = self.weights.get(modality, 0.0)
            weighted_sum += w * score
            total_weight += w

        if total_weight == 0:
            return 0.0
        return weighted_sum / total_weight
