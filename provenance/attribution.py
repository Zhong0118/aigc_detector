"""模型来源归因 scaffold。

这个文件尝试回答：“如果内容是 AI 生成的，它可能来自哪个模型/模型家族？”

当前实现只是一个未训练的 MLP scaffold：
- 没有训练权重时不能作为真实归因依据
- 输入 features 是简单 dict
- 输出 KNOWN_MODELS 上的 Top-K 概率

后续建议：
- 使用真实检测 API 的 generator scores 作为主来源
- 训练多模态 attribution 模型
- 按 text/image/audio/video 分别维护候选模型列表
- 保存训练数据版本和模型版本
"""

from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class AttributionResult:
    """模型归因输出结果。"""

    top_k: List[Dict[str, float]]
    confidence: float
    features_used: List[str]


class AttributionMLP(nn.Module):
    """简单 MLP 归因模型。

    输入固定长度特征向量，输出已知模型列表上的 logits。
    """

    def __init__(self, input_dim: int = 64, num_models: int = 10):
        """构建 MLP 网络结构。"""
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
        """前向传播。"""
        return self.net(x)


KNOWN_MODELS = [
    "chatgpt-4", "chatgpt-3.5", "claude-3", "gemini-pro",
    "stable-diffusion-xl", "midjourney-v6", "dall-e-3",
    "elevenlabs", "bark", "suno",
]


class ModelAttribution:
    """模型来源归因器 scaffold。"""

    def __init__(self, device: str | None = None):
        """初始化归因模型。"""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AttributionMLP(
            input_dim=64, num_models=len(KNOWN_MODELS)
        ).to(self.device)
        self.model.eval()

    def attribute(self, features: Dict[str, float], top_k: int = 3) -> AttributionResult:
        """根据特征输出 Top-K 可能来源模型。

        注意：未加载训练权重时，输出没有实际可信度。
        """
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
        """把 dict 特征整理成固定长度 64 维向量。"""
        vec = np.zeros(64, dtype=np.float32)
        for i, (_, val) in enumerate(sorted(features.items())):
            if i >= 64:
                break
            vec[i] = val
        return vec

    def load_weights(self, path: str):
        """加载训练好的归因模型权重。"""
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
