"""本地文本 AIGC 检测 scaffold。

这个文件目前不是 API-first 主流程的默认路径。
它提供一个本地文本检测器雏形：用语言模型困惑度 perplexity
和句子长度波动 burstiness 做启发式判断。

注意：
- 当前默认模型是 distilgpt2，需要联网/本地缓存模型权重。
- 这不是强可靠的 AIGC 检测器，只适合作为 baseline。
- 后续可以替换为 AdaDetectGPT、Fast-DetectGPT、RADAR 等文本检测模型。
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class DetectionResult:
    """本地 detector 的统一返回结构。"""

    score: float
    label: str
    details: dict


class TextDetector:
    """基于 perplexity + burstiness 的文本检测器。"""

    def __init__(self, model_name: str = "distilgpt2", device: str | None = None):
        """加载 tokenizer 和 causal language model。

        device 未指定时优先使用 CUDA，否则使用 CPU。
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def detect(self, text: str) -> DetectionResult:
        """对文本进行 AI/Human 检测。

        输出 score 越接近 1，越倾向 AI。
        """
        ppl = self._compute_perplexity(text)
        burstiness = self._compute_burstiness(text)

        score = self._score_from_features(ppl, burstiness)
        label = "ai" if score > 0.5 else "human"

        return DetectionResult(
            score=score,
            label=label,
            details={"perplexity": ppl, "burstiness": burstiness},
        )

    def _compute_perplexity(self, text: str) -> float:
        """计算文本困惑度。

        粗略直觉：某些 AI 文本可能更“顺滑”，困惑度更低。
        但这不是绝对规律，所以只能作为一个特征。
        """
        encodings = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = encodings.input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss

        return math.exp(loss.item())

    def _compute_burstiness(self, text: str) -> float:
        """计算句子长度波动。

        burstiness 越低，句子长度越均匀；部分 AI 文本可能更均匀。
        """
        sentences = text.replace("!", ".").replace("?", ".").split(".")
        lengths = [len(s.split()) for s in sentences if s.strip()]
        if len(lengths) < 2:
            return 0.0
        return float(np.std(lengths) / (np.mean(lengths) + 1e-8))

    def _score_from_features(self, ppl: float, burstiness: float) -> float:
        """把 perplexity 和 burstiness 映射成 AI 概率。"""
        # 低困惑度 + 低 burstiness 在这个启发式里更偏 AI。
        ppl_score = 1.0 / (1.0 + math.exp((ppl - 60) / 20))
        burst_score = 1.0 / (1.0 + math.exp((burstiness - 0.5) / 0.2))
        return 0.7 * ppl_score + 0.3 * burst_score
