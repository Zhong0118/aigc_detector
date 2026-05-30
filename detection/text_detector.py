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
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer


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


class RoutedTextAigcDetector:
    """YuchuanTian AIGC_text_detector 的本地路由封装。

    这个 detector 会先判断文本语言和长度，再选择中文/英文、长文/短文模型。
    权重通过 Hugging Face Transformers 加载，并缓存到项目内的 cache_dir。
    """

    def __init__(
        self,
        models: dict[str, str],
        cache_dir: str = "models/huggingface",
        short_text_chars: int = 200,
        threshold: float = 0.5,
        device: str = "auto",
        mixed_strategy: str = "max",
        max_length: int = 512,
    ) -> None:
        """初始化本地文本路由器。

        models 需要包含 zh_long/zh_short/en_long/en_short 四个键。
        cache_dir 建议固定为 models/huggingface，方便后续部署服务器。
        """
        self.models = models
        self.cache_dir = str(Path(cache_dir))
        self.short_text_chars = int(short_text_chars)
        self.threshold = float(threshold)
        self.device = self._resolve_device(device)
        self.mixed_strategy = mixed_strategy
        self.max_length = int(max_length)
        self._loaded: dict[str, tuple[Any, Any]] = {}

    def detect(self, text: str) -> DetectionResult:
        """执行一次本地文本检测。"""
        selected_keys = self.select_model_keys(text)
        branch_scores: dict[str, float] = {}
        branch_details: dict[str, Any] = {}

        for key in selected_keys:
            model_name = self.models[key]
            score = self._run_model(key, model_name, text)
            public_name = f"local-text-{key.replace('_', '-')}"
            branch_scores[public_name] = round(score, 4)
            branch_details[key] = {"model": model_name, "score": round(score, 4)}

        final_score = self._merge_branch_scores(list(branch_scores.values()))
        return DetectionResult(
            score=round(final_score, 4),
            label="ai" if final_score >= self.threshold else "human",
            details={
                "language": self.detect_language(text),
                "length_bucket": self.length_bucket(text),
                "selected_models": selected_keys,
                "cache_dir": self.cache_dir,
                "model_scores": branch_scores,
                "branch_details": branch_details,
                "note": "Local open-source text detector branch with language/length routing.",
            },
        )

    def select_model_keys(self, text: str) -> list[str]:
        """根据语言和长度选择模型键。"""
        language = self.detect_language(text)
        length = self.length_bucket(text)
        if language == "mixed":
            return [f"zh_{length}", f"en_{length}"]
        return [f"{language}_{length}"]

    def detect_language(self, text: str) -> str:
        """用轻量规则判断中文、英文或中英混合。

        为了少引入依赖，这里不用额外语言识别库。
        """
        chinese_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
        english_chars = sum(1 for char in text if char.isascii() and char.isalpha())
        total = chinese_chars + english_chars
        if total == 0:
            return "en"
        zh_ratio = chinese_chars / total
        en_ratio = english_chars / total
        if zh_ratio >= 0.1 and en_ratio >= 0.1:
            return "mixed"
        return "zh" if zh_ratio > en_ratio else "en"

    def length_bucket(self, text: str) -> str:
        """判断短文本或长文本。"""
        return "short" if len(text.strip()) <= self.short_text_chars else "long"

    def _run_model(self, key: str, model_name: str, text: str) -> float:
        """加载并运行 Hugging Face sequence classification 模型。"""
        tokenizer, model = self._load_model(key, model_name)
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {name: value.to(self.device) for name, value in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        return self._ai_probability(probs, model.config.id2label)

    def _load_model(self, key: str, model_name: str) -> tuple[Any, Any]:
        """懒加载 tokenizer/model，并把权重缓存到 cache_dir。"""
        if key not in self._loaded:
            model_source = self._local_snapshot_path(model_name) or model_name
            tokenizer = AutoTokenizer.from_pretrained(model_source, cache_dir=self.cache_dir)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_source,
                cache_dir=self.cache_dir,
            ).to(self.device)
            model.eval()
            self._loaded[key] = (tokenizer, model)
        return self._loaded[key]

    def _local_snapshot_path(self, model_name: str) -> str | None:
        """如果项目缓存里已有模型 snapshot，则优先返回本地路径。

        Hugging Face 的 repo-id 缓存偶尔会因为索引状态导致离线加载不稳定；
        服务器部署时直接用 snapshot 目录更可控。
        """
        if "/" not in model_name:
            return None
        owner, repo = model_name.split("/", 1)
        repo_dir = Path(self.cache_dir) / f"models--{owner}--{repo}"
        refs_main = repo_dir / "refs" / "main"
        if not refs_main.exists():
            return None
        revision = refs_main.read_text(encoding="utf-8").strip()
        snapshot = repo_dir / "snapshots" / revision
        if snapshot.exists():
            return str(snapshot)
        return None

    def _ai_probability(self, probs: torch.Tensor, id2label: dict[int, str]) -> float:
        """从分类器输出里提取 AI 类概率。"""
        normalized_labels = {idx: str(label).lower() for idx, label in id2label.items()}
        for idx, label in normalized_labels.items():
            if any(token in label for token in ["ai", "machine", "generated", "gpt"]):
                return float(probs[int(idx)].item())
        if len(probs) == 1:
            return float(probs[0].item())
        return float(probs[-1].item())

    def _merge_branch_scores(self, scores: list[float]) -> float:
        """合并中英混合文本的多个分支分数。"""
        if not scores:
            return 0.0
        if self.mixed_strategy == "mean":
            return float(sum(scores) / len(scores))
        return float(max(scores))

    def _resolve_device(self, device: str) -> str:
        """解析 device 配置。"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
