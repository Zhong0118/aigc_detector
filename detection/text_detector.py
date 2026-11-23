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
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from detection.text_segmentation import SentenceAwareSegmenter
from detection.text_threshold_artifact import model_calibration_group


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


def margin_probability(margin: float, temperature: float) -> float:
    """Stable sigmoid in double precision, with finite-precision endpoints excluded."""
    scaled = margin / temperature
    e = math.exp(-abs(scaled))
    score = 1 / (1 + e) if scaled >= 0 else e / (1 + e)
    return min(math.nextafter(1.0, 0.0), max(math.nextafter(0.0, 1.0), score))


def branch_scoring_protocol(raw_protocol: str, routes: dict[str, str], temperatures: dict[str, float]) -> str:
    payload = [raw_protocol, "executed-weight-calibration-v1", routes, temperatures]
    return "token-logit-v1:" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


class RoutedTextAigcDetector:
    """YuchuanTian AIGC_text_detector 的本地路由封装。

    这个 detector 会先判断文本语言和长度，再选择中文/英文、长文/短文模型。
    权重通过 Hugging Face Transformers 加载，并缓存到项目内的 cache_dir。
    """

    scoring_protocol = "token-complete-v1"

    def __init__(
        self,
        models: dict[str, str],
        cache_dir: str = "models/huggingface",
        short_text_chars: int = 200,
        threshold: float = 0.5,
        device: str = "auto",
        mixed_strategy: str = "max",
        max_length: int = 512,
        segment_target_chars: int = 480,
        segment_max_chars: int = 640,
        en_segment_target_chars: int | None = None,
        en_segment_max_chars: int | None = None,
        segment_overlap_sentences: int = 1,
        aggregation_mode: str = "probability_mean",
        temperature: float = 1.0,
        length_routing: str = "segment",
        branch_calibration_path: str | None = None,
    ) -> None:
        """初始化本地文本路由器。

        models 需要包含 zh_long/zh_short/en_long/en_short 四个键。
        cache_dir 建议固定为 models/huggingface，方便后续部署服务器。
        中文窗口按 BERT 512 token 贴近字符数；英文窗口更大，避免长模型吃不饱。
        """
        self.models = models
        self.cache_dir = str(Path(cache_dir))
        self.short_text_chars = int(short_text_chars)
        if length_routing not in {"segment", "document"}:
            raise ValueError(f"Unknown text length routing: {length_routing}")
        self.length_routing = length_routing
        self.threshold = float(threshold)
        self.device = self._resolve_device(device)
        self.mixed_strategy = mixed_strategy
        self.max_length = int(max_length)
        if aggregation_mode not in {"probability_mean", "logit_mean"}:
            raise ValueError(f"Unknown text aggregation mode: {aggregation_mode}")
        self.aggregation_mode = aggregation_mode
        self.temperature = float(temperature)
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        overlap = int(segment_overlap_sentences)
        self.zh_segmenter = SentenceAwareSegmenter(
            target_chars=int(segment_target_chars),
            max_chars=int(segment_max_chars),
            overlap_sentences=overlap,
        )
        self.en_segmenter = SentenceAwareSegmenter(
            target_chars=int(en_segment_target_chars or 1400),
            max_chars=int(en_segment_max_chars or 1800),
            overlap_sentences=overlap,
        )
        self.segmenter = self.zh_segmenter
        self._loaded: dict[str, tuple[Any, Any]] = {}
        protocol_config = dict(models=self.models, mixed_strategy=mixed_strategy,
                               max_length=self.max_length, short_text_chars=self.short_text_chars,
                               zh_window=[segment_target_chars, segment_max_chars],
                               en_window=[en_segment_target_chars or 1400, en_segment_max_chars or 1800],
                               overlap=overlap)
        if self.length_routing == "document":
            protocol_config.update(length_routing="document", length_rule="non_whitespace_lt")
        protocol_name = "token-complete-v1"
        if self.aggregation_mode == "logit_mean":
            protocol_name = "token-logit-v1"
            protocol_config.update(aggregation_mode=self.aggregation_mode, temperature=self.temperature)
        self.scoring_protocol = protocol_name + ":" + hashlib.sha256(
            json.dumps(protocol_config, sort_keys=True).encode()
        ).hexdigest()[:16]
        self.model_routes: dict[str, str] = {}
        self.branch_temperatures: dict[str, float] = {}
        if branch_calibration_path:
            self._load_branch_calibration(Path(branch_calibration_path))

    def _load_branch_calibration(self, path: Path) -> None:
        if self.aggregation_mode != "logit_mean" or self.length_routing != "document" or self.temperature != 1.0:
            raise ValueError("Branch calibration requires document logit scoring with a unit base temperature")
        fitted = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(fitted, dict) or fitted.get("raw_scoring_protocol") != self.scoring_protocol:
            raise ValueError("Branch calibration does not match the raw scoring protocol")
        routes = fitted.get("model_routes")
        groups = fitted.get("groups")
        if (not isinstance(routes, dict) or set(routes) != set(self.models)
                or any(not isinstance(value, str) or not value or "+" in value for value in routes.values())
                or not isinstance(groups, dict) or not groups or not groups.keys() <= set(routes.values())):
            raise ValueError("Invalid branch calibration routes or groups")
        weights_by_group: dict[str, set[str]] = {}
        for key, group in routes.items():
            weights_by_group.setdefault(group, set()).add(self.models[key])
        if any(len(weights) != 1 for weights in weights_by_group.values()):
            raise ValueError("Branch calibration cannot share one temperature between distinct weights")
        temperatures = {}
        for group, evidence in groups.items():
            temperature = evidence.get("temperature") if isinstance(evidence, dict) else None
            if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
                raise ValueError(f"Invalid calibration temperature for {group}")
            temperatures[group] = temperature
        protocol = branch_scoring_protocol(self.scoring_protocol, routes, temperatures)
        if fitted.get("scoring_protocol") != protocol:
            raise ValueError("Branch calibration scoring protocol does not match its temperatures")
        self.model_routes, self.branch_temperatures, self.scoring_protocol = routes, temperatures, protocol

    def detect(self, text: str) -> DetectionResult:
        """执行一次本地文本检测。"""
        language = self.detect_language(text)
        segments = self._segmenter_for(language).split(text)
        branch_score_lists: dict[str, list[float]] = {}
        branch_details: dict[str, Any] = {}
        segment_details: list[dict[str, Any]] = []
        weighted_scores: list[tuple[float, int]] = []
        previous_end = 0
        use_margin = self.aggregation_mode == "logit_mean"
        # Keep the document's route for each segment, including short tails.
        document_keys = self.select_model_keys(text) if self.length_routing == "document" else None
        group = model_calibration_group({"selected_models": document_keys}, self.model_routes)
        temperature = self.branch_temperatures.get(group, self.temperature)

        for segment in segments:
            selected_keys = document_keys if document_keys is not None else self.select_model_keys(segment.text)
            segment_branch_scores: dict[str, float] = {}
            for key in selected_keys:
                model_name = self.models[key]
                score = self._run_model(key, model_name, segment.text)
                public_name = f"local-text-{key.replace('_', '-')}"
                segment_branch_scores[public_name] = score if use_margin else round(score, 4)
                branch_score_lists.setdefault(public_name, []).append(score)
                branch_details[key] = {"model": model_name}

            segment_score = self._merge_branch_scores(list(segment_branch_scores.values()))
            unique_start = max(segment.start, previous_end)
            weight = max(0 if use_margin else 1, segment.end - unique_start)
            previous_end = max(previous_end, segment.end)
            weighted_scores.append((segment_score, weight))
            segment_details.append(
                {
                    "index": segment.index,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "score": self._as_probability(segment_score, temperature),
                    "model_scores": {name: self._as_probability(value, temperature) for name, value in segment_branch_scores.items()},
                    **({"raw_margin": segment_score} if use_margin else {}),
                }
            )

        total_weight = sum(weight for _, weight in weighted_scores)
        final_score = (
            sum(score * weight for score, weight in weighted_scores) / total_weight
            if total_weight
            else 0.0
        )
        branch_scores = {
            name: self._as_probability(sum(scores) / len(scores), temperature)
            for name, scores in branch_score_lists.items()
            if scores
        }
        probability = self._as_probability(final_score, temperature)
        return DetectionResult(
            score=probability,
            label="ai" if (probability if use_margin else final_score) >= self.threshold else "human",
            details={
                "language": language,
                "length_bucket": self.length_bucket(text),
                "effective_chars": sum(not char.isspace() for char in text),
                "selected_models": sorted(branch_details),
                "cache_dir": self.cache_dir,
                "model_scores": branch_scores,
                "screening_threshold": self.threshold,
                "branch_details": branch_details,
                "segments": segment_details,
                "aggregation": "unique-character-weighted-logit-mean" if use_margin else "unique-character-weighted-mean",
                "scoring_protocol": self.scoring_protocol,
                **({"aggregation_mode": self.aggregation_mode, "raw_margin": final_score,
                    "temperature": temperature} if use_margin else {}),
                "note": "Local open-source text detector branch with language/length routing.",
            },
        )

    def _segmenter_for(self, language: str) -> SentenceAwareSegmenter:
        """英文用更大窗口；中文和中英混合仍用接近 512 token 的中文窗口。"""
        if language == "en":
            return self.en_segmenter
        return self.zh_segmenter

    def select_model_keys(self, text: str) -> list[str]:
        """根据语言和长度选择模型键。"""
        language = self.detect_language(text)
        length = self.length_bucket(text)
        if language == "mixed":
            if self.mixed_strategy == "dominant":
                chinese = sum("\u4e00" <= char <= "\u9fff" for char in text)
                english = sum(char.isascii() and char.isalpha() for char in text)
                return [f"{'zh' if chinese >= english else 'en'}_{length}"]
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
        if self.length_routing == "document":
            return "short" if sum(not char.isspace() for char in text) < self.short_text_chars else "long"
        return "short" if len(text.strip()) <= self.short_text_chars else "long"

    def _run_model(self, key: str, model_name: str, text: str) -> float:
        """Return the token-weighted segment value in the configured score space."""
        tokenizer, model = self._load_model(key, model_name)
        limit = min(
            self.max_length,
            getattr(model.config, "max_position_embeddings", self.max_length),
            tokenizer.model_max_length,
        )
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=limit,
            return_overflowing_tokens=True,
            padding=True,
        )
        encoded.pop("overflow_to_sample_mapping", None)
        special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
        weighted_score = 0.0
        total_tokens = 0

        with torch.no_grad():
            for index in range(encoded["input_ids"].shape[0]):
                window = {name: value[index:index + 1].to(self.device) for name, value in encoded.items()}
                outputs = model(**window)
                content_tokens = int(window["attention_mask"].sum().item()) - special_tokens
                logits = outputs.logits[0]
                if self.aggregation_mode == "logit_mean":
                    if logits.numel() != 2:
                        raise ValueError("logit_mean requires two output classes")
                    ai = self._ai_label_index(model.config.id2label, 2)
                    value = float(logits[ai].item()) - float(logits[1 - ai].item())
                    if not math.isfinite(value):
                        raise ValueError("Non-finite binary classifier margin")
                    weight = max(0, content_tokens)
                else:
                    probs = torch.softmax(outputs.logits, dim=-1)[0]
                    value = self._ai_probability(probs, model.config.id2label)
                    weight = max(1, content_tokens)
                weighted_score += value * weight
                total_tokens += weight

        return weighted_score / total_tokens if total_tokens else 0.0

    def _as_probability(self, value: float, temperature: float | None = None) -> float:
        if self.aggregation_mode == "logit_mean":
            return margin_probability(value, self.temperature if temperature is None else temperature)
        return round(value, 4)

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
        return float(probs[self._ai_label_index(id2label, len(probs))].item())

    @staticmethod
    def _ai_label_index(id2label: dict[int, str], size: int) -> int:
        normalized_labels = {idx: str(label).lower() for idx, label in id2label.items()}
        for idx, label in normalized_labels.items():
            if any(token in label for token in ["ai", "machine", "generated", "gpt"]):
                return int(idx)
        return size - 1

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
