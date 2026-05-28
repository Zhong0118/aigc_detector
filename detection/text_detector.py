import math
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class DetectionResult:
    score: float
    label: str
    details: dict


class TextDetector:
    def __init__(self, model_name: str = "distilgpt2", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def detect(self, text: str) -> DetectionResult:
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
        encodings = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = encodings.input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss

        return math.exp(loss.item())

    def _compute_burstiness(self, text: str) -> float:
        sentences = text.replace("!", ".").replace("?", ".").split(".")
        lengths = [len(s.split()) for s in sentences if s.strip()]
        if len(lengths) < 2:
            return 0.0
        return float(np.std(lengths) / (np.mean(lengths) + 1e-8))

    def _score_from_features(self, ppl: float, burstiness: float) -> float:
        # Low perplexity + low burstiness -> more likely AI
        ppl_score = 1.0 / (1.0 + math.exp((ppl - 60) / 20))
        burst_score = 1.0 / (1.0 + math.exp((burstiness - 0.5) / 0.2))
        return 0.7 * ppl_score + 0.3 * burst_score
