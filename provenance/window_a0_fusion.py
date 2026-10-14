"""Production family attribution: window-split A0 seed42 + B0 + B1 + B3 equal mix.

Official neural weights are the window-split A0 closed seed-42 run. Seed 62 is
kept only as a research control and is never loaded here. A1–A4 are not used.
"""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from provenance.high_score_blocks import unit_weight
from research.text_attribution.classical_models import predict_family_probabilities
from research.text_attribution.hierarchical_model import load_phase3_model
from research.text_attribution.phase3_data import tokenize_phase3_windows
from research.text_attribution.window_split import split_token_ids


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_SEED = 42
CONTROL_SEED = 62
FAMILY_LABELS = (
    "openai",
    "claude",
    "gemini",
    "deepseek",
    "qwen",
    "glm",
    "minimax",
    "kimi",
    "doubao",
    "mimo",
)
UNKNOWN_FAMILY = "unknown"
DEFAULT_BRANCH_WEIGHTS = {"A0": 0.25, "B0": 0.25, "B1": 0.25, "B3": 0.25}
WEIGHTS_ARTIFACT = (
    PROJECT_ROOT / "outputs/attribution/family-window-split/fusion/seed42-weights.json"
)
# Closed 10-way softmax always picks a family. Product unknown is only used
# when Top-1 itself is below this floor. Do not invent unknown = 1 - Top-1.
DEFAULT_MIN_FAMILY_PROBABILITY = 0.20
DEFAULT_PATHS = {
    "a0_model_dir": PROJECT_ROOT
    / "outputs/attribution/family-window-split/A0-closed-seed42/model",
    "a0_tokenizer_dir": PROJECT_ROOT
    / "outputs/attribution/family-window-split/A0-closed-seed42/tokenizer",
    "b0_model_path": PROJECT_ROOT
    / "outputs/attribution/family-phase4/fast-track-v1/classical/closed-seed42/B0/model.joblib",
    "b1_model_path": PROJECT_ROOT
    / "outputs/attribution/family-phase4/fast-track-v1/classical/closed-seed42/B1/model.joblib",
    "b3_model_path": PROJECT_ROOT
    / "outputs/attribution/family-phase1/final/models/B3.joblib",
}


class WindowA0EqualFusion:
    """Mix window A0, B0, B1, and Phase1 B3 on one text unit.

    Weights default to equal 0.25; deployment may supply explicit validated weights.
    """

    def __init__(self, cfg: Mapping[str, Any] | None = None) -> None:
        self.cfg = dict(cfg or {})
        self.family_labels = tuple(self.cfg.get("family_labels") or FAMILY_LABELS)
        self.top_k = int(self.cfg.get("top_k", 5))
        self.max_length = int(self.cfg.get("max_length", 512))
        self.device_name = str(self.cfg.get("device") or "auto")
        self.min_family_probability = float(
            self.cfg.get("min_family_probability", DEFAULT_MIN_FAMILY_PROBABILITY)
        )
        self.branch_weights = load_branch_weights(self.cfg.get("weights"))
        self._loaded = False
        self._a0 = None
        self._tokenizer = None
        self._classical: dict[str, Any] = {}
        self._torch = None

    def predict_units(self, units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Score each attribution unit, then mix by char_len × detection score."""

        if not units:
            return {
                "status": "skipped",
                "source": "window_a0_equal_fusion",
                "provider": "window_a0_equal_fusion",
                "top_k": [],
                "confidence": 0.0,
                "blocks": [],
                "note": "No high-score detection blocks were selected for family attribution.",
            }
        try:
            self._ensure_loaded()
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "source": "window_a0_equal_fusion",
                "provider": "window_a0_equal_fusion",
                "top_k": [],
                "confidence": 0.0,
                "blocks": [],
                "error": str(exc),
                "install_hint": (
                    "Window-split A0 seed42 plus B0/B1/B3 artifacts must be present. "
                    "Do not train A1–A4 for this path."
                ),
            }

        block_rows: list[dict[str, Any]] = []
        mixed = np.zeros(len(self.family_labels), dtype=np.float64)
        weight_sum = 0.0
        for unit in units:
            text = str(unit.get("text") or "")
            if not text.strip():
                continue
            probabilities = self.predict_text(text)
            decision = decide_family_status(
                probabilities,
                self.family_labels,
                min_family_probability=self.min_family_probability,
                top_k=self.top_k,
            )
            weight = unit_weight(unit)
            mixed += probabilities * weight
            weight_sum += weight
            block_rows.append(
                {
                    "block_indices": list(unit.get("block_indices") or []),
                    "start": int(unit.get("start") or 0),
                    "end": int(unit.get("end") or 0),
                    "char_len": int(unit.get("char_len") or max(1, len(text))),
                    "detection_score": float(unit.get("detection_score") or 0.0),
                    "weight": weight,
                    "top_k": decision["top_k"],
                    "verdict": decision["verdict"],
                    "family_status": decision["status"],
                    "predicted_family": decision["predicted_family"],
                    "confidence": decision["confidence"],
                    "preview": text if len(text) <= 220 else text[:220] + "…",
                }
            )
        if weight_sum <= 0 or not block_rows:
            return {
                "status": "skipped",
                "source": "window_a0_equal_fusion",
                "provider": "window_a0_equal_fusion",
                "top_k": [],
                "confidence": 0.0,
                "blocks": block_rows,
                "note": "Selected blocks were empty after stripping.",
            }
        mixed = mixed / weight_sum
        decision = decide_family_status(
            mixed,
            self.family_labels,
            min_family_probability=self.min_family_probability,
            top_k=self.top_k,
        )
        return {
            "status": "ok",
            "source": "window_a0_equal_fusion",
            "provider": "window_a0_equal_fusion",
            "seed": OFFICIAL_SEED,
            "control_seed": CONTROL_SEED,
            "recipe": "window-A0+B0+B1+B3",
            "branch_weights": dict(self.branch_weights),
            "top_k": decision["top_k"],
            "verdict": decision["verdict"],
            "family_status": decision["status"],
            "predicted_family": decision["predicted_family"],
            "confidence": decision["confidence"],
            "blocks": block_rows,
            "family_probabilities": {
                label: float(value) for label, value in zip(self.family_labels, mixed)
            },
            "note": (
                "Official production mix: window-split A0 seed42 with B0, B1, and "
                "Phase1 B3. Weights are explicitly configured or default to equal "
                "0.25; research locks are not loaded automatically. Seed 62 is "
                "control only. Closed softmax has no 11th class; unknown means the "
                "fused Top-1 is too weak or too close to call a family."
            ),
        }

    def predict_text(self, text: str) -> np.ndarray:
        """Return a length-|families| probability vector for one contiguous unit."""

        self._ensure_loaded()
        branches = self.predict_branches(text)
        scores = np.zeros(len(self.family_labels), dtype=np.float64)
        for name, weight in self.branch_weights.items():
            scores += float(weight) * branches[name]
        total = float(scores.sum())
        if total <= 0:
            raise RuntimeError("fusion produced an empty probability vector")
        return scores / total

    def predict_branches(self, text: str) -> dict[str, np.ndarray]:
        """Return per-branch probability vectors for one contiguous unit."""

        self._ensure_loaded()
        branches = {"A0": self._predict_a0(text)}
        for name in ("B0", "B1", "B3"):
            branches[name] = np.asarray(
                predict_family_probabilities(
                    self._classical[name], [text], self.family_labels
                )[0],
                dtype=np.float64,
            )
        return branches

    def _predict_a0(self, text: str) -> np.ndarray:
        torch = self._torch
        tokenizer = self._tokenizer
        special_count = int(tokenizer.num_special_tokens_to_add(pair=False))
        content_length = self.max_length - special_count
        original_max = getattr(tokenizer, "model_max_length", None)
        if original_max is not None:
            tokenizer.model_max_length = int(10**9)
        try:
            raw = tokenizer(text, add_special_tokens=False, truncation=False)
            token_ids = [int(token_id) for token_id in raw["input_ids"]]
        finally:
            if original_max is not None:
                tokenizer.model_max_length = original_max
        spans = split_token_ids(token_ids, content_length, content_length) or [(0, 0)]
        rows: list[np.ndarray] = []
        for start, end in spans:
            if end <= start:
                window_text = text
            else:
                window_text = tokenizer.decode(token_ids[start:end], skip_special_tokens=True)
                if not str(window_text).strip():
                    window_text = text
            encoded = tokenize_phase3_windows(
                tokenizer,
                window_text,
                self.max_length,
                "single",
            )
            input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=self._device)
            attention_mask = torch.tensor(
                [encoded["attention_mask"]], dtype=torch.long, device=self._device
            )
            window_mask = torch.tensor(
                [encoded["window_mask"]], dtype=torch.long, device=self._device
            )
            with torch.no_grad():
                logits = self._a0(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    window_mask=window_mask,
                ).family_logits
                probabilities = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
            rows.append(np.asarray(probabilities, dtype=np.float64))
        stacked = np.stack(rows, axis=0)
        return stacked.mean(axis=0)

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        import torch
        from transformers import AutoTokenizer

        self._torch = torch
        device_name = self.device_name
        if device_name == "auto":
            device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device_name)
        a0_dir = _resolve_artifact(self.cfg.get("a0_model_dir"), DEFAULT_PATHS["a0_model_dir"])
        tokenizer_dir = _resolve_artifact(
            self.cfg.get("a0_tokenizer_dir"), DEFAULT_PATHS["a0_tokenizer_dir"]
        )
        if not a0_dir.exists():
            raise FileNotFoundError(f"official window-split A0 seed42 model missing: {a0_dir}")
        self._a0 = load_phase3_model(
            a0_dir, device=str(self._device), expected_family_labels=self.family_labels
        )
        recipe = self._a0.recipe
        if (
            recipe.recipe_id != "A0-window-split"
            or recipe.window_mode != "single"
            or recipe.validation_window_mode != "nonoverlap-probability-mean"
            or recipe.max_length != self.max_length
            or recipe.train_window_stride is None
        ):
            raise ValueError("incompatible window-A0 recipe or inference max_length")
        self._a0.eval()
        if tuple(self._a0.family_labels) != self.family_labels:
            raise ValueError("A0 family label order does not match the production catalog")
        self._tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
        self._classical = {
            "B0": _load_sklearn(_resolve_artifact(self.cfg.get("b0_model_path"), DEFAULT_PATHS["b0_model_path"])),
            "B1": _load_sklearn(_resolve_artifact(self.cfg.get("b1_model_path"), DEFAULT_PATHS["b1_model_path"])),
            "B3": _load_sklearn(_resolve_artifact(self.cfg.get("b3_model_path"), DEFAULT_PATHS["b3_model_path"])),
        }
        self._loaded = True


def decide_family_status(
    probabilities: np.ndarray,
    labels: Sequence[str],
    *,
    min_family_probability: float = DEFAULT_MIN_FAMILY_PROBABILITY,
    top_k: int = 5,
) -> dict[str, Any]:
    """Map a closed-set vector to family_only or unknown for the workbench.

    Unknown is only the Top-1 family when that probability is below the floor.
    The displayed Top-K stays the closed-set ranking; do not prepend 1-Top-1.
    """

    ranked = _top_k(probabilities, labels, top_k)
    if not ranked:
        return {
            "status": "unknown",
            "verdict": "AI-家族不明",
            "predicted_family": UNKNOWN_FAMILY,
            "confidence": 0.0,
            "top_k": [{"model": UNKNOWN_FAMILY, "probability": 1.0}],
        }
    top = ranked[0]
    if float(top["probability"]) >= min_family_probability:
        return {
            "status": "family_only",
            "verdict": f"AI-家族明确 · {top['model']}",
            "predicted_family": top["model"],
            "confidence": float(top["probability"]),
            "top_k": ranked,
        }
    return {
        "status": "unknown",
        "verdict": "AI-家族不明",
        "predicted_family": UNKNOWN_FAMILY,
        "confidence": float(top["probability"]),
        "top_k": ranked,
    }


def load_branch_weights(explicit: Mapping[str, float] | None = None) -> dict[str, float]:
    """Use explicit deployment weights or equal weights; never discover old locks."""

    if explicit is not None:
        return _normalize_weights(explicit)
    return dict(DEFAULT_BRANCH_WEIGHTS)


def _normalize_weights(raw: Mapping[str, Any]) -> dict[str, float]:
    if set(raw) != set(DEFAULT_BRANCH_WEIGHTS):
        raise ValueError("fusion weights must contain exactly A0, B0, B1, B3")
    values = {name: float(raw[name]) for name in DEFAULT_BRANCH_WEIGHTS}
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError("fusion weights must be finite and nonnegative")
    total = sum(values.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("fusion weights must have a finite positive sum")
    return {name: value / total for name, value in values.items()}


def mix_equal_probabilities(
    stores: Mapping[str, Mapping[str, float]],
    labels: Sequence[str],
    weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Average named probability maps; used by tests and offline fusion."""

    mix_weights = weights or DEFAULT_BRANCH_WEIGHTS
    scores: dict[str, float] = defaultdict(float)
    for name, weight in mix_weights.items():
        for label in labels:
            scores[label] += float(weight) * float(stores[name][label])
    total = sum(scores.values())
    if total <= 0:
        raise ValueError("equal-weight mix is empty")
    return {label: scores[label] / total for label in labels}


def _resolve_artifact(value: Any, default: Path) -> Path:
    if value in (None, ""):
        return default
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _load_sklearn(path: Path) -> Any:
    import joblib

    if not path.exists():
        raise FileNotFoundError(f"classical attribution artifact missing: {path}")
    artifact = joblib.load(path)
    if isinstance(artifact, dict):
        model = artifact.get("model") or artifact.get("pipeline") or artifact.get("classifier")
        if model is None:
            raise RuntimeError(f"{path} does not contain a sklearn model")
        return model
    return artifact


def _top_k(probabilities: np.ndarray, labels: Sequence[str], top_k: int) -> list[dict[str, Any]]:
    ranked = sorted(
        zip(labels, probabilities.tolist()),
        key=lambda item: float(item[1]),
        reverse=True,
    )[:top_k]
    return [
        {"model": str(label), "probability": round(float(score), 6)}
        for label, score in ranked
    ]
