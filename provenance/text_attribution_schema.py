"""Canonical records and deterministic split assignment for text attribution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Mapping


DEFAULT_SPLIT_RATIOS: dict[str, float] = {
    "train": 0.70,
    "calibration": 0.10,
    "test": 0.20,
}


def assign_prompt_split(
    prompt_group_id: str,
    *,
    seed: int,
    ratios: Mapping[str, float] | None = None,
) -> str:
    """Assign a prompt group to one deterministic split.

    Every output produced from the same prompt group receives the same split,
    which prevents content leakage between training and evaluation.
    """

    if not str(prompt_group_id).strip():
        raise ValueError("prompt_group_id must not be blank")
    split_ratios = dict(ratios or DEFAULT_SPLIT_RATIOS)
    if not split_ratios:
        raise ValueError("ratios must define at least one split")
    if any(not name.strip() or value < 0 for name, value in split_ratios.items()):
        raise ValueError("ratios must use non-empty names and non-negative values")
    total = sum(split_ratios.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1.0, got {total}")

    digest = hashlib.sha256(f"{seed}:{prompt_group_id}".encode("utf-8")).digest()
    position = int.from_bytes(digest[:8], "big") / float(1 << 64)
    cumulative = 0.0
    for name, ratio in split_ratios.items():
        cumulative += ratio
        if position < cumulative:
            return name
    return next(reversed(split_ratios))


@dataclass(frozen=True)
class AttributionSample:
    """One immutable, provenance-safe generated-text training record."""

    prompt_group_id: str
    prompt_template_id: str
    text: str
    language: str
    domain: str
    task_type: str
    provider: str
    model_family: str
    model_id: str
    deployment_snapshot: str
    collection_time: str
    split: str
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    request_id: str | None = None
    prompt_hash: str | None = None
    system_prompt_hash: str | None = None
    original_generator: str | None = None
    last_transformer: str | None = None
    generation_chain: tuple[str, ...] = ()
    transformation_type: str = "direct"
    dataset_revision: str = "unversioned"
    sample_id: str = field(init=False)

    def __post_init__(self) -> None:
        required = {
            "prompt_group_id": self.prompt_group_id,
            "prompt_template_id": self.prompt_template_id,
            "text": self.text,
            "language": self.language,
            "domain": self.domain,
            "task_type": self.task_type,
            "provider": self.provider,
            "model_family": self.model_family,
            "model_id": self.model_id,
            "deployment_snapshot": self.deployment_snapshot,
            "collection_time": self.collection_time,
            "split": self.split,
        }
        for name, value in required.items():
            if not str(value).strip():
                raise ValueError(f"{name} must not be blank")

        original = self.original_generator or self.model_id
        last = self.last_transformer or self.model_id
        chain = tuple(item for item in self.generation_chain if str(item).strip())
        if not chain:
            chain = (original,) if original == last else (original, last)

        object.__setattr__(self, "original_generator", original)
        object.__setattr__(self, "last_transformer", last)
        object.__setattr__(self, "generation_chain", chain)
        object.__setattr__(self, "sample_id", self._stable_sample_id())

    def _stable_sample_id(self) -> str:
        payload = {
            "prompt_group_id": self.prompt_group_id,
            "prompt_template_id": self.prompt_template_id,
            "text": self.text,
            "provider": self.provider,
            "model_id": self.model_id,
            "deployment_snapshot": self.deployment_snapshot,
            "collection_time": self.collection_time,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "generation_chain": self.generation_chain,
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_record(self) -> dict[str, Any]:
        """Return the allow-listed JSON record; secret-bearing fields do not exist."""

        record = asdict(self)
        record["generation_chain"] = list(self.generation_chain)
        return record

