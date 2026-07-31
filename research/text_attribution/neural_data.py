"""Neural-model data adapters for text-family attribution."""

from __future__ import annotations

from collections import Counter, defaultdict
import random
from typing import Any, Mapping, Sequence

from torch.utils.data import Dataset

from research.text_attribution.training_data import AttributionSample


class NeuralAttributionDataset(Dataset):
    """Tokenize attribution samples without exposing their metadata."""

    def __init__(
        self,
        samples: Sequence[AttributionSample],
        tokenizer: Any,
        family_to_id: Mapping[str, int],
        max_length: int,
    ) -> None:
        if not isinstance(max_length, int) or isinstance(max_length, bool):
            raise ValueError("max_length must be a positive integer")
        if max_length <= 0:
            raise ValueError("max_length must be a positive integer")

        stored_samples = tuple(samples)
        if not stored_samples:
            raise ValueError("samples must not be empty")

        missing_families = sorted(
            {
                sample.family_id
                for sample in stored_samples
                if sample.family_id not in family_to_id
            }
        )
        if missing_families:
            raise ValueError(
                f"family_to_id is missing sample families: {missing_families}"
            )
        if any(
            not isinstance(label_id, int) or isinstance(label_id, bool)
            for label_id in family_to_id.values()
        ):
            raise ValueError("family_to_id values must be integers")

        self._samples = stored_samples
        self._tokenizer = tokenizer
        self._family_to_id = dict(family_to_id)
        self._max_length = max_length

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._samples[index]
        model_input = dict(
            self._tokenizer(
                sample.text,
                truncation=True,
                max_length=self._max_length,
                padding=False,
            )
        )
        model_input["labels"] = self._family_to_id[sample.family_id]
        return model_input


def build_family_label_maps(
    family_labels: Sequence[str],
) -> tuple[dict[str, int], dict[int, str]]:
    """Build order-preserving forward and inverse family label maps."""

    labels = tuple(family_labels)
    if not labels:
        raise ValueError("family_labels must not be empty")
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("family labels must be non-empty strings")
    if len(set(labels)) != len(labels):
        raise ValueError("family labels must not contain duplicates")

    family_to_id = {label: index for index, label in enumerate(labels)}
    id_to_family = {index: label for label, index in family_to_id.items()}
    return family_to_id, id_to_family


def model_balanced_weights(
    samples: Sequence[AttributionSample],
) -> tuple[float, ...]:
    """Give each sample the inverse frequency of its concrete model."""

    stored_samples = tuple(samples)
    if not stored_samples:
        raise ValueError("samples must not be empty")
    counts = Counter(sample.model_id for sample in stored_samples)
    return tuple(1.0 / counts[sample.model_id] for sample in stored_samples)


def _sample_sort_key(sample: AttributionSample) -> tuple[Any, ...]:
    return (
        sample.model_id,
        sample.request_key,
        sample.text,
        sample.family_id,
        sample.prompt_id,
        sample.semantic_prompt_group,
        sample.language,
        sample.task_type,
        sample.target_length,
        sample.temperature,
        sample.repeat_index,
    )


def select_smoke_samples(
    samples: Sequence[AttributionSample],
    per_model: int,
    seed: int,
) -> tuple[AttributionSample, ...]:
    """Select a deterministic, model-balanced subset for smoke runs."""

    if (
        not isinstance(per_model, int)
        or isinstance(per_model, bool)
        or per_model <= 0
    ):
        raise ValueError("per_model must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")

    by_model: dict[str, list[AttributionSample]] = defaultdict(list)
    for sample in samples:
        by_model[sample.model_id].append(sample)

    generator = random.Random(seed)
    selected: list[AttributionSample] = []
    for model_id in sorted(by_model):
        candidates = sorted(by_model[model_id], key=_sample_sort_key)
        if len(candidates) <= per_model:
            selected.extend(candidates)
        else:
            selected.extend(generator.sample(candidates, per_model))

    return tuple(sorted(selected, key=_sample_sort_key))
