"""Data preparation for leakage-safe text-attribution baselines."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable, Sequence


_LANGUAGE_PREFIX = re.compile(r"^(?:zh|en)-", flags=re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_LATIN = re.compile(r"[A-Za-z]")


@dataclass(frozen=True)
class BaselineSample:
    """One immutable, quality-approved attribution example."""

    request_key: str
    text: str
    label: str
    model: str
    prompt_id: str
    semantic_prompt_group: str
    language: str
    task_type: str
    target_length: str
    temperature: float
    repeat_index: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def semantic_prompt_group(prompt_id: str) -> str:
    """Return the language-independent identifier for a paired prompt."""

    return _LANGUAGE_PREFIX.sub("", str(prompt_id), count=1)


def normalize_text(text: str) -> str:
    """Normalize only whitespace so exact-content deduplication stays conservative."""

    return _WHITESPACE.sub(" ", str(text)).strip()


def load_clean_samples(
    path: str | Path,
    included_semantic_groups: frozenset[str] | None = None,
) -> tuple[list[BaselineSample], dict[str, Any]]:
    """Load JSONL and apply deterministic attribution-quality filters.

    Raw data is never modified. Exclusions are assigned in a fixed order so each
    row contributes to exactly one reason in the returned audit.
    """

    input_path = Path(path)
    excluded: Counter[str] = Counter()
    samples: list[BaselineSample] = []
    seen: set[tuple[str, str, float, str]] = set()
    raw_records = 0

    for row in _read_jsonl(input_path):
        raw_records += 1
        if included_semantic_groups is not None:
            prompt = row.get("prompt")
            prompt_id = (
                str(prompt.get("prompt_id", "")).strip()
                if isinstance(prompt, dict)
                else ""
            )
            if semantic_prompt_group(prompt_id) not in included_semantic_groups:
                excluded["outside_requested_groups"] += 1
                continue
        text = str(row.get("text", "")).strip()
        if not text:
            excluded["empty_text"] += 1
            continue

        finish_reason = str(row.get("finish_reason", "")).strip().lower()
        if finish_reason in {"length", "max_tokens"}:
            excluded["truncated"] += 1
            continue
        if finish_reason == "content_filter":
            excluded["content_filtered"] += 1
            continue
        if finish_reason != "stop":
            excluded["non_stop_finish"] += 1
            continue

        if bool(row.get("route_mismatch")):
            excluded["route_mismatch"] += 1
            continue

        prompt = row.get("prompt")
        parameters = row.get("parameters")
        if not isinstance(prompt, dict) or not isinstance(parameters, dict):
            excluded["invalid_metadata"] += 1
            continue

        language = str(prompt.get("language", "")).strip().lower()
        if not _language_matches(text, language):
            excluded["language_mismatch"] += 1
            continue

        label = str(row.get("family_id", "")).strip()
        prompt_id = str(prompt.get("prompt_id", "")).strip()
        request_key = str(row.get("request_key", "")).strip()
        if not label or not prompt_id or not request_key:
            excluded["invalid_metadata"] += 1
            continue

        try:
            temperature = float(parameters.get("temperature", 0.0))
            repeat_index = int(row.get("repeat_index", 0))
        except (TypeError, ValueError):
            excluded["invalid_metadata"] += 1
            continue

        normalized = normalize_text(text)
        duplicate_key = (label, prompt_id, temperature, normalized)
        if duplicate_key in seen:
            excluded["exact_duplicate"] += 1
            continue
        seen.add(duplicate_key)

        samples.append(
            BaselineSample(
                request_key=request_key,
                text=text,
                label=label,
                model=str(row.get("request_model", "")).strip(),
                prompt_id=prompt_id,
                semantic_prompt_group=semantic_prompt_group(prompt_id),
                language=language,
                task_type=str(prompt.get("task_type", "unknown")).strip() or "unknown",
                target_length=(
                    str(prompt.get("target_length", "unknown")).strip() or "unknown"
                ),
                temperature=temperature,
                repeat_index=repeat_index,
            )
        )

    audit: dict[str, Any] = {
        "input_path": str(input_path),
        "raw_records": raw_records,
        "kept_records": len(samples),
        "excluded_records": raw_records - len(samples),
        "excluded_by_reason": dict(sorted(excluded.items())),
        "counts_by_label": _sample_counts(samples, "label"),
        "counts_by_language": _sample_counts(samples, "language"),
        "counts_by_target_length": _sample_counts(samples, "target_length"),
        "semantic_prompt_groups": len(
            {sample.semantic_prompt_group for sample in samples}
        ),
    }
    return samples, audit


def assign_group_folds(
    samples: Sequence[BaselineSample],
    *,
    n_splits: int = 6,
    seed: int = 42,
) -> list[int]:
    """Assign deterministic folds without splitting semantic prompt pairs."""

    groups = sorted({sample.semantic_prompt_group for sample in samples})
    if n_splits < 2 or n_splits > len(groups):
        raise ValueError(
            "n_splits must be between 2 and the number of semantic prompt "
            f"groups ({len(groups)}), got {n_splits}"
        )

    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    group_to_fold = {
        group: index % n_splits for index, group in enumerate(shuffled)
    }
    folds = [group_to_fold[sample.semantic_prompt_group] for sample in samples]

    all_labels = {sample.label for sample in samples}
    for fold in range(n_splits):
        training_labels = {
            sample.label
            for sample, assigned in zip(samples, folds, strict=True)
            if assigned != fold
        }
        missing = sorted(all_labels - training_labels)
        if missing:
            raise ValueError(
                f"Fold {fold} leaves labels without training examples: {missing}"
            )
    return folds


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Attribution JSONL does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            yield row


def _language_matches(text: str, expected: str) -> bool:
    if expected not in {"zh", "en"}:
        return True
    cjk_count = len(_CJK.findall(text))
    latin_count = len(_LATIN.findall(text))
    visible_count = cjk_count + latin_count
    if visible_count == 0:
        return True
    cjk_ratio = cjk_count / visible_count
    if expected == "en":
        return not (cjk_count >= 8 and cjk_ratio > 0.35)
    return not (latin_count >= 20 and cjk_ratio < 0.15)


def _sample_counts(
    samples: Sequence[BaselineSample],
    attribute: str,
) -> dict[str, int]:
    counts = Counter(str(getattr(sample, attribute)) for sample in samples)
    return dict(sorted(counts.items()))
