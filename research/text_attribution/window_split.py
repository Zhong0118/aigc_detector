"""Expand long attribution documents into labeled 512-token training windows.

A0 currently truncates each JSONL row to the first 512 tokens. Window splitting
copies family_id / model_id / prompt group onto each non-overlapping window so
the discarded tail also receives gradient. Windows from one document stay in
the same split; evaluation should still be aggregated by original document.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

from research.text_attribution.training_data import AttributionSample


def split_token_ids(token_ids: Sequence[int], content_length: int, stride: int) -> list[tuple[int, int]]:
    """Return [start, end) spans covering token_ids with a fixed window."""

    if content_length <= 0 or stride <= 0:
        raise ValueError("content_length and stride must be positive")
    if not token_ids:
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    total = len(token_ids)
    while start < total:
        end = min(start + content_length, total)
        spans.append((start, end))
        if end >= total:
            break
        start += stride
    return spans


def expand_samples_to_windows(
    samples: Sequence[AttributionSample],
    tokenizer: Any,
    *,
    max_length: int = 512,
    stride: int | None = None,
) -> tuple[tuple[AttributionSample, ...], dict[str, int]]:
    """Turn each document into one or more window samples with copied labels.

    Validation/test callers should keep original documents and only expand train.
    """

    if max_length <= 2:
        raise ValueError("max_length must leave room for special tokens")
    special_count = int(tokenizer.num_special_tokens_to_add(pair=False))
    content_length = max_length - special_count
    if content_length < 1:
        raise ValueError("max_length must leave room for at least one content token")
    window_stride = content_length if stride is None else int(stride)
    if window_stride <= 0:
        raise ValueError("stride must be positive")

    expanded: list[AttributionSample] = []
    document_count = 0
    multi_window_documents = 0
    original_max = getattr(tokenizer, "model_max_length", None)
    if original_max is not None:
        tokenizer.model_max_length = int(10**9)
    try:
        for sample in samples:
            document_count += 1
            raw = tokenizer(sample.text, add_special_tokens=False, truncation=False)
            token_ids = [int(token_id) for token_id in raw["input_ids"]]
            spans = split_token_ids(token_ids, content_length, window_stride) or [(0, 0)]
            if len(spans) > 1:
                multi_window_documents += 1
            for index, (start, end) in enumerate(spans):
                if end <= start:
                    window_text = sample.text
                else:
                    window_text = tokenizer.decode(
                        token_ids[start:end],
                        skip_special_tokens=True,
                    )
                if not str(window_text).strip():
                    window_text = sample.text
                expanded.append(
                    replace(
                        sample,
                        request_key=f"{sample.request_key}::w{index}",
                        text=window_text,
                    )
                )
    finally:
        if original_max is not None:
            tokenizer.model_max_length = original_max
    audit = {
        "source_documents": document_count,
        "window_rows": len(expanded),
        "multi_window_documents": multi_window_documents,
        "content_length": content_length,
        "stride": window_stride,
    }
    return tuple(expanded), audit


def expand_validation_windows(
    samples: Sequence[AttributionSample], tokenizer: Any, *, max_length: int
) -> tuple[tuple[AttributionSample, ...], tuple[int, ...]]:
    """Return flattened non-overlap windows and counts aligned to source documents."""

    all_windows: list[AttributionSample] = []
    counts: list[int] = []
    for sample in samples:
        windows, _ = expand_samples_to_windows((sample,), tokenizer, max_length=max_length)
        all_windows.extend(windows)
        counts.append(len(windows))
    return tuple(all_windows), tuple(counts)
