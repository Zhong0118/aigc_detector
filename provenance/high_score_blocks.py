"""Select detection blocks for family attribution.

Product rule (discussion notes 2026-09-09 §2.3):

- Reuse detector windows; do not recut the whole document.
- Prefer blocks above the stylistic high threshold; fill from the low
  threshold only if that set is thin; cap at 8–12 blocks.
- Merge only adjacent or overlapping selected blocks.
- Never concatenate non-adjacent high-score spans.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


DEFAULT_LOW_THRESHOLD = 0.35
DEFAULT_HIGH_THRESHOLD = 0.88
DEFAULT_MIN_BLOCKS = 8
DEFAULT_MAX_BLOCKS = 12
DEFAULT_HIGH_COVERAGE_GATE = 0.15
DEFAULT_MIN_SUSPECTED_BLOCKS = 2
DEFAULT_SHORT_BLOCK_SCORE = 0.35


def collect_detection_segments(detection: Any) -> list[dict[str, Any]]:
    """Pull detector windows from a DetectionPackage or analysis dict."""

    if detection is None:
        return []
    decision = _mapping(getattr(detection, "decision_details", None))
    if not decision and isinstance(detection, Mapping):
        nested = _mapping(detection.get("detection"))
        decision = _mapping(nested.get("decision") or detection.get("decision"))
    segments = _as_segment_list(decision.get("segments"))
    if segments:
        return segments
    providers = list(getattr(detection, "provider_results", None) or [])
    if not providers and isinstance(detection, Mapping):
        nested = _mapping(detection.get("detection"))
        providers = list(nested.get("providers") or detection.get("providers") or [])
    for result in providers:
        details = _mapping(getattr(result, "details", None))
        if not details and isinstance(result, Mapping):
            details = _mapping(result.get("details"))
        segments = _as_segment_list(details.get("segments"))
        if segments:
            return segments
    return []


def thresholds_from_detection(detection: Any) -> tuple[float, float]:
    """Return (low, high) detector thresholds, with academic-technical defaults."""

    decision = _mapping(getattr(detection, "decision_details", None))
    if not decision and isinstance(detection, Mapping):
        nested = _mapping(detection.get("detection"))
        decision = _mapping(nested.get("decision") or detection.get("decision"))
    low = decision.get("low_threshold", DEFAULT_LOW_THRESHOLD)
    high = decision.get("high_threshold", DEFAULT_HIGH_THRESHOLD)
    return float(low), float(high)


def high_score_coverage(segments: Sequence[Mapping[str, Any]], high_threshold: float) -> float:
    """Character share of blocks whose detector score exceeds the high threshold."""

    total = 0
    high = 0
    for segment in _normalize_segments(segments):
        length = segment["char_len"]
        total += length
        if segment["score"] > high_threshold:
            high += length
    if total <= 0:
        return 0.0
    return high / total


def suspected_ai_count(segments: Sequence[Mapping[str, Any]], high_threshold: float) -> int:
    return sum(1 for segment in _normalize_segments(segments) if segment["score"] > high_threshold)


def should_run_text_attribution(
    *,
    document_score: float,
    segments: Sequence[Mapping[str, Any]],
    low_threshold: float = DEFAULT_LOW_THRESHOLD,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
    deep_threshold: float = 0.6,
    high_coverage_gate: float = DEFAULT_HIGH_COVERAGE_GATE,
    min_suspected_blocks: int = DEFAULT_MIN_SUSPECTED_BLOCKS,
    short_block_score: float = DEFAULT_SHORT_BLOCK_SCORE,
) -> bool:
    """Attribution gate is independent of the `high_risk_ai` document label."""

    normalized = _normalize_segments(segments)
    if float(document_score) >= float(deep_threshold):
        return True
    if high_score_coverage(normalized, high_threshold) >= high_coverage_gate:
        return True
    if suspected_ai_count(normalized, high_threshold) >= min_suspected_blocks:
        return True
    if len(normalized) <= 1:
        score = normalized[0]["score"] if normalized else float(document_score)
        return score >= short_block_score
    return False


def select_attribution_units(
    segments: Sequence[Mapping[str, Any]],
    *,
    full_text: str | None = None,
    low_threshold: float = DEFAULT_LOW_THRESHOLD,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
    min_blocks: int = DEFAULT_MIN_BLOCKS,
    max_blocks: int = DEFAULT_MAX_BLOCKS,
) -> list[dict[str, Any]]:
    """Return adjacent-merged attribution units from selected detector blocks."""

    if max_blocks < 1:
        raise ValueError("max_blocks must be positive")
    if min_blocks < 1:
        raise ValueError("min_blocks must be positive")
    normalized = _normalize_segments(segments, full_text=full_text)
    selected = _select_blocks(
        normalized,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        min_blocks=min_blocks,
        max_blocks=max_blocks,
    )
    return _merge_adjacent(selected)


def unit_weight(unit: Mapping[str, Any]) -> float:
    """Document-level mix weight: character length times detector score."""

    char_len = max(0, int(unit.get("char_len") or 0))
    score = max(0.0, float(unit.get("detection_score") or 0.0))
    weight = char_len * score
    if weight > 0:
        return weight
    return float(max(char_len, 1))


def _select_blocks(
    segments: Sequence[dict[str, Any]],
    *,
    low_threshold: float,
    high_threshold: float,
    min_blocks: int,
    max_blocks: int,
) -> list[dict[str, Any]]:
    high = [item for item in segments if item["score"] > high_threshold]
    high.sort(key=lambda item: (-item["score"], item["index"]))
    chosen = high[:max_blocks]
    if len(chosen) >= min(min_blocks, max_blocks) or len(chosen) >= max_blocks:
        chosen.sort(key=lambda item: (item["start"], item["index"]))
        return chosen
    remaining = [
        item
        for item in segments
        if item["score"] > low_threshold and item["index"] not in {row["index"] for row in chosen}
    ]
    remaining.sort(key=lambda item: (-item["score"], item["index"]))
    fill_to = min(max_blocks, max(min_blocks, len(chosen)))
    for item in remaining:
        if len(chosen) >= fill_to:
            break
        chosen.append(item)
    if not chosen and segments:
        # Gate already decided to attribute; take the strongest remaining windows.
        ranked = sorted(segments, key=lambda item: (-item["score"], item["index"]))
        chosen = ranked[: min(max_blocks, max(1, min_blocks if len(segments) > 1 else 1))]
    chosen.sort(key=lambda item: (item["start"], item["index"]))
    return chosen[:max_blocks]


def _merge_adjacent(blocks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for block in blocks:
        if units and _adjacent(units[-1], block):
            units[-1] = _extend_unit(units[-1], block)
        else:
            units.append(_new_unit(block))
    return units


def _adjacent(unit: Mapping[str, Any], block: Mapping[str, Any]) -> bool:
    return int(block["start"]) <= int(unit["end"])


def _new_unit(block: Mapping[str, Any]) -> dict[str, Any]:
    char_len = int(block["char_len"])
    score = float(block["score"])
    return {
        "block_indices": [int(block["index"])],
        "start": int(block["start"]),
        "end": int(block["end"]),
        "text": str(block.get("text") or ""),
        "char_len": char_len,
        "detection_score": score,
        "member_scores": [score],
        "member_char_lens": [char_len],
    }


def _extend_unit(unit: dict[str, Any], block: Mapping[str, Any]) -> dict[str, Any]:
    start = min(int(unit["start"]), int(block["start"]))
    end = max(int(unit["end"]), int(block["end"]))
    left = str(unit.get("text") or "")
    right = str(block.get("text") or "")
    if int(block["start"]) < int(unit["end"]) and left and right:
        overlap = int(unit["end"]) - int(block["start"])
        if overlap > 0:
            right = right[overlap:] if overlap < len(right) else ""
    text = left + right
    scores = list(unit.get("member_scores") or []) + [float(block["score"])]
    lens = list(unit.get("member_char_lens") or []) + [int(block["char_len"])]
    mass = sum(max(1, length) for length in lens)
    weighted = sum(float(score) * max(1, length) for score, length in zip(scores, lens))
    return {
        "block_indices": list(unit["block_indices"]) + [int(block["index"])],
        "start": start,
        "end": end,
        "text": text,
        "char_len": max(1, end - start, len(text)),
        "detection_score": weighted / mass if mass else float(block["score"]),
        "member_scores": scores,
        "member_char_lens": lens,
    }


def _normalize_segments(
    segments: Sequence[Mapping[str, Any]],
    *,
    full_text: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset, raw in enumerate(segments or []):
        if not isinstance(raw, Mapping) or raw.get("score") is None:
            continue
        start = int(raw.get("start") or 0)
        end = int(raw.get("end") or start)
        text = str(raw.get("text") or "")
        if not text and full_text is not None:
            text = full_text[start:end]
        char_len = max(1, end - start, len(text))
        rows.append(
            {
                "index": int(raw.get("index", offset)),
                "start": start,
                "end": end,
                "text": text,
                "score": float(raw["score"]),
                "char_len": char_len,
            }
        )
    rows.sort(key=lambda item: (item["start"], item["index"]))
    return rows


def _as_segment_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict) and item.get("score") is not None]


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}
