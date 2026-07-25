"""Normalize supported text datasets into canonical calibration samples."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from numbers import Real
from typing import Any, Iterable, Iterator, Literal, Mapping


TextLabel = Literal["human", "ai"]
TextLanguage = Literal["zh", "en"]

CANONICAL_DOMAINS = frozenset(
    {"literature", "academic_technical", "official_news", "general_qa"}
)

_SOURCE_DOMAIN_ALIASES = {
    "literature": "literature",
    "literary": "literature",
    "book": "literature",
    "books": "literature",
    "fiction": "literature",
    "poem": "literature",
    "poems": "literature",
    "poetry": "literature",
    "academic": "academic_technical",
    "academic_technical": "academic_technical",
    "technical": "academic_technical",
    "abstract": "academic_technical",
    "abstracts": "academic_technical",
    "arxiv": "academic_technical",
    "code": "academic_technical",
    "computer_science": "academic_technical",
    "medicine": "academic_technical",
    "medical": "academic_technical",
    "peer_read": "academic_technical",
    "peer_review": "academic_technical",
    "peerread": "academic_technical",
    "peerreview": "academic_technical",
    "science": "academic_technical",
    "scientific": "academic_technical",
    "wiki_csai": "academic_technical",
    "government": "official_news",
    "news": "official_news",
    "news_article": "official_news",
    "news_articles": "official_news",
    "official": "official_news",
    "official_news": "official_news",
    "tnews": "official_news",
    "finance": "general_qa",
    "general_qa": "general_qa",
    "law": "general_qa",
    "open_qa": "general_qa",
    "psychology": "general_qa",
    "qa": "general_qa",
    "question_answering": "general_qa",
    "recipe": "general_qa",
    "recipes": "general_qa",
    "reddit": "general_qa",
    "reddit_eli5": "general_qa",
    "review": "general_qa",
    "reviews": "general_qa",
    "wiki": "general_qa",
    "wiki_how": "general_qa",
    "wikihow": "general_qa",
    "wikipedia": "general_qa",
}

_NONFINITE_STRINGS = frozenset(
    {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}
)
_ROW_ID_FIELDS = frozenset({"source_id", "question_id", "id"})


def _require_nonblank_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    if normalized.casefold() in _NONFINITE_STRINGS:
        raise ValueError(f"{field_name} must not be NaN or infinite")
    return normalized


def normalize_source_domain(source_domain: Any) -> str:
    """Map dataset source/domain metadata to one canonical domain."""
    normalized = re.sub(r"[^a-z0-9]+", "_", str(source_domain or "").casefold()).strip("_")
    return _SOURCE_DOMAIN_ALIASES.get(normalized, "general_qa")


@dataclass(frozen=True)
class TextSample:
    text: str
    label: TextLabel
    language: TextLanguage
    domain: str
    source: str
    source_id: str
    generator: str | None = None
    split: str = "train"
    partition_id: str | None = None

    def __post_init__(self) -> None:
        _require_nonblank_string(self.text, "text")
        _require_nonblank_string(self.source, "source")
        _require_nonblank_string(self.source_id, "source_id")
        _require_nonblank_string(self.split, "split")
        if self.generator is not None:
            _require_nonblank_string(self.generator, "generator")
        if self.partition_id is not None:
            _require_nonblank_string(self.partition_id, "partition_id")
        if not isinstance(self.label, str):
            raise TypeError("label must be a string")
        if self.label not in {"human", "ai"}:
            raise ValueError(f"Unsupported label: {self.label}")
        _validate_language(self.language)
        if not isinstance(self.domain, str):
            raise TypeError("domain must be a string")
        if self.domain not in CANONICAL_DOMAINS:
            raise ValueError(f"Unsupported domain: {self.domain}")

    def group_key(self, length_bucket: str) -> str:
        if not isinstance(length_bucket, str):
            raise TypeError("length_bucket must be a string")
        if length_bucket not in {"short", "long"}:
            raise ValueError("length_bucket must be 'short' or 'long'")
        return f"{self.language}:{self.domain}:{length_bucket}"

    @property
    def partition_key(self) -> str:
        return self.partition_id or self.source_id

    @property
    def identity_key(self) -> str:
        return _stable_json_hash(
            {
                "source": self.source,
                "split": self.split,
                "source_id": self.source_id,
                "label": self.label,
                "generator": self.generator,
                "text": self.text,
            }
        )


def adapt_hc3_rows(
    rows: Iterable[Mapping[str, Any]],
    source_name: str,
    language: str,
    split: str = "train",
) -> Iterator[TextSample]:
    """Flatten HC3 answer lists while retaining question-level identity."""
    source_name = _require_nonblank_string(source_name, "source_name")
    split = _require_nonblank_string(split, "split")
    valid_language = _validate_language(language)
    for row in rows:
        question_id = _row_id(
            row,
            source_name,
            split,
            ("question_id", "id", "source_id"),
        )
        prefix = _source_row_key(source_name, split, question_id)
        for label, field, generator in (
            ("human", "human_answers", None),
            ("ai", "chatgpt_answers", "chatgpt"),
        ):
            for answer_index, answer in enumerate(_text_values(row.get(field))):
                yield TextSample(
                    text=answer,
                    label=label,
                    language=valid_language,
                    domain=normalize_source_domain(row.get("domain") or source_name),
                    source=source_name,
                    source_id=f"{prefix}:{label}:{answer_index}",
                    generator=generator,
                    split=split,
                    partition_id=prefix,
                )


def adapt_tnews_rows(
    rows: Iterable[Mapping[str, Any]],
    source_name: str = "tnews",
    language: str = "zh",
    split: str = "train",
) -> Iterator[TextSample]:
    """Adapt TNEWS rows as Chinese human official-news samples."""
    source_name = _require_nonblank_string(source_name, "source_name")
    split = _require_nonblank_string(split, "split")
    valid_language = _validate_language(language)
    if valid_language != "zh":
        raise ValueError("TNEWS rows must use language 'zh'")
    for row in rows:
        text = _first_text(row, ("sentence", "text"))
        if text is None:
            continue
        row_id = _row_id(row, source_name, split)
        yield TextSample(
            text=text,
            label="human",
            language="zh",
            domain="official_news",
            source=source_name,
            source_id=_source_row_key(source_name, split, row_id),
            split=split,
        )


def adapt_m4_rows(
    rows: Iterable[Mapping[str, Any]],
    source_name: str,
    language: str | None = None,
    split: str = "train",
) -> Iterator[TextSample]:
    """Adapt M4/SemEval rows, preserving their source and generator metadata."""
    source_name = _require_nonblank_string(source_name, "source_name")
    split = _require_nonblank_string(split, "split")
    default_language = _validate_language(language) if language is not None else None
    for row in rows:
        text = _first_text(row, ("text", "generation"))
        if text is None:
            continue
        row_language = _validate_language(row.get("language") or default_language)
        label = _m4_label(row.get("label"))
        generator = _first_string(row, ("model", "generator")) if label == "ai" else None
        domain_metadata = row.get("domain") or row.get("source") or source_name
        row_id = _row_id(row, source_name, split)
        yield TextSample(
            text=text,
            label=label,
            language=row_language,
            domain=normalize_source_domain(domain_metadata),
            source=source_name,
            source_id=_source_row_key(source_name, split, row_id),
            generator=generator,
            split=split,
        )


def adapt_raid_rows(
    rows: Iterable[Mapping[str, Any]],
    source_name: str = "raid_no_adversarial",
    language: str = "en",
    split: str = "train",
) -> Iterator[TextSample]:
    """Adapt non-adversarial RAID rows and discard attacked variants."""
    source_name = _require_nonblank_string(source_name, "source_name")
    split = _require_nonblank_string(split, "split")
    valid_language = _validate_language(language)
    for row in rows:
        if not _is_unattacked(row.get("attack")):
            continue
        text = _first_text(row, ("generation", "text"))
        if text is None:
            continue
        row_id = _row_id(row, source_name, split)
        source_id = _source_row_key(source_name, split, row_id)
        model = _raid_model(row.get("model"), source_id)
        is_human = model.casefold() == "human"
        yield TextSample(
            text=text,
            label="human" if is_human else "ai",
            language=valid_language,
            domain=normalize_source_domain(row.get("domain") or row.get("source")),
            source=source_name,
            source_id=source_id,
            generator=None if is_human else model,
            split=split,
        )


def _validate_language(language: Any) -> TextLanguage:
    if not isinstance(language, str):
        raise TypeError("language must be a string")
    if language not in {"zh", "en"}:
        raise ValueError(f"Unsupported language: {language}")
    return language


def _m4_label(value: Any) -> TextLabel:
    if not isinstance(value, bool) and value in (0, "0"):
        return "human"
    if not isinstance(value, bool) and value in (1, "1"):
        return "ai"
    raise ValueError(f"Unsupported M4 label: {value}")


def _first_text(row: Mapping[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _first_string(row: Mapping[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = row.get(key)
        if not isinstance(value, str):
            continue
        try:
            return _require_nonblank_string(value, key)
        except ValueError:
            continue
    return None


def _text_values(value: Any) -> Iterator[str]:
    values = value if isinstance(value, (list, tuple)) else (value,)
    for item in values:
        if isinstance(item, str) and item.strip():
            yield item.strip()


def _row_id(
    row: Mapping[str, Any],
    source_name: str,
    split: str,
    keys: tuple[str, ...] = ("source_id", "question_id", "id"),
) -> str:
    for key in keys:
        identifier = _valid_row_id(row.get(key))
        if identifier is not None:
            return identifier
    content = {key: value for key, value in row.items() if key not in _ROW_ID_FIELDS}
    return f"sha256-{_stable_json_hash({'source': source_name, 'split': split, 'row': content})}"


def _valid_row_id(value: Any) -> str | None:
    if isinstance(value, str):
        try:
            return _require_nonblank_string(value, "row id")
        except ValueError:
            return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if not isinstance(value, Real):
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return str(value)


def _source_row_key(source_name: str, split: str, row_id: str) -> str:
    return f"{source_name}:{split}:{row_id}"


def _stable_json_hash(value: Any) -> str:
    payload = json.dumps(
        _canonical_json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_canonical_json_value(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, default=str))
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, Real):
        numeric = float(value)
        if not math.isfinite(numeric):
            return None
        return int(value) if numeric.is_integer() else numeric
    return {"type": type(value).__qualname__, "value": str(value)}


def _raid_model(value: Any, row_context: str) -> str:
    try:
        return _require_nonblank_string(value, "RAID model")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid RAID model for {row_context}: {exc}") from None


def _is_unattacked(attack: Any) -> bool:
    if attack is None:
        return True
    if isinstance(attack, float) and math.isnan(attack):
        return True
    return str(attack).strip().casefold() in {"", "none", "null"}
