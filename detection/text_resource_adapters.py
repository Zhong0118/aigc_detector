"""Normalize supported text datasets into canonical calibration samples."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
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

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("TextSample text must not be blank")
        if self.label not in {"human", "ai"}:
            raise ValueError(f"Unsupported label: {self.label}")
        _validate_language(self.language)
        if self.domain not in CANONICAL_DOMAINS:
            raise ValueError(f"Unsupported domain: {self.domain}")
        for name in ("source", "source_id", "split"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"TextSample {name} must not be blank")

    def group_key(self, length_bucket: str) -> str:
        return f"{self.language}:{self.domain}:{length_bucket}"


def adapt_hc3_rows(
    rows: Iterable[Mapping[str, Any]],
    source_name: str,
    language: str,
    split: str = "train",
) -> Iterator[TextSample]:
    """Flatten HC3 answer lists while retaining question-level identity."""
    valid_language = _validate_language(language)
    for row_index, row in enumerate(rows):
        question_id = _row_id(row, row_index, ("question_id", "id"))
        prefix = f"{source_name}:{question_id}"
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
                )


def adapt_tnews_rows(
    rows: Iterable[Mapping[str, Any]],
    source_name: str = "tnews",
    language: str = "zh",
    split: str = "train",
) -> Iterator[TextSample]:
    """Adapt TNEWS rows as Chinese human official-news samples."""
    valid_language = _validate_language(language)
    if valid_language != "zh":
        raise ValueError("TNEWS rows must use language 'zh'")
    for row_index, row in enumerate(rows):
        text = _first_text(row, ("sentence", "text"))
        if text is None:
            continue
        yield TextSample(
            text=text,
            label="human",
            language="zh",
            domain="official_news",
            source=source_name,
            source_id=f"{source_name}:{_row_id(row, row_index)}",
            split=split,
        )


def adapt_m4_rows(
    rows: Iterable[Mapping[str, Any]],
    source_name: str,
    language: str | None = None,
    split: str = "train",
) -> Iterator[TextSample]:
    """Adapt M4/SemEval rows, preserving their source and generator metadata."""
    default_language = _validate_language(language) if language is not None else None
    for row_index, row in enumerate(rows):
        text = _first_text(row, ("text", "generation"))
        if text is None:
            continue
        row_language = _validate_language(row.get("language") or default_language)
        label = _m4_label(row.get("label"))
        generator = _first_string(row, ("model", "generator")) if label == "ai" else None
        domain_metadata = row.get("domain") or row.get("source") or source_name
        yield TextSample(
            text=text,
            label=label,
            language=row_language,
            domain=normalize_source_domain(domain_metadata),
            source=source_name,
            source_id=f"{source_name}:{_row_id(row, row_index)}",
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
    valid_language = _validate_language(language)
    for row_index, row in enumerate(rows):
        if not _is_unattacked(row.get("attack")):
            continue
        text = _first_text(row, ("generation", "text"))
        if text is None:
            continue
        model = str(row.get("model") or "").strip()
        is_human = model.casefold() == "human"
        yield TextSample(
            text=text,
            label="human" if is_human else "ai",
            language=valid_language,
            domain=normalize_source_domain(row.get("domain") or row.get("source")),
            source=source_name,
            source_id=f"{source_name}:{_row_id(row, row_index)}",
            generator=None if is_human else model or None,
            split=split,
        )


def _validate_language(language: Any) -> TextLanguage:
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
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _text_values(value: Any) -> Iterator[str]:
    values = value if isinstance(value, (list, tuple)) else (value,)
    for item in values:
        if isinstance(item, str) and item.strip():
            yield item.strip()


def _row_id(
    row: Mapping[str, Any],
    row_index: int,
    keys: tuple[str, ...] = ("source_id", "id"),
) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return str(row_index)


def _is_unattacked(attack: Any) -> bool:
    if attack is None:
        return True
    if isinstance(attack, float) and math.isnan(attack):
        return True
    return str(attack).strip().casefold() in {"", "none", "null"}
