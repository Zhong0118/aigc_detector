"""Explicit, corpus-independent stylometric features for text attribution."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


_CJK_CHARACTER = r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
_LATIN_CHARACTER = r"A-Za-z\u00c0-\u024f"
_TOKEN_PATTERN = re.compile(
    rf"[{_CJK_CHARACTER}]|[{_LATIN_CHARACTER}]+|\d+",
    re.UNICODE,
)
_SENTENCE_PATTERN = re.compile(r"[^.!?\u3002\uff01\uff1f]+")
_PARAGRAPH_PATTERN = re.compile(r"(?:\r?\n\s*){2,}")

_LENGTH_FEATURE_NAMES = (
    "char_count_log1p",
    "word_count_log1p",
    "sentence_count_log1p",
    "paragraph_count_log1p",
    "mean_sentence_chars_log1p",
    "std_sentence_chars_log1p",
    "mean_paragraph_chars_log1p",
)
_STYLE_FEATURE_NAMES = (
    "cjk_ratio",
    "latin_ratio",
    "digit_ratio",
    "whitespace_ratio",
    "newline_ratio",
    "comma_per_1000",
    "period_per_1000",
    "colon_per_1000",
    "semicolon_per_1000",
    "question_per_1000",
    "exclamation_per_1000",
    "quote_per_1000",
    "parenthesis_per_1000",
    "dash_per_1000",
    "markdown_heading_ratio",
    "bullet_line_ratio",
    "numbered_line_ratio",
    "indented_line_ratio",
    "bold_per_1000",
    "code_fence_per_1000",
    "type_token_ratio",
    "repeated_bigram_ratio",
    "zh_de_ratio",
    "zh_shi_ratio",
    "zh_yinwei_ratio",
    "zh_danshi_ratio",
    "en_the_ratio",
    "en_and_ratio",
    "en_because_ratio",
    "en_however_ratio",
    "starts_with_transition",
    "ends_with_summary",
)


class StylometricFeatureExtractor(BaseEstimator, TransformerMixin):
    """Transform strings into deterministic, interpretable style features."""

    def __init__(self, include_length_features: bool = True) -> None:
        self.include_length_features = include_length_features

    def fit(self, texts: Sequence[str] | np.ndarray, y: object = None) -> "StylometricFeatureExtractor":
        """Validate inputs; no corpus statistics are learned."""

        self._validate_texts(texts)
        return self

    def transform(self, texts: Sequence[str] | np.ndarray) -> np.ndarray:
        """Return a float64 feature matrix with one row for each input text."""

        values = self._validate_texts(texts)
        matrix = np.empty((len(values), len(self.get_feature_names_out())), dtype=np.float64)
        for index, text in enumerate(values):
            matrix[index] = self._features_for_text(text)
        return matrix

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return stable names in exactly the same order as ``transform`` columns."""

        names = _STYLE_FEATURE_NAMES
        if self.include_length_features:
            names = _LENGTH_FEATURE_NAMES + names
        return np.asarray(names, dtype=object)

    @staticmethod
    def _validate_texts(texts: Sequence[str] | np.ndarray) -> list[str]:
        if isinstance(texts, (str, bytes)) or not isinstance(texts, (list, tuple, np.ndarray)):
            raise ValueError("texts must be a one-dimensional sequence of strings")
        if isinstance(texts, np.ndarray):
            if texts.ndim != 1:
                raise ValueError("texts must be a one-dimensional sequence of strings")
            values = texts.tolist()
        else:
            values = list(texts)
        if not all(isinstance(value, str) for value in values):
            raise TypeError("texts must contain only strings")
        return values

    def _features_for_text(self, text: str) -> list[float]:
        character_count = len(text)
        denominator = max(character_count, 1)
        tokens = _TOKEN_PATTERN.findall(text)
        normalized_tokens = [token.casefold() for token in tokens]
        sentence_lengths = [len(match.group().strip()) for match in _SENTENCE_PATTERN.finditer(text)]
        sentence_lengths = [length for length in sentence_lengths if length]
        paragraphs = [part.strip() for part in _PARAGRAPH_PATTERN.split(text) if part.strip()]
        paragraph_lengths = [len(part) for part in paragraphs]
        nonempty_lines = [line for line in text.splitlines() if line.strip()]
        line_denominator = max(len(nonempty_lines), 1)

        style = [
            self._count_pattern(text, rf"[{_CJK_CHARACTER}]") / denominator,
            self._count_pattern(text, rf"[{_LATIN_CHARACTER}]") / denominator,
            sum(character.isdigit() for character in text) / denominator,
            sum(character.isspace() for character in text) / denominator,
            (text.count("\n") + text.count("\r")) / denominator,
            self._per_1000(text.count(",") + text.count("，"), denominator),
            self._per_1000(text.count(".") + text.count("。"), denominator),
            self._per_1000(text.count(":") + text.count("："), denominator),
            self._per_1000(text.count(";") + text.count("；"), denominator),
            self._per_1000(text.count("?") + text.count("？"), denominator),
            self._per_1000(text.count("!") + text.count("！"), denominator),
            self._per_1000(self._count_pattern(text, r"[\"'“”‘’]"), denominator),
            self._per_1000(self._count_pattern(text, r"[()（）\[\]【】]"), denominator),
            self._per_1000(self._count_pattern(text, r"[-–—]"), denominator),
            self._line_ratio(nonempty_lines, r"\s{0,3}#{1,6}\s+"),
            self._line_ratio(nonempty_lines, r"\s*[-*+]\s+"),
            self._line_ratio(nonempty_lines, r"\s*\d+[.)]\s+"),
            self._line_ratio(nonempty_lines, r"(?: {2,}|\t+)"),
            self._per_1000(len(re.findall(r"(?:\*\*|__).+?(?:\*\*|__)", text, re.DOTALL)), denominator),
            self._per_1000(text.count("```"), denominator),
            len(set(normalized_tokens)) / max(len(normalized_tokens), 1),
            self._repeated_bigram_ratio(normalized_tokens),
            self._substring_ratio(text, "的"),
            self._substring_ratio(text, "是"),
            self._substring_ratio(text, "因为"),
            self._substring_ratio(text, "但是"),
            self._token_ratio(normalized_tokens, "the"),
            self._token_ratio(normalized_tokens, "and"),
            self._token_ratio(normalized_tokens, "because"),
            self._token_ratio(normalized_tokens, "however"),
            float(self._starts_with_transition(text)),
            float(self._ends_with_summary(text)),
        ]
        if not self.include_length_features:
            return style
        return [
            float(np.log1p(character_count)),
            float(np.log1p(len(tokens))),
            float(np.log1p(len(sentence_lengths))),
            float(np.log1p(len(paragraph_lengths))),
            float(np.log1p(np.mean(sentence_lengths) if sentence_lengths else 0.0)),
            float(np.log1p(np.std(sentence_lengths) if sentence_lengths else 0.0)),
            float(np.log1p(np.mean(paragraph_lengths) if paragraph_lengths else 0.0)),
            *style,
        ]

    @staticmethod
    def _count_pattern(text: str, pattern: str) -> int:
        return len(re.findall(pattern, text))

    @staticmethod
    def _per_1000(count: int, denominator: int) -> float:
        return 1000.0 * count / denominator

    @staticmethod
    def _line_ratio(lines: list[str], pattern: str) -> float:
        return sum(bool(re.match(pattern, line)) for line in lines) / max(len(lines), 1)

    @staticmethod
    def _token_ratio(tokens: list[str], token: str) -> float:
        return tokens.count(token) / max(len(tokens), 1)

    @staticmethod
    def _substring_ratio(text: str, token: str) -> float:
        return text.count(token) / max(len(_TOKEN_PATTERN.findall(text)), 1)

    @staticmethod
    def _repeated_bigram_ratio(tokens: list[str]) -> float:
        bigrams = list(zip(tokens, tokens[1:]))
        if not bigrams:
            return 0.0
        counts = Counter(bigrams)
        return sum(count for count in counts.values() if count > 1) / len(bigrams)

    @staticmethod
    def _starts_with_transition(text: str) -> bool:
        start = text.strip().casefold()
        return bool(re.match(r"(?:however|therefore|moreover|meanwhile|first|其次|首先|因此|但是|不过)", start))

    @staticmethod
    def _ends_with_summary(text: str) -> bool:
        end = text.strip().casefold().rstrip(".!?。！？")
        return bool(re.search(r"(?:in summary|to summarize|in conclusion|overall|总结|综上|结论)$", end))
