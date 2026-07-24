"""Sentence-aware text windows used by local AIGC detectors."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TextSegment:
    """A text window with coordinates in the original input."""

    index: int
    start: int
    end: int
    text: str


class SentenceAwareSegmenter:
    """Build bounded, overlapping windows without losing source offsets."""

    _SENTENCE_RE = re.compile(r".+?(?:[。！？!?；;\n]+|$)", re.DOTALL)

    def __init__(
        self,
        target_chars: int = 480,
        max_chars: int = 640,
        overlap_sentences: int = 1,
    ) -> None:
        if target_chars <= 0 or max_chars < target_chars:
            raise ValueError("max_chars must be greater than or equal to target_chars")
        self.target_chars = int(target_chars)
        self.max_chars = int(max_chars)
        self.overlap_sentences = max(0, int(overlap_sentences))

    def split(self, text: str) -> list[TextSegment]:
        if not text:
            return []
        if len(text) <= self.target_chars:
            return [TextSegment(index=0, start=0, end=len(text), text=text)]

        spans = self._sentence_spans(text)
        windows: list[TextSegment] = []
        cursor = 0
        while cursor < len(spans):
            start = spans[cursor][0]
            end = start
            probe = cursor
            last_included = cursor - 1
            while probe < len(spans):
                candidate_end = spans[probe][1]
                if candidate_end - start > self.max_chars and last_included >= cursor:
                    break
                end = candidate_end
                last_included = probe
                if end - start >= self.target_chars or probe == len(spans) - 1:
                    break
                probe += 1

            windows.append(
                TextSegment(
                    index=len(windows),
                    start=start,
                    end=end,
                    text=text[start:end],
                )
            )
            if last_included >= len(spans) - 1:
                break
            next_cursor = max(cursor + 1, last_included + 1 - self.overlap_sentences)
            cursor = next_cursor
        return windows

    def _sentence_spans(self, text: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        for match in self._SENTENCE_RE.finditer(text):
            start, end = match.span()
            while end - start > self.max_chars:
                spans.append((start, start + self.max_chars))
                start += self.max_chars
            if end > start:
                spans.append((start, end))
        return spans or [(0, len(text))]
