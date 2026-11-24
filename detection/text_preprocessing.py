"""Conservatively remove structured front and back matter before text scoring."""
from dataclasses import dataclass
import re
import unicodedata

RULE_VERSION = "structured-text-v1"

_REFERENCE_HEADING = re.compile(
    r"(?:参考文献|参考书目|参考资料|主要参考文献|references|bibliography)(?:\s*\(references?\))?\s*[:：]?",
    re.I,
)
_ENTRY = re.compile(r"\[(\d{1,3})\]\s*(.{13,300})")
_YEAR = re.compile(r"(?<!\d)(?:18|19|20)\d{2}(?!\d)")
_CITATION_CUE = re.compile(
    r"\[(?:J|M|D|C|R|S|N|P|EB|A|DS|DB)(?:/OL)?\]|doi\s*:|\b10\.\d{4,9}/\S+|"
    r"https?://|出版社|\d+\s*\(\d+\)|\d+\s*[:：]\s*\d+|\d+\s*[–—-]\s*\d+", re.I,
)
_TOC_HEADING = re.compile(r"目\s*录|(?:table of )?contents", re.I)
_TOC_ROW = re.compile(r"\S.{1,90}(?:[.．…·]{3,}[ \t]*|[ \t]{2,})(?:\d{1,4}|[IVXLCDM]{1,8})", re.I)


@dataclass(frozen=True)
class ExcludedRange:
    kind: str
    start: int
    end: int
    chars: int
    rule: str = RULE_VERSION


@dataclass(frozen=True)
class PreprocessedText:
    text: str
    excluded_ranges: tuple[ExcludedRange, ...]
    retained_ranges: tuple[tuple[int, int], ...]

    def source_ranges(self, start: int, end: int) -> list[tuple[int, int]]:
        """Map a half-open scored span to the exact retained original intervals."""
        if not 0 <= start <= end <= len(self.text):
            raise ValueError("Scored span is outside the retained text")
        ranges = []
        offset = 0
        for left, right in self.retained_ranges:
            width = right-left
            first, last = max(0, start-offset), min(width, end-offset)
            if first < last:
                ranges.append((left+first, left+last))
            offset += width
        return ranges


def _normalized(line: str) -> str:
    return unicodedata.normalize("NFKC", line).strip()


def complement(length: int, ranges: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """Return the surviving ranges for ordered, non-overlapping exclusions."""
    retained = []
    cursor = 0
    for start, end in ranges:
        if not cursor <= start < end <= length:
            raise ValueError("Invalid excluded text range")
        if cursor < start:
            retained.append((cursor, start))
        cursor = end
    if cursor < length:
        retained.append((cursor, length))
    return tuple(retained)


def preprocess_text(text: str) -> PreprocessedText:
    """Retain ambiguous text, including inline citations and references followed by prose."""
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1]+len(line))
    rows = [_normalized(line) for line in lines]
    candidates = []
    for index, line in enumerate(rows[:21]):
        if offsets[index] > len(text)*.2:
            break
        if not _TOC_HEADING.fullmatch(line):
            continue
        cursor, entries = index+1, 0
        while cursor < len(rows):
            if _TOC_ROW.fullmatch(rows[cursor]):
                entries += 1
            elif rows[cursor]:
                break
            cursor += 1
        if entries >= 4 and cursor < len(rows):
            candidates.append(("table_of_contents", offsets[index], offsets[cursor]))
        break

    for index in range(len(rows)-1, -1, -1):
        if not _REFERENCE_HEADING.fullmatch(rows[index]):
            continue
        entries = [_ENTRY.fullmatch(line) for line in rows[index+1:] if line]
        if len(entries) >= 4 and all(entries):
            numbers = [int(entry[1]) for entry in entries]
            if (numbers == list(range(1, len(numbers)+1))
                    and all(_YEAR.search(entry[2]) and _CITATION_CUE.search(entry[2]) for entry in entries)):
                candidates.append(("references", offsets[index], len(text)))
        break

    ranges = complement(len(text), [(start,end) for _,start,end in candidates])
    clean = "".join(text[start:end] for start,end in ranges)
    if not clean.strip():
        return PreprocessedText(text, (), ((0,len(text)),))
    exclusions = tuple(ExcludedRange(kind,start,end,sum(not ch.isspace() for ch in text[start:end]))
                       for kind,start,end in candidates)
    return PreprocessedText(clean, exclusions, ranges)
