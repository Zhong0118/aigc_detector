"""从文本类文件里抽出可检测的纯文本。

检测器、指纹和 Hive 文本接口都应该吃同一份字符串，
避免 .docx 被当成二进制乱码、或指纹和检测各读一遍导致不一致。
"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_DOCX_SUFFIXES = {".docx"}
_PLAIN_SUFFIXES = {".txt", ".md", ".csv", ".json", ".html"}


def extract_document_text(path: str | Path) -> str:
    """按扩展名抽取文本。docx 走 OOXML，其余按 utf-8/gbk 读。"""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in _DOCX_SUFFIXES:
        return _extract_docx_text(path)
    return _read_plain_text(path)


def is_document_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in _DOCX_SUFFIXES | _PLAIN_SUFFIXES


def _read_plain_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def _extract_docx_text(path: Path) -> str:
    """从 Word 文档抽出段落文本，忽略样式、图片和页眉页脚。"""
    with ZipFile(path) as archive:
        try:
            xml = archive.read("word/document.xml")
        except KeyError as exc:
            raise ValueError(f"Not a valid .docx file: {path.name}") from exc

    root = ET.fromstring(xml)
    paragraphs: list[str] = []
    for paragraph in root.iter(f"{_W}p"):
        parts: list[str] = []
        for node in paragraph.iter(f"{_W}t"):
            if node.text:
                parts.append(node.text)
            if node.tail:
                parts.append(node.tail)
        line = "".join(parts).strip()
        if line:
            paragraphs.append(line)
    if not paragraphs:
        raise ValueError(f"No readable text in {path.name}")
    return "\n".join(paragraphs)
