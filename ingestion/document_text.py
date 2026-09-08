"""从文本类文件里抽出可检测的纯文本。

检测器、指纹和 Hive 文本接口都应该吃同一份字符串，
避免 .docx/.doc 被当成二进制乱码、或指纹和检测各读一遍导致不一致。
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import BadZipFile, ZipFile

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_DOCX_SUFFIXES = {".docx"}
_DOC_SUFFIXES = {".doc"}
_PLAIN_SUFFIXES = {".txt", ".md", ".csv", ".json", ".html"}
_WORD_SUFFIXES = _DOCX_SUFFIXES | _DOC_SUFFIXES
_LIBREOFFICE_BINARIES = ("soffice", "libreoffice")


def extract_document_text(path: str | Path) -> str:
    """按扩展名抽取文本。docx 走 OOXML，.doc 转成 docx 后再抽，其余按 utf-8/gbk 读。"""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in _DOCX_SUFFIXES:
        return _extract_docx_text(path)
    if suffix in _DOC_SUFFIXES:
        return _extract_doc_text(path)
    return _read_plain_text(path)


def is_document_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in _WORD_SUFFIXES | _PLAIN_SUFFIXES


def _read_plain_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def _extract_doc_text(path: Path) -> str:
    """抽取旧版 Word .doc 文本。

    有些文件虽然后缀是 .doc，实际已经是 OOXML；这种直接当 .docx 读。
    真正的 OLE .doc 则用本机 LibreOffice 转成 .docx 后再抽文本。
    """
    if _looks_like_docx(path):
        return _extract_docx_text(path)

    soffice = _find_libreoffice()
    if soffice is None:
        raise ValueError(
            "Cannot read .doc without LibreOffice/soffice. "
            "Install LibreOffice, or save the file as .docx/.txt first."
        )

    with tempfile.TemporaryDirectory(prefix="aigc-doc-") as tmpdir:
        converted = _convert_doc_to_docx(path, Path(tmpdir), soffice)
        return _extract_docx_text(converted)


def _looks_like_docx(path: Path) -> bool:
    try:
        with ZipFile(path) as archive:
            return "word/document.xml" in archive.namelist()
    except BadZipFile:
        return False


def _find_libreoffice() -> str | None:
    for name in _LIBREOFFICE_BINARIES:
        found = shutil.which(name)
        if found:
            return found
    return None


def _convert_doc_to_docx(path: Path, outdir: Path, soffice: str) -> Path:
    command = [
        soffice,
        "--headless",
        "--norestore",
        "--convert-to",
        "docx",
        "--outdir",
        str(outdir),
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"Timed out converting {path.name} with LibreOffice") from exc

    converted = outdir / f"{path.stem}.docx"
    if completed.returncode != 0 or not converted.exists():
        detail = (completed.stderr or completed.stdout or "").strip()
        raise ValueError(
            f"Failed to convert {path.name} to .docx"
            + (f": {detail}" if detail else "")
        )
    return converted


def _extract_docx_text(path: Path) -> str:
    """从 Word 文档抽出段落文本，忽略样式、图片和页眉页脚。"""
    try:
        archive_ctx = ZipFile(path)
    except BadZipFile as exc:
        raise ValueError(f"Not a valid .docx file: {path.name}") from exc

    with archive_ctx as archive:
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
