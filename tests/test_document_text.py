from __future__ import annotations

import io
import zipfile
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring

from ingestion.document_text import extract_document_text, is_document_path
from ingestion.loader import detect_modality


_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _write_docx(path: Path, paragraphs: list[str]) -> None:
    document = Element(f"{_W}document")
    body = SubElement(document, f"{_W}body")
    for text in paragraphs:
        paragraph = SubElement(body, f"{_W}p")
        run = SubElement(paragraph, f"{_W}r")
        node = SubElement(run, f"{_W}t")
        node.text = text
    xml = tostring(document, encoding="utf-8", xml_declaration=True)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types></Types>")
        archive.writestr("word/document.xml", xml)
    path.write_bytes(buffer.getvalue())


def test_is_document_path_includes_doc_and_docx() -> None:
    assert is_document_path("paper.doc")
    assert is_document_path("paper.docx")
    assert detect_modality(Path("paper.doc")) == "text"


def test_docx_text_extraction(tmp_path: Path) -> None:
    path = tmp_path / "sample.docx"
    _write_docx(path, ["第一段", "第二段"])
    assert extract_document_text(path) == "第一段\n第二段"


def test_doc_suffix_with_docx_payload_does_not_need_libreoffice(tmp_path: Path) -> None:
    path = tmp_path / "legacy.doc"
    _write_docx(path, ["旧后缀但实际是 docx"])
    assert extract_document_text(path) == "旧后缀但实际是 docx"
