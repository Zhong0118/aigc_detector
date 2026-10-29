"""Drive the real workbench pages with a mocked API (AppTest)."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from tests.test_workbench_image_engine import FakeModels, png
from workbench import export
from workbench.image_engine import CONDITIONAL_STATEMENT, ImageService

CAPS = {"preview": True, "text": {"available": True}, "history": {"stores_original_images": False},
        "image": {"state": "ready", "device": "cpu"}}


def fake_api(monkeypatch, analyze_status=200, score=0.9):
    calls = []
    image = png()
    body = ImageService(dict(research_root="/x", research_ofa=False), models=FakeModels(score=score)).analyze(image, "a.png")

    def reply(method, url, **kwargs):
        calls.append((method, url))
        if url.endswith("/capabilities"):
            payload, status = CAPS, 200
        elif url.endswith("/analyze/file") and analyze_status == 200:
            payload, status = {"image_result": body, "modality": "image"}, 200
        else:
            payload = {"error": {"code": "model_unavailable", "message": "未配置", "request_id": "r1", "retryable": False}}
            status = analyze_status
        return SimpleNamespace(status_code=status, json=lambda: payload, text=json.dumps(payload))

    monkeypatch.setattr("requests.request", reply)
    from workbench.ui.common import capabilities
    capabilities.clear()
    return calls, image


def image_app(image):
    def script():
        from workbench.ui.common import inject_css
        from workbench.ui.pages import image_page
        inject_css()
        image_page()
    app = AppTest.from_function(script)
    app.session_state["img_file"] = dict(name="a.png", data=image, sha256=hashlib.sha256(image).hexdigest())
    return app


def markdown_text(app):
    return "\n".join(m.value for m in app.markdown)


def test_image_flow_renders_semantics_and_never_reinfers(monkeypatch):
    calls, image = fake_api(monkeypatch)
    app = image_app(image)
    app.run(timeout=30)
    start = next(b for b in app.button if b.label == "开始分析")
    assert not start.disabled
    start.click().run(timeout=30)
    assert not app.exception
    text = markdown_text(app)
    assert "模型检测分数（不是校准概率）" in text and CONDITIONAL_STATEMENT in text
    assert "未知来源" in text and "不可用" in text
    posts = [c for c in calls if c[0] == "POST"]
    assert len(posts) == 1
    next(b for b in app.button if b.label == "开始分析").click().run(timeout=30)
    assert len([c for c in calls if c[0] == "POST"]) == 1  # same image + mode reuses the stored result


def test_below_threshold_shows_no_source(monkeypatch):
    _, image = fake_api(monkeypatch, score=0.1)
    app = image_app(image)
    app.run(timeout=30)
    next(b for b in app.button if b.label == "开始分析").click().run(timeout=30)
    text = markdown_text(app)
    assert "检测分数低于阈值，主结论不给出来源候选" in text and CONDITIONAL_STATEMENT not in text


def test_model_unavailable_is_an_error_not_a_result(monkeypatch):
    _, image = fake_api(monkeypatch, analyze_status=503)
    app = image_app(image)
    app.run(timeout=30)
    next(b for b in app.button if b.label == "开始分析").click().run(timeout=30)
    assert any("模型不可用" in e.value for e in app.error)
    assert "模型检测分数" not in markdown_text(app)
    assert not next(b for b in app.button if b.label == "开始分析").disabled


def test_button_disabled_until_model_ready(monkeypatch):
    fake_api(monkeypatch)
    monkeypatch.setitem(CAPS["image"], "state", "loading")
    app = image_app(png())
    app.run(timeout=30)
    assert next(b for b in app.button if b.label == "开始分析").disabled


def test_markdown_export_keeps_semantics():
    result = ImageService(dict(research_root="/x", research_ofa=False), models=FakeModels()).analyze(png(), "a.png", research=True)
    text = export.to_markdown("image", result)
    assert "不是校准概率" in text and "score = null" in text and "/home/" not in text
    assert CONDITIONAL_STATEMENT in text and "研究：校准阈值" in text


@pytest.mark.skipif(not (Path(__file__).resolve().parents[2] / "attribution-pilot-baseline/outputs").is_dir(),
                    reason="registered results only on the research server")
def test_experiments_page_shows_denominators(monkeypatch):
    import yaml
    from workbench.catalog import build_catalog
    root = Path(__file__).resolve().parents[1]
    overlay = (root / "configs/workbench/preview.overlay.example.yaml").read_text().replace(
        "${RESEARCH_ROOT}", str(root.parent / "attribution-pilot-baseline"))
    catalog = build_catalog(yaml.safe_load(overlay)["workbench"])

    def reply(method, url, **kwargs):
        payload = CAPS if url.endswith("/capabilities") else catalog
        return SimpleNamespace(status_code=200, json=lambda: payload)

    monkeypatch.setattr("requests.request", reply)

    def script():
        from workbench.ui.pages import experiments_page
        experiments_page()
    app = AppTest.from_function(script)
    app.run(timeout=30)
    assert not app.exception
    frames = "\n".join(df.value.to_csv() for df in app.dataframe)
    assert "（19/979）" in frames and "（15/93）" in frames and "（56/93）" in frames
    assert "576/1212" in frames and "539/1212" in frames and "59/63" in frames


def test_broken_file_preview_does_not_crash_and_shows_typed_error(monkeypatch):
    fake_api(monkeypatch, analyze_status=422)
    broken = b"\x89PNG\r\n\x1a\n" + b"x" * 64
    app = image_app(broken)
    app.run(timeout=30)
    assert not app.exception
    next(b for b in app.button if b.label == "开始分析").click().run(timeout=30)
    assert not app.exception and app.error
