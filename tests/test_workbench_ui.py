"""Drive the real workbench pages with a mocked HTTP layer (Streamlit AppTest)."""
import hashlib
import json
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from tests.test_workbench_image_engine import FakeModels, acceptance, png, settings, stored
from workbench import export
from workbench.image_engine import CONDITIONAL_STATEMENT, ImageService


def image_result(score=0.9, **model_flags):
    svc = ImageService(settings(), models=FakeModels(score=score, **model_flags), acceptance=acceptance("research-dna-lp22"))
    return svc.analyze(png(), persist=stored)[0]


class FakeAPI:
    def __init__(self, monkeypatch, score=0.9, status=200, error=None, **model_flags):
        self.posts, self.score, self.status, self.error, self.flags = [], score, status, error, model_flags
        self.caps = ImageService(settings(), models=FakeModels(), acceptance=acceptance("research-dna-lp22")).capabilities()
        self.reports, self.report_params = {}, []
        monkeypatch.setattr("requests.request", self.reply)
        from workbench.ui.common import capabilities
        capabilities.clear()

    def reply(self, method, url, **kwargs):
        if url.endswith("/capabilities"):
            return self._ok(dict(preview=True, text={"available": True}, image=self.caps, history={}, experiments={}))
        if url.endswith("/analyze/file"):
            self.posts.append(kwargs.get("data") or {})
            if self.error == "timeout":
                import requests
                raise requests.Timeout()
            if self.status != 200:
                return self._ok({"error": dict(code="model_unavailable", message="未加载", request_id="r", retryable=False)},
                                self.status)
            data = kwargs["files"]["file"][1]
            result = image_result(self.score, **self.flags)
            result["request_id"] = kwargs["data"]["request_id"]
            result["input"]["sha256"] = hashlib.sha256(data).hexdigest()
            return self._ok({"image_result": result, "modality": "image"})
        if "/reports" in url:
            self.report_params.append(dict(kwargs.get("params") or {}))
            items = [dict(content_id=k, created_at="2026-10-30T00:00:00Z", modality=v[0], filename="a.png", status=v[2],
                          label="ai", score=0.9, bundle_id="image-standard-q179-v1") for k, v in self.reports.items()]
            return self._ok(dict(items=items, total=len(items), limit=15, offset=0))
        if "/report/" in url:
            kind, payload, _ = self.reports[int(url.rsplit("/", 1)[1])]
            return self._ok(payload)
        raise AssertionError(url)

    @staticmethod
    def _ok(payload, status=200):
        return SimpleNamespace(status_code=status, json=lambda: payload, text=json.dumps(payload))


def image_app(data=None):
    def script():
        import streamlit as st
        from workbench.ui.common import inject_css
        from workbench.ui.pages import image_page, text_page
        inject_css()
        (text_page if st.session_state.get("page") == "text" else image_page)()
    app = AppTest.from_function(script)
    data = data if data is not None else png()
    app.session_state["img_file"] = dict(name="a.png", data=data, sha256=hashlib.sha256(data).hexdigest())
    return app


def text_of(app):
    return "\n".join(m.value for m in app.markdown) + "\n".join(c.value for c in app.caption)


def button(app, label):
    return next(b for b in app.button if b.label == label)


def test_success_renders_semantics_and_reuses_result(monkeypatch):
    api = FakeAPI(monkeypatch)
    app = image_app()
    app.run(timeout=30)
    assert not button(app, "开始分析").disabled
    button(app, "开始分析").click().run(timeout=30)
    assert not app.exception
    text = text_of(app)
    assert "AI 生成倾向" in text and "AI 生成分数 90.00%" in text and "越高越像 AI 生成" in text and CONDITIONAL_STATEMENT in text
    assert "不可用 · score = null" in text and "报告已保存" in text
    assert len(api.posts) == 1 and api.posts[0]["mode"] == "standard" and "research_method_id" not in api.posts[0]
    button(app, "开始分析").click().run(timeout=30)
    assert len(api.posts) == 1  # same image + parameters: stored result, no second inference


def test_below_threshold_skips_source(monkeypatch):
    FakeAPI(monkeypatch, score=0.1)
    app = image_app()
    app.run(timeout=30)
    button(app, "开始分析").click().run(timeout=30)
    text = text_of(app)
    assert "真实图片倾向" in text and "未运行：检测分数低于阈值" in text and CONDITIONAL_STATEMENT not in text
    assert "AI 生成分数 10.00%" in text and "不是“真实图片”的置信度" in text  # low AI score is not real-confidence
    assert "det-openclip-q179-v1" not in "\n".join(m.value for m in app.markdown if "wb-card" in m.value)


def test_partial_keeps_detection(monkeypatch):
    FakeAPI(monkeypatch, attribution_fails=True)
    app = image_app()
    app.run(timeout=30)
    button(app, "开始分析").click().run(timeout=30)
    assert any("部分成功" in w.value for w in app.warning)
    assert "AI 生成倾向" in text_of(app) and "来源模型推理失败" in text_of(app)


def test_failure_is_error_not_result(monkeypatch):
    FakeAPI(monkeypatch, status=503)
    app = image_app()
    app.run(timeout=30)
    button(app, "开始分析").click().run(timeout=30)
    assert any("模型不可用" in e.value for e in app.error)
    assert "AI 生成分数" not in text_of(app) and not button(app, "开始分析").disabled


def test_timeout_does_not_resubmit_and_retry_reuses_request_id(monkeypatch):
    api = FakeAPI(monkeypatch, error="timeout")
    app = image_app()
    app.run(timeout=30)
    button(app, "开始分析").click().run(timeout=30)
    assert any("结果暂未收到" in e.value for e in app.error) and len(api.posts) == 1
    app.run(timeout=30)  # a plain rerun never re-submits
    assert len(api.posts) == 1
    api.error = None
    button(app, "重新查询（同一请求 ID）").click().run(timeout=30)
    assert len(api.posts) == 2 and api.posts[0]["request_id"] == api.posts[1]["request_id"]
    assert "AI 生成倾向" in text_of(app)


def test_navigation_round_trip_keeps_result_without_reinference(monkeypatch):
    api = FakeAPI(monkeypatch)
    app = image_app()
    app.run(timeout=30)
    button(app, "开始分析").click().run(timeout=30)
    app.session_state["page"] = "text"
    app.run(timeout=30)
    assert "文本检测" in text_of(app)
    app.session_state["page"] = "image"
    app.run(timeout=30)
    assert "AI 生成倾向" in text_of(app) and len(api.posts) == 1


def test_changed_image_never_shows_old_result(monkeypatch):
    FakeAPI(monkeypatch)
    app = image_app()
    app.run(timeout=30)
    button(app, "开始分析").click().run(timeout=30)
    other = png(size=(80, 20))
    app.session_state["img_file"] = dict(name="b.png", data=other, sha256=hashlib.sha256(other).hexdigest())
    app.run(timeout=30)
    assert "AI 生成倾向" not in text_of(app) and 'class="wb-verdict' not in text_of(app) and "已选择图片" in text_of(app)


def test_clear_drops_file_and_session_results(monkeypatch):
    FakeAPI(monkeypatch)
    app = image_app()
    app.run(timeout=30)
    button(app, "开始分析").click().run(timeout=30)
    button(app, "清空").click().run(timeout=30)
    assert "img_file" not in app.session_state and app.session_state["img_results"] == {}
    assert button(app, "开始分析").disabled


def test_buttons_locked_while_pending(monkeypatch):
    FakeAPI(monkeypatch)
    monkeypatch.setattr("workbench.ui.pages._finish_pending", lambda: None)
    monkeypatch.setattr("streamlit.rerun", lambda: None)
    app = image_app()
    app.session_state["img_pending"] = dict(key=("x",), request_id="r", job=dict(done=False))
    app.run(timeout=30)
    assert button(app, "分析中…").disabled and button(app, "清空").disabled


def test_research_mode_sends_one_allowlisted_method(monkeypatch):
    api = FakeAPI(monkeypatch)
    app = image_app()
    app.session_state["img_mode_saved"] = "研究模式"
    app.run(timeout=30)
    assert [o for o in app.selectbox[0].options] == ["DNA-Det 冻结骨干 + 22 类线性头", "A0 来源排序（不经检测门控）",
                                                     "D-OpenCLIP 已登记校准阈值（1% / 5%）"]
    button(app, "开始分析").click().run(timeout=30)
    assert api.posts[0]["mode"] == "research" and api.posts[0]["research_method_id"] == "research-dna-lp22"


def test_model_not_ready_disables_submit(monkeypatch):
    api = FakeAPI(monkeypatch)
    api.caps["state"] = "loading"
    app = image_app()
    app.run(timeout=30)
    assert button(app, "开始分析").disabled


HIST_TABLE = None  # set lazily: the selection key is derived from the active filters


def test_history_restores_snapshot_and_legacy(monkeypatch):
    global HIST_TABLE
    from workbench.ui.pages import _filter_key
    HIST_TABLE = "hist_table_" + _filter_key(dict(limit=15, offset=0))
    api = FakeAPI(monkeypatch)
    snap = image_result()
    api.reports = {2: ("image", dict(image_result=snap, report_snapshot={}, summary={}), "ok"),
                   1: ("text", dict(detection_results=[{"score": 0.1}], image_result=None, report_snapshot=None), "legacy")}

    def script():
        from workbench.ui.pages import history_page
        history_page()
    app = AppTest.from_function(script)
    app.run(timeout=30)
    assert not app.exception and app.dataframe
    app.session_state[HIST_TABLE] = {"selection": {"rows": [0], "columns": []}}
    app.run(timeout=30)
    text = text_of(app)
    assert "历史快照" in text and "不会改写或重算" in text
    app.session_state[HIST_TABLE] = {"selection": {"rows": [1], "columns": []}}
    app.run(timeout=30)
    assert any("旧记录" in i.value for i in app.info)


def test_markdown_export_keeps_semantics():
    r = image_result()
    md = export.image_markdown(r)
    assert "AI 生成倾向" in md and "不是校准概率" in md and "score = null" in md and CONDITIONAL_STATEMENT in md
    assert "/home/" not in md and "未重新归一化" in md
    low_result = image_result(score=0.2)
    low_result["runtime"]["timing_ms"].update(attribution_ms=None, research_ms=None)
    low = export.image_markdown(low_result)
    assert "未运行来源模型" in low and "真实图片倾向" in low
    assert "来源 未运行，研究 未运行" in low and "None" not in low


@pytest.mark.skipif(not __import__("pathlib").Path(__file__).resolve().parents[1].joinpath(
    "docs/image_research_results/2026-10-30T0444Z-final/overview.json").is_file(), reason="sealed overview not present")
def test_experiments_page_shows_denominators_and_status(monkeypatch):
    from pathlib import Path
    from tests.test_workbench_catalog import CATALOG
    from workbench.catalog import build_catalog
    catalog = build_catalog(dict(research_root=str(Path(__file__).resolve().parents[1]), catalog=CATALOG))
    api = FakeAPI(monkeypatch)
    original = api.reply
    monkeypatch.setattr("requests.request", lambda method, url, **kw: FakeAPI._ok(catalog) if url.endswith("/experiments")
                        else original(method, url, **kw))

    def script():
        from workbench.ui.pages import experiments_page
        experiments_page()
    app = AppTest.from_function(script)
    app.run(timeout=30)
    assert not app.exception
    frames = "\n".join(df.value.to_csv() for df in app.dataframe)
    assert "19/979" in frames and "15/93" in frames and "56/93" in frames and "7/93" in frames
    assert "58/1212" in frames and "28/1212" in frames and "0/63" in frames
    assert "未接入（已有离线结果）" in frames and "不提供入口" in frames
    assert "None" not in frames and "计数未登记" in frames  # DINOv2 has rates but no registered counts
    assert "未登记 / 28" in frames and "30 / 29" in frames and "不适用（KNN 无训练轮次）" in frames
    assert "train 37,338 · selection 2,311" in frames and "held-FFHQ" in frames
    text = text_of(app)
    assert "15 张" in text and "58/1212" in text and "7/93" in text  # key limitations stay above the tables


@pytest.mark.skipif(not __import__("pathlib").Path(__file__).resolve().parents[1].joinpath(
    "docs/image_research_results/2026-10-30T0444Z-final/overview.json").is_file(), reason="sealed overview not present")
def test_experiments_page_separates_prototype_epochs_from_its_support_head(monkeypatch):
    from pathlib import Path
    from tests.test_workbench_catalog import CATALOG
    from workbench.catalog import build_catalog
    catalog = build_catalog(dict(research_root=str(Path(__file__).resolve().parents[1]), catalog=CATALOG))
    api = FakeAPI(monkeypatch)
    original = api.reply
    monkeypatch.setattr("requests.request", lambda method, url, **kw: FakeAPI._ok(catalog) if url.endswith("/experiments")
                        else original(method, url, **kw))

    def script():
        from workbench.ui.pages import experiments_page
        experiments_page()
    app = AppTest.from_function(script)
    app.run(timeout=30)
    assert not app.exception
    unknown = next(df.value for df in app.dataframe if "OpenCLIP open20 原型余弦" in set(df.value.get("方法", [])))
    rows = unknown.set_index("方法")
    proto, msp = rows.loc["OpenCLIP open20 原型余弦"], rows.loc["OpenCLIP open20 MSP"]
    assert proto["训练 / 选模 epoch"].startswith("不适用") and "30" not in proto["训练 / 选模 epoch"]
    assert proto["配套分类头 epoch"] == "30 / 29" and msp["配套分类头 epoch"] == "—"
    assert msp["训练 / 选模 epoch"] == "30 / 29"
    assert "train 特征" in proto["说明"] and "线性分类头" in proto["说明"]
    text = text_of(app)
    assert "原型拒识分数本身没有训练轮次" in text and "30 / 29 属于配套 open20 线性分类头，该分类头提供已知类预测" in text
    assert "）（" not in text and "））" not in text  # no nested brackets in the caption


def test_history_date_filter_sends_utc_half_open_bounds_and_resets_page(monkeypatch):
    import datetime as dt
    api = FakeAPI(monkeypatch)

    def script():
        from workbench.ui.pages import history_page
        history_page()
    app = AppTest.from_function(script)
    app.run(timeout=30)
    assert "created_from" not in api.report_params[-1] and "created_to" not in api.report_params[-1]
    assert any("没有" not in i.value and "暂无记录" in i.value for i in app.info)
    app.session_state["hist_page"] = 2
    app.date_input(key="hist_from").set_value(dt.date(2026, 10, 30)).run(timeout=30)
    assert app.session_state["hist_page"] == 0
    assert api.report_params[-1]["created_from"] == "2026-10-30T00:00:00Z" and "created_to" not in api.report_params[-1]
    app.date_input(key="hist_to").set_value(dt.date(2026, 10, 31)).run(timeout=30)
    params = api.report_params[-1]
    assert params["created_to"] == "2026-11-01T00:00:00Z" and params["offset"] == 0  # end day inclusive
    assert any("没有符合筛选条件的记录" in i.value for i in app.info)
    assert "UTC" in text_of(app)


def test_rapid_double_submit_keeps_the_first_job(monkeypatch):
    """Two click callbacks before the rerun: one job, same request_id, the first job is not replaced."""
    from workbench.ui import pages
    jobs = []
    monkeypatch.setattr(pages, "start_job", lambda fn: jobs.append(fn) or dict(done=False, n=len(jobs)))
    data = png()
    state = dict(img_file=dict(name="a.png", data=data, sha256=hashlib.sha256(data).hexdigest()),
                 img_mode_saved="标准模式")
    monkeypatch.setattr(pages, "S", state)
    pages._submit_image()
    first = state["img_pending"]
    pages._submit_image()
    assert len(jobs) == 1 and state["img_pending"] is first and first["job"]["n"] == 1
