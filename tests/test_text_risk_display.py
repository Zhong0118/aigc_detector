"""The report and the Streamlit header use the same text-risk vocabulary."""
from pathlib import Path
import textwrap

import pytest
from streamlit.testing.v1 import AppTest

from reports.generator import ReportGenerator


STATES = (
    ("uncertain", 0.858, "严格判定不确定 · 未触发复核"),
    ("uncertain", 0.9029, "不确定 · 疑似 AI，需复核"),
    ("high_risk_ai", 0.96, "高风险疑似 AI"),
    ("low_risk", 0.15, "低风险（更偏人工）"),
    ("insufficient_evidence", 0.999, "未校准 / 证据不足"),
)


def sample(label, score):
    return {
        "modality": "text", "content_id": 1,
        "detection": {
            "score": score, "label": label, "threshold": 0.9456, "providers": [],
            "model_scores": {},
            "decision": {
                "decision": label, "threshold_group": "zh_long", "screening_threshold": 0.9029,
                "low_threshold": 0.4, "high_threshold": 0.9456, "high_risk_coverage": 0.6,
                "segments": [{"start": 0, "end": 60, "score": 0.95},
                             {"start": 40, "end": 100, "score": 0.93},
                             {"start": 70, "end": 100, "score": 0.9456}],
                "uncertainty_reasons": ["model_branch_not_calibrated"] if label == "insufficient_evidence" else [],
            },
        },
        "provenance": {"deep_triggered": False, "c2pa": {}, "watermark": {}, "fingerprint_registry": {}},
        "report": {"summary": "历史模板", "evidence": [], "limitations": [], "recommendation": ""},
    }


@pytest.mark.parametrize(("label", "score", "expected"), STATES)
def test_report_promotes_conclusion_and_preserves_machine_fields(label, score, expected):
    data = sample(label, score)
    report = ReportGenerator(config={"report": {"provider": "template"}}).generate(data)
    assert report["summary"].startswith(expected)
    assert set(report) >= {"provider", "status", "review_label", "summary", "evidence", "limitations", "recommendation"}
    assert data["detection"]["score"] == score
    if label == "insufficient_evidence":
        assert "99.90%" not in str(report)
        return
    assert f"{score:.2%}" not in report["summary"]
    evidence = " ".join(report["evidence"])
    assert all(word in evidence for word in ("zh_long", "90.29%", "94.56%", "1/3", "60.00%", "模型校准风险分"))


@pytest.mark.parametrize(("label", "score", "expected"), STATES)
def test_dashboard_uses_conclusion_as_header(label, score, expected):
    path = Path(__file__).resolve().parents[1] / "api" / "dashboard.py"
    data = sample(label, score)
    script = textwrap.dedent(f"""\
        import runpy
        view = runpy.run_path({str(path)!r}, run_name="text_dashboard_test")
        view["render_analysis"]({data!r})
    """)
    app = AppTest.from_string(script).run(timeout=20)
    assert not app.exception
    headings = [h.value for h in [*app.header, *app.subheader]]
    assert expected in headings
    captions = [c.value for c in app.caption]
    assert all("AI率" not in line for line in [*headings, *captions])
    if label == "insufficient_evidence":
        assert not app.get("plotly_chart")
        assert all("99.90%" not in line for line in captions)
        return
    assert any("模型校准风险分" in line and f"{score:.2%}" in line for line in captions)
    assert any("单块超过严格线 94.56%" in line for line in captions)
    charts = [chart.proto.spec for chart in app.get("plotly_chart")]
    assert any("窗口风险分=%{y:.1%}" in chart for chart in charts)
    assert all("AI率" not in chart for chart in charts)
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["复核线"] == "90.29%"
    assert metrics["严格线"] == "94.56%"
    assert metrics["严格线以上窗口"] == "1 / 3"
    assert metrics["去重叠高分字符覆盖率"] == "60.00%"
