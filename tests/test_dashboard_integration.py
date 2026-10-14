"""Exercise the actual Streamlit UI with API replies supplied locally."""
from pathlib import Path
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest


def test_adjusted_threshold_ui_shows_unmeasured_metrics_and_block_results(monkeypatch):
    providers = {
        "mode": "api_first", "ports": {"api": 8001, "dashboard": 8501},
        "deep_provenance_threshold": 0.6, "providers": [],
    }
    analysis = {
        "modality": "text", "content_id": 1,
        "detection": {
            "score": 0.4, "label": "uncertain", "threshold": 0.8,
            "providers": [], "model_scores": {},
            "decision": {
                "low_threshold": 0.35, "high_threshold": 0.88,
                "threshold_validation_samples": 1000,
                "threshold_ai_validation_samples": 900,
                "threshold_validation_fpr": None, "threshold_ai_recall": None,
                "threshold_validation_auroc": 0.7,
                "threshold_warnings": ["effective_high_threshold_not_evaluated"],
            },
        },
        "provenance": {
            "deep_triggered": False, "attribution_triggered": True,
            "c2pa": {}, "watermark": {}, "fingerprint_registry": {},
            "attribution": {
                "status": "ok", "family_status": "family_only", "confidence": 0.7,
                "top_k": [{"model": "qwen", "probability": 0.7}],
                "blocks": [{"block_indices": [1], "detection_score": 0.9,
                            "top_k": [{"model": "qwen", "probability": 0.7}]}],
            },
        },
        "report": {"summary": "待人工复核", "evidence": [], "limitations": [],
                   "recommendation": "人工复核"},
    }

    def reply(method, url, **kwargs):
        payload = providers if method == "GET" else analysis
        return SimpleNamespace(status_code=200, json=lambda: payload)

    monkeypatch.setattr("requests.request", reply)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "api/dashboard.py"))
    app.run(timeout=20)
    app.text_area[0].input("A local UI regression fixture.")
    app.button[0].click().run(timeout=20)
    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["真人误报率"] == "n/a"
    assert metrics["AI 召回率"] == "n/a"
    assert any("当前运行阈值未完成独立验证" in warning.value for warning in app.warning)
    assert any(header.value == "高分块家族 Top-K" for header in app.subheader)
