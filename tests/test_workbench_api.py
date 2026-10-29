"""Workbench HTTP contract: image_result + legacy keys, typed errors, history, isolation."""
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tests.test_workbench_image_engine import FakeModels, png
from workbench.image_engine import ImageService


@pytest.fixture
def client(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text(f"database:\n  path: {tmp_path / 'preview.db'}\nworkbench:\n  research_root: /nonexistent\n"
                      "  research_ofa: false\n")
    monkeypatch.setenv("AIGC_CONFIG", str(config))
    from storage import database
    database.close_db()
    import api.analysis as analysis
    import api.workbench_routes as routes

    class NoText:
        def providers(self):
            return {"mode": "api_first", "providers": [{"name": "local", "type": "local", "enabled": True, "configured": True}]}

        def analyze_upload(self, *args):
            raise AssertionError("images must never reach the legacy text/scaffold pipeline")

    monkeypatch.setattr(analysis, "get_analysis_service", lambda: NoText())
    monkeypatch.setattr("api.routes.get_analysis_service", lambda: NoText())
    monkeypatch.setattr(routes, "_image_service", ImageService(dict(research_root="/nonexistent", research_ofa=False),
                                                               models=FakeModels()))
    from api.main import app
    with TestClient(app) as test_client:
        yield test_client, routes, monkeypatch
    database.close_db()


def upload(test_client, data, name="a.png", **form):
    return test_client.post("/api/v1/analyze/file", files={"file": (name, data)}, data=form)


def test_image_upload_returns_legacy_fields_and_image_result_and_history(client):
    test_client, *_ = client
    response = upload(test_client, png(), name="C:\\Users\\me\\secret dir\\cat.png", request_id="req-12345678")
    assert response.status_code == 200
    body = response.json()
    for key in ("filename", "modality", "detection", "provenance", "report", "fingerprint", "metadata"):
        assert key in body
    result = body["image_result"]
    assert body["filename"] == "cat.png" and result["request_id"] == "req-12345678"
    assert result["unknown"]["score"] is None and result["schema_version"] == "image_result.v1"
    page = test_client.get("/api/v1/history", params={"kind": "image"}).json()
    assert page["total"] == 1 and page["items"][0]["display_name"] == "cat.png"
    item = test_client.get(f"/api/v1/history/{result['analysis_id']}").json()
    assert item["snapshot"]["model_bundle"] == result["model_bundle"]
    assert "secret dir" not in str(item)
    again = upload(test_client, png(), request_id="req-12345678").json()
    assert again["image_result"]["deduplicated"]
    assert test_client.get("/api/v1/history").json()["total"] == 1


def test_typed_errors_carry_request_id_and_retryable(client):
    test_client, routes, monkeypatch = client
    bad = upload(test_client, b"\x89PNG\r\n\x1a\n" + b"0" * 40)
    assert bad.status_code in (400, 422)
    error = bad.json()["error"]
    assert set(error) == {"code", "message", "request_id", "retryable"} and error["retryable"] is False
    empty = upload(test_client, b"", name="empty.png")
    assert empty.status_code == 400 and empty.json()["error"]["code"] == "invalid_input"
    missing = test_client.get("/api/v1/history/does-not-exist")
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "report_not_found"
    monkeypatch.setattr(routes, "_image_service", ImageService({}))
    unavailable = upload(test_client, png())
    assert unavailable.status_code == 503 and unavailable.json()["error"]["code"] == "model_unavailable"
    assert test_client.get("/api/v1/history").json()["total"] == 0


def test_capabilities_and_experiments_without_catalog(client):
    test_client, *_ = client
    caps = test_client.get("/api/v1/capabilities").json()
    assert caps["preview"] and caps["image"]["state"] == "ready"
    assert caps["image"]["unknown"] == {"status": "unavailable", "score": None}
    assert caps["history"]["stores_original_images"] is False
    assert test_client.get("/api/v1/experiments").status_code == 503


def test_detect_alias_keeps_working_for_images(client):
    test_client, *_ = client
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buffer, format="JPEG")
    response = test_client.post("/api/v1/detect", files={"file": ("x.jpg", buffer.getvalue())})
    assert response.status_code == 200 and response.json()["modality"] == "image"
