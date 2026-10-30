"""HTTP contract: image_result + legacy projection, typed errors, history snapshots, isolation."""
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from tests.test_workbench_image_engine import FakeModels, OVERLAY, acceptance, image_bytes, png, settings
from workbench.image_engine import ImageService

RID = "00000000-0000-4000-8000-00000000c0de"


@pytest.fixture
def client(tmp_path, monkeypatch):
    db = tmp_path / "preview.db"
    config = tmp_path / "config.yaml"
    config.write_text(json.dumps({"database": {"path": str(db)}, "workbench": dict(settings())}))
    monkeypatch.setenv("AIGC_CONFIG", str(config))
    from storage import database
    database.close_db()
    import api.analysis as analysis
    import api.workbench_routes as routes

    class FakeText:
        calls = []

        def providers(self):
            return {"mode": "api_first", "providers": [{"name": "local", "type": "local", "enabled": True,
                                                        "configured": True}]}

        def analyze_upload(self, filename, content):
            self.calls.append(filename)
            return {"filename": filename, "modality": "text", "detection": {"score": 0.1, "label": "low_risk"}}

    text = FakeText()
    monkeypatch.setattr(analysis, "get_analysis_service", lambda: text)
    models = FakeModels()
    monkeypatch.setattr(routes, "_image_service", ImageService(settings(), models=models,
                                                               acceptance=acceptance("research-dna-lp22")))

    def no_network(*args, **kwargs):
        raise AssertionError("the image path must not call external services")
    monkeypatch.setattr("requests.sessions.Session.request", no_network)
    from api.main import app
    with TestClient(app) as test_client:
        yield test_client, routes, text, models, db
    database.close_db()


def upload(test_client, data, name="a.png", path="/api/v1/analyze/file", **form):
    return test_client.post(path, files={"file": (name, data)}, data=form)


def test_image_upload_returns_contract_projection_and_snapshot(client):
    test_client, _, text, _, db = client
    response = upload(test_client, png(), name="C:\\Users\\me\\secret dir\\cat.png", request_id=RID)
    assert response.status_code == 200
    body = response.json()
    r = body["image_result"]
    assert r["request_id"] == RID and r["status"] == "ok" and r["persistence"]["status"] == "ok"
    assert body["content_id"] == r["persistence"]["content_id"] and body["filename"] == "cat.png"
    assert body["detection"]["label"] == "ai" and body["detection"]["score"] == r["detection"]["score"]
    assert body["provenance"]["attribution"]["top_k"][0]["model"] == r["attribution"]["top_k"][0]["source"]
    assert body["report"]["provider"] == "template" and body["deduplicated"] is False and not text.calls
    detail = test_client.get(f"/api/v1/report/{body['content_id']}").json()
    assert detail["image_result"] == r and detail["report_snapshot"]["schema_version"] == "image-report-snapshot-v1"
    assert detail["content"]["filename"] == "cat.png" and detail["summary"]["status"] == "ok"
    with sqlite3.connect(db) as conn:
        dump = "\n".join(conn.iterdump())
    assert "secret dir" not in dump and "Users" not in dump and png()[16:40].hex() not in dump
    assert r["input"]["sha256"] in dump  # Content.file_hash holds the original-byte SHA


def test_low_score_projects_to_real_tendency_without_source(client):
    test_client, routes, *_ = client
    routes._image_service.models.score = 0.2
    body = upload(test_client, png()).json()
    assert body["detection"]["label"] == "human" and body["image_result"]["detection"]["label"] == "real"
    assert body["provenance"]["attribution"]["status"] == "skipped" and body["provenance"]["attribution"]["top_k"] == []


def test_dedupe_and_conflict_over_http(client):
    test_client, _, _, models, _ = client
    first = upload(test_client, png(), request_id=RID).json()
    again = upload(test_client, png(), request_id=RID).json()
    assert again["deduplicated"] is True and again["image_result"] == first["image_result"]
    assert models.calls["detect"] == 1
    conflict = upload(test_client, png(size=(70, 40)), request_id=RID)
    assert conflict.status_code == 409 and conflict.json()["error"]["code"] == "request_id_conflict"


@pytest.mark.parametrize("data,name,form,status,code", [
    (b"", "a.png", {}, 400, "empty_file"),
    (b"not really a png", "a.png", {}, 400, "decode_failed"),
    (png(mode="RGBA"), "a.png", {}, 400, "unsupported_image_policy"),
    (image_bytes(fmt="GIF"), "x.bin", {}, 415, "unsupported_media_type"),
    (b"\xff\xd8\xff" + b"\0" * (10 * 1024 * 1024 + 10), "big.jpg", {}, 413, "too_large"),
    (png(), "a.png", {"mode": "default"}, 422, "unsupported_mode"),
    (png(), "a.png", {"mode": "research", "research_method_id": "research-ofa-pair"}, 422, "unknown_method_id"),
    (png(), "a.png", {"request_id": "abc"}, 422, "invalid_request_id"),
])
def test_typed_errors_never_fall_back(client, data, name, form, status, code):
    test_client, _, text, _, _ = client
    response = upload(test_client, data, name=name, **form)
    assert response.status_code == status
    body = response.json()
    assert body["error"]["code"] == code and set(body["error"]) == {"code", "message", "request_id", "retryable"}
    assert "detection" not in body and "/home" not in json.dumps(body) and not text.calls


def test_model_failure_and_busy_are_errors_not_human(client):
    test_client, routes, *_ = client
    routes._image_service.models.detect_fails = True
    for path in ("/api/v1/analyze/file", "/api/v1/detect", "/api/v1/provenance"):
        response = upload(test_client, png(), path=path)
        assert response.status_code == 500 and response.json()["error"]["code"] == "inference_failed"
        assert "human" not in response.text
    routes._image_service.models.detect_fails = False
    for _ in range(2):
        routes._image_service._gate.acquire()
    response = upload(test_client, png())
    assert response.status_code == 503 and response.json()["error"]["code"] == "busy"
    assert response.json()["error"]["retryable"] is True
    routes._image_service.state, routes._image_service.error = "unavailable", None
    response = upload(test_client, png(), path="/api/v1/provenance")
    assert response.status_code == 503 and response.json()["error"]["code"] == "model_unavailable"


def test_storage_failure_returns_partial_without_content_id(client, monkeypatch):
    test_client, *_ = client
    monkeypatch.setattr("workbench.history.save_image", lambda result, filename: (_ for _ in ()).throw(OSError("ro")))
    body = upload(test_client, png()).json()
    r = body["image_result"]
    assert r["status"] == "partial" and r["persistence"] == dict(status="error", content_id=None, reason_code="storage_failed")
    assert body["content_id"] is None and r["detection"]["status"] == "ok"


def test_research_upload_and_text_research_rejection(client):
    test_client, _, text, _, _ = client
    body = upload(test_client, png(), mode="research", research_method_id="research-dna-lp22").json()
    assert body["image_result"]["research"][0]["method_id"] == "research-dna-lp22"
    response = upload(test_client, b"plain text", name="note.txt", mode="research", research_method_id="research-dna-lp22")
    assert response.status_code == 422 and response.json()["error"]["code"] == "unsupported_mode" and not text.calls
    assert upload(test_client, b"plain text", name="note.txt").json()["modality"] == "text" and text.calls == ["note.txt"]


def test_detect_and_provenance_aliases_use_the_same_image_branch(client):
    test_client, *_ = client
    detect = upload(test_client, png(), path="/api/v1/detect").json()
    assert detect["image_result"]["mode"] == "standard"
    prov = upload(test_client, png(), path="/api/v1/provenance").json()
    assert prov["modality"] == "image" and prov["detection_score"] == detect["detection"]["score"]


def test_reports_paging_filters_and_not_found(client):
    test_client, routes, *_ = client
    for i in range(3):
        upload(test_client, png(size=(40 + i, 40)))
    routes._image_service.models.attribution_fails = True
    upload(test_client, png(size=(90, 40)))
    page = test_client.get("/api/v1/reports", params=dict(limit=2, offset=0, modality="image")).json()
    assert page["total"] == 4 and len(page["items"]) == 2 and page["items"][0]["status"] == "partial"
    assert test_client.get("/api/v1/reports", params=dict(status="partial")).json()["total"] == 1
    assert test_client.get("/api/v1/reports", params=dict(limit=2, offset=2)).json()["items"][-1]["content_id"] == 1
    missing = test_client.get("/api/v1/report/9999")
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "report_not_found"
    assert missing.json()["detail"] == "Content not found"


def test_capabilities_are_path_free(client):
    test_client, *_ = client
    caps = test_client.get("/api/v1/capabilities").json()
    text = json.dumps(caps)
    assert caps["image"]["state"] == "ready" and caps["history"]["stores_original_images"] is False
    assert "/home/" not in text and "nonexistent" not in text and "api_key" not in text.lower()
    assert caps["image"]["bundle_id"] == OVERLAY["registry"]["bundle_id"]
