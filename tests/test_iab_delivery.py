import json
from pathlib import Path
import pytest


def test_http_range_rejects_full_response_before_reading_body(monkeypatch):
    import requests
    from scripts.download_iab_subset import Ranges
    class Response:
        status_code=200;headers={}
        def __enter__(self):return self
        def __exit__(self,*args):pass
        @property
        def content(self):raise AssertionError('must not materialize full response')
    class Session:
        def get(self,*args,**kwargs):
            assert kwargs.get('stream') is True
            return Response()
    monkeypatch.setattr(requests,'Session',Session)
    monkeypatch.setattr('scripts.download_iab_subset.time.sleep',lambda seconds:None)
    with pytest.raises(RuntimeError,match='ValueError'):
        Ranges([{'id':1,'size':10}],{1:'https://example.invalid'})(0,0,2)


def test_receipt_write_preserves_previous_json_if_replace_fails(tmp_path,monkeypatch):
    from scripts.download_iab_subset import write_json
    path=tmp_path/'receipt.json';path.write_text('{"old":true}')
    replace=Path.replace
    def fail(self,target):
        if target==path:raise OSError('simulated interruption')
        return replace(self,target)
    monkeypatch.setattr(Path,'replace',fail)
    with pytest.raises(OSError):write_json(path,{'new':True})
    assert json.loads(path.read_text())=={'old':True}


def test_stratified_attribution_and_truth_metrics_are_separate():
    import numpy as np
    from scripts.run_image_experiment import stratified_metrics
    rows=[{'semantic_class':'cat','source_label':'a'}, {'semantic_class':'dog','source_label':'b'},
          {'semantic_class':'cat','source_label':None}, {'semantic_class':'dog','source_label':None}]
    scores=np.array([[.1,.9],[.7,.3],[.9,.1],[.4,.6]])
    d=stratified_metrics(rows,[1,1,0,0],['real','ai'],scores,'detection')
    assert d['per_source']['a']['tpr_0_5']==1
    assert d['per_source']['b']['tpr_0_5']==0
    assert d['per_semantic']['cat']['auroc']==1
    a=stratified_metrics(rows[:2],[0,1],['a','b'],np.array([[.8,.2],[.7,.3]]),'attribution')
    assert a['per_source']['b']['recall']==0
    assert a['per_semantic']['dog']['accuracy']==0


def test_reuse_verified_image_hardlinks_legacy_member(tmp_path):
    import hashlib
    import os
    from research.image_attribution.protocol import write_json
    from scripts.download_iab_subset import reuse_verified_image

    old_root = tmp_path / "v1"
    new_root = tmp_path / "v2"
    source, semantic, filename = "model", "cat", "x_p1.png"
    identifier = hashlib.sha256(f"{source}/{semantic}/{filename}".encode()).hexdigest()[:24]
    old_image = old_root / "images" / source / semantic / f"{identifier}.png"
    old_image.parent.mkdir(parents=True)
    old_image.write_bytes(b"verified image bytes")
    row = {
        "sample_id": "iab-" + identifier,
        "relative_path": str(old_image.relative_to(old_root)),
        "sha256": hashlib.sha256(old_image.read_bytes()).hexdigest(),
        "source_label": source,
        "semantic_class": semantic,
        "source_evidence": {"zip_member": filename},
    }
    write_json(old_image.with_suffix(".json"), row)

    reused = reuse_verified_image(old_root, new_root, source, semantic, filename, identifier)

    new_image = new_root / "images" / source / semantic / old_image.name
    assert reused["relative_path"] == str(new_image.relative_to(new_root))
    assert new_image.read_bytes() == old_image.read_bytes()
    assert os.stat(new_image).st_ino == os.stat(old_image).st_ino


def test_catalog_cell_reuses_matching_legacy_catalog(tmp_path, monkeypatch):
    from scripts import download_iab_subset
    from scripts.download_iab_subset import catalog_cell, write_json

    old_root = tmp_path / "v1"
    new_root = tmp_path / "v2"
    parts = [{"id": 9, "label": "model_cat.zip", "size": 10}]
    cached = {
        "source": "model",
        "semantic": "cat",
        "parts": parts,
        "entries": [{"filename": "x_p1.png", "crc32": 7}],
        "unavailable_member_count": 0,
        "range_bytes": 4,
    }
    old_catalog = old_root / "catalogs" / "model__cat.json"
    old_catalog.parent.mkdir(parents=True)
    write_json(old_catalog, cached)
    monkeypatch.setattr(download_iab_subset, "archive_catalog",
                        lambda *args: (_ for _ in ()).throw(AssertionError("catalog should be reused")))

    assert catalog_cell(("model", "cat"), parts, None, new_root, old_root) == cached
    assert (new_root / "catalogs" / old_catalog.name).is_file()


def test_public_links_falls_back_to_stable_dataverse_range_endpoint():
    from scripts.download_iab_subset import PublicLinks

    links = PublicLinks(None)

    assert links[12345] == "https://dataverse.harvard.edu/api/access/datafile/12345"


def test_range_reader_records_retry_timeout_and_effective_throughput(monkeypatch):
    import io
    import requests
    from scripts.download_iab_subset import PublicLinks, Ranges

    class Response:
        status_code = 206
        headers = {"Content-Range": "bytes 0-2/3", "Content-Encoding": "identity"}
        raw = io.BytesIO(b"abc")
        def __enter__(self): return self
        def __exit__(self, *args): pass

    class Session:
        def __init__(self): self.calls = 0
        def get(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise requests.Timeout()
            return Response()

    import requests as requests_module
    session = Session()
    monkeypatch.setattr(requests_module, "Session", lambda: session)
    monkeypatch.setattr("scripts.download_iab_subset.time.sleep", lambda _: None)
    reader = Ranges([{"id": 12345, "size": 3}], PublicLinks(None))

    assert reader(0, 0, 3) == b"abc"
    stats = reader.stats()
    assert stats["requests_attempted"] == 2
    assert stats["requests_succeeded"] == 1
    assert stats["request_failures"] == 1
    assert stats["timeouts"] == 1
    assert stats["bytes"] == 3
    assert stats["elapsed_seconds"] >= 0
    assert stats["throughput_mib_s"] >= 0


def test_real_acquisition_reads_selected_members_without_whole_archive(tmp_path, monkeypatch):
    import hashlib
    import io
    from PIL import Image
    from scripts import download_iab_subset
    from scripts.download_iab_subset import PublicLinks, acquire_cell

    content = io.BytesIO()
    Image.new("RGB", (2, 2), "white").save(content, format="PNG")
    image_bytes = content.getvalue()
    monkeypatch.setattr(download_iab_subset, "read_member", lambda *args: image_bytes)
    monkeypatch.setattr(download_iab_subset, "real_archive",
                        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("whole archive download")),
                        raising=False)
    record = {
        "source": "real", "semantic": "cat",
        "parts": [{"id": 12345, "size": 3}],
    }
    (tmp_path / "receipts").mkdir()
    entry = {"filename": "real_cat_1.png", "size": len(image_bytes),
             "compressed_size": 10, "crc32": 0, "flags": 0, "compression": 8,
             "offset": 0, "volume": 0}

    result = acquire_cell(record, [entry], PublicLinks(None), tmp_path)

    assert result["selected"] == 1
    receipt = __import__("json").loads((tmp_path / "receipts/real__cat.json").read_text())
    assert receipt["rows"][0]["source_evidence"]["whole_volume_md5_verified"] is False
    assert receipt["network"]["bytes"] == 0


def test_plan_only_freezes_selection_without_fetching_images(tmp_path, monkeypatch):
    import json
    import sys
    from scripts import download_iab_subset

    metadata = tmp_path/"metadata.json"
    metadata.write_text(json.dumps({"id": 11127188, "version": [3, 2], "files": []}))
    reuse = tmp_path/"v1"
    reuse.mkdir()
    (reuse/"selection.json").write_text(json.dumps({"entries": {}}))
    output = tmp_path/"iab-balanced-v2-200"
    records = {
        ("a", "cat"): [{"filename": "a.png", "size": 11, "compressed_size": 7}],
        ("b", "cat"): [{"filename": "b.png", "size": 13, "compressed_size": 9}],
        ("real", "cat"): [{"filename": "r.png", "size": 17, "compressed_size": 12}],
    }
    monkeypatch.setattr(download_iab_subset, "cells", lambda _: {key: [] for key in records})
    monkeypatch.setattr(download_iab_subset, "catalog_cell",
                        lambda key, *_args, **_kwargs: dict(source=key[0], semantic=key[1],
                                                            entries=records[key]))
    monkeypatch.setattr(download_iab_subset, "acquire_cell",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("image acquisition started")))
    monkeypatch.setattr(sys, "argv", [
        "download_iab_subset.py", "--metadata", str(metadata), "--output", str(output),
        "--reuse-from", str(reuse), "--per-cell", "1", "--plan-only",
    ])

    download_iab_subset.main()

    selection = json.loads((output/"selection.json").read_text())
    assert selection["audit"]["selected_uncompressed_bytes"] == 41
    assert not list((output/"receipts").glob("*.json"))
    assert not list((output/"images").rglob("*.png"))


def test_http_403_is_not_retried_or_logged_with_a_url(monkeypatch):
    import json
    import requests
    from scripts.download_iab_subset import PublicLinks, RangeResponseError, Ranges

    class Response:
        status_code = 403
        headers = {}
        def __enter__(self): return self
        def __exit__(self, *args): pass

    class Session:
        calls = 0
        def get(self, *args, **kwargs):
            self.calls += 1
            return Response()

    session = Session()
    monkeypatch.setattr(requests, "Session", lambda: session)
    reader = Ranges([{"id": 12345, "size": 3}], PublicLinks())
    with pytest.raises(RangeResponseError) as error:
        reader(0, 0, 1)

    assert error.value.status_code == 403
    assert session.calls == 1
    assert reader.stats()["request_failures"] == 1
    assert "dataverse.harvard.edu" not in str(error.value)
