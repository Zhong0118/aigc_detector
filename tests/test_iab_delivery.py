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
