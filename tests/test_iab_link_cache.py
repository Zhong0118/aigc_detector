import io
from datetime import datetime, timezone
from urllib.parse import urlsplit
from types import SimpleNamespace
import pytest
from scripts import download_iab_subset as d

PUBLIC = 'https://dataverse.harvard.edu/api/access/datafile/7'
SIGNED = 'https://dvn-cloud-iqss.s3.amazonaws.com/object?X-Amz-Date=20261020T090000Z&X-Amz-Expires=3600&X-Amz-Signature=SECRET'
NOW = datetime(2026,10,20,9,10,tzinfo=timezone.utc).timestamp()

def response(url=SIGNED, history=True, status=206):
    return SimpleNamespace(url=url, status_code=status, history=[SimpleNamespace(url=PUBLIC,status_code=303)] if history else [])

def test_reuses_only_official_redirect_until_expiry(monkeypatch):
    clock=[NOW];monkeypatch.setattr(d.time,'time',lambda:clock[0])
    links=d.PublicLinks()
    links.remember(7,response())
    assert links[7] == SIGNED
    assert links[8].endswith('/8')
    clock[0]=NOW+50*60
    assert links[7] == PUBLIC

@pytest.mark.parametrize('r',[response('http://dvn-cloud-iqss.s3.amazonaws.com/object'),response('https://evil.invalid/object'),response(history=False),response(status=403)])
def test_rejects_unverified_or_failed_redirect(monkeypatch,r):
    monkeypatch.setattr(d.time,'time',lambda:NOW)
    links=d.PublicLinks();links.remember(7,r)
    assert links[7] == PUBLIC

def test_many_ranges_make_one_dataverse_request_and_do_not_log_url(monkeypatch):
    import requests,json
    monkeypatch.setattr(d.time,'time',lambda:NOW)
    calls=[]
    class Response:
        status_code=206
        headers={'Content-Range':'bytes 0-2/3'}
        url=SIGNED
        def __init__(self,public):
            self.raw=io.BytesIO(b'abc');self.history=response().history if public else []
        def __enter__(self):return self
        def __exit__(self,*a):pass
    class Session:
        def get(self,url,**kw):
            calls.append(url);return Response(url==PUBLIC)
    monkeypatch.setattr(requests,'Session',Session)
    links=d.PublicLinks();paced=[];monkeypatch.setattr(links,'wait_for_request',lambda:paced.append(1))
    reader=d.Ranges([dict(id=7,size=3)],links)
    for _ in range(10):assert reader(0,0,3)==b'abc'
    assert calls.count(PUBLIC)==1 and calls.count(SIGNED)==9
    assert len(paced)==1
    stats=reader.stats()
    assert stats['dataverse_requests']==1 and stats['storage_requests']==10 and stats['link_reuses']==9
    assert 'SECRET' not in json.dumps(stats)

def test_access_denial_stops_other_cells_without_requests(monkeypatch):
    import requests
    calls=[]
    class Response:
        status_code=403;headers={}
        def __enter__(self):return self
        def __exit__(self,*a):pass
    class Session:
        def get(self,*a,**kw):calls.append(1);return Response()
    monkeypatch.setattr(requests,'Session',Session)
    links=d.PublicLinks()
    for file_id in [7,8]:
        reader=d.Ranges([dict(id=file_id,size=3)],links)
        with pytest.raises(d.RangeResponseError):reader(0,0,3)
    assert len(calls)==1
