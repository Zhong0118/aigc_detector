import json
from pathlib import Path


def test_refreshed_public_links_are_reloaded_without_restart(tmp_path):
    from scripts.download_iab_subset import PublicLinks,write_json
    path=tmp_path/'links.json'
    write_json(path,{'results':[{'id':7,'url':'https://example.invalid/old','status':200}]})
    links=PublicLinks(path);assert links[7].endswith('/old')
    write_json(path,{'results':[{'id':7,'url':'https://example.invalid/new','status':200}]})
    assert links[7].endswith('/new')


def test_public_request_pacing_is_shared_and_waits_before_next_start(monkeypatch):
    import scripts.download_iab_subset as download
    clock=[10.0];sleeps=[]
    monkeypatch.setattr(download.time,'monotonic',lambda:clock[0])
    def sleep(seconds):
        sleeps.append(seconds);clock[0]+=seconds
    monkeypatch.setattr(download.time,'sleep',sleep)
    links=download.PublicLinks(request_interval=1.0)
    links.wait_for_request()
    links.wait_for_request()
    clock[0]+=2.0
    links.wait_for_request()
    assert sleeps == [1.0]
