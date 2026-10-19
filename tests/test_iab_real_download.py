import hashlib
import io
import json
from pathlib import Path
import zipfile


def test_real_archive_resumes_and_checks_publisher_md5(tmp_path,monkeypatch):
    import requests
    from scripts.download_iab_subset import real_archive
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w') as z:z.writestr('x.jpg',b'image-fixture')
    raw=b.getvalue();folder=tmp_path/'full_real_archives';folder.mkdir()
    (folder/'7.zip.partial').write_bytes(raw[:10])
    class Response:
        status_code=206
        headers={'Content-Range':f'bytes 10-{len(raw)-1}/{len(raw)}'}
        def __init__(self):self.raw=io.BytesIO(raw[10:])
        def __enter__(self):return self
        def __exit__(self,*args):pass
    def get(url,**kwargs):
        assert kwargs['stream'] and kwargs['headers']['Range']==f'bytes=10-{len(raw)-1}'
        return Response()
    monkeypatch.setattr(requests,'get',get)
    record={'parts':[{'id':7,'size':len(raw),'checksum':{'type':'MD5','value':hashlib.md5(raw).hexdigest()}}]}
    with real_archive(record,{7:'https://example.invalid'},tmp_path) as z:assert z.read('x.jpg')==b'image-fixture'
    assert (folder/'7.zip').read_bytes()==raw
    assert not (folder/'7.zip.partial').exists()
    monkeypatch.setattr(requests,'get',lambda *a,**k:(_ for _ in ()).throw(AssertionError('must reuse verified archive')))
    with real_archive(record,{},tmp_path) as z:assert z.read('x.jpg')==b'image-fixture'


def test_refreshed_public_links_are_reloaded_without_restart(tmp_path):
    from scripts.download_iab_subset import PublicLinks,write_json
    path=tmp_path/'links.json'
    write_json(path,{'results':[{'id':7,'url':'https://example.invalid/old','status':200}]})
    links=PublicLinks(path);assert links[7].endswith('/old')
    write_json(path,{'results':[{'id':7,'url':'https://example.invalid/new','status':200}]})
    assert links[7].endswith('/new')
