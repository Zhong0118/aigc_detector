"""Acquire an auditable source/semantic-balanced public IAB V3.2 subset.

Temporary signed download URLs are supplied separately and never written to
receipts. Only selected ZIP members are downloaded; whole-volume MD5 is NOT
verified. Member CRC32, SHA256, image decoding and pixel hashes are recorded.
"""
import argparse
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import time
import threading
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlsplit

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research.image_attribution.iab_archives import archive_catalog, read_member
from research.image_attribution.iab_selection import select_available, select_prefix
from research.image_attribution.protocol import sha256, write_json as write_plain_json


def write_json(path, value):
    temporary=path.with_name(path.name+".partial-json")
    write_plain_json(temporary,value)
    temporary.replace(path)


class PublicLinks:
    """Resolve stable Dataverse endpoints, with optional public URL overrides."""
    def __init__(self, path=None, request_interval=0.0):
        self.path = Path(path) if path is not None else None
        self.mtime = None
        self.links = {}
        self.request_interval = request_interval
        self._request_lock = threading.Lock()
        self._next_request = 0.0
        self._redirects = {}
        self._cache_lock = threading.Lock()
        self.blocked_status = None

    def remember(self, key, response):
        # Reuse only a successful official public redirect; never persist its URL.
        url = getattr(response, 'url', '')
        parsed = urlsplit(url)
        history = getattr(response, 'history', [])
        if (response.status_code != 206 or parsed.scheme != 'https' or
                parsed.hostname != 'dvn-cloud-iqss.s3.amazonaws.com' or
                parsed.username or parsed.password or
                not any(urlsplit(getattr(r, 'url', '')).hostname == 'dataverse.harvard.edu'
                        and r.status_code in (301, 302, 303, 307, 308) for r in history)):
            return
        try:
            query = parse_qs(parsed.query)
            issued = datetime.strptime(query['X-Amz-Date'][0], '%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc).timestamp()
            expires = issued + int(query['X-Amz-Expires'][0]) - 60
        except (KeyError, ValueError, IndexError):
            return
        if expires > time.time():
            with self._cache_lock:
                self._redirects[key] = (url, expires)

    def wait_for_request(self):
        # All archive workers share this object: pace total public API load.
        with self._request_lock:
            delay = self._next_request - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._next_request = time.monotonic() + self.request_interval

    def __getitem__(self, key):
        if self.blocked_status is not None:
            raise RangeResponseError(self.blocked_status)
        with self._cache_lock:
            cached = self._redirects.get(key)
            if cached and cached[1] > time.time():
                return cached[0]
            self._redirects.pop(key, None)
        fallback = f"https://dataverse.harvard.edu/api/access/datafile/{key}"
        if self.path is None:
            return fallback
        try:
            stat = self.path.stat()
        except OSError:
            return fallback
        stamp = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
        if stamp != self.mtime:
            self.links = {item['id']: item['url'] for item in
                          json.loads(self.path.read_text()).get('results', [])
                          if item.get('status') == 200}
            self.mtime = stamp
        return self.links.get(key, fallback)


def prompt_id(name):
    match=re.search(r'(?:^|_)p(\d+)(?:_|\.)',Path(name).name)
    return match.group(1) if match else None



class RangeResponseError(RuntimeError):
    def __init__(self, status_code):
        self.status_code = status_code
        super().__init__(f"HTTP Range request returned status {status_code}")

class Ranges:
    def __init__(self, parts, links):
        import requests
        self.parts, self.links, self.session = parts, links, requests.Session()
        self.started = time.monotonic()
        self.bytes = 0
        self.requests_attempted = 0
        self.requests_succeeded = 0
        self.request_failures = 0
        self.timeouts = 0
        self.request_seconds = 0.0
        self.dataverse_requests = 0
        self.storage_requests = 0
        self.link_reuses = 0

    def stats(self):
        elapsed = time.monotonic() - self.started
        return dict(
            dataverse_requests=self.dataverse_requests,
            storage_requests=self.storage_requests,
            link_reuses=self.link_reuses,
            requests_attempted=self.requests_attempted,
            requests_succeeded=self.requests_succeeded,
            request_failures=self.request_failures,
            timeouts=self.timeouts,
            success_rate=self.requests_succeeded / self.requests_attempted if self.requests_attempted else None,
            bytes=self.bytes,
            request_seconds=self.request_seconds,
            mean_request_seconds=self.request_seconds / self.requests_attempted if self.requests_attempted else None,
            elapsed_seconds=elapsed,
            throughput_mib_s=self.bytes / (1024 * 1024 * elapsed) if elapsed else 0.0,
        )

    def __call__(self, disk, start, length):
        import requests
        part = self.parts[disk]
        if length <= 0 or start < 0 or start + length > part['size']:
            raise ValueError('invalid range')
        for attempt in range(4):
            url = self.links[part['id']]
            public = urlsplit(url).hostname == 'dataverse.harvard.edu'
            if public and hasattr(self.links, "wait_for_request"):
                self.links.wait_for_request()
                # Another worker may have received an access denial while waiting.
                if self.links.blocked_status is not None:
                    raise RangeResponseError(self.links.blocked_status)
            self.dataverse_requests += int(public)
            self.storage_requests += int(not public)
            self.link_reuses += int(not public)
            self.requests_attempted += 1
            request_started = time.monotonic()
            try:
                with self.session.get(url,
                        headers={'Range': f'bytes={start}-{start+length-1}', 'Accept-Encoding': 'identity'},
                        timeout=(20, 120), stream=True) as response:
                    if public and urlsplit(getattr(response, 'url', '')).hostname == 'dvn-cloud-iqss.s3.amazonaws.com':
                        self.storage_requests += 1
                    expected = f"bytes {start}-{start+length-1}/{part['size']}"
                    if response.status_code in (401, 403, 429):
                        if isinstance(self.links, PublicLinks):
                            self.links.blocked_status = response.status_code
                        raise RangeResponseError(response.status_code)
                    if response.status_code != 206 or response.headers.get('Content-Range') != expected:
                        raise ValueError(f"file {part['id']}: range status {response.status_code}")
                    if response.headers.get('Content-Encoding', 'identity') != 'identity':
                        raise ValueError('encoded range response')
                    data = response.raw.read(length + 1)
                    if len(data) != length:
                        raise ValueError('range length mismatch')
                    if isinstance(self.links, PublicLinks):
                        self.links.remember(part['id'], response)
                self.request_seconds += time.monotonic() - request_started
                self.requests_succeeded += 1
                self.bytes += length
                return data
            except Exception as exc:
                self.request_seconds += time.monotonic() - request_started
                self.request_failures += 1
                if isinstance(exc, requests.Timeout):
                    self.timeouts += 1
                if isinstance(exc, RangeResponseError):
                    exc.network_stats = self.stats()
                    raise exc from None
                if attempt == 3:
                    failure = RuntimeError(f"file {part['id']}: range failed ({type(exc).__name__})")
                    failure.network_stats = self.stats()
                    raise failure from None
                time.sleep(2 ** attempt)


def cells(metadata):
    groups=defaultdict(list)
    for f in metadata['files']:
        if f['directory']=='_test_imgs': continue
        source,semantic=f['directory'].split('/')
        if semantic=='ImageNet-1k': semantic='ImageNet'
        groups[(source,semantic)].append(f)
    result={}
    for key,parts in groups.items():
        parts.sort(key=lambda p:(p['label'].endswith('.zip'),p['label']))
        if not parts[-1]['label'].endswith('.zip'): raise ValueError('missing last ZIP volume')
        if any(p['restricted'] for p in parts): raise ValueError('restricted file')
        result[key]=parts
    if len(result)!=230 or len({s for s,c in result})!=23: raise ValueError('unexpected IAB coverage')
    return result


def catalog_cell(key, parts, links, output, reuse_from=None):
    source,semantic=key
    path=output/'catalogs'/f'{source}__{semantic}.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        cached=json.loads(path.read_text())
        if cached['source']!=source or cached['semantic']!=semantic or cached['parts']!=parts:
            raise ValueError('catalog identity changed')
        return cached
    if reuse_from:
        old_path=reuse_from/'catalogs'/path.name
        if old_path.is_file():
            cached=json.loads(old_path.read_text())
            if cached['source']!=source or cached['semantic']!=semantic or cached['parts']!=parts:
                raise ValueError('legacy catalog identity changed')
            write_json(path,cached)
            return cached
    reader=Ranges(parts,links)
    entries=archive_catalog([p['size'] for p in parts],reader)
    missing_count=sum(e.get('physical_volume',e['volume'])<0 for e in entries)
    entries=[e for e in entries if e.get('physical_volume',e['volume'])>=0 and Path(e['filename']).suffix.lower() in {'.png','.jpg','.jpeg','.webp'} and '__MACOSX' not in e['filename']]
    record=dict(source=source,semantic=semantic,parts=parts,entries=entries,unavailable_member_count=missing_count,range_bytes=reader.bytes)
    write_json(path,record)
    return record



def reuse_verified_image(reuse_from, output, source, semantic, filename, identifier):
    if reuse_from is None:
        return None
    suffix=Path(filename).suffix.lower()
    old_target=reuse_from/'images'/source/semantic/(identifier+suffix)
    old_sidecar=old_target.with_suffix('.json')
    target=output/'images'/source/semantic/old_target.name
    if target.exists() or not old_target.is_file() or not old_sidecar.is_file():
        return None
    row=json.loads(old_sidecar.read_text())
    if (row.get('source_evidence',{}).get('zip_member')!=filename or
        row.get('semantic_class')!=semantic or
        row.get('source_label')!=(source if source!='real' else None) or
        sha256(old_target)!=row.get('sha256')):
        return None
    target.parent.mkdir(parents=True,exist_ok=True)
    try:
        os.link(old_target,target)
    except OSError:
        import shutil
        shutil.copyfile(old_target,target)
    new_row=dict(row,relative_path=str(target.relative_to(output)))
    write_json(target.with_suffix('.json'),new_row)
    return new_row


def acquire_cell(record, chosen, links, output, reuse_from=None):
    from PIL import Image
    import imagehash
    source,semantic=record['source'],record['semantic']
    reader=Ranges(record['parts'],links)
    sizes=[p['size'] for p in record['parts']]
    rows=[]
    folder=output/'images'/source/semantic
    folder.mkdir(parents=True,exist_ok=True)
    for entry in chosen:
        identifier=hashlib.sha256(f'{source}/{semantic}/{entry["filename"]}'.encode()).hexdigest()[:24]
        target=folder/(identifier+Path(entry['filename']).suffix.lower())
        sidecar=folder/(identifier+'.json')
        if target.exists() and sidecar.exists():
            row=json.loads(sidecar.read_text())
            if sha256(target)!=row['sha256']: raise ValueError('resume hash mismatch')
        else:
            row=reuse_verified_image(reuse_from,output,source,semantic,entry['filename'],identifier)
            if row is None:
                data=read_member(entry,sizes,reader)
                with Image.open(io.BytesIO(data)) as image:
                    image.load();rgb=image.convert('RGB');w,h=rgb.size
                    phash=str(imagehash.phash(rgb))
                    pixel=hashlib.sha256(str((w,h)).encode()+rgb.tobytes()).hexdigest()
                    fmt=image.format
                target.write_bytes(data)
                row=dict(sample_id='iab-'+identifier,relative_path=str(target.relative_to(output)),
                    sha256=hashlib.sha256(data).hexdigest(),pixel_sha256=pixel,phash=phash,
                    is_ai=int(source!='real'),source_label=source if source!='real' else None,
                    source_label_level='publisher_directory',source_evidence=dict(
                        doi='10.7910/DVN/O4S4IV',version='3.2',directory=f'{source}/{semantic}',
                        file_ids=[p['id'] for p in record['parts']],zip_member=entry['filename'],
                        zip_crc32=entry['crc32'],member_crc_verified=True,whole_volume_md5_verified=False),
                    dataset_id='ImageAttributionBench',dataset_revision='3.2',original_split='unsplit_public_release',
                    split=None,group_id=None,semantic_class=semantic,prompt_id=prompt_id(entry['filename']) if source!='real' else None,
                    task_type='dataset_declared_generation' if source!='real' else 'real',
                    parent_sample_id=None,transform_id='original_bytes',format=fmt,width=w,height=h,
                    watermark_status='not_verified')
                write_json(sidecar,row)
        rows.append(row)
        if reader.requests_attempted and len(rows) % 25 == 0:
            print(json.dumps(dict(stage='progress', source=source, semantic=semantic,
                                  verified=len(rows), target=len(chosen), network=reader.stats())), flush=True)
    receipt=dict(source=source,semantic=semantic,selected=len(rows),range_bytes=reader.bytes,
                 network=reader.stats(),rows=rows)
    write_json(output/'receipts'/f'{source}__{semantic}.json',receipt)
    return dict(source=source,semantic=semantic,selected=len(rows),range_bytes=reader.bytes,
                network=reader.stats())



def acquisition_jobs(pool, records, selections, links, output, reuse_from, workers):
    """Keep real and generated cells active; share freed slots when either ends."""
    real = deque(k for k in sorted(selections) if k[0] == 'real')
    ai = deque(k for k in sorted(selections) if k[0] != 'real')
    pending = {}
    while real or ai or pending:
        while len(pending) < workers and (real or ai):
            active_real = sum(key[0] == 'real' for key in pending.values())
            queue = real if real and (not ai or active_real < max(1, workers // 2)) else ai
            key = queue.popleft()
            future = pool.submit(acquire_cell, records[key], selections[key], links, output, reuse_from)
            pending[future] = key
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            yield pending.pop(future), future


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--per-cell',type=int,default=200)
    p.add_argument('--workers',type=int,default=16);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--request-interval',type=float,default=1.0,help='minimum seconds between Dataverse entry requests; cached S3 reads are not paced')
    p.add_argument('--reuse-from',type=Path,help='extend a frozen subset and reuse its verified image files')
    p.add_argument('--prefix-selection',action='store_true',help='take fixed prefixes of the reuse source selection, without resampling')
    p.add_argument('--catalog-only',action='store_true')
    p.add_argument('--plan-only',action='store_true',help='freeze the availability-capped selection without acquiring images')
    args=p.parse_args()
    if args.request_interval < 0:
        p.error('--request-interval must be nonnegative')
    if args.reuse_from and args.reuse_from.resolve()==args.output.resolve():
        raise ValueError('reuse source and output must differ')
    if args.prefix_selection and not args.reuse_from:
        p.error('--prefix-selection requires --reuse-from')
    metadata=json.loads(args.metadata.read_text())
    if metadata['version']!=[3,2] or metadata['id']!=11127188: raise ValueError('unverified dataset revision')
    links=PublicLinks(request_interval=args.request_interval)
    groups=cells(metadata)
    for name in ('catalogs','receipts','images'): (args.output/name).mkdir(parents=True,exist_ok=True)
    records={}
    with ThreadPoolExecutor(args.workers) as pool:
        jobs={pool.submit(catalog_cell,k,v,links,args.output,args.reuse_from):k for k,v in groups.items()}
        for f in as_completed(jobs):
            key=jobs[f];records[key]=f.result()
            print(json.dumps(dict(stage='catalog',source=key[0],semantic=key[1],available=len(records[key]['entries']))),flush=True)
    if args.catalog_only: return
    legacy={}
    if args.reuse_from:
        old_path=args.reuse_from/'selection.json'
        if not old_path.is_file():raise ValueError('reuse source has no frozen selection.json')
        legacy=json.loads(old_path.read_text()).get('entries',{})
    select = select_prefix if args.prefix_selection else select_available
    selections,audit=select({key:record['entries'] for key,record in records.items()},
                                       legacy,args.per_cell,args.seed)
    selection=dict(dataset='ImageAttributionBench',version='3.2',seed=args.seed,per_cell=args.per_cell,
        metadata_sha256=sha256(args.metadata),
        selection_policy=('fixed prefix of parent selection; no resampling' if args.prefix_selection else
                          'preserve legacy selections then deterministic availability-capped extension'),
        reused_from=str(args.reuse_from.resolve()) if args.reuse_from else None,
        audit=audit,entries={f'{s}/{c}':v for (s,c),v in selections.items()})
    dest=args.output/'selection.json'
    if dest.exists() and json.loads(dest.read_text())!=selection: raise ValueError('selection changed; use a new output directory')
    write_json(dest,selection)
    if args.plan_only:
        print(json.dumps(dict(stage='selection', audit=audit), separators=(',', ':')), flush=True)
        return
    download_started=time.monotonic()
    results=[]
    with ThreadPoolExecutor(args.workers) as pool:
        failures=[]
        for key, f in acquisition_jobs(pool, records, selections, links, args.output, args.reuse_from, args.workers):
            try:
                result=f.result()
                results.append(result)
                print(json.dumps(dict(stage='download',**result)),flush=True)
            except Exception as exc:
                failure=dict(source=key[0],semantic=key[1],error=str(exc),
                             network=getattr(exc,'network_stats',None))
                failures.append(failure)
                print(json.dumps(dict(stage='cell_failed',**failure)),flush=True)
        if failures:
            write_json(args.output/'download_failures.json',failures)
            raise RuntimeError(f'{len(failures)} cells failed; resume the unchanged selection')
    network_rows=[row['network'] for row in results]
    network_rows.extend(row['network'] for row in failures if row.get('network'))
    network=dict(
        requests_attempted=sum(row.get('requests_attempted',0) for row in network_rows),
        requests_succeeded=sum(row.get('requests_succeeded',0) for row in network_rows),
        request_failures=sum(row.get('request_failures',0) for row in network_rows),
        timeouts=sum(row.get('timeouts',0) for row in network_rows),
        bytes=sum(row.get('bytes',0) for row in network_rows),
        wall_seconds=time.monotonic()-download_started,
        cell_failures=len(failures),
    )
    network['success_rate']=(network['requests_succeeded']/network['requests_attempted']
                             if network['requests_attempted'] else None)
    network['throughput_mib_s']=(network['bytes']/(1024*1024*network['wall_seconds'])
                                 if network['wall_seconds'] else 0.0)
    write_json(args.output/'download_status.json',dict(status='completed',cells=len(selections),
        selected=sum(map(len,selections.values())),ai_selected=audit['ai_selected'],
        real_selected=audit['real_selected'],seed=args.seed,per_cell=args.per_cell,
        selected_by_cell=audit['selected_by_cell'],selected_uncompressed_bytes=audit['selected_uncompressed_bytes'],
        selected_compressed_bytes=audit['selected_compressed_bytes'],network=network))


if __name__=='__main__':main()
