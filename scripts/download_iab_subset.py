"""Acquire an auditable source/semantic-balanced public IAB V3.2 subset.

Temporary signed download URLs are supplied separately and never written to
receipts. Only selected ZIP members are downloaded; whole-volume MD5 is NOT
verified. Member CRC32, SHA256, image decoding and pixel hashes are recorded.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import random
import re
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research.image_attribution.iab_archives import archive_catalog, read_member
from research.image_attribution.protocol import sha256, write_json as write_plain_json


def write_json(path, value):
    temporary=path.with_name(path.name+".partial-json")
    write_plain_json(temporary,value)
    temporary.replace(path)


def prompt_id(name):
    match=re.search(r'(?:^|_)p(\d+)(?:_|\.)',Path(name).name)
    return match.group(1) if match else None


class Ranges:
    def __init__(self, parts, links):
        import requests
        self.parts,self.links,self.session=parts,links,requests.Session()
        self.bytes=0

    def __call__(self,disk,start,length):
        part=self.parts[disk]
        if length<=0 or start<0 or start+length>part['size']: raise ValueError('invalid range')
        for attempt in range(4):
            try:
                with self.session.get(self.links[part['id']],headers={'Range':f'bytes={start}-{start+length-1}','Accept-Encoding':'identity'},timeout=(20,120),stream=True) as r:
                    expected=f"bytes {start}-{start+length-1}/{part['size']}"
                    if r.status_code != 206 or r.headers.get('Content-Range') != expected:
                        raise ValueError(f'file {part["id"]}: range status {r.status_code}')
                    if r.headers.get('Content-Encoding','identity')!='identity':raise ValueError('encoded range response')
                    data=r.raw.read(length+1)
                if len(data)!=length: raise ValueError('range length mismatch')
                self.bytes+=length
                return data
            except Exception as exc:
                # requests exceptions can contain signed URLs; do not log them.
                kind=type(exc).__name__
                if attempt==3: raise RuntimeError(f'file {part["id"]}: range failed ({kind})') from None
                time.sleep(2**attempt)


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


def catalog_cell(key, parts, links, output):
    source,semantic=key
    path=output/'catalogs'/f'{source}__{semantic}.json'
    if path.exists():
        cached=json.loads(path.read_text())
        if cached['source']!=source or cached['semantic']!=semantic or cached['parts']!=parts:
            raise ValueError('catalog identity changed')
        return cached
    reader=Ranges(parts,links)
    entries=archive_catalog([p['size'] for p in parts],reader)
    missing_count=sum(e.get('physical_volume',e['volume'])<0 for e in entries)
    entries=[e for e in entries if e.get('physical_volume',e['volume'])>=0 and Path(e['filename']).suffix.lower() in {'.png','.jpg','.jpeg','.webp'} and '__MACOSX' not in e['filename']]
    record=dict(source=source,semantic=semantic,parts=parts,entries=entries,unavailable_member_count=missing_count,range_bytes=reader.bytes)
    write_json(path,record)
    return record


def acquire_cell(record, chosen, links, output):
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
    receipt=dict(source=source,semantic=semantic,selected=len(rows),range_bytes=reader.bytes,rows=rows)
    write_json(output/'receipts'/f'{source}__{semantic}.json',receipt)
    return dict(source=source,semantic=semantic,selected=len(rows),range_bytes=reader.bytes)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata',type=Path,required=True);p.add_argument('--links',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--per-cell',type=int,default=50)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--catalog-only',action='store_true')
    args=p.parse_args()
    metadata=json.loads(args.metadata.read_text())
    if metadata['version']!=[3,2] or metadata['id']!=11127188: raise ValueError('unverified dataset revision')
    links={x['id']:x['url'] for x in json.loads(args.links.read_text())['results'] if x['status']==200}
    groups=cells(metadata)
    for name in ('catalogs','receipts','images'): (args.output/name).mkdir(parents=True,exist_ok=True)
    records={}
    with ThreadPoolExecutor(args.workers) as pool:
        jobs={pool.submit(catalog_cell,k,v,links,args.output):k for k,v in groups.items()}
        for f in as_completed(jobs):
            key=jobs[f];records[key]=f.result()
            print(json.dumps(dict(stage='catalog',source=key[0],semantic=key[1],available=len(records[key]['entries']))),flush=True)
    if args.catalog_only: return
    # Select the same prompt IDs across every source in each semantic category.
    # One deterministic image per prompt avoids sibling-image leakage and keeps
    # all 22 source x 10 semantic cells exactly balanced before deduplication.
    selections={};audit=[]
    rng=random.Random(args.seed)
    semantics=sorted({c for s,c in groups});sources=sorted({s for s,c in groups if s!='real'})
    for semantic in semantics:
        by_source={}
        for source in sources:
            buckets=defaultdict(list)
            for entry in records[(source,semantic)]['entries']:
                prompt=prompt_id(entry['filename'])
                if prompt is not None: buckets[prompt].append(entry)
            by_source[source]=buckets
        common=set.intersection(*(set(b) for b in by_source.values()))
        if len(common)<args.per_cell: raise ValueError(f'{semantic}: only {len(common)} shared verified prompt IDs')
        selected=sorted(rng.sample(sorted(common),args.per_cell),key=int)
        for source in sources:
            selections[(source,semantic)]=[sorted(by_source[source][v],key=lambda x:x['filename'])[0] for v in selected]
        real=records[('real',semantic)]['entries'];n_real=args.per_cell*len(sources)
        if len(real)<n_real: raise ValueError(f'{semantic}: insufficient real images')
        selections[('real',semantic)]=rng.sample(sorted(real,key=lambda x:x['filename']),n_real)
        audit.append(dict(semantic=semantic,common_prompt_count=len(common),selected_prompt_ids=selected,
            ai_per_source=args.per_cell,real=n_real))
    selection=dict(dataset='ImageAttributionBench',version='3.2',seed=args.seed,per_cell=args.per_cell,
        metadata_sha256=sha256(args.metadata),semantics=audit,
        entries={f'{s}/{c}':v for (s,c),v in selections.items()})
    dest=args.output/'selection.json'
    if dest.exists() and json.loads(dest.read_text())!=selection: raise ValueError('selection changed; use a new output directory')
    write_json(dest,selection)
    with ThreadPoolExecutor(args.workers) as pool:
        jobs={pool.submit(acquire_cell,records[k],v,links,args.output):k for k,v in selections.items()}
        failures=[]
        for f in as_completed(jobs):
            try:print(json.dumps(dict(stage='download',**f.result())),flush=True)
            except Exception as exc:
                failure=dict(source=jobs[f][0],semantic=jobs[f][1],error=str(exc))
                failures.append(failure)
                print(json.dumps(dict(stage='cell_failed',**failure)),flush=True)
        if failures:
            write_json(args.output/'download_failures.json',failures)
            raise RuntimeError(f'{len(failures)} cells failed; resume the unchanged selection')
    write_json(args.output/'download_status.json',dict(status='completed',cells=len(selections),
        selected=sum(map(len,selections.values())),seed=args.seed,per_cell=args.per_cell))


if __name__=='__main__':main()
