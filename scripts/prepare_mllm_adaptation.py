#!/usr/bin/env python3
"""Pinned public MLLM acquisition only: metadata, original bytes, size and SHA256.

Use a fresh output for every attempt. Preflight and execute each obtain metadata;
check-existing is offline/read-only. No retry, resume, image decoding or training.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import ssl
import sys
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, unquote, urlsplit
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener

DATASET = 'zr-zhang/MLLM-Generated-Image-Detection-Dataset'
REVISION = '651a325f0ece2744c21d6c68280d2aaac227177b'
SOURCES = ('Real', 'GPT Image 2', 'Nano Banana 2')
DEFAULT_OUTPUT = 'outputs/image-new-adaptation-20261109/mllm-audit-sample-v1'
RESERVE_BYTES = 20 * 1024**3
MAX_METADATA_BYTES = 16 * 1024**2
PROTOCOL = 'mllm-original-bytes-v1'


class AcquisitionError(RuntimeError):
    """Contains only a safe error category, never a response URL or body."""


class OfficialRedirect(HTTPRedirectHandler):
    @staticmethod
    def validate(url):
        parsed = urlsplit(url)
        host = parsed.hostname or ''
        if (parsed.scheme != 'https' or parsed.username or parsed.password or
                parsed.port not in (None, 443) or parsed.fragment or
                not (host == 'huggingface.co' or host.endswith('.huggingface.co') or host.endswith('.hf.co'))):
            raise ValueError('untrusted_https_endpoint')

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self.validate(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class PublicHTTP:
    def __init__(self):
        # Standard certificate verification, no tokens, cookies or custom proxies.
        self.opener = build_opener(OfficialRedirect(), HTTPSHandler(context=ssl.create_default_context()))
        self.lock = threading.Lock()
        self.next_request = 0.0

    def open(self, url):
        OfficialRedirect.validate(url)
        # Pace aggregate entry requests across workers; redirect URLs stay in memory.
        with self.lock:
            delay = self.next_request - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self.next_request = time.monotonic() + 0.25
        return self.opener.open(Request(url, headers={
            'User-Agent': 'aigc-detector-mllm-audit/1', 'Accept-Encoding': 'identity',
        }), timeout=60)


def category(error):
    if isinstance(error, HTTPError):
        return f'http_{error.code}'
    if isinstance(error, AcquisitionError):
        return str(error)
    if isinstance(error, (URLError, TimeoutError, ConnectionError)):
        return 'network_error'
    if isinstance(error, ValueError):
        return 'validation_error'
    if isinstance(error, OSError):
        return 'filesystem_error'
    return 'acquisition_error'


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + '\n', encoding='utf-8')


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')


def hash_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def relative_parts(value):
    if not isinstance(value, str) or not value or '\\' in value:
        raise ValueError('unsafe_path')
    path = PurePosixPath(value)
    if path.is_absolute() or any(p in ('', '.', '..') for p in value.split('/')):
        raise ValueError('unsafe_path')
    return path.parts


def safe_child(root, relative):
    parts = relative_parts(relative)
    child = root
    for part in parts:
        child = child / part
        if child.is_symlink():
            raise ValueError('symlink_path')
    return child


def output_path(root, relative):
    root = Path(root)
    if not root.is_absolute() or not root.is_dir() or root.resolve() != root:
        raise ValueError('root_must_be_existing_absolute_non_symlink')
    return safe_child(root, relative)


def resolve_url(path):
    return f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{quote(path, safe="/")}'


def small_read(response):
    body = response.read(MAX_METADATA_BYTES + 1)
    if len(body) > MAX_METADATA_BYTES:
        raise ValueError('metadata_too_large')
    return body


def fetch_tree(http):
    entries, pages = [], []
    for source in SOURCES:
        scope = f'images/{source}/Texture Images'
        base = f'https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}/{quote(scope, safe="/")}'
        url, seen = base + '?recursive=true&limit=1000', set()
        while url:
            parsed, expected = urlsplit(url), urlsplit(base)
            query = parse_qs(parsed.query, keep_blank_values=True)
            decoded_path = unquote(parsed.path)
            relative_parts(decoded_path.lstrip('/'))
            if (parsed.scheme != 'https' or parsed.netloc != 'huggingface.co' or
                    decoded_path != unquote(expected.path) or parsed.fragment or
                    set(query) - {'recursive', 'limit', 'cursor', 'expand'} or
                    query.get('expand', ['false']) != ['false'] or
                    query.get('recursive') != ['true'] or query.get('limit') != ['1000'] or
                    url in seen):
                raise ValueError('invalid_pagination_endpoint')
            seen.add(url)
            with http.open(url) as response:
                body = small_read(response)
                rows = json.loads(body)
                link = response.headers.get('Link', '')
            if not isinstance(rows, list):
                raise ValueError('invalid_tree_response')
            for row in rows:
                parts = relative_parts(row.get('path'))
                if parts[:3] != ('images', source, 'Texture Images'):
                    raise ValueError('tree_scope_mismatch')
            entries.extend(rows)
            # Keep the exact small API body, but never any redirect or Link URLs.
            pages.append(dict(source_label=source, page=len(seen), entries=rows,
                              response_sha256=hashlib.sha256(body).hexdigest(), response_bytes=body.decode('utf-8')))
            matches = [re.match(r'\s*<([^>]+)>\s*;\s*rel="?next"?', item) for item in link.split(',')]
            next_links = [match.group(1) for match in matches if match]
            if len(next_links) > 1:
                raise ValueError('ambiguous_pagination')
            url = next_links[0] if next_links else None
    return entries, pages


def select_entries(entries, limit):
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError('limit_must_be_positive')
    groups, paths, counts = {}, set(), dict.fromkeys(SOURCES, 0)
    for item in entries:
        parts = relative_parts(item.get('path'))
        if item.get('type') == 'directory':
            continue
        if (len(parts) < 4 or parts[0] != 'images' or parts[1] not in SOURCES or
                parts[2] != 'Texture Images' or Path(parts[-1]).suffix.lower() not in ('.png', '.jpg', '.jpeg')):
            continue
        if item.get('type') != 'file':
            raise ValueError('invalid_file_type')
        path, source = item['path'], parts[1]
        if path in paths:
            raise ValueError('duplicate_metadata_path')
        paths.add(path)
        lfs = item.get('lfs')
        size = item.get('size')
        if (not isinstance(lfs, dict) or not re.fullmatch('[0-9a-f]{64}', str(lfs.get('oid', ''))) or
                not isinstance(size, int) or isinstance(size, bool) or size <= 0 or lfs.get('size') != size):
            raise ValueError('invalid_lfs_identity')
        counts[source] += 1
        groups.setdefault(lfs['oid'], []).append(dict(source_path=path, source_label=source, size=size, sha256=lfs['oid']))
    candidates = {source: [] for source in SOURCES}
    duplicate_files = conflicting = 0
    for sha, group in groups.items():
        if len({r['size'] for r in group}) != 1:
            raise ValueError('sha_size_conflict')
        if len({r['source_label'] for r in group}) > 1:
            conflicting += 1
            continue
        duplicate_files += len(group) - 1
        row = min(group, key=lambda r: r['source_path'])
        source = row['source_label']
        row['aliases'] = sorted(r['source_path'] for r in group)
        row['selection_rank'] = hashlib.sha256(b'mllm-sample-v1\0' + b'42\0' + source.encode() + b'\0' + sha.encode()).hexdigest()
        row['sample_id'] = 'mllm-' + sha[:24]
        row['is_ai'] = int(source != 'Real')
        row['source_url'] = resolve_url(row['source_path'])
        slug = {'Real': 'real', 'GPT Image 2': 'gpt-image-2', 'Nano Banana 2': 'nano-banana-2'}[source]
        row['relative_path'] = f'images/{slug}/{sha}{Path(row["source_path"]).suffix.lower()}'
        candidates[source].append(row)
    selected = []
    for source in SOURCES:
        selected.extend(sorted(candidates[source], key=lambda r: (r['selection_rank'], r['sha256']))[:limit])
    audit = dict(candidate_files_by_source=counts, unique_files_by_source={s: len(candidates[s]) for s in SOURCES},
                 selected_files_by_source={s: sum(r['source_label'] == s for r in selected) for s in SOURCES},
                 duplicate_files=duplicate_files, conflicting_sha256_groups=conflicting,
                 selected_bytes=sum(r['size'] for r in selected))
    return selected, audit


def download_one(http, output, row, transport):
    target = safe_child(output, row['relative_path'])
    part = safe_child(output, row['relative_path'] + '.part')
    if target.exists() or part.exists():
        raise FileExistsError('download_target_exists')
    target.parent.mkdir(parents=True, exist_ok=True)
    digest, size = hashlib.sha256(), 0
    with part.open('xb') as stream:
        with http.open(row['source_url']) as response:
            if response.headers.get('Content-Encoding', 'identity') != 'identity':
                raise ValueError('encoded_image_response')
            while True:
                chunk = response.read(1024**2)
                if not chunk:
                    break
                stream.write(chunk)
                digest.update(chunk)
                size += len(chunk)
                if size > row['size']:
                    raise ValueError('size_mismatch')
    if size != row['size'] or digest.hexdigest() != row['sha256']:
        raise ValueError('size_or_sha256_mismatch')
    # Exclusive link publishes verified bytes without replacing any existing path.
    os.link(part, target)
    part.unlink()
    return dict(row, status='verified', verified_sha256=digest.hexdigest(), verified_size=size,
                transport=transport, verified_at=timestamp())


def prepare(root, output, mode, limit=32, workers=4, transport='school-direct'):
    target = output_path(root, output)
    if target.exists():
        raise FileExistsError('output_already_exists')
    if mode not in ('preflight', 'execute') or not 1 <= workers <= 4 or limit <= 0:
        raise ValueError('invalid_acquisition_options')
    if transport not in ('school-direct', 'local-https-relay'):
        raise ValueError('invalid_transport')
    target.mkdir(parents=True, exist_ok=False)
    status = dict(protocol=PROTOCOL, dataset=DATASET, revision=REVISION, status='running',
                  mode=mode, transport=transport, started_at=timestamp(), exit_code=None)
    write_json(target / 'status.json', status)
    try:
        http = PublicHTTP()
        entries, pages = fetch_tree(http)
        rows, audit = select_entries(entries, limit)
        if any(audit['selected_files_by_source'][s] == 0 for s in SOURCES):
            raise ValueError('empty_source_after_dedup')
        write_json(target / 'metadata-pages.json', pages)
        with http.open(resolve_url('README.md')) as response:
            card = small_read(response)
        (target / 'source-data-card.md').write_bytes(card)
        write_jsonl(target / 'manifest.jsonl', rows)
        selection = dict(protocol=PROTOCOL, dataset=DATASET, revision=REVISION,
                         seed=42, limit_per_source=limit, workers=workers, transport=transport,
                         ranking='sha256(mllm-sample-v1 NUL 42 NUL publisher UTF8 NUL LFS-sha256 ASCII)',
                         deduplication='LFS SHA256; cross-publisher conflicts excluded',
                         labels='Publisher directory labels; card Hybrid/Structure/Texture labels unused',
                         reserve_bytes=RESERVE_BYTES, metadata_only=True,
                         source_card_sha256=hash_file(target / 'source-data-card.md'),
                         metadata_sha256=hash_file(target / 'metadata-pages.json'),
                         manifest_sha256=hash_file(target / 'manifest.jsonl'), **audit)
        write_json(target / 'selection.json', selection)
        if mode == 'execute':
            if shutil.disk_usage(target).free < RESERVE_BYTES + audit['selected_bytes']:
                raise AcquisitionError('insufficient_disk_reserve')
            stop = threading.Event()
            def acquire(row):
                if stop.is_set():
                    return dict(row, status='skipped', error_category='cancelled_after_failure', transport=transport)
                try:
                    return download_one(http, target, row, transport)
                except Exception as error:
                    stop.set()
                    return dict(row, status='failed', error_category=category(error), transport=transport)
            receipts = []
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for receipt in pool.map(acquire, rows):
                    receipts.append(receipt)
                    write_jsonl(target / 'receipts.jsonl', receipts)
            failures = [r for r in receipts if r['status'] == 'failed']
            if failures:
                raise AcquisitionError(failures[0]['error_category'])
            status.update(status='completed', verified_files=len(receipts),
                          receipts_sha256=hash_file(target / 'receipts.jsonl'))
        else:
            status.update(status='metadata_only', selected_files=len(rows))
        status.update(exit_code=0, finished_at=timestamp(), selection_sha256=hash_file(target / 'selection.json'))
        write_json(target / 'status.json', status)
        return status
    except Exception as error:
        code = category(error)
        status.update(status='failed', exit_code=1, error_category=code, finished_at=timestamp())
        write_json(target / 'status.json', status)
        raise AcquisitionError(code) from None


def check_existing(root, output):
    target = output_path(root, output)
    for child in target.rglob('*'):
        if child.is_symlink():
            raise ValueError('symlink_in_existing_output')
    status = json.loads((target / 'status.json').read_text())
    if (status.get('status') != 'completed' or status.get('exit_code') != 0 or
            status.get('dataset') != DATASET or status.get('revision') != REVISION or status.get('protocol') != PROTOCOL):
        raise ValueError('acquisition_not_completed')
    selection_path = target / 'selection.json'
    if hash_file(selection_path) != status['selection_sha256']:
        raise ValueError('selection_hash_mismatch')
    selection = json.loads(selection_path.read_text())
    for name, key in [('source-data-card.md', 'source_card_sha256'), ('metadata-pages.json', 'metadata_sha256'), ('manifest.jsonl', 'manifest_sha256')]:
        if hash_file(target / name) != selection[key]:
            raise ValueError('metadata_hash_mismatch')
    pages = json.loads((target / 'metadata-pages.json').read_text())
    entries = []
    for page in pages:
        body = page['response_bytes'].encode()
        if hashlib.sha256(body).hexdigest() != page['response_sha256'] or json.loads(body) != page['entries']:
            raise ValueError('api_page_identity_mismatch')
        entries.extend(page['entries'])
    expected, _ = select_entries(entries, selection['limit_per_source'])
    rows = [json.loads(line) for line in (target / 'manifest.jsonl').read_text().splitlines()]
    if rows != expected:
        raise ValueError('selection_reproduction_mismatch')
    if hash_file(target / 'receipts.jsonl') != status['receipts_sha256']:
        raise ValueError('receipts_hash_mismatch')
    receipts = [json.loads(line) for line in (target / 'receipts.jsonl').read_text().splitlines()]
    if len(rows) != len(receipts) or status['verified_files'] != len(rows):
        raise ValueError('receipt_count_mismatch')
    for row, receipt in zip(rows, receipts):
        if (any(receipt.get(k) != v for k, v in row.items()) or receipt.get('status') != 'verified' or
                receipt.get('verified_sha256') != row['sha256'] or receipt.get('verified_size') != row['size'] or
                receipt.get('transport') != status['transport']):
            raise ValueError('receipt_identity_mismatch')
        image = safe_child(target, row['relative_path'])
        if not image.is_file() or image.stat().st_size != row['size'] or hash_file(image) != row['sha256']:
            raise ValueError('image_size_or_sha256_mismatch')
    if any(target.rglob('*.part')):
        raise ValueError('partial_files_present')
    return dict(status='verified', protocol=PROTOCOL, revision=REVISION, verified_files=len(rows), transport=status['transport'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='Existing absolute root, without symlink components')
    parser.add_argument('--output', default=DEFAULT_OUTPUT, help='Fresh relative output directory under root')
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--preflight', action='store_true', help='Fetch public metadata and data card only')
    modes.add_argument('--execute', action='store_true', help='Acquire selected original image files once')
    modes.add_argument('--check-existing', action='store_true', help='Offline read-only size/SHA and receipt verification')
    parser.add_argument('--limit-per-source', type=int, default=32, help='After exact dedup; default 32, never all by default')
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--transport', choices=['school-direct', 'local-https-relay'], default='school-direct')
    args = parser.parse_args(argv)
    if args.limit_per_source <= 0:
        parser.error('--limit-per-source must be positive')
    try:
        if args.check_existing:
            result = check_existing(args.root, args.output)
        else:
            result = prepare(args.root, args.output, 'execute' if args.execute else 'preflight',
                             args.limit_per_source, args.workers, args.transport)
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as error:
        print(json.dumps(dict(status='failed', exit_code=1, error_category=category(error))), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
