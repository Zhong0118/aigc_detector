"""Offline acquisition contract tests; no image decoding, network, GPU or training."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'prepare_mllm_adaptation.py'
REVISION = '651a325f0ece2744c21d6c68280d2aaac227177b'
SOURCES = ('Real', 'GPT Image 2', 'Nano Banana 2')


def entry(source='Real', name='a.png', payload=b'image fixture'):
    return dict(type='file', path=f'images/{source}/Texture Images/{name}',
                size=len(payload), lfs=dict(oid=hashlib.sha256(payload).hexdigest(), size=len(payload)))


class Response(io.BytesIO):
    def __init__(self, body, link=None):
        super().__init__(body)
        self.headers = {'Link': link} if link else {}


class MLLMAcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.is_file(), 'acquisition entry point is missing')
        spec = importlib.util.spec_from_file_location('mllm_preparation', SCRIPT)
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def fake_http(self, rows=None, broken_download=False):
        rows = rows if rows is not None else [entry(s, payload=s.encode()) for s in SOURCES]
        calls = []
        def open_url(url):
            calls.append(url)
            if '/tree/' in url:
                from urllib.parse import unquote
                source = next(s for s in SOURCES if f'images/{s}/Texture Images' in unquote(url))
                return Response(json.dumps([r for r in rows if f'images/{source}/' in r['path']]).encode())
            if url.endswith('/README.md'):
                return Response(b'---\nlicense: mit\n---\n# Public dataset card\n')
            if broken_download:
                raise HTTPError('https://cas-bridge.xethub.hf.co/file?secret=NEVER_PERSIST', 403,
                                'signed URL https://secret.invalid?token=LEAK', {}, None)
            from urllib.parse import unquote
            source = next(s for s in SOURCES if f'images/{s}/' in unquote(url))
            return Response(source.encode())
        class HTTP:
            open = staticmethod(open_url)
        return HTTP(), calls

    def test_hash_dedup_happens_before_balanced_selection(self):
        rows = [entry(s, payload=s.encode()) for s in SOURCES]
        rows += [entry('Real', 'alias.jpg', b'Real'), entry('Real', 'z.png', b'other real')]
        rows += [entry('Real', 'conflict.png', b'conflict'), entry('GPT Image 2', 'conflict.png', b'conflict')]
        selected, audit = self.m.select_entries(rows, 1)
        self.assertEqual([r['source_label'] for r in selected], list(SOURCES))
        self.assertEqual(len({r['sha256'] for r in selected}), 3)
        self.assertEqual(audit['duplicate_files'], 1)
        self.assertEqual(audit['conflicting_sha256_groups'], 1)
        self.assertNotIn(hashlib.sha256(b'conflict').hexdigest(), [r['sha256'] for r in selected])
        self.assertEqual(self.m.select_entries(list(reversed(rows)), 1), (selected, audit))
        self.assertTrue(all('/resolve/' in r['source_url'] and REVISION in r['source_url'] for r in selected))
        # Publisher labels, never the card's unrelated Hybrid/Structure/Texture labels.
        self.assertEqual([r['is_ai'] for r in selected], [0, 1, 1])

    def test_rejects_untrusted_paths_and_invalid_lfs_before_selection(self):
        for value in [entry(name='../x.png'), entry(name='/x.png'), entry(name='nested/../../x.png')]:
            with self.subTest(path=value['path']), self.assertRaises(ValueError):
                self.m.select_entries([value], 1)
        for change in [dict(lfs={'oid': '0'*40, 'size': 1}), dict(size=-1), dict(lfs=None)]:
            value = entry(); value.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.m.select_entries([value], 1)

    def test_tree_pagination_keeps_fixed_revision_and_scope(self):
        first = entry()
        base = f'https://huggingface.co/api/datasets/zr-zhang/MLLM-Generated-Image-Detection-Dataset/tree/{REVISION}/images/Real/Texture%20Images'
        next_url = base + '?recursive=true&limit=1000&cursor=two'
        class HTTP:
            def open(_, url):
                if url == next_url:
                    return Response(json.dumps([entry(name='b.png', payload=b'b')]).encode())
                if '/Real/' in url:
                    return Response(json.dumps([first]).encode(), f'<{next_url}>; rel="next"')
                return Response(b'[]')
        rows, pages = self.m.fetch_tree(HTTP())
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(pages), 4)
        for evil in ['http://huggingface.co/x', 'https://evil.invalid/x', base.replace(REVISION, 'main'), base.replace('/Real/', '/Other/')]:
            class Evil:
                def open(_, url):
                    return Response(b'[]', f'<{evil}>; rel="next"')
            with self.subTest(evil=evil), self.assertRaises(ValueError):
                self.m.fetch_tree(Evil())

    def test_hf_encoded_scope_pagination_with_expand_false(self):
        base = f'https://huggingface.co/api/datasets/zr-zhang/MLLM-Generated-Image-Detection-Dataset/tree/{REVISION}/'
        next_url = base + 'images%2FReal%2FTexture%20Images?expand=false&recursive=true&limit=1000&cursor=two'
        class HTTP:
            def open(_, url):
                if url == next_url:
                    return Response(json.dumps([entry(name='b.png', payload=b'b')]).encode())
                if '/Real/' in url:
                    return Response(json.dumps([entry()]).encode(), f'<{next_url}>; rel="next"')
                return Response(b'[]')
        rows, pages = self.m.fetch_tree(HTTP())
        self.assertEqual([r['path'] for r in rows], ['images/Real/Texture Images/a.png', 'images/Real/Texture Images/b.png'])
        self.assertEqual(len(pages), 4)
        for evil in [next_url.replace('expand=false', 'expand=true'),
                     next_url.replace('images%2FReal', 'images%2FOther'),
                     next_url.replace('images%2FReal', 'images%2F..%2FReal'),
                     next_url.replace(REVISION, 'main'),
                     next_url.replace('huggingface.co', 'evil.invalid')]:
            class Evil:
                def open(_, url):
                    return Response(b'[]', f'<{evil}>; rel="next"')
            with self.subTest(evil=evil), self.assertRaises(ValueError):
                self.m.fetch_tree(Evil())

    def test_preflight_never_requests_images_and_refuses_existing_output(self):
        http, calls = self.fake_http()
        with patch.object(self.m, 'PublicHTTP', return_value=http):
            result = self.m.prepare(self.root, 'outputs/preflight', 'preflight', limit=1)
        self.assertEqual(result['status'], 'metadata_only')
        self.assertFalse(any('/resolve/' in u and not u.endswith('/README.md') for u in calls))
        out = self.root / 'outputs/preflight'
        self.assertEqual((out / 'source-data-card.md').read_bytes(), b'---\nlicense: mit\n---\n# Public dataset card\n')
        self.assertEqual(len((out / 'manifest.jsonl').read_text().splitlines()), 3)
        before = (out / 'status.json').read_bytes()
        with self.assertRaises(FileExistsError):
            self.m.prepare(self.root, 'outputs/preflight', 'execute', limit=1)
        self.assertEqual((out / 'status.json').read_bytes(), before)

    def test_execute_checks_original_bytes_receipts_and_readonly_recheck(self):
        http, _ = self.fake_http()
        with patch.object(self.m, 'PublicHTTP', return_value=http), patch.object(self.m.shutil, 'disk_usage', return_value=type('Disk', (), {'free': 10**12})()):
            result = self.m.prepare(self.root, 'outputs/sample', 'execute', limit=1, workers=2, transport='local-https-relay')
        self.assertEqual(result['status'], 'completed')
        out = self.root / 'outputs/sample'
        receipts = [json.loads(line) for line in (out / 'receipts.jsonl').read_text().splitlines()]
        self.assertEqual(len(receipts), 3)
        self.assertTrue(all(r['verified_sha256'] == r['sha256'] and r['verified_size'] == r['size'] for r in receipts))
        before = {str(p): (p.stat().st_mtime_ns, p.read_bytes()) for p in out.rglob('*') if p.is_file()}
        self.assertEqual(self.m.check_existing(self.root, 'outputs/sample')['verified_files'], 3)
        after = {str(p): (p.stat().st_mtime_ns, p.read_bytes()) for p in out.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        (out / receipts[0]['relative_path']).write_bytes(b'tampered')
        with self.assertRaises(ValueError):
            self.m.check_existing(self.root, 'outputs/sample')

    def test_failure_keeps_partial_file_records_safe_error_and_never_retries(self):
        http, calls = self.fake_http(broken_download=True)
        with patch.object(self.m, 'PublicHTTP', return_value=http), patch.object(self.m.shutil, 'disk_usage', return_value=type('Disk', (), {'free': 10**12})()):
            with self.assertRaises(self.m.AcquisitionError):
                self.m.prepare(self.root, 'outputs/failure', 'execute', limit=1, workers=1)
        out = self.root / 'outputs/failure'
        status = json.loads((out / 'status.json').read_text())
        self.assertEqual(status['status'], 'failed')
        self.assertEqual(status['exit_code'], 1)
        self.assertEqual(status['error_category'], 'http_403')
        self.assertEqual(len(list(out.rglob('*.part'))), 1)
        self.assertEqual(len([u for u in calls if '/resolve/' in u and not u.endswith('/README.md')]), 1)
        saved = b''.join(p.read_bytes() for p in out.rglob('*') if p.is_file())
        for forbidden in (b'NEVER_PERSIST', b'LEAK', b'secret.invalid', b'cas-bridge'):
            self.assertNotIn(forbidden, saved)
        with self.assertRaises(ValueError):
            self.m.check_existing(self.root, 'outputs/failure')

    def test_size_sha_and_symlink_checks_fail_closed(self):
        selected, _ = self.m.select_entries([entry()], 1)
        out = self.root / 'case'; out.mkdir()
        class HTTP:
            def open(_, url):
                return Response(b'corruption')
        with self.assertRaises(ValueError):
            self.m.download_one(HTTP(), out, selected[0], 'school-direct')
        self.assertEqual(len(list(out.rglob('*.part'))), 1)
        self.assertFalse((out / selected[0]['relative_path']).exists())
        for relative in ('../escape', '/absolute', 'outputs/../../escape'):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                self.m.prepare(self.root, relative, 'preflight')
        (self.root / 'link').symlink_to(out, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.m.prepare(self.root, 'link/sample', 'preflight')

    def test_reserve_and_cli_invalid_arguments_fail_before_download(self):
        http, calls = self.fake_http()
        with patch.object(self.m, 'PublicHTTP', return_value=http), patch.object(self.m.shutil, 'disk_usage', return_value=type('Disk', (), {'free': 20 * 1024**3})()):
            with self.assertRaises(self.m.AcquisitionError):
                self.m.prepare(self.root, 'outputs/no-space', 'execute', limit=1)
        self.assertFalse(any('/resolve/' in u and not u.endswith('/README.md') for u in calls))
        for args in [[], ['--preflight', '--limit-per-source', '0'], ['--execute', '--workers', '5']]:
            result = subprocess.run([sys.executable, str(SCRIPT), '--root', str(self.root), *args], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
        help_result = subprocess.run([sys.executable, str(SCRIPT), '--help'], capture_output=True, text=True)
        self.assertEqual(help_result.returncode, 0)
        self.assertIn('--check-existing', help_result.stdout)

    def test_redirect_policy_rejects_nonofficial_and_http_urls(self):
        handler = self.m.OfficialRedirect()
        for url in ['http://huggingface.co/file', 'https://evil.invalid/file', 'https://huggingface.co.evil.invalid/file', 'https://user@huggingface.co/file']:
            with self.subTest(url=url), self.assertRaises(ValueError):
                handler.validate(url)
        handler.validate('https://cas-bridge.xethub.hf.co/file?signature=memory-only')


if __name__ == '__main__':
    unittest.main()
