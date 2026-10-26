import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

spec = importlib.util.spec_from_file_location('monitor', Path(__file__).resolve().parents[1]/'scripts/monitor_iab_v2.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_lock_excludes_second_holder(tmp_path):
    with m.lock(tmp_path/'lock') as first:
        assert first is not None
        with m.lock(tmp_path/'lock') as second:
            assert second is None


def test_pid_reuse_is_not_alive(monkeypatch):
    monkeypatch.setattr(m, 'identity', lambda pid: ['boot', 'new'])
    assert not m.alive(dict(pid=12, identity=['boot', 'old']))


def test_failed_pipeline_never_restarts(tmp_path, monkeypatch):
    folder = tmp_path/'outputs/iab-v2-monitor'
    m.write(folder/'pipeline.json', dict(status='failed', exit_code=1))
    monkeypatch.setattr(m, 'launch', lambda *a: (_ for _ in ()).throw(AssertionError('duplicate launch')))
    report = dict(completion=None, download=dict(status='completed'))
    m.advance(tmp_path, report)
    assert 'needs_attention' in report


def test_worker_records_real_subprocess_failure(tmp_path, monkeypatch):
    folder = tmp_path/'outputs/iab-v2-monitor'
    m.write(folder/'pipeline.json', dict(status='running'))
    monkeypatch.setattr(m, 'command', lambda root, kind: [sys.executable, '-c', 'raise SystemExit(7)'])
    assert m.worker(tmp_path, 'pipeline') == 7
    assert m.read(folder/'pipeline.json')['status'] == 'failed'
    assert not (folder/'completion.json').exists()


def test_expiry_never_returns_url(tmp_path):
    path=tmp_path/'links.json'
    m.write(path, dict(results=[dict(url='https://private.invalid/secret?Expires=1')]))
    result=m.expiry(path)
    assert result['earliest_unix'] == 1
    assert 'private' not in json.dumps(result)


def test_read_only_once_creates_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'gpu', lambda index=0: dict(index=index, unavailable=True))
    m.snapshot(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_completed_dummy_pipeline_is_validated_before_completion(tmp_path, monkeypatch):
    folder = tmp_path/'outputs/iab-v2-monitor'
    m.write(folder/'pipeline.json', dict(status='running'))
    scripts=tmp_path/'scripts'; scripts.mkdir()
    runtime=tmp_path/'.venv-image/bin'; runtime.mkdir(parents=True)
    (runtime/'python').symlink_to(sys.executable)
    (scripts/'summarize_iab_m2.py').write_text(
        'import json,sys\nfrom pathlib import Path\n'
        'Path(sys.argv[sys.argv.index("--output")+1]).write_text(json.dumps({"manifest_sha256":"validated"}))\n')
    monkeypatch.setattr(m, 'command', lambda root, kind: [sys.executable, '-c', 'print("dummy train finished")'])
    assert m.worker(tmp_path, 'pipeline') == 0
    assert m.read(folder/'completion.json')['manifest_sha256'] == 'validated'


def test_busy_gpu_blocks_pipeline_without_intent(tmp_path):
    report = dict(completion=None, download=dict(status='completed'), gpu0=dict(used_mib=2048))
    m.advance(tmp_path, report)
    assert report['waiting']
    assert not (tmp_path/'outputs/iab-v2-monitor/pipeline.json').exists()


def test_worker_environment_only_passes_a_verified_proxy(monkeypatch):
    from scripts.monitor_iab_v2 import worker_environment

    monkeypatch.setenv("HTTP_PROXY", "http://unverified.invalid:8080")
    monkeypatch.setenv("https_proxy", "http://unverified.invalid:8080")

    direct = worker_environment(None)
    assert not any(key.lower() in {"http_proxy", "https_proxy", "all_proxy"} for key in direct)
    proxied = worker_environment("http://127.0.0.1:17898")
    assert proxied["HTTP_PROXY"] == "http://127.0.0.1:17898"
    assert proxied["HTTPS_PROXY"] == "http://127.0.0.1:17898"


def test_download_waits_for_a_successful_public_transport_probe(tmp_path, monkeypatch):
    folder = tmp_path/'outputs/iab-v2-monitor'
    folder.mkdir(parents=True)
    (folder/'iab-public-v3.2-metadata.json').write_text("{}")
    monkeypatch.setattr(m, 'launch', lambda *a, **k: (_ for _ in ()).throw(AssertionError('network not verified')))
    report = dict(completion=None, download=None, gpu0={"used_mib": 0},
                  transport={"direct": {"success": False}, "proxy": {"success": False}})
    m.advance(tmp_path, report)
    assert "waiting" in report or "needs_attention" in report


def test_pipeline_waits_until_model_assets_exist(tmp_path):
    folder = tmp_path/'outputs/iab-v2-monitor'
    m.write(tmp_path/'outputs/image_experiments/iab-balanced-v2-200/download_status.json',
            dict(status="completed"))
    report = dict(completion=None, download=dict(status="completed"),
                  gpu0={"used_mib": 0}, transport={"direct": {"success": True}})
    m.advance(tmp_path, report)
    assert report["needs_attention"] == "required image model assets missing"
    assert not (folder/'pipeline.json').exists()


def test_probe_transports_reports_both_public_paths(tmp_path, monkeypatch):
    metadata = tmp_path/"metadata.json"
    metadata.write_text(json.dumps({"files": [{"id": 9, "size": 4096, "restricted": False}]}))
    calls = []
    def probe(file_id, total_size, proxy_url=None):
        calls.append((file_id, total_size, proxy_url))
        return {"success": proxy_url is None, "bytes": total_size, "elapsed_seconds": 1,
                "throughput_mib_s": 0.0039, "timeouts": 0, "request_failures": int(bool(proxy_url))}
    monkeypatch.setattr(m, "request_probe", probe)

    result = m.probe_transports(metadata)

    assert result["direct"]["success"] is True
    assert result["proxy"]["success"] is False
    assert calls == [(9, 4096, None), (9, 4096, m.PROXY_URL)]


def test_download_launches_only_with_a_successfully_probed_route(tmp_path, monkeypatch):
    folder = tmp_path/"outputs/iab-v2-monitor"
    folder.mkdir(parents=True)
    (folder/"iab-public-v3.2-metadata.json").write_text("{}")
    launched = []
    monkeypatch.setattr(m, "launch", lambda *args, **kwargs: launched.append((args, kwargs)))
    report = dict(completion=None, download=None, gpu0={"used_mib": 0},
                  transport={"direct": {"success": False}, "proxy": {"success": True}})
    m.advance(tmp_path, report)

    assert report["launched"] == "download"
    assert launched[0][1]["proxy_url"] == m.PROXY_URL


def test_remaining_selected_bytes_subtracts_completed_receipt_samples(tmp_path):
    import hashlib

    data = tmp_path/"data"
    selection = {
        "entries": {
            "model/cat": [
                {"filename": "a.png", "size": 10},
                {"filename": "b.png", "size": 20},
            ],
        },
    }
    (data/"receipts").mkdir(parents=True)
    (data/"selection.json").write_text(json.dumps(selection))
    sample_id = "iab-" + hashlib.sha256(b"model/cat/a.png").hexdigest()[:24]
    (data/"receipts/model__cat.json").write_text(json.dumps(
        {"rows": [{"sample_id": sample_id}]}))

    assert m.remaining_selected_bytes(data) == 20


def test_download_waits_when_remaining_data_plus_reserve_exceeds_free_disk(tmp_path, monkeypatch):
    import hashlib

    folder = tmp_path/"outputs/iab-v2-monitor"
    folder.mkdir(parents=True)
    (folder/"iab-public-v3.2-metadata.json").write_text("{}")
    data = tmp_path/"outputs/image_experiments/iab-balanced-v2-200"
    data.mkdir(parents=True)
    (data/"selection.json").write_text(json.dumps({"entries": {
        "model/cat": [{"filename": "a.png", "size": 9 * 1024}],
    }}))
    launched = []
    monkeypatch.setattr(m, "launch", lambda *args, **kwargs: launched.append(args))
    report = dict(completion=None, download=None, gpu0={"used_mib": 0},
                  disk_free_bytes=10 * 1024,
                  transport={"direct": {"success": True}, "proxy": {"success": False}})
    m.advance(tmp_path, report)

    assert report["waiting"] == "insufficient free disk for remaining selected data and reserve"
    assert not launched


def test_failed_download_worker_is_never_restarted_automatically(tmp_path, monkeypatch):
    folder = tmp_path/"outputs/iab-v2-monitor"
    folder.mkdir(parents=True)
    (folder/"iab-public-v3.2-metadata.json").write_text("{}")
    m.write(folder/"download.json", {"status": "failed", "exit_code": 1})
    monkeypatch.setattr(m, "launch",
                        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("duplicate download")))
    report = dict(completion=None, download=None, gpu0={"used_mib": 0},
                  disk_free_bytes=100 * 1024**3, remaining_selected_bytes=0,
                  transport={"direct": {"success": True}, "proxy": {"success": False}})
    m.advance(tmp_path, report)

    assert "previous download worker failed" in report["needs_attention"]


def test_independent_monitor_namespace_preserves_completed_v4(tmp_path,monkeypatch):
    old=tmp_path/'outputs/iab-v2-monitor'
    m.write(old/'completion.json',dict(status='completed'))
    folder=tmp_path/'outputs/iab-v5-50-monitor'
    m.write(folder/'config.json',dict(dataset='outputs/data50',suffix='v5-50',per_cell=50,
        route='direct',parent_manifest='outputs/data20/manifest.jsonl'))
    monkeypatch.setenv('IAB_MONITOR_STATE_DIR','outputs/iab-v5-50-monitor')
    monkeypatch.setattr(m,'gpu',lambda index=0:dict(index=index,used_mib=0))
    report=m.snapshot(tmp_path)
    assert report['completion'] is None
    assert report['dataset']=='outputs/data50'
    assert str(folder/'iab-public-v3.2-metadata.json') in m.command(tmp_path,'download')
    assert m.command(tmp_path,'pipeline')[5]==str(tmp_path/'outputs/data20/manifest.jsonl')
    assert m.read(old/'completion.json')==dict(status='completed')
