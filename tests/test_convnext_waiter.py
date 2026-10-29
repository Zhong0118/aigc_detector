import json
import pytest
from scripts import wait_convnext_q179 as waiter


def test_binding_changes_fail_closed(tmp_path,monkeypatch):
    monkeypatch.setattr(waiter,'ROOT',tmp_path)
    p=tmp_path/'source.py';p.write_text('original')
    binding={'source.py':waiter.sha(p)};waiter.check_registration(binding)
    p.write_text('changed')
    with pytest.raises(ValueError,match='changed'):waiter.check_registration(binding)


def test_dead_identity_is_not_pid_only():
    import os
    record=waiter.identity(os.getpid())
    assert waiter.alive(record)
    assert not waiter.alive(dict(record,process_start_ticks='0'))
    assert not waiter.alive(dict(record,boot_id='wrong'))


def test_atomic_status_can_be_read(tmp_path):
    p=tmp_path/'status.json';waiter.write(p,dict(status='waiting_gpu0',used_mib=1287))
    assert json.loads(p.read_text())['status']=='waiting_gpu0'
    assert not p.with_suffix('.tmp').exists()
