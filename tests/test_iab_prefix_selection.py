from pathlib import Path
import pytest
from research.image_attribution import iab_selection as selection
from scripts import monitor_iab_v2 as monitor


def test_prefix_selection_keeps_order_caps_availability_and_does_not_fill():
    records = {(s, 'cat'): [{'filename': f'{s}{i}', 'size': i+1} for i in range(n)]
               for s,n in [('a', 6), ('b', 1), ('real', 3)]}
    parent = {'a/cat': [records[('a','cat')][i] for i in (4,1,3,0)],
              'b/cat': records[('b','cat')], 'real/cat': records[('real','cat')]}
    chosen,audit = selection.select_prefix(records, parent, per_cell=2)
    assert chosen[('a','cat')] == parent['a/cat'][:2]
    assert chosen[('b','cat')] == parent['b/cat']
    assert chosen[('real','cat')] == parent['real/cat']
    assert audit['ai_selected'] == 3 and audit['real_selected'] == 3
    assert len(parent['a/cat']) == 4


def test_prefix_rejects_an_incomplete_parent_instead_of_resampling():
    records = {('a','cat'): [{'filename': str(i)} for i in range(3)],
               ('real','cat'): [{'filename': 'r'}]}
    with pytest.raises(ValueError, match='parent selection'):
        selection.select_prefix(records, {'a/cat': [], 'real/cat': records[('real','cat')]}, per_cell=2)


def test_new_target_drives_download_pipeline_and_summary(tmp_path):
    monitor.write(tmp_path/'outputs/iab-v2-monitor/config.json', dict(
        dataset='outputs/image_experiments/iab-balanced-v3-100', suffix='v3-100',
        per_cell=100, workers=4, reuse_from='outputs/image_experiments/iab-balanced-v2-200',
        prefix_selection=True, route='direct'))
    download = monitor.command(tmp_path,'download')
    assert download[download.index('--per-cell')+1] == '100'
    assert '--prefix-selection' in download
    pipeline = monitor.command(tmp_path,'pipeline')
    assert pipeline[3:5] == ['v3-100','outputs/image_experiments/iab-m2-v3-100-summary.json']
    assert pipeline[5:] == ['', '20', '0']
    assert str(tmp_path/'outputs/image_experiments/iab-balanced-v3-100') in pipeline


def test_direct_policy_does_not_select_proxy_when_both_probes_succeed(tmp_path, monkeypatch):
    folder=tmp_path/'outputs/iab-v2-monitor'
    monitor.write(folder/'config.json', {'route':'direct'})
    monitor.write(folder/'iab-public-v3.2-metadata.json', {})
    calls=[]
    monkeypatch.setattr(monitor,'launch',lambda *a,**kw:calls.append(kw))
    report=dict(completion=None,download=None,gpu0={},disk_free_bytes=100000,
                remaining_selected_bytes=0,transport={'direct':{'success':True},'proxy':{'success':True}})
    monitor.advance(tmp_path,report)
    assert calls == [{'proxy_url':None}]


def test_worker_guard_matches_output_not_reuse_source_or_another_root(tmp_path):
    target=tmp_path/'outputs/image_experiments/iab-balanced-v2-200'
    other=tmp_path/'outputs/image_experiments/iab-balanced-v3-100'
    args=[b'python', b'scripts/download_iab_subset.py', b'--output', str(other).encode(),
          b'--reuse-from',str(target).encode()]
    assert not monitor.matches_worker(args, target, 'download')
    assert monitor.matches_worker(args, other, 'download')
    assert not monitor.matches_worker(args, Path('/other-root')/other.name, 'download')


def test_explicit_proxy_policy_does_not_silently_fall_back_to_direct(tmp_path,monkeypatch):
    folder=tmp_path/'outputs/iab-v2-monitor'
    monitor.write(folder/'config.json',{'route':'proxy'})
    monitor.write(folder/'iab-public-v3.2-metadata.json',{})
    calls=[]
    monkeypatch.setattr(monitor,'launch',lambda *a,**kw:calls.append(kw))
    report=dict(completion=None,download=None,gpu0={},disk_free_bytes=100000,
                remaining_selected_bytes=0,transport={'direct':{'success':True},'proxy':{'success':False}})
    monitor.advance(tmp_path,report)
    assert not calls and report['waiting']=='configured proxy route unavailable'
    report['transport']['proxy']['success']=True
    monitor.advance(tmp_path,report)
    assert calls==[{'proxy_url':monitor.PROXY_URL}]
