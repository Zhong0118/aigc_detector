from research.image_attribution.iab_protocol import group_and_split, reference_indices


def test_prompt_pixel_and_near_hash_groups_never_cross():
    rows=[]
    for i in range(100):
        for source in ['a','b']:
            rows.append(dict(sample_id=f'{source}-{i}',source_label=source,is_ai=1,
                semantic_class='cat',prompt_id=str(i),pixel_sha256=f'{source}-{i}',
                sha256=f'{source}-{i}',phash=f'{(i*12345678901234567)%(2**64):016x}'))
    rows[4]['pixel_sha256']=rows[0]['pixel_sha256']
    rows[8]['phash']=f'{int(rows[12]["phash"],16)^7:016x}'
    a,audit=group_and_split(rows,42)
    b,_=group_and_split(list(reversed(rows)),42)
    assert {r['sample_id']:r['split'] for r in a}=={r['sample_id']:r['split'] for r in b}
    byid={r['sample_id']:r for r in a}
    assert byid['a-0']['group_id']==byid['a-2']['group_id']
    assert byid['a-4']['group_id']==byid['a-6']['group_id']
    for row in a: assert row['split']==byid['a-'+row['prompt_id']]['split']
    assert {r['split'] for r in a}=={'train','dev','holdout'}


def test_ofa_references_are_balanced_semantically_and_train_only():
    rows=[dict(sample_id=f'{s}-{i}',semantic_class=s,split='train') for s in ['cat','dog'] for i in range(40)]
    rows.append(dict(sample_id='dev',semantic_class='cat',split='dev'))
    labels=[0]*len(rows)
    selected=reference_indices(rows,labels,1,limit=20,seed=42)[0]
    assert len(selected)==20
    assert sum(rows[i]['semantic_class']=='cat' for i in selected)==10
    assert all(rows[i]['split']=='train' for i in selected)
    assert selected==reference_indices(rows,labels,1,limit=20,seed=42)[0]


def expansion_rows(n):
    import random
    rng=random.Random(716)
    return [dict(sample_id=f'image-{i:03d}',source_label='model',is_ai=1,
                 semantic_class='cat',prompt_id=str(i),pixel_sha256=f'pixels-{i}',
                 sha256=f'bytes-{i}',phash=f'{rng.getrandbits(64):016x}') for i in range(n)]


def test_expansion_preserves_frozen_splits_and_group_ids():
    old,_=group_and_split(expansion_rows(20),42)
    expanded=expansion_rows(50)
    expanded.append(dict(expanded[0],sample_id='earlier-name',sha256='new-bytes'))
    actual,audit=group_and_split(expanded,91,frozen_rows=old)
    byid={r['sample_id']:r for r in actual}
    for row in old:
        assert (byid[row['sample_id']]['split'],byid[row['sample_id']]['group_id']) == (row['split'],row['group_id'])
    assert byid['earlier-name']['group_id']==old[0]['group_id']
    assert audit['frozen_samples']==20
    reverse,_=group_and_split(list(reversed(expanded)),91,frozen_rows=old)
    assert actual==reverse


def test_expansion_rejects_changed_or_missing_parent_samples():
    import pytest
    old,_=group_and_split(expansion_rows(20))
    changed=expansion_rows(50);changed[0]['sha256']='changed'
    with pytest.raises(ValueError,match='frozen sample changed'):
        group_and_split(changed,frozen_rows=old)
    with pytest.raises(ValueError,match='frozen samples missing'):
        group_and_split(expansion_rows(19),frozen_rows=old)


def test_expansion_rejects_new_duplicate_bridging_old_splits_or_groups():
    import pytest
    old,_=group_and_split(expansion_rows(20))
    first=old[0];second=next(r for r in old if r['split']!=first['split'])
    bridge=dict(expansion_rows(21)[-1],sha256=first['sha256'],pixel_sha256=second['pixel_sha256'])
    with pytest.raises(ValueError,match='frozen split conflict'):
        group_and_split(old+[bridge],frozen_rows=old)
    second=next(r for r in old if r['split']==first['split'] and r['group_id']!=first['group_id'])
    bridge['pixel_sha256']=second['pixel_sha256']
    with pytest.raises(ValueError,match='frozen group conflict'):
        group_and_split(old+[bridge],frozen_rows=old)


def test_existing_expansion_cannot_bypass_parent_holdout_constraint():
    import pytest
    from scripts.prepare_iab_manifest import validate_existing_split
    parent,_=group_and_split(expansion_rows(20),42)
    expanded,_=group_and_split(expansion_rows(50),42,frozen_rows=parent)
    validate_existing_split(expanded,parent,42)
    held=next(r['sample_id'] for r in parent if r['split']=='holdout')
    next(r for r in expanded if r['sample_id']==held)['split']='train'
    with pytest.raises(ValueError,match='existing manifest split/group mismatch'):
        validate_existing_split(expanded,parent,42)


def test_pipeline_checks_existing_manifest_before_training(tmp_path):
    from pathlib import Path
    import json,os,shutil,subprocess,sys
    repo=Path(__file__).resolve().parents[1]
    (tmp_path/'scripts').mkdir();(tmp_path/'.venv-image/bin').mkdir(parents=True)
    shutil.copyfile(repo/'scripts/run_iab_m2.sh',tmp_path/'scripts/run_iab_m2.sh')
    log=tmp_path/'calls.jsonl';py=tmp_path/'.venv-image/bin/python'
    py.write_text('#!'+sys.executable+'\nimport json,sys\nfrom pathlib import Path\n'
        +f'with Path({str(log)!r}).open("a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\n'
        +'if "--check-existing" in sys.argv:raise SystemExit(7)\n')
    py.chmod(0o755)
    gpu=tmp_path/'.venv-image/bin/nvidia-smi';gpu.write_text('#!/bin/sh\necho 0\n');gpu.chmod(0o755)
    data=tmp_path/'data';data.mkdir();(data/'manifest.jsonl').write_text('already frozen')
    env=dict(os.environ,PATH=str(py.parent)+':'+os.environ['PATH'])
    result=subprocess.run(['bash','scripts/run_iab_m2.sh',str(data),'test50','summary.json','parent/manifest.jsonl'],cwd=tmp_path,env=env,capture_output=True,text=True)
    assert result.returncode==7,result.stderr
    calls=[json.loads(l) for l in log.read_text().splitlines()]
    assert any('--check-existing' in call and '--parent-manifest' in call for call in calls)
    assert not any('scripts/run_image_experiment.py' in call for call in calls)
