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
