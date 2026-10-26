"""Prompt/duplicate isolation and semantic-balanced OFA reference selection."""
from collections import defaultdict
import random


def group_and_split(input_rows,seed=42,frozen_rows=None):
    rows=sorted((dict(r) for r in input_rows),key=lambda r:r['sample_id'])
    frozen={} if frozen_rows is None else {r['sample_id']:r for r in frozen_rows}
    if frozen_rows is not None:
        if not frozen or len(frozen)!=len(frozen_rows):
            raise ValueError('empty or duplicate frozen samples')
        byid={r['sample_id']:r for r in rows}
        if not set(frozen)<=set(byid):
            raise ValueError('frozen samples missing from expansion')
        fields=('sha256','pixel_sha256','phash','is_ai','source_label','semantic_class','prompt_id')
        for sid,old in frozen.items():
            if any(byid[sid].get(key)!=old.get(key) for key in fields):
                raise ValueError(f'frozen sample changed: {sid}')
            if old.get('split') not in {'train','dev','holdout'} or not old.get('group_id'):
                raise ValueError('invalid frozen split/group')
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):
        a,b=find(i),find(j)
        if a!=b:parent[max(a,b)]=min(a,b)
    identities={};bands=defaultdict(list);links=[]
    hashes=[int(r['phash'],16) for r in rows]
    # Five disjoint bands: distance <=4 implies at least one unchanged band.
    # Candidate filtering is exact for this threshold, unlike approximate ANN.
    widths=[13,13,13,13,12]
    for i,row in enumerate(rows):
        keys=[('bytes',row['sha256']),('pixels',row['pixel_sha256'])]
        if row.get('prompt_id') is not None and row['is_ai']:
            keys.append(('prompt',row['semantic_class'],row['prompt_id']))
        for key in keys:
            if key in identities:union(i,identities[key])
            else:identities[key]=i
        candidate=set();offset=0;band_keys=[]
        for band,width in enumerate(widths):
            key=(band,(hashes[i]>>offset)&((1<<width)-1));offset+=width
            candidate.update(bands[key]);band_keys.append(key)
        for j in sorted(candidate):
            distance=(hashes[i]^hashes[j]).bit_count()
            if distance<=4:
                union(i,j)
                links.append(dict(first=rows[j]['sample_id'],second=row['sample_id'],phash_distance=distance))
        for key in band_keys:bands[key].append(i)
    raw_groups=defaultdict(list)
    for i,row in enumerate(rows):raw_groups[find(i)].append(row)
    groups={};anchors={}
    for members in raw_groups.values():
        old=[frozen[r['sample_id']] for r in members if r['sample_id'] in frozen]
        splits={r['split'] for r in old};old_groups={r['group_id'] for r in old}
        if len(splits)>1:raise ValueError('frozen split conflict: new duplicates bridge old partitions')
        if len(old_groups)>1:raise ValueError('frozen group conflict: preserve development roles; review new duplicates')
        group=next(iter(old_groups)) if old else members[0]['sample_id']
        groups[group]=members
        if old:anchors[group]=next(iter(splits))
        for row in members:
            row['group_id']=group
            if old:row['split']=anchors[group]
    strata=defaultdict(list)
    for group,members in sorted(groups.items()):
        key=(tuple(sorted({r['semantic_class'] for r in members})),tuple(sorted({str(r['source_label']) for r in members})))
        strata[key].append(group)
    rng=random.Random(seed)
    for _,ids in sorted(strata.items()):
        ids=[group for group in ids if group not in anchors]
        rng.shuffle(ids);a=int(.7*len(ids));b=a+int(.15*len(ids))
        for n,group in enumerate(ids):
            split='train' if n<a else 'dev' if n<b else 'holdout'
            for row in groups[group]:row['split']=split
    return rows,dict(group_count=len(groups),largest_group=max(map(len,groups.values())),
        near_duplicate_links=links,strata_count=len(strata),frozen_samples=len(frozen),
        split_policy='preserve frozen partitions/groups; split new unanchored groups 70/15/15' if frozen else 'grouped 70/15/15',grouping='semantic+publisher prompt index; exact bytes/pixels; pHash Hamming<=4')


def reference_indices(rows,labels,class_count,limit=100,seed=42):
    rng=random.Random(seed);result={}
    for label in range(class_count):
        buckets=defaultdict(list)
        for i,(row,y) in enumerate(zip(rows,labels)):
            if y==label and row['split']=='train':buckets[str(row.get('semantic_class'))].append(i)
        for ids in buckets.values():
            ids.sort(key=lambda i:rows[i]['sample_id']);rng.shuffle(ids)
        selected=[]
        while len(selected)<limit and any(buckets.values()):
            for key in sorted(buckets):
                if buckets[key] and len(selected)<limit:selected.append(buckets[key].pop())
        result[label]=selected
    return result
