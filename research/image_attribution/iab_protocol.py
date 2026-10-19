"""Prompt/duplicate isolation and semantic-balanced OFA reference selection."""
from collections import defaultdict
import random


def group_and_split(input_rows,seed=42):
    rows=sorted((dict(r) for r in input_rows),key=lambda r:r['sample_id'])
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
    groups=defaultdict(list)
    for i,row in enumerate(rows):
        row['group_id']=rows[find(i)]['sample_id'];groups[row['group_id']].append(row)
    strata=defaultdict(list)
    for group,members in sorted(groups.items()):
        key=(tuple(sorted({r['semantic_class'] for r in members})),tuple(sorted({str(r['source_label']) for r in members})))
        strata[key].append(group)
    rng=random.Random(seed)
    for _,ids in sorted(strata.items()):
        rng.shuffle(ids);a=int(.7*len(ids));b=a+int(.15*len(ids))
        for n,group in enumerate(ids):
            split='train' if n<a else 'dev' if n<b else 'holdout'
            for row in groups[group]:row['split']=split
    return rows,dict(group_count=len(groups),largest_group=max(map(len,groups.values())),
        near_duplicate_links=links,strata_count=len(strata),grouping='semantic+publisher prompt index; exact bytes/pixels; pHash Hamming<=4')


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
