"""Build a research-only subset from explicitly downloaded Tiny-GenImage train shards."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.image_attribution.protocol import assign_splits, sha256, validate_manifest, write_json
from scripts.train_image_attribution import decode_dataset_label


def main():
    import imagehash
    import pyarrow.parquet as pq
    from PIL import Image

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parquet', type=Path, nargs='+', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-source', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'images').mkdir()
    rng = random.Random(args.seed)
    candidates, assets, features = [], [], None
    for path in sorted(args.parquet):
        if not path.name.startswith('train-'):
            raise ValueError('Only official train shards are allowed')
        pf = pq.ParquetFile(path)
        current = json.loads(pf.schema_arrow.metadata[b'huggingface'])['info']['features']
        if features is not None and current != features:
            raise ValueError('feature schema changed between shards')
        features = current
        assets.append({'name': path.name, 'sha256': sha256(path), 'bytes': path.stat().st_size,
                       'rows': pf.metadata.num_rows})
        labels = pf.read(columns=['label', 'generator']).to_pylist()
        for index, row in enumerate(labels):
            row.update(shard=path.name, index=index)
            candidates.append(row)
    if features['label']['names'] != ['real', 'fake']:
        raise ValueError('Unexpected truth label definition')
    names = features['generator']['names']
    buckets = defaultdict(list)
    for row in candidates:
        if (names[row['generator']] == 'Real') != (row['label'] == 0):
            raise ValueError('Conflicting truth/source labels')
        buckets[row['generator']].append(row)
    nonempty_ai = [k for k in buckets if names[k] != 'Real']
    selected = {}
    for k, bucket in sorted(buckets.items()):
        limit = args.per_source * len(nonempty_ai) if names[k] == 'Real' else args.per_source
        for row in rng.sample(bucket, min(len(bucket), limit)):
            selected[(row['shard'], row['index'])] = row
    rows, failures = [], []
    for path in sorted(args.parquet):
        offset = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
            for raw in batch.to_pylist():
                index = offset
                offset += 1
                if (path.name, index) not in selected:
                    continue
                sample_id = f'{path.stem}-{index:05d}'
                try:
                    content = raw['image']['bytes']
                    with Image.open(io.BytesIO(content)) as image:
                        image.load()
                        fmt, size = image.format, image.size
                        rgb = image.convert('RGB')
                        phash = str(imagehash.phash(rgb))
                        pixel_hash = hashlib.sha256(str(size).encode() + rgb.tobytes()).hexdigest()
                    suffix = {'JPEG': '.jpg', 'PNG': '.png', 'WEBP': '.webp'}.get(fmt)
                    if suffix is None:
                        raise ValueError(f'unsupported format {fmt}')
                    relative = f'images/{sample_id}{suffix}'
                    (args.output / relative).write_bytes(content)
                    label = decode_dataset_label(features, 'generator', raw['generator'])
                    rows.append(dict(sample_id=sample_id, relative_path=relative,
                        sha256=hashlib.sha256(content).hexdigest(), pixel_sha256=pixel_hash, phash=phash,
                        is_ai=raw['label'], source_label=label if raw['label'] else None,
                        source_label_level='dataset_declared_generator' if raw['label'] else 'dataset_declared_real',
                        source_evidence=f'TheKernel01/Tiny-GenImage@{args.revision}/{path.name}:row={index}',
                        dataset_id='TheKernel01/Tiny-GenImage', dataset_revision=args.revision,
                        original_split='train', split=None, group_id=None, semantic_class=None,
                        task_type='dataset_declared_generation' if raw['label'] else 'real',
                        parent_sample_id=None, transform_id='original_bytes', prompt_id=None,
                        original_label=raw['label'], original_generator=raw['generator'],
                        format=fmt, width=size[0], height=size[1], watermark_status='not_verified'))
                except Exception as exc:
                    failures.append({'sample_id': sample_id, 'error': str(exc)})
    # Conservative transitive near-duplicate grouping; semantic/prompt identity is unavailable.
    rows.sort(key=lambda r: r['sample_id'])
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    links = []
    for i, first in enumerate(rows):
        for j in range(i):
            second = rows[j]
            distance = (int(first['phash'], 16) ^ int(second['phash'], 16)).bit_count()
            if first['pixel_sha256'] == second['pixel_sha256'] or distance <= 4:
                parent[find(i)] = find(j)
                links.append({'first': first['sample_id'], 'second': second['sample_id'], 'phash_distance': distance})
    for i, row in enumerate(rows):
        row['group_id'] = rows[find(i)]['sample_id']
    rows = assign_splits(rows, args.seed)
    validate_manifest(rows, args.output)
    manifest = args.output / 'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    report = dict(protocol='tiny-train-shard-v1', seed=args.seed, manifest_sha256=sha256(manifest),
        dataset_revision=args.revision, upstream_assets=assets, features=features,
        shard_counts={names[k]: len(v) for k, v in buckets.items()},
        declared_but_absent=[n for i, n in enumerate(names) if i not in buckets],
        selected_counts=dict(Counter(str(r['source_label']) for r in rows)),
        split_counts=dict(Counter(r['split'] for r in rows)),
        group_count=len({r['group_id'] for r in rows}), near_duplicate_links=links, failures=failures,
        license='CC-BY-NC-SA-4.0; upstream terms not fully audited; research only',
        limitations=['Random single/multiple shard subset, not full dataset coverage',
                     'No prompt/semantic metadata; no claim of prompt isolation',
                     'pHash distance <=4 and exact decoded pixels; not exhaustive semantic deduplication',
                     'Official validation never downloaded or used',
                     'Source versions and real-image origin not independently verified'])
    write_json(args.output / 'audit.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
