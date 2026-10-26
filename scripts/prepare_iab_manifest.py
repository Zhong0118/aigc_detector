"""Freeze the acquired IAB subset with prompt/duplicate-isolated development splits."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research.image_attribution.iab_protocol import group_and_split
from research.image_attribution.protocol import sha256,validate_manifest,write_json,load_manifest



def validate_receipts(selection, status, receipts):
    expected={key:len(entries) for key,entries in selection.get('entries',{}).items()}
    if not expected or status.get('selected')!=sum(expected.values()):
        raise ValueError('selection/status count mismatch')
    if status.get('selected_by_cell')!=selection.get('audit',{}).get('selected_by_cell'):
        raise ValueError('selection/status cell count mismatch')
    indexed={}
    for receipt in receipts:
        key=f"{receipt['source']}/{receipt['semantic']}"
        if key in indexed:
            raise ValueError('duplicate receipt cell')
        indexed[key]=receipt
    if set(indexed)!=set(expected):
        raise ValueError('missing or unexpected source/semantic receipt cell')
    rows=[]
    sample_ids=set()
    for key,count in expected.items():
        source,semantic=key.split('/',1)
        receipt=indexed[key]
        cell_rows=receipt.get('rows',[])
        if receipt.get('selected')!=count or len(cell_rows)!=count:
            raise ValueError(f'receipt count mismatch for {key}')
        for row in cell_rows:
            row_source=row.get('source_label') or 'real'
            if row_source!=source or row.get('semantic_class')!=semantic:
                raise ValueError(f'receipt label mismatch for {key}')
            if row.get('sample_id') in sample_ids:
                raise ValueError('duplicate sample_id in receipts')
            sample_ids.add(row['sample_id'])
        rows.extend(cell_rows)
    if len(rows)!=status['selected']:
        raise ValueError('acquisition count mismatch')
    return rows

def validate_existing_split(rows,frozen,seed=42):
    """An existing manifest must satisfy the same anchored protocol as a new one."""
    expected,_=group_and_split(rows,seed,frozen_rows=frozen)
    actual={r['sample_id']:r for r in rows}
    for row in expected:
        before=actual[row['sample_id']]
        if (before['split'],before['group_id'])!=(row['split'],row['group_id']):
            raise ValueError(f"existing manifest split/group mismatch: {row['sample_id']}")
    return dict(status='verified',samples=len(rows),frozen_samples=len(frozen))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--check-existing',action='store_true',help='verify existing manifest against its frozen parent; never rewrite')
    p.add_argument('--parent-manifest',type=Path,help='preserve every parent sample split and group; fail on conflicts')
    p.add_argument('--seed',type=int,default=42);args=p.parse_args();root=args.dataset
    if args.check_existing:
        if not args.parent_manifest:p.error('--check-existing requires --parent-manifest')
        rows=[json.loads(line) for line in (root/'manifest.jsonl').read_text().splitlines() if line.strip()]
        frozen=[json.loads(line) for line in args.parent_manifest.read_text().splitlines() if line.strip()]
        # Training already rehashes the image files; this gate checks metadata/partition identity.
        validate_manifest(rows,root,verify_files=False)
        validate_manifest(frozen,args.parent_manifest.parent,verify_files=False)
        print(json.dumps(validate_existing_split(rows,frozen,args.seed)),flush=True)
        return
    if (root/'manifest.jsonl').exists():raise ValueError('manifest already frozen')
    status=json.loads((root/'download_status.json').read_text())
    if status['status']!='completed':raise ValueError('acquisition incomplete')
    selection=json.loads((root/'selection.json').read_text())
    receipts=[json.loads(p.read_text()) for p in sorted((root/'receipts').glob('*.json'))]
    if len(receipts)!=230:raise ValueError('missing source/semantic cells')
    rows=validate_receipts(selection,status,receipts)
    frozen=load_manifest(args.parent_manifest) if args.parent_manifest else None
    rows,audit=group_and_split(rows,args.seed,frozen_rows=frozen)
    if args.parent_manifest:
        audit.update(parent_manifest=str(args.parent_manifest.resolve()),parent_manifest_sha256=sha256(args.parent_manifest))
    validate_manifest(rows,root)
    manifest=root/'manifest.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    audit.update(dataset='ImageAttributionBench',version='3.2',seed=args.seed,manifest_sha256=sha256(manifest),
        selected=len(rows),requested_per_cell=selection['audit']['requested_ai_per_source_semantic'],
        selected_by_cell=selection['audit']['selected_by_cell'],
        real_selected=selection['audit']['real_selected'],
        sources=sorted({r['source_label'] for r in rows if r['is_ai']}),
        split_counts=dict(Counter(r['split'] for r in rows)),
        source_semantic_split_counts={f'{s}/{c}/{split}':n for (s,c,split),n in sorted(Counter((r['source_label'] or 'real',r['semantic_class'],r['split']) for r in rows).items())},
        license='CC BY 4.0 (Dataverse V3.2 metadata); underlying dataset/model conditions retained',
        official_test_used=False,whole_volume_md5_verified_for_all_inputs=False,
        rows_with_whole_archive_md5_verification=sum(bool(r['source_evidence']['whole_volume_md5_verified']) for r in rows),
        unique_byte_hashes=len({r['sha256'] for r in rows}), unique_pixel_hashes=len({r['pixel_sha256'] for r in rows}),
        limitations=['Development subset, not full paper reproduction','Publisher prompt indices assumed aligned across generator folders; prompt text not in ZIP archives',
        'Real filenames do not establish correspondence to generator prompts','Member CRC32 plus downloaded SHA256, not full archive MD5 verification',
        'Grouped 70/15/15 custom split; holdout unopened by training/evaluation','No semantic-disjoint or unseen-generator generalization claim'])
    write_json(root/'audit.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in {'near_duplicate_links','source_semantic_split_counts'}}),flush=True)


if __name__=='__main__':main()
