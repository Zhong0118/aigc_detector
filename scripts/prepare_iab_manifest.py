"""Freeze the acquired IAB subset with prompt/duplicate-isolated development splits."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research.image_attribution.iab_protocol import group_and_split
from research.image_attribution.protocol import sha256,validate_manifest,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42);args=p.parse_args();root=args.dataset
    if (root/'manifest.jsonl').exists():raise ValueError('manifest already frozen')
    status=json.loads((root/'download_status.json').read_text())
    if status['status']!='completed':raise ValueError('acquisition incomplete')
    receipts=[json.loads(p.read_text()) for p in sorted((root/'receipts').glob('*.json'))]
    if len(receipts)!=230:raise ValueError('missing source/semantic cells')
    rows=[r for receipt in receipts for r in receipt['rows']]
    if len(rows)!=status['selected']:raise ValueError('acquisition count mismatch')
    counts=Counter((r['source_label'] or 'real',r['semantic_class']) for r in rows)
    if any(n!=(status['per_cell']*22 if source=='real' else status['per_cell']) for (source,semantic),n in counts.items()):
        raise ValueError('unbalanced acquisition')
    rows,audit=group_and_split(rows,args.seed)
    validate_manifest(rows,root)
    manifest=root/'manifest.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    audit.update(dataset='ImageAttributionBench',version='3.2',seed=args.seed,manifest_sha256=sha256(manifest),
        selected=len(rows),sources=sorted({r['source_label'] for r in rows if r['is_ai']}),
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
