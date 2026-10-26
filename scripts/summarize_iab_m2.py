"""Join completed independent tasks; report the fixed-threshold cascade separately."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research.image_attribution.protocol import sha256


def joined_cascade(rows,detection,attribution,ofa):
    dev={r['sample_id']:r for r in rows if r['split']=='dev'}
    ai={k for k,r in dev.items() if r['is_ai']}
    def index(records,expected):
        indexed={r['sample_id']:r for r in records}
        if len(indexed)!=len(records) or set(indexed)!=expected:raise ValueError('prediction/query ID mismatch')
        return indexed
    d=index(detection,set(dev));a=index(attribution,ai);o=index(ofa,ai)
    if any(r['classes']!=['real','ai'] for r in d.values()):raise ValueError('binary class order mismatch')
    flags={k for k,r in d.items() if r['scores'][1]>=.5}
    result=dict(threshold=.5,ai_queries=len(ai),real_queries=len(dev)-len(ai),
        flagged_ai=len(flags & ai),flagged_real=len(flags-ai),
        ai_detection_recall=len(flags & ai)/len(ai),models={})
    classes=sorted({dev[k]['source_label'] for k in ai})
    for name,records,key in [('resnet50',a,'scores'),('ofa',o,'match_scores')]:
        if any(r['classes']!=classes for r in records.values()):raise ValueError('attribution class order mismatch')
        correct={k for k,r in records.items() if classes[max(range(len(classes)),key=lambda i:r[key][i])]==dev[k]['source_label']}
        result['models'][name]=dict(correctly_detected_and_attributed=len(correct & flags),
            correctly_detected_and_attributed_over_all_ai=len(correct & flags)/len(ai),
            attribution_top1_among_flagged_ai=len(correct & flags)/len(flags & ai) if flags & ai else None,
            note='Closed-set known-generator development cascade; real false alarms counted separately, no unknown-source guarantee')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('outputs/image_experiments'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dataset',type=Path)
    p.add_argument('--run-suffix',default='v1')
    args=p.parse_args();root=args.root
    manifest=(args.dataset or root/'iab-balanced-v1')/'manifest.jsonl';digest=sha256(manifest)
    rows=[json.loads(line) for line in manifest.read_text().splitlines()]
    runs={};predictions={}
    for key,name in [('detection',f'd0-resnet50-iab-{args.run_suffix}'),('attribution',f'a0-resnet50-iab-{args.run_suffix}'),('ofa',f'ofa-iab-{args.run_suffix}')]:
        folder=root/name;status=json.loads((folder/'status.json').read_text());config=json.loads((folder/'resolved_config.json').read_text())
        if status['status']!='completed' or config['manifest_sha256']!=digest:raise ValueError('incomplete or mismatched run')
        predictions[key]=[json.loads(line) for line in (folder/'predictions.jsonl').read_text().splitlines()]
        runs[key]=dict(directory=str(folder.resolve()),metrics=json.loads((folder/'metrics.json').read_text()),
            stratified_metrics=json.loads((folder/'stratified_metrics.json').read_text()),code_commit=config['code_commit'],
            elapsed_seconds=status['elapsed_seconds'],weights=str(Path(config['weights']).resolve()),
            learned_asset=str((folder/('reference_bank.pt' if key=='ofa' else 'model.pth')).resolve()))
        if key=='ofa':runs[key]['denoiser']=str(Path(config['denoiser']).resolve())
    report=dict(status='completed',dataset='ImageAttributionBench V3.2',manifest_sha256=digest,
        selected=len(rows),truth_counts=dict(Counter('ai' if r['is_ai'] else 'real' for r in rows)),
        split_counts=dict(Counter(r['split'] for r in rows)),source_counts=dict(Counter(r['source_label'] or 'real' for r in rows)),
        source_semantic_split_counts={f'{s}/{c}/{split}':n for (s,c,split),n in sorted(Counter((r['source_label'] or 'real',r['semantic_class'],r['split']) for r in rows).items())},
        format_truth_counts={f'{t}/{fmt}':n for (t,fmt),n in sorted(Counter(('ai' if r['is_ai'] else 'real',r['format']) for r in rows).items())},
        runs=runs,cascade=joined_cascade(rows,predictions['detection'],predictions['attribution'],predictions['ofa']),
        evaluation='development only; holdout not evaluated; selected-epoch metrics are not independent test results')
    with args.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(output=str(args.output.resolve()),selected=len(rows),cascade=report['cascade']),indent=2))


if __name__=='__main__':main()
