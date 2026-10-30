"""Verify completed source adaptations from saved scores and immutable identities only."""
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import torch
ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
from research.image_attribution.protocol import sha256
from research.image_attribution.openclip_route import digest,job_parts
from research.image_attribution.followups import validate_probabilities
from scripts.run_image_followup import class_report
from scripts.compare_image_scale_epochs import indexed
from scripts.run_image_dna import prepare as dna_prepare
from scripts.run_image_omnidfa import prepare as omni_prepare
from research.image_attribution.omnidfa_route import support_manifest
OUT=Path(__file__).resolve().parent
sources={};copied=[];summaries={}
def read(path,lines=False):
    path=Path(path);sources[str(path.relative_to(ROOT))]=sha256(path)
    return [json.loads(x) for x in path.read_text().splitlines()] if lines else json.loads(path.read_text())
def copy(path,relative):
    target=OUT/'artifacts'/relative;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(path,target);h=sha256(path);assert sha256(target)==h
    copied.append(dict(source=str(path.relative_to(ROOT)),path=str(target.relative_to(OUT)),sha256=h,bytes=path.stat().st_size))
def state(path):
    s=read(path);assert s['status']=='completed' and s['exit_code']==0
def check_reports(a,b):
    for k in ('n','accuracy','macro_f1','top1','top3'):assert np.isclose(a[k],b[k],atol=1e-10,rtol=0),(k,a[k],b[k])
assert not (OUT/'snapshot.json').exists()
for route,prepare,names in [('dna',dna_prepare,['A-DNA-Wild-LP22']),('omnidfa',omni_prepare,['A-OmniDFA-LP22','A-OmniDFA-prototype22-K10'])]:
    cp=ROOT/f'configs/image_experiments/iab_{route}_q179_20261029.json';c=read(cp)
    data,rows,cache_rows,identity,job=prepare(c)
    root=ROOT/c['output_root'];state(root/'status.json')
    reg=read(root/'registration.json');assert reg['config']==c and reg['config_sha256']==digest(c) and reg['job']==job
    cache=read(root/'cache.json');assert cache['identity']==identity and cache['n']==27545 and cache['holdout_read'] is False
    assert sha256(root/'features.pt')==cache['sha256'];sources[str((root/'features.pt').relative_to(ROOT))]=cache['sha256']
    for key,h in c['implementation_sha256'].items():assert sha256(ROOT/key)==h;sources[key]=h
    sources[c['asset']['path']]=c['asset']['sha256']
    copy(cp,Path(route)/'config.json')
    for f in ('registration.json','launch.json','status.json'):read(root/f);copy(root/f,Path(route)/f)
    for name in names:
        p=root/name;state(p/'status.json');m=read(p/'metrics.json');pr=read(p/'provenance.json')
        parts,classes=job_parts(rows,c['experiments'][name],42)
        ids={k:[r['sample_id'] for r in v] for k,v in parts.items()}
        assert read(p/'roles.json')==ids and classes==m['classes']==pr['classes']
        assert {k:len(v) for k,v in parts.items()}==pr['role_counts']
        assert pr['role_ids_sha256']=={k:digest(v) for k,v in ids.items()}
        groups={k:{r['group_id'] for r in v} for k,v in parts.items()}
        for a in groups:
            for b in groups:
                if a!=b:assert not groups[a]&groups[b]
        assert m['holdout_scored'] is False and pr['holdout_scored'] is False and m['failed_samples']==[]
        assert pr['code']==c['implementation_sha256'] and pr['manifest_sha256']==c['manifest_sha256']
        assert pr['selection_sha256']==c['selection_sha256'] and pr['feature_cache_sha256']==cache['sha256']
        checks=m['actual_image_verification']
        assert checks['full_offline_backbone_reload'] is True and checks['images_opened']==2
        assert checks['selection_sample_ids']==ids['selection'][:2]
        assert all(np.isfinite(v) and v<=1e-5 for k,v in checks.items() if k.endswith('max_error'))
        records=read(p/'predictions.jsonl',True);assert {r['role'] for r in records}=={'calibration','evaluation'}
        values={}
        for role in ('calibration','evaluation'):
            selected=indexed([r for r in records if r['role']==role]);assert set(selected)==set(ids[role])
            for row in parts[role]:
                r=selected[row['sample_id']];assert r['condition']=='clean' and r['classes']==classes and row['split']=='dev'
                assert all(r[k]==row[k] for k in ('group_id','source_label','is_ai','semantic_class'))
            values[role]=validate_probabilities([selected[s]['scores'] for s in ids[role]],len(ids[role]),22)
        report=class_report(parts['evaluation'],classes,values['evaluation']);check_reports(report,m)
        if 'prototype' in name:
            assert m['training_epochs'] is None and m['epoch'] is None and m['k']==10
            support=read(p/'support.json');expected=support_manifest(parts['train'],classes,42)
            assert support==expected and digest(support)==c['expected_support_sha256']==m['support_sha256']
            # Every support identity is in a distinct original train group for its class.
            for source in classes:
                ss=[r for r in support['samples'] if r['source_label']==source]
                assert len(ss)==len({r['group_id'] for r in ss})==10 and all(r['split']=='train' for r in ss)
            hp=p/'prototypes.pth'
            saved=torch.load(hp,map_location='cpu',weights_only=True)
            assert saved['classes']==classes and saved['support']==support
            centers=saved['prototypes'];assert centers.shape==(22,128) and torch.isfinite(centers).all()
            for role in ('calibration','evaluation'):
                spec=m['raw_cosine_files'][role];path=p/spec['path']
                assert sha256(path)==spec['sha256'] and spec['classes']==classes and spec['sample_ids_sha256']==digest(ids[role])
                sources[str(path.relative_to(ROOT))]=spec['sha256']
                raw=np.load(path,allow_pickle=False);assert raw.shape==values[role].shape and np.isfinite(raw).all()
                recomputed=torch.tensor(raw,dtype=torch.float64).softmax(1).numpy()
                assert np.allclose(recomputed,values[role],rtol=0,atol=1e-12)
            copy(p/'support.json',Path(route)/name/'support.json')
            copy(p/'cosine-score-files.json',Path(route)/name/'cosine-score-files.json')
        else:
            assert m['training_epochs']==30
            state(p/'linear/status.json')
            history=read(p/'linear/history.jsonl',True);assert [r['epoch'] for r in history]==list(range(1,31))
            best=max(history,key=lambda r:r['selection']['macro_f1'])
            assert m['best_epoch']==best['epoch'] and np.isclose(best['selection']['macro_f1'],m['selection_macro_f1'])
            assert m['reload_max_error']<=1e-5 and m['single_batch_max_error']<=1e-5
            hp=p/'linear/head.pth';head=torch.load(hp,map_location='cpu',weights_only=True)
            assert head['classes']==classes and head['input_dimensions']==pr['feature_dimensions'] and head['best_epoch']==m['best_epoch']
            assert head['variant']==route
            for f in ('history.jsonl','status.json','provenance.json'):read(p/'linear'/f,f.endswith('jsonl'));copy(p/'linear'/f,Path(route)/name/'linear'/f)
        h=sha256(hp);assert h==m['head_asset_sha256']==pr['head_asset_sha256'];sources[str(hp.relative_to(ROOT))]=h
        summaries[name]=dict(n=m['n'],top1=m['top1'],top3=m['top3'],macro_f1=m['macro_f1'],
            epochs=m['training_epochs'],selected_epoch=m.get('best_epoch'),role_counts={k:len(v) for k,v in parts.items()},
            head_sha256=h,actual_image_verification=checks,confusion_matrix=report['confusion_matrix'],
            per_source=report['per_source'],per_semantic=report['per_semantic'],classes=classes,status='completed')
        for f in ('status.json','metrics.json','provenance.json'):copy(p/f,Path(route)/name/f)
for path,h in sources.items():assert sha256(ROOT/path)==h
(OUT/'summary.json').write_text(json.dumps(dict(experiments=summaries,holdout_scored=False,paired_comparison_pending=True),indent=2)+'\n')
(OUT/'snapshot.json').write_text(json.dumps(dict(sources=sources,copied=copied,images_read=False,models_or_features_copied=False),indent=2)+'\n')
(OUT/'README.md').write_text('DNA-Det骨干适配与OmniDFA两项来源对照已完成。线性头真实30轮、selection选模；OmniDFA K10原型epoch=null。共同1331张AI evaluation：DNA Top1 70.0225% / Top3 90.4583%；OmniDFA线性58.0766% / 83.6965%；K10原型45.5297% / 73.7040%。完整角色/标签/词表、保存scores复算、模型/代码/缓存SHA、实际两张selection重载及单批一致性通过。它们是来源适配，不能当作FFHQ真假误报改善证据。配对group区间待现有队列完成后另归档。单seed开发结果，官方预训练重叠和许可局限保留。没有读取holdout图像推理，模型/缓存/每图预测未复制Git。\n')
print(json.dumps({k:{f:v[f] for f in ('n','top1','top3','epochs','selected_epoch')} for k,v in summaries.items()},indent=2))
