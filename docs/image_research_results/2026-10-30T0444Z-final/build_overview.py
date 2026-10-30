"""Build the final aggregate overview from accepted score-only reports."""
import json,csv
from pathlib import Path
import sys
ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
from research.image_attribution.protocol import sha256
OUT=Path(__file__).resolve().parent
sources={}
def read(path):
    p=Path(path)
    if not p.is_absolute():p=ROOT/p
    sources[str(p.relative_to(ROOT))]=sha256(p)
    return json.loads(p.read_text())
def mpath(run):return ROOT/'outputs/image_experiments'/run/'metrics.json'
def state(run):
    p=ROOT/'outputs/image_experiments'/run/'status.json';s=read(p);assert s['status']=='completed' and s['exit_code']==0
def points(m,fields):return {k:m[k] for k in fields if k in m}
def table(headers,rows):
    return ['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,row))+'|' for row in rows]
def pct(x):return f'{100*x:.2f}%'
delivery=read('outputs/image-next-routes-20261029/completion-audit/delivery.json')
prior_path=ROOT/delivery['snapshot']/'summary.json'
assert sha256(prior_path)==delivery['summary_sha256']
prior=read(prior_path);assert prior['all_registered_routes_verified'] and prior['holdout_scored'] is False
comparisons=read(OUT/'comparisons.json');acceptance=read(OUT/'acceptance.json')
assert acceptance['all_ten_stages_completed'] and acceptance['all_seven_reports_point_metrics_recomputed']
ffhq=read(OUT/'ffhq-threshold-transfer.json')
rejection=comparisons['ofa-tplus1-comparison']
base='followups-v7-200-q179-e30'
extensions={}
for name in ['D3','D1-control','D1','D2','D2-semantic','A1','A2','A2-semantic']:
    run=f'{base}/{name}/evaluate';state(run);m=read(mpath(run));assert m['holdout_evaluated'] is False
    extensions[name]=dict(source=run,conditions={key:points(v,['n','accuracy','macro_f1','top1','top3','known_closed_set_accuracy','known_acceptance','known_correct_and_accepted_rate','unknown_false_acceptance','unknown_rejection_rate','unknown_auroc'])|({'fixed_0_5':v['fixed_0_5']} if 'fixed_0_5' in v else {}) for key,v in m['conditions'].items()})
clean={}
m=read(mpath(base+'/D1-control/evaluate'));clean['ResNet-control']=m['conditions']['clean']['fixed_0_5']
clean['ResNet-augmented']=read(mpath(base+'/D1/evaluate'))['conditions']['clean']['fixed_0_5']
for name,run in [('OmniAID','omniaid-v2-q179-20261029'),('OpenCLIP','openclip-q179-20261029/D-OpenCLIP'),('ConvNeXt','convnext-q179-20261029/D-ConvNeXt')]:
    state(run);clean[name]=read(mpath(run))['fixed_0_5']
ablation='independent-image-tasks-v7-200-q179-e30-gpu'
dino=read(mpath(ablation+'/D-DINO'));clean['DINOv2']=points(dino,['n','accuracy','auroc','macro_f1'])|dict(fpr=dino['fpr_0_5'],tpr=dino['tpr_0_5'])
source={}
a2=read(mpath(base+'/A2/evaluate'));source['ResNet-A0-parent-dev-selected']=a2['conditions']['clean']
for name in ['A-RGB','A-FP','A-Fusion']:
    state(ablation+'/'+name);source[name]=read(mpath(ablation+'/'+name))
for label,run in [('OpenCLIP-linear','openclip-q179-20261029/A-OpenCLIP22'),('OpenCLIP-KNN','openclip-q179-20261029/A-OpenCLIP-KNN22'),('ConvNeXt-linear','convnext-q179-20261029/A-ConvNeXt22'),('DNA-Det-linear','dna-wild-q179-20261029/A-DNA-Wild-LP22'),('OmniDFA-linear','omnidfa-part1-q179-20261029/A-OmniDFA-LP22'),('OmniDFA-K10-prototype','omnidfa-part1-q179-20261029/A-OmniDFA-prototype22-K10')]:
    state(run);source[label]=read(mpath(run))
source={k:points(m,['n','top1','top3','macro_f1','best_epoch','training_epochs','epoch','k','best_k']) for k,m in source.items()}
fusion=read('outputs/image_experiments/resnet-feature-score-fusion-iab-v7-200-q179-e30/summary.json')
assert fusion['status']=='completed' and fusion['alpha_resnet']==.75 and fusion['holdout_scored'] is False
for k,v in fusion['metrics'].items():source['score-fusion-'+k]=v
for v in clean.values():assert v['n']==2310
for v in source.values():assert v['n']==1331
full={}
for name in ['d0-resnet50','a0-resnet50','ofa']:
    m=read(mpath(name+'-iab-v7-200-q179-e30'));assert m['holdout_evaluated'] is False
    full[name]=points(m,['task','n','query_n','accuracy','auroc','fpr_0_5','tpr_0_5','top1','top3','macro_auroc','epoch','selected_epoch','train_n','train_counts','class_weights','reference_counts'])
scope=dict(dataset='iab-balanced-v7-200-q179',effective_images=63821,ai=43821,real=20000,train=37338,dev=7189,holdout=19294,
    manifest_sha256='9dd85636e2a19dbe833d9ebe2bee0c2a6b6eaf43015699e4fd9f27db3694b4c3',nominal_per_cell=200,actual_generated_per_cell=[187,200],quarantined_new_members=179,seed=42,holdout_scored=False)
limitations=[
 'Single seed, previously inspected developer evaluation, with parent-dev selection bias in A0/legacy baselines; no independent external test.',
 'Frozen-head comparisons share exact role IDs, but backbones, pretraining data, image size and normalization differ. ResNet fine-tuning is not a controlled backbone-only comparison.',
 'Class imbalance and D0 inverse-frequency weights,100-version GPU1 versus200-version GPU0, only new unanchored groups using corrected split assignment, and179 new duplicate-bridge exclusions prevent pure sample-size causal claims.',
 'Official pretraining overlap with IAB images or generator versions is unknown. OmniDFA has documented source-name overlap; DNA source/weight and OmniDFA source licensing remain unestablished.',
 'FFHQ evaluation contains93real and132AI, not a broad external domain. No mechanism for the misclassification was proven by the file-format audit.',
 'Unknown transfer only tests64kling calibration to63grok3 evaluation with20known classes. Zero false accepts does not establish a zero population error rate. Wilson intervals ignore group correlation; paired group bootstrap is supplied separately.',
 'OSCR is a score-ranking statistic; current intervals cover the specified rejection metrics, not OSCR. No unknown threshold is transferred to the22-source UI.',
 'OFA20+1 is frozen official simulator and extractor plus a project head,1024train-only reconstructions and8fixed architectures. Full simulator/extractor pretraining and exact paper reproduction were not performed.',
 'DNA is the official wild backbone plus a project linear probe, not full contrastive/AWL retraining. OmniDFA is one fixed part1 checkpoint and deterministic per-image views, not the original random episodic benchmark.',
 'D1 augmentation was evaluated in the clean matched condition; matched augmented-detector JPEG/resize robustness tests were not registered, so robustness improvement cannot be claimed.',
 'KNN chose k1; Top3 partly depends on zero-vote class tie ordering. K10 prototype and linear probes have different training/support budgets.',
 'A3 full architecture-family mapping remains unreviewed/blocked. Complete OFA pretraining and further routes remain outside the selected scope.',
 'Canonical batch32 calibration scores are used for calibrated detector results. Re-exported batch128 scores have small float32 differences; the previous OpenCLIP acceptance documented one changed5% calibration decision, with evaluation decisions unchanged. ConvNeXt export max error<=5.37e-7 is archived separately.']
overview=dict(status='completed',scope=scope,all_registered_calculations_completed=True,clean_detection_2310=clean,
    held_ffhq_225=ffhq,source_classification_1331=source,unknown_rejection_1275=rejection,
    baseline_full_dev_separate_denominators=full,previous_extensions=extensions,score_fusion_calibration_only=fusion,
    registered_comparisons=comparisons,limitations=limitations,
    result_snapshots=['docs/image_research_results/2026-10-30-convnext-completed','docs/image_research_results/2026-10-30T0414Z-source-adaptations',str(OUT.relative_to(ROOT)),delivery['snapshot']],
    historical_comparisons=['outputs/image-followups-20261027/epochs20-30-comparison','outputs/image-followups-20261028/v6-100-e30-vs-v7-200-q179-e30'])
(OUT/'overview.json').write_text(json.dumps(overview,indent=2,allow_nan=False)+'\n')
lines=['# 图片检测与来源研究阶段总览（2026-10-30）','',
 '本轮全部已登记GPU计算、七项保存分数对照与验收均完成。原始结果、失败诊断、数据和模型保留。最终入口见overview.json，复制工件与源SHA见snapshot.json和overview-inputs.json。此目录为研究结果归档，不部署或修改线上服务。','',
 '结论：FFHQ跨语义检测仍以原OpenCLIP路线较好，本轮ConvNeXt没有进一步降低固定.5误报；OFA20+1以大幅丢失已知覆盖换取63/63未知拒识，综合拒识能力没有改善。新冻结来源头中DNA优于OpenCLIP参考，OmniDFA线性差异区间跨0。','',
 '## 数据与口径','',
 '200-q179有效63821张：43821AI+20000真实；train37338/dev7189/holdout19294。原64000文件保留，用户批准隔离179新增冲突图，实际187–200张/生成小格。旧42000ID/split/group/标签/hash不变，holdout未推理。所有比例用各自下述分母；不能将7189完整dev、2310检测evaluation、1331来源evaluation、1275开放集evaluation混算。单seed42，训练头30epochs/batch32/Adam.001/StepLR(1,.9)，最早最佳selection epoch；冻结/原型/KNN epoch=null。','',
 '## 真假检测：同2310 evaluation（979真实+1331AI），阈值固定.5','']
lines+=table(['路线','AUROC','FPR','TPR'],[[name,pct(v['auroc']),pct(v['fpr']),pct(v['tpr'])] for name,v in clean.items()])
lines+=['','OpenCLIP仍是此clean开发比较中的优选。ResNet-control/D3曾有完全相同真实图概率向量，不能当独立复现。D1增强的clean TPR上升但FPR也上升，未验证增强后的JPEG/缩放鲁棒性。原D0完整7189dev AUROC99.5535%、FPR2.8743%、TPR95.9733%另列于overview，不能与2310表直接作改善比较。','',
 '## FFHQ跨语义：同225 evaluation（93真实+132AI），训练/selection/cal均排除FFHQ','']
lines+=table(['路线','误报/93','FPR@.5','TPR@.5','AUROC'],[[name,f"{v['fixed_0_5']['false_positives']}/93",pct(v['fixed_0_5']['fpr']),pct(v['fixed_0_5']['tpr']),pct(v['fixed_0_5']['auroc'])] for name,v in ffhq['methods'].items()])
ci=ffhq['fixed_0_5_paired_group_intervals']['OpenCLIP-heldFFHQ_minus_ResNet-heldFFHQ']['delta_95_ci']['fpr_0_5']
lines+=['',f'OpenCLIP相对ResNet固定.5的FPR变化为−44.09个百分点，1000次原group配对95%区间[{100*ci[0]:.2f},{100*ci[1]:.2f}]个百分点。ConvNeXt相对ResNet固定.5误报计数不变，但AUROC排序改善；这不等于误报问题解决。该文件格式审计尚未证明具体像素成因或语义捷径。','',
 '阈值只用非FFHQ calibration的914张真实图选择。下表“目标1%/5%”是cal目标，FFHQ evaluation始终只报告：','']
lines+=table(['路线','目标cal FPR','cal FPR','FFHQ误报/93','FFHQ FPR','FFHQ TPR','阈值'],[[name,target,pct(r['calibration']['fpr']),f"{r['evaluation']['false_positives']}/93",pct(r['evaluation']['fpr']),pct(r['evaluation']['tpr']),f"{r['threshold']:.10f}"] for name,v in ffhq['methods'].items() for target,r in v['calibrated'].items()])
lines+=['','例如OpenCLIP cal目标1%在FFHQ实际仍误报7/93=7.53%，TPR98.48%；Wilson95% FPR约3.69–14.73%。不是跨域1%保证。ffhq-threshold-transfer.json同时保存每模型阈值稳定性、1000次共享cal/eval原group抽样并在cal重拟阈值的FPR/TPR区间和配对差值。','',
 '## 22来源：同1331张已知AI evaluation','']
lines+=table(['路线','Top1','Top3','MacroF1'],[[name,pct(v['top1']),pct(v['top3']),pct(v['macro_f1'])] for name,v in source.items()])
lines+=['',
 '主参考为同role/selection协议的冻结OpenCLIP线性头：DNA Top1提升12.55pp，group95%区间[9.20,15.92]pp；ConvNeXt提升4.43pp，[1.78,7.40]pp；OmniDFA线性提升0.60pp，[-2.79,4.02]pp跨0；K10原型下降11.95pp，[-15.36,-8.57]pp。七报告保存逐来源和原group配对区间。K10固定每来源10不同train组，epoch=null。','',
 'A0 ResNet及其score fusion被单列为parent-dev选模的探索参考，不是同等独立选模排名。原RGB/FP特征融合优于各单分支，但容量和表示均不同。分数融合alpha=.75只由1554张AI calibration最小NLL确定，固定50/50也事前登记；虽50/50的evaluation Top1更高，不能据此改选权重。闭集概率以“输入已知AI”为条件，不代表未知来源概率。','',
 '原冻结OFA配对验证在完整4197AI dev的宏AUC89.22%，每来源100个train参考，epoch=null；其配对分数不与分类softmax混为一个排名。参考ID变化、D0频率权重和规模比较的限制保留在原100/200配对报告。','',
 '## 未知拒识：同20known、cal64kling、eval63grok3+1212known','']
methods=rejection['methods']
lines+=table(['方法','未知误接收/63','未知拒绝率','已知保留/1212','正确且保留/1212','未知AUROC','OSCR'],[[name,f"{v['unknown_false_acceptance_count']}/63",pct(v['unknown_rejection']),f"{v['known_retention_count']}/1212 ({pct(v['known_retention'])})",f"{v['known_correct_and_retained_count']}/1212 ({pct(v['known_correct_and_retained'])})",pct(v['unknown_auroc']),pct(v['oscr'])] for name,v in methods.items()])
lines+=['',
 'OFA20+1 30轮、selection选第18轮：未知0/63误接收，但已知仅保留58/1212=4.79%，正确且保留28/1212=2.31%，未知AUROC43.38%、OSCR21.70%。相对ResNet MSP，已知保留下降42.74pp，cal重拟阈值+原group配对95%区间约[-54.72,-20.78]pp；未知误接收变化区间[-18.30,0]pp含0边界，不能只凭63/63拒绝宣称有效提升。零误接收的Wilson95%上界仍约5.75%。','',
 '所有拒识阈值仅用64张kling拟5%未知误接收（实际cal3/64），grok3从不选阈值。OFA训练unknown来自1024不同real-train组、8固定架构各128张的官方模拟器重建；没有将kling/grok3或holdout加入训练。1-p_unknown/MSP/cosine均不是已校准未知概率。仅验证kling→grok3这一个迁移设定；不把20类阈值移到22类主接口。OSCR只报点值，未提供其bootstrap区间。','',
 '## 既有拓展与未执行范围','',
 'D2排除4o：59AI+979真实的1038evaluation，AUROC99.6849%、FPR4.8008%、TPR98.3051%，正例少且单一来源，不代表普遍未知检测。A2来源六条件同1331AI：clean Top1 77.16%、JPEG95 76.26%、JPEG75 75.88%、JPEG50 73.25%、resize.75 76.71%、resize.5 74.00%。A2-semantic在132张FFHQ AI上Top1 80.30%、Top3 93.94%，它是来源分类，不是93真实FFHQ的误报测试。D3无训练epoch，阈值校准和父dev偏差保留。','',
 '本次已完成：原100/20、100/30、200-q179/30基线与四分支、融合、D1/D2/D3/A1/A2及语义拓展、FFHQ审计、冻结OmniAID/OpenCLIP、冻结ConvNeXt/DNA/OmniDFA及OFA20+1适配。旧CPU30诊断、raw200分组桥接失败、资产传输/Namespace恢复证据均保留，不冒充新失败或删除。','',
 '未执行：用户选择范围之外的OFA模拟器100万步及指纹提取器10万步完整预训练；DNA完整contrastive/AWL重训与OmniDFA原随机episodic任务；A3完整22类架构家族映射因证据未齐仍blocked；不扩500、不下载额外骨干、不新增未登记路线。','',
 '## 局限与使用建议','']
lines += [f'- {s}' for s in limitations]
lines += ['', '页面交接继续以已验收OpenCLIP检测及A0 ResNet来源排序为默认候选；OFA/融合/拒识只在研究模式明确范围和覆盖损失。研究结果不足以支持生产未知拒识保证。未改变任何线上服务。', '',
 '## 工件与复核','',
 '- overview.json：各路线机器可读总表、同分母结果、局限与原路径',
 '- acceptance.json、verify_final.py：十阶段成功、OFA完整训练/图像重载/保存scores及七报告点指标复核',
 '- ffhq_score_audit.py、ffhq-threshold-transfer.json、ffhq-analysis-inputs.json：FFHQ保存scores阈值迁移及group区间，无图片推理',
 '- comparisons.json及artifacts/*comparison：七个原配对报告、逐来源变化与95%区间',
 '- snapshot.json、overview-inputs.json：复制文件与所有输入source路径/SHA；训练history/config/模型SHA在artifacts',
 '- 先前ConvNeXt、DNA/OmniDFA和OmniAID/OpenCLIP结果快照保持不覆盖。仅聚合结果/代码/配置/哈希入Git，模型、图像、缓存与每图预测保留服务器。','']
(OUT/'README.md').write_text('\n'.join(lines).rstrip()+'\n')
with (OUT/'route-table.csv').open('w',newline='') as f:
    w=csv.writer(f,lineterminator="\n");w.writerow(['task','route','n','auroc','fpr','tpr','top1','top3','macro_f1'])
    for name,v in clean.items():w.writerow(['clean_detection',name,v['n'],v['auroc'],v['fpr'],v['tpr'],'','',v.get('macro_f1','')])
    for name,v in source.items():w.writerow(['known_source22',name,v['n'],'','','',v['top1'],v['top3'],v['macro_f1']])
for p,h in sources.items():assert sha256(ROOT/p)==h
sources[str(Path(__file__).relative_to(ROOT))]=sha256(__file__)
(OUT/'overview-inputs.json').write_text(json.dumps(dict(inputs=sources,images_read=False,holdout_scored=False),indent=2)+'\n')
print(json.dumps(dict(status='overview_built',clean_routes=len(clean),source_routes=len(source),registered_comparisons=7,unknown_methods=len(methods),holdout_scored=False)))
