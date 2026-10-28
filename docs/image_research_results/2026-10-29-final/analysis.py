from pathlib import Path
import argparse,csv,datetime,itertools,json,re,sys,subprocess
import numpy as np
parser=argparse.ArgumentParser(description="Read-only saved IAB follow-up verification and paired analysis.")
parser.add_argument("--output",default="docs/image_research_results/2026-10-29-final")
args=parser.parse_args()
ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
from scripts.compare_image_scale_epochs import Evidence,indexed,stats,group_bootstrap,sha256
from scripts.compare_iab_v6_100_vs_v7_200_q179 import detection_bootstrap,source_rows
from research.image_attribution.followups import choose_threshold,open_set_report
out=ROOT/args.output
assert not out.exists(),"Existing output must be preserved."
e=Evidence();cp=ROOT/"configs/image_experiments/iab_followups_v7_200_q179_e30.json";config=e.read(cp)
assert sha256(cp)=="3cd648391e4f60c04c64e38f057e6f9e6c703e34adba602deb8a90f46d1e3888"
mp=ROOT/config["dataset"]/"manifest.jsonl";manifest=indexed(e.read(mp,lines=True))
assert len(manifest)==63821 and sha256(mp)==config["manifest_sha256"]
base=ROOT/config["output_root"];qd=ROOT/"outputs/image-followups-20261028/followups-q179-queue"
queue=e.read(qd/"queue.json");items=queue["items"];assert len(items)==14 and items[-1]=="A2-semantic:evaluate"
ql=(qd/"queue.log").read_text();e.inputs[str((qd/"queue.log").resolve())]=sha256(qd/"queue.log")
assert ql.rstrip().splitlines()[-1].startswith("queue=completed ")
proc=Path("/proc")/str(queue["pid"]);alive=False
if (proc/"stat").exists():
 st=(proc/"stat").read_text();t=st[st.rfind(")")+2:].split()
 alive=queue["identity"]==[Path("/proc/sys/kernel/random/boot_id").read_text().strip(),t[19]]
assert not alive
mapping=e.read(ROOT/config["experiments"]["A3"]["family_mapping"])
assert mapping.get("reviewed") is False and not (base/"A3").exists()
sources=[];runs={};predictions={};roles_by_exp={}
for item in items:
 exp,action=item.split(":");folder=base/exp/action
 status=e.read(folder/"status.json");metric=e.read(folder/"metrics.json");resolved=e.read(folder/"resolved_config.json")
 roles=e.read(folder/"roles.json");assets=e.read(folder/"asset_manifest.json")
 assert status["status"]=="completed" and status["exit_code"]==0,item
 assert metric["holdout_evaluated"] is False and resolved["holdout_evaluated"] is False
 assert resolved["manifest_sha256"]==config["manifest_sha256"] and resolved["config_sha256"]==sha256(cp)
 assert resolved["roles_sha256"]==sha256(folder/"roles.json")
 for name,meta in assets.items():
  f=folder/name;assert f.resolve().is_relative_to(folder.resolve()) and sha256(f)==meta["sha256"],(item,name)
  e.inputs[str(f.resolve())]=meta["sha256"]
 sets={k:set(v) for k,v in roles.items()};groups={k:{manifest[s]["group_id"] for s in v} for k,v in sets.items()}
 for k,v in sets.items():
  assert len(v)==len(roles[k]) and all(manifest[s]["split"]==("train" if k=="train" else "dev") for s in v)
  if config["experiments"][exp]["task"]=="attribution":assert all(manifest[s]["is_ai"] for s in v)
 for a,b in itertools.combinations(sets,2):assert sets[a].isdisjoint(sets[b]) and groups[a].isdisjoint(groups[b])
 if exp in roles_by_exp:assert roles_by_exp[exp]==roles
 roles_by_exp[exp]=roles
 if exp=="D2":
  for k,v in sets.items():
   if k!="evaluation":assert all(manifest[s]["source_label"]!="4o" for s in v)
   else:assert all(not manifest[s]["is_ai"] or manifest[s]["source_label"]=="4o" for s in v)
 if exp in {"D2-semantic","A2-semantic"}:
  for k,v in sets.items():assert all((manifest[s]["semantic_class"]=="FFHQ")==(k=="evaluation") for s in v)
 if exp=="A1":
  for k,v in sets.items():
   if k!="calibration":assert all(manifest[s]["source_label"]!="kling" for s in v)
   if k!="evaluation":assert all(manifest[s]["source_label"]!="grok3" for s in v)
 record={"status":status,"metrics":metric,"role_counts":{k:len(v) for k,v in roles.items()},
         "roles_sha256":resolved["roles_sha256"],"assets_verified":len(assets),
         "ids_and_groups_disjoint":True,"holdout_evaluated":False}
 if action=="train":
  history=e.read(folder/"history.json");assert [x["epoch"] for x in history]==list(range(1,31))
  assert 1<=metric["selected_epoch"]<=30 and metric["failed_samples"]==0
  assert metric["reload_max_abs_error"]<=1e-5 and metric["single_vs_batch_max_abs_error"]<=1e-5
  effective=e.read(folder/"effective_training_rows.jsonl",lines=True)
  assert {x["sample_id"] for x in effective if x["split"]=="train"}==sets["train"]
  assert {x["sample_id"] for x in effective if x["split"]=="dev"}==sets["selection"]
  assert {x["sample_id"] for x in effective}==sets["train"]|sets["selection"]
  model=folder/"model.pth";digest=sha256(model);e.inputs[str(model.resolve())]=digest
  record.update(completed_epochs=30,model_sha256=digest,effective_train_selection_verified=True)
 else:
  checkpoint=Path(metric["checkpoint"]);assert sha256(checkpoint)==metric["checkpoint_sha256"]
  e.inputs[str(checkpoint.resolve())]=metric["checkpoint_sha256"];record["checkpoint_sha256"]=metric["checkpoint_sha256"]
  pred=e.read(folder/"predictions.jsonl",lines=True);assert all(x["role"] in {"calibration","evaluation"} for x in pred)
  bycondition={}
  for condition in metric["conditions"]:
   entries=[x for x in pred if x["condition"]==condition]
   for k in ("calibration","evaluation"):
    part=[x for x in entries if x["role"]==k];assert len(part)==len(sets[k]) and {x["sample_id"] for x in part}==sets[k]
   for row in entries:assert all(row[k]==manifest[row["sample_id"]][k] for k in ("group_id","source_label","semantic_class","is_ai"))
   er=sorted((x for x in entries if x["role"]=="evaluation"),key=lambda x:x["sample_id"])
   classes=er[0]["classes"];assert all(x["classes"]==classes for x in entries);scores=np.asarray([x["scores"] for x in er])
   if config["experiments"][exp]["task"]=="detection":
    y=np.array([classes.index("ai") if x["is_ai"] else classes.index("real") for x in er])
    recalculated=stats(y,scores,"D0",classes);actual=metric["conditions"][condition]["fixed_0_5"]
    for k,o in (("n","n"),("accuracy","accuracy"),("macro_f1","macro_f1"),("auroc","auroc"),("fpr_0_5","fpr"),("tpr_0_5","tpr")):assert abs(recalculated[k]-actual[o])<1e-10,(item,k)
   elif exp=="A1":
    assert len(classes)==20 and not {"kling","grok3"}&set(classes)
    cu=[x for x in entries if x["role"]=="calibration" and x["source_label"] not in classes]
    assert len(cu)==64 and {x["source_label"] for x in cu}=={"kling"}
    unknown=[x for x in er if x["source_label"] not in classes];assert len(unknown)==63 and {x["source_label"] for x in unknown}=={"grok3"}
    threshold=choose_threshold([max(x["scores"]) for x in cu],.05)
    recalculated=open_set_report([x["source_label"] for x in er],classes,scores,threshold);actual=metric["conditions"][condition]
    for k,v in recalculated.items():
     if isinstance(v,(int,float)):assert abs(v-actual[k])<1e-10,(item,k)
    assert abs(sum(max(x["scores"])>=threshold for x in cu)/64-actual["calibration_unknown_false_acceptance"])<1e-12
    y=None
   else:
    y=np.array([classes.index(x["source_label"]) for x in er]);recalculated=stats(y,scores,"A0",classes);actual=metric["conditions"][condition]
    for k in ("n","accuracy","macro_f1","top1","top3"):assert abs(recalculated[k]-actual[k])<1e-10,(item,k)
   bycondition[condition]={"rows":er,"scores":scores,"classes":classes,"y":y}
  predictions[exp]=bycondition;record["evaluation_metrics_recomputed"]=True
 runs[item]=record
 for name in ("status.json","metrics.json","history.json","resolved_config.json","stratified_metrics.json","confusion_matrix.csv","asset_manifest.json","launcher.json"):
  if (folder/name).is_file():sources.append((folder/name,f"{exp}/{action}/{name}"))
 if action=="train":sources.append((folder/"run.log",f"{exp}/train/training-log.txt"))
print("VERIFIED_REGISTERED_ACTIONS",len(runs),flush=True)
def paired(exp,other,condition="clean",other_condition="clean"):
 cur=predictions[exp][condition];ref=predictions[other][other_condition]
 lookup={x["sample_id"]:(x,s) for x,s in zip(ref["rows"],ref["scores"])};assert cur["classes"]==ref["classes"]
 p=[];q=[]
 for row,score in zip(cur["rows"],cur["scores"]):
  before,ps=lookup[row["sample_id"]];assert all(row[k]==before[k] for k in ("group_id","source_label","is_ai","semantic_class","classes"))
  p.append(ps);q.append(score)
 p,q=np.asarray(p),np.asarray(q);classes=cur["classes"];y=cur["y"];g=[x["group_id"] for x in cur["rows"]]
 detection=config["experiments"][exp]["task"]=="detection";task="D0" if detection else "A0";a,b=stats(y,p,task,classes),stats(y,q,task,classes)
 boot=detection_bootstrap(y,p,q,g,classes) if detection else group_bootstrap(y,p,q,g)
 result={"n":len(y),"same_ids_labels_groups_verified":True,"before":a,"after":b,
 "delta_after_minus_before":{k:b[k]-a[k] for k in b if k!="n"},"group_bootstrap":boot,
 "scope":"conditional on frozen checkpoints; no retraining uncertainty"}
 src=[] if detection else source_rows(f"{exp}:{condition} minus {other}:{other_condition}",y,p,q,classes,g)
 return result,src
comparisons={};per_source=[]
for key,exp,other in [("D1_vs_clean_control","D1","D1-control"),("D2_vs_in_domain_control_same_held_source_subset","D2","D1-control"),("D2_semantic_vs_in_domain_control_same_FFHQ_subset","D2-semantic","D1-control"),("A2_semantic_vs_in_domain_A0_same_FFHQ_subset","A2-semantic","A2")]:
 comparisons[key],src=paired(exp,other);per_source.extend(src)
for condition in ("jpeg95","jpeg75","jpeg50","resize075","resize050"):
 comparisons[f"A2_{condition}_vs_clean"],src=paired("A2","A2",condition,"clean");per_source.extend(src)
print("PAIRED_COMPARISONS",len(comparisons),flush=True)
main=ROOT/"outputs/image-followups-20261028/v6-100-e30-vs-v7-200-q179-e30";done=e.read(main/"completion.json")
assert done["status"]=="completed" and done["holdout_scored"] is False
for name,key in (("summary.json","summary_sha256"),("provenance.json","provenance_sha256")):
 assert sha256(main/name)==done[key];e.inputs[str((main/name).resolve())]=done[key]
for name in ("scripts/compare_image_scale_epochs.py","scripts/compare_iab_v6_100_vs_v7_200_q179.py","research/image_attribution/followups.py"):
 e.inputs[str((ROOT/name).resolve())]=sha256(ROOT/name)
e.unchanged()
now=datetime.datetime.now(datetime.timezone.utc)
summary={"scope":"registered q179 follow-ups only, not production readiness","verified_at":now.astimezone(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
"source_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
"config_sha256":sha256(cp),"manifest_sha256":config["manifest_sha256"],
"dataset":{"total":63821,"ai":43821,"real":20000,"quarantined_new":179,"raw_preserved":64000,"parent_ids_preserved":42000},
"registered_actions_verified":14,"runs":runs,"A3":"not run: family mapping reviewed=false",
"queue_completion_logged":True,"matching_queue_process_alive":False,"queue_pid":queue["pid"],"paired_comparisons":comparisons,
"main_stage_summary_sha256":done["summary_sha256"],"holdout_evaluated":False,"training_executed_by_analysis":False,"image_inference_executed_by_analysis":False}
out.mkdir(parents=True,exist_ok=False);index=[]
sensitive=re.compile(r"hf_[A-Za-z0-9]{24,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|X-Amz-(?:Signature|Credential)=|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY")
for src,name in sources:
 data=src.read_bytes();assert len(data)<2000000 and not sensitive.search(data.decode())
 dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data);assert sha256(dest)==sha256(src)
 index.append({"file":name,"source":str(src.relative_to(ROOT)),"bytes":len(data),"sha256":sha256(src)})
def write_json(name,value):(out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")
write_json("source-files.json",index);write_json("summary.json",summary);write_json("paired-comparisons.json",comparisons)
with (out/"paired-per-source.csv").open("w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in per_source for k in row)),lineterminator="\n");w.writeheader();w.writerows(per_source)
(out/".gitattributes").write_text("*.csv -text whitespace=cr-at-eol\n")
source=globals().get("ANALYSIS_SOURCE")
if source is None:source=Path(__file__).read_text()
(out/"analysis.py").write_text(source)
def pct(v):return f"{100*v:.2f}%"
def dci(c,k):
 v=c["delta_after_minus_before"][k];lo,hi=c["group_bootstrap"]["delta_95_ci"][k]
 return f"{100*v:+.2f}pp [{100*lo:+.2f},{100*hi:+.2f}]"
d1=comparisons["D1_vs_clean_control"];sem=comparisons["A2_semantic_vs_in_domain_A0_same_FFHQ_subset"]
a1=runs["A1:evaluate"]["metrics"]["conditions"]["clean"];a2=runs["A2-semantic:evaluate"]["metrics"]["conditions"]["clean"]
text=f"""# IAB 200-q179：登记拓展实验全部完成

核验时间：{summary['verified_at']}。本批14项train/evaluate均completed/exit0；6项训练各连续完成30轮。唯一队列于2026-10-29 01:46:16（+08）记录queue=completed，原身份进程已结束。本次是已登记实验验收，不代表生产高可靠和通用未知拒识目标已经达到。

[完整指标](summary.json) · [原group配对比较](paired-comparisons.json) · [逐来源CSV](paired-per-source.csv) · [来源SHA](source-files.json) · [只读分析脚本](analysis.py)

|路线|实际逻辑|状态|
|---|---|---|
|D1-control / D1|同role/seed ResNet50，无增强与JPEG/缩放增强|各30轮及clean评估完成|
|D2|train/selection/calibration排除4o，只在未见4o AI与真实图验证|30轮及评估完成|
|D2-semantic|train/selection/calibration排除FFHQ，只在FFHQ验证|30轮及评估完成|
|D3|冻结D0，calibration真实图选择1%/2%/5%目标FPR阈值|完成，不训练|
|A1|20已知来源，kling未知校准、grok3未知评估；最大softmax拒识|30轮及评估完成|
|A2|固定A0，clean/JPEG95/75/50/缩放0.75/0.5六条件同ID比较|完成，不新增训练|
|A2-semantic|22类来源分类，训练及选模、校准排除FFHQ|30轮及评估完成|
|A3|来源到家族的映射聚合|未运行，reviewed=false|

主体100张20/30轮与200-q17930轮D0/A0、冻结OFA、D-DINO/A-RGB/A-FP/A-Fusion、来源分数融合及同ID规模比较此前已完成。主体封口summary SHA仍为{done['summary_sha256']}。有效63821=43821AI+20000真实；179新增冲突图已获批隔离，原64000文件和42000父成员保留，holdout未评分。

## 真假检测结果与局限

|evaluation|n（真实/AI）|AUROC|FPR@0.5|TPR@0.5|
|---|---:|---:|---:|---:|
|D1-control，clean|2310（979/1331）|99.6147%|3.1665%|96.2434%|
|D1增强，clean|同2310|99.5461%|5.2094%|98.1968%|
|D2，未见4o|1038（979/59）|99.6849%|4.8008%|98.3051%|
|D2-semantic，未见FFHQ|225（93/132）|88.6527%|60.2151%|98.4848%|

D1减同ID对照，原group配对bootstrap1000次：AUROC {dci(d1,'auroc')}；FPR {dci(d1,'fpr_0_5')}；TPR {dci(d1,'tpr_0_5')}。召回提升同时增加真实误报，不能称总体更好。没有增强检测器的JPEG/缩放匹配测试，不能宣称增强鲁棒性已验证。

FFHQ跨语义56/93真实被误判AI，Wilson95%约[50.05%,69.57%]，暴露明显泛化局限；不能用FFHQ evaluation调阈值/选模。4o仅59张正例，不能推广所有新来源。配对文件另列D2与同批in-domain对照ID的差异；训练域不同，不宣称纯因果或独立测试。

D3仅用calibration真实图校准：目标FPR1%在2310 evaluation实际FPR1.3279%、TPR91.4350%；5%目标对应4.6987%/98.4222%。目标不是评估保证；D3继承主D0按父完整dev选模的偏差。

## 来源分类、后处理与未知拒识

A2同1331张AI evaluation Top1：clean77.1600%、JPEG95 76.2585%、JPEG75 75.8828%、JPEG50 73.2532%、缩放0.75 76.7092%、缩放0.5 74.0045%。JPEG50减clean Top1 {dci(comparisons['A2_jpeg50_vs_clean'],'top1')}；缩放0.5减clean {dci(comparisons['A2_resize050_vs_clean'],'top1')}。Top3/MacroF1、逐来源差异与同ID原group1000次区间保存在配对结果，未混入calibration。

A2-semantic训练20986张AI，selection1190、calibration1384均不含FFHQ，选第26轮。132张未见FFHQ AI evaluation：Top1={pct(a2['top1'])}、Top3={pct(a2['top3'])}、MacroF1={pct(a2['macro_f1'])}。与包含FFHQ训练的A0在完全相同132ID上配对：Top1 {dci(sem,'top1')}；Top3 {dci(sem,'top3')}；MacroF1 {dci(sem,'macro_f1')}。这只验证一种语义，每来源3–11张；in-domain A0仍有完整dev选模偏差，不能推广全部跨语义能力。

A1仅64张kling未知calibration选阈值0.9986448884010316，cal误接收3/64=4.6875%。evaluation为1212已知＋63张grok3未知：未知AUROC={pct(a1['unknown_auroc'])}、拒识59/63={pct(a1['unknown_rejection_rate'])}、误接收4/63={pct(a1['unknown_false_acceptance'])}；已知保留576/1212={pct(a1['known_acceptance'])}，错误拒绝52.4752%，正确且保留539/1212=44.4719%。不能只报未知拒识率掩盖已知覆盖损失。20类A1不等于主22类A0/Fusion，阈值不能直接移植；1-maxsoftmax不是未知概率，仅kling→grok3设定，不保证任意未知来源，不部署生产。

## 技术骨架与剩余研究目标

RGB实际用冻结DINOv2-S/14；指纹是冻结DnCNN残差→FFT2幅度→log1p→作者OFA ResNet50编码器。RGB-only、Fingerprint-only、特征Fusion和与ResNet的分数融合均已运行；真假二分类、22类来源softmax、OFA配对分数分别报告。

同原1968张AI dev的100/30→200-q179/30 A0 Top1为67.2764%→77.1341%；同606AI evaluation，特征Fusion69.1419%低于ResNet75.2475%，校准分数融合75.9076%，不能宣布复杂融合稳定最优。需针对跨语义误报、拒识覆盖损失和外部泛化另定后续协议。

ConvNeXt、OpenCLIP/UniversalAttribution、OmniAID、OmniDFA、DNA-Det没有本批模型成绩；A3映射未核验。没有自动扩500、换骨干或混入GenImage。文献数值、接口和合成测试不能冒充项目成绩。

## 验收与证据

14项status/exit、manifest/config/role/model/source SHA，6项连续1..30轮、失败样本0、重载/单图批量门槛均通过。所有角色ID/group互斥，只来自train/dev，无holdout；排除域严格不入训练、selection和calibration。effective_training_rows按split分别核对train和作为dev的selection；predictions按role/condition筛选，用保存scores重算最终指标；A1阈值只由kling calibration重算，未用evaluation调参。

9项配对比较复用仓库stats/group_bootstrap/detection_bootstrap/source_rows，不执行旧硬编码main；固定seed42、1000次原group抽样、同ID及固定词表。区间条件于已有模型/阈值，不覆盖重训或选模不确定性；单seed开发结果，不宣称独立检验、SOTA或纯数据因果。类别不平衡/权重、旧100分配偏差及新组修正、物理GPU和OFA参考ID差异仍适用。

本目录保存{len(sources)}份完成产物副本、汇总、配对、provenance和完成封口；模型、图像、数据manifest、角色ID全集、逐图预测、特征缓存均不进Git，原资产保留服务器。只新增独立结果目录，不覆盖旧报告，不改sync生产树、文本服务或线上模型。队列已完成，不再启动新实验；归档验证与推送后删除本定时任务。

在研究根目录可用.venv-image/bin/python运行本目录analysis.py，并用--output指定新的独占目录复核；需要原服务器证据，只读CPU分析，不能覆盖现有目录。
"""
(out/"README.md").write_text(text)
outputs={str(p.relative_to(out)):sha256(p) for p in out.rglob("*") if p.is_file()}
provenance={"inputs_sha256":e.inputs,"outputs_sha256":outputs,"analysis_script_sha256":sha256(out/"analysis.py"),
"numpy_version":np.__version__,"original_files_overwritten":False,"holdout_scored":False,
"training_executed":False,"image_inference_executed":False,"queue_matching_process_alive":False}
write_json("provenance.json",provenance)
write_json("completion.json",{"status":"completed","exit_code":0,"scope":"registered_followups_only",
"registered_actions_verified":14,"training_runs_30_epochs_verified":6,"A3_not_run":True,
"config_sha256":sha256(cp),"manifest_sha256":config["manifest_sha256"],
"summary_sha256":sha256(out/"summary.json"),"provenance_sha256":sha256(out/"provenance.json"),
"holdout_scored":False,"training_executed_by_analysis":False,"image_inference_executed_by_analysis":False})
print(json.dumps({"output":args.output,"inputs_verified":len(e.inputs),"outputs_sealed":len(outputs),
"copied_files":len(index),"paired_comparisons":len(comparisons),"A2_semantic_top1":a2["top1"],
"A2_semantic_top3":a2["top3"],"A2_semantic_macro_f1":a2["macro_f1"],"summary_sha256":sha256(out/"summary.json")},ensure_ascii=False))
