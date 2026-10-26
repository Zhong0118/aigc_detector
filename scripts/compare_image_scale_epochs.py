#!/usr/bin/env python3
"""Read-only saved-dev comparison: 50/12, 50/20, 100/20; never infer or train.

Only numpy is required. Exit 0: all three verified; 2: pending inputs;
1: invalid evidence or existing output directory. No images/holdout scores/models
are opened. Cluster bootstrap follows the prior iab-v5-50-comparison/compare.py
algorithm, without importing or executing its hard-coded main().
"""
import argparse
from collections import Counter
import csv
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np


RUNS = {"D0": "d0-resnet50", "A0": "a0-resnet50", "OFA": "ofa"}
ANCHOR_SHA = "0dd207abf3fb477a89d8c767144a4869b425722415258a3e031ff2bedbf26553"
IMMUTABLE = ("split", "group_id", "sha256", "pixel_sha256", "source_label",
             "is_ai", "semantic_class", "prompt_id")
PROTOCOL_FIELDS = ("pretrained_sha256", "batch_size", "seed", "training_protocol",
                   "ofa_denoiser_microbatch")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def indexed(records):
    result = {}
    for record in records:
        sid = record.get("sample_id")
        require(isinstance(sid, str) and sid and sid not in result,
                "missing or duplicate sample_id")
        result[sid] = record
    return result


def verify_extension(old, new):
    require(set(old) <= set(new), "new manifest lost original sample IDs")
    for sid, before in old.items():
        for field in IMMUTABLE:
            require(field in before and field in new[sid], f"missing manifest field: {field}")
            require(before[field] == new[sid][field], f"manifest changed {sid}: {field}")
    return len(old)


def verify_cell_scale(old, new):
    """100/cell must double each anchored 50/cell source/semantic stratum."""
    def cells(rows):
        return Counter((r["source_label"], r["semantic_class"]) for r in rows.values() if r["is_ai"])
    before, after = cells(old), cells(new)
    require(set(before) == set(after) and all(after[key] == 2*count for key, count in before.items()),
            "100/cell manifest must double every original AI source/semantic cell")
    return len(before)


def dev_rows(manifest, task):
    return {sid: row for sid, row in manifest.items()
            if row["split"] == "dev" and (task == "D0" or row["is_ai"])}


def align(rows, old, new, task):
    """Require exactly these dev IDs in both inputs, then align by sample_id."""
    a, b = indexed(old), indexed(new)
    ids = sorted(rows)
    require(ids and set(a) == set(ids) == set(b), "missing or unexpected dev IDs")
    classes = a[ids[0]].get("classes")
    require(isinstance(classes, list) and len(classes) >= 2 and
            len(classes) == len(set(classes)), "invalid class vocabulary")
    if task == "D0":
        require(set(classes) == {"real", "ai"}, "D0 requires real/ai vocabulary")
    else:
        require(set(classes) == {r["source_label"] for r in rows.values()},
                "source vocabulary differs from manifest")
    field = "match_scores" if task == "OFA" else "scores"
    truth, pa, pb = [], [], []
    for sid in ids:
        row = rows[sid]
        require(row["split"] == "dev", "non-dev manifest row")
        label = ("ai" if row["is_ai"] else "real") if task == "D0" else row["source_label"]
        for prediction in (a[sid], b[sid]):
            require(prediction.get("split") == "dev" and prediction.get("status") == "ok",
                    "non-dev or failed prediction")
            require(prediction.get("true_label") == label, f"truth mismatch: {sid}")
            require(prediction.get("classes") == classes, "vocabulary mismatch")
            scores = np.asarray(prediction.get(field), dtype=float)
            require(scores.shape == (len(classes),) and np.isfinite(scores).all() and
                    ((scores >= 0) & (scores <= 1)).all(), "invalid scores")
            if task != "OFA":
                require(np.isclose(scores.sum(), 1, atol=1e-5), "invalid probability sum")
        truth.append(classes.index(label)); pa.append(a[sid][field]); pb.append(b[sid][field])
    return ids, np.asarray(truth), np.asarray(pa), np.asarray(pb), classes


def auc(truth, scores):
    """Mann–Whitney AUC with average tie ranks; both labels are required."""
    truth = np.asarray(truth, dtype=bool)
    positives = int(truth.sum()); negatives = len(truth) - positives
    require(positives and negatives, "AUC requires positive and negative samples")
    order = np.argsort(scores, kind="stable")
    sorted_scores = np.asarray(scores)[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    ends = np.r_[starts[1:], len(truth)]
    ranks = np.empty(len(truth), dtype=float)
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = (start + 1 + end) / 2
    return float((ranks[truth].sum() - positives * (positives + 1) / 2) /
                 (positives * negatives))


def confusion(y, scores):
    k = scores.shape[1]
    return np.bincount(k * y + scores.argmax(1), minlength=k*k).reshape(k, k)


def f1_from_cm(cm):
    denominator = cm.sum(0) + cm.sum(1)
    return np.divide(2*np.diag(cm), denominator, out=np.zeros(len(cm)), where=denominator > 0)


def top3_hits(y, scores):
    return (np.argsort(scores, axis=1)[:, -min(3, scores.shape[1]):] == y[:, None]).any(1)


def stats(y, scores, task, classes):
    if task == "OFA":
        return {"n": len(y), "macro_auroc": float(np.mean([
            auc(y == i, scores[:, i]) for i in range(len(classes))]))}
    cm = confusion(y, scores)
    result = dict(n=len(y), accuracy=float(np.trace(cm)/len(y)),
                  macro_f1=float(f1_from_cm(cm).mean()))
    if task == "D0":
        ai = classes.index("ai"); truth = y == ai; positive = scores[:, ai] >= .5
        result.update(auroc=auc(truth, scores[:, ai]),
                      fpr_0_5=float(positive[~truth].mean()),
                      tpr_0_5=float(positive[truth].mean()))
    else:
        result.update(top1=result["accuracy"], top3=float(top3_hits(y, scores).mean()))
    return result


def group_bootstrap(y, p, q, groups, repeats=1000, seed=42):
    """Same original-group draws for both predictions; fixed class vocabulary."""
    require(len(y) == len(p) == len(q) == len(groups) and len(y) > 0,
            "bootstrap row count mismatch")
    names = sorted(set(groups)); groups = np.asarray(groups)
    cms, sizes, hits = [], [], []
    for name in names:
        idx = np.flatnonzero(groups == name); sizes.append(len(idx))
        cms.append([confusion(y[idx], s[idx]) for s in (p, q)])
        hits.append([top3_hits(y[idx], s[idx]).sum() for s in (p, q)])
    cms, sizes, hits = np.asarray(cms), np.asarray(sizes), np.asarray(hits)
    rng = np.random.default_rng(seed); deltas = []
    for _ in range(repeats):
        weights = np.bincount(rng.integers(0, len(names), len(names)), minlength=len(names))
        n = weights @ sizes; pair = np.tensordot(weights, cms, axes=(0, 0))
        values = [[np.trace(cm)/n, (weights@hits[:, i])/n, f1_from_cm(cm).mean()]
                  for i, cm in enumerate(pair)]
        deltas.append(np.asarray(values[1]) - values[0])
    bounds = np.quantile(deltas, [.025, .975], axis=0)
    return dict(group_n=len(names), repeats=repeats, seed=seed,
                method="paired percentile cluster bootstrap; fixed class vocabulary",
                delta_95_ci={name: bounds[:, i].tolist()
                             for i, name in enumerate(("top1", "top3", "macro_f1"))})


class Evidence:
    def __init__(self):
        self.inputs = {}

    def read(self, path, lines=False):
        path = Path(path)
        self.inputs[str(path.resolve())] = sha256(path)
        text = path.read_text()
        return [json.loads(line) for line in text.splitlines() if line.strip()] if lines else json.loads(text)

    def unchanged(self):
        for path, digest in self.inputs.items():
            require(sha256(path) == digest, f"input changed during analysis: {path}")


def load_manifest(path, evidence):
    rows = indexed(evidence.read(path, lines=True)); groups = {}
    require(rows, "empty manifest")
    for sid, row in rows.items():
        require(all(field in row for field in IMMUTABLE), f"missing manifest metadata: {sid}")
        require(row["split"] in {"train", "dev", "holdout"}, "unexpected manifest split")
        require(row["is_ai"] in (0, 1) and row["group_id"] and row["sha256"] and
                row["pixel_sha256"], "invalid manifest identity")
        require(not row["is_ai"] or isinstance(row["source_label"], str), "missing AI source")
        group = row["group_id"]
        require(group not in groups or groups[group] == row["split"], "group crosses splits")
        groups[group] = row["split"]
    return rows


def load_run(folder, manifest, digest, task, epochs, evidence):
    required = ["status.json", "resolved_config.json", "metrics.json", "predictions.jsonl",
                "asset_manifest.json", "reference_manifest.json" if task == "OFA" else "history.json"]
    missing = [name for name in required if not (folder/name).is_file()]
    if missing:
        return {"status": "pending", "reason": "missing: " + ", ".join(missing), "path": str(folder)}
    status = evidence.read(folder/"status.json")
    if status.get("status") != "completed" or status.get("exit_code") != 0:
        state = "failed" if status.get("exit_code") not in (None, 0) or status.get("status") == "failed" else "pending"
        return {"status": state, "reason": f"run status: {status}", "path": str(folder)}
    config = evidence.read(folder/"resolved_config.json")
    metrics = evidence.read(folder/"metrics.json")
    assets = evidence.read(folder/"asset_manifest.json")
    require(config.get("manifest_sha256") == digest, f"manifest hash mismatch: {folder}")
    require(metrics.get("holdout_evaluated") is False, f"holdout evaluation flag invalid: {folder}")
    require(config.get("seed") == 42, f"seed differs from protocol: {folder}")
    require(all(field in config for field in PROTOCOL_FIELDS), f"missing protocol fields: {folder}")
    for filename in required:
        if filename in {"status.json", "asset_manifest.json"}:
            continue
        require(filename in assets and assets[filename].get("sha256") == sha256(folder/filename),
                f"asset hash mismatch: {folder/filename}")
    predictions = evidence.read(folder/"predictions.jsonl", lines=True)
    selected = dev_rows(manifest, task)
    ids, y, p, _, classes = align(selected, predictions, predictions, task)
    computed = stats(y, p, task, classes)
    for field, value in computed.items():
        if field in metrics:
            require(np.isclose(value, metrics[field], atol=1e-8, rtol=0),
                    f"saved metric disagrees with predictions: {folder}: {field}")
    if "classes" in metrics:
        require(metrics["classes"] == classes, "metrics vocabulary mismatch")
    history = []
    reference_counts = None
    if task == "OFA":
        references = evidence.read(folder/"reference_manifest.json")
        require(set(references) == set(classes), "reference vocabulary mismatch")
        seen = set()
        for source, samples in references.items():
            require(samples and len(samples) == len(set(samples)), "empty/duplicate references")
            for sid in samples:
                require(sid in manifest and sid not in seen and manifest[sid]["split"] == "train"
                        and manifest[sid]["source_label"] == source, "reference leakage or wrong label")
                seen.add(sid)
        reference_counts = {source: len(samples) for source, samples in references.items()}
    else:
        require(config.get("epochs") == epochs, f"unexpected training epochs: {folder}")
        history = evidence.read(folder/"history.json")
        require([entry.get("epoch") for entry in history] == list(range(1, epochs+1)),
                f"incomplete epoch history: {folder}")
        criterion = "auroc" if task == "D0" else "macro_f1"
        require(all(np.isfinite(entry["dev"][criterion]) for entry in history), "invalid history metric")
        best = max(history, key=lambda entry: entry["dev"][criterion])["epoch"]
        require(metrics.get("selected_epoch") == best, f"selected epoch mismatch: {folder}")
        require(all(entry["dev"]["n"] == len(ids) for entry in history), "history dev size mismatch")
    return dict(status="verified", path=str(folder), config=config, selected_epoch=metrics.get("selected_epoch"),
                history=history, predictions=predictions, metrics=computed, classes=classes,
                reference_counts=reference_counts, elapsed_seconds=status.get("elapsed_seconds"),
                declared_model_assets={k: v for k, v in assets.items() if k.endswith((".pth", ".pt"))})


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path, records):
    fields = list(dict.fromkeys(key for record in records for key in record))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(records)


def render(summary):
    lines = ["# 图像实验数据规模与训练轮数比较", "", f"状态：{summary['status']}", "",
             "同 dev 使用原 v5 sample_id 对齐；完整 dev 单列。OFA 是冻结模型参考匹配，条件中的轮数不表示 OFA 训练。", "",
             "|条件|任务|范围|样本数|AUROC / 验证宏 AUC|Accuracy / Top1|Top3|Macro-F1|FPR@0.5|TPR@0.5|",
             "|---|---|---|---|---|---|---|---|---|---|"]
    def fmt(value):
        return "—" if value is None else f"{100*value:.2f}%"
    for scope in ("same_dev", "full_dev"):
        for condition, tasks in summary[scope].items():
            for task, values in tasks.items():
                fields = [values.get("macro_auroc", values.get("auroc")), values.get("top1", values.get("accuracy")),
                          values.get("top3"), values.get("macro_f1"), values.get("fpr_0_5"), values.get("tpr_0_5")]
                lines.append("|"+"|".join([condition, task, scope, str(values["n"])] + [fmt(v) for v in fields])+"|")
    lines += ["", "## A0 原分组配对 bootstrap", "", "差值为后条件减前条件；seed42、1000次、95%百分位区间。", ""]
    for pair, tasks in summary["paired"].items():
        if "A0" in tasks:
            item = tasks["A0"]
            for metric, interval in item["paired_group_bootstrap"]["delta_95_ci"].items():
                lines.append(f"- {pair} {metric}：{100*item['delta'][metric]:+.2f} pp；95% CI [{100*interval[0]:+.2f}, {100*interval[1]:+.2f}] pp")
    lines += ["", "## 未完成与边界", ""]
    for condition, details in summary["conditions"].items():
        if details["status"] != "verified":
            lines.append(f"- {condition}：{details['status']}；{details.get('reason', '见任务状态')}")
            for task, result in details.get("runs", {}).items():
                if result["status"] != "verified":
                    lines.append(f"- {condition}/{task}：{result['status']}；{result.get('reason')}")
    lines += ["- 单一种子；dev参与checkpoint选择，本结果为探索性比较，未评价holdout。",
              "- 每epoch曲线来自各条件自己的dev；只有已保存最终checkpoint可以在同一dev上配对。",
              "- 数据增量与优化步数可能同时变化；本比较不能单独归因于数据多样性。",
              "- 模型文件只记录asset manifest声明的hash，本脚本不读取二进制模型，不重训、不推理。",
              "- per-source.csv提供A0逐来源差异；OFA来源验证宏AUC独立报告，不作分类概率解释。"]
    return "\n".join(lines)+"\n"


def analyze(args, evidence):
    base = args.base.resolve()
    paths = {"50/12": args.manifest_50 or base/"iab-balanced-v5-50/manifest.jsonl",
             "50/20": args.manifest_50 or base/"iab-balanced-v5-50/manifest.jsonl",
             "100/20": args.manifest_100 or base/"iab-balanced-v6-100/manifest.jsonl"}
    suffixes = {"50/12": args.suffix_50_12, "50/20": args.suffix_50_20, "100/20": args.suffix_100_20}
    require(paths["50/12"].is_file(), "anchor manifest missing")
    require(sha256(paths["50/12"]) == args.expected_anchor_sha, "anchor manifest SHA mismatch")
    anchor = load_manifest(paths["50/12"], evidence)
    require(len(dev_rows(anchor, "D0")) == args.expected_d0 and
            len(dev_rows(anchor, "A0")) == args.expected_attribution, "anchor dev count mismatch")
    vocabulary = sorted({row["source_label"] for row in anchor.values() if row["is_ai"]})
    require(len(vocabulary) == args.expected_sources, "anchor source count mismatch")
    summary = dict(status="pending", holdout_evaluated=False, inference_executed=False,
                   bootstrap=dict(seed=42, repeats=1000), anchor_manifest_sha256=args.expected_anchor_sha,
                   anchor_dev_counts=dict(D0=args.expected_d0, A0=args.expected_attribution, OFA=args.expected_attribution),
                   conditions={}, same_dev={}, full_dev={}, paired={})
    available, epochs_rows, source_rows = {}, [], []
    for condition, path in paths.items():
        if not path.is_file():
            summary["conditions"][condition] = dict(status="pending", reason=f"manifest missing: {path}")
            continue
        manifest = load_manifest(path, evidence); digest = sha256(path)
        preserved = verify_extension(anchor, manifest)
        require(sorted({row["source_label"] for row in manifest.values() if row["is_ai"]}) == vocabulary,
                "manifest source vocabulary changed")
        if condition == "50/20":
            require(digest == args.expected_anchor_sha, "50/20 must use original v5 manifest")
        if condition == "100/20":
            verify_cell_scale(anchor, manifest)
        info = dict(status="verified", manifest=str(path), manifest_sha256=digest, old_rows_preserved=preserved,
                    manifest_rows=len(manifest), split_counts=dict(Counter(r["split"] for r in manifest.values())), runs={})
        summary["conditions"][condition] = info
        summary["same_dev"][condition] = {}; summary["full_dev"][condition] = {}
        for task, prefix in RUNS.items():
            result = load_run(base/f"{prefix}-iab-{suffixes[condition]}", manifest, digest, task,
                              12 if condition == "50/12" else 20, evidence)
            info["runs"][task] = {k: v for k, v in result.items() if k not in {"predictions", "history", "config"}}
            if result["status"] != "verified":
                info["status"] = "failed" if result["status"] == "failed" or info["status"] == "failed" else "pending"
                continue
            fixed = dev_rows(anchor, task)
            predictions = [r for r in result["predictions"] if r["sample_id"] in fixed]
            ids, y, p, _, classes = align(fixed, predictions, predictions, task)
            summary["same_dev"][condition][task] = stats(y, p, task, classes)
            summary["full_dev"][condition][task] = result["metrics"]
            available[condition, task] = (ids, y, p, classes, result)
            for entry in result["history"]:
                record = dict(condition=condition, task=task, epoch=entry["epoch"],
                              selected=entry["epoch"] == result["selected_epoch"],
                              lr=entry.get("lr"), seconds=entry.get("seconds"), dev_scope="own_full_dev")
                for partition in ("train", "dev"):
                    for field in ("n", "loss", "accuracy", "macro_f1", "auroc", "top1", "top3", "fpr_0_5", "tpr_0_5"):
                        record[f"{partition}_{field}"] = entry.get(partition, {}).get(field)
                epochs_rows.append(record)
    for before, after in combinations(paths, 2):
        pair = {}
        for task in RUNS:
            if (before, task) not in available or (after, task) not in available:
                continue
            ids, y, p, classes, old = available[before, task]
            ids2, y2, q, classes2, new = available[after, task]
            require(ids == ids2 and classes == classes2 and np.array_equal(y, y2), "paired identity mismatch")
            for field in PROTOCOL_FIELDS:
                require(old["config"][field] == new["config"][field], f"protocol mismatch: {task} {field}")
            a = summary["same_dev"][before][task]; b = summary["same_dev"][after][task]
            item = dict(n=len(ids), before=a, after=b, delta={k:b[k]-v for k,v in a.items() if k != "n"})
            if task == "A0":
                item["paired_group_bootstrap"] = group_bootstrap(y, p, q, [anchor[sid]["group_id"] for sid in ids])
                cm_a, cm_b = confusion(y, p), confusion(y, q)
                for i, name in enumerate(classes):
                    ix = y == i
                    row = dict(comparison=f"{before}→{after}", source=name, n=int(ix.sum()))
                    for metric, va, vb in [("top1", float((p[ix].argmax(1)==i).mean()), float((q[ix].argmax(1)==i).mean())),
                                           ("top3", float(top3_hits(y[ix], p[ix]).mean()), float(top3_hits(y[ix], q[ix]).mean())),
                                           ("f1", float(f1_from_cm(cm_a)[i]), float(f1_from_cm(cm_b)[i]))]:
                        row.update({f"before_{metric}":va, f"after_{metric}":vb, f"delta_{metric}":vb-va})
                    source_rows.append(row)
            pair[task] = item
        if pair:
            summary["paired"][f"{before}→{after}"] = pair
    statuses = [value["status"] for value in summary["conditions"].values()]
    summary["status"] = "failed" if "failed" in statuses else "completed" if all(s == "verified" for s in statuses) else "pending"
    return summary, epochs_rows, source_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("outputs/image_experiments"))
    parser.add_argument("--output", type=Path, required=True, help="new directory; existing directories refused")
    parser.add_argument("--manifest-50", type=Path)
    parser.add_argument("--manifest-100", type=Path)
    parser.add_argument("--suffix-50-12", default="v5-50")
    parser.add_argument("--suffix-50-20", default="v5-50-e20")
    parser.add_argument("--suffix-100-20", default="v6-100-e20")
    parser.add_argument("--expected-anchor-sha", default=ANCHOR_SHA)
    parser.add_argument("--expected-d0", type=int, default=3140)
    parser.add_argument("--expected-attribution", type=int, default=1496)
    parser.add_argument("--expected-sources", type=int, default=22)
    args = parser.parse_args(argv)
    evidence = Evidence()
    try:
        args.output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"Refusing existing output directory: {args.output}", file=sys.stderr)
        return 1
    try:
        summary, epochs_rows, source_rows = analyze(args, evidence)
        evidence.unchanged()
        write_json(args.output/"summary.json", summary)
        write_csv(args.output/"epochs.csv", epochs_rows)
        write_csv(args.output/"per-source.csv", source_rows)
        (args.output/"report.md").write_text(render(summary))
        write_json(args.output/"provenance.json", dict(
            inputs=evidence.inputs, analysis_script_sha256=sha256(Path(__file__)),
            numpy_version=np.__version__, model_hashes_verified_from_binary=False,
            outputs={p.name:sha256(p) for p in args.output.iterdir() if p.is_file()}))
        if summary["status"] == "completed":
            write_json(args.output/"completion.json", dict(status="completed", holdout_evaluated=False,
                       same_dev_ids_verified=True, summary_sha256=sha256(args.output/"summary.json")))
        print(json.dumps({"status":summary["status"], "output":str(args.output)}, ensure_ascii=False))
        return 0 if summary["status"] == "completed" else 2 if summary["status"] == "pending" else 1
    except (ValueError, KeyError, TypeError, OSError) as error:
        write_json(args.output/"summary.json", dict(status="invalid", error=str(error), holdout_evaluated=False))
        write_json(args.output/"provenance.json", dict(inputs=evidence.inputs, analysis_script_sha256=sha256(Path(__file__))))
        print(f"Evidence rejected: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
