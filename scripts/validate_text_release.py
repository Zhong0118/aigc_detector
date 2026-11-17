"""Calibrate and validate a trained Chinese detector through the product decision policy."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from detection.false_positive_controller import FalsePositiveController
from detection.text_resources import conformal_high_threshold
from detection.text_threshold_artifact import validate_text_threshold_artifact
from scripts.bootstrap_text_resources import _build_detector, _group_artifact
from scripts.evaluate_text_detector import binary_metrics
from scripts.finetune_text_detector import load_dataset_rows


def build_calibration_artifact(scored, protocol, version):
    router = FalsePositiveController()
    groups = {}
    for language in ("zh", "mixed"):
        values = [r for r in scored if (router._language(r["text"]) if "text" in r else "zh") == language]
        calibration = [r["result"]["score"] for r in values if r["split"] == "calibration" and r["label"] == 0]
        if len(calibration) < 50:
            continue
        high = conformal_high_threshold(calibration, alpha=0.02)
        if not 0 < high < 1:
            raise ValueError(f"No useful independently calibrated cutoff for {language}: {high}")
        test_human = [r["result"]["score"] for r in values if r["split"] == "test" and r["label"] == 0]
        test_ai = [r["result"]["score"] for r in values if r["split"] == "test" and r["label"] == 1]
        groups[language] = _group_artifact(human_calibration=calibration, human_validation=test_human,
            ai_validation=test_ai, threshold=high, threshold_floor=0.0, low_threshold=min(0.4, high / 2), alpha=0.02,
            minimum_validation_samples=200, minimum_ai_validation_samples=200, minimum_ai_recall=0.7)
    if "zh" not in groups:
        raise ValueError("Insufficient Chinese calibration documents")
    return dict(artifact_schema_version=2, dataset_profile="binary-chinese-v1",
        version=version, scoring_protocol=protocol, metric_partition="test",
        calibration_kind="human_only_document_holdout", decision_operator="score > high_threshold", target_fpr=0.02,
        groups=groups)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regression", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    rows = [r for r in load_dataset_rows(args.data) if r["split"] != "train"]
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    local = cfg["detection"]["local_models"]["text_yuchuan"]
    local.update(device=args.device, mixed_strategy="dominant")
    for key in ("zh_long", "zh_short"):
        local["models"][key] = str(args.model.resolve())
    detector = _build_detector(cfg)
    torch.set_num_threads(2)
    scored = []
    with (args.output / "scores.jsonl").open("w", encoding="utf-8") as handle:
        for i, row in enumerate(rows):
            result = detector.detect(row["text"])
            item = dict(row, result=dict(score=result.score, details=result.details))
            scored.append(item)
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            if i % 100 == 0:
                handle.flush()
                print(f"scored {i + 1}/{len(rows)}", flush=True)
    artifact = build_calibration_artifact(scored, detector.scoring_protocol, args.model.name)
    high = artifact["groups"]["zh"]["high_threshold"]
    valid = validate_text_threshold_artifact(artifact, mode="readiness", expected_dataset_profile="binary-chinese-v1")
    if not valid.valid:
        raise ValueError(valid.errors)
    artifact_path = args.output.resolve() / "thresholds.json"
    artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    fp_config = cfg["detection"]["text_false_positive_control"]
    fp_config["artifact_path"] = str(artifact_path)
    controller = FalsePositiveController(fp_config)
    groups = defaultdict(list)
    decisions = []
    for row in scored:
        if row["split"] != "test":
            continue
        decision = controller.evaluate(row["text"], [SimpleNamespace(status="ok", **row["result"])])
        item = dict(source=row["source"], label=row["label"], identity=row["identity"],
                    generator=row.get("generator"), heldout_generator=row.get("heldout_generator", False),
                    chars=len(row["text"]), score=row["result"]["score"], decision=decision.to_dict())
        decisions.append(item)
        groups["all"].append(item)
        groups[row["source"]].append(item)
        if item["heldout_generator"]:
            groups["heldout_modern_model_and_topic"].append(item)
            if row.get("family") == "claude":
                groups["unseen_claude_family_and_topic"].append(item)
        if item["chars"] >= 1500:
            groups["long_documents"].append(item)
    metrics = {}
    for name, values in groups.items():
        labels = ["ai" if r["label"] else "human" for r in values]
        scores = [r["score"] for r in values]
        positive = [1.0 if r["decision"]["decision"] == "high_risk_ai" else 0.0 for r in values]
        metrics[name] = dict(raw=binary_metrics(labels, scores, high),
                             product=binary_metrics(labels, positive, 0.5),
                             decisions=dict(Counter(r["decision"]["decision"] for r in values)))
    text = json.loads(args.regression.read_text(encoding="utf-8"))["text"]
    result = detector.detect(text)
    regression = controller.evaluate(text, [SimpleNamespace(status="ok", score=result.score, details=result.details)]).to_dict()
    summary = dict(model=str(args.model), protocol=detector.scoring_protocol, high_threshold=high,
                   calibration_samples=artifact["groups"]["zh"]["human_calibration_samples"], independent_test_group=artifact["groups"]["zh"],
                   test=metrics, regression=regression)
    (args.output / "test-decisions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in decisions), encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "regression"}, ensure_ascii=False, indent=2), flush=True)
    print("REGRESSION", result.score, regression["decision"], flush=True)


if __name__ == "__main__":
    main()
