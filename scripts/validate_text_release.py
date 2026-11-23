"""Calibrate and validate a trained Chinese detector through the product decision policy."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unicodedata

import numpy as np
from scipy.optimize import minimize_scalar
import torch
import yaml

from detection.false_positive_controller import FalsePositiveController
from detection.text_detector import branch_scoring_protocol, margin_probability
from detection.text_resources import conformal_high_threshold
from detection.text_threshold_artifact import model_calibration_group, validate_text_threshold_artifact
from scripts.bootstrap_text_resources import _build_detector, _group_artifact
from scripts.evaluate_text_detector import binary_metrics
from scripts.finetune_text_detector import load_dataset_rows
from scripts.prepare_text_detection_v2 import fingerprints


def fit_temperature(validation_rows):
    """Fit one temperature on document margins from the validation partition."""
    if not validation_rows or any(row.get("split") != "validation" for row in validation_rows):
        raise ValueError("Temperature fitting requires validation rows only")
    labels = np.array([row["label"] for row in validation_rows], dtype=np.float64)
    margins = np.array([row["result"]["details"]["raw_margin"] for row in validation_rows], dtype=np.float64)
    if set(labels) != {0, 1} or not np.isfinite(margins).all():
        raise ValueError("Temperature fitting requires finite validation margins and both labels")
    signed_margins = (2 * labels - 1) * margins

    def nll(log_temperature):
        return float(np.logaddexp(0, -signed_margins / np.exp(log_temperature)).mean())

    bounds = np.log([0.05, 100.0])
    fitted = minimize_scalar(nll, method="bounded", bounds=bounds, options={"xatol": 1e-9})
    if not fitted.success:
        raise RuntimeError(f"Temperature fitting failed: {fitted.message}")
    return dict(temperature=float(np.exp(fitted.x)), partition="validation", samples=len(labels),
                human=int((labels == 0).sum()), ai=int((labels == 1).sum()),
                validation_nll_before=nll(0), validation_nll_after=float(fitted.fun),
                temperature_bounds=[0.05, 100.0])


def fit_screening_threshold(validation_rows, target_recall):
    """Fit one inclusive screening cutoff with recall constraints on all and long AI."""
    if not validation_rows or any(row["split"] != "validation" for row in validation_rows):
        raise ValueError("Screening threshold fitting requires validation rows only")
    if not 0 < target_recall <= 1:
        raise ValueError("Target screening recall must be in (0, 1]")
    ai = [row for row in validation_rows if row["label"] == 1]
    partitions = {}
    for name, rows in (("all_ai", ai), ("long_ai", [row for row in ai if _effective_chars(row["text"]) >= 1500])):
        scores = sorted(row["result"]["score"] for row in rows)
        if not scores or any(not 0 <= score <= 1 for score in scores):
            raise ValueError(f"Screening validation requires finite scores for {name}")
        partitions[name] = dict(samples=len(scores), cutoff=scores[len(scores) - math.ceil(target_recall * len(scores))])
    return dict(partition="validation", target_recall=target_recall, decision_operator="score >= threshold",
                threshold=min(p["cutoff"] for p in partitions.values()), partitions=partitions)


def screening_positive(decision, score, threshold):
    return decision == "high_risk_ai" or (decision != "insufficient_evidence" and score >= threshold)


def _effective_chars(text):
    return sum(not char.isspace() for char in text)


def _length_band(chars):
    if chars < 500:
        return "<500"
    if chars < 1500:
        return "500–1499"
    return "1500–2499" if chars < 2500 else "≥2500"


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
        groups[language]["high_threshold"] = high
    if "zh" not in groups:
        raise ValueError("Insufficient Chinese calibration documents")
    return dict(artifact_schema_version=2, dataset_profile="binary-chinese-v1",
        version=version, scoring_protocol=protocol, metric_partition="test",
        calibration_kind="human_only_document_holdout", decision_operator="score > high_threshold", target_fpr=0.02,
        groups=groups)


def _branch_rows(rows, model_routes):
    groups = defaultdict(list)
    for row in rows:
        group = model_calibration_group(row["result"]["details"], model_routes)
        if group is None:
            raise ValueError("Missing or unknown executed model route")
        groups[group].append(row)
    return groups


def _model_routes(models):
    by_weight = defaultdict(list)
    for name, weight in models.items():
        by_weight[weight].append(name)
    routes = {}
    for names in by_weight.values():
        family = min(names).split("_")[0]
        alias = family if set(names) == {f"{family}_short", f"{family}_long"} else min(names)
        routes.update(dict.fromkeys(names, alias))
    return routes


def fit_branch_temperatures(rows, model_routes):
    groups = {}
    skipped = {}
    for name, values in _branch_rows(rows, model_routes).items():
        validation = [r for r in values if r["split"] == "validation"]
        calibration_count = sum(r["split"] == "calibration" and r["label"] == 0 for r in values)
        counts = Counter(r["label"] for r in validation)
        if "+" in name or calibration_count < 50 or set(counts) != {0, 1}:
            skipped[name] = dict(human_calibration_samples=calibration_count,
                                 validation_human=counts[0], validation_ai=counts[1],
                                 reason="multiple_weights" if "+" in name else "insufficient_calibration_or_validation")
            continue
        groups[name] = fit_temperature(validation)
        groups[name]["validation_identity_sha256"] = hashlib.sha256(
            "\n".join(sorted(row["identity"] for row in validation if "identity" in row)).encode()
        ).hexdigest()
    return dict(groups=groups, uncalibrated=skipped)


def fit_branch_screening(scores, *, high_threshold, target_fpr):
    if not scores or not 0 < target_fpr < 1 or not all(0 <= s <= 1 for s in scores):
        raise ValueError("Screening requires human calibration scores and a valid FPR budget")
    # The inclusive comparison must skip every tie at the first excluded score.
    budget = math.floor(len(scores) * target_fpr)
    threshold = math.nextafter(sorted(scores, reverse=True)[budget], math.inf)
    if not threshold < high_threshold:
        raise ValueError("Human screening budget is infeasible below the strict threshold")
    fp = sum(score >= threshold for score in scores)
    return dict(threshold=threshold, high_threshold=high_threshold, partition="calibration",
                target_fpr=target_fpr, human_samples=len(scores), human_flagged=fp,
                observed_fpr=fp / len(scores), decision_operator="score >= threshold")


def build_branch_artifact(rows, protocol, version, model_routes, policy, *, eligible=None, target_fpr_by_group=None):
    if policy not in {"pooled", "source_max"}:
        raise ValueError(f"Unknown calibration policy: {policy}")
    groups = {}
    for name, values in _branch_rows(rows, model_routes).items():
        by_source = defaultdict(list)
        for row in values:
            if row["split"] == "calibration" and row["label"] == 0:
                by_source[row["source"]].append(row["result"]["score"])
        calibration = [score for scores in by_source.values() for score in scores]
        if len(calibration) < 50 or "+" in name or (eligible is not None and name not in eligible):
            continue
        alpha = (target_fpr_by_group or {}).get(name, 0.02)
        pooled = conformal_high_threshold(calibration, alpha=alpha)
        sources = {source: dict(samples=len(scores), cutoff=conformal_high_threshold(scores, alpha=alpha)
                               if len(scores) >= max(50, math.ceil(1 / alpha) - 1) else None)
                   for source, scores in sorted(by_source.items())}
        high = max([pooled, *[s["cutoff"] for s in sources.values() if s["cutoff"] is not None]]) if policy == "source_max" else pooled
        if not 0 < high < 1:
            raise ValueError(f"No usable independently calibrated cutoff for {name}: {high}")
        test = [r for r in values if r["split"] == "test"]
        group = _group_artifact(
            human_calibration=calibration, human_validation=[r["result"]["score"] for r in test if r["label"] == 0],
            ai_validation=[r["result"]["score"] for r in test if r["label"] == 1], threshold=high,
            threshold_floor=0.0, low_threshold=min(0.4, high / 2), alpha=alpha,
            minimum_validation_samples=200, minimum_ai_validation_samples=200, minimum_ai_recall=0.7,
        )
        groups[name] = dict(group, high_threshold=high, pooled_high_threshold=pooled, source_calibration=sources)
    if not groups:
        raise ValueError("No model branch has sufficient calibration evidence")
    return dict(artifact_schema_version=2, dataset_profile="binary-chinese-v1", version=version,
                scoring_protocol=protocol, model_routes=model_routes, metric_partition="test",
                calibration_kind="human_only_document_holdout", threshold_policy=policy,
                decision_operator="score > high_threshold", target_fpr=0.02, groups=groups)


def _apply_branch_temperatures(result, model_routes, temperatures, protocol):
    details = result["details"]
    branch = model_calibration_group(details, model_routes)
    temperature = temperatures.get(branch, {}).get("temperature", 1.0)
    return dict(score=margin_probability(details["raw_margin"], temperature), details=dict(
        details, temperature=temperature, scoring_protocol=protocol,
        segments=[dict(s, score=margin_probability(s["raw_margin"], temperature), model_scores={
            name: margin_probability(math.log(p) - math.log1p(-p), temperature)
            for name, p in s["model_scores"].items()}) for s in details["segments"]],
    ))


def write_branch_evaluations(scored, detector, cfg, model_path, output, regression_text, raw_regression,
                             policies=("pooled", "source_max")):
    routes = _model_routes(detector.models)
    fits = fit_branch_temperatures(scored, routes)
    temperatures = fits["groups"]
    protocol = branch_scoring_protocol(detector.scoring_protocol, routes,
                                       {group: fit["temperature"] for group, fit in temperatures.items()})
    fits.update(scoring_protocol=protocol, raw_scoring_protocol=detector.scoring_protocol, model_routes=routes)
    (output / "temperature.json").write_text(json.dumps(fits, indent=2) + "\n", encoding="utf-8")
    scored = [dict(row, result=_apply_branch_temperatures(row["result"], routes, temperatures, protocol)) for row in scored]
    regression = _apply_branch_temperatures(raw_regression, routes, temperatures, protocol)
    grouped = _branch_rows(scored, routes)
    for policy in policies:
        run = output / policy
        run.mkdir()
        artifact = build_branch_artifact(scored, protocol, model_path.name, routes, policy, eligible=temperatures)
        screening = {}
        for name, group in artifact["groups"].items():
            calibration = [r["result"]["score"] for r in grouped[name] if r["split"] == "calibration" and r["label"] == 0]
            screening[name] = fit_branch_screening(calibration, high_threshold=group["high_threshold"], target_fpr=0.05)
            group["screening_threshold"] = screening[name]["threshold"]
        (run / "screening.json").write_text(json.dumps(screening, indent=2) + "\n", encoding="utf-8")
        write_release_evaluation(scored, detector, cfg, model_path, run, regression_text, fits, screening,
                                 artifact_override=artifact, regression_result=regression)


def _text_digest(text):
    normalized = " ".join(unicodedata.normalize("NFKC", text).split()).casefold()
    return hashlib.sha256(normalized.encode()).hexdigest()


def _external_rows(specs):
    for spec in specs:
        name, separator, path = spec.partition("=")
        if not name or not separator or not path:
            raise ValueError("External datasets must use NAME=PATH")
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("split", "test") != "test":
                    continue
                if type(row.get("label")) is not int or row["label"] not in (0, 1) or not row.get("text", "").strip():
                    raise ValueError(f"Invalid external sample in {name}")
                yield dict(row, split="external", external=name, identity=row["id"])


def _score_stats(scores, labels):
    p = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels) == "ai"
    actual = np.where(y, p, 1 - p)
    quantiles = dict(zip(("p50", "p95", "p98", "p99", "max"),
                         np.quantile(p, [0.5, 0.95, 0.98, 0.99, 1]).tolist()))
    return dict(quantiles=quantiles, unique_scores=len(set(scores)),
                at_or_above_09995=int((p >= 0.9995).sum()),
                nll=float(-np.log(np.maximum(actual, np.finfo(float).tiny)).mean()),
                brier=float(np.mean((p - y) ** 2)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--zh-short-model", type=Path, help="Independent snapshot for the Chinese short-text branch")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regression", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--aggregation", choices=("probability_mean", "logit_mean"))
    parser.add_argument("--fit-temperature", action="store_true")
    parser.add_argument("--screen-validation-recall", type=float)
    parser.add_argument("--branch-calibration", action="store_true",
                        help="Fit separate executed-weight temperatures and compare two human-only calibrations")
    parser.add_argument("--calibration-policy", choices=("pooled", "source_max"),
                        help="Evaluate one policy instead of both branch calibration policies")
    parser.add_argument("--raw-evaluation", type=Path,
                        help="Replay scores from a complete T=1 evaluation of the identical rows and protocol")
    parser.add_argument("--external", action="append", default=[], metavar="NAME=PATH")
    args = parser.parse_args(argv)
    if (args.raw_evaluation or args.zh_short_model or args.calibration_policy) and not args.branch_calibration:
        parser.error("--raw-evaluation, --zh-short-model and --calibration-policy require --branch-calibration")
    if args.zh_short_model is not None and not args.zh_short_model.is_dir():
        parser.error("--zh-short-model must be a local snapshot directory")
    if args.branch_calibration and (args.aggregation != "logit_mean" or args.screen_validation_recall is not None):
        parser.error("--branch-calibration requires --aggregation logit_mean and uses a human screening budget")
    dataset = load_dataset_rows(args.data)
    rows = [r for r in dataset if r["split"] != "train"]
    external = list(_external_rows(args.external))
    regression_text = json.loads(args.regression.read_text(encoding="utf-8"))["text"]
    validation = [row for row in rows if row["split"] == "validation"]
    if args.screen_validation_recall is not None:
        if not args.fit_temperature or not 0 < args.screen_validation_recall <= 1:
            parser.error("--screen-validation-recall requires --fit-temperature and a recall in (0, 1]")
    if args.screen_validation_recall is not None or args.branch_calibration:
        protected = [fingerprints(row["text"]) for row in [*external, *[r for r in rows if r["split"] == "test"]]]
        protected.append(fingerprints(regression_text))
        protected_full, protected_leads = map(set, zip(*protected))
        for row in dataset:
            if row["split"] != "test":
                full, lead = fingerprints(row["text"])
                if full in protected_full or lead in protected_leads:
                    raise ValueError(f"Protected full text or prefix found in {row['split']}")
    if args.fit_temperature:
        validation_hashes = {_text_digest(row["text"]) for row in validation}
        if _text_digest(regression_text) in validation_hashes or any(
            _text_digest(row["text"]) in validation_hashes for row in external
        ):
            raise ValueError("Temperature validation overlaps an external or regression sample")
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    local = cfg["detection"]["local_models"]["text_yuchuan"]
    local.update(device=args.device, mixed_strategy="dominant")
    local["models"].update(zh_long=str(args.model.resolve()),
                           zh_short=str((args.zh_short_model or args.model).resolve()))
    if args.aggregation:
        local["aggregation_mode"] = args.aggregation
    torch.set_num_threads(2)
    if args.branch_calibration:
        local["temperature"] = 1.0
        detector = _build_detector(cfg)
        if args.raw_evaluation:
            scored = _load_raw_evaluation(args.raw_evaluation, [*rows, *external], detector.scoring_protocol)
        else:
            scored = _score_rows(detector, [*rows, *external], args.output)
        result = detector.detect(regression_text)
        raw_regression = dict(score=result.score, details=result.details)
        policies = (args.calibration_policy,) if args.calibration_policy else ("pooled", "source_max")
        write_branch_evaluations(scored, detector, cfg, args.model, args.output, regression_text, raw_regression, policies)
        return
    temperature_fit = None
    screening_fit = None
    if args.fit_temperature:
        if local.get("aggregation_mode") != "logit_mean":
            parser.error("--fit-temperature requires logit_mean aggregation")
        local["temperature"] = 1.0
        unscaled = _build_detector(cfg)
        validation_margins = [dict(row, result={"details": {"raw_margin": unscaled.detect(row["text"]).details["raw_margin"]}})
                              for row in validation]
        temperature_fit = fit_temperature(validation_margins)
        temperature_fit["validation_identity_sha256"] = hashlib.sha256(
            "\n".join(sorted(row["identity"] for row in validation)).encode()
        ).hexdigest()
        local["temperature"] = temperature_fit["temperature"]
        (args.output / "temperature.json").write_text(json.dumps(temperature_fit, indent=2) + "\n", encoding="utf-8")
        if args.screen_validation_recall is not None:
            for row in validation_margins:
                row["result"]["score"] = margin_probability(row["result"]["details"]["raw_margin"], local["temperature"])
            screening_fit = fit_screening_threshold(validation_margins, args.screen_validation_recall)
            screening_fit["validation_identity_sha256"] = temperature_fit["validation_identity_sha256"]
            (args.output / "screening.json").write_text(json.dumps(screening_fit, indent=2) + "\n", encoding="utf-8")
        del unscaled
    detector = _build_detector(cfg)
    scored = _score_rows(detector, [*rows, *external], args.output)
    write_release_evaluation(
        scored, detector, cfg, args.model, args.output, regression_text, temperature_fit, screening_fit,
    )


def _score_rows(detector, rows, output):
    scored = []
    with (output / "scores.jsonl").open("w", encoding="utf-8") as handle:
        for i, row in enumerate(rows):
            result = detector.detect(row["text"])
            item = dict(row, result=dict(score=result.score, details=result.details))
            scored.append(item)
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            if i % 100 == 0:
                handle.flush()
                print(f"scored {i + 1}/{len(rows)}", flush=True)
    return scored


def _load_raw_evaluation(path, rows, protocol):
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    if summary["protocol"] != protocol or summary["temperature"] != 1.0:
        raise ValueError("Raw evaluation has a different scoring protocol or temperature")
    scored = []
    with (path / "scores.jsonl").open(encoding="utf-8") as handle:
        for row, line in zip(rows, handle, strict=True):
            item = json.loads(line)
            result = item.pop("result")
            if item != row or result["details"]["scoring_protocol"] != protocol:
                raise ValueError("Raw evaluation does not match the requested rows and protocol")
            scored.append(dict(row, result=result))
    return scored


def write_release_evaluation(scored, detector, cfg, model_path, output, regression_text, temperature_fit=None,
                             screening_fit=None, *, artifact_override=None, regression_result=None):
    """Calibrate and publish metrics from frozen scores and the same detector protocol."""
    artifact = artifact_override or build_calibration_artifact(scored, detector.scoring_protocol, model_path.name)
    model_routes = artifact.get("model_routes")
    primary = "zh" if "zh" in artifact["groups"] else "zh_long"
    chinese = artifact["groups"].get(primary, {})
    high = chinese.get("high_threshold")
    valid = validate_text_threshold_artifact(artifact, mode="readiness", expected_dataset_profile="binary-chinese-v1")
    if not valid.valid:
        raise ValueError(valid.errors)
    artifact_path = output.resolve() / "thresholds.json"
    artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    fp_config = cfg["detection"]["text_false_positive_control"]
    fp_config["artifact_path"] = str(artifact_path)
    controller = FalsePositiveController(fp_config)
    groups = defaultdict(list)
    decisions = []
    for row in scored:
        if row["split"] not in {"test", "external"}:
            continue
        decision = controller.evaluate(row["text"], [SimpleNamespace(status="ok", **row["result"])])
        details = row["result"]["details"]
        item = dict(source=row.get("source"), label=row["label"], identity=row["identity"],
                    generator=row.get("generator"), heldout_generator=row.get("heldout_generator", False),
                    chars=len(row["text"]), score=row["result"]["score"], decision=decision.to_dict(),
                    raw_margin=details.get("raw_margin"), external=row.get("external"))
        if screening_fit is not None:
            threshold = screening_fit.get(decision.threshold_group, {}).get("threshold") if model_routes else screening_fit["threshold"]
            item["screened"] = threshold is not None and screening_positive(decision.decision, item["score"], threshold)
            item["topic"] = row.get("domain") or decision.domain
            item["effective_chars"] = details.get("effective_chars", _effective_chars(row["text"]))
            item["suite"] = row.get("suite", row.get("corpus"))
            scope = f"external:{row['external']}" if row["split"] == "external" else "all"
            scopes = [scope]
            if item["suite"]:
                scopes.append(f"{scope}/{item['suite']}")
            for group in scopes:
                groups[f"{group}/topic={item['topic']}"].append(item)
                groups[f"{group}/length={_length_band(item['effective_chars'])}"].append(item)
                if item["generator"] and row["label"] == 1:
                    groups[f"{group}/generator={item['generator']}"].append(item)
            if model_routes:
                item["branch"] = model_calibration_group(details, model_routes)
                item["input_language"] = controller._language(row["text"])
                band = _length_band(item["effective_chars"])
                for group in scopes:
                    branch = f"{group}/branch={item['branch']}"
                    groups[branch].append(item)
                    groups[f"{branch}/length={band}"].append(item)
                    groups[f"{branch}/input={item['input_language']}"].append(item)
                    if row["split"] == "test":
                        groups[f"{branch}/source={row['source']}"].append(item)
                        groups[f"{branch}/source={row['source']}/length={band}"].append(item)
        decisions.append(item)
        if row["split"] == "external":
            name = f"external:{row['external']}"
            groups[name].append(item)
            if subgroup := row.get("suite", row.get("corpus")):
                groups[f"{name}/{subgroup}"].append(item)
            continue
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
        raw_metrics = binary_metrics(labels, scores, high if high is not None else 0.5)
        if model_routes:
            above = [float(r["decision"]["high_threshold"] is not None and r["score"] > r["decision"]["high_threshold"])
                     for r in values]
            raw_metrics = dict(binary_metrics(labels, above, 0.5), auroc=raw_metrics["auroc"],
                               calibrated_samples=sum(r["decision"]["high_threshold"] is not None for r in values))
        metrics[name] = dict(raw=raw_metrics,
                             product=binary_metrics(labels, positive, 0.5),
                             decisions=dict(Counter(r["decision"]["decision"] for r in values)),
                             scores=_score_stats(scores, labels))
        if screening_fit is not None:
            metrics[name]["screening"] = binary_metrics(labels, [float(r["screened"]) for r in values], 0.5)
            for policy in ("product", "screening"):
                values_for_policy = metrics[name][policy]
                values_for_policy["fnr"] = 1 - values_for_policy["recall"] if values_for_policy["ai"] else None
                interval = values_for_policy["recall_interval_95"]
                values_for_policy["fnr_interval_95"] = [1 - interval[1], 1 - interval[0]] if interval else None
        for label in ("human", "ai"):
            subset = [score for score, target in zip(scores, labels) if target == label]
            if subset:
                metrics[name][label + "_scores"] = _score_stats(subset, [label] * len(subset))
        if detector.aggregation_mode == "logit_mean":
            unscaled = [margin_probability(r["raw_margin"], 1.0) for r in values]
            metrics[name]["logit_at_temperature_1"] = dict(
                **_score_stats(unscaled, labels), auroc=binary_metrics(labels, unscaled, 0.5)["auroc"],
            )
    if regression_result is None:
        result = detector.detect(regression_text)
        regression_result = dict(score=result.score, details=result.details)
    regression = controller.evaluate(regression_text, [SimpleNamespace(status="ok", **regression_result)]).to_dict()
    regression["raw_margin"] = regression_result["details"].get("raw_margin")
    if screening_fit is not None:
        threshold = screening_fit.get(regression["threshold_group"], {}).get("threshold") if model_routes else screening_fit["threshold"]
        regression["screened"] = threshold is not None and screening_positive(regression["decision"], regression["score"], threshold)
    if model_routes:
        regression["branch"] = model_calibration_group(regression_result["details"], model_routes)
        regression["input_language"] = controller._language(regression_text)
    summary = dict(model=str(model_path), protocol=artifact["scoring_protocol"], high_threshold=high,
                   calibration_samples=chinese.get("human_calibration_samples"), independent_test_group=chinese,
                   temperature=temperature_fit["groups"].get(primary, {}).get("temperature") if model_routes else detector.temperature,
                   temperature_fit=temperature_fit, test=metrics, regression=regression)
    if screening_fit is not None:
        summary["screening_fit"] = screening_fit
    (output / "test-decisions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in decisions), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if not model_routes:
        (output / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        print(json.dumps({k: v for k, v in summary.items() if k != "regression"}, ensure_ascii=False, indent=2), flush=True)
    else:
        print(f"{artifact['threshold_policy']}: {len(decisions)} holdout decisions across {len(artifact['groups'])} calibrated model groups", flush=True)
    print("REGRESSION", regression["score"], regression["decision"], flush=True)


if __name__ == "__main__":
    main()
