#!/usr/bin/env python3
"""Fine-tune a local binary BERT detector using isolated document splits.

JSONL rows require text, label (0=human, 1=ai), split, group and identity.
Only train and validation are used; calibration and test remain untouched.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Any, Sequence
import unicodedata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detection.text_segmentation import SentenceAwareSegmenter


def load_dataset_rows(path: Path | str) -> list[dict[str, Any]]:
    """Validate labels and prohibit group, identity or normalized-text leakage."""
    rows: list[dict[str, Any]] = []
    seen: dict[str, dict[str, str]] = {name: {} for name in ("group", "identity", "text")}
    text_labels: dict[str, int] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: expected a JSON object")
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"line {line_number}: text must be nonempty")
            if type(row.get("label")) is not int or row["label"] not in (0, 1):
                raise ValueError(f"line {line_number}: label must be integer 0 or 1")
            split = row.get("split")
            if split not in ("train", "validation", "calibration", "test"):
                raise ValueError(f"line {line_number}: unsupported split {split!r}")
            normalized = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()
            text_hash = hashlib.sha256(normalized.encode()).hexdigest()
            for name, value in (("group", row.get("group")), ("identity", row.get("identity")), ("text", text_hash)):
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"line {line_number}: {name} must be nonempty")
                previous = seen[name].setdefault(value, split)
                if previous != split:
                    raise ValueError(f"line {line_number}: {name} leakage between {previous} and {split}")
            previous_label = text_labels.setdefault(text_hash, row["label"])
            if previous_label != row["label"]:
                raise ValueError(f"line {line_number}: conflicting labels for identical text")
            rows.append(row)
    if not rows:
        raise ValueError("dataset is empty")
    return rows


def select_split_rows(
    rows: list[dict[str, Any]], split: str, *, max_per_label: int, seed: int
) -> list[dict[str, Any]]:
    """Deterministically cap each label while retaining every available source."""
    if split not in ("train", "validation"):
        raise ValueError("selection is restricted to train and validation")
    generator = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for label in (0, 1):
        sources: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row["split"] == split and row["label"] == label:
                sources[str(row.get("source", "unknown"))].append(row)
        for source_rows in sources.values():
            generator.shuffle(source_rows)
        total = sum(map(len, sources.values()))
        limit = min(total, max_per_label) if max_per_label > 0 else total
        source_names = sorted(sources)
        generator.shuffle(source_names)
        count = 0
        while count < limit:
            for source in source_names:
                if sources[source] and count < limit:
                    selected.append(sources[source].pop())
                    count += 1
    generator.shuffle(selected)
    return selected


def document_windows(
    text: str, tokenizer: Any, *, max_length: int, min_weight: int = 1,
) -> list[dict[str, Any]]:
    """Cover every character segment and its tokenizer overflow, including tails."""
    windows: list[dict[str, Any]] = []
    previous_end = 0
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    for segment in SentenceAwareSegmenter(target_chars=480, max_chars=640, overlap_sentences=1).split(text):
        character_weight = max(min_weight, segment.end - max(segment.start, previous_end))
        previous_end = max(previous_end, segment.end)
        encoded = tokenizer(
            segment.text, truncation=True, max_length=max_length,
            return_overflowing_tokens=True, padding=False,
        )
        for index, input_ids in enumerate(encoded["input_ids"]):
            features = {name: encoded[name][index] for name in tokenizer.model_input_names if name in encoded}
            windows.append({
                "features": features,
                "token_weight": max(min_weight, sum(features.get("attention_mask", [1] * len(input_ids))) - special_tokens),
                "segment_index": segment.index,
                "character_weight": character_weight,
            })
    return windows


def evenly_spaced_windows(windows: list[dict[str, Any]], limit: int = 2) -> list[dict[str, Any]]:
    if limit < 1:
        raise ValueError("window limit must be positive")
    if len(windows) <= limit:
        return windows
    if limit == 1:
        return [windows[len(windows) // 2]]
    return [windows[round(index * (len(windows) - 1) / (limit - 1))] for index in range(limit)]


def training_sample_weights(
    rows: list[dict[str, Any]], window_counts: list[int], *, balance_sources: bool = False
) -> list[float]:
    """Give labels equal mass, optionally divided equally among their sources."""
    if len(rows) != len(window_counts) or any(count < 1 for count in window_counts):
        raise ValueError("each document must have one positive window count")
    label_counts = Counter(row["label"] for row in rows)
    source_documents = Counter((row["label"], str(row.get("source", "unknown"))) for row in rows)
    sources_per_label = Counter(label for label, _ in source_documents)
    weights: list[float] = []
    for row, count in zip(rows, window_counts):
        label = row["label"]
        documents = label_counts[label]
        if balance_sources:
            documents = sources_per_label[label] * source_documents[(label, str(row.get("source", "unknown")))]
        weights.extend([1.0 / (documents * count)] * count)
    return weights


def freeze_bert_layers(model: Any, last_layers: int = 4) -> dict[str, int]:
    """Keep the final encoder blocks, pooler and binary classifier trainable."""
    base = model.base_model
    if not hasattr(base, "encoder") or not hasattr(base.encoder, "layer"):
        raise ValueError("this fine-tuner requires a BERT-compatible encoder classifier")
    if last_layers < 1:
        raise ValueError("last_layers must be positive")
    for parameter in model.parameters():
        parameter.requires_grad = False
    modules = list(base.encoder.layer[-last_layers:]) + [model.classifier]
    for pooler in (getattr(base, "pooler", None), getattr(model, "pooler", None)):
        if pooler is not None:
            modules.append(pooler)
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


def score_documents(
    model: Any, tokenizer: Any, rows: list[dict[str, Any]], *,
    device: str, batch_size: int, max_length: int, aggregation_mode: str = "probability_mean",
) -> list[float]:
    """Batch GPU inference with the runtime's token then unique-character means."""
    import torch

    if aggregation_mode not in {"logit_mean", "probability_mean"}:
        raise ValueError(f"Unknown text aggregation mode: {aggregation_mode}")
    use_margin = aggregation_mode == "logit_mean"
    model.eval()
    windows: list[tuple[int, dict[str, Any]]] = []
    for document_index, row in enumerate(rows):
        windows.extend((document_index, window) for window in document_windows(
            row["text"], tokenizer, max_length=max_length, min_weight=0 if use_margin else 1,
        ))
    segment_scores: dict[tuple[int, int], list[float]] = {}
    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            chunk = windows[start:start + batch_size]
            batch = tokenizer.pad([window["features"] for _, window in chunk], padding=True, return_tensors="pt")
            logits = model(**{name: value.to(device) for name, value in batch.items()}).logits
            values = (logits[:, 1].double() - logits[:, 0].double() if use_margin else logits.softmax(dim=-1)[:, 1])
            for (document_index, window), value in zip(chunk, values.cpu().tolist()):
                key = (document_index, window["segment_index"])
                aggregate = segment_scores.setdefault(key, [0.0, 0, window["character_weight"]])
                aggregate[0] += value * window["token_weight"]
                aggregate[1] += window["token_weight"]
    totals = [[0.0, 0] for _ in rows]
    for (document_index, _), (weighted_score, tokens, characters) in segment_scores.items():
        segment_score = weighted_score / tokens if tokens else 0.0
        totals[document_index][0] += (segment_score if use_margin else round(segment_score, 4)) * characters
        totals[document_index][1] += characters
    if use_margin:
        margins = torch.tensor([score / weight if weight else 0.0 for score, weight in totals], dtype=torch.float64)
        exponential = (-margins.abs()).exp()
        probabilities = torch.where(margins >= 0, 1 / (1 + exponential), exponential / (1 + exponential))
        return probabilities.clamp(min=math.nextafter(0.0, 1.0), max=math.nextafter(1.0, 0.0)).tolist()
    return [round(score / weight, 4) if weight else 0.0 for score, weight in totals]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    identities = sorted(row["identity"] for row in rows)
    return {
        "documents": len(rows),
        "labels": dict(Counter(str(row["label"]) for row in rows)),
        "sources": dict(Counter(str(row.get("source", "unknown")) for row in rows)),
        "identities_sha256": hashlib.sha256("\n".join(identities).encode()).hexdigest(),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True, help="Existing local BERT snapshot")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-train-per-label", type=int, default=5000)
    parser.add_argument("--max-validation-per-label", type=int, default=400)
    parser.add_argument("--max-windows-per-document", type=int, default=2)
    parser.add_argument("--validation-aggregation", choices=("probability_mean", "logit_mean"), default="probability_mean")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--balance-sources", action="store_true", help="Give each source within a label equal sampling mass")
    parser.add_argument("--trainable-layers", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.eval_batch_size < 1 or args.lr <= 0 or args.max_length < 3:
        raise ValueError("epochs, batch sizes, learning rate and max_length must be positive")
    if args.max_windows_per_document < 1 or not 0 <= args.label_smoothing < 1:
        raise ValueError("window limit must be positive and label smoothing must be in [0, 1)")
    if not args.base_model.is_dir():
        raise ValueError("--base-model must be an existing local snapshot directory")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("--output must be empty to avoid overwriting an existing model")
    rows = load_dataset_rows(args.data)
    train = select_split_rows(rows, "train", max_per_label=args.max_train_per_label, seed=args.seed)
    validation = select_split_rows(rows, "validation", max_per_label=args.max_validation_per_label, seed=args.seed)
    for split, split_rows in (("train", train), ("validation", validation)):
        if {row["label"] for row in split_rows} != {0, 1}:
            raise ValueError(f"{split} must contain both human and ai labels")

    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model), local_files_only=True, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("a fast tokenizer is required for complete overflow windows")
    model = AutoModelForSequenceClassification.from_pretrained(str(args.base_model), local_files_only=True)
    if model.config.num_labels != 2:
        raise ValueError("base model must have exactly two labels")
    model.config.id2label = {0: "human", 1: "ai"}
    model.config.label2id = {"human": 0, "ai": 1}
    model.config.problem_type = "single_label_classification"
    parameter_counts = freeze_bert_layers(model, last_layers=args.trainable_layers)
    max_length = min(args.max_length, model.config.max_position_embeddings, tokenizer.model_max_length)
    model.to(args.device)
    training_windows: list[dict[str, Any]] = []
    window_counts: list[int] = []
    for row in train:
        windows = evenly_spaced_windows(
            document_windows(row["text"], tokenizer, max_length=max_length), limit=args.max_windows_per_document,
        )
        window_counts.append(len(windows))
        for window in windows:
            training_windows.append({**window["features"], "labels": row["label"]})
    weights = training_sample_weights(train, window_counts, balance_sources=args.balance_sources)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(weights, num_samples=len(training_windows), replacement=True, generator=generator)

    def collate(items):
        return tokenizer.pad(items, padding=True, return_tensors="pt")

    loader = DataLoader(training_windows, batch_size=args.batch_size, sampler=sampler, collate_fn=collate)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.lr, weight_decay=0.01)
    cuda = torch.device(args.device).type == "cuda"
    amp_dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=cuda and amp_dtype == torch.float16)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_sha256": _sha256(args.data),
        "script_sha256": _sha256(Path(__file__)),
        "base_model": str(args.base_model.resolve()),
        "base_config_sha256": _sha256(args.base_model / "config.json"),
        "base_model_sha256": {path.name: _sha256(path) for path in sorted(args.base_model.glob("*.safetensors"))},
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "label_mapping": {"0": "human", "1": "ai"},
        "parameters": parameter_counts,
        "splits": {split: _row_manifest([row for row in rows if row["split"] == split]) for split in ("train", "validation", "calibration", "test")},
        "selected_train": _row_manifest(train),
        "selected_validation": _row_manifest(validation),
        "training_windows": len(training_windows),
        "windows_per_document_counts": dict(sorted(Counter(window_counts).items())),
        "protocol": {
            "scoring": "token-complete-v1",
            "segments": {"target_chars": 480, "max_chars": 640, "overlap_sentences": 1},
            "max_length": max_length,
            "training_windows_per_document": args.max_windows_per_document,
            "window_selection": "evenly spaced including first and last",
            "sampling": (
                "class-balanced, then source-balanced, then equal probability per document within source"
                if args.balance_sources else "class-balanced with equal probability per document"
            ),
            "aggregation": "token-weighted segment mean then unique-character-weighted document mean",
            "validation_aggregation": args.validation_aggregation,
            "model_selection": "validation AUROC only",
            "held_out": ["calibration", "test", "user regression"],
        },
        "versions": {"torch": torch.__version__},
        "epochs": [],
    }
    print(json.dumps({"event": "ready", "train": len(train), "validation": len(validation), "windows": len(training_windows), "parameters": parameter_counts}), flush=True)
    best_auc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        examples = 0
        for step, batch in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            batch = {name: value.to(args.device) for name, value in batch.items()}
            labels = batch.pop("labels")
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=cuda):
                logits = model(**batch).logits
                loss = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=args.label_smoothing)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss at epoch {epoch}, step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            scaler.step(optimizer)
            scaler.update()
            count = labels.shape[0]
            total_loss += float(loss.detach()) * count
            examples += count
            if step % 50 == 0:
                print(json.dumps({"event": "train", "epoch": epoch, "step": step, "steps": len(loader), "loss": total_loss / examples}), flush=True)
        probabilities = score_documents(
            model, tokenizer, validation, device=args.device, batch_size=args.eval_batch_size,
            max_length=max_length, aggregation_mode=args.validation_aggregation,
        )
        auc = float(roc_auc_score([row["label"] for row in validation], probabilities))
        record = {"epoch": epoch, "train_loss": total_loss / examples, "validation_auroc": auc}
        metadata["epochs"].append(record)
        if auc > best_auc:
            best_auc = auc
            metadata["best_epoch"] = epoch
            metadata["best_validation_auroc"] = auc
            model.save_pretrained(args.output, safe_serialization=True)
            tokenizer.save_pretrained(args.output)
        (args.output / "training_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"event": "epoch", **record, "best_epoch": metadata["best_epoch"]}), flush=True)
    metadata["model_sha256"] = {path.name: _sha256(path) for path in sorted(args.output.glob("*.safetensors"))}
    (args.output / "training_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "output": str(args.output), "best_epoch": metadata["best_epoch"], "validation_auroc": best_auc}), flush=True)


if __name__ == "__main__":
    main()
