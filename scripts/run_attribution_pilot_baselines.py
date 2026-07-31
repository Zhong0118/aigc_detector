"""Run leakage-safe text-attribution baselines on a collected JSONL dataset."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback
from typing import Sequence

from research.text_attribution.baseline_data import (
    assign_group_folds,
    load_clean_samples,
)
from research.text_attribution.baseline_models import (
    EvaluationResult,
    evaluate_embedding_probe,
    evaluate_sparse_method,
    load_or_extract_embeddings,
)
from research.text_attribution.baseline_reporting import write_baseline_report


_METHODS = ("char_lr", "tfidf_svm", "frozen_e5")
_SPARSE_METHODS = ("char_lr", "tfidf_svm")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate leakage-safe sparse and frozen-encoder attribution "
            "baselines on collected JSONL."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=_METHODS,
        default=list(_METHODS),
    )
    parser.add_argument(
        "--encoder-model",
        default="intfloat/multilingual-e5-base",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--hf-cache", type=Path, default=Path("models/huggingface"))
    parser.add_argument("--embedding-cache", type=Path)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    samples, audit = load_clean_samples(args.input)
    folds = assign_group_folds(
        samples,
        n_splits=args.folds,
        seed=args.seed,
    )
    summary = {
        "input": str(args.input),
        "raw_records": audit["raw_records"],
        "clean_records": audit["kept_records"],
        "excluded_records": audit["excluded_records"],
        "excluded_by_reason": audit["excluded_by_reason"],
        "labels": audit["counts_by_label"],
        "semantic_prompt_groups": audit["semantic_prompt_groups"],
        "folds": args.folds,
        "methods": list(args.methods),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        print("Dry run complete; no output directory was created.", flush=True)
        return 0

    run_config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "methods": list(args.methods),
        "encoder_model": args.encoder_model,
        "device": args.device,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "hf_cache": str(args.hf_cache),
        "embedding_cache": (
            str(args.embedding_cache) if args.embedding_cache is not None else None
        ),
        "folds": args.folds,
        "seed": args.seed,
        "split_strategy": "semantic_prompt_group_round_robin",
    }
    results: list[EvaluationResult] = []
    failures: list[dict[str, str]] = []

    for method in _SPARSE_METHODS:
        if method not in args.methods:
            continue
        print(f"[{method}] fitting {args.folds} OOF folds ...", flush=True)
        try:
            result = evaluate_sparse_method(
                samples,
                folds,
                method=method,
                seed=args.seed,
            )
            results.append(result)
            print(
                f"[{method}] macro-F1={result.oof_metrics['macro_f1']:.4f}",
                flush=True,
            )
        except Exception as exc:  # keep independent methods and diagnostics
            failures.append(_failure_record(method, exc))
            traceback.print_exc()

    # Persist lightweight results before any model download or GPU allocation.
    if results:
        write_baseline_report(
            samples=samples,
            audit=audit,
            results=results,
            output_dir=args.output_dir,
            run_config=run_config,
        )

    if "frozen_e5" in args.methods:
        method = "frozen_e5"
        cache_path = args.embedding_cache or (
            args.output_dir
            / "embeddings"
            / f"{_safe_model_name(args.encoder_model)}.npz"
        )
        print(
            f"[{method}] loading/extracting embeddings at {cache_path} ...",
            flush=True,
        )
        try:
            embeddings, cache_hit = load_or_extract_embeddings(
                samples,
                cache_path=cache_path,
                model_id=args.encoder_model,
                device=args.device,
                batch_size=args.batch_size,
                max_length=args.max_length,
                hf_cache=args.hf_cache,
            )
            print(
                f"[{method}] embeddings ready "
                f"({embeddings.shape[0]}x{embeddings.shape[1]}, "
                f"cache_hit={cache_hit})",
                flush=True,
            )
            result = evaluate_embedding_probe(
                samples,
                folds,
                embeddings,
                seed=args.seed,
            )
            results.append(result)
            print(
                f"[{method}] macro-F1={result.oof_metrics['macro_f1']:.4f}",
                flush=True,
            )
        except Exception as exc:  # sparse artifacts remain available
            failures.append(_failure_record(method, exc))
            traceback.print_exc()

    if results:
        artifacts = write_baseline_report(
            samples=samples,
            audit=audit,
            results=results,
            output_dir=args.output_dir,
            run_config=run_config,
        )
        print(
            f"Baseline artifacts written to {args.output_dir.resolve()}",
            flush=True,
        )
        print(
            f"Markdown report: {artifacts['report_markdown']}",
            flush=True,
        )

    if failures:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        failure_path = args.output_dir / "method_failures.json"
        failure_path.write_text(
            json.dumps(failures, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Method failures recorded at {failure_path}", file=sys.stderr)
    if not results:
        print("No baseline method completed successfully.", file=sys.stderr)
        return 1
    return 1 if failures else 0


def _safe_model_name(model_id: str) -> str:
    return model_id.replace("/", "__").replace("\\", "__")


def _failure_record(method: str, error: Exception) -> dict[str, str]:
    return {
        "method": method,
        "error_type": type(error).__name__,
        "error_message": str(error),
    }


if __name__ == "__main__":
    raise SystemExit(main())
