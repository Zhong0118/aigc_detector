"""Select the top neural attribution experiments from validation artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Sequence

from research.text_attribution.neural_selection import (
    NeuralSelectionSummary,
    select_neural_experiments,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank formal N0/N1/N2 family-attribution runs using validation "
            "metrics only."
        )
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        type=Path,
        help="Completed neural experiment directories.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New directory for the screening summary.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir: Path = args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")

    summary = select_neural_experiments(args.inputs)
    _publish_summary(output_dir, summary)
    return 0


def _publish_summary(
    output_dir: Path,
    summary: NeuralSelectionSummary,
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-",
            dir=output_dir.parent,
        )
    )
    try:
        json_path = staging / "neural_screening_summary.json"
        markdown_path = staging / "neural_screening_summary.md"
        _write_text_durable(
            json_path,
            json.dumps(
                summary.to_dict(),
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            + "\n",
        )
        _write_text_durable(markdown_path, _markdown(summary))
        _validate_staged_summary(json_path, markdown_path)
        if output_dir.exists():
            raise FileExistsError(
                f"output directory already exists: {output_dir}"
            )
        os.replace(staging, output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _write_text_durable(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _validate_staged_summary(json_path: Path, markdown_path: Path) -> None:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("selection_scope") != (
        "validation_only"
    ):
        raise RuntimeError("staged JSON summary is not validation-only")
    if not markdown_path.is_file() or not markdown_path.read_text(
        encoding="utf-8"
    ).strip():
        raise RuntimeError("staged Markdown summary is empty")


def _markdown(summary: NeuralSelectionSummary) -> str:
    lines = [
        "# Neural Attribution Screening",
        "",
        "Selection scope: **validation only**. Test metrics were neither "
        "accepted nor used.",
        "",
        f"- Dataset fingerprint: `{summary.dataset_fingerprint}`",
        f"- Split manifest: `{summary.split_manifest or 'not recorded'}`",
        f"- Top two: {', '.join(summary.top_two)}",
        "",
        "| Rank | Experiment | Source | Validation macro-F1 | "
        "Worst-family F1 | Validation NLL |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for rank, item in enumerate(summary.ranking, start=1):
        source = item.source.replace("|", "\\|")
        lines.append(
            f"| {rank} | {item.experiment_id} | `{source}` | "
            f"{item.macro_f1:.6f} | {item.worst_family_f1:.6f} | "
            f"{item.negative_log_likelihood:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
