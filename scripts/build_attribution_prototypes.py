"""构建来源归因 prototype 样本库。

这个脚本不训练模型，只把“候选模型 -> 少量代表样本”整理成本项目可读的
JSONL 原型库，供 provenance.attribution 的 prototype 分支做 top-k 检索。

常用方式：
- OpenTuringBench 文本样本：
  python scripts/build_attribution_prototypes.py text-openturingbench --output data/attribution/text/openturingbench_prototypes.jsonl
- 自建图片样本目录：
  python scripts/build_attribution_prototypes.py image-folder --input data/attribution/image_samples --output data/attribution/image/image_prototypes.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def build_text_openturingbench(args: argparse.Namespace) -> None:
    """从 Hugging Face OpenTuringBench 数据集抽样生成文本原型库。"""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing optional dependency `datasets`. Install it with: "
            "pip install datasets"
        ) from exc

    if args.cache_dir:
        cache_root = Path(args.cache_dir)
        os.environ["HF_HOME"] = str(cache_root)
        os.environ["HF_HUB_CACHE"] = str(cache_root / "hub")

    dataset = load_dataset(
        args.dataset_name,
        args.subset,
        split=args.split,
        cache_dir=args.cache_dir,
    )
    counts: dict[str, int] = defaultdict(int)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in dataset:
            model = row.get(args.model_column)
            text = row.get(args.text_column)
            if not model or not isinstance(text, str) or not text.strip():
                continue
            if counts[str(model)] >= args.max_per_model:
                continue
            handle.write(json.dumps({"model": str(model), "text": text}, ensure_ascii=False) + "\n")
            counts[str(model)] += 1
            if len(counts) >= args.max_models and all(value >= args.max_per_model for value in counts.values()):
                break
    print(f"Wrote {sum(counts.values())} text prototypes for {len(counts)} models to {output}")


def build_image_folder(args: argparse.Namespace) -> None:
    """从按模型名分组的图片目录生成 pHash 原型库。

    目录结构示例：
    data/attribution/image_samples/
      midjourney/
        001.png
      stable-diffusion-xl/
        001.jpg
    """
    try:
        import imagehash
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            "Missing optional dependency for image prototypes. Install with: "
            "pip install imagehash Pillow"
        ) from exc

    root = Path(args.input)
    if not root.exists():
        raise SystemExit(f"Input folder does not exist: {root}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffixes = {suffix.lower() for suffix in args.suffixes}
    written = 0
    with output.open("w", encoding="utf-8") as handle:
        for model_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            count = 0
            for image_path in iter_images(model_dir, suffixes):
                if count >= args.max_per_model:
                    break
                try:
                    phash = str(imagehash.phash(Image.open(image_path).convert("RGB")))
                except Exception as exc:  # noqa: BLE001 - 坏图跳过即可。
                    print(f"Skip {image_path}: {exc}")
                    continue
                handle.write(
                    json.dumps(
                        {
                            "model": model_dir.name,
                            "path": str(image_path),
                            "phash": phash,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                count += 1
                written += 1
    print(f"Wrote {written} image prototypes to {output}")


def iter_images(root: Path, suffixes: set[str]) -> Iterable[Path]:
    """遍历图片文件。"""
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in suffixes:
            yield path


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Build local attribution prototype files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    text_parser = subparsers.add_parser("text-openturingbench", help="Build text prototypes from OpenTuringBench.")
    text_parser.add_argument("--dataset-name", default="MLNTeam-Unical/OpenTuringBench")
    text_parser.add_argument("--subset", default="in_domain")
    text_parser.add_argument("--split", default="train")
    text_parser.add_argument("--text-column", default="content")
    text_parser.add_argument("--model-column", default="model")
    text_parser.add_argument("--max-per-model", type=int, default=50)
    text_parser.add_argument("--max-models", type=int, default=32)
    text_parser.add_argument("--cache-dir", default="models/huggingface")
    text_parser.add_argument("--output", default="data/attribution/text/openturingbench_prototypes.jsonl")
    text_parser.set_defaults(func=build_text_openturingbench)

    image_parser = subparsers.add_parser("image-folder", help="Build image pHash prototypes from labeled folders.")
    image_parser.add_argument("--input", default="data/attribution/image_samples")
    image_parser.add_argument("--output", default="data/attribution/image/image_prototypes.jsonl")
    image_parser.add_argument("--max-per-model", type=int, default=100)
    image_parser.add_argument("--suffixes", nargs="+", default=[".jpg", ".jpeg", ".png", ".webp"])
    image_parser.set_defaults(func=build_image_folder)

    return parser.parse_args()


def main() -> None:
    """脚本入口。"""
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
