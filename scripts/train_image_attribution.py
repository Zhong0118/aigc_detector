"""训练图片来源归因分类器。

默认路线：
- 从 Hugging Face Tiny-GenImage 或本地图片目录读取图片和 generator 标签
- 提取轻量图像统计 + pHash/dHash 特征
- 训练 LogisticRegression 多分类器
- 保存为 provenance.attribution 可直接加载的 joblib 文件

示例：
python scripts/train_image_attribution.py --dataset tiny-genimage --max-samples 10000
"""

from __future__ import annotations

import argparse
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def log(message: str) -> None:
    """输出带阶段感的训练日志。"""
    print(f"[image-attribution] {message}", flush=True)


def progress(iterable: Iterable, total: int | None = None, desc: str = "progress") -> Iterable:
    """优先使用 tqdm；没有安装时退化成定期打印。"""
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, total=total, desc=desc, unit="items")
    except ImportError:
        return simple_progress(iterable, total=total, desc=desc)


def simple_progress(iterable: Iterable, total: int | None = None, desc: str = "progress") -> Iterable:
    """无 tqdm 时的简易进度输出。"""
    for index, item in enumerate(iterable, start=1):
        if index == 1 or index % 500 == 0:
            suffix = f"/{total}" if total else ""
            log(f"{desc}: {index}{suffix}")
        yield item


def load_tiny_genimage(args: argparse.Namespace) -> tuple[list[Image.Image], list[str]]:
    """从 Hugging Face Tiny-GenImage 加载图片和 generator 标签。"""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("请先安装 datasets：python -m pip install datasets") from exc

    configure_hf_cache(args.cache_dir)
    log(f"loading dataset {args.dataset_name} split={args.split}")
    dataset = load_dataset(args.dataset_name, split=args.split, cache_dir=args.cache_dir)
    log(f"dataset loaded, rows={len(dataset)}")
    images: list[Image.Image] = []
    labels: list[str] = []
    counts: dict[str, int] = {}
    for row in progress(dataset, total=len(dataset), desc="collect image rows"):
        image = row.get(args.image_column) or row.get("image")
        raw_label = row.get(args.label_column) if args.label_column in row else row.get("generator")
        label = decode_dataset_label(dataset.features, args.label_column, raw_label)
        if image is None or label is None:
            continue
        label = str(label)
        if max_per_label_reached(counts, label, args.max_per_label):
            continue
        images.append(image.convert("RGB") if hasattr(image, "convert") else Image.open(image).convert("RGB"))
        labels.append(label)
        counts[label] = counts.get(label, 0) + 1
        if args.max_samples and len(images) >= args.max_samples:
            break
    log(f"collected images={len(images)}, labels={len(counts)}")
    return images, labels


def load_image_folder(args: argparse.Namespace) -> tuple[list[Image.Image], list[str]]:
    """从按来源模型分组的本地目录加载图片。"""
    root = Path(args.input)
    if not root.exists():
        raise SystemExit(f"本地图片目录不存在：{root}")
    images: list[Image.Image] = []
    labels: list[str] = []
    counts: dict[str, int] = {}
    for model_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        label = model_dir.name
        for image_path in iter_image_paths(model_dir):
            if max_per_label_reached(counts, label, args.max_per_label):
                break
            try:
                images.append(Image.open(image_path).convert("RGB"))
            except Exception as exc:  # noqa: BLE001 - 坏图跳过。
                print(f"Skip {image_path}: {exc}")
                continue
            labels.append(label)
            counts[label] = counts.get(label, 0) + 1
            if args.max_samples and len(images) >= args.max_samples:
                log(f"collected images={len(images)}, labels={len(counts)}")
                return images, labels
    log(f"collected images={len(images)}, labels={len(counts)}")
    return images, labels


def train(args: argparse.Namespace) -> None:
    """训练并保存图片来源分类器。"""
    loaders = {
        "tiny-genimage": load_tiny_genimage,
        "image-folder": load_image_folder,
    }
    log(f"start training, dataset={args.dataset}")
    images, labels = loaders[args.dataset](args)
    if len(set(labels)) < 2:
        raise SystemExit("至少需要 2 个图片来源类别才能训练。")
    if len(images) < 20:
        raise SystemExit(f"样本太少：{len(images)}。建议至少准备几十到几百张。")
    print_label_distribution(labels)

    log(f"extracting image features with {args.feature_extractor}")
    if args.feature_extractor == "clip":
        features = extract_clip_features(images, args)
        features_used = ["clip_image_embedding", args.clip_model_name]
    else:
        features = np.asarray(
            [extract_image_features(image) for image in progress(images, total=len(images), desc="extract features")],
            dtype=np.float32,
        )
        features_used = ["rgb_stats", "gray_percentiles", "phash", "dhash"]
    log("splitting train/test")
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels if min(labels.count(label) for label in set(labels)) >= 2 else None,
    )
    log(f"training classifier: train={len(x_train)}, test={len(x_test)}, labels={len(set(labels))}")
    model = LogisticRegression(
        max_iter=args.max_iter,
        class_weight="balanced",
        n_jobs=args.n_jobs,
        random_state=args.seed,
    )
    model.fit(x_train, y_train)
    log("evaluating classifier")
    predictions = model.predict(x_test)
    print(classification_report(y_test, predictions, zero_division=0))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    log(f"saving model to {output}")
    joblib.dump(
        {
            "model": model,
            "dataset": args.dataset,
            "feature_type": args.feature_extractor,
            "clip_model_name": args.clip_model_name if args.feature_extractor == "clip" else None,
            "cache_dir": args.cache_dir,
            "features_used": features_used,
            "label_count": len(set(labels)),
            "sample_count": len(images),
            "classes": sorted(set(labels)),
        },
        output,
    )
    log(f"saved image attribution classifier to {output}")


def extract_image_features(image: Image.Image) -> list[float]:
    """提取与 provenance.attribution 保持一致的图片特征。"""
    import imagehash

    image = image.convert("RGB").resize((128, 128))
    array = np.asarray(image).astype(np.float32) / 255.0
    means = array.mean(axis=(0, 1))
    stds = array.std(axis=(0, 1))
    mins = array.min(axis=(0, 1))
    maxs = array.max(axis=(0, 1))
    gray = array.mean(axis=2)
    phash = imagehash.phash(image)
    dhash = imagehash.dhash(image)
    hash_values = [float(bit) for bit in hash_to_bits(str(phash)) + hash_to_bits(str(dhash))]
    stats = [
        float(gray.mean()),
        float(gray.std()),
        float(np.percentile(gray, 10)),
        float(np.percentile(gray, 50)),
        float(np.percentile(gray, 90)),
    ]
    return [*means.tolist(), *stds.tolist(), *mins.tolist(), *maxs.tolist(), *stats, *hash_values]


def extract_clip_features(images: list[Image.Image], args: argparse.Namespace) -> np.ndarray:
    """使用 CLIP 图像编码器提取 embedding。"""
    import torch
    from transformers import CLIPImageProcessor, CLIPModel

    configure_hf_cache(args.cache_dir)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log(f"loading CLIP model {args.clip_model_name} on {device}")
    processor = CLIPImageProcessor.from_pretrained(args.clip_model_name, cache_dir=args.cache_dir)
    model = CLIPModel.from_pretrained(args.clip_model_name, cache_dir=args.cache_dir).to(device)
    model.eval()

    embeddings: list[np.ndarray] = []
    batch_size = max(1, int(args.batch_size))
    with torch.no_grad():
        for start in progress(range(0, len(images), batch_size), total=(len(images) + batch_size - 1) // batch_size, desc="clip batches"):
            batch = [image.convert("RGB") for image in images[start:start + batch_size]]
            inputs = processor(images=batch, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            features = model.get_image_features(pixel_values=pixel_values)
            features = features / features.norm(dim=-1, keepdim=True)
            embeddings.append(features.cpu().numpy())
    return np.vstack(embeddings).astype(np.float32)


def decode_dataset_label(features: Any, column: str, value: Any) -> str | None:
    """把 Hugging Face ClassLabel 数字还原为可读模型名。"""
    if value is None:
        return None
    feature = None
    if isinstance(features, dict):
        feature = features.get(column)
    else:
        try:
            feature = features[column]
        except Exception:  # noqa: BLE001
            feature = None
    if feature is not None and hasattr(feature, "int2str") and isinstance(value, (int, np.integer)):
        try:
            return str(feature.int2str(int(value)))
        except Exception:  # noqa: BLE001
            pass
    if column == "generator" and isinstance(value, (int, np.integer)):
        fallback = {
            0: "Real",
            1: "ADM",
            2: "BigGAN",
            3: "GLIDE",
            4: "Midjourney",
            5: "SD14",
            6: "SD15",
            7: "VQDM",
            8: "Wukong",
        }
        return fallback.get(int(value), str(value))
    return str(value)


def hash_to_bits(hex_hash: str) -> list[int]:
    """把 imagehash 十六进制哈希转换成 bit 列表。"""
    bits = bin(int(hex_hash, 16))[2:].zfill(len(hex_hash) * 4)
    return [int(bit) for bit in bits]


def iter_image_paths(root: Path) -> Iterable[Path]:
    """遍历本地图片路径。"""
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def max_per_label_reached(counts: dict[str, int], label: str, max_per_label: int) -> bool:
    """判断某个类别是否达到抽样上限。"""
    return bool(max_per_label and counts.get(label, 0) >= max_per_label)


def configure_hf_cache(cache_dir: str) -> None:
    """把 Hugging Face 下载缓存固定在项目目录。"""
    root = Path(cache_dir)
    os.environ["HF_HOME"] = str(root)
    os.environ["HF_HUB_CACHE"] = str(root / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(root / "datasets")
    os.environ["TRANSFORMERS_CACHE"] = str(root / "hub")


def print_label_distribution(labels: list[str]) -> None:
    """打印训练类别分布，确认数字标签已经还原为模型名。"""
    counts = Counter(labels)
    log("label distribution:")
    for label, count in counts.most_common():
        print(f"  {label}: {count}", flush=True)


def parse_args() -> argparse.Namespace:
    """解析训练参数。"""
    parser = argparse.ArgumentParser(description="Train image source attribution classifier.")
    parser.add_argument("--dataset", choices=["tiny-genimage", "image-folder"], default="tiny-genimage")
    parser.add_argument("--dataset-name", default="TheKernel01/Tiny-GenImage")
    parser.add_argument("--split", default="train")
    parser.add_argument("--input", default="data/attribution/image_samples")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--label-column", default="generator")
    parser.add_argument("--cache-dir", default="models/huggingface")
    parser.add_argument("--output", default="models/attribution/image_source_classifier.joblib")
    parser.add_argument("--feature-extractor", choices=["clip", "basic"], default="clip")
    parser.add_argument("--clip-model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="")
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--max-per-label", type=int, default=1000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True):
        train(parse_args())
