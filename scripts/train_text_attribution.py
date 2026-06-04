"""训练文本来源归因分类器。

默认路线：
- 下载/读取带来源模型标签的数据集
- 使用字符级 TF-IDF 提取语言风格特征
- 训练 LogisticRegression 多分类器
- 保存为 provenance.attribution 可直接加载的 joblib 文件

示例：
python scripts/train_text_attribution.py --dataset openturingbench --max-samples 20000
python scripts/train_text_attribution.py --dataset openturingbench --extra-jsonl data/attribution/text/custom_text_sources.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Iterable

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def log(message: str) -> None:
    """输出带阶段感的训练日志。"""
    print(f"[text-attribution] {message}", flush=True)


def progress(iterable: Iterable, total: int | None = None, desc: str = "progress") -> Iterable:
    """优先使用 tqdm；没有安装时退化成定期打印。"""
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, total=total, desc=desc, unit="rows")
    except ImportError:
        return simple_progress(iterable, total=total, desc=desc)


def simple_progress(iterable: Iterable, total: int | None = None, desc: str = "progress") -> Iterable:
    """无 tqdm 时的简易进度输出。"""
    for index, item in enumerate(iterable, start=1):
        if index == 1 or index % 1000 == 0:
            suffix = f"/{total}" if total else ""
            log(f"{desc}: {index}{suffix}")
        yield item


def load_openturingbench(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    """从 Hugging Face OpenTuringBench 加载文本和来源模型标签。"""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("请先安装 datasets：python -m pip install datasets") from exc

    configure_hf_cache(args.cache_dir)
    log(f"loading dataset {args.dataset_name}/{args.subset} split={args.split}")
    dataset = load_dataset(
        args.dataset_name,
        args.subset,
        split=args.split,
        cache_dir=args.cache_dir,
    )
    log(f"dataset loaded, rows={len(dataset)}")
    return collect_text_rows(
        dataset,
        text_column=args.text_column or "content",
        label_column=args.label_column or "model",
        max_samples=per_dataset_max_samples(args),
        min_chars=args.min_chars,
        max_per_label=args.max_per_label,
    )


def load_mage(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    """从 Hugging Face MAGE 加载文本和 source 信息。"""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("请先安装 datasets：python -m pip install datasets") from exc

    configure_hf_cache(args.cache_dir)
    log("loading dataset yaful/MAGE")
    dataset = load_dataset("yaful/MAGE", split=args.split, cache_dir=args.cache_dir)
    log(f"dataset loaded, rows={len(dataset)}")
    text_column = args.mage_text_column or "text"
    label_column = args.mage_label_column or "src"
    return collect_text_rows(
        dataset,
        text_column=text_column,
        label_column=label_column,
        max_samples=per_dataset_max_samples(args),
        min_chars=args.min_chars,
        max_per_label=args.max_per_label,
        label_normalizer=lambda value: normalize_mage_label(value, enabled=args.normalize_mage_labels),
    )


def load_jsonl(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    """读取本地 JSONL：每行包含 text 和 model/source_model/label。"""
    input_path = args.input or args.extra_jsonl
    if not input_path:
        raise SystemExit("--input or --extra-jsonl is required for local-jsonl dataset.")
    texts: list[str] = []
    labels: list[str] = []
    for line in Path(input_path).read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        text = row.get(args.text_column) or row.get("text") or row.get("content")
        label = row.get(args.label_column) or row.get("model") or row.get("source_model") or row.get("label")
        if isinstance(text, str) and text.strip() and label:
            texts.append(text)
            labels.append(str(label))
        if per_dataset_max_samples(args) and len(texts) >= per_dataset_max_samples(args):
            break
    return texts, labels


def per_dataset_max_samples(args: argparse.Namespace) -> int:
    """返回每个数据源的最大样本数。"""
    return int(args.max_samples_per_dataset or args.max_samples or 0)


def merge_labeled_samples(parts: list[tuple[list[str], list[str]]]) -> tuple[list[str], list[str]]:
    """合并多个数据源的文本和标签。"""
    texts: list[str] = []
    labels: list[str] = []
    for part_texts, part_labels in parts:
        texts.extend(part_texts)
        labels.extend(part_labels)
    return texts, labels


def normalize_mage_label(label: str, enabled: bool = False) -> str:
    """把 MAGE 的任务型标签规整成更粗粒度的来源模型名。"""
    if not enabled:
        return str(label)
    value = str(label)
    if value.endswith("_human") or value == "human" or "_human" in value:
        return "human"
    replacements = {
        "gpt-3.5-trubo": "gpt-3.5-turbo",
        "text-davinci-003": "text-davinci-003",
        "text-davinci-002": "text-davinci-002",
        "gpt_neox": "gpt_neox",
        "gpt_j": "gpt_j",
        "bloom_7b": "bloom_7b",
        "GLM130B": "GLM130B",
    }
    for key, normalized in replacements.items():
        if key in value:
            return normalized
    prefixes = [
        "cmv_machine_continuation_",
        "cmv_machine_specified_",
        "cmv_machine_topical_",
        "eli5_machine_continuation_",
        "eli5_machine_specified_",
    ]
    for prefix in prefixes:
        if value.startswith(prefix):
            return value.removeprefix(prefix)
    return value


def load_datasets(args: argparse.Namespace) -> tuple[list[str], list[str], list[str]]:
    """按逗号分隔的 dataset 参数加载并合并多个数据源。"""
    loaders = {
        "openturingbench": load_openturingbench,
        "mage": load_mage,
        "local-jsonl": load_jsonl,
    }
    names = [name.strip() for name in str(args.dataset).split(",") if name.strip()]
    parts: list[tuple[list[str], list[str]]] = []
    loaded_names: list[str] = []
    for name in names:
        if name not in loaders:
            raise SystemExit(f"Unsupported dataset: {name}. Choose from {', '.join(loaders)}")
        log(f"loading source dataset={name}")
        source_args = argparse.Namespace(**vars(args))
        source_args.dataset = name
        texts, labels = loaders[name](source_args)
        log(f"source dataset={name} collected samples={len(texts)}, labels={len(set(labels))}")
        parts.append((texts, labels))
        loaded_names.append(name)
    texts, labels = merge_labeled_samples(parts)
    if args.extra_jsonl and "local-jsonl" not in loaded_names:
        log(f"loading extra jsonl={args.extra_jsonl}")
        extra_args = argparse.Namespace(**vars(args))
        extra_args.input = args.extra_jsonl
        extra_texts, extra_labels = load_jsonl(extra_args)
        texts.extend(extra_texts)
        labels.extend(extra_labels)
        loaded_names.append("extra-jsonl")
    if args.max_samples and len(texts) > args.max_samples:
        texts, labels = texts[:args.max_samples], labels[:args.max_samples]
    log(f"merged samples={len(texts)}, labels={len(set(labels))}, sources={loaded_names}")
    return texts, labels, loaded_names


def collect_text_rows(
    rows: Iterable[dict],
    text_column: str,
    label_column: str,
    max_samples: int,
    min_chars: int,
    max_per_label: int,
    label_normalizer=None,
) -> tuple[list[str], list[str]]:
    """从 dataset rows 中抽取文本和标签，并做每类限量。"""
    texts: list[str] = []
    labels: list[str] = []
    counts: dict[str, int] = {}
    total = len(rows) if hasattr(rows, "__len__") else None
    for row in progress(rows, total=total, desc="collect text rows"):
        text = row.get(text_column) or row.get("text") or row.get("content")
        label = row.get(label_column) or row.get("model") or row.get("src") or row.get("source")
        if not isinstance(text, str) or len(text.strip()) < min_chars or not label:
            continue
        label = str(label)
        if label_normalizer is not None:
            label = label_normalizer(label)
        if max_per_label and counts.get(label, 0) >= max_per_label:
            continue
        texts.append(text)
        labels.append(label)
        counts[label] = counts.get(label, 0) + 1
        if max_samples and len(texts) >= max_samples:
            break
    log(f"collected samples={len(texts)}, labels={len(counts)}")
    return texts, labels


def train(args: argparse.Namespace) -> None:
    """训练并保存文本来源分类器。"""
    log(f"start training, dataset={args.dataset}")
    texts, labels, loaded_names = load_datasets(args)
    if len(set(labels)) < 2:
        raise SystemExit("至少需要 2 个来源模型类别才能训练。")
    if len(texts) < 20:
        raise SystemExit(f"样本太少：{len(texts)}。建议至少准备几十到几百条。")
    print_label_distribution(labels)

    log("splitting train/test")
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels if min(labels.count(label) for label in set(labels)) >= 2 else None,
    )
    log(f"training classifier: train={len(x_train)}, test={len(x_test)}, labels={len(set(labels))}")
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer=args.analyzer,
                    ngram_range=(args.ngram_min, args.ngram_max),
                    max_features=args.max_features,
                    min_df=args.min_df,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=args.max_iter,
                    class_weight="balanced",
                    n_jobs=args.n_jobs,
                    random_state=args.seed,
                ),
            ),
        ]
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
            "dataset": loaded_names,
            "features_used": ["char_tfidf", "logistic_regression"],
            "label_count": len(set(labels)),
            "sample_count": len(texts),
        },
        output,
    )
    log(f"saved text attribution classifier to {output}")


def configure_hf_cache(cache_dir: str) -> None:
    """把 Hugging Face 下载缓存固定在项目目录。"""
    root = Path(cache_dir)
    os.environ["HF_HOME"] = str(root)
    os.environ["HF_HUB_CACHE"] = str(root / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(root / "datasets")
    os.environ["TRANSFORMERS_CACHE"] = str(root / "hub")


def print_label_distribution(labels: list[str]) -> None:
    """打印训练类别分布，避免用户不知道实际训练了哪些来源模型。"""
    counts = Counter(labels)
    log("label distribution:")
    for label, count in counts.most_common():
        print(f"  {label}: {count}", flush=True)


def parse_args() -> argparse.Namespace:
    """解析训练参数。"""
    parser = argparse.ArgumentParser(description="Train text source attribution classifier.")
    parser.add_argument(
        "--dataset",
        default="openturingbench",
        help="Dataset name or comma-separated list: openturingbench,mage,local-jsonl.",
    )
    parser.add_argument("--dataset-name", default="MLNTeam-Unical/OpenTuringBench")
    parser.add_argument("--subset", default="in_domain")
    parser.add_argument("--split", default="train")
    parser.add_argument("--input", default="")
    parser.add_argument("--extra-jsonl", default="", help="Optional local JSONL samples appended to the selected dataset.")
    parser.add_argument("--text-column", default="content")
    parser.add_argument("--label-column", default="model")
    parser.add_argument("--mage-text-column", default="text")
    parser.add_argument("--mage-label-column", default="src")
    parser.add_argument("--normalize-mage-labels", action="store_true", help="Collapse MAGE task labels into coarser model names.")
    parser.add_argument("--cache-dir", default="models/huggingface")
    parser.add_argument("--output", default="models/attribution/text_source_classifier.joblib")
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--max-samples-per-dataset", type=int, default=0)
    parser.add_argument("--max-per-label", type=int, default=1000)
    parser.add_argument("--min-chars", type=int, default=50)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--analyzer", default="char_wb")
    parser.add_argument("--ngram-min", type=int, default=3)
    parser.add_argument("--ngram-max", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=120000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
