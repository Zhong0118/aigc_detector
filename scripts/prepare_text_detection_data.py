"""Prepare cached full Chinese documents for binary detector training; no downloads."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import re
from pathlib import Path


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def partition(group):
    bucket = int(digest("binary-v2:" + group)[:12], 16) % 10
    return "test" if bucket < 2 else "validation" if bucket < 3 else "calibration" if bucket < 4 else "train"


def prepare(records, *, excluded=(), train_limit=1500, evaluation_limit=800):
    records = [r for r in records if isinstance(r["text"], str) and len(r["text"].strip()) >= 120]
    parent = {r["group"]: r["group"] for r in records}

    def root(group):
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group

    aliases = {}
    for r in records:
        r["identity"] = digest(" ".join(r["text"].split()))
        other = aliases.setdefault(r["identity"], r["group"])
        left, right = sorted((root(r["group"]), root(other)))
        parent[right] = left
    roles = {}
    priority = {"train": 0, "calibration": 1, "validation": 2, "test": 3}
    for r in records:
        group = root(r["group"])
        roles[group] = max(roles.get(group, "train"), r["split"], key=priority.get)
    unique = {}
    excluded_groups = {root(r["group"]) for r in records if r["identity"] in excluded}
    for r in sorted(records, key=lambda r: (r["identity"], r["group"])):
        group = root(r["group"])
        if group in excluded_groups:
            continue
        r = dict(r, group=group, split=roles[group])
        previous = unique.get(r["identity"])
        if previous and previous["label"] != r["label"]:
            raise ValueError("Conflicting labels for identical text")
        unique.setdefault(r["identity"], r)
    groups = defaultdict(list)
    for r in unique.values():
        if r["split"] == "calibration" and r["label"]:
            continue
        groups[r["split"], r["source"], r["label"]].append(r)
    selected = []
    for (split, source, label), values in sorted(groups.items()):
        limit = train_limit if split == "train" else min(400, evaluation_limit) if split == "validation" else evaluation_limit
        selected.extend(sorted(values, key=lambda r: digest("sample:" + r["identity"]))[:limit])
    return selected


def load_records(data_root, attribution, *, include_modern_families=False):
    from datasets import Dataset

    records = []
    hc3 = data_root / "text_false_positive/corpora/hc3_zh/all.jsonl"
    for line in hc3.open():
        row = json.loads(line)
        group = "hc3:" + digest(row["question"])
        for label, field in ((0, "human_answers"), (1, "chatgpt_answers")):
            for text in row[field] or []:
                records.append(dict(text=text, label=label, group=group, split=partition(group),
                                    source="hc3", domain=row["source"], generator="chatgpt" if label else None))
    for path in (data_root / "text_false_positive/corpora/m4_multilingual").rglob("*.arrow"):
        native = "train" if path.stem.endswith("train") else "validation" if path.stem.endswith("dev") else "test"
        for row in Dataset.from_file(str(path)):
            if row["source"] != "chinese":
                continue
            group = f"m4:{native}:{row['id']}"
            records.append(dict(text=row["text"], label=row["label"], group=group,
                                split=partition(group) if native == "train" else native,
                                source="m4", domain=row.get("domain"), generator=row["model"]))
    for line in attribution.open():
        row = json.loads(line)
        prompt = row["prompt"]
        if prompt["language"] != "zh" or row["route_mismatch"] or row["finish_reason"] != "stop":
            continue
        group = "modern:" + prompt["semantic_group_id"]
        split = partition(group)
        heldout = not include_modern_families and row["family_id"] in {"openai", "claude"}
        if heldout and split != "test":
            continue
        records.append(dict(text=row["text"], label=1, group=group, split=split,
                            source="modern", domain=prompt["domain"], generator=row["response_model"],
                            family=row["family_id"], heldout_generator=heldout))
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--attribution", type=Path, required=True)
    parser.add_argument("--exclude-json", type=Path, help="JSON containing regression text; excluded from every split")
    parser.add_argument("--extra-reports", type=Path, help="Independent generated reports with explicit group and split")
    parser.add_argument("--include-modern-families", action="store_true", help="Train across every available modern family; hold out prompt groups instead")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    excluded = []
    if args.exclude_json:
        excluded.append(digest(" ".join(json.loads(args.exclude_json.read_text(encoding="utf-8"))["text"].split())))
    records = load_records(args.data_root, args.attribution, include_modern_families=args.include_modern_families)
    if args.extra_reports:
        for line in args.extra_reports.open(encoding="utf-8"):
            row = json.loads(line)
            text = re.sub(r"(?m)^#{1,6}\s*", "", row["text"]).replace("**", "")
            records.append(dict(text=text, label=1, group=row["group"], split=row["split"],
                                source="long_reports", domain="academic_technical", generator=row["request_model"],
                                heldout_generator=row["request_model"] == "gpt-5.4"))
    records = prepare(records, excluded=excluded)
    args.output.mkdir(parents=True, exist_ok=False)
    dataset = args.output / "dataset.jsonl"
    dataset.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    manifest = dict(dataset_sha256=digest(dataset.read_text(encoding="utf-8")), excluded_sha256=excluded,
                    counts={str(k): v for k, v in sorted(Counter((r["split"], r["source"], r["label"]) for r in records).items())},
                    heldout_modern_families=[] if args.include_modern_families else ["openai", "claude"],
                    notes=["All available modern families may enter training; prompt groups are held out."
                           if args.include_modern_families else
                           "Legacy ChatGPT examples remain in training; OpenAI is a heldout modern version, not an unseen family."],
                    arguments={k: str(v) for k, v in vars(args).items()})
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
