"""Build the offline v2 Chinese corpus from pinned, previously downloaded sources."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import csv
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from scripts.finetune_text_detector import _sha256, load_dataset_rows

SEED = "chinese-text-v2-20261123:"
TOPICS = {0: "体育", 1: "文娱", 5: "教育", 7: "时政", 10: "社会", 12: "财经", 13: "财经"}
HAN = re.compile(r"[\u3400-\u9fff]")
REPORT = re.compile(r"工作报告|调研报告|分析报告|技术报告|新闻稿|新闻报道|通讯稿|公文")
SOURCES = {
    "cred": ("https://github.com/HeraldOfLight/C-ReD/tree/b90072cd218b6ebdbd1d1478ce6e439677f18192/benchmark%20data", "Public benchmark; no standalone dataset license found"),
    "detectrlx": ("https://huggingface.co/datasets/WUJUNCHAO/DetectRL-X/blob/ccc9a0a74793dd3448e0da1650b430f5df1b0402/Binary/binary_general_open.json", "Author repo limits use to non-commercial academic research"),
    "thucnews": ("https://huggingface.co/datasets/joao-luz/thucnews-sample/tree/4560e380dcdb89bc11c4ad8dffd7324e7fd48fc0", "Sina RSS archive; no blanket reuse license asserted"),
    "cnewsum": ("https://huggingface.co/datasets/ethanhao2077/cnewsum-processed/tree/05c98a632bcc73017c08b5bf6fd0abb0ecd75e8a", "Mirror marks MIT; source articles retain their terms"),
    "longwriter": ("https://huggingface.co/datasets/zai-org/LongWriter-6k/blob/0db15c0624f19d63e2efe1021595af933cc5b6cc/long.jsonl", "Dataset card Apache-2.0; responses identified as GPT-4o AgentWrite"),
    "official": ("https://www.gov.cn/", "Publicly published official documents; exact URLs in source metadata"),
}


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def fingerprints(text):
    normalized = "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())
    return digest(normalized), digest(normalized[:160])


def rows_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def size(text):
    return len("".join(text.split()))


def bucket(n):
    if n < 500:
        return "<500"
    if n < 1500:
        return "500–1499"
    return "1500+"


def suitable(text):
    n = size(text)
    return 500 <= n <= 8000 and len(HAN.findall(text)) >= 0.65 * n


def partition(group):
    point = int(digest(SEED + group)[:10], 16) % 100
    return "calibration" if point < 12 else "validation" if point < 28 else "train"


def item(source, key, group, topic, label, text, generator=None, uri=None):
    # Remove external markdown markers without discarding substantive headings or paragraphs.
    body = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", text.strip())
    return dict(text=body, label=label, source=source, source_id=str(key), domain=topic,
                generator=generator, source_url=uri or SOURCES[source][0], group=f"{source}:{group}")


class SplitGuard:
    """Prevent a protected full text or lead from entering any fit partition."""

    def __init__(self, protected):
        pairs = [fingerprints(r["text"]) for r in protected]
        self.protected_full = {full for full, _ in pairs}
        self.protected_lead = {lead for _, lead in pairs}
        self.protected_groups = {r["group"] for r in protected if "group" in r}
        self.used_full = {}
        self.used_lead = {}
        self.groups = {}
        self.excluded = Counter()

    def accept(self, candidates, split):
        keys = [fingerprints(row["text"]) for row in candidates]
        labels = [row["label"] for row in candidates]
        for row, (full, lead) in zip(candidates, keys):
            src = row["source"]
            if row["group"] in self.protected_groups or full in self.protected_full or lead in self.protected_lead:
                self.excluded[src, "protected_full_prefix_or_group"] += 1
                return False
            if full in self.used_full:
                self.excluded[src, "duplicate_full"] += 1
                return False
            previous = {self.used_lead.get(lead), self.groups.get(row["group"])} - {None, split}
            if previous:
                self.excluded[src, "cross_split_prefix_or_group"] += 1
                return False
            if lead in self.used_lead and src in SOURCES:
                self.excluded[src, "repeated_lead"] += 1
                return False
        if len(set(labels)) > 1 and (len({a for a, _ in keys}) != len(keys) or len({b for _, b in keys}) != len(keys)):
            self.excluded[candidates[0]["source"], "paired_copy"] += 1
            return False
        for row, (full, lead) in zip(candidates, keys):
            self.used_full[full] = self.used_lead[lead] = self.groups[row["group"]] = split
        return True


def cred_pairs(base):
    for dir_name, stem, topic_key, title_key in (
        ("news", "news", "news_category", "news_title"),
        ("composition", "composition", None, "composition_title"),
        ("question answer", "question_answer", None, "question_title"),
        ("paper", "paper", None, "paper_title"),
    ):
        folder = base / dir_name
        def read(path):
            with path.open(encoding="utf-8-sig") as handle:
                yield from csv.DictReader(handle)
        originals = {r["id"]: r for r in read(folder/f"CReD_{stem}_human.csv")}
        for file in sorted(folder.glob(f"CReD_{stem}_*.csv")):
            if file.stem.endswith("_human"):
                continue
            for ai in read(file):
                human = originals.get(ai["original_id"])
                if not human or human["attribution"] != "human" or ai["attribution"] == "human":
                    continue
                topic = human[topic_key] if topic_key else {"composition":"考试作文","question_answer":"长评问答","paper":"技术报告"}[stem]
                topic = "时政" if topic == "政治" else "财经" if topic in {"金融", "财政"} else "文娱" if topic == "娱乐" else topic
                group = digest(stem + ":" + "".join(c for c in human[title_key] if c.isalnum()))
                yield [item("cred", f"{stem}:{human['id']}", group, topic, 0, human["text"]),
                       item("cred", f"{stem}:{file.stem}:{ai['id']}", group, topic, 1, ai["text"], ai["attribution"])]


def detectrl_pairs(path):
    import ijson

    with path.open("rb") as handle:
        for index, row in enumerate(ijson.items(handle, "item")):
            if row["lang"] != "chinese" or row["split"] != "train":
                continue
            human, ai = row["human_written_text"], row["llm_generated_text"]
            group = fingerprints(human)[1]
            topic = {"News":"新闻", "Novel":"小说叙事", "SEO":"商业文案", "Wiki":"百科",
                     "Webtext":"博客与长评", "Academic":"技术报告"}.get(row["domain"], row["domain"])
            yield [item("detectrlx", f"{index}:human", group, topic, 0, human),
                   item("detectrlx", f"{index}:ai", group, topic, 1, ai, row["model"])]


def parquet_rows(path):
    import pyarrow.parquet as pq

    for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
        yield from batch.to_pylist()


def additional_rows(args):
    for path in args.thuc:
        for index, row in enumerate(parquet_rows(path)):
            if row["label"] in TOPICS and suitable(row["text"]):
                yield item("thucnews", f"{path.name}:{index}", fingerprints(row["text"])[1],
                           TOPICS[row["label"]], 0, row["text"])
    for row in parquet_rows(args.cnewsum):
        if suitable(row["article"]):
            yield item("cnewsum", row["id"], fingerprints(row["article"])[1], "新闻", 0, row["article"])
    for index, row in enumerate(rows_jsonl(args.longwriter), 1):
        msgs = row["messages"]
        if len(msgs) != 2 or [m["role"] for m in msgs] != ["user", "assistant"]:
            continue
        prompt, text = [m["content"] for m in msgs]
        if len(HAN.findall(prompt)) < 12 or not suitable(text):
            continue
        topic = "正式报告与公文" if REPORT.search(prompt) else "技术报告" if re.search(r"论文|研究报告|技术|实验", prompt) else "长文写作"
        yield item("longwriter", index, fingerprints(prompt)[0], topic, 1, text, "GPT-4o (AgentWrite)")
    if args.official:
        from bs4 import BeautifulSoup

        urls = json.loads((args.official/"urls.json").read_text(encoding="utf-8"))
        for year, url in sorted(urls.items()):
            soup = BeautifulSoup((args.official/f"{year}.html").read_bytes(), "html.parser")
            article = soup.select_one("#BodyLabel, #articleins, .TRS_Editor") or soup
            paragraphs = [p.get_text(" ", strip=True) for p in article.find_all("p")]
            text = "\n".join(p for p in paragraphs if p and len(HAN.findall(p)) >= 15)
            if len(HAN.findall(text)) < 5000:
                continue
            # Both size buckets, grouped as excerpts from one dated source report.
            pos = 0
            for index in range(9):
                width = 1000 if index % 3 == 0 else 2200
                tail = min(len(text), pos + width)
                if tail < len(text):
                    end = text.rfind("。", pos + width // 2, tail)
                    if end >= 0:
                        tail = end + 1
                chunk = text[pos:tail]
                if suitable(chunk):
                    yield item("official", f"{year}:{index}", year, "正式报告与公文", 0, chunk, uri=url)
                pos = tail
                if pos >= len(text):
                    break


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("original", "regression", "cred", "detectrlx", "cnewsum", "longwriter", "output"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--protect", type=Path, action="append", required=True)
    parser.add_argument("--thuc", type=Path, action="append", required=True)
    parser.add_argument("--official", type=Path)
    args = parser.parse_args(argv)

    base = load_dataset_rows(args.original)
    original_test = [r for r in base if r["split"] == "test"]
    protected = [*original_test]
    for path in args.protect:
        protected.extend(rows_jsonl(path))  # Conservatively protect entire supplied files.
    protected.append(json.loads(args.regression.read_text(encoding="utf-8")))
    guard = SplitGuard(protected)
    result = list(original_test)
    for split in ("calibration", "validation", "train"):
        result.extend(r for r in base if r["split"] == split and guard.accept([r], split))

    def add(rows):
        split = partition(rows[0]["group"])
        if not any(split != "calibration" or r["label"] == 0 for r in rows) or not guard.accept(rows, split):
            return False
        for r in rows:
            if split != "calibration" or r["label"] == 0:
                result.append(dict(r, split=split, identity=fingerprints(r["text"])[0]))
        return True

    paired = defaultdict(lambda: defaultdict(list))
    from itertools import chain
    for pair in chain(cred_pairs(args.cred), detectrl_pairs(args.detectrlx)):
        a, b = [size(x["text"]) for x in pair]
        if all(suitable(x["text"]) for x in pair) and bucket(a) == bucket(b) and min(a,b) >= 0.67 * max(a,b):
            human, ai = pair
            paired[human["source"],human["domain"],bucket(a)][ai["generator"]].append(pair)
    for (source, _, _), models in sorted(paired.items()):
        limit = 160 if source == "cred" else 105
        queues = [deque(sorted(m, key=lambda p:digest(SEED+p[1]["source_id"]))) for _,m in sorted(models.items())]
        count = 0
        while count < limit and any(queues):
            for queue in queues:
                while queue and count < limit:
                    if add(queue.popleft()):
                        count += 1
                        break

    unpaired = defaultdict(list)
    for row in additional_rows(args):
        unpaired[row["source"],row["domain"],bucket(size(row["text"]))].append(row)
    for (source, _, _), candidates in sorted(unpaired.items()):
        cap = {"thucnews": 130, "cnewsum": 230, "longwriter": 170, "official": 90}[source]
        count = 0
        for row in sorted(candidates, key=lambda r:digest(SEED+r["source_id"])):
            if add([row]):
                count += 1
                if count >= cap:
                    break

    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output/"dataset.jsonl"
    with path.open("w",encoding="utf-8") as handle:
        for row in result:
            handle.write(json.dumps(row,ensure_ascii=False)+"\n")
    load_dataset_rows(path)
    assert all(
        f not in guard.protected_full and p not in guard.protected_lead and r["group"] not in guard.protected_groups
        for r in result if r["split"] != "test" for f,p in [fingerprints(r["text"])]
    )
    summary = dict(
        dataset_sha256=_sha256(path),
        source_manifest={s:{"source":v[0],"terms_note":v[1]} for s,v in SOURCES.items()},
        input_sha256={str(p):_sha256(p) for p in [args.original,args.regression,*args.protect,*args.thuc,args.detectrlx,args.cnewsum,args.longwriter]},
        extra_file_sha256={str(p):_sha256(p) for p in sorted(args.cred.rglob("*.csv"))},
        protected={"rows":len(protected),"full":len(guard.protected_full),"lead":len(guard.protected_lead)},
        exclusions={" / ".join(k):v for k,v in sorted(guard.excluded.items())},
        splits={split:{"documents":sum(r["split"]==split for r in result),
                       "groups_sha256":digest("\n".join(sorted({r["group"] for r in result if r["split"]==split})))}
                for split in ("train","validation","calibration","test")},
        distributions={" / ".join(map(str,k)):v for k,v in sorted(Counter(
            (r["split"],r["source"],r["label"],r.get("domain") or "general",bucket(size(r["text"]))) for r in result).items())},
        generated_by=dict(Counter(r["generator"] for r in result if r["split"] == "train" and r["label"] == 1 and r.get("generator"))),
        group_split="SHA-256 source + original ID/title/prompt or article lead; 72% train, 16% validation, 12% human calibration",
    )
    if args.official:
        summary["official_sources"]=json.loads((args.official/"urls.json").read_text(encoding="utf-8"))
        summary["official_sha256"]={p.name:_sha256(p) for p in sorted(args.official.glob("*.html"))}
    (args.output/"manifest.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"splits":summary["splits"],"protected":summary["protected"],"exclusions":summary["exclusions"],
                      "generators":summary["generated_by"]},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
