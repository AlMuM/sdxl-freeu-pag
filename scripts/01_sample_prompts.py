"""从 MS-COCO 2017 val captions 中按类别分层抽样 prompt，并为每条 prompt 固定随机种子。

用法：
    python scripts/01_sample_prompts.py --n 150 --subset 50
输出：data/prompts.csv（prompt_id, image_id, category, caption, seed, in_subset）
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common import ROOT  # noqa: E402

COCO_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
CAPTIONS_MEMBER = "annotations/captions_val2017.json"

# 简单关键词分类，按优先级从上到下匹配。
CATEGORY_KEYWORDS = {
    "person": r"\b(man|men|woman|women|person|people|boy|girl|child|children|kid|kids|player|guy|lady|skier|surfer|skateboarder)\b",
    "animal": r"\b(dog|dogs|cat|cats|horse|horses|cow|cows|sheep|bird|birds|elephant|elephants|giraffe|giraffes|zebra|zebras|bear|bears)\b",
    "indoor": r"\b(kitchen|bathroom|bedroom|living room|table|room|desk|sink|toilet|couch|bed|refrigerator|oven)\b",
    "outdoor": r"\b(street|road|beach|field|park|city|mountain|snow|ocean|sky|river|building|train|bus)\b",
}
# 至少包含两个不同物体名词的描述视为 multi_object
OBJECT_WORDS = r"\b(dog|cat|horse|cow|sheep|bird|elephant|giraffe|zebra|bear|car|bus|train|truck|bike|bicycle|motorcycle|boat|plane|airplane|umbrella|chair|table|bench|laptop|phone|clock|vase|bottle|cup|bowl|pizza|cake|sandwich|banana|apple|orange|broccoli|kite|frisbee|ball|bag|book|tv|teddy bear)\b"


def categorize(caption: str) -> str:
    text = caption.lower()
    objs = set(re.findall(OBJECT_WORDS, text))
    if len(objs) >= 2:
        return "multi_object"
    for cat, pattern in CATEGORY_KEYWORDS.items():
        if re.search(pattern, text):
            return cat
    return "other"


def load_captions(path: Path | None) -> list[dict]:
    cache = ROOT / "data" / "captions_val2017.json"
    if path is None:
        path = cache
        if not cache.exists():
            print(f"下载 COCO annotations（约 240MB）: {COCO_ZIP_URL}")
            with urllib.request.urlopen(COCO_ZIP_URL) as resp:
                data = resp.read()
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_bytes(zf.read(CAPTIONS_MEMBER))
    with open(path, encoding="utf-8") as f:
        return json.load(f)["annotations"]


def clean(caption: str) -> str:
    caption = " ".join(caption.strip().split())
    return caption.rstrip(".") + "."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150, help="总 prompt 数（建议 100–200）")
    ap.add_argument("--subset", type=int, default=50, help="超参数实验子集大小")
    ap.add_argument("--seed", type=int, default=2024, help="抽样随机种子")
    ap.add_argument("--captions", type=Path, default=None, help="已下载的 captions_val2017.json")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "prompts.csv")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    anns = load_captions(args.captions)

    # 每张图片只取一条 caption，避免同一张图出现多次
    by_image: dict[int, list[str]] = defaultdict(list)
    for a in anns:
        by_image[a["image_id"]].append(a["caption"])
    pool = defaultdict(list)
    for image_id in sorted(by_image):
        cap = clean(rng.choice(by_image[image_id]))
        if 6 <= len(cap.split()) <= 25:
            pool[categorize(cap)].append((image_id, cap))

    cats = [c for c in ["person", "animal", "indoor", "outdoor", "multi_object"] if pool[c]]
    per_cat = args.n // len(cats)
    chosen = []
    for c in cats:
        rng.shuffle(pool[c])
        chosen += [(c, *x) for x in pool[c][:per_cat]]
    # 余数从 other + 剩余样本中补齐
    rest = [(c, *x) for c in pool for x in pool[c][per_cat if c in cats else 0:]]
    rng.shuffle(rest)
    chosen += rest[: args.n - len(chosen)]

    # 子集按类别轮流取，保证类别均衡
    rng.shuffle(chosen)
    by_cat = defaultdict(list)
    for r in chosen:
        by_cat[r[0]].append(r[1])
    subset_ids = set()
    target = min(args.subset, len(chosen))
    while len(subset_ids) < target:
        for c in by_cat:
            if by_cat[c] and len(subset_ids) < target:
                subset_ids.add(by_cat[c].pop(0))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["prompt_id", "image_id", "category", "caption", "seed", "in_subset"])
        for i, (cat, image_id, cap) in enumerate(chosen):
            # seed 由抽样种子派生，可复现
            w.writerow([f"p{i:03d}", image_id, cat, cap, rng.randrange(2**31), int(image_id in subset_ids)])

    counts = defaultdict(int)
    for c, *_ in chosen:
        counts[c] += 1
    print(f"写入 {len(chosen)} 条 prompt -> {args.out}")
    print("类别分布:", dict(counts), "| 子集:", len(subset_ids))


if __name__ == "__main__":
    main()
