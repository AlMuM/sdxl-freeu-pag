"""打包人工盲评材料。

对每条 prompt，把 configs 中 blind_eval.runs 的各组图片随机打乱成 A/B/C/D，
复制到 blind_eval/（发给评审），映射关系写到 blind_eval_key.json（不要发给评审）。

用法：
    python scripts/04_build_blind_eval.py
    然后把 blind_eval/ 整个文件夹压缩发给三位评审，各自双击 index.html 评分。
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common import ROOT, image_path, load_config, load_prompts  # noqa: E402

LETTERS = "ABCDEF"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--size", type=int, default=768, help="评审图片边长（压缩体积）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.output_dir:
        cfg["common"]["output_dir"] = args.output_dir
    runs = cfg["blind_eval"]["runs"]
    assert 2 <= len(runs) <= len(LETTERS), f"盲评支持 2–{len(LETTERS)} 组"
    slots = list(LETTERS[: len(runs)])
    n = cfg["blind_eval"]["num_prompts"]

    prompts = load_prompts(cfg)
    ready = [p for p in prompts if all(image_path(cfg, r, p["prompt_id"]).exists() for r in runs)]
    if len(ready) < n:
        print(f"注意：只有 {len(ready)} 条 prompt 三组图齐全，少于配置的 {n} 条")
    rng = random.Random(args.seed)
    chosen = rng.sample(ready, min(n, len(ready)))

    out_dir = ROOT / "blind_eval"
    img_dir = out_dir / "images"
    if img_dir.exists():
        shutil.rmtree(img_dir)
    img_dir.mkdir(parents=True)

    items, key = [], {}
    for p in chosen:
        order = runs[:]
        rng.shuffle(order)
        key[p["prompt_id"]] = dict(zip(slots, order))
        images = {}
        for slot, run in zip(slots, order):
            name = f"{p['prompt_id']}_{slot}.jpg"
            im = Image.open(image_path(cfg, run, p["prompt_id"])).convert("RGB")
            im.thumbnail((args.size, args.size))
            im.save(img_dir / name, quality=90)
            images[slot] = f"images/{name}"
        items.append({"prompt_id": p["prompt_id"], "caption": p["caption"], "images": images})

    (out_dir / "manifest.js").write_text(
        "window.MANIFEST = " + json.dumps({"slots": slots, "items": items}, ensure_ascii=False, indent=1) + ";\n",
        encoding="utf-8")
    shutil.copy(ROOT / "src" / "blind_eval.html", out_dir / "index.html")
    key_path = ROOT / "blind_eval_key.json"
    key_path.write_text(json.dumps(key, indent=1), encoding="utf-8")

    print(f"已打包 {len(items)} 条 prompt -> {out_dir}")
    print(f"答案映射 -> {key_path}（不要发给评审）")
    print("下一步：压缩 blind_eval/ 发给三位评审；收回的 results_*.json 放到 eval_results/")


if __name__ == "__main__":
    main()
