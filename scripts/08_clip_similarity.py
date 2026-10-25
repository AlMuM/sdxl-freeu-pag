"""计算每种设置的图片与同一 prompt、同一 seed 的原始 SDXL 图片之间的 CLIP 图像相似度。

定义与 DUNE 论文（Lim et al., 2026）的 "CLIP similarity" 一致：两张图的 CLIP 图像向量余弦相似度。
它衡量增强方法把原图改动了多少（1.0 = 与原图语义完全相同），可以和 CLIPScore 互补：
论文指出过饱和的图可能抬高图文 CLIPScore，而图图相似度不受这个影响。

用法：
    python scripts/08_clip_similarity.py
输出：results/clip_similarity.csv（run, prompt_id, sim），已算过的会跳过。
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common import resolve, load_config, results_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    cfg = load_config(args.config)
    img_root = resolve(cfg["common"]["output_dir"]) / "images"
    out_path = results_dir(cfg) / "clip_similarity.csv"
    base_dir = img_root / "baseline"

    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            done = {(r["run"], r["prompt_id"]) for r in csv.DictReader(f)}
    todo = [(d.name, p.stem, p) for d in sorted(img_root.iterdir()) if d.is_dir() and d.name != "baseline"
            for p in sorted(d.glob("*.png"))
            if (d.name, p.stem) not in done and (base_dir / p.name).exists()]
    print(f"待计算 {len(todo)} 张（已完成 {len(done)}）")
    if not todo:
        return

    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(args.model).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.model)

    @torch.no_grad()
    def embed(paths):
        images = [Image.open(p).convert("RGB") for p in paths]
        e = model.get_image_features(**processor(images=images, return_tensors="pt").to(device))
        return e / e.norm(dim=-1, keepdim=True)

    new = not out_path.exists()
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["run", "prompt_id", "sim"])
        for i in range(0, len(todo), args.batch_size):
            batch = todo[i: i + args.batch_size]
            a = embed([p for _, _, p in batch])
            b = embed([base_dir / p.name for _, _, p in batch])
            for (run, pid, _), s in zip(batch, (a * b).sum(-1).tolist()):
                w.writerow([run, pid, round(s, 5)])
            f.flush()
            print(f"  {min(i + args.batch_size, len(todo))}/{len(todo)}")
    print(f"完成 -> {out_path}")


if __name__ == "__main__":
    main()
