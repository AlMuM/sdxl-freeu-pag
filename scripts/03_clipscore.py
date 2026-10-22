"""计算每张生成图与其 prompt 的 CLIPScore。

定义与 torchmetrics 一致：CLIPScore = 100 * max(cos(image_emb, text_emb), 0)，
模型默认 openai/clip-vit-large-patch14。prompt 超过 77 token 时会截断（COCO caption 一般很短）。

用法：
    python scripts/03_clipscore.py                 # 给 outputs/images 下所有图打分
    python scripts/03_clipscore.py --dry-run       # 不加载 CLIP，用随机分数测试流程
输出：results/clipscore.csv（run, prompt_id, clipscore），已打过分的图会跳过。
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common import load_config, load_prompts, resolve, results_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.output_dir:
        cfg["common"]["output_dir"] = args.output_dir
    captions = {p["prompt_id"]: p["caption"] for p in load_prompts(cfg)}
    img_root = resolve(cfg["common"]["output_dir"]) / "images"
    out_path = results_dir(cfg) / "clipscore.csv"

    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            done = {(r["run"], r["prompt_id"]) for r in csv.DictReader(f)}

    todo = []
    for run_dir in sorted(p for p in img_root.iterdir() if p.is_dir()):
        for img in sorted(run_dir.glob("*.png")):
            key = (run_dir.name, img.stem)
            if key not in done and img.stem in captions:
                todo.append((key, img))
    print(f"待打分 {len(todo)} 张（已完成 {len(done)}）")
    if not todo:
        return

    if args.dry_run:
        bias = {"baseline": 0.0, "freeu": 0.3, "pag": 0.5}
        def score_batch(batch):
            return [26 + bias.get(k[0], 0.2) + random.Random(f"{k}").gauss(0, 2) for k, _ in batch]
    else:
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CLIPModel.from_pretrained(args.model).to(device).eval()
        processor = CLIPProcessor.from_pretrained(args.model)

        @torch.no_grad()
        def score_batch(batch):
            images = [Image.open(p).convert("RGB") for _, p in batch]
            texts = [captions[k[1]] for k, _ in batch]
            inputs = processor(text=texts, images=images, return_tensors="pt",
                               padding=True, truncation=True).to(device)
            img_emb = model.get_image_features(pixel_values=inputs["pixel_values"])
            txt_emb = model.get_text_features(input_ids=inputs["input_ids"],
                                              attention_mask=inputs["attention_mask"])
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
            return (100 * (img_emb * txt_emb).sum(-1).clamp(min=0)).tolist()

    new = not out_path.exists()
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["run", "prompt_id", "clipscore"])
        for i in range(0, len(todo), args.batch_size):
            batch = todo[i: i + args.batch_size]
            for (key, _), s in zip(batch, score_batch(batch)):
                w.writerow([key[0], key[1], round(s, 4)])
            f.flush()
            print(f"  {min(i + args.batch_size, len(todo))}/{len(todo)}")
    print(f"完成 -> {out_path}")


if __name__ == "__main__":
    main()
