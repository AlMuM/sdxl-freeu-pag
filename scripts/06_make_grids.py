"""生成并排对比图（报告 / slides 用）。

自动挑选方案中建议的五类案例（基于 CLIPScore，若有盲评结果则结合人工排名）：
    big_diff        各方法差异最大
    similar         各方法最接近
    <方法>_better   该增强方法明显更好（freeu / pag / freeu_pag）
    all_fail        所有增强方法都失败
再为 freeu_sweep / pag_sweep 生成参数网格。

用法：
    python scripts/06_make_grids.py
    python scripts/06_make_grids.py --ids p003 p017 p042 --name my_pick   # 手动指定
注意：自动挑选只是起点，最终放进报告的案例请组员人工确认。
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common import ROOT, image_path, load_config, resolve, results_dir  # noqa: E402

LABELS = {"baseline": "SDXL Base", "freeu": "SDXL + FreeU", "pag": "SDXL + PAG",
          "freeu_weak": "FreeU weak", "freeu_strong": "FreeU strong",
          "pag_s1": "PAG s=1", "pag_s5": "PAG s=5", "pag_up": "PAG up_blocks.0", "freeu_pag": "FreeU + PAG", "dune": "SDXL + DUNE"}


def _font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name), size)


def grid(cfg, runs, ids, captions, path, col_titles=None, thumb=512, gap=12):
    """列 = 方法，行 = prompt；每行上方写 prompt 文本。"""
    ids = [i for i in ids if all(image_path(cfg, r, i).exists() for r in runs)]
    if not ids:
        print(f"跳过 {path.name}：没有可用图片")
        return
    head_font, cap_font = _font(26, bold=True), _font(20)
    width = len(runs) * thumb + (len(runs) + 1) * gap
    wrap = max(20, width // 11)
    caps = ["\n".join(textwrap.wrap(f'{pid}: "{captions[pid]}"', wrap)) for pid in ids]
    cap_h = [24 * (c.count("\n") + 1) + 10 for c in caps]
    head_h = 48
    height = head_h + sum(cap_h) + len(ids) * (thumb + gap) + gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for j, r in enumerate(runs):
        x = gap + j * (thumb + gap)
        title = (col_titles or {}).get(r, LABELS.get(r, r))
        tw = draw.textlength(title, font=head_font)
        draw.text((x + (thumb - tw) / 2, 10), title, fill="black", font=head_font)
    y = head_h
    for pid, cap, ch in zip(ids, caps, cap_h):
        draw.multiline_text((gap, y + 4), cap, fill=(60, 60, 60), font=cap_font, spacing=4)
        y += ch
        for j, r in enumerate(runs):
            im = Image.open(image_path(cfg, r, pid)).convert("RGB").resize((thumb, thumb), Image.LANCZOS)
            canvas.paste(im, (gap + j * (thumb + gap), y))
        y += thumb + gap
    canvas.save(path)
    print(f"写入 {path}")


def human_ranks(cfg):
    key_path, files = ROOT / "blind_eval_key.json", sorted((ROOT / "eval_results").glob("results_*.json"))
    if not key_path.exists() or not files:
        return None
    key = json.loads(key_path.read_text())
    rows = []
    for f in files:
        for pid, a in json.loads(f.read_text(encoding="utf-8"))["answers"].items():
            if pid not in key or not (a.get("tie") or (a.get("best") and a.get("worst"))):
                continue
            n = len(key[pid])
            mid = (n + 1) / 2
            for slot, run in key[pid].items():
                rank = mid if a.get("tie") else 1 if slot == a["best"] else n if slot == a["worst"] else mid
                rows.append({"prompt_id": pid, "run": run, "rank": rank, "fail": len(a.get("tags", {}).get(slot, []))})
    return pd.DataFrame(rows).groupby(["prompt_id", "run"]).mean().unstack() if rows else None


def pick_cases(clip, h, runs, k):
    """runs[0] 是 baseline，其余是增强方法；为每种增强方法挑出它明显更好的案例。"""
    enhanced = runs[1:]
    piv = clip.pivot(index="prompt_id", columns="run", values="clipscore")[runs].dropna()
    rng = piv.max(axis=1) - piv.min(axis=1)
    cases = {"big_diff": rng.nlargest(k).index, "similar": rng.nsmallest(k).index}
    for r in enhanced:
        others = [o for o in runs if o != r]
        cases[f"{r}_better"] = (piv[r] - piv[others].max(axis=1)).nlargest(k).index
    cases["all_fail"] = piv[enhanced].mean(axis=1).nsmallest(k).index
    if h is not None:  # 有人工评价时，用人工名次覆盖“某方法更好”和“都失败”
        rank, fail = h["rank"], h["fail"]
        common = rank.dropna().index
        rank, fail = rank.loc[common], fail.loc[common]
        rated = [r for r in runs if r in rank.columns]  # 人工只评了部分方法（如没有 DUNE）
        for r in rated[1:]:
            others = [o for o in rated if o != r]
            cases[f"{r}_better"] = (rank[others].min(axis=1) - rank[r]).nlargest(k).index
        cases["all_fail"] = fail[rated[1:]].sum(axis=1).nlargest(k).index
    return {name: list(ids) for name, ids in cases.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--k", type=int, default=3, help="每类案例的 prompt 数")
    ap.add_argument("--ids", nargs="*", help="手动指定 prompt_id")
    ap.add_argument("--name", default="custom")
    ap.add_argument("--runs", nargs="*", default=None, help="默认使用 groups.main")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.output_dir:
        cfg["common"]["output_dir"] = args.output_dir
    out = results_dir(cfg) / "grids"
    out.mkdir(exist_ok=True)
    prompts = pd.read_csv(resolve(cfg["prompts"]["file"]))
    captions = dict(zip(prompts.prompt_id, prompts.caption))

    main_runs = args.runs or cfg["groups"]["main"]
    if args.ids:
        grid(cfg, main_runs, args.ids, captions, out / f"grid_{args.name}.png")
        return

    clip = pd.read_csv(results_dir(cfg) / "clipscore.csv").drop_duplicates(["run", "prompt_id"], keep="last")
    cases = pick_cases(clip, human_ranks(cfg), main_runs, args.k)
    for name, ids in cases.items():
        grid(cfg, main_runs, ids, captions, out / f"grid_{name}.png")
    (out / "picked_cases.json").write_text(json.dumps(cases, indent=1))

    subset = list(prompts[prompts.in_subset == 1].prompt_id)
    show = subset[:: max(1, len(subset) // args.k)][: args.k]
    for g in ["freeu_sweep", "pag_sweep", "pag_layers"]:
        runs = cfg["groups"].get(g)
        if runs:
            titles = None
            if g == "pag_sweep":
                titles = {r: f"pag_scale = {cfg['runs'][r]['pag_scale']:g}" for r in runs}
            grid(cfg, runs, show, captions, out / f"grid_{g}.png", col_titles=titles)
    print(f"自动挑选的案例 -> {out / 'picked_cases.json'}（请人工复核）")


if __name__ == "__main__":
    main()
