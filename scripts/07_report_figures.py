"""报告用的英文图表（从原始数据重新计算，不手抄数字）。

输出 results/report_figs/：
    fig_metric_vs_human.png   CLIPScore 排名 vs 人工排名 + 相对耗时（斜率图）
    fig_best_worst.png        被选为最好 / 最差的比例（双向条形图）
    fig_failure_tags.png      失败标签占比热力图
    fig_category_rank.png     各类别人工平均名次热力图
    fig_sweeps.png            PAG scale 与 FreeU 强度的 CLIPScore（含 95% CI）

用法：
    python scripts/07_report_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common import ROOT, load_config, records_path, resolve, results_dir  # noqa: E402

MAIN = ["baseline", "freeu", "pag", "freeu_pag"]
NAME = {"baseline": "SDXL", "freeu": "SDXL + FreeU", "pag": "SDXL + PAG", "freeu_pag": "SDXL + FreeU + PAG"}
GREY, BLUE, ORANGE = "#9A9890", "#2F6FDE", "#E2662C"
GOOD, BAD = "#4FA36B", "#D9534F"
TAGS = ["结构/肢体错误", "过饱和/伪影", "与描述不符", "物体缺失", "物体重复"]
TAG_EN = ["Structure /\nlimb error", "Over-\nsaturation /\nartifact", "Mismatch\nwith\ndescription", "Missing\nobject", "Duplicated\nobject"]
CAT_EN = {"person": "Person", "animal": "Animal", "indoor": "Indoor", "outdoor": "Outdoor", "multi_object": "Multi-object"}
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 220, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    means = np.random.default_rng(seed).choice(x, size=(n, len(x))).mean(1)
    return np.percentile(means, [2.5, 97.5])


def human_rows():
    key = json.loads((ROOT / "blind_eval_key.json").read_text())
    rows = []
    for f in sorted((ROOT / "eval_results").glob("results_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for pid, a in d["answers"].items():
            if pid not in key:
                continue
            n = len(key[pid])
            mid = (n + 1) / 2
            for slot, run in key[pid].items():
                tie = bool(a.get("tie"))
                rank = mid if tie else 1 if slot == a.get("best") else n if slot == a.get("worst") else mid
                rows.append({"rater": d.get("rater", f.stem), "prompt_id": pid, "run": run, "rank": rank,
                             "best": int(not tie and slot == a.get("best")),
                             "worst": int(not tie and slot == a.get("worst")),
                             "tags": (a.get("tags") or {}).get(slot, [])})
    return pd.DataFrame(rows)


def fig_metric_vs_human(clip, rec, h, out):
    piv = clip.pivot(index="prompt_id", columns="run", values="clipscore")[MAIN].dropna()
    cs = piv.mean()
    rk = h.groupby("run")["rank"].mean()[MAIN]
    t = rec[rec.run.isin(MAIN)].groupby("run").time_s.mean()
    rel = t / t["baseline"]
    left = list(cs.sort_values(ascending=False).index)
    right = list(rk.sort_values().index)
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    for r in MAIN:
        yl, yr = -left.index(r), -right.index(r)
        c, lw = (BLUE, 2.6) if r == "pag" else (GREY, 1.6)
        ax.plot([0, 1], [yl, yr], color=c, lw=lw, marker="o", zorder=3 if r == "pag" else 2)
        w = "bold" if r == "pag" else "normal"
        ax.text(-0.04, yl, f"{NAME[r]}  {cs[r]:.2f}", ha="right", va="center", fontweight=w)
        ax.text(1.04, yr, f"{rk[r]:.2f}  {NAME[r]}", ha="left", va="center", fontweight=w)
        ax.text(2.35, yr, f"×{rel[r]:.2f}", ha="right", va="center", color="#555")
    ax.text(0, 0.7, "Rank by CLIPScore (↑)", ha="center", fontweight="bold")
    ax.text(1, 0.7, "Rank by human mean rank (↓)", ha="center", fontweight="bold")
    ax.text(2.35, 0.7, "Time", ha="right", fontweight="bold")
    ax.set_xlim(-1.05, 2.4)
    ax.set_ylim(-3.5, 1.0)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "fig_metric_vs_human.png", bbox_inches="tight")
    plt.close(fig)


def fig_best_worst(h, out):
    g = h.groupby("run").agg(best=("best", "mean"), worst=("worst", "mean"), rank=("rank", "mean")).loc[MAIN]
    g = g.sort_values("rank", ascending=False)  # 最好的方法画在最上面
    y = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    ax.barh(y, -g.worst, color=BAD, alpha=0.75, height=0.55, label="chosen as worst")
    ax.barh(y, g.best, color=GOOD, alpha=0.75, height=0.55, label="chosen as best")
    for i, (r, row) in enumerate(g.iterrows()):
        ax.text(-row.worst - 0.01, i, f"{row.worst:.0%}", ha="right", va="center")
        ax.text(row.best + 0.01, i, f"{row.best:.0%}", ha="left", va="center")
    ax.set_yticks(y, [NAME[r] for r in g.index])
    ax.axvline(0, color="#666", lw=1)
    ax.set_xlim(-0.55, 0.5)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{abs(v):.0%}"))
    ax.set_xlabel("share of 180 judgements per setting (3 raters × 60 prompts)")
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / "fig_best_worst.png", bbox_inches="tight")
    plt.close(fig)


def heatmap(ax, mat, rows, cols, fmt, cmap, vmin, vmax, white_above=0.6):
    ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            dark = (v - vmin) / (vmax - vmin) > white_above
            ax.text(j, i, fmt(v), ha="center", va="center", color="white" if dark else "#222",
                    fontweight="bold" if dark else "normal")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def fig_failure_tags(h, out):
    mat = np.array([[h[h.run == r].tags.apply(lambda ts: t in ts).mean() for t in TAGS] for r in MAIN])
    fig, ax = plt.subplots(figsize=(8.0, 2.9))
    heatmap(ax, mat, [NAME[r] for r in MAIN], TAG_EN, lambda v: f"{v:.0%}", "Reds", 0, 0.9)
    ax.xaxis.tick_top()
    fig.tight_layout()
    fig.savefig(out / "fig_failure_tags.png", bbox_inches="tight")
    plt.close(fig)


def fig_category_rank(h, prompts, out):
    d = h.merge(prompts[["prompt_id", "category"]], on="prompt_id")
    cats = ["person", "animal", "indoor", "outdoor", "multi_object"]
    tab = d.groupby(["category", "run"])["rank"].mean().unstack().loc[cats, MAIN]
    counts = d.groupby("category").prompt_id.nunique().loc[cats]
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    heatmap(ax, tab.values, [f"{CAT_EN[c]} (n={counts[c]})" for c in cats], [NAME[r].replace("SDXL + ", "+ ") for r in MAIN],
            lambda v: f"{v:.2f}", "RdYlGn_r", 1.5, 3.5, white_above=0.8)
    ax.xaxis.tick_top()
    fig.tight_layout()
    fig.savefig(out / "fig_category_rank.png", bbox_inches="tight")
    plt.close(fig)


def fig_sweeps(clip, cfg, out):
    piv = clip.pivot(index="prompt_id", columns="run", values="clipscore")
    sub = piv[["baseline", "pag_s1", "pag", "pag_s5", "freeu_weak", "freeu", "freeu_strong"]].dropna()
    stats = {r: (sub[r].mean(), *boot_ci(sub[r])) for r in sub.columns}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    panels = [("PAG scale", ["baseline", "pag_s1", "pag", "pag_s5"], [0, 1, 3, 5], ["0", "1", "3", "5"], BLUE),
              ("FreeU strength", ["baseline", "freeu_weak", "freeu", "freeu_strong"], [0, 1, 2, 3],
               ["off", "weak", "default", "strong"], ORANGE)]
    for ax, (title, runs, xs, labels, c) in zip(axes, panels):
        m = [stats[r][0] for r in runs]
        lo = [stats[r][0] - stats[r][1] for r in runs]
        hi = [stats[r][2] - stats[r][0] for r in runs]
        ax.errorbar(xs, m, yerr=[lo, hi], color=c, marker="o", capsize=3, lw=1.8)
        ax.axhline(stats["baseline"][0], color="#888", ls="--", lw=1)
        ax.set_xticks(xs, labels)
        ax.set_title(title, fontsize=10.5)
    axes[0].set_ylabel("CLIPScore (50-prompt subset)")
    axes[0].text(0.02, 0.04, "dashed: SDXL baseline; bars: 95% CI", transform=axes[0].transAxes, fontsize=8, color="#555")
    fig.tight_layout()
    fig.savefig(out / "fig_sweeps.png", bbox_inches="tight")
    plt.close(fig)


def fig_similarity(sim, prompts, out):
    """左：主实验各设置与原图的相似度；右：子集上随 PAG scale / FreeU 强度的变化。"""
    stat = lambda x: (x.mean(), *boot_ci(x))
    main_runs = ["dune", "pag", "freeu", "freeu_pag"]
    names = {"dune": "SDXL + DUNE", **NAME}
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    for i, r in enumerate(main_runs):
        m, lo, hi = stat(sim[sim.run == r].sim)
        c = {"pag": BLUE, "freeu": ORANGE, "dune": "#7A5AA6"}.get(r, GREY)
        ax.errorbar(m, -i, xerr=[[m - lo], [hi - m]], fmt="o", color=c, capsize=3)
        if hi > 0.97:  # 靠近 1.0 的虚线时，数字放在点的左边
            ax.text(lo - 0.006, -i, f"{m:.3f}", va="center", ha="right", fontsize=9)
        else:
            ax.text(hi + 0.006, -i, f"{m:.3f}", va="center", fontsize=9)
    ax.set_yticks([-i for i in range(len(main_runs))], [names[r] for r in main_runs])
    ax.set_xlim(0.80, 1.01)
    ax.axvline(1.0, color="#888", ls="--", lw=1)
    ax.set_xlabel("CLIP image similarity to SDXL (150 prompts)")
    ax = axes[1]
    subset = set(prompts[prompts.in_subset == 1].prompt_id)
    sub = sim[sim.prompt_id.isin(subset)]
    for runs, xs, c, lab in [(["pag_s1", "pag", "pag_s5"], [1, 3, 5], BLUE, "PAG scale (1, 3, 5)"),
                             (["freeu_weak", "freeu", "freeu_strong"], [1, 3, 5], ORANGE, "FreeU (weak, default, strong)")]:
        ms = [sub[sub.run == r].sim.mean() for r in runs]
        ax.plot([0] + xs, [1.0] + ms, marker="o", color=c, label=lab)
    ax.set_xticks([0, 1, 3, 5], ["off", "1 / weak", "3 / default", "5 / strong"], fontsize=8.5)
    ax.set_ylabel("similarity (50-prompt subset)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_similarity.png", bbox_inches="tight")
    plt.close(fig)


def main():
    cfg = load_config()
    out = results_dir(cfg) / "report_figs"
    out.mkdir(exist_ok=True)
    rec = pd.read_csv(records_path(cfg)).drop_duplicates(["run", "prompt_id"], keep="last")
    clip = pd.read_csv(results_dir(cfg) / "clipscore.csv").drop_duplicates(["run", "prompt_id"], keep="last")
    prompts = pd.read_csv(resolve(cfg["prompts"]["file"]))
    h = human_rows()
    fig_metric_vs_human(clip, rec, h, out)
    fig_best_worst(h, out)
    fig_failure_tags(h, out)
    fig_category_rank(h, prompts, out)
    fig_sweeps(clip, cfg, out)
    sim_path = results_dir(cfg) / "clip_similarity.csv"
    if sim_path.exists():
        fig_similarity(pd.read_csv(sim_path).drop_duplicates(["run", "prompt_id"], keep="last"), prompts, out)
    print("written:", ", ".join(sorted(p.name for p in out.glob("*.png"))))


if __name__ == "__main__":
    main()
