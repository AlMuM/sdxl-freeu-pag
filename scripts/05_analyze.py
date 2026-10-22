"""汇总所有实验结果，生成报告用的表格和图表。

读取：
    outputs/records.csv        生成时间、显存
    results/clipscore.csv      CLIPScore
    data/prompts.csv           类别
    blind_eval_key.json + eval_results/results_*.json   人工盲评（可选）
输出（results/）：
    summary.md                 所有表格（可直接复制进报告）
    table_*.csv                各表 CSV
    fig_*.png                  图表
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common import ROOT, load_config, records_path, resolve, results_dir  # noqa: E402

LABELS = {
    "baseline": "SDXL", "freeu": "SDXL+FreeU", "pag": "SDXL+PAG",
    "freeu_weak": "FreeU (weak)", "freeu_strong": "FreeU (strong)",
    "pag_s1": "PAG s=1", "pag_s5": "PAG s=5", "pag_up": "PAG up_blocks.0",
    "freeu_pag": "FreeU+PAG",
}
COLORS = {"baseline": "#888780", "freeu": "#1D9E75", "pag": "#7F77DD", "freeu_pag": "#D85A30"}
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})


def label(run):
    return LABELS.get(run, run)


def color(run):
    return COLORS.get(run, "#B4B2A9")


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------- 自动指标
def group_table(runs, clip, rec):
    """在同一批 prompt（所有 run 都有分数）上比较。"""
    piv = clip[clip.run.isin(runs)].pivot(index="prompt_id", columns="run", values="clipscore")
    runs = [r for r in runs if r in piv.columns]
    piv = piv[runs].dropna()
    if piv.empty:
        return None, piv
    base = runs[0]
    rows = []
    for r in runs:
        x = piv[r]
        lo, hi = boot_ci(x)
        t = rec[(rec.run == r) & rec.prompt_id.isin(piv.index)]
        row = {
            "method": label(r), "run": r, "n": len(x),
            "CLIPScore": f"{x.mean():.2f}", "95% CI": f"[{lo:.2f}, {hi:.2f}]",
            "Δ vs base": "—" if r == base else f"{(x - piv[base]).mean():+.2f}",
            "Wilcoxon p": "—" if r == base else f"{stats.wilcoxon(x, piv[base]).pvalue:.3g}",
            "time (s/img)": f"{t.time_s.mean():.2f} ± {t.time_s.std():.2f}" if len(t) else "—",
            "rel. time": f"{t.time_s.mean() / rec[(rec.run == base) & rec.prompt_id.isin(piv.index)].time_s.mean():.2f}×" if len(t) else "—",
            "peak mem (GB)": f"{t.peak_mem_gb.max():.2f}" if len(t) and t.peak_mem_gb.notna().any() else "—",
        }
        rows.append(row)
    return pd.DataFrame(rows), piv


def fig_bar_clip(piv, runs, path, title):
    fig, ax = plt.subplots(figsize=(1.4 * len(runs) + 1.5, 3.4))
    means = [piv[r].mean() for r in runs]
    cis = [boot_ci(piv[r]) for r in runs]
    err = np.array([[m - lo for m, (lo, _) in zip(means, cis)], [hi - m for m, (_, hi) in zip(means, cis)]])
    ax.bar([label(r) for r in runs], means, yerr=err, capsize=4, color=[color(r) for r in runs])
    lo = min(c[0] for c in cis)
    hi = max(c[1] for c in cis)
    ax.set_ylim(lo - (hi - lo) * 1.5, hi + (hi - lo) * 0.6)
    for i, m in enumerate(means):
        ax.text(i, m, f"{m:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("CLIPScore (↑)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_time_mem(rec, runs, path):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    t = [rec[rec.run == r].time_s for r in runs]
    axes[0].bar([label(r) for r in runs], [x.mean() for x in t], yerr=[x.std() for x in t],
                capsize=4, color=[color(r) for r in runs])
    axes[0].set_ylabel("seconds / image (↓)")
    axes[0].set_title("Generation time")
    m = [rec[rec.run == r].peak_mem_gb.max() for r in runs]
    axes[1].bar([label(r) for r in runs], m, color=[color(r) for r in runs])
    axes[1].set_ylabel("GB (↓)")
    axes[1].set_title("Peak GPU memory")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_pag_scale(piv, rec, runs, path):
    scales = [float(rec[rec.run == r].pag_scale.iloc[0]) for r in runs]
    order = np.argsort(scales)
    scales = [scales[i] for i in order]
    runs = [runs[i] for i in order]
    clip_m = [piv[r].mean() for r in runs]
    clip_ci = [boot_ci(piv[r]) for r in runs]
    times = [rec[(rec.run == r) & rec.prompt_id.isin(piv.index)].time_s.mean() for r in runs]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.errorbar(scales, clip_m, yerr=[[m - c[0] for m, c in zip(clip_m, clip_ci)],
                                       [c[1] - m for m, c in zip(clip_m, clip_ci)]],
                marker="o", color="#7F77DD", capsize=4, label="CLIPScore")
    ax.set_xlabel("pag_scale")
    ax.set_ylabel("CLIPScore (↑)", color="#534AB7")
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(scales, times, marker="s", ls="--", color="#BA7517", label="time")
    ax2.set_ylabel("seconds / image", color="#854F0B")
    ax.set_xticks(scales)
    ax.set_title("Effect of PAG scale")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_category(clip, prompts, runs, path):
    d = clip[clip.run.isin(runs)].merge(prompts[["prompt_id", "category"]], on="prompt_id")
    tab = d.groupby(["category", "run"]).clipscore.mean().unstack()[runs]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    w = 0.8 / len(runs)
    x = np.arange(len(tab))
    for i, r in enumerate(runs):
        ax.bar(x + i * w - 0.4 + w / 2, tab[r], w, label=label(r), color=color(r))
    ax.set_xticks(x, tab.index)
    lo, hi = np.nanmin(tab.values), np.nanmax(tab.values)
    ax.set_ylim(lo - (hi - lo) * 1.2, hi + (hi - lo) * 0.4)
    ax.set_ylabel("CLIPScore")
    ax.set_title("CLIPScore by prompt category")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return tab.round(2).reset_index()


# ---------------------------------------------------------------- 人工评价
def load_human(cfg):
    key_path = ROOT / "blind_eval_key.json"
    files = sorted((ROOT / "eval_results").glob("results_*.json"))
    if not key_path.exists() or not files:
        return None
    key = json.loads(key_path.read_text())
    rows = []
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        rater = data.get("rater", f.stem)
        for pid, a in data["answers"].items():
            if pid not in key:
                continue
            complete = a.get("tie") or (a.get("best") and a.get("worst") and a["best"] != a["worst"])
            if not complete:
                continue
            n = len(key[pid])
            mid = (n + 1) / 2  # 未被选为最好/最差的图并列中间名次
            for slot, run in key[pid].items():
                if a.get("tie"):
                    rank = mid
                elif slot == a["best"]:
                    rank = 1.0
                elif slot == a["worst"]:
                    rank = float(n)
                else:
                    rank = mid
                rows.append({"rater": rater, "prompt_id": pid, "run": run, "rank": rank,
                             "best": int(not a.get("tie") and slot == a["best"]),
                             "worst": int(not a.get("tie") and slot == a["worst"]),
                             "tie": int(bool(a.get("tie"))),
                             "tags": a.get("tags", {}).get(slot, [])})
    return pd.DataFrame(rows) if rows else None


def fleiss_kappa(table: np.ndarray) -> float:
    """table: items × categories，元素为该类别被选中的评审数。每行总数相同。"""
    n = table.sum(1)[0]
    p_j = table.sum(0) / table.sum()
    p_i = ((table ** 2).sum(1) - n) / (n * (n - 1))
    p_bar, p_e = p_i.mean(), (p_j ** 2).sum()
    return float((p_bar - p_e) / (1 - p_e)) if p_e < 1 else float("nan")


def human_tables(h, runs):
    n_raters = h.rater.nunique()
    per_run = []
    for r in runs:
        d = h[h.run == r]
        per_run.append({"method": label(r), "run": r, "judgments": len(d),
                        "best rate": f"{d.best.mean():.1%}", "worst rate": f"{d.worst.mean():.1%}",
                        f"mean rank (1–{len(runs)}, ↓)": f"{d['rank'].mean():.2f}"})
    per_run = pd.DataFrame(per_run)

    # 两两胜率：同一评审、同一 prompt 内，名次更靠前即为胜；平局不计入分母
    piv = h.pivot_table(index=["rater", "prompt_id"], columns="run", values="rank")
    pair_rows = []
    for a, b in itertools.combinations(runs, 2):
        diff = piv[a] - piv[b]
        wins, losses, ties = (diff < 0).sum(), (diff > 0).sum(), (diff == 0).sum()
        p = stats.binomtest(int(wins), int(wins + losses)).pvalue if wins + losses else float("nan")
        pair_rows.append({"pair": f"{label(a)} vs {label(b)}", "A wins": int(wins), "B wins": int(losses),
                          "ties": int(ties),
                          "A win rate (excl. ties)": f"{wins / (wins + losses):.1%}" if wins + losses else "—",
                          "sign test p": f"{p:.3g}"})
    pairs = pd.DataFrame(pair_rows)

    # 评审一致性：对“最好”的选择（含平局类别）计算 Fleiss' kappa
    choice = (h[h.best == 1].set_index(["rater", "prompt_id"]).run
              .reindex(piv.index).fillna("tie").unstack("rater"))
    choice = choice.dropna()
    kappa = float("nan")
    if n_raters >= 2 and len(choice):
        cats = runs + ["tie"]
        table = np.array([[(row == c).sum() for c in cats] for _, row in choice.iterrows()])
        kappa = fleiss_kappa(table)

    by_rater = (h.groupby(["rater", "run"]).best.mean().unstack()[runs]
                .map(lambda v: f"{v:.0%}").rename(columns=label).reset_index())

    tag_rows = []
    for r in runs:
        tags = pd.Series([t for ts in h[h.run == r].tags for t in ts], dtype=object)
        counts = tags.value_counts()
        tag_rows.append({"method": label(r), **{k: int(v) for k, v in counts.items()},
                         "any failure (% judgments)": f"{(h[h.run == r].tags.str.len() > 0).mean():.1%}"})
    tags = pd.DataFrame(tag_rows).fillna(0)
    return per_run, pairs, kappa, len(choice), by_rater, tags


def fig_human(h, runs, path):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    best = [h[h.run == r].best.mean() for r in runs]
    worst = [h[h.run == r].worst.mean() for r in runs]
    x = np.arange(len(runs))
    ax.bar(x - 0.2, best, 0.4, label="chosen as best", color=[color(r) for r in runs])
    ax.bar(x + 0.2, worst, 0.4, label="chosen as worst", color=[color(r) for r in runs], alpha=0.35,
           hatch="//", edgecolor="#444")
    for i, (b, w) in enumerate(zip(best, worst)):
        ax.text(i - 0.2, b, f"{b:.0%}", ha="center", va="bottom", fontsize=8)
        ax.text(i + 0.2, w, f"{w:.0%}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [label(r) for r in runs])
    ax.set_ylabel("share of judgments")
    ax.set_title(f"Blind human preference ({h.rater.nunique()} raters)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.output_dir:
        cfg["common"]["output_dir"] = args.output_dir
    out = results_dir(cfg)

    rec = pd.read_csv(records_path(cfg)).drop_duplicates(["run", "prompt_id"], keep="last")
    clip = pd.read_csv(out / "clipscore.csv").drop_duplicates(["run", "prompt_id"], keep="last")
    prompts = pd.read_csv(resolve(cfg["prompts"]["file"]))

    md = ["# Experiment summary", ""]
    if rec.dry_run.any():
        md += ["> ⚠️ 包含 DRY RUN 占位数据，结果没有意义，只用于测试流程。", ""]
    env = resolve(cfg["common"]["output_dir"]) / "env.json"
    if env.exists():
        e = json.loads(env.read_text())
        c = e["common"]
        md += [f"Setup: `{c['model_id']}`, {c['width']}×{c['height']}, {c['num_inference_steps']} steps, "
               f"CFG {c['guidance_scale']}, scheduler {c['scheduler']}, {c['dtype']}, "
               f"device {e.get('gpu', e['device'])}, torch {e.get('torch', '-')}, diffusers {e.get('diffusers', '-')}", ""]

    titles = {"main": "Main comparison", "freeu_sweep": "FreeU hyper-parameters",
              "pag_sweep": "PAG scale", "pag_layers": "PAG applied layers", "combo": "FreeU + PAG"}
    for g, runs in cfg["groups"].items():
        tab, piv = group_table(runs, clip, rec)
        if tab is None or len(tab) < 2:
            print(f"跳过分组 {g}：数据不足")
            continue
        runs = list(tab.run)
        tab.drop(columns="run").to_csv(out / f"table_{g}.csv", index=False)
        md += [f"## {titles.get(g, g)} (`{g}`, n={len(piv)} prompts)", "", md_table(tab.drop(columns="run")), ""]
        fig_bar_clip(piv, runs, out / f"fig_clipscore_{g}.png", titles.get(g, g))
        if g == "main":
            fig_time_mem(rec[rec.prompt_id.isin(piv.index)], runs, out / "fig_time_memory.png")
            cat = fig_category(clip[clip.prompt_id.isin(piv.index)], prompts, runs, out / "fig_category.png")
            cat.columns = ["category"] + [label(r) for r in cat.columns[1:]]
            cat.to_csv(out / "table_category.csv", index=False)
            md += ["### CLIPScore by category", "", md_table(cat), ""]
        if g == "pag_sweep":
            fig_pag_scale(piv, rec, runs, out / "fig_pag_scale.png")

    h = load_human(cfg)
    if h is None:
        md += ["## Human evaluation", "", "（还没有盲评结果：把 results_*.json 放进 eval_results/ 后重跑本脚本）", ""]
    else:
        runs = cfg["blind_eval"]["runs"]
        per_run, pairs, kappa, n_items, by_rater, tags = human_tables(h, runs)
        for name, t in [("human_overall", per_run), ("human_pairwise", pairs),
                        ("human_by_rater", by_rater), ("failure_tags", tags)]:
            t.to_csv(out / f"table_{name}.csv", index=False)
        fig_human(h, runs, out / "fig_human_preference.png")
        md += [f"## Human evaluation ({h.rater.nunique()} raters: {', '.join(sorted(h.rater.unique()))}; "
               f"{h.prompt_id.nunique()} prompts)", "",
               md_table(per_run.drop(columns="run")), "",
               "### Pairwise preference (derived from best/worst ranking)", "", md_table(pairs), "",
               f"Inter-rater agreement on the 'best' choice: Fleiss' κ = {kappa:.3f} "
               f"(over {n_items} prompts rated by all raters; <0.2 slight, 0.2–0.4 fair, 0.4–0.6 moderate)", "",
               "### Best-choice rate by rater", "", md_table(by_rater), "",
               "### Failure tags (count of judgments)", "", md_table(tags), ""]

    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\n表格与图表已写入 {out}")


if __name__ == "__main__":
    main()
