"""用同一个 SDXL pipeline 依次生成 baseline / FreeU / PAG 等各组图片。

- 只加载一次模型：PAG 版 pipeline 在 pag_scale=0 时与原始 SDXL 完全一致，
  FreeU 通过 enable_freeu / disable_freeu 在 U-Net 上开关。
- 每条 prompt 使用 prompts.csv 中固定的 seed，所有 run 相同。
- 已存在的图片会跳过，Colab 断线后重跑同一命令即可续跑。
- 每张图记录生成时间和峰值显存，写入 outputs/records.csv。

用法：
    python scripts/02_generate.py --group main            # 主实验三组
    python scripts/02_generate.py --runs freeu_weak freeu_strong
    python scripts/02_generate.py --group main --limit 10 # 冒烟测试
    python scripts/02_generate.py --group main --dry-run  # 不加载模型，生成占位图测试流程
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common import (  # noqa: E402
    image_path,
    load_config,
    load_prompts,
    prompts_for_run,
    records_path,
    resolve,
    run_settings,
)

RECORD_FIELDS = [
    "run", "prompt_id", "seed", "time_s", "peak_mem_gb", "freeu", "pag_scale",
    "pag_applied_layers", "steps", "guidance_scale", "height", "width",
    "device", "dry_run", "timestamp",
]


def build_pipeline(cfg: dict, cpu_offload: bool):
    import torch
    from diffusers import (
        AutoPipelineForText2Image,
        DPMSolverMultistepScheduler,
        EulerDiscreteScheduler,
    )

    common = cfg["common"]
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[common["dtype"]]
    kwargs = dict(torch_dtype=dtype, use_safetensors=True)
    if common["dtype"] == "fp16":
        kwargs["variant"] = "fp16"
    pipe = AutoPipelineForText2Image.from_pretrained(
        common["model_id"],
        enable_pag=True,
        pag_applied_layers=common["pag_applied_layers"],
        **kwargs,
    )
    if common["scheduler"] == "euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    elif common["scheduler"] == "dpm++":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    else:
        raise ValueError(f"未知 scheduler: {common['scheduler']}")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    if cpu_offload and device == "cuda":
        pipe.enable_model_cpu_offload()  # 显存不足时用；计时会变慢，报告里要注明
    else:
        pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe, device


def apply_settings(pipe, s: dict):
    if s["freeu"]:
        pipe.enable_freeu(**s["freeu"])
    else:
        pipe.disable_freeu()
    pipe.set_pag_applied_layers(s["pag_applied_layers"])


def generate_one(pipe, device: str, prompt: str, seed: int, s: dict):
    import torch

    gen_device = "cpu" if device == "mps" else device
    generator = torch.Generator(gen_device).manual_seed(seed)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    image = pipe(
        prompt=prompt,
        height=s["height"],
        width=s["width"],
        num_inference_steps=s["num_inference_steps"],
        guidance_scale=s["guidance_scale"],
        pag_scale=s["pag_scale"],
        generator=generator,
    ).images[0]
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else float("nan")
    return image, elapsed, peak


def dry_run_image(run: str, prompt_id: str, seed: int, s: dict):
    """不加载模型时生成的占位图（随机色块 + 文字），只用来测试整条流程。"""
    from PIL import Image, ImageDraw

    rng = random.Random(f"{run}-{seed}")
    img = Image.new("RGB", (256, 256), tuple(rng.randrange(256) for _ in range(3)))
    d = ImageDraw.Draw(img)
    for _ in range(12):
        x, y = rng.randrange(256), rng.randrange(256)
        d.ellipse([x, y, x + rng.randrange(20, 80), y + rng.randrange(20, 80)],
                  fill=tuple(rng.randrange(256) for _ in range(3)))
    d.text((8, 8), f"DRY RUN\n{prompt_id}", fill=(255, 255, 255))
    base = 6.0 * (1 + 0.5 * (s["pag_scale"] > 0)) * (1.02 if s["freeu"] else 1.0)
    return img, base * rng.uniform(0.95, 1.05), float("nan")


def append_record(path: Path, row: dict):
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RECORD_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def save_env(cfg: dict, device: str, dry_run: bool):
    info = {"time": datetime.now().isoformat(timespec="seconds"), "device": device,
            "dry_run": dry_run, "python": platform.python_version(), "common": cfg["common"]}
    if not dry_run:
        import diffusers
        import torch
        info.update(torch=torch.__version__, diffusers=diffusers.__version__)
        if device == "cuda":
            info["gpu"] = torch.cuda.get_device_name(0)
    out = resolve(cfg["common"]["output_dir"]) / "env.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print("环境信息:", json.dumps({k: v for k, v in info.items() if k != "common"}, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--group", action="append", default=[], help="configs 里 groups 的名字，可重复")
    ap.add_argument("--runs", nargs="*", default=[], help="直接指定 run 名")
    ap.add_argument("--limit", type=int, default=None, help="每个 run 只跑前 N 条（冒烟测试）")
    ap.add_argument("--output-dir", default=None, help="覆盖输出目录，例如挂载的 Google Drive")
    ap.add_argument("--cpu-offload", action="store_true", help="显存不足时启用（会影响计时）")
    ap.add_argument("--dry-run", action="store_true", help="不加载模型，生成占位图测试流程")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.output_dir:
        cfg["common"]["output_dir"] = args.output_dir

    runs = list(args.runs)
    for g in args.group:
        runs += cfg["groups"][g]
    runs = list(dict.fromkeys(runs)) or list(cfg["runs"])  # 去重并保持顺序
    unknown = [r for r in runs if r not in cfg["runs"]]
    if unknown:
        raise SystemExit(f"configs 中没有这些 run: {unknown}")

    prompts = load_prompts(cfg)
    rec_path = records_path(cfg)

    if args.dry_run:
        pipe, device = None, "dry-run"
    else:
        pipe, device = build_pipeline(cfg, args.cpu_offload)
    save_env(cfg, device, args.dry_run)

    warmed_up = args.dry_run
    for run in runs:
        s = run_settings(cfg, run)
        todo = prompts_for_run(cfg, run, prompts)[: args.limit]
        todo = [p for p in todo if not image_path(cfg, run, p["prompt_id"]).exists()]
        print(f"\n== {run}: freeu={s['freeu']} pag_scale={s['pag_scale']} "
              f"layers={s['pag_applied_layers']} | 待生成 {len(todo)} 张")
        if not todo:
            continue
        if pipe is not None:
            apply_settings(pipe, s)
        if not warmed_up:
            # 第一次调用包含 CUDA 初始化等开销，不计入计时
            print("预热中...")
            generate_one(pipe, device, "a photo of a cat", 0, {**s, "num_inference_steps": 2})
            warmed_up = True

        t_run = time.time()
        for i, p in enumerate(todo, 1):
            if args.dry_run:
                image, elapsed, peak = dry_run_image(run, p["prompt_id"], p["seed"], s)
            else:
                image, elapsed, peak = generate_one(pipe, device, p["caption"], p["seed"], s)
            out = image_path(cfg, run, p["prompt_id"])
            out.parent.mkdir(parents=True, exist_ok=True)
            image.save(out)
            append_record(rec_path, {
                "run": run, "prompt_id": p["prompt_id"], "seed": p["seed"],
                "time_s": round(elapsed, 3), "peak_mem_gb": round(peak, 3),
                "freeu": json.dumps(s["freeu"]) if s["freeu"] else "",
                "pag_scale": s["pag_scale"], "pag_applied_layers": "|".join(s["pag_applied_layers"]),
                "steps": s["num_inference_steps"], "guidance_scale": s["guidance_scale"],
                "height": s["height"], "width": s["width"], "device": device,
                "dry_run": int(args.dry_run), "timestamp": datetime.now().isoformat(timespec="seconds"),
            })
            eta = (time.time() - t_run) / i * (len(todo) - i)
            print(f"  [{i}/{len(todo)}] {p['prompt_id']} {elapsed:.1f}s peak={peak:.2f}GB "
                  f"剩余约 {eta / 60:.1f} 分钟")
    print(f"\n完成。记录文件: {rec_path}")


if __name__ == "__main__":
    main()
