"""各脚本共享的工具函数：读配置、读 prompt、解析 run 参数。"""
from __future__ import annotations

import csv
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "experiments.yaml"


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """相对路径一律相对于工程根目录。"""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_prompts(cfg: dict) -> list[dict]:
    path = resolve(cfg["prompts"]["file"])
    if not path.exists():
        raise SystemExit(f"找不到 {path}，请先运行 scripts/01_sample_prompts.py")
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["seed"] = int(r["seed"])
        r["in_subset"] = r["in_subset"] == "1"
    return rows


def prompts_for_run(cfg: dict, run_name: str, prompts: list[dict]) -> list[dict]:
    scope = cfg["runs"][run_name].get("prompts", "all")
    if scope == "all":
        return prompts
    if scope == "subset":
        return [p for p in prompts if p["in_subset"]]
    raise ValueError(f"run {run_name}: prompts 只能是 all 或 subset，得到 {scope}")


def run_settings(cfg: dict, run_name: str) -> dict:
    """合并 common 与 run 自己的设置，返回这个 run 实际使用的全部参数。"""
    common = cfg["common"]
    run = cfg["runs"][run_name]
    return {
        "freeu": run.get("freeu"),
        "pag_scale": float(run.get("pag_scale", 0.0)),
        "pag_applied_layers": run.get("pag_applied_layers", common["pag_applied_layers"]),
        "guidance_scale": run.get("guidance_scale", common["guidance_scale"]),
        "num_inference_steps": run.get("num_inference_steps", common["num_inference_steps"]),
        "height": common["height"],
        "width": common["width"],
    }


def image_path(cfg: dict, run_name: str, prompt_id: str) -> Path:
    return resolve(cfg["common"]["output_dir"]) / "images" / run_name / f"{prompt_id}.png"


def records_path(cfg: dict) -> Path:
    return resolve(cfg["common"]["output_dir"]) / "records.csv"


def results_dir(cfg: dict) -> Path:
    d = resolve("results")
    d.mkdir(exist_ok=True)
    return d
