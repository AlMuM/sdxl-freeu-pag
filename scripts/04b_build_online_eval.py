"""把 04_build_blind_eval.py 打包好的盲评材料做成在线版页面。

在线版和离线版用同一份 blind_eval/manifest.js 和 blind_eval_key.json，
所以分析脚本不用改。页面发布成 claude.ai Artifact 后，评分自动保存到共享数据库，
评审不用下载压缩包，也不用导出 JSON。

用法：
    python scripts/04_build_blind_eval.py              # 先生成离线版（如果还没有）
    python scripts/04b_build_online_eval.py            # 生成 online_eval/，发布为 Artifact
    python scripts/04b_build_online_eval.py --server   # 生成 online_eval_server/，配合 eval_server.py 自托管
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common import ROOT  # noqa: E402


# 自托管时页面不经过 Artifact 的外壳，需要自己补上 HTML 骨架
SERVER_HEAD = ('<!doctype html><html lang="zh"><head><meta charset="utf-8">'
               '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"></head><body>\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", action="store_true", help="生成自托管版本（评分保存到 eval_server.py）")
    args = ap.parse_args()

    src_dir = ROOT / "blind_eval"
    manifest = src_dir / "manifest.js"
    if not manifest.exists():
        raise SystemExit("找不到 blind_eval/manifest.js，请先运行 scripts/04_build_blind_eval.py")

    out = ROOT / ("online_eval_server" if args.server else "online_eval")
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src_dir / "images", out / "images")
    html = (ROOT / "src" / "online_eval.html").read_text(encoding="utf-8")
    data = manifest.read_text(encoding="utf-8").strip()
    if args.server:
        data += "\nwindow.EVAL_SERVER = true;"
    html = html.replace("/*__MANIFEST__*/", data)
    if args.server:
        html = SERVER_HEAD + html + "\n</body></html>\n"
    (out / "index.html").write_text(html, encoding="utf-8")

    n = len(list((out / "images").glob("*.jpg")))
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 2**20
    print(f"在线版 -> {out}（{n} 张图，共 {size:.1f} MB）")


if __name__ == "__main__":
    main()
