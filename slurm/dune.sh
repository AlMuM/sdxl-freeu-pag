#!/bin/bash
# DUNE 第五组 + 与原图相似度。先用 LIMIT=3 冒烟，再不带 LIMIT 跑全部：
#   LIMIT=3 sbatch slurm/dune.sh
#   sbatch slurm/dune.sh
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --job-name=sdxl_dune
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -eo pipefail   # 不用 -u：conda 的激活脚本会引用未定义变量
source slurm/env.sh

if [ -n "$LIMIT" ]; then
  python -u scripts/02_generate.py --runs dune --limit "$LIMIT"
  exit 0
fi
python -u scripts/02_generate.py --runs dune
python -u scripts/03_clipscore.py
python -u scripts/08_clip_similarity.py
python -u scripts/05_analyze.py
python -u scripts/06_make_grids.py
echo "ALL DONE"
