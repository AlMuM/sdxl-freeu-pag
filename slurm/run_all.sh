#!/bin/bash
# 正式实验：主实验 + 全部超参实验 + CLIPScore。
# 已生成的图片会跳过，如果超时被中断，重新 sbatch 同一脚本即可续跑。
# 在工程根目录提交：sbatch slurm/run_all.sh
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --job-name=sdxl_all
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -eo pipefail   # 不用 -u：conda 的激活脚本会引用未定义变量
source slurm/env.sh

python -u scripts/02_generate.py --group main
python -u scripts/02_generate.py --group freeu_sweep --group pag_sweep
python -u scripts/02_generate.py --runs pag_up freeu_pag
python -u scripts/03_clipscore.py
python -u scripts/05_analyze.py
python -u scripts/06_make_grids.py
echo "ALL DONE"
