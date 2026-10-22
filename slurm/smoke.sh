#!/bin/bash
# 冒烟测试：下载模型与 COCO 标注、抽样 prompt、每组只生成 5 张。
# 在工程根目录提交：sbatch slurm/smoke.sh
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --job-name=sdxl_smoke
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -eo pipefail   # 不用 -u：conda 的激活脚本会引用未定义变量
source slurm/env.sh

echo "== 网络检查"
curl -sS -o /dev/null -w "huggingface.co %{http_code}\n" --max-time 15 https://huggingface.co || echo "计算节点无法访问外网"

echo "== 下载模型（已缓存则跳过）"
python -u -c "
from huggingface_hub import snapshot_download
p = snapshot_download('stabilityai/stable-diffusion-xl-base-1.0',
    allow_patterns=['*.json', '*.txt', 'tokenizer*/*', '*fp16.safetensors'])
print('model at', p)
"

echo "== 抽样 prompt"
[ -f data/prompts.csv ] || python -u scripts/01_sample_prompts.py --n 150 --subset 50

echo "== 每组生成 5 张"
python -u scripts/02_generate.py --group main --limit 5
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
