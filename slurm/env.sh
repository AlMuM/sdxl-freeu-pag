# 各作业脚本共用的环境设置（被 source，不单独提交）
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate sdxl
export HF_HOME="$HOME/.cache/huggingface"
export PYTHONUNBUFFERED=1
echo "节点: $(hostname)  作业: ${SLURM_JOB_ID:-local}  开始: $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
