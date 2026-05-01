#!/bin/bash
#SBATCH --job-name=eval_sfm_uni_200k
#SBATCH --partition=gpu_lowp
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/nfs/ghome/live/cmarouani/FREE/outputs/cifar10_self_fm_uniform/eval_200k_%j.log

set -e

PYBIN=/nfs/ghome/live/cmarouani/.conda/envs/code-drifting/bin/python
REPO=/nfs/ghome/live/cmarouani/FREE
OUT_DIR=$REPO/outputs/cifar10_self_fm_uniform
DATA_DIR=/tmp/cifar10_eval_$$

mkdir -p "$DATA_DIR"

echo "=== Self-FM CIFAR-10 — NFE sweep eval, step 200K ==="
echo "Node : $(hostname)"
echo "GPU  : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
date

PYTHONPATH=$REPO \
$PYBIN $REPO/scripts/eval_self_fm_nfe.py \
    --ckpt_dir   "$OUT_DIR/checkpoints"  \
    --out_dir    "$OUT_DIR"              \
    --nfe_list   5 10 20 35 50 100 200   \
    --n_samples  10000                   \
    --only_steps 200000                  \
    --data_dir   "$DATA_DIR"

echo "=== Eval done ===" && date
