#!/bin/bash
#SBATCH --job-name=self_fm_uni
#SBATCH --partition=gpu_lowp
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/nfs/ghome/live/cmarouani/FREE/outputs/cifar10_self_fm_uniform/train_%j.log

set -e

PYBIN=/nfs/ghome/live/cmarouani/.conda/envs/code-drifting/bin/python
REPO=/nfs/ghome/live/cmarouani/FREE
CKPT_RESUME=$REPO/outputs/cifar10_self_fm/checkpoints/ema_step_0100000.pt
OUT_DIR=$REPO/outputs/cifar10_self_fm_uniform
DATA_DIR=/tmp/cifar10_self_fm_uni_$$

mkdir -p "$OUT_DIR" "$DATA_DIR"

echo "=== Self-FM CIFAR-10 — Uniform t ablation (resume from 100K, no curriculum) ==="
echo "Node : $(hostname)"
echo "GPU  : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo "Out  : $OUT_DIR"
echo "Resume: $CKPT_RESUME"
date

PYTHONPATH=$REPO \
$PYBIN $REPO/examples/images/cifar10/training/train_self_fm.py \
    --total_steps   200001          \
    --batch_size    128             \
    --lr            2e-4            \
    --warmup        5000            \
    --ema_decay     0.9999          \
    --num_channel   128             \
    --save_step     50000           \
    --num_workers   4               \
    --data_dir      "$DATA_DIR"     \
    --out_dir       "$OUT_DIR"      \
    --resume        "$CKPT_RESUME"  \
    --speed_step    999999

echo "=== Training done ===" && date
