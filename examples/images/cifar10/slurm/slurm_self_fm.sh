#!/bin/bash
#SBATCH --job-name=self_fm
#SBATCH --partition=gpu_lowp
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/nfs/ghome/live/cmarouani/FREE/outputs/cifar10_self_fm/train_%j.log

set -e

PYBIN=/nfs/ghome/live/cmarouani/.conda/envs/code-drifting/bin/python
REPO=/nfs/ghome/live/cmarouani/FREE
OUT_DIR=$REPO/outputs/cifar10_self_fm
DATA_DIR=/tmp/cifar10_self_fm_$$

mkdir -p "$OUT_DIR" "$DATA_DIR"

echo "=== Self-FM CIFAR-10: standard 100k + FR speed + arc-length curriculum ==="
echo "Node : $(hostname)"
echo "GPU  : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo "Out  : $OUT_DIR"
date

PYTHONPATH=$REPO \
$PYBIN $REPO/examples/images/cifar10/training/train_self_fm.py \
    --total_steps   200001 \
    --batch_size    128 \
    --lr            2e-4 \
    --warmup        5000 \
    --ema_decay     0.9999 \
    --num_channel   128 \
    --save_step     50000 \
    --num_workers   4 \
    --data_dir      "$DATA_DIR" \
    --out_dir       "$OUT_DIR" \
    --speed_step    100000 \
    --blend_steps   25000 \
    --fr_n_t        1000 \
    --fr_n_epochs   5 \
    --fr_n_hutch    5 \
    --fr_B_per_t    2 \
    --fr_chunk      128 \
    --fr_smooth     3.0 \
    --fr_n_ref      2000

echo "=== Done ==="
date
