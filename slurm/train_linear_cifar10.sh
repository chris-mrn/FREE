#!/bin/bash
#SBATCH --job-name=fm_lin_cifar
#SBATCH --partition=gpu_lowp
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/ghome/live/cmarouani/FREE/outputs/cifar10_linear/train_%j.log

set -e
REPO=/nfs/ghome/live/cmarouani/FREE
PYBIN=/nfs/ghome/live/cmarouani/.conda/envs/code-drifting/bin/python
OUT_DIR=$REPO/outputs/cifar10_linear
DATA_DIR=/tmp/cifar10_lin_$$
mkdir -p "$OUT_DIR" "$DATA_DIR"

echo "=== Linear FM on CIFAR-10 ===" && echo "Node: $(hostname)" && date

PYTHONPATH=$REPO $PYBIN $REPO/train.py \
    --path          linear              \
    --dataset       cifar10             \
    --coupling      independent         \
    --t_mode        uniform             \
    --total_steps   400001              \
    --batch_size    128                 \
    --lr            2e-4                \
    --warmup        5000                \
    --ema_decay     0.9999              \
    --grad_clip     1.0                 \
    --num_channel   128                 \
    --eval_every    20000               \
    --fid_samples   10000               \
    --keep_ckpts    2                   \
    --num_workers   4                   \
    --data_dir      "$DATA_DIR"         \
    --out_dir       "$OUT_DIR"          \
    --resume        auto

echo "=== Done ===" && date
