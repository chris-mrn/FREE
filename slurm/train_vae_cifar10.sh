#!/bin/bash
#SBATCH --job-name=vae_cifar10
#SBATCH --partition=gpu_lowp
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --output=/nfs/ghome/live/cmarouani/FREE/outputs/cifar10_vae/train_%j.log

set -e
REPO=/nfs/ghome/live/cmarouani/FREE
PYBIN=/nfs/ghome/live/cmarouani/.conda/envs/code-drifting/bin/python
OUT_DIR=$REPO/outputs/cifar10_vae
DATA_DIR=/tmp/cifar10_vae_$$
mkdir -p "$OUT_DIR"

echo "=== VAE Training — CIFAR-10 ===" && echo "Node: $(hostname)" && date

LAST_CKPT=$(ls -t "$OUT_DIR"/vae_step_*.pt 2>/dev/null | head -1 || true)
RESUME_ARG=${LAST_CKPT:+"--resume $LAST_CKPT"}

PYTHONPATH=$REPO $PYBIN $REPO/scripts/train_vae_cifar10.py \
    --out_dir      "$OUT_DIR"   \
    --data_dir     "$DATA_DIR"  \
    --total_steps  100000       \
    --batch_size   128          \
    --lr           4.5e-6       \
    --disc_lr      4.5e-6       \
    --kl_weight    1e-6         \
    --perc_weight  1.0          \
    --disc_weight  0.5          \
    --disc_start   10000        \
    --kl_warmup    5000         \
    --ema_decay    0.9999       \
    --grad_clip    1.0          \
    --z_ch         4            \
    --base_ch      128          \
    --num_workers  4            \
    --log_every    100          \
    --save_every   20000        \
    --recon_every  5000         \
    $RESUME_ARG

echo "=== Done ===" && date
