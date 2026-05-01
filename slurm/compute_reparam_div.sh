#!/bin/bash
#SBATCH --job-name=reparam_div
#SBATCH --partition=gatsby_ws
#SBATCH --gres=gpu:rtx4500:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --output=/nfs/ghome/live/cmarouani/FREE/outputs/cifar10_self_fm/reparam_div_%j.log

set -e

PYBIN=/nfs/ghome/live/cmarouani/.conda/envs/code-drifting/bin/python
REPO=/nfs/ghome/live/cmarouani/FREE
SELF_FM=$REPO/outputs/cifar10_self_fm
DATA_DIR=/tmp/cifar10_div_$$

mkdir -p "$DATA_DIR"

echo "=== E[(div u_{alpha(t)})^2] sweep — Self-FM step 200K ==="
echo "Node : $(hostname)"
echo "GPU  : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
date

PYTHONPATH=$REPO \
$PYBIN $REPO/scripts/compute_reparam_div.py \
    --ckpt      "$SELF_FM/checkpoints/ema_step_0200000.pt" \
    --speed_t   "$SELF_FM/fr_t_grid_step100000.npy"        \
    --speed_v   "$SELF_FM/fr_speed_step100000.npy"         \
    --out_dir   "$SELF_FM"                                  \
    --data_dir  "$DATA_DIR"                                 \
    --n_t       200                                         \
    --B         256                                         \
    --n_hutch   5                                           \
    --n_epochs  3                                           \
    --num_channel 128

echo "=== Done ===" && date
