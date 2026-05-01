#!/bin/bash
#SBATCH --job-name=fm_sph_ot_cur
#SBATCH --partition=gpu_lowp
#SBATCH --gres=gpu:h100:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=14:00:00
#SBATCH --output=/nfs/ghome/live/cmarouani/FREE/outputs/cifar10_sph_ot_curriculum/train_%j.log

set -e
REPO=/nfs/ghome/live/cmarouani/FREE
TORCHRUN=/nfs/ghome/live/cmarouani/.conda/envs/code-drifting/bin/torchrun
OUT_DIR=$REPO/outputs/cifar10_sph_ot_curriculum
DATA_DIR=/tmp/cifar10_sph_curr_$$
mkdir -p "$OUT_DIR" "$DATA_DIR"
NGPU=$(nvidia-smi -L | wc -l)

echo "=== Spherical FM + OT coupling + Curriculum (${NGPU}×H100 DDP) ==="
echo "Node: $(hostname)" && date

PYTHONPATH=$REPO $TORCHRUN \
    --nproc_per_node=$NGPU \
    --nnodes=1 --node_rank=0 \
    --master_addr=localhost --master_port=29502 \
    $REPO/main.py train \
    --path              spherical           \
    --dataset           cifar10             \
    --coupling          ot                  \
    --t_mode            curriculum          \
    --speed_type        ot                  \
    --curriculum_start  100000              \
    --curriculum_blend  25000               \
    --speed_n_t         100                 \
    --speed_B           512                 \
    --speed_epochs      3                   \
    --total_steps       400001              \
    --batch_size        128                 \
    --lr                2e-4                \
    --warmup            5000                \
    --ema_decay         0.9999              \
    --grad_clip         1.0                 \
    --num_channel       128                 \
    --eval_every        20000               \
    --fid_samples       10000               \
    --keep_ckpts        2                   \
    --num_workers       4                   \
    --data_dir          "$DATA_DIR"         \
    --out_dir           "$OUT_DIR"          \
    --resume            auto                \
    --ddp

echo "=== Done ===" && date
