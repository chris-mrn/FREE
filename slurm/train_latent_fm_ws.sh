#!/bin/bash
#SBATCH --job-name=latent_lfm
#SBATCH --partition=gatsby_ws
#SBATCH --gres=gpu:rtx4500:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=10-00:00:00
#SBATCH --output=/nfs/ghome/live/cmarouani/FREE/outputs/cifar10_latent_linear/train_%j.log

set -e
REPO=/nfs/ghome/live/cmarouani/FREE
TORCHRUN=/nfs/ghome/live/cmarouani/.conda/envs/code-drifting/bin/torchrun
VAE_DIR=$REPO/outputs/cifar10_vae
OUT_DIR=$REPO/outputs/cifar10_latent_linear
DATA_DIR=/tmp/cifar10_latent_$$
mkdir -p "$OUT_DIR" "$DATA_DIR"
NGPU=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}

if [ ! -f "$VAE_DIR/vae_final.pt" ]; then
    echo "ERROR: VAE checkpoint not found." && exit 1
fi
VAE_CKPT=$VAE_DIR/vae_final.pt
[ -f "$VAE_DIR/vae_ema_final.pt" ] && VAE_CKPT=$VAE_DIR/vae_ema_final.pt

echo "=== Latent Linear FM (${NGPU}×RTX4500) ===" && echo "Node: $(hostname)" && date

PYTHONPATH=$REPO $TORCHRUN \
    --nproc_per_node=$NGPU \
    --nnodes=1 --node_rank=0 \
    --master_addr=localhost --master_port=29503 \
    $REPO/scripts/train_fm_latent_linear_ddp.py \
    --vae_ckpt     "$VAE_CKPT"                      \
    --latent_stats "$VAE_DIR/latent_stats.pt"       \
    --out_dir      "$OUT_DIR"                       \
    --data_dir     "$DATA_DIR"                      \
    --total_steps  200000                           \
    --batch_size   128                              \
    --lr           2e-4                             \
    --warmup       1000                             \
    --ema_decay    0.9999                           \
    --grad_clip    1.0                              \
    --num_channel  128                              \
    --num_workers  4                                \
    --eval_every   20000                            \
    --fid_samples  10000                            \
    --n_steps      35                               \
    --vae_z_ch     4                                \
    --vae_base_ch  128                              \
    --resume       auto

echo "=== Done ===" && date
