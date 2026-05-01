#!/bin/bash
#SBATCH --job-name=latent_uniform_300k
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu-350-02
#SBATCH --gres=gpu:a4500:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/ghome/live/cmarouani/FREE/logs/latent_uniform_300k_%j.log
#SBATCH --error=/nfs/ghome/live/cmarouani/FREE/logs/latent_uniform_300k_%j.err

set -e

REPO=/nfs/ghome/live/cmarouani/FREE
OUT_DIR=$REPO/outputs/cifar10_latent_linear_uniform_300k
CKPT_200K=$REPO/outputs/cifar10_latent_linear/checkpoints/ckpt_step_0200000.pt
DATA_DIR=/tmp/cifar10_latent_$$

module load miniconda/23.10.0
eval "$(/ceph/apps/ubuntu-20/packages/miniconda/23.10.0/bin/conda shell.bash hook)"
conda activate code-drifting

TORCHRUN=$CONDA_PREFIX/bin/torchrun

echo "=== Latent FM Uniform Baseline (200K→300K) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

mkdir -p "$OUT_DIR/checkpoints" "$DATA_DIR" "$REPO/logs"

# Reuse pre-computed normalised latents to skip re-encoding
LATENTS_CACHE=$REPO/outputs/cifar10_latent_linear/latents_norm.pt
if [ -f "$LATENTS_CACHE" ] && [ ! -f "$OUT_DIR/latents_norm.pt" ]; then
    ln -s "$LATENTS_CACHE" "$OUT_DIR/latents_norm.pt"
    echo "Linked latents cache: $LATENTS_CACHE"
fi

PYTHONPATH=$REPO $TORCHRUN \
    --nproc_per_node=1 \
    --nnodes=1 --node_rank=0 \
    --master_addr=localhost --master_port=29504 \
    $REPO/scripts/train_fm_latent_linear_ddp.py \
    --vae_ckpt     "$REPO/outputs/cifar10_vae/vae_best.pt" \
    --latent_stats "$REPO/outputs/cifar10_vae/latent_stats.pt"  \
    --out_dir      "$OUT_DIR"                                    \
    --data_dir     "$DATA_DIR"                                   \
    --total_steps  300000                                        \
    --batch_size   128                                           \
    --lr           2e-4                                          \
    --warmup       1000                                          \
    --ema_decay    0.9999                                        \
    --grad_clip    1.0                                           \
    --num_channel  128                                           \
    --num_workers  4                                             \
    --eval_every   50000                                         \
    --fid_samples  10000                                         \
    --n_steps      35                                            \
    --vae_z_ch     4                                             \
    --vae_base_ch  128                                           \
    --resume       "$CKPT_200K"

echo "End time: $(date)"
echo "=== Training complete ==="
