#!/bin/bash
#SBATCH --job-name=latent_fr_speed
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu-350-03
#SBATCH --gres=gpu:a4500:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/nfs/ghome/live/cmarouani/FREE/logs/latent_fr_speed_%j.log
#SBATCH --error=/nfs/ghome/live/cmarouani/FREE/logs/latent_fr_speed_%j.err

set -e

REPO=/nfs/ghome/live/cmarouani/FREE

module load miniconda/23.10.0
eval "$(/ceph/apps/ubuntu-20/packages/miniconda/23.10.0/bin/conda shell.bash hook)"
conda activate code-drifting

echo "=== Latent FR Speed Estimation + Plot ==="
echo "Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES  Start: $(date)"

mkdir -p "$REPO/logs"

# Estimate FR speed on final curriculum checkpoint (300K)
PYTHONPATH=$REPO python scripts/compute_latent_fr_speed.py \
    --ckpt         outputs/cifar10_latent_linear_curriculum/ckpt_step_0300000.pt \
    --latent_stats outputs/cifar10_vae/latent_stats.pt \
    --out_dir      outputs/cifar10_latent_linear_curriculum/speed_profile_300k \
    --speed_n_t    100 \
    --speed_B      512 \
    --speed_epochs 5 \
    --speed_hutch  5

echo "Speed estimation done. Plotting..."

# Generate 4-panel plot
PYTHONPATH=$REPO python scripts/plot_latent_speed.py \
    --speed_dir outputs/cifar10_latent_linear_curriculum/speed_profile_300k \
    --out_dir   outputs/cifar10_latent_linear_curriculum \
    --smooth_sigma 0.05

echo "Done: $(date)"
echo "Plot: outputs/cifar10_latent_linear_curriculum/latent_speed_profile.png"
