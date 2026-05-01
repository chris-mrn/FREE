#!/bin/bash
#SBATCH --job-name=latent_fm_curriculum
#SBATCH --partition=a100
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=80GB
#SBATCH --output=logs/latent_fm_curriculum_%j.log
#SBATCH --error=logs/latent_fm_curriculum_%j.err

set -e

cd /nfs/ghome/live/cmarouani/FREE

# Load miniconda module and activate conda environment
module load miniconda/23.10.0
eval "$(/ceph/apps/ubuntu-20/packages/miniconda/23.10.0/bin/conda shell.bash hook)"
conda activate code-drifting

echo "=== Training Latent FM with Fisher-Rao Curriculum ==="
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

mkdir -p logs

PYTHONPATH=. python scripts/train_latent_fm_curriculum.py \
    --ckpt_dir outputs/cifar10_latent_linear/checkpoints \
    --latent_stats outputs/cifar10_vae/latent_stats.pt \
    --out_dir outputs/cifar10_latent_linear_curriculum \
    --speed_n_t 100 \
    --speed_epochs 3 \
    --speed_hutch 5 \
    --curriculum_blend 50000 \
    --additional_steps 100000 \
    --batch_size 256 \
    --seed 42

echo "End time: $(date)"
echo "=== Training complete ==="
