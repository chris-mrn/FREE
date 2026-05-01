#!/bin/bash
#SBATCH --job-name=reparam_div_2d
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=/nfs/ghome/live/cmarouani/FREE/outputs/self_fm_2d/reparam_div_2d_%j.log

set -e

PYBIN=/nfs/ghome/live/cmarouani/.conda/envs/code-drifting/bin/python
REPO=/nfs/ghome/live/cmarouani/FREE
OUT=$REPO/outputs/self_fm_2d

mkdir -p "$OUT"

echo "=== 2D self-FM energy repartition sweep (CPU) ==="
echo "Node : $(hostname)"
date

for DATASET in 8gaussians moons circles checkerboard; do
    echo ""
    echo "── Dataset: $DATASET ──"
    PYTHONPATH=$REPO \
    $PYBIN $REPO/scripts/compute_reparam_div_2d.py \
        --dataset          "$DATASET"              \
        --out_dir          "$OUT/$DATASET"          \
        --total_steps      50000                    \
        --batch_size       512                      \
        --curriculum_start 20000                    \
        --curriculum_blend 5000                     \
        --speed_n_t        100                      \
        --speed_B          2000                     \
        --speed_epochs     5                        \
        --div_n_t          200                      \
        --div_B            2000                     \
        --div_epochs       5
done

echo ""
echo "=== All datasets done ==="
date
