#!/bin/bash
#SBATCH --job-name=sph_nfe_eval
#SBATCH --partition=gpu_lowp
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/nfs/ghome/live/cmarouani/FREE/outputs/cifar10_spherical/fm_standard/eval_nfe_%j.log

set -e

PYBIN=/nfs/ghome/live/cmarouani/.conda/envs/code-drifting/bin/python
REPO=/nfs/ghome/live/cmarouani/FREE
CKPT_DIR=$REPO/outputs/cifar10_spherical/fm_220k_only
OUT_DIR=$REPO/outputs/cifar10_spherical/fm_standard
DATA_DIR=/tmp/cifar10_sph_eval_$$

mkdir -p "$DATA_DIR"

echo "=== Spherical FM NFE sweep: 10,35,50,100,200,500,1000 ==="
echo "Node : $(hostname)"
echo "GPU  : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo "Ckpt : $CKPT_DIR"
echo "Out  : $OUT_DIR"
date

PYTHONPATH=$REPO \
$PYBIN $REPO/scripts/eval_fid_nfe.py \
    --ckpt_dir    "$CKPT_DIR" \
    --out_dir     "$OUT_DIR" \
    --metrics_csv "$OUT_DIR/metrics.csv" \
    --nfe_list    10 35 50 100 200 500 1000 \
    --n_samples   10000 \
    --data_dir    "$DATA_DIR" \
    --spherical

echo "=== Done ==="
date
