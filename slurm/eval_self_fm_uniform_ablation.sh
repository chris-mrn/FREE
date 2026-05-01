#!/bin/bash
#SBATCH --job-name=self_fm_uni_eval
#SBATCH --partition=gatsby_ws
#SBATCH --gres=gpu:rtx4500:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=8:00:00
#SBATCH --output=/nfs/ghome/live/cmarouani/FREE/outputs/cifar10_self_fm_uniform/eval_%j.log

set -e

PYBIN=/nfs/ghome/live/cmarouani/.conda/envs/code-drifting/bin/python
REPO=/nfs/ghome/live/cmarouani/FREE
OUT_DIR=$REPO/outputs/cifar10_self_fm_uniform
DATA_DIR=/tmp/cifar10_self_fm_uni_eval_$$

mkdir -p "$DATA_DIR"

echo "=== Self-FM Uniform Ablation — NFE Evaluation ==="
echo "Node : $(hostname)"
echo "GPU  : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo "Ckpts: $OUT_DIR/checkpoints"
echo "Out  : $OUT_DIR"
date

PYTHONPATH=$REPO \
$PYBIN $REPO/scripts/eval_self_fm_nfe.py \
    --ckpt_dir   "$OUT_DIR/checkpoints" \
    --out_dir    "$OUT_DIR"             \
    --nfe_list   5 10 20 35 50 100 200  \
    --n_samples  10000                  \
    --data_dir   "$DATA_DIR"            \
    --num_channel 128                   \
    --batch_size  256

echo "=== Eval done ===" && date
