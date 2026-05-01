#!/bin/bash
set -e

echo "========================================"
echo "8-Gaussian FR Curriculum Experiment"
echo "========================================"

# Train curriculum model (200k steps, FR-adaptive from step 100k, 200-point t-grid)
echo ""
echo "Step 1: Training curriculum model (200k steps, FR switch at 100k, 200-point t-grid)..."
PYTHONPATH=. python scripts/train_8gauss_curriculum.py \
    --dataset 8gaussians --total_steps 200001 --phase2_start 100000 \
    --batch_size 256 --lr 2e-4 --ema_decay 0.9999 \
    --hidden 256 --depth 4 \
    --speed_n_t 200 --speed_B 2000 --speed_epochs 5 --speed_smooth 0.05 \
    --eval_every 50000 --seed 42 \
    --out_dir outputs/8gauss_curriculum

# Train baseline model (200k steps, uniform t-sampling throughout)
echo ""
echo "Step 2: Training baseline model (200k steps, uniform t-sampling)..."
PYTHONPATH=. python train.py \
    --path linear --dataset 8gaussians --t_mode uniform \
    --total_steps 200001 --batch_size 256 --lr 2e-4 \
    --eval_every 50000 --keep_ckpts -1 \
    --out_dir outputs/8gauss_uniform_200k --resume disabled

# Generate energy comparison plots (200-point energy sweep)
echo ""
echo "Step 3: Generating comparison plots (200-point energy sweep)..."
PYTHONPATH=. python scripts/plot_8gauss_energy.py \
    --curriculum_ckpt outputs/8gauss_curriculum/checkpoints/ckpt_step_0100000.pt \
    --curriculum_speed_npy outputs/8gauss_curriculum/fr_speed.npy \
    --baseline_ckpt outputs/8gauss_uniform_200k/checkpoints/ckpt_step_0100000.pt \
    --dataset 8gaussians --n_t 200 --B 2000 --n_epochs 5 \
    --out_dir outputs/8gauss_energy_comparison

echo ""
echo "========================================"
echo "Experiment complete!"
echo "Results: outputs/8gauss_energy_comparison/energy_comparison.png"
echo "========================================"
