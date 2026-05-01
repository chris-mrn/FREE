#!/usr/bin/env python
"""Run full 8-Gaussian FR curriculum experiment: train both models + compare."""
import subprocess
import sys

def run_command(cmd, description):
    """Run shell command and report status."""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n❌ Failed: {description}")
        sys.exit(1)
    print(f"✓ Complete: {description}")

def main():
    print("\n" + "="*60)
    print("8-Gaussian FR Curriculum Experiment")
    print("="*60)

    # Step 1: Train curriculum model (4x longer for convergence, 2x FR precision, direct switch)
    run_command(
        "PYTHONPATH=. python scripts/train_8gauss_curriculum.py "
        "--dataset 8gaussians --total_steps 80001 --phase2_start 20000 "
        "--batch_size 256 --lr 2e-4 --ema_decay 0.9999 "
        "--hidden 256 --depth 4 "
        "--speed_n_t 500 --speed_B 2000 --speed_epochs 10 --speed_smooth 0.01 "
        "--eval_every 20000 --seed 42 "
        "--out_dir outputs/8gauss_curriculum",
        "Training curriculum model (400k steps, FR switch at 100k, 200-point t-grid, direct switch)"
    )

    # Step 2: Train baseline model (4x longer for fair comparison)
    run_command(
        "PYTHONPATH=. python train.py "
        "--path linear --dataset 8gaussians --t_mode uniform "
        "--total_steps 80001 --batch_size 256 --lr 2e-4 "
        "--eval_every 20000 --keep_ckpts -1 "
        "--out_dir outputs/8gauss_uniform_400k --resume disabled",
        "Training baseline model (400k steps, uniform t-sampling)"
    )

    # Step 3: Generate comparison plots at final checkpoints (200-point energy sweep)
    run_command(
        "PYTHONPATH=. python scripts/plot_8gauss_energy.py "
        "--curriculum_ckpt outputs/8gauss_curriculum/checkpoints/ckpt_step_0080001.pt "
        "--curriculum_speed_npy outputs/8gauss_curriculum/fr_speed.npy "
        "--baseline_ckpt outputs/8gauss_uniform_400k/checkpoints/ckpt_step_0080001.pt "
        "--dataset 8gaussians --n_t 200 --B 2000 --n_epochs 5 "
        "--out_dir outputs/8gauss_energy_comparison",
        "Generating energy comparison plots (200-point energy sweep, final checkpoints)"
    )

    print("\n" + "="*60)
    print("✅ Experiment complete!")
    print("Results: outputs/8gauss_energy_comparison/energy_comparison.png")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
