"""
Compare FR energy allocation along the flow for two trained 2D models.

Fisher-Rao energy: D(t) = E[(div u_t^theta(X_t))^2]
  - Curriculum model: trained with FR-adaptive curriculum (outputs/8gauss_curriculum)
  - Baseline model:   trained with uniform t-sampling throughout (outputs/8gauss_uniform_300k)

2-panel plot:
  Panel A: D(t) for both models with ±std bands and CV annotations
  Panel B: Sampling density p(t) under each training strategy

Usage:
    PYTHONPATH=. python scripts/plot_8gauss_energy.py \\
        --curriculum_ckpt outputs/8gauss_curriculum/checkpoints/ckpt_step_0200000.pt \\
        --curriculum_speed_npy outputs/8gauss_curriculum/fr_speed.npy \\
        --baseline_ckpt outputs/8gauss_uniform_300k/checkpoints/ckpt_step_0200000.pt \\
        --dataset 8gaussians \\
        --out_dir outputs/8gauss_energy_comparison
"""
import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.models import MLP2D
from scripts.compute_reparam_div_2d import (
    exact_div,
    sample_interpolant_classical,
)

T_MIN = 0.2
T_MAX = 0.8


def sweep_fr_energy(model, dataset, t_grid, B, n_epochs, device):
    """
    Sweep D(t) = E[(div u_t)^2] over t_grid via exact divergence.
    Returns (mean, std) both shape (len(t_grid),).
    """
    """v_t^FR = sqrt(E[(div u_t)^2]) via exact divergence (valid for dim=2)."""
    model.eval()
    sq_all = np.zeros((n_epochs, len(t_grid)))

    for ep in range(n_epochs):
        for i, t_val in enumerate(tqdm(
                t_grid, desc=f'  FR speed epoch {ep+1}/{n_epochs}', ncols=80, leave=False)):

            t_val_c = float(np.clip(t_val, T_MIN + 1e-3, T_MAX - 1e-3))

            xt, _ = sample_interpolant_classical(dataset, B, t_val_c, device)

            t_tensor = torch.full((xt.shape[0],), t_val_c,
                      device=device, dtype=torch.float32)

            u = model(t_tensor, xt)  # shape: (B, d)

            # divergence (should be per-sample or already batchwise scalar per sample)
            div_est = exact_div(model, t_val_c, xt, device)  # ideally shape: (B,)

            # ---- FIX 1: norm must be per-sample, not global ----
            scaled_norm = t_val_c / (1 - t_val_c) * torch.linalg.norm(u, dim=1)  # (B,)

            # ---- FIX 2: dot product must be per-sample ----
            # xt.T @ u is WRONG for batch data
            scaled_dot_product = (1.0 / t_val_c) * torch.sum(xt * u, dim=1)  # (B,)

            # combine
            err = div_est - scaled_norm - scaled_dot_product  # (B,)

            sq_all[ep, i] = torch.mean(err ** 2).detach().cpu().numpy()

    mean = sq_all.mean(axis=0)
    std  = sq_all.std(axis=0)
    return mean, std



def load_sampling_density(speed_npy_path, t_grid):
    """
    Reconstruct sampling density p(t) from saved FR speed profile.
    p(t) ∝ v_t^FR, normalised to unit integral over t_grid.
    Returns uniform density array if speed_npy_path is None.
    """
    if speed_npy_path is None or not os.path.exists(speed_npy_path):
        return np.ones(len(t_grid)) / (t_grid[-1] - t_grid[0])

    v_t = np.load(speed_npy_path)
    t_speed = np.load(speed_npy_path.replace('fr_speed.npy', 'fr_t_grid.npy')) \
        if os.path.exists(speed_npy_path.replace('fr_speed.npy', 'fr_t_grid.npy')) \
        else np.linspace(T_MIN, T_MAX, len(v_t))

    # Interpolate to the analysis t_grid
    v_interp = np.interp(t_grid, t_speed, v_t)
    v_interp = np.maximum(v_interp, 1e-10)

    # p(t) ∝ v_t, normalised
    density = (1/v_interp) / np.trapezoid((1/v_interp), t_grid)
    return density


def load_model(ckpt_path, hidden, depth, device):
    """Load EMA weights from checkpoint into MLP2D."""
    net = MLP2D(dim=2, hidden=hidden, depth=depth).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    # Try EMA weights first, fall back to net weights
    state = ckpt.get('ema', ckpt.get('net', ckpt))
    net.load_state_dict(state)
    net.eval()
    return net


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--curriculum_ckpt',      required=True,
                        help='Path to curriculum model checkpoint')
    parser.add_argument('--curriculum_speed_npy', default=None,
                        help='Path to fr_speed.npy from curriculum training dir (optional)')
    parser.add_argument('--baseline_ckpt',        required=True,
                        help='Path to uniform baseline checkpoint')
    parser.add_argument('--dataset',   default='8gaussians')
    parser.add_argument('--n_t',       type=int,   default=100,
                        help='Number of t grid points for energy sweep')
    parser.add_argument('--B',         type=int,   default=2_000,
                        help='Batch size per t point per epoch')
    parser.add_argument('--n_epochs',  type=int,   default=5,
                        help='Epochs for energy sweep')
    parser.add_argument('--hidden',    type=int,   default=256)
    parser.add_argument('--depth',     type=int,   default=4)
    parser.add_argument('--out_dir',   required=True)
    parser.add_argument('--seed',      type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    t_grid = np.linspace(T_MIN + 0.01, T_MAX - 0.01, args.n_t, dtype=np.float32)

    # ── Load models ───────────────────────────────────────────────────────────
    print(f'Loading curriculum model from {args.curriculum_ckpt}')
    curr_model = load_model(args.curriculum_ckpt, args.hidden, args.depth, device)

    print(f'Loading baseline model from {args.baseline_ckpt}')
    base_model = load_model(args.baseline_ckpt, args.hidden, args.depth, device)

    # ── Compute FR energy sweep ───────────────────────────────────────────────
    print('\nSweeping FR energy for curriculum model...')
    curr_mean, curr_std = sweep_fr_energy(curr_model, args.dataset, t_grid,
                                          args.B, args.n_epochs, device)

    print('\nSweeping FR energy for baseline model...')
    base_mean, base_std = sweep_fr_energy(base_model, args.dataset, t_grid,
                                          args.B, args.n_epochs, device)

    # ── Sampling densities ────────────────────────────────────────────────────
    p_curr = load_sampling_density(args.curriculum_speed_npy, t_grid)
    p_base = np.ones(len(t_grid)) / (t_grid[-1] - t_grid[0])  # uniform

    # ── Save arrays ───────────────────────────────────────────────────────────
    np.save(os.path.join(args.out_dir, 't_grid.npy'),              t_grid)
    np.save(os.path.join(args.out_dir, 'curriculum_fr_mean.npy'),  curr_mean)
    np.save(os.path.join(args.out_dir, 'curriculum_fr_std.npy'),   curr_std)
    np.save(os.path.join(args.out_dir, 'baseline_fr_mean.npy'),    base_mean)
    np.save(os.path.join(args.out_dir, 'baseline_fr_std.npy'),     base_std)
    np.save(os.path.join(args.out_dir, 'curriculum_density.npy'),  p_curr)
    print('Arrays saved.')

    # ── Statistics ────────────────────────────────────────────────────────────
    cv_curr = curr_mean.std() / (curr_mean.mean() + 1e-10)
    cv_base = base_mean.std() / (base_mean.mean() + 1e-10)
    print(f'\nCurriculum  CV(D)={cv_curr:.4f}  mean={curr_mean.mean():.4f}')
    print(f'Baseline    CV(D)={cv_base:.4f}  mean={base_mean.mean():.4f}')
    change = (1.0 - cv_curr / max(cv_base, 1e-10)) * 100
    print(f'CV change: {cv_base:.4f} → {cv_curr:.4f}  ({change:+.1f}%)')

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(f'Fisher-Rao Energy Allocation — {args.dataset}', fontsize=13, fontweight='bold')

    # Panel A: FR energy D(t)
    ax = axes[0]
    ax.fill_between(t_grid, base_mean - base_std, base_mean + base_std,
                    alpha=0.15, color='steelblue')
    ax.plot(t_grid, base_mean, color='steelblue', lw=2,
            label=f'Baseline (uniform)  CV={cv_base:.3f}')
    ax.fill_between(t_grid, curr_mean - curr_std, curr_mean + curr_std,
                    alpha=0.15, color='tomato')
    ax.plot(t_grid, curr_mean, color='tomato', lw=2,
            label=f'Curriculum (FR)  CV={cv_curr:.3f}')
    ax.axhline(curr_mean.mean(), color='tomato', lw=1, ls='--', alpha=0.6)
    ax.axhline(base_mean.mean(), color='steelblue', lw=1, ls='--', alpha=0.6)
    ax.set_xlabel('$t$', fontsize=12)
    ax.set_ylabel(r'$E[(\mathrm{div}\,u_t^\theta)^2]$', fontsize=12)
    ax.set_title('(A) Fisher-Rao Energy $D(t)$', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel B: Sampling density
    ax = axes[1]
    ax.plot(t_grid, p_base, color='steelblue', lw=2, ls='--', label='Baseline (uniform)')
    ax.plot(t_grid, p_curr, color='tomato',    lw=2,           label='Curriculum (FR-weighted)')
    ax.set_xlabel('$t$', fontsize=12)
    ax.set_ylabel('$p(t)$', fontsize=12)
    ax.set_title('(B) Sampling Density $p(t)$', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = os.path.join(args.out_dir, 'energy_comparison.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nPlot saved → {out_png}')
    print('Done.')


if __name__ == '__main__':
    main()
