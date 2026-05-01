"""
Plot Fisher-Rao speed profile for latent FM: speed, weighting, and arc-length schedule.

Takes the output from compute_latent_fr_speed.py and creates a 3-panel visualization:
  Panel 1: Speed v_t^FR (Fisher-Rao speed)
  Panel 2: Weighting w(t) = L / v_t (raw and smoothed)
  Panel 3: Arc-length schedule α(t) — reparameterisation function

Usage:
  python plot_latent_speed.py \\
      --speed_dir outputs/latent_speed_fr \\
      --out_dir outputs/latent_speed_fr \\
      --smooth_sigma 0.05
"""
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, '/nfs/ghome/live/cmarouani/FREE')


def schedule_from_weighting(w, t_grid):
    """
    Compute arc-length schedule α(t) from weighting w(t).

    Arc-length reparameterisation: (α^{-1})'(t) = 1/w(t)
    Returns α(t) such that uniform sampling in α space → weighted sampling in t space.
    """
    integrand = 1.0 / np.maximum(w, 1e-12)
    cum = np.zeros(len(t_grid))
    for i in range(1, len(t_grid)):
        cum[i] = np.trapezoid(integrand[:i+1], t_grid[:i+1])
    cum_normalized = cum / max(cum[-1], 1e-12)
    # Invert: α(t) is the function such that integrating 1/w gives α
    return np.interp(t_grid, cum_normalized, t_grid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--speed_dir', type=str, required=True,
                        help='Directory containing fr_speed.npy, fr_weighting.npy, t_grid.npy')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory for the plot')
    parser.add_argument('--smooth_sigma', type=float, default=0.05,
                        help='Gaussian smoothing bandwidth (in t-units)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ─── Load speed profile ────────────────────────────────────────────────────
    speed_path = os.path.join(args.speed_dir, 'fr_speed.npy')
    weighting_path = os.path.join(args.speed_dir, 'fr_weighting.npy')
    t_grid_path = os.path.join(args.speed_dir, 't_grid.npy')

    if not os.path.exists(speed_path):
        print(f'Error: {speed_path} not found')
        return

    v_fr = np.load(speed_path)
    w_raw = np.load(weighting_path) if os.path.exists(weighting_path) else None
    t_grid = np.load(t_grid_path)

    print(f'Loaded speed profile from {args.speed_dir}')
    print(f'  t_grid shape: {t_grid.shape}')
    print(f'  v_fr shape: {v_fr.shape}')
    print(f'  v_fr range: [{v_fr.min():.4f}, {v_fr.max():.4f}]')

    # ─── Compute weighting if not provided ─────────────────────────────────────
    if w_raw is None:
        print('Computing weighting from speed...')
        cum = np.zeros(len(t_grid))
        for i in range(1, len(t_grid)):
            cum[i] = np.trapezoid(v_fr[:i+1], t_grid[:i+1])
        L = max(cum[-1], 1e-12)
        w_raw = np.where(v_fr > 1e-12, L / v_fr, 0.0)

    # ─── Smooth weighting ─────────────────────────────────────────────────────
    if args.smooth_sigma > 0:
        dt_avg = (t_grid[-1] - t_grid[0]) / max(len(t_grid) - 1, 1)
        sigma_pts = args.smooth_sigma / max(dt_avg, 1e-9)
        w_smooth = gaussian_filter1d(w_raw, sigma=sigma_pts, mode='reflect')
        w_smooth = np.minimum(w_smooth, 100.0)
    else:
        w_smooth = w_raw

    print(f'  w_raw range: [{w_raw.min():.4f}, {w_raw.max():.4f}]')
    print(f'  w_smooth range: [{w_smooth.min():.4f}, {w_smooth.max():.4f}]')

    # ─── Compute arc-length schedule ──────────────────────────────────────────
    alpha = schedule_from_weighting(w_smooth, t_grid)

    # ─── Compute sampling density p(t) ∝ v_t (CdfSampler samples ∝ speed) ─────
    density = v_fr / np.trapezoid(v_fr, t_grid)
    density_uniform = np.ones_like(t_grid) / (t_grid[-1] - t_grid[0])

    # ─── Create 4-panel plot ──────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)
    fig.suptitle('Latent FM Speed Profile (Fisher-Rao)', fontsize=14, fontweight='bold')

    # Panel 1: Speed
    axes[0].plot(t_grid, v_fr, color='steelblue', lw=2.5, label=r'$v_t^{FR}$')
    axes[0].axhline(v_fr.mean(), color='steelblue', lw=1.2, ls='--', alpha=0.6,
                    label=f'mean = {v_fr.mean():.1f}')
    cv = v_fr.std() / v_fr.mean()
    axes[0].set_ylabel(r'Speed $v_t^{FR}$', fontsize=11)
    axes[0].set_title(f'Panel A: Fisher-Rao Speed  (CV = {cv:.3f})', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10, loc='best')
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Sampling density p(t) ∝ v_t
    axes[1].plot(t_grid, density, color='mediumorchid', lw=2.5,
                 label=r'$p(t) \propto v_t^{FR}$  (FR curriculum)')
    axes[1].fill_between(t_grid, density, density_uniform, alpha=0.15, color='mediumorchid')
    axes[1].plot(t_grid, density_uniform, 'k--', lw=1.5, alpha=0.6, label='uniform $p(t)$')
    axes[1].set_ylabel(r'Sampling density $p(t)$', fontsize=11)
    axes[1].set_title('Panel B: t-Sampling Density', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10, loc='best')
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Weighting
    axes[2].plot(t_grid, w_raw, color='coral', lw=1.5, alpha=0.5,
                 label=r'$w(t)=L/v_t$ (raw)')
    axes[2].plot(t_grid, w_smooth, color='coral', lw=2.5,
                 label=r'$\tilde{w}(t)$ (smoothed)')
    axes[2].axhline(1.0, color='k', lw=1.2, ls='--', alpha=0.5, label='uniform baseline')
    axes[2].set_ylabel(r'Weighting $w(t)$', fontsize=11)
    axes[2].set_title('Panel C: Inverse-Speed Weighting', fontsize=12, fontweight='bold')
    axes[2].legend(fontsize=10, loc='best')
    axes[2].grid(True, alpha=0.3)

    # Panel 4: Arc-length schedule
    axes[3].plot(t_grid, alpha, color='forestgreen', lw=2.5, label=r'$\alpha(t)$ (schedule)')
    axes[3].plot(t_grid, t_grid, 'k--', lw=1.5, alpha=0.5, label='identity (uniform)')
    axes[3].set_ylabel(r'Schedule $\alpha(t)$', fontsize=11)
    axes[3].set_xlabel('Time $t$', fontsize=11)
    axes[3].set_title('Panel D: Arc-Length Reparameterisation', fontsize=12, fontweight='bold')
    axes[3].legend(fontsize=10, loc='best')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(args.out_dir, 'latent_speed_profile.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n✓ Saved plot to {plot_path}')

    # ─── Compute and print statistics ──────────────────────────────────────────
    print('\n' + '='*60)
    print('Speed Profile Summary')
    print('='*60)
    print(f'Speed v_t^FR:')
    print(f'  min: {v_fr.min():.4f},  max: {v_fr.max():.4f},  mean: {v_fr.mean():.4f}')
    print(f'  ratio (max/min): {v_fr.max() / (v_fr.min() + 1e-12):.2f}x')

    print(f'\nWeighting w(t) (smoothed):')
    print(f'  min: {w_smooth.min():.4f},  max: {w_smooth.max():.4f},  mean: {w_smooth.mean():.4f}')
    print(f'  ratio (max/min): {w_smooth.max() / (w_smooth.min() + 1e-12):.2f}x')

    # Compute coefficient of variation for speed
    cv_speed = v_fr.std() / (v_fr.mean() + 1e-12)
    print(f'\nCoefficient of Variation:')
    print(f'  CV(v_t^FR) = {cv_speed:.4f}  (lower is more constant)')

    # Check how much α(t) deviates from uniform (t_grid)
    deviation = np.linalg.norm(alpha - t_grid) / np.linalg.norm(t_grid)
    print(f'  ||α(t) - t|| / ||t|| = {deviation:.4f}  (deviation from uniform)')

    print(f'\nSampling density p(t) ∝ 1/w(t):')
    density = 1.0 / np.maximum(w_smooth, 1e-12)
    density /= np.trapezoid(density, t_grid)  # Normalise to unit integral
    print(f'  density range: [{density.min():.4f}, {density.max():.4f}]')
    print(f'  density ratio (max/min): {density.max() / (density.min() + 1e-12):.2f}x')

    print(f'\nNotes:')
    print(f'  - Speed is high where the flow changes rapidly')
    print(f'  - Weighting w(t)=L/v(t) is inversely proportional to speed')
    print(f'  - Arc-length schedule α(t) maps uniform sampling in α to weighted in t')
    print(f'  - If α(t) ≈ t (identity), then speed is approximately constant')
    print('='*60)


if __name__ == '__main__':
    main()
