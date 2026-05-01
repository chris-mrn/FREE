#!/usr/bin/env python3
"""
curriculum_speed_plot.py

Plot speed / weighting / sampling density for the two FR speed estimates
stored inside the curriculum checkpoints:

  speed1  — estimated at step 50K  (stored in ckpt_step_0050000.pt)
  speed2  — estimated at step 75K  (stored in ckpt_step_0100000.pt as speed2_*)

Quantities plotted:
  Row 0: v_t                       (Hutchinson FR speed)
  Row 1: w_t = (∫₀¹ v_s ds) / v_t  (importance-sampling weight)
  Row 2: p(t) = (1/v_t) / ∫₀¹ (1/v_s) ds  (sampling density, normalised)

Output:
  outputs/cifar10/comparison/curriculum_speed_plot.png
"""

import os, sys, argparse
sys.path.insert(0, '/nfs/ghome/live/cmarouani/FREE')

import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ─── helpers ──────────────────────────────────────────────────────────────────

def compute_weighting(speeds: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """w(t) = (∫₀¹ v_s ds) / v_t  (trapezoidal integral)."""
    dt     = np.diff(t_grid)
    mean_v = float(np.sum(0.5 * (speeds[:-1] + speeds[1:]) * dt))
    return mean_v / np.maximum(speeds, 1e-12)


def smooth_weighting(w: np.ndarray, t_grid: np.ndarray,
                     sigma_t: float = 0.05) -> np.ndarray:
    """Gaussian-smooth w(t) with sigma in t-units."""
    if sigma_t <= 0:
        return w.copy()
    dt = (t_grid[-1] - t_grid[0]) / max(len(t_grid) - 1, 1)
    return gaussian_filter1d(w, sigma=sigma_t / dt, mode='reflect')


def compute_density(w: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """p(t) = (1/v_t) / ∫₀¹ (1/v_s) ds  ≡  w(t) / ∫ w(s) ds  (trapezoidal)."""
    dt = np.diff(t_grid)
    Z  = float(np.sum(0.5 * (w[:-1] + w[1:]) * dt))
    return w / max(Z, 1e-12)


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_50k', type=str,
                        default='/nfs/ghome/live/cmarouani/FREE/outputs/'
                                'cifar10_curriculum/checkpoints/ckpt_step_0050000.pt')
    parser.add_argument('--ckpt_100k', type=str,
                        default='/nfs/ghome/live/cmarouani/FREE/outputs/'
                                'cifar10_curriculum/checkpoints/ckpt_step_0100000.pt')
    parser.add_argument('--smooth_sigma', type=float, default=0.05)
    parser.add_argument('--out_dir', type=str,
                        default='/nfs/ghome/live/cmarouani/FREE/outputs/cifar10/comparison')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── load speed curves ────────────────────────────────────────────────────
    ck50  = torch.load(args.ckpt_50k,  map_location='cpu')
    ck100 = torch.load(args.ckpt_100k, map_location='cpu')
    c50   = ck50['curriculum']
    c100  = ck100['curriculum']

    t1 = np.array(c50['speed1_t']);   v1 = np.array(c50['speed1_v'])
    t2 = np.array(c100['speed2_t']);  v2 = np.array(c100['speed2_v'])

    print(f'speed1 (@50K):  t {t1.shape}  v [{v1.min():.1f}, {v1.max():.1f}]')
    print(f'speed2 (@75K):  t {t2.shape}  v [{v2.min():.1f}, {v2.max():.1f}]')

    # ── common t grid ─────────────────────────────────────────────────────────
    t_lo = max(float(t1.min()), float(t2.min()))
    t_hi = min(float(t1.max()), float(t2.max()))
    t    = np.linspace(t_lo, t_hi, 500)

    v1_i = np.interp(t, t1, v1)
    v2_i = np.interp(t, t2, v2)

    # ── weighting ─────────────────────────────────────────────────────────────
    w1_raw = compute_weighting(v1_i, t)
    w2_raw = compute_weighting(v2_i, t)
    w1 = smooth_weighting(w1_raw, t, args.smooth_sigma)
    w2 = smooth_weighting(w2_raw, t, args.smooth_sigma)

    # ── density ───────────────────────────────────────────────────────────────
    d1   = compute_density(w1, t)
    d2   = compute_density(w2, t)
    d_un = np.ones_like(t) / (t[-1] - t[0])   # uniform reference

    # ── print summary ─────────────────────────────────────────────────────────
    for label, w, d in [('@50K', w1, d1), ('@75K', w2, d2)]:
        dt   = np.diff(t)
        m30  = float(np.sum(0.5*(d[:-1]+d[1:])*dt*(t[:-1]<=0.30)))
        m70  = float(np.sum(0.5*(d[:-1]+d[1:])*dt*(t[:-1]>=0.70)))
        print(f'{label}:  mass t<0.30 = {m30:.4f},  mass t>0.70 = {m70:.4f}')

    # ── plot: 3 rows × 2 columns ──────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle(
        'Curriculum FM — Hutchinson FR speed  (@50K = speed1,  @75K = speed2)\n'
        r'$w_t = \int_0^1 v_s\,ds\,/\,v_t$,  '
        r'$p(t) = (1/v_t)\,/\,\int_0^1(1/v_s)\,ds$',
        fontsize=12, y=1.01)

    COLS   = ['C3', 'C4']
    labels = ['@50K (speed1)', '@75K (speed2)']
    vs     = [v1_i, v2_i]
    ws_raw = [w1_raw, w2_raw]
    ws     = [w1, w2]
    ds     = [d1, d2]

    for col, (label, v, w_r, w, d, c) in enumerate(
            zip(labels, vs, ws_raw, ws, ds, COLS)):

        # Row 0 — speed (log scale)
        ax = axes[0, col]
        ax.semilogy(t, v, color=c, lw=2.0)
        ax.set_title(f'Speed  $v_t$  —  {label}')
        ax.set_xlabel('$t$');  ax.set_ylabel('$v_t$')
        ax.grid(True, alpha=0.3)

        # Row 1 — weighting (raw faint + smoothed solid)
        ax = axes[1, col]
        ax.plot(t, w_r, color=c, lw=1.0, alpha=0.3, label='raw')
        ax.plot(t, w,   color=c, lw=2.0,             label=f'smoothed ($\\sigma_t$={args.smooth_sigma})')
        ax.axhline(1.0, color='k', lw=0.7, ls='--', alpha=0.4, label='1')
        ax.set_title(f'Weighting  $w_t$  —  {label}')
        ax.set_xlabel('$t$');  ax.set_ylabel(r'$w_t = \int v_s\,ds\;/\;v_t$')
        ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)

        # Row 2 — sampling density
        ax = axes[2, col]
        ax.plot(t, d,    color=c, lw=2.0, label='$p(t)$')
        ax.plot(t, d_un, color='k', lw=0.9, ls='--', alpha=0.5, label='uniform')
        ax.fill_between(t, d, d_un, where=(d > d_un),
                        alpha=0.20, color=c, label='over-sampled')
        ax.fill_between(t, d, d_un, where=(d < d_un),
                        alpha=0.10, color='grey', label='under-sampled')
        ax.set_title(f'Sampling density  $p(t)$  —  {label}')
        ax.set_xlabel('$t$');  ax.set_ylabel(r'$p(t) = (1/v_t)\,/\,\int(1/v_s)\,ds$')
        ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = f'{args.out_dir}/curriculum_speed_plot.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nPlot saved → {out}')


if __name__ == '__main__':
    main()
