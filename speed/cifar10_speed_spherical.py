#!/usr/bin/env python3
"""
cifar10_speed_spherical.py

Compute OT / Score / Fisher-Rao speed, weighting, schedule, and sampling
density for CIFAR-10 using the **spherical** interpolation path:

    X_t = cos(t) * X_0 + sin(t) * X_1,    t ∈ [0, π/2]
    X_0 ~ N(0, I),   X_1 ~ CIFAR-10

Marginal density:
    p_t(x) = (1/N) sum_i  N(x;  sin(t)*x1_i,  cos²(t)*I)

Mixture weights:
    w_i(x) = softmax_i( -||x - sin(t)*x1_i||² / (2 cos²(t)) )   [log-softmax]

Marginal fields:
    Wx1(x)   = sum_i w_i * x1_i                               [weighted mean]
    u_t(x)   = (Wx1 - sin(t)*x) / cos(t)                     [OT velocity]
    s_t(x)   = (sin(t)*Wx1 - x) / cos²(t)                    [score]
    logp_t(x)= logsumexp_i(-||x-sin(t)x1_i||²/2cos²t) - logN - (d/2)log2π - d·log(cos t)

Speed definitions (same as linear case, different path):
    v_OT(t)    = sqrt( E[ ||d/dt u_t(X_t)||²     ] )
    v_Score(t) = sqrt( E[ ||d/dt s_t(X_t)||²     ] )
    v_FR(t)    = sqrt( E[ (d/dt log p_t(X_t))²   ] )

All three time-derivatives are computed in a single JVP call per (epoch, t).

Default parameters chosen so that σ(t_max) = cos(t_max) ≈ 0.05 matches
the linear case σ(t_max) = 1 - 0.95 = 0.05.

Outputs saved to --out_dir (default outputs/cifar10_spherical/):
    t_grid_sph.npy
    {ot,score,fr}_{speed,weighting,schedule}_sph.npy
    cifar10_speed_spherical.png
"""

import os, sys, math, time, argparse

import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch
from torch.func import jvp
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/nfs/ghome/live/cmarouani/FREE')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS_T  = 1e-4   # clip cos(t) away from 0


# ─── data ─────────────────────────────────────────────────────────────────────

def load_cifar10(data_dir: str) -> torch.Tensor:
    """Load all 50 k CIFAR-10 training images → (50000, 3072) float32 in [-1,1] on CPU."""
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=tfm)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=5000, shuffle=False, num_workers=0)
    chunks = []
    for imgs, _ in loader:
        chunks.append(imgs.view(imgs.size(0), -1).float())
    return torch.cat(chunks, dim=0).cpu()   # (50000, 3072) on CPU


# ─── core: single JVP for all three fields ────────────────────────────────────

def compute_speeds_at_t(t_val: float, x1_all: torch.Tensor, B: int) -> tuple:
    """
    Estimate E[||du_t/dt||²], E[||ds_t/dt||²], E[(d logp_t/dt)²]
    for the spherical path at time t_val using B query + B reference images.

    Returns (sq_ot, sq_score, sq_fr) as Python floats.
    """
    N, d = x1_all.shape

    # Clip t so cos(t) stays safely positive
    t_clipped = float(np.clip(t_val, EPS_T, math.pi / 2 - EPS_T))
    t = torch.tensor(t_clipped, dtype=torch.float32, device=DEVICE)

    # ── query points x_t ~ p_t  (spherical path) ──────────────────────────────
    i_q   = torch.randint(0, N, (B,))
    x1_q  = x1_all[i_q].to(DEVICE)              # (B, d)
    x0_q  = torch.randn(B, d, device=DEVICE)
    cost0 = math.cos(t_clipped)
    sint0 = math.sin(t_clipped)
    xt    = (cost0 * x0_q + sint0 * x1_q).detach()   # (B, d) — fixed for JVP

    # ── reference images (independent draw) ────────────────────────────────────
    i_r   = torch.randint(0, N, (B,))
    x1_r  = x1_all[i_r].to(DEVICE).detach()     # (B, d)

    # ── precompute t-independent dot-product quantities ─────────────────────────
    xt_n2   = (xt    ** 2).sum(-1, keepdim=True)     # (B, 1)
    x1r_n2  = (x1_r  ** 2).sum(-1, keepdim=True)     # (B, 1)
    cross   = xt @ x1_r.T                             # (B, B)  <xt_i, x1_r_j>

    log_N_d2_log2pi = math.log(B) + 0.5 * d * math.log(2.0 * math.pi)

    # ── JVP function ────────────────────────────────────────────────────────────
    def f(t_: torch.Tensor) -> torch.Tensor:
        cost   = torch.cos(t_)
        sint   = torch.sin(t_)
        sigma2 = cost ** 2                                       # scalar

        # ||xt - sin(t)*x1_r||² = ||xt||² + sin²(t)||x1_r||² - 2sin(t)<xt,x1_r>
        dist2   = xt_n2 + sint ** 2 * x1r_n2.T - 2.0 * sint * cross   # (B,B)

        # log-softmax weights (numerically stable via logsumexp)
        log_w   = -dist2 / (2.0 * sigma2)               # (B,B) unnormalised
        lse     = torch.logsumexp(log_w, dim=-1)         # (B,)
        log_w_n = log_w - lse.unsqueeze(-1)              # (B,B) log-softmax
        w       = log_w_n.exp()                          # (B,B) weights
        Wx1     = w @ x1_r                               # (B,d) = sum_j w_j x1_r_j

        # OT velocity:  u_t(x) = (Wx1 - sin(t)*x) / cos(t)
        u = (Wx1 - sint * xt) / cost                     # (B,d)

        # Score:        s_t(x) = (sin(t)*Wx1 - x) / cos²(t)
        s = (sint * Wx1 - xt) / sigma2                   # (B,d)

        # log p_t(x) = lse - log N - (d/2) log(2π) - d * log(cos t)
        logp = lse - log_N_d2_log2pi - d * torch.log(cost)  # (B,)

        return torch.cat([u.reshape(-1), s.reshape(-1), logp])   # (2*B*d + B,)

    _, deriv = jvp(f, (t,), (torch.ones_like(t),))

    du_dt    = deriv[:B * d].reshape(B, d)
    ds_dt    = deriv[B * d : 2 * B * d].reshape(B, d)
    dlogp_dt = deriv[2 * B * d:]                            # (B,)

    sq_ot    = (du_dt    ** 2).mean(-1).mean().item()
    sq_score = (ds_dt    ** 2).mean(-1).mean().item()
    sq_fr    = (dlogp_dt ** 2).mean().item()

    return sq_ot, sq_score, sq_fr


# ─── weighting and schedule (same helpers as linear) ──────────────────────────

def compute_weighting(speeds: np.ndarray, w_cap: float = -1.0,
                      w_pow: float = 1.0) -> np.ndarray:
    """w(t) = v(t)^{-w_pow}, normalised so mean(w)=1, optionally capped."""
    w = np.maximum(speeds, 1e-12) ** (-w_pow)
    w = w / w.mean()
    if w_cap > 0.0:
        w = np.minimum(w, w_cap)
        w = w / w.mean()
    return w


def smooth_weighting(w: np.ndarray, t_grid: np.ndarray, sigma_t: float = 0.05) -> np.ndarray:
    """Gaussian-smooth w(t) with bandwidth sigma_t in t-units."""
    if sigma_t <= 0:
        return w.copy()
    dt = (t_grid[-1] - t_grid[0]) / max(len(t_grid) - 1, 1)
    sigma_pts = sigma_t / dt
    return gaussian_filter1d(w, sigma=sigma_pts, mode='reflect')


def compute_schedule(w: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """
    alpha(t) = (alpha^{-1})^{-1}(t)  where  (alpha^{-1})'(t) = 1/w(t).

    Integrates 1/w on t_grid, normalises to [0,1], then inverts.
    Queries at t/t_max so the result maps [t_min, t_max] → [t_min, t_max].
    """
    integrand = 1.0 / np.maximum(w, 1e-12)
    dt  = np.diff(t_grid)
    cum = np.concatenate(
        [[0.0], np.cumsum(0.5 * (integrand[:-1] + integrand[1:]) * dt)])
    if cum[-1] < 1e-12:
        return t_grid.copy()
    cum /= cum[-1]                                        # normalise to [0, 1]
    return np.interp(t_grid / t_grid[-1], cum, t_grid)   # alpha(t) ∈ [t_min, t_max]


def compute_density(w: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Normalised IS density p(t) = w(t) / int_0^T w(s) ds."""
    dt = np.diff(t_grid)
    Z  = np.sum(0.5 * (w[:-1] + w[1:]) * dt)
    return w / max(Z, 1e-12)


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--T',            type=int,   default=200,
                        help='Number of intervals (T+1 grid points)')
    parser.add_argument('--t_min',        type=float, default=0.01,
                        help='Lower bound of t grid (sin(t_min) ≈ t_min, safe near 0)')
    parser.add_argument('--t_max',        type=float, default=0.9,
                        help='Upper bound of t grid in radians. Default 0.9 avoids '
                             'the speed explosion near π/2≈1.571 (cos(0.9)≈0.62). '
                             'Use π/2-0.05≈1.521 for the full range.')
    parser.add_argument('--B',            type=int,   default=4096,
                        help='Batch size for query and reference draws')
    parser.add_argument('--n_epochs',     type=int,   default=1,
                        help='Independent epochs; median taken for robustness')
    parser.add_argument('--smooth_sigma', type=float, default=0.05,
                        help='Gaussian smoothing bandwidth in t-units (0 to disable)')
    parser.add_argument('--w_cap',        type=float, default=100.0,
                        help='Hard cap on normalised weighting (-1 to disable)')
    parser.add_argument('--w_pow',        type=float, default=0.5,
                        help='w ∝ v^{-w_pow}: 1.0=inverse, 0.5=sqrt-inverse, 0.0=uniform')
    parser.add_argument('--data_dir', type=str, default='/tmp/fm_results/data')
    parser.add_argument('--out_dir',  type=str,
                        default='/nfs/ghome/live/cmarouani/FREE/outputs/cifar10_spherical')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f'Device : {DEVICE}')
    if torch.cuda.is_available():
        print(f'GPU    : {torch.cuda.get_device_name(0)}')

    print('Loading CIFAR-10 …')
    x1_all = load_cifar10(args.data_dir)
    N, d   = x1_all.shape
    print(f'Loaded {N:,} images  d={d}')

    t_grid = np.linspace(args.t_min, args.t_max, args.T + 1)  # (T+1,)
    M      = len(t_grid)
    print(f'\nSpherical path:  X_t = cos(t)*X_0 + sin(t)*X_1')
    print(f't ∈ [{args.t_min:.4f}, {args.t_max:.4f}]  '
          f'(cos(t_max) = {math.cos(args.t_max):.4f})')
    print(f'Config: T={args.T}, B={args.B}, n_epochs={args.n_epochs}, '
          f'smooth_sigma={args.smooth_sigma}, w_cap={args.w_cap}, w_pow={args.w_pow}')
    print(f'Total JVP calls: {M * args.n_epochs:,}')

    sq_ot_all    = np.zeros((args.n_epochs, M), dtype=np.float64)
    sq_score_all = np.zeros((args.n_epochs, M), dtype=np.float64)
    sq_fr_all    = np.zeros((args.n_epochs, M), dtype=np.float64)

    t0 = time.time()
    for epoch in range(args.n_epochs):
        print(f'\nEpoch {epoch + 1}/{args.n_epochs}')
        for i, t_val in enumerate(tqdm(t_grid, ncols=80)):
            a, b, c = compute_speeds_at_t(t_val, x1_all, B=args.B)
            sq_ot_all[epoch, i]    = a
            sq_score_all[epoch, i] = b
            sq_fr_all[epoch, i]    = c

        elapsed = time.time() - t0
        eta     = elapsed / (epoch + 1) * (args.n_epochs - epoch - 1)
        print(f'  elapsed {elapsed/60:.1f} min  ETA {eta/60:.1f} min')

    sq_ot    = np.median(sq_ot_all,    axis=0)
    sq_score = np.median(sq_score_all, axis=0)
    sq_fr    = np.median(sq_fr_all,    axis=0)

    v_ot    = np.sqrt(np.maximum(sq_ot,    0.0))
    v_score = np.sqrt(np.maximum(sq_score, 0.0))
    v_fr    = np.sqrt(np.maximum(sq_fr,    0.0))

    elapsed = time.time() - t0
    print(f'\nTotal time: {elapsed / 60:.1f} min')
    print(f'OT    speed : [{v_ot.min():.4g},  {v_ot.max():.4g}]')
    print(f'Score speed : [{v_score.min():.4g}, {v_score.max():.4g}]')
    print(f'FR    speed : [{v_fr.min():.4g},    {v_fr.max():.4g}]')

    # ── weighting / schedule / density ────────────────────────────────────────
    w_ot    = compute_weighting(v_ot,    w_cap=args.w_cap, w_pow=args.w_pow)
    w_score = compute_weighting(v_score, w_cap=args.w_cap, w_pow=args.w_pow)
    w_fr    = compute_weighting(v_fr,    w_cap=args.w_cap, w_pow=args.w_pow)

    ws_ot    = smooth_weighting(w_ot,    t_grid, sigma_t=args.smooth_sigma)
    ws_score = smooth_weighting(w_score, t_grid, sigma_t=args.smooth_sigma)
    ws_fr    = smooth_weighting(w_fr,    t_grid, sigma_t=args.smooth_sigma)

    sched_ot    = compute_schedule(ws_ot,    t_grid)
    sched_score = compute_schedule(ws_score, t_grid)
    sched_fr    = compute_schedule(ws_fr,    t_grid)

    # ── save arrays ────────────────────────────────────────────────────────────
    base = args.out_dir
    np.save(f'{base}/t_grid_sph.npy',           t_grid)
    np.save(f'{base}/ot_speed_sph.npy',         v_ot)
    np.save(f'{base}/score_speed_sph.npy',      v_score)
    np.save(f'{base}/fr_speed_sph.npy',         v_fr)
    np.save(f'{base}/ot_weighting_sph.npy',     ws_ot)
    np.save(f'{base}/score_weighting_sph.npy',  ws_score)
    np.save(f'{base}/fr_weighting_sph.npy',     ws_fr)
    np.save(f'{base}/ot_schedule_sph.npy',      sched_ot)
    np.save(f'{base}/score_schedule_sph.npy',   sched_score)
    np.save(f'{base}/fr_schedule_sph.npy',      sched_fr)
    print(f'\nSaved .npy files → {base}/')

    # ── plot ───────────────────────────────────────────────────────────────────
    METHODS   = ['OT', 'Score', 'Fisher-Rao']
    COLORS    = ['C0', 'C1', 'C2']
    speeds_l  = [v_ot,     v_score,    v_fr]
    weights_l = [w_ot,     w_score,    w_fr]
    ws_l      = [ws_ot,    ws_score,   ws_fr]
    scheds_l  = [sched_ot, sched_score, sched_fr]

    dens_ot    = compute_density(ws_ot,    t_grid)
    dens_score = compute_density(ws_score, t_grid)
    dens_fr    = compute_density(ws_fr,    t_grid)
    dens_l     = [dens_ot, dens_score, dens_fr]

    # Normalise t to [0, 1] so schedule plots are directly comparable to the linear case
    t_norm = t_grid / t_grid[-1]                          # ∈ [0, 1]
    t_range_norm = t_norm[-1] - t_norm[0]                # ≈ 1
    uniform_density_norm = np.ones_like(t_grid) / t_range_norm  # density on [0,1]

    fig, axes = plt.subplots(4, 3, figsize=(16, 16))
    fig.suptitle(
        f'CIFAR-10 — Spherical path  $X_t = \\cos(t)X_0 + \\sin(t)X_1$\n'
        f'T={args.T}, t∈[{args.t_min:.3f}, {args.t_max:.3f}] rad  '
        f'(schedule normalised to [0,1]), B={args.B}, '
        f'{args.n_epochs} epoch(s) (median), '
        f'smooth_sigma={args.smooth_sigma}, w_cap={args.w_cap}, w_pow={args.w_pow}',
        fontsize=11)

    for col, (name, v, w, ws, sched, dens, c) in enumerate(
            zip(METHODS, speeds_l, weights_l, ws_l, scheds_l, dens_l, COLORS)):

        # Row 0: speed  (raw t axis in radians)
        ax = axes[0, col]
        ax.plot(t_grid, v, color=c, lw=1.3)
        ax.set_title(f'{name}  —  speed $v_t$')
        ax.set_xlabel('$t$ (rad)')
        ax.set_ylabel('$v_t$')
        ax.grid(True, alpha=0.3)

        # Row 1: weighting — raw (faint) + smoothed (solid)
        ax = axes[1, col]
        ax.plot(t_grid, w,  color=c, lw=1.0, alpha=0.35, label='raw')
        ax.plot(t_grid, ws, color=c, lw=1.8,              label='smoothed')
        ax.axhline(1.0, color='k', lw=0.7, ls='--', alpha=0.4)
        ax.set_title(f'{name}  —  weighting $w(t)$')
        ax.set_xlabel('$t$ (rad)')
        ax.set_ylabel('$w(t)$')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Row 2: schedule alpha(s), s = t/t_max ∈ [0,1] → alpha(s) ∈ [0,1]
        sched_norm = sched / t_grid[-1]                  # normalise output to [0,1]
        ax = axes[2, col]
        ax.plot(t_norm, sched_norm, color=c, lw=1.3, label=name)
        ax.plot([0, 1],  [0, 1],   'k--',   lw=0.8, alpha=0.4, label='identity')
        ax.set_title(f'{name}  —  schedule $\\alpha(s)$,  $s=t/t_{{\\max}}$')
        ax.set_xlabel('$s = t / t_{\\max}$  (normalised, $\\in[0,1]$)')
        ax.set_ylabel('$\\alpha(s)$')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Row 3: sampling density p(s) on normalised [0,1] axis
        # Rescale density: p_norm(s) = p(t) * t_max  (change of variables)
        dens_norm = dens * t_grid[-1]
        ax = axes[3, col]
        ax.plot(t_norm, dens_norm, color=c, lw=1.8, label='$p(s)$')
        ax.plot(t_norm, uniform_density_norm, 'k--', lw=0.8, alpha=0.5, label='uniform')
        ax.fill_between(t_norm, dens_norm, uniform_density_norm,
                        where=(dens_norm > uniform_density_norm),
                        alpha=0.18, color=c,      label='over-sampled')
        ax.fill_between(t_norm, dens_norm, uniform_density_norm,
                        where=(dens_norm < uniform_density_norm),
                        alpha=0.18, color='grey',  label='under-sampled')
        ax.set_title(f'{name}  —  sampling density $p(s)$')
        ax.set_xlabel('$s = t / t_{\\max}$  (normalised, $\\in[0,1]$)')
        ax.set_ylabel('$p(s)$')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = f'{base}/cifar10_speed_spherical.png'
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved plot → {plot_path}')
    print('Done!')


if __name__ == '__main__':
    main()
