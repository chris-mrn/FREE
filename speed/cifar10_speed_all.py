#!/usr/bin/env python3
"""
cifar10_speed_all.py

Compute OT / Score / Fisher-Rao speed, weighting, and schedule for CIFAR-10.

  T           = 500   (501 time points, linspace(0, t_max, T+1))
  t_max       = 0.95  (avoid blowup / high-variance region near t=1)
  B           = 4096  (query batch = reference batch size)
  n_epochs    = 1     (independent epochs; MEDIAN used for robustness)
  smooth_sigma= 0.05  (Gaussian bandwidth in t-units applied to weighting before schedule)
  w_cap       = 3.5   (hard cap on normalised weighting)

Memory-efficient via dot-product trick for squared distances (avoids (B,N,d) tensor).
Numerically stable via logsumexp throughout.
All three speeds computed in a single combined JVP per (epoch, t).

Robustness notes:
  - t_max < 1 avoids the O(1/(1-t)^2) blowup region where estimates are noisy.
  - Median over epochs is robust to occasional outlier batches near t_max.
  - Gaussian smoothing of the speed curve before arc-length integration prevents
    a noisy spike at t_max from collapsing the schedule into a near-Dirac.
  - Weighting cap prevents extreme w(t)=1/v(t) values (e.g. at small t where v
    is small) from dominating the weighted loss.

Speed definitions:
  v_OT(t)    = sqrt( E[ ||d/dt u_t(X_t)||^2     ] )
  v_Score(t) = sqrt( E[ ||d/dt s_t(X_t)||^2     ] )
  v_FR(t)    = sqrt( E[ (d/dt log p_t(X_t))^2   ] )

where u_t is marginal OT velocity, s_t = grad_x log p_t, and X_t ~ p_t.

Formulas (TargetCFM, sigma=0, source = N(0,I)):
  p_t(x)  = (1/N) sum_i N(x; t*x1_i, (1-t)^2 I)
  w_i(x)  = softmax_i( -||x - t*x1_i||^2 / 2(1-t)^2 )   [log-softmax trick]
  u_t(x)  = ( sum_i w_i * x1_i  - x ) / (1-t)
  s_t(x)  = ( t * sum_i w_i * x1_i - x ) / (1-t)^2
  logp_t  = logsumexp_i(-||x-t*x1_i||^2 / 2(1-t)^2) - logN - (d/2)log(2pi) - d*log(1-t)

Outputs saved to --out_dir:
  t_grid_v2.npy
  {ot,score,fr}_{speed,weighting,schedule}_v2.npy
  cifar10_speed_all.png
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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS_T  = 1e-3   # clip t away from 0 and 1


# ─── data ─────────────────────────────────────────────────────────────────────

def load_cifar10(data_dir: str) -> torch.Tensor:
    """Load all 50 k CIFAR-10 training images → (50000, 3072) float32 in [-1,1] on CPU."""
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=tfm)
    # Load in chunks of 5000 to avoid large collation peak
    loader = torch.utils.data.DataLoader(
        ds, batch_size=5000, shuffle=False, num_workers=0)
    chunks = []
    for imgs, _ in loader:
        chunks.append(imgs.view(imgs.size(0), -1).float())
    return torch.cat(chunks, dim=0).cpu()   # keep on CPU, (50000, 3072)


# ─── core: single JVP for all three fields ────────────────────────────────────

def compute_speeds_at_t(t_val: float, x1_all: torch.Tensor, B: int) -> tuple:
    """
    Estimate E[||du_t/dt||^2], E[||ds_t/dt||^2], E[(d logp_t/dt)^2]
    at time t_val using B query points and B reference images.

    Returns (sq_ot, sq_score, sq_fr) as Python floats.
    """
    N, d = x1_all.shape

    t = torch.tensor(float(np.clip(t_val, EPS_T, 1.0 - EPS_T)),
                     dtype=torch.float32, device=DEVICE)

    # ── query points x_t ~ p_t ───────────────────────────────────────────────
    i_q  = torch.randint(0, N, (B,))
    x1_q = x1_all[i_q].to(DEVICE)               # (B, d) — CPU→GPU
    x0_q = torch.randn(B, d, device=DEVICE)
    xt   = (t * x1_q + (1 - t) * x0_q).detach()  # (B, d) — fixed for JVP

    # ── reference images for the mixture (independent draw) ──────────────────
    i_r  = torch.randint(0, N, (B,))
    x1_r = x1_all[i_r].to(DEVICE).detach()      # (B, d) — CPU→GPU, fixed for JVP

    # ── precompute t-independent quantities (zero tangent in JVP) ─────────────
    # squared-distance trick: ||xt - t*x1_r||^2 = ||xt||^2 + t^2||x1_r||^2 - 2t <xt,x1_r>
    xt_n2   = (xt   ** 2).sum(-1, keepdim=True)     # (B, 1)
    x1r_n2  = (x1_r ** 2).sum(-1, keepdim=True)     # (B, 1)
    cross   = xt @ x1_r.T                            # (B, B) — inner products

    # log N + (d/2) log(2pi): Python float constant, zero tangent
    log_N_d2_log2pi = math.log(B) + 0.5 * d * math.log(2.0 * math.pi)

    # ── JVP function: differentiate all three fields w.r.t. t simultaneously ──
    def f(t_: torch.Tensor) -> torch.Tensor:
        sigma2 = (1.0 - t_) ** 2                             # scalar

        # ||xt - t_ * x1_r||^2  shape (B, B)
        dist2   = xt_n2 + t_ ** 2 * x1r_n2.T - 2.0 * t_ * cross

        # log-softmax weights (numerically stable)
        log_w   = -dist2 / (2.0 * sigma2)                   # (B, B) unnorm
        lse     = torch.logsumexp(log_w, dim=-1)             # (B,)
        log_w_n = log_w - lse.unsqueeze(-1)                  # (B, B) log-softmax
        w       = log_w_n.exp()                              # (B, B) weights
        Wx1     = w @ x1_r                                   # (B, d) = sum_i w_i * x1_r_i

        # OT velocity: u_t(x) = (W·x1 - x) / (1-t)
        u = (Wx1 - xt) / (1.0 - t_)                         # (B, d)

        # Score: s_t(x) = (t·W·x1 - x) / (1-t)^2
        s = (t_ * Wx1 - xt) / sigma2                        # (B, d)

        # log p_t(x) = lse - log N - (d/2) log(2pi) - d*log(1-t)
        logp = lse - log_N_d2_log2pi - d * torch.log(1.0 - t_)   # (B,)

        # pack into a single flat tensor for combined JVP
        return torch.cat([u.reshape(-1), s.reshape(-1), logp])   # (2*B*d + B,)

    _, deriv = jvp(f, (t,), (torch.ones_like(t),))

    du_dt    = deriv[:B * d].reshape(B, d)
    ds_dt    = deriv[B * d : 2 * B * d].reshape(B, d)
    dlogp_dt = deriv[2 * B * d :]                            # (B,)

    sq_ot    = (du_dt    ** 2).mean(-1).mean().item()   # mean over d → per-dim normalised
    sq_score = (ds_dt    ** 2).mean(-1).mean().item()   # mean over d → per-dim normalised
    sq_fr    = (dlogp_dt ** 2).mean().item()             # scalar, no d factor

    return sq_ot, sq_score, sq_fr


# ─── weighting and schedule ───────────────────────────────────────────────────

def compute_weighting(speeds: np.ndarray, w_cap: float = -1.0,
                      w_pow: float = 1.0) -> np.ndarray:
    """w(t) = v(t)^{-w_pow}, normalised so mean(w)=1, optionally capped.

    w_pow=1.0 : standard inverse speed (w → 0 fast as v → inf)
    w_pow=0.5 : square-root inverse (much less extreme near t=1)
    w_pow=0.0 : uniform weighting
    """
    w = np.maximum(speeds, 1e-12) ** (-w_pow)
    w = w / w.mean()
    if w_cap > 0.0:
        w = np.minimum(w, w_cap)
        w = w / w.mean()                    # re-normalise after cap
    return w


def smooth_weighting(w: np.ndarray, t_grid: np.ndarray, sigma_t: float = 0.05) -> np.ndarray:
    """Gaussian-smooth w(t) with reflection padding.

    sigma_t is the bandwidth in t-units (not grid steps), so smoothing is
    independent of T.  sigma_t=0.05 means a 1-sigma radius of 5% of [0, t_max].
    """
    if sigma_t <= 0:
        return w.copy()
    dt = (t_grid[-1] - t_grid[0]) / max(len(t_grid) - 1, 1)
    sigma_pts = sigma_t / dt
    return gaussian_filter1d(w, sigma=sigma_pts, mode='reflect')


def compute_schedule(w: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """
    alpha(t) = (alpha^{-1})^{-1}(t)  where  (alpha^{-1})'(t) = 1/w(t).

    Integrates 1/w on t_grid, normalises to [0,1], then numerically inverts.
    Uses the smoothed weighting so the schedule is stable near t_max.
    """
    integrand = 1.0 / np.maximum(w, 1e-12)
    dt  = np.diff(t_grid)
    cum = np.concatenate(
        [[0.0], np.cumsum(0.5 * (integrand[:-1] + integrand[1:]) * dt)])
    if cum[-1] < 1e-12:
        return t_grid.copy()
    cum /= cum[-1]                                        # normalise to [0, 1]
    return np.interp(t_grid / t_grid[-1], cum, t_grid)    # alpha(t), query at t/t_max ∈ [0,1]


def compute_density(w: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Normalised importance-sampling density p(t) = w(t) / int_0^T w(s) ds.

    This is the density from which t is sampled during training when using
    importance sampling instead of a weighted loss.  It satisfies:
        E_{t~p}[ loss(t) ] = E_{t~U}[ w(t) * loss(t) ]
    so both formulations are equivalent in expectation.
    """
    dt  = np.diff(t_grid)
    Z   = np.sum(0.5 * (w[:-1] + w[1:]) * dt)   # trapezoidal normalisation
    return w / max(Z, 1e-12)


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--T',            type=int,   default=500,
                        help='Number of intervals (T+1 grid points)')
    parser.add_argument('--t_max',        type=float, default=0.95,
                        help='Upper bound of t grid (< 1 to avoid blowup near t=1)')
    parser.add_argument('--B',            type=int,   default=4096,
                        help='Batch size for query and reference')
    parser.add_argument('--n_epochs',     type=int,   default=1,
                        help='Number of independent epochs; median used for robustness')
    parser.add_argument('--smooth_sigma', type=float, default=0.05,
                        help='Gaussian sigma in t-units (not grid steps) for smoothing '
                             'weighting before schedule integration (0 to disable)')
    parser.add_argument('--w_cap',        type=float, default=100.0,
                        help='Hard cap on normalised weighting (-1 to disable)')
    parser.add_argument('--w_pow',        type=float, default=0.5,
                        help='Exponent for speed→weighting: w ∝ v^{-w_pow}. '
                             '1.0=standard inverse, 0.5=sqrt-inverse, 0.0=uniform')
    parser.add_argument('--data_dir', type=str, default='/tmp/fm_results/data')
    parser.add_argument('--out_dir',  type=str,
                        default='/nfs/ghome/live/cmarouani/FREE/outputs/cifar10')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f'Device : {DEVICE}')
    if torch.cuda.is_available():
        print(f'GPU    : {torch.cuda.get_device_name(0)}')

    print('Loading CIFAR-10 …')
    x1_all = load_cifar10(args.data_dir)        # (50000, 3072) on GPU
    N, d   = x1_all.shape
    print(f'Loaded {N:,} images  d={d}')

    t_grid = np.linspace(0.0, args.t_max, args.T + 1)  # (T+1,)
    M      = len(t_grid)

    print(f'\nConfig: T={args.T}, t_max={args.t_max}, B={args.B}, '
          f'n_epochs={args.n_epochs}, smooth_sigma={args.smooth_sigma}, '
          f'w_cap={args.w_cap}')
    print(f'Total JVP calls : {M * args.n_epochs:,}  '
          f'(≈{args.B} query pts × {args.B} ref pts × {M * args.n_epochs} calls)')

    # Per-epoch accumulators — we keep all epochs for median computation
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

    # Median over epochs: robust to outlier batches (especially near t_max)
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

    # Weighting, smoothing, and schedule for each method
    w_ot    = compute_weighting(v_ot,    w_cap=args.w_cap, w_pow=args.w_pow)
    w_score = compute_weighting(v_score, w_cap=args.w_cap, w_pow=args.w_pow)
    w_fr    = compute_weighting(v_fr,    w_cap=args.w_cap, w_pow=args.w_pow)

    ws_ot    = smooth_weighting(w_ot,    t_grid, sigma_t=args.smooth_sigma)
    ws_score = smooth_weighting(w_score, t_grid, sigma_t=args.smooth_sigma)
    ws_fr    = smooth_weighting(w_fr,    t_grid, sigma_t=args.smooth_sigma)

    sched_ot    = compute_schedule(ws_ot,    t_grid)
    sched_score = compute_schedule(ws_score, t_grid)
    sched_fr    = compute_schedule(ws_fr,    t_grid)

    # ── save ──────────────────────────────────────────────────────────────────
    base = args.out_dir
    np.save(f'{base}/t_grid_v2.npy',          t_grid)
    np.save(f'{base}/ot_speed_v2.npy',        v_ot)
    np.save(f'{base}/score_speed_v2.npy',     v_score)
    np.save(f'{base}/fr_speed_v2.npy',        v_fr)
    np.save(f'{base}/ot_weighting_v2.npy',    ws_ot)
    np.save(f'{base}/score_weighting_v2.npy', ws_score)
    np.save(f'{base}/fr_weighting_v2.npy',    ws_fr)
    np.save(f'{base}/ot_schedule_v2.npy',     sched_ot)
    np.save(f'{base}/score_schedule_v2.npy',  sched_score)
    np.save(f'{base}/fr_schedule_v2.npy',     sched_fr)
    print(f'\nSaved .npy files → {base}/')

    # ── plot ──────────────────────────────────────────────────────────────────
    METHODS   = ['OT', 'Score', 'Fisher-Rao']
    COLORS    = ['C0', 'C1', 'C2']
    speeds_l  = [v_ot,     v_score,    v_fr]
    weights_l = [w_ot,     w_score,    w_fr]
    ws_l      = [ws_ot,    ws_score,   ws_fr]
    scheds_l  = [sched_ot, sched_score, sched_fr]

    # Compute sampling densities from smoothed weightings
    dens_ot    = compute_density(ws_ot,    t_grid)
    dens_score = compute_density(ws_score, t_grid)
    dens_fr    = compute_density(ws_fr,    t_grid)
    dens_l     = [dens_ot, dens_score, dens_fr]
    # Uniform density for reference
    uniform_density = np.ones_like(t_grid) / (t_grid[-1] - t_grid[0])

    fig, axes = plt.subplots(4, 3, figsize=(16, 16))
    fig.suptitle(
        f'CIFAR-10 — OT / Score / Fisher-Rao\n'
        f'T={args.T}, t_max={args.t_max}, B={args.B}, {args.n_epochs} epochs '
        f'(median), smooth_sigma={args.smooth_sigma}, w_cap={args.w_cap}, w_pow={args.w_pow}',
        fontsize=11)

    for col, (name, v, w, ws, sched, dens, c) in enumerate(
            zip(METHODS, speeds_l, weights_l, ws_l, scheds_l, dens_l, COLORS)):

        # Row 0: speed
        ax = axes[0, col]
        ax.plot(t_grid, v, color=c, lw=1.3)
        ax.set_title(f'{name}  —  speed $v_t$')
        ax.set_xlabel('$t$')
        ax.set_ylabel('$v_t$')
        ax.grid(True, alpha=0.3)

        # Row 1: weighting — raw (faint) + smoothed (solid)
        ax = axes[1, col]
        ax.plot(t_grid, w,  color=c, lw=1.0, alpha=0.35, label='raw')
        ax.plot(t_grid, ws, color=c, lw=1.8,              label='smoothed')
        ax.axhline(1.0, color='k', lw=0.7, ls='--', alpha=0.4)
        ax.set_title(f'{name}  —  weighting $w(t)$')
        ax.set_xlabel('$t$')
        ax.set_ylabel('$w(t)$')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Row 2: schedule alpha(t), derived from smoothed weighting
        ax = axes[2, col]
        ax.plot(t_grid, sched, color=c, lw=1.3, label=name)
        ax.plot(t_grid, t_grid, 'k--', lw=0.8, alpha=0.4, label='identity')
        ax.set_title(f'{name}  —  schedule $\\alpha(t)$')
        ax.set_xlabel('$t$')
        ax.set_ylabel('$\\alpha(t)$')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Row 3: sampling density p(t) = w(t) / int w(s) ds
        ax = axes[3, col]
        ax.plot(t_grid, dens, color=c, lw=1.8, label='$p(t)$')
        ax.plot(t_grid, uniform_density, 'k--', lw=0.8, alpha=0.5, label='uniform')
        ax.fill_between(t_grid, dens, uniform_density,
                        where=(dens > uniform_density),
                        alpha=0.18, color=c, label='over-sampled')
        ax.fill_between(t_grid, dens, uniform_density,
                        where=(dens < uniform_density),
                        alpha=0.18, color='grey', label='under-sampled')
        ax.set_title(f'{name}  —  sampling density $p(t)$')
        ax.set_xlabel('$t$')
        ax.set_ylabel('$p(t) = w(t)/\\int w$')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = f'{base}/cifar10_speed_all.png'
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved plot → {plot_path}')


if __name__ == '__main__':
    main()
