#!/usr/bin/env python3
"""
compare_speed_methods.py

Compare FR speed / weighting / sampling density from:

  (A) BATCH closed-form (OT / Score / FR):
        Exact formulas from the OT marginal  p_t = (1/N) sum_i N(t·x_i, (1-t)^2 I)
        differentiated via JVP.  Pre-computed and loaded from .npy files.
        Three variants:
          v_t^OT    = sqrt( E[||d/dt u_t(X_t)||^2] )
          v_t^Score = sqrt( E[||d/dt s_t(X_t)||^2] )
          v_t^FR    = sqrt( E[(d/dt log p_t(X_t))^2] )

  (B) MODEL Hutchinson (u_t^θ):
        Trained neural network v_θ; Hutchinson divergence estimator gives
          v_t^FR-model ≈ sqrt( E[(div v_θ(t, X_t))^2] )
        Sources (in order of preference):
          1. curriculum checkpoint metadata (pre-stored at step 50k / 100k)
          2. fresh inference via --ckpt_model (needs GPU + CIFAR-10)

Produces:
  compare_speed_methods.png   — comparison plot
  compare_metrics.txt         — quantitative metrics table

Run (fast, no GPU — uses stored curriculum metadata speeds):
  python compare_speed_methods.py

Run with fresh inference on a checkpoint (needs GPU + CIFAR-10):
  python compare_speed_methods.py \\
      --ckpt_model /path/to/ckpt_step_0100000.pt \\
      --data_dir   /tmp/cifar10
"""

import os, sys, argparse, time
sys.path.insert(0, '/nfs/ghome/live/cmarouani/FREE')

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr, pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch


# ─── shared helpers ───────────────────────────────────────────────────────────

def compute_weighting(speeds: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """w(t) = (∫₀¹ v_s ds) / v_t"""
    dt     = np.diff(t_grid)
    mean_v = float(np.sum(0.5 * (speeds[:-1] + speeds[1:]) * dt))
    return mean_v / np.maximum(speeds, 1e-12)


def smooth_weighting(w: np.ndarray, t_grid: np.ndarray,
                     sigma_t: float = 0.05) -> np.ndarray:
    """Gaussian-smooth w(t) with sigma in t-units (not grid steps)."""
    if sigma_t <= 0:
        return w.copy()
    dt = (t_grid[-1] - t_grid[0]) / max(len(t_grid) - 1, 1)
    return gaussian_filter1d(w, sigma=sigma_t / dt, mode='reflect')


def compute_density(w: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """p(t) = (1/v_t) / ∫₀¹ (1/v_s) ds  ≡  w(t) / ∫ w(s) ds  (trapezoidal)."""
    dt = np.diff(t_grid)
    Z  = np.sum(0.5 * (w[:-1] + w[1:]) * dt)
    return w / max(Z, 1e-12)


def interp_to(t_src, v_src, t_dst):
    """Interpolate (t_src, v_src) onto t_dst."""
    return np.interp(t_dst, t_src, v_src)


# ─── Hutchinson FR speed from a checkpoint ────────────────────────────────────

def compute_hutchinson_fr(ckpt_path: str, x1_ref: torch.Tensor,
                          t_grid: np.ndarray, n_hutch: int = 8,
                          B: int = 128, smooth_sigma: float = 3.0,
                          num_channel: int = 128) -> np.ndarray:
    """
    Load EMA from checkpoint, run Hutchinson divergence estimator over t_grid.
    Returns smoothed v_t^FR array of shape (len(t_grid),).
    """
    from torchcfm.models.unet.unet import UNetModelWrapper

    device = x1_ref.device
    net = UNetModelWrapper(
        dim=(3, 32, 32), num_res_blocks=2, num_channels=num_channel,
        channel_mult=[1, 2, 2, 2], num_heads=4, num_head_channels=64,
        attention_resolutions='16', dropout=0.0,
    ).to(device)
    ck = torch.load(ckpt_path, map_location=device)
    state = ck['ema'] if 'ema' in ck else ck
    net.load_state_dict(state)
    net.eval()

    N  = len(x1_ref)
    speeds_sq = np.zeros(len(t_grid), dtype=np.float64)
    t0 = time.time()

    for i, t_val in enumerate(t_grid):
        if i % 10 == 0:
            print(f'  Hutchinson [{i+1}/{len(t_grid)}]  t={t_val:.3f}', flush=True)
        t_v = float(np.clip(t_val, 1e-3, 1.0 - 1e-3))

        idx  = torch.randint(0, N, (B,), device=device)
        x1   = x1_ref[idx]
        x0   = torch.randn_like(x1)
        xt   = (t_v * x1 + (1 - t_v) * x0).detach().requires_grad_(True)
        t_ts = torch.full((B,), t_v, device=device)

        divs = []
        for _ in range(n_hutch):
            z  = torch.randint(0, 2, xt.shape, device=device).float() * 2 - 1
            v  = net(t_ts, xt)
            g  = torch.autograd.grad(
                (v * z).sum(), xt,
                create_graph=False, retain_graph=True
            )[0]
            divs.append((g * z).sum(dim=(1, 2, 3)).detach())

        div_mean = torch.stack(divs, 0).mean(0)   # (B,)
        speeds_sq[i] = (div_mean ** 2).mean().item()
        del xt, v, divs, div_mean

    print(f'  Done in {(time.time()-t0)/60:.1f} min', flush=True)
    raw = np.sqrt(np.maximum(speeds_sq, 0.0))
    return np.clip(gaussian_filter1d(raw, sigma=smooth_sigma), 1e-6, None)


# ─── metrics ──────────────────────────────────────────────────────────────────

def pearson_log(v_a, v_b):
    """Pearson correlation on log-normalised speeds."""
    a = np.log1p(v_a / v_a.mean())
    b = np.log1p(v_b / v_b.mean())
    r, _ = pearsonr(a, b)
    return float(r)


def spearman_rank(v_a, v_b):
    r, _ = spearmanr(v_a, v_b)
    return float(r)


def kl_divergence(p, q, t_grid):
    """KL(p || q) via trapezoidal integration."""
    dt  = np.diff(t_grid)
    log_ratio = np.log(np.maximum(p, 1e-30) / np.maximum(q, 1e-30))
    integrand = p * log_ratio
    return float(np.sum(0.5 * (integrand[:-1] + integrand[1:]) * dt))


def wasserstein1(p, q, t_grid):
    """W_1 distance between two 1D densities via |CDF_p - CDF_q|."""
    dt    = np.diff(t_grid)
    cdf_p = np.concatenate([[0.], np.cumsum(0.5 * (p[:-1] + p[1:]) * dt)])
    cdf_q = np.concatenate([[0.], np.cumsum(0.5 * (q[:-1] + q[1:]) * dt)])
    return float(np.sum(np.abs(cdf_p[:-1] - cdf_q[:-1]) * dt))


def peak_t(density, t_grid):
    """t where sampling density is highest."""
    return float(t_grid[np.argmax(density)])


def mass_early(density, t_grid, threshold=0.3):
    """Fraction of sampling mass in [t_lo, threshold]."""
    dt   = np.diff(t_grid)
    mask = t_grid[:-1] <= threshold
    total = np.sum(0.5 * (density[:-1] + density[1:]) * dt)
    early = np.sum(0.5 * (density[:-1] + density[1:]) * dt * mask)
    return float(early / max(total, 1e-12))


def mass_late(density, t_grid, threshold=0.7):
    """Fraction of sampling mass in [threshold, t_hi]."""
    dt   = np.diff(t_grid)
    mask = t_grid[:-1] >= threshold
    total = np.sum(0.5 * (density[:-1] + density[1:]) * dt)
    late  = np.sum(0.5 * (density[:-1] + density[1:]) * dt * mask)
    return float(late / max(total, 1e-12))


# ─── plot ─────────────────────────────────────────────────────────────────────

def make_plot(t_common, batch_curves, model_curves, out_path, smooth_sigma):
    """
    batch_curves : list of dicts (OT, Score, FR batch) — reference ground truth
    model_curves : list of dicts (Hutchinson on u_t^θ)
    Each dict has keys: label, color, lw, ls, speed, weight, density
    """
    d_unif = np.ones_like(t_common) / (t_common[-1] - t_common[0])
    all_curves = batch_curves + model_curves

    fig, axes = plt.subplots(3, 3, figsize=(17, 14))
    fig.suptitle(
        f'Batch closed-form (OT / Score / FR)  vs  Model Hutchinson ($u_t^\\theta$) — CIFAR-10\n'
        r'$w_t = \int_0^1 v_s\,ds\,/\,v_t$,  '
        r'$p(t) = (1/v_t)\,/\,\int_0^1 (1/v_s)\,ds$,  '
        f'smooth $\\sigma_t$={smooth_sigma}',
        fontsize=12, y=1.01)

    # ── (0,0) All speeds — log scale ─────────────────────────────────────────
    ax = axes[0, 0]
    for c in all_curves:
        ax.semilogy(t_common, c['speed'], color=c['color'], lw=c['lw'],
                    ls=c['ls'], label=c['label'])
    ax.set_title('Speed (log scale)')
    ax.set_xlabel('$t$'); ax.set_ylabel('speed')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── (0,1) Normalised speed shapes ────────────────────────────────────────
    ax = axes[0, 1]
    for c in all_curves:
        v_n = c['speed'] / c['speed'].mean()
        ax.plot(t_common, v_n, color=c['color'], lw=c['lw'],
                ls=c['ls'], label=c['label'])
    ax.set_title('Normalised speed  $v_t / \\bar{v}$')
    ax.set_xlabel('$t$'); ax.set_ylabel('$v_t / \\bar{v}$')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── (0,2) Scatter: log(v_FR_batch) vs log(v_model) ───────────────────────
    ax     = axes[0, 2]
    v_fr   = batch_curves[-1]   # last batch curve = FR
    x_ref  = np.log1p(v_fr['speed'] / v_fr['speed'].mean())
    for c in model_curves:
        r = pearson_log(v_fr['speed'], c['speed'])
        ax.scatter(x_ref, np.log1p(c['speed'] / c['speed'].mean()),
                   alpha=0.4, s=7, color=c['color'],
                   label=f'{c["label"]}  r={r:.3f}')
    lim = x_ref.max() * 1.05
    ax.plot([0, lim], [0, lim], 'k--', lw=0.8, alpha=0.5, label='$y=x$')
    ax.set_title('Scatter: $\\log(1+v_{FR}^{batch})$ vs $\\log(1+v^{model})$')
    ax.set_xlabel('FR batch (log-norm)'); ax.set_ylabel('Model (log-norm)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # ── (1,0) Weighting — batch curves ───────────────────────────────────────
    ax = axes[1, 0]
    for c in batch_curves:
        ax.plot(t_common, c['weight'], color=c['color'], lw=c['lw'],
                ls=c['ls'], label=c['label'])
    ax.axhline(1.0, color='k', lw=0.7, ls='--', alpha=0.4)
    ax.set_title(f'Weighting $w(t)$ — batch methods')
    ax.set_xlabel('$t$'); ax.set_ylabel('$w(t)$')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── (1,1) Weighting — model curves vs FR batch ───────────────────────────
    ax = axes[1, 1]
    ax.plot(t_common, batch_curves[-1]['weight'],
            color=batch_curves[-1]['color'], lw=2.0, ls='-',
            label=batch_curves[-1]['label'] + ' (ref)')
    for c in model_curves:
        ax.plot(t_common, c['weight'], color=c['color'], lw=c['lw'],
                ls=c['ls'], label=c['label'])
    ax.axhline(1.0, color='k', lw=0.7, ls='--', alpha=0.4)
    ax.set_title(f'Weighting $w(t)$ — model vs FR batch')
    ax.set_xlabel('$t$'); ax.set_ylabel('$w(t)$')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── (1,2) Weighting difference model - FR_batch ───────────────────────────
    ax    = axes[1, 2]
    w_ref = batch_curves[-1]['weight']
    for c in model_curves:
        ax.plot(t_common, c['weight'] - w_ref, color=c['color'], lw=c['lw'],
                ls=c['ls'], label=c['label'])
    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5, label='FR batch ref')
    ax.set_title('Weighting diff  $w_{model}(t) - w_{FR}^{batch}(t)$')
    ax.set_xlabel('$t$'); ax.set_ylabel('$\\Delta w(t)$')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── (2,0) Sampling density — batch curves ────────────────────────────────
    ax = axes[2, 0]
    for c in batch_curves:
        ax.plot(t_common, c['density'], color=c['color'], lw=c['lw'],
                ls=c['ls'], label=c['label'])
    ax.plot(t_common, d_unif, color='grey', lw=0.9, ls=':', label='Uniform')
    ax.set_title('Sampling density $p(t)$ — batch methods')
    ax.set_xlabel('$t$'); ax.set_ylabel('$p(t)$')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── (2,1) Sampling density — model curves vs FR batch ────────────────────
    ax = axes[2, 1]
    ax.plot(t_common, batch_curves[-1]['density'],
            color=batch_curves[-1]['color'], lw=2.0, ls='-',
            label=batch_curves[-1]['label'] + ' (ref)')
    for c in model_curves:
        ax.plot(t_common, c['density'], color=c['color'], lw=c['lw'],
                ls=c['ls'], label=c['label'])
    ax.plot(t_common, d_unif, color='grey', lw=0.9, ls=':', label='Uniform')
    ax.set_title('Sampling density $p(t)$ — model vs FR batch')
    ax.set_xlabel('$t$'); ax.set_ylabel('$p(t)$')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── (2,2) Density difference model - FR_batch ────────────────────────────
    ax    = axes[2, 2]
    d_ref = batch_curves[-1]['density']
    for c in model_curves:
        ax.plot(t_common, c['density'] - d_ref, color=c['color'], lw=c['lw'],
                ls=c['ls'], label=c['label'])
    ax.fill_between(t_common, t_common * 0, 0, alpha=0)
    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5, label='FR batch ref')
    ax.set_title('Density diff  $p_{model}(t) - p_{FR}^{batch}(t)$')
    ax.set_xlabel('$t$'); ax.set_ylabel('$\\Delta p(t)$')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot saved → {out_path}')


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--batch_dir', type=str,
                        default='/nfs/ghome/live/cmarouani/FREE/outputs/cifar10',
                        help='Dir with t_grid_v2, ot/score/fr_speed_v2 .npy files')
    parser.add_argument('--ckpt_50k',  type=str,
                        default='/nfs/ghome/live/cmarouani/FREE/outputs/cifar10_curriculum/'
                                'checkpoints/ckpt_step_0050000.pt')
    parser.add_argument('--ckpt_100k', type=str,
                        default='/nfs/ghome/live/cmarouani/FREE/outputs/cifar10_curriculum/'
                                'checkpoints/ckpt_step_0100000.pt')
    parser.add_argument('--ckpt_model', type=str, default=None,
                        help='Optional checkpoint for fresh Hutchinson inference (needs GPU)')
    parser.add_argument('--ckpt_model_label', type=str, default='Model (fresh Hutchinson)')
    parser.add_argument('--data_dir',  type=str, default='/tmp/cifar10_data')
    parser.add_argument('--n_ref',     type=int, default=2000)
    parser.add_argument('--n_hutch',   type=int, default=8)
    parser.add_argument('--B',         type=int, default=128)
    parser.add_argument('--T',         type=int, default=100)
    parser.add_argument('--num_channel', type=int, default=128)
    parser.add_argument('--smooth_sigma', type=float, default=0.05)
    parser.add_argument('--out_dir',   type=str,
                        default='/nfs/ghome/live/cmarouani/FREE/outputs/cifar10/comparison')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── 1. Load batch closed-form speeds (OT / Score / FR) ───────────────────
    print('\n=== Loading batch closed-form speeds ===')
    t_batch  = np.load(f'{args.batch_dir}/t_grid_v2.npy')
    v_ot     = np.load(f'{args.batch_dir}/ot_speed_v2.npy')
    v_score  = np.load(f'{args.batch_dir}/score_speed_v2.npy')
    v_fr_bat = np.load(f'{args.batch_dir}/fr_speed_v2.npy')
    print(f'  t_grid  : {t_batch.shape}  [{t_batch[0]:.3f}, {t_batch[-1]:.3f}]')
    print(f'  OT speed: [{v_ot.min():.3f}, {v_ot.max():.3f}]')
    print(f'  Score   : [{v_score.min():.3f}, {v_score.max():.3f}]')
    print(f'  FR      : [{v_fr_bat.min():.3f}, {v_fr_bat.max():.3f}]')

    # ── 2. Load curriculum model Hutchinson speeds from checkpoint metadata ───
    print('\n=== Loading model Hutchinson speeds (from curriculum metadata) ===')
    c50_t = c50_v = c100_t = c100_v = None

    if args.ckpt_50k and os.path.exists(args.ckpt_50k):
        ck   = torch.load(args.ckpt_50k, map_location='cpu')
        curr = ck.get('curriculum', {})
        if curr.get('speed1_t') is not None:
            c50_t = np.array(curr['speed1_t'])
            c50_v = np.array(curr['speed1_v'])
            print(f'  @50K  : t {c50_t.shape}  v [{c50_v.min():.2f}, {c50_v.max():.2f}]')
    else:
        print(f'  @50K  : not found')

    if args.ckpt_100k and os.path.exists(args.ckpt_100k):
        ck   = torch.load(args.ckpt_100k, map_location='cpu')
        curr = ck.get('curriculum', {})
        # prefer speed2 (re-estimated at 100K); fall back to speed1
        key_t = 'speed2_t' if curr.get('speed2_t') is not None else 'speed1_t'
        key_v = 'speed2_v' if curr.get('speed2_v') is not None else 'speed1_v'
        if curr.get(key_t) is not None:
            c100_t = np.array(curr[key_t])
            c100_v = np.array(curr[key_v])
            tag    = '@100K (speed2)' if 'speed2' in key_v else '@100K (speed1)'
            print(f'  {tag}: t {c100_t.shape}  v [{c100_v.min():.2f}, {c100_v.max():.2f}]')
    else:
        print(f'  @100K : not found')

    # ── 3. Optional fresh Hutchinson inference ────────────────────────────────
    hutch_t = hutch_v = None
    if args.ckpt_model and os.path.exists(args.ckpt_model):
        from torchvision import datasets, transforms
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'\n=== Fresh Hutchinson inference on {args.ckpt_model_label} ===')
        ds = datasets.CIFAR10(
            root=args.data_dir, train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,)*3, (0.5,)*3),
            ])
        )
        x1_ref = next(iter(
            torch.utils.data.DataLoader(ds, batch_size=args.n_ref, shuffle=True)
        ))[0].to(device)
        hutch_t = np.linspace(0.01, 0.95, args.T)
        hutch_v = compute_hutchinson_fr(
            args.ckpt_model, x1_ref, hutch_t,
            n_hutch=args.n_hutch, B=args.B, smooth_sigma=3.0,
            num_channel=args.num_channel,
        )
        np.save(f'{args.out_dir}/hutch_t.npy', hutch_t)
        np.save(f'{args.out_dir}/hutch_v.npy', hutch_v)
        print(f'  Saved hutch_t/v.npy → {args.out_dir}')
    elif os.path.exists(f'{args.out_dir}/hutch_t.npy'):
        hutch_t = np.load(f'{args.out_dir}/hutch_t.npy')
        hutch_v = np.load(f'{args.out_dir}/hutch_v.npy')
        print(f'\n=== Loaded cached Hutchinson from {args.out_dir} ===')
        print(f'  t {hutch_t.shape}  v [{hutch_v.min():.2f}, {hutch_v.max():.2f}]')

    # ── 4. Common t-grid ──────────────────────────────────────────────────────
    t_lo, t_hi = t_batch[1], t_batch[-1]   # skip t=0 (OT speed is tiny there)
    for arr in [c50_t, c100_t, hutch_t]:
        if arr is not None:
            t_lo = max(t_lo, float(arr.min()))
            t_hi = min(t_hi, float(arr.max()))

    t_common = np.linspace(t_lo, t_hi, 500)
    print(f'\n=== Common grid: [{t_lo:.4f}, {t_hi:.4f}], 500 pts ===')

    def make_curve(label, color, lw, ls, t_src, v_src):
        v = interp_to(t_src, v_src, t_common)
        w = smooth_weighting(compute_weighting(v, t_common),
                             t_common, args.smooth_sigma)
        d = compute_density(w, t_common)
        return dict(label=label, color=color, lw=lw, ls=ls,
                    speed=v, weight=w, density=d)

    # Batch curves (ground truth)
    batch_curves = [
        make_curve('OT  (batch)',    'C0', 2.0, '-',  t_batch, v_ot),
        make_curve('Score (batch)',  'C1', 2.0, '-',  t_batch, v_score),
        make_curve('FR  (batch)',    'C2', 2.5, '-',  t_batch, v_fr_bat),
    ]

    # Model Hutchinson curves
    model_curves = []
    if c50_t is not None:
        model_curves.append(make_curve('Model @50K  (Hutchinson)',
                                       'C3', 1.8, '--', c50_t, c50_v))
    if c100_t is not None:
        model_curves.append(make_curve('Model @100K (Hutchinson)',
                                       'C4', 1.8, ':',  c100_t, c100_v))
    if hutch_t is not None:
        model_curves.append(make_curve(args.ckpt_model_label,
                                       'C5', 1.8, '-.', hutch_t, hutch_v))

    if not model_curves:
        print('WARNING: no model curves found — nothing to compare against batch methods.')
        return

    # ── 5. Metrics ────────────────────────────────────────────────────────────
    fr_ref  = batch_curves[-1]   # FR batch is the primary reference
    v_ref   = fr_ref['speed']
    d_ref   = fr_ref['density']
    d_unif  = np.ones_like(t_common) / (t_common[-1] - t_common[0])

    col_names  = [c['label'][:22] for c in model_curves] + ['Uniform']
    col_vals_v = [c['speed']   for c in model_curves] + [d_unif * 0]   # placeholder
    col_vals_d = [c['density'] for c in model_curves] + [d_unif]

    hdr = f'\n{"Metric":<40}' + ''.join(f'  {n:<22}' for n in col_names)
    sep = '-' * (40 + 24 * len(col_names))
    print(hdr); print(sep)

    lines = [hdr + '\n' + sep]

    def row(name, vals):
        line = f'  {name:<38}' + ''.join(
            f'  {v:>22.4f}' if not (isinstance(v, float) and v != v)
            else f'  {"—":>22}' for v in vals)
        print(line); lines.append(line)

    row('Pearson r (log-norm speed, vs FR batch)',
        [pearson_log(v_ref, c['speed']) for c in model_curves] + [float('nan')])
    row('Spearman ρ (speed, vs FR batch)',
        [spearman_rank(v_ref, c['speed']) for c in model_curves] + [float('nan')])
    row('KL(model ∥ FR batch)',
        [kl_divergence(c['density'], d_ref, t_common) for c in model_curves] +
        [kl_divergence(d_unif, d_ref, t_common)])
    row('KL(FR batch ∥ model)',
        [kl_divergence(d_ref, c['density'], t_common) for c in model_curves] +
        [kl_divergence(d_ref, d_unif, t_common)])
    row('Wasserstein-1 (density)',
        [wasserstein1(c['density'], d_ref, t_common) for c in model_curves] +
        [wasserstein1(d_unif, d_ref, t_common)])
    row('Peak t of p(t)',
        [peak_t(c['density'], t_common) for c in model_curves] +
        [peak_t(d_unif, t_common)])
    row('Mass t < 0.30',
        [mass_early(c['density'], t_common) for c in model_curves] +
        [mass_early(d_unif, t_common)])
    row('Mass t > 0.70',
        [mass_late(c['density'], t_common) for c in model_curves] +
        [mass_late(d_unif, t_common)])

    print(f'\n  FR batch reference:')
    print(f'    Peak p(t) at t = {peak_t(d_ref, t_common):.4f}')
    print(f'    Mass t < 0.30  = {mass_early(d_ref, t_common):.4f}')
    print(f'    Mass t > 0.70  = {mass_late(d_ref, t_common):.4f}')

    metrics_path = f'{args.out_dir}/compare_metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nMetrics saved → {metrics_path}')

    # ── 6. Plot ────────────────────────────────────────────────────────────────
    plot_path = f'{args.out_dir}/compare_speed_methods.png'
    make_plot(t_common, batch_curves, model_curves, plot_path, args.smooth_sigma)


if __name__ == '__main__':
    main()
