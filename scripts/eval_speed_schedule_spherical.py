#!/usr/bin/env python3
"""
eval_speed_schedule_spherical.py

Compares Euler sampling schedules for the trained spherical-FM model:

  1. Linear:          t_k = k/n * (pi/2)
  2. Speed-adaptive:  t_k = F^{-1}(k/n)       where F = CDF of q(t) ~ v_t
  3. Mixed(alpha):    t_k = (1-alpha)*t_k^lin + alpha*t_k^speed   for alpha in --alphas

The mixed schedule interpolates linearly between the two extremes, giving the
best of both: enough early steps (from linear) while concentrating extra steps
where the speed is high (from speed-adaptive).

Speed profile v_t = sqrt(E[||partial_t u_t^theta(X_t)||^2]) is either:
  (a) Loaded from pre-computed .npy files (default)
  (b) Estimated from the trained model via JVP  (--use_model_speed)

Any time derivative of the model is computed with torch.func.jvp (forward-mode AD),
never with finite differences.

Outputs (all in --out_dir):
    schedule_comparison.csv
    report.txt
    speed_and_schedule.png
    metrics_comparison.png   (FID/KID/IS vs NFE for all schemes)
    alpha_sweep.png          (FID vs alpha for each NFE)
"""

import os, sys, math, time, csv, argparse
import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import datasets, transforms

sys.path.insert(0, '/nfs/ghome/live/cmarouani/FREE')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'training'))

from torchcfm.models.unet.unet import UNetModelWrapper
from evaluation.metrics import InceptionMetrics

T_MAX = math.pi / 2   # spherical path: t in [0, pi/2]


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(ckpt_path, num_channel, device):
    net = UNetModelWrapper(
        dim=(3, 32, 32), num_res_blocks=2, num_channels=num_channel,
        channel_mult=[1, 2, 2, 2], num_heads=4, num_head_channels=64,
        attention_resolutions='16', dropout=0.0,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    key = 'ema' if 'ema' in ckpt else 'net'
    state = ckpt[key]
    if any(k.startswith('module.') for k in state):
        state = {k[7:]: v for k, v in state.items()}
    net.load_state_dict(state)
    net.eval()
    print(f'Loaded {"EMA " if key == "ema" else ""}model  step={ckpt.get("step", "?")}')
    return net


# ── Model speed via JVP ────────────────────────────────────────────────────────
# Time derivatives of model outputs are always computed via forward-mode AD (JVP),
# never by finite differences.

def estimate_model_speed_jvp(model, t_grid, data_tensor, B, n_epochs, device):
    """
    Estimate v_t = sqrt(E[||partial_t u_t^theta(X_t)||^2]) on t_grid.

    Uses torch.func.jvp (forward-mode AD) with functional_call so that the
    function passed to jvp is purely functional.

    Args:
        model       : trained EMA UNet in eval()
        t_grid      : (M,) numpy array of time points in (0, pi/2)
        data_tensor : (N, C, H, W) CIFAR-10 images on CPU, float, [-1, 1]
        B           : batch size per time step
        n_epochs    : independent epochs; median taken over epochs
        device      : torch device

    Returns:
        v_t : (M,) numpy array of speed values
    """
    from torch.func import jvp, functional_call

    N, C, H, W = data_tensor.shape
    params  = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    sq_all = np.zeros((n_epochs, len(t_grid)), dtype=np.float64)

    for epoch in range(n_epochs):
        print(f'  JVP speed epoch {epoch + 1}/{n_epochs}')
        for i, t_val in enumerate(tqdm(t_grid, ncols=80, leave=False)):
            t_val = float(np.clip(t_val, 1e-4, T_MAX - 1e-4))

            # Sample X_t ~ p_t from the spherical path
            idx = torch.randint(0, N, (B,))
            x1  = data_tensor[idx].to(device)
            x0  = torch.randn(B, C, H, W, device=device)
            xt  = (math.cos(t_val) * x0 + math.sin(t_val) * x1).detach()

            # JVP: d/dt u_theta(t, xt) at fixed xt
            # f is purely functional: only t_scalar varies; xt/params/buffers are closed over
            def f(t_scalar):
                t_b = t_scalar.view(1).expand(B)
                return functional_call(model, (params, buffers), (t_b, xt))

            t_tensor = torch.tensor(t_val, dtype=torch.float32, device=device)
            _, du_dt = jvp(f, (t_tensor,), (torch.ones_like(t_tensor),))
            # du_dt : (B, C, H, W)

            sq_all[epoch, i] = du_dt.detach().pow(2).mean().item()

    sq_med = np.median(sq_all, axis=0)
    return np.sqrt(np.maximum(sq_med, 0.0))


# ── CDF and schedule ───────────────────────────────────────────────────────────

def build_speed_schedule(t_full, v_full, n_steps, smooth_sigma=0.05):
    """
    Build a speed-adaptive schedule of n_steps+1 time points.

    q(t) proportional to v_t, F(t) = integral_0^t v_s ds / Z.
    t_k = F^{-1}(k/n),  k = 0 .. n_steps.

    Returns:
        t_schedule : (n_steps+1,) numpy array in [0, pi/2]
    """
    v = v_full.copy()
    if smooth_sigma > 0:
        span = t_full[-1] - t_full[0]
        dt_avg = span / max(len(t_full) - 1, 1)
        v = gaussian_filter1d(v, sigma=smooth_sigma / dt_avg, mode='reflect')

    v = np.maximum(v, 1e-12)
    dt  = np.diff(t_full)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (v[:-1] + v[1:]) * dt)])
    cum /= cum[-1]   # normalise to [0, 1]

    u_vals     = np.linspace(0.0, 1.0, n_steps + 1)
    t_schedule = np.interp(u_vals, cum, t_full)
    t_schedule[0]  = 0.0
    t_schedule[-1] = T_MAX
    return t_schedule


# ── Euler sampler with arbitrary schedule ──────────────────────────────────────

@torch.no_grad()
def euler_sample_schedule(model, n_samples, t_schedule, device, bs=256):
    """
    Euler sampler for the spherical path with a custom time schedule.

    t_schedule : 1-D array, t_schedule[0]=0, t_schedule[-1]=pi/2, length n+1.
    """
    model.eval()
    t_s = torch.tensor(t_schedule, dtype=torch.float32, device=device)
    out = []
    for i in range(0, n_samples, bs):
        b = min(bs, n_samples - i)
        x = torch.randn(b, 3, 32, 32, device=device)
        for k in range(len(t_s) - 1):
            t_k  = t_s[k].item()
            dt_k = (t_s[k + 1] - t_s[k]).item()
            tv   = torch.full((b,), t_k, device=device)
            x    = x + model(tv, x) * dt_k
        out.append(x.cpu())
    return torch.cat(out, 0)


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_speed_and_schedules(t_full, v_full, out_dir, nfe_list, source_label):
    v = np.maximum(v_full, 1e-12)
    dt  = np.diff(t_full)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (v[:-1] + v[1:]) * dt)])
    cum /= cum[-1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Spherical FM — speed ({source_label}) and sampling schedules',
                 fontsize=11)

    ax = axes[0]
    ax.plot(t_full, v_full, 'C0', lw=1.5)
    ax.set_xlabel('$t$ (rad)')
    ax.set_ylabel('$v_t$')
    ax.set_title(r'Speed $v_t = \sqrt{E[\|\partial_t u_t(X_t)\|^2]}$')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t_full, cum, 'C0', lw=1.5, label='$F(t)$')
    ax.plot([0, T_MAX], [0, 1], 'k--', lw=0.8, alpha=0.5, label='linear')
    ax.set_xlabel('$t$ (rad)')
    ax.set_ylabel('$F(t)$')
    ax.set_title('CDF $F(t)$ of $q(t) \\propto v_t$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    n_ex = nfe_list[len(nfe_list) // 2]
    t_lin = np.linspace(0.0, T_MAX, n_ex + 1)
    t_spd = build_speed_schedule(t_full, v_full, n_ex)

    ax = axes[2]
    ax.plot(range(n_ex + 1), t_lin, 'k-o', ms=4, lw=1.5, label='Linear')
    ax.plot(range(n_ex + 1), t_spd, 'C1-o', ms=4, lw=1.5, label='Speed-adaptive')
    ax.set_xlabel('Step $k$')
    ax.set_ylabel('$t_k$ (rad)')
    ax.set_title(f'Schedule comparison ($n={n_ex}$)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, 'speed_and_schedule.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {path}')


def plot_metrics(results, out_dir, alphas):
    import pandas as pd
    df  = pd.DataFrame(results)
    # Build a colormap across alpha values
    cmap  = plt.cm.coolwarm
    norms = {a: i / max(len(alphas) - 1, 1) for i, a in enumerate(alphas)}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Spherical FM — Mixed schedule sweep  '
                 r'$t_k = (1-\alpha)\,t_k^{\rm lin} + \alpha\,t_k^{\rm speed}$',
                 fontsize=11)

    for ax, key, ylabel, title in [
        (axes[0], 'fid',      'FID', 'FID vs NFE'),
        (axes[1], 'kid_mean', 'KID', 'KID vs NFE'),
        (axes[2], 'is_mean',  'IS',  'IS vs NFE'),
    ]:
        for alpha in alphas:
            sub = df[df['alpha'] == alpha].sort_values('n')
            lw  = 2.0 if alpha in (0.0, 1.0) else 1.0
            ls  = '--' if alpha == 0.0 else (':' if alpha == 1.0 else '-')
            label = (f'α={alpha:.1f}' if alpha not in (0.0, 1.0)
                     else ('linear (α=0)' if alpha == 0.0 else 'speed (α=1)'))
            ax.plot(sub['n'], sub[key], color=cmap(norms[alpha]),
                    lw=lw, ls=ls, marker='o', ms=4, label=label)
        ax.set_xlabel('NFE')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, 'metrics_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {path}')


def plot_alpha_sweep(results, out_dir, alphas):
    """FID vs alpha for each NFE — shows the optimal mix."""
    import pandas as pd
    df  = pd.DataFrame(results)
    nfe_list = sorted(df['n'].unique())
    cmap = plt.cm.viridis

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(r'FID / KID / IS vs mix weight $\alpha$  '
                 r'($\alpha=0$: linear,  $\alpha=1$: speed-adaptive)', fontsize=11)

    for ax, key, ylabel, title in [
        (axes[0], 'fid',      'FID', 'FID vs α'),
        (axes[1], 'kid_mean', 'KID', 'KID vs α'),
        (axes[2], 'is_mean',  'IS',  'IS vs α'),
    ]:
        for i, n in enumerate(nfe_list):
            sub = df[df['n'] == n].sort_values('alpha')
            ax.plot(sub['alpha'], sub[key],
                    color=cmap(i / max(len(nfe_list) - 1, 1)),
                    marker='o', ms=5, lw=1.5, label=f'NFE={n}')
            # Mark the best alpha per NFE
            best_idx = sub[key].idxmin() if key != 'is_mean' else sub[key].idxmax()
            best_row = sub.loc[best_idx]
            ax.axvline(best_row['alpha'], color=cmap(i / max(len(nfe_list) - 1, 1)),
                       lw=0.6, ls='--', alpha=0.5)
        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, 'alpha_sweep.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {path}')


# ── Text report ────────────────────────────────────────────────────────────────

def write_report(results, t_full, v_full, out_dir, ckpt_path,
                 nfe_list, alphas, source_label):
    import pandas as pd
    df  = pd.DataFrame(results)
    sep = '=' * 72

    path = os.path.join(out_dir, 'report.txt')
    with open(path, 'w') as f:
        f.write(sep + '\n')
        f.write('Spherical FM — Mixed Schedule Alpha Sweep\n')
        f.write(sep + '\n')
        f.write(f'Checkpoint  : {ckpt_path}\n')
        f.write(f'Speed source: {source_label}\n')
        f.write(f'Path        : X_t = cos(t)*X_0 + sin(t)*X_1,  t in [0, pi/2]\n')
        f.write(f'Schedule    : t_k = (1-a)*t_k^lin + a*t_k^speed\n')
        f.write(f'Alphas      : {alphas}\n')
        f.write(sep + '\n\n')

        f.write('Speed profile\n')
        f.write(f'  t range : [{t_full[0]:.4f}, {t_full[-1]:.4f}] rad\n')
        f.write(f'  v_t min : {v_full.min():.4f}\n')
        f.write(f'  v_t max : {v_full.max():.4f}\n')
        f.write(f'  v_t mean: {v_full.mean():.4f}\n\n')

        for n in sorted(df['n'].unique()):
            sub = df[df['n'] == n].sort_values('alpha')
            best_fid = sub.loc[sub['fid'].idxmin()]
            f.write(f'NFE = {n}  (best FID: {best_fid["fid"]:.3f} at α={best_fid["alpha"]:.2f})\n')
            f.write(f'  {"alpha":>6}  {"FID":>8}  {"KID":>8}  {"IS":>8}  {"dFID vs lin":>12}\n')
            f.write(f'  {"-"*50}\n')
            fid_lin = float(sub[sub['alpha'] == 0.0]['fid'].values[0])
            for _, row in sub.iterrows():
                delta = fid_lin - row['fid']
                sign  = '+' if delta > 0 else ''
                marker = ' ◄ best' if row['alpha'] == best_fid['alpha'] else ''
                f.write(f'  {row["alpha"]:>6.2f}  {row["fid"]:>8.3f}  '
                        f'{row["kid_mean"]:>8.4f}  {row["is_mean"]:>8.3f}  '
                        f'{sign}{delta:>10.3f}{marker}\n')
            f.write('\n')

        f.write(sep + '\n')
        f.write('dFID vs lin = FID(linear) - FID(alpha)  [positive = alpha wins]\n')

    print(f'Saved {path}')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str,
        default='/nfs/ghome/live/cmarouani/FREE/outputs/cifar10_spherical/'
                'fm_standard/checkpoints/ema_step_0220000.pt')
    parser.add_argument('--speed_dir', type=str,
        default='/nfs/ghome/live/cmarouani/FREE/outputs/cifar10_spherical',
        help='Directory with pre-computed t_grid_sph.npy and ot_speed_sph.npy')
    parser.add_argument('--out_dir', type=str,
        default='/nfs/ghome/live/cmarouani/FREE/outputs/cifar10_spherical/'
                'schedule_comparison')
    parser.add_argument('--data_dir',  type=str, default='/tmp/cifar10_sched_eval')
    parser.add_argument('--n_samples', type=int, default=10000)
    parser.add_argument('--nfe_list',  type=int, nargs='+', default=[10, 20, 35, 50, 100])
    parser.add_argument('--alphas', type=float, nargs='+',
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        help='Mix weights: 0=pure linear, 1=pure speed-adaptive, '
             'intermediate=(1-a)*linear + a*speed')
    parser.add_argument('--smooth_sigma', type=float, default=0.05,
        help='Gaussian smoothing bandwidth (t-units) applied to v_t before CDF. 0=off')
    parser.add_argument('--num_channel', type=int, default=128)
    parser.add_argument('--gen_bs',    type=int, default=256)
    # Model-based speed estimation (optional)
    parser.add_argument('--use_model_speed', action='store_true',
        help='Estimate v_t from the trained model via JVP instead of loading '
             'pre-computed analytical OT speed')
    parser.add_argument('--n_grid',       type=int, default=100)
    parser.add_argument('--speed_B',      type=int, default=512)
    parser.add_argument('--speed_epochs', type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')
    if torch.cuda.is_available():
        print(f'GPU    : {torch.cuda.get_device_name(0)}')

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_model(args.ckpt, args.num_channel, device)

    # ── Speed profile ─────────────────────────────────────────────────────────
    if args.use_model_speed:
        print('\n=== Model speed via JVP ===')
        # Need data for X_t sampling
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        ds = datasets.CIFAR10(root=args.data_dir, train=True,
                               download=True, transform=tfm)
        ldr = torch.utils.data.DataLoader(
            ds, batch_size=256, shuffle=False, num_workers=4)
        imgs = torch.cat([x for x, _ in tqdm(ldr, desc='Loading data', ncols=80)])

        t_grid_raw = np.linspace(0.01, T_MAX - 0.01, args.n_grid)
        v_t_raw = estimate_model_speed_jvp(
            model, t_grid_raw, imgs,
            B=args.speed_B, n_epochs=args.speed_epochs, device=device)

        np.save(os.path.join(args.out_dir, 'model_speed_t_grid.npy'), t_grid_raw)
        np.save(os.path.join(args.out_dir, 'model_speed_v_t.npy'),    v_t_raw)
        print(f'v_t range: [{v_t_raw.min():.4f}, {v_t_raw.max():.4f}]')
        source_label = f'model JVP  (n_grid={args.n_grid}, B={args.speed_B})'
    else:
        print('\n=== Loading pre-computed OT speed ===')
        t_grid_raw = np.load(os.path.join(args.speed_dir, 't_grid_sph.npy'))
        v_t_raw    = np.load(os.path.join(args.speed_dir, 'ot_speed_sph.npy'))
        print(f't_grid : [{t_grid_raw[0]:.4f}, {t_grid_raw[-1]:.4f}]  '
              f'({len(t_grid_raw)} pts)')
        print(f'v_t    : [{v_t_raw.min():.4f}, {v_t_raw.max():.4f}]')
        source_label = 'pre-computed analytical OT speed (ot_speed_sph.npy)'

    # Extend grid to cover full [0, pi/2] by flat extrapolation at edges
    t_full = np.concatenate([[0.0],      t_grid_raw, [T_MAX]])
    v_full = np.concatenate([[v_t_raw[0]], v_t_raw, [v_t_raw[-1]]])

    # ── Inception reference statistics ───────────────────────────────────────
    print('\n=== Inception reference statistics ===')
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    ds  = datasets.CIFAR10(root=args.data_dir, train=True,
                            download=True, transform=tfm)
    loader    = torch.utils.data.DataLoader(
        ds, batch_size=256, shuffle=False, num_workers=4)
    inception = InceptionMetrics(device=device, batch_size=256)
    real_mu, real_sig, real_feats = inception.compute_real_stats(loader)
    print(f'Reference: {len(real_feats):,} images')

    # ── Evaluate schedules ────────────────────────────────────────────────────
    print('\n=== Evaluating sampling schedules ===')
    # Ensure 0.0 and 1.0 are always included, dedup and sort
    alphas = sorted(set([0.0, 1.0] + list(args.alphas)))
    print(f'Alpha sweep: {alphas}')
    results = []

    for n in args.nfe_list:
        t_linear = np.linspace(0.0, T_MAX, n + 1)
        t_speed  = build_speed_schedule(t_full, v_full, n,
                                         smooth_sigma=args.smooth_sigma)
        for alpha in alphas:
            # Mixed schedule: convex combination of linear and speed-adaptive
            t_sched = (1.0 - alpha) * t_linear + alpha * t_speed
            scheme_name = (f'alpha={alpha:.2f}' if alpha not in (0.0, 1.0)
                           else ('linear' if alpha == 0.0 else 'speed_adaptive'))

            print(f'\n  NFE={n:3d}  [{scheme_name}]  generating {args.n_samples} samples...',
                  flush=True)
            t0      = time.time()
            samples = euler_sample_schedule(
                model, args.n_samples, t_sched, device, args.gen_bs)
            gen_t   = time.time() - t0

            feats_f, probs_f = inception.get_activations(samples)
            fid            = inception.compute_fid(real_mu, real_sig, feats_f)
            kid_m, kid_s   = inception.compute_kid(real_feats, feats_f)
            is_m, is_s     = inception.compute_is(probs_f)

            print(f'    FID={fid:.3f}  KID={kid_m:.4f}±{kid_s:.4f}'
                  f'  IS={is_m:.3f}±{is_s:.3f}  ({gen_t:.0f}s)')

            results.append({
                'n': n, 'alpha': alpha, 'scheme': scheme_name,
                'fid': fid, 'kid_mean': kid_m, 'kid_std': kid_s,
                'is_mean': is_m, 'is_std': is_s,
            })

    # ── Save outputs ──────────────────────────────────────────────────────────
    csv_path = os.path.join(args.out_dir, 'schedule_comparison.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f,
            fieldnames=['n', 'alpha', 'scheme', 'fid', 'kid_mean', 'kid_std',
                        'is_mean', 'is_std'])
        w.writeheader()
        w.writerows(results)
    print(f'\nCSV → {csv_path}')

    plot_speed_and_schedules(t_full, v_full, args.out_dir, args.nfe_list,
                              source_label)
    plot_metrics(results, args.out_dir, alphas)
    plot_alpha_sweep(results, args.out_dir, alphas)
    write_report(results, t_full, v_full, args.out_dir, args.ckpt,
                 args.nfe_list, alphas, source_label)

    print('\nAll done!')


if __name__ == '__main__':
    main()
