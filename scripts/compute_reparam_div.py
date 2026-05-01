"""
Compute E[(div u_t^theta(X_{alpha(t)}))^2] under the arc-length reparametrisation alpha.

For the second stage of self-FM training, the time-sampling distribution is
    p(t) ∝ 1 / v_t^FR
which induces an arc-length reparametrisation
    alpha : [0,1] → [T_MIN, T_MAX],   alpha = CDF^{-1} of p(t)
such that if s ~ Uniform(0,1) then alpha(s) ~ p(t).

The velocity field u_s^theta is trained with t ~ p(t) ∝ 1/v_t^FR, so the
network's internal "time" is the arc-length parameter s.  When evaluated at s,
it approximates the dynamics of X_{alpha(s)}.  Therefore D(s) is:

    D(s) = E[(div u_s^theta(X_{alpha(s)}))^2]

i.e. the network is called with argument s (not tau=alpha(s)), but the sample
is drawn from the interpolant at time tau=alpha(s).

If alpha truly makes the flow constant-speed in the Fisher-Rao sense, then
    s ↦ D(s) = E[(div u_s^theta(X_{alpha(s)}))^2]
should be approximately constant.

We plot both:
    (A) t ↦ E[(div u_t^theta(X_t))^2]                   raw (no reparam)
    (B) s ↦ E[(div u_s^theta(X_{alpha(s)}))^2]           after reparam

The divergence is estimated via the Hutchinson trace estimator with
n_hutch=5 Rademacher probes.

Usage:
    python compute_reparam_div.py \\
        --ckpt      outputs/cifar10_self_fm/checkpoints/ema_step_0200000.pt \\
        --speed_t   outputs/cifar10_self_fm/fr_t_grid_step100000.npy \\
        --speed_v   outputs/cifar10_self_fm/fr_speed_step100000.npy \\
        --out_dir   outputs/cifar10_self_fm \\
        --data_dir  /tmp/cifar10_div \\
        --n_t       200 \\
        --B         256 \\
        --n_hutch   5
"""
import os, sys, argparse
import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch
import torch.nn as nn
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, '/nfs/ghome/live/cmarouani/FREE')
from torchcfm.models.unet.unet import UNetModelWrapper

T_MIN = 0.2
T_MAX = 0.8


# ── Arc-length reparametrisation ───────────────────────────────────────────────

def build_alpha(t_grid, v_t, T_MIN=T_MIN, T_MAX=T_MAX):
    """
    Build alpha = F^{-1} where F(t) = int_0^t v^FR(s) ds / int_0^T v^FR(s) ds.
    This is the true arc-length parametrisation: equal increments of s correspond
    to equal Fisher-Rao distance travelled.
    Returns alpha: (n_s,) -> (n_s,) numpy function via interpolation.
    """
    # Ensure v_t > 0
    v = np.maximum(v_t, 1e-10).copy()
    # Density: w(t) = v_t^FR  (arc-length weight)
    w = v
    # Normalise to a PDF over [t_grid[0], t_grid[-1]]
    dt = np.diff(t_grid)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (w[:-1] + w[1:]) * dt)])
    cum /= cum[-1]
    # Extend to cover full [0, 1] in s-space with flat extrapolation at edges
    t_ext = np.concatenate([[T_MIN], t_grid, [T_MAX]])
    c_ext = np.concatenate([[0.0],   cum,    [1.0]])
    # alpha(s) = CDF^{-1}(s): interpolate
    def alpha(s_arr):
        return np.interp(s_arr, c_ext, t_ext).astype(np.float32)
    return alpha, t_ext, c_ext, w


# ── Self-interpolant sampler ───────────────────────────────────────────────────

def sample_xt(x1_pool, t_val, B, device):
    """
    Sample X_t from the self-interpolant at fixed t_val.
    X_t = (1-t)*X1 + t*X1_tilde + sqrt(t(1-t))*eps
    """
    N  = len(x1_pool)
    i1 = torch.randint(0, N, (B,))
    i2 = torch.randint(0, N, (B,))
    x1       = x1_pool[i1].to(device)
    x1_tilde = x1_pool[i2].to(device)
    sigma_t  = float(np.sqrt(t_val * (1.0 - t_val)))
    eps      = torch.randn_like(x1)
    xt       = (1.0 - t_val) * x1 + t_val * x1_tilde + sigma_t * eps
    return xt.detach()


# ── Hutchinson divergence estimator ───────────────────────────────────────────

def hutchinson_div_sq(model, t_val, xt, n_hutch, device):
    """
    Estimate E[(div u_t^theta(X_t))^2] via Hutchinson with n_hutch probes.
    div u ≈ z^T (∂u/∂x) z  for z ~ Rademacher.
    Returns scalar: mean over batch of (div_estimate)^2.
    """
    B   = xt.shape[0]
    t   = torch.full((B,), t_val, device=device)
    div_estimates = []

    for _ in range(n_hutch):
        x_req = xt.detach().requires_grad_(True)
        u     = model(t, x_req)
        z     = (torch.randint(0, 2, xt.shape, device=device).float() * 2 - 1)
        # Hutchinson: div(u) ≈ z^T ∇_x u z = (u·z).sum() backprop onto z.T
        grad  = torch.autograd.grad(
            (u * z).sum(), x_req,
            retain_graph=False, create_graph=False
        )[0]
        div_k = (grad * z).sum(dim=(1, 2, 3))   # (B,)
        div_estimates.append(div_k.detach())

    div_avg = torch.stack(div_estimates).mean(0)   # (B,)  averaged over probes
    return (div_avg ** 2).mean().item()             # scalar E[(div)^2]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',      required=True,
                        help='Path to EMA checkpoint (.pt)')
    parser.add_argument('--speed_t',   required=True,
                        help='Saved t_grid .npy from FR speed estimation')
    parser.add_argument('--speed_v',   required=True,
                        help='Saved v_t .npy (smoothed FR speed)')
    parser.add_argument('--out_dir',   required=True)
    parser.add_argument('--data_dir',  default='/tmp/cifar10_div')
    parser.add_argument('--n_t',       type=int, default=200,
                        help='Number of t-grid points for the sweep')
    parser.add_argument('--B',         type=int, default=256,
                        help='Batch size per time point')
    parser.add_argument('--n_hutch',   type=int, default=5,
                        help='Hutchinson probes per time point')
    parser.add_argument('--n_epochs',  type=int, default=3,
                        help='Repeat sweep N times and average for stability')
    parser.add_argument('--num_channel', type=int, default=128)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device  : {device}')
    if device.type == 'cuda':
        print(f'GPU     : {torch.cuda.get_device_name(0)}')
    print(f'Ckpt    : {args.ckpt}')
    print(f'n_t={args.n_t}, B={args.B}, n_hutch={args.n_hutch}, n_epochs={args.n_epochs}')

    # ── Load speed curve and build alpha ──────────────────────────────────────
    t_grid = np.load(args.speed_t)
    v_t    = np.load(args.speed_v)
    print(f'\nSpeed   : t∈[{t_grid[0]:.3f},{t_grid[-1]:.3f}]  '
          f'v∈[{v_t.min():.1f},{v_t.max():.1f}]  ratio={v_t.max()/v_t.min():.0f}x')

    alpha, t_ext, cdf, w_raw = build_alpha(t_grid, v_t)

    # ── Uniform s-grid and its alpha-mapped t values ──────────────────────────
    s_grid    = np.linspace(0.0, 1.0, args.n_t, dtype=np.float32)
    tau_grid  = alpha(s_grid)                   # t values after reparam
    # Also define a direct uniform t-grid for comparison (no reparam)
    t_uniform = np.linspace(T_MIN + 0.01, T_MAX - 0.01, args.n_t, dtype=np.float32)

    print(f'tau range after reparam: [{tau_grid.min():.4f}, {tau_grid.max():.4f}]')

    # ── Load CIFAR-10 (reference pool for X_t sampling) ──────────────────────
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    ds      = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=tfm)
    loader  = DataLoader(ds, batch_size=512, shuffle=True, num_workers=4)
    x1_pool = next(iter(loader))[0]             # (512, 3, 32, 32) reference pool
    print(f'Reference pool: {x1_pool.shape}')

    # ── Load model ────────────────────────────────────────────────────────────
    net = UNetModelWrapper(
        dim=(3, 32, 32), num_res_blocks=2, num_channels=args.num_channel,
        channel_mult=[1, 2, 2, 2], num_heads=4, num_head_channels=64,
        attention_resolutions='16', dropout=0.1,
    ).to(device)

    ck = torch.load(args.ckpt, map_location=device)
    # checkpoint may store 'ema' or 'net' key
    state = ck.get('ema', ck.get('net', ck))
    net.load_state_dict(state)
    net.eval()
    print(f'Loaded  : {args.ckpt}')

    # ── Sweep ─────────────────────────────────────────────────────────────────
    div_sq_reparam  = np.zeros((args.n_epochs, args.n_t))   # E[(div u_{alpha(s)})^2]
    div_sq_uniform  = np.zeros((args.n_epochs, args.n_t))   # E[(div u_t)^2]  raw

    for ep in range(args.n_epochs):
        print(f'\n── Epoch {ep+1}/{args.n_epochs} ──')
        # Refresh reference pool each epoch for variance reduction
        x1_pool = next(iter(loader))[0]

        for i, (s_val, tau_val, t_val) in enumerate(tqdm(
                zip(s_grid, tau_grid, t_uniform),
                total=args.n_t, ncols=80, desc=f'  sweep ep{ep+1}')):

            # Reparametrised: sample X at time tau = alpha(s), but call net with s
            # u_s^theta approximates the dynamics of X_{alpha(s)}, so the network
            # is queried at the arc-length parameter s, not at tau = alpha(s).
            xt_reparam = sample_xt(x1_pool, float(tau_val), args.B, device)
            with torch.no_grad():
                pass  # ensure clean grads before hutchinson
            div_sq_reparam[ep, i] = hutchinson_div_sq(
                net, float(s_val), xt_reparam, args.n_hutch, device)

            # Raw uniform t: sample X at time t
            xt_uniform = sample_xt(x1_pool, float(t_val), args.B, device)
            div_sq_uniform[ep, i] = hutchinson_div_sq(
                net, float(t_val), xt_uniform, args.n_hutch, device)

    # Average over epochs
    div_sq_reparam_mean = div_sq_reparam.mean(0)
    div_sq_uniform_mean = div_sq_uniform.mean(0)
    div_sq_reparam_std  = div_sq_reparam.std(0)
    div_sq_uniform_std  = div_sq_uniform.std(0)

    # ── Save arrays ───────────────────────────────────────────────────────────
    np.save(os.path.join(args.out_dir, 'div_sq_reparam_s_grid.npy'), s_grid)
    np.save(os.path.join(args.out_dir, 'div_sq_reparam_tau_grid.npy'), tau_grid)
    np.save(os.path.join(args.out_dir, 'div_sq_reparam_mean.npy'), div_sq_reparam_mean)
    np.save(os.path.join(args.out_dir, 'div_sq_reparam_std.npy'),  div_sq_reparam_std)
    np.save(os.path.join(args.out_dir, 'div_sq_uniform_t_grid.npy'), t_uniform)
    np.save(os.path.join(args.out_dir, 'div_sq_uniform_mean.npy'), div_sq_uniform_mean)
    np.save(os.path.join(args.out_dir, 'div_sq_uniform_std.npy'),  div_sq_uniform_std)
    print('\nArrays saved.')

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(r'Energy repartition: $E[(\mathrm{div}\,u_t^\theta(X_t))^2]$'
                 '\n(Self-FM, step 200K)', fontsize=12)

    # Panel 1: raw (no reparam)
    ax = axes[0]
    ax.fill_between(t_uniform,
                    div_sq_uniform_mean - div_sq_uniform_std,
                    div_sq_uniform_mean + div_sq_uniform_std, alpha=0.2, color='steelblue')
    ax.plot(t_uniform, div_sq_uniform_mean, color='steelblue', lw=1.5)
    ax.set_xlabel('$t$ (uniform)')
    ax.set_ylabel(r'$E[(\mathrm{div}\,u_t^\theta)^2]$')
    ax.set_title('(A) Raw — no reparametrisation')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Panel 2: after arc-length reparam
    ax = axes[1]
    ax.fill_between(s_grid,
                    div_sq_reparam_mean - div_sq_reparam_std,
                    div_sq_reparam_mean + div_sq_reparam_std, alpha=0.2, color='tomato')
    ax.plot(s_grid, div_sq_reparam_mean, color='tomato', lw=1.5)
    ax.axhline(div_sq_reparam_mean.mean(), color='k', lw=1, ls='--',
               label=f'mean={div_sq_reparam_mean.mean():.1f}')
    ax.set_xlabel(r'$s \in [0,1]$ (arc-length parameter)')
    ax.set_ylabel(r'$E[(\mathrm{div}\,u_{\alpha(s)}^\theta)^2]$')
    ax.set_title(r'(B) After arc-length reparam $\alpha$')
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: tau = alpha(s) to show where alpha maps
    ax = axes[2]
    ax.plot(s_grid, tau_grid, color='seagreen', lw=1.5)
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='identity')
    ax.set_xlabel('$s$ (arc-length parameter)')
    ax.set_ylabel(r'$\tau = \alpha(s)$  (original $t$)')
    ax.set_title(r'(C) Reparametrisation $\alpha$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = os.path.join(args.out_dir, 'div_sq_reparam.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot saved → {out_png}')

    # ── Summary stats ─────────────────────────────────────────────────────────
    cv_raw    = div_sq_uniform_mean.std() / div_sq_uniform_mean.mean()
    cv_reparm = div_sq_reparam_mean.std() / div_sq_reparam_mean.mean()
    print(f'\n── Summary ──')
    print(f'Raw (uniform t):')
    print(f'  mean E[(div)^2] = {div_sq_uniform_mean.mean():.2f}')
    print(f'  std             = {div_sq_uniform_mean.std():.2f}')
    print(f'  CoeffVar (CV)   = {cv_raw:.3f}')
    print(f'\nReparametrised (s = alpha^-1(t)):')
    print(f'  mean E[(div)^2] = {div_sq_reparam_mean.mean():.2f}')
    print(f'  std             = {div_sq_reparam_mean.std():.2f}')
    print(f'  CoeffVar (CV)   = {cv_reparm:.3f}')
    print(f'\nCV reduction: {cv_raw:.3f} → {cv_reparm:.3f}  '
          f'({(1 - cv_reparm/cv_raw)*100:.1f}% more uniform)')
    print('\nDone.')


if __name__ == '__main__':
    main()
