"""
Compute and plot OT speed, weighting, and schedule for CIFAR-10 (TargetCFM).

Source: N(0,I), Target: CIFAR-10
Path:  x_t = t*x1 + (1-t)*x0
Marginal velocity: u_t(x) = (E[X1|X_t=x] - x) / (1-t)
OT speed: v_t^OT = sqrt(E[||∂_t u_t(X_t)||^2])

Saves:
  /tmp/fm_results/cifar10_speed.png  -- plot
  /tmp/fm_results/ot_weighting.npy   -- array of shape (T+1,) with w(t) values
  /tmp/fm_results/t_grid.npy         -- array of shape (T+1,) with t values
"""
import os, sys
sys.path.insert(0, '/nfs/ghome/live/cmarouani/FREE')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.func import jvp
from torchvision import datasets, transforms

OUTDIR = '/nfs/ghome/live/cmarouani/FREE/outputs/cifar10'
os.makedirs(OUTDIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

EPS_T = 1e-3  # avoid singularity at 0 and 1

# ─── load CIFAR-10 reference images ──────────────────────────────────────────

def load_cifar10_ref(n=200, data_dir='/tmp/fm_results/data'):
    """Load n CIFAR-10 training images as flattened float32 tensors in [-1,1]."""
    os.makedirs(data_dir, exist_ok=True)
    ds = datasets.CIFAR10(
        root=data_dir, train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=n, shuffle=True)
    x1, _ = next(iter(loader))
    return x1.view(n, -1).to(DEVICE)  # (N, 3072) in [-1,1]

# ─── marginal fields ──────────────────────────────────────────────────────────

def compute_u_t(x, t, x1):
    """Marginal OT velocity. x:(B,d), t:scalar, x1:(N,d) -> (B,d)"""
    # Softmax mixture weights: w_i ∝ exp(-||x - t*x1_i||^2 / (2*(1-t)^2))
    diff = x.unsqueeze(1) - t * x1.unsqueeze(0)      # (B,N,d)
    log_w = -(diff ** 2).sum(-1) / (2 * (1 - t) ** 2)  # (B,N)
    w = torch.softmax(log_w, dim=-1)                    # (B,N)
    # u_t(x) = (E[X1|X_t=x] - x) / (1-t)
    vel = (x1.unsqueeze(0) - x.unsqueeze(1)) / (1 - t)  # (B,N,d)
    return (w.unsqueeze(-1) * vel).sum(1)               # (B,d)

# ─── speed estimators ─────────────────────────────────────────────────────────

def _sample_xt(t_val, x1_ref, B=128):
    """Sample x_t via forward process: x_t = t*x1 + (1-t)*x0, x0~N(0,I)."""
    N, d = x1_ref.shape
    t_val = float(np.clip(t_val, EPS_T, 1 - EPS_T))
    idx = torch.randint(0, N, (B,), device=DEVICE)
    x1_s = x1_ref[idx]                              # (B,d)
    x0 = torch.randn(B, d, device=DEVICE)
    xt = t_val * x1_s + (1 - t_val) * x0
    t = torch.tensor(t_val, device=DEVICE)
    return xt.detach(), t

def ot_speed(t_val, x1_ref, B=128):
    """OT speed v_t^OT = sqrt(E[||∂_t u_t(X_t)||^2]) via JVP."""
    xt, t = _sample_xt(t_val, x1_ref, B)

    def f(t_):
        return compute_u_t(xt, t_, x1_ref)

    _, du_dt = jvp(f, (t,), (torch.ones_like(t),))
    return (du_dt ** 2).sum(-1).mean().sqrt().item()

def compute_ot_speeds(x1_ref, t_grid):
    vot = []
    T = len(t_grid)
    for i, t in enumerate(t_grid):
        if i % 5 == 0:
            print(f'  t={t:.3f} ({i+1}/{T})', flush=True)
        vot.append(ot_speed(t, x1_ref))
    return np.array(vot)

# ─── weighting & schedule ─────────────────────────────────────────────────────

def alpha_inv(speeds, t_grid):
    """(1/L) int_0^t v_s ds — used internally to get L for weighting."""
    cum = np.zeros(len(t_grid))
    for i in range(1, len(t_grid)):
        cum[i] = np.trapz(speeds[:i+1], t_grid[:i+1])
    L = max(cum[-1], 1e-12)
    return cum / L, L

def weighting(speeds, t_grid):
    """w(t) = L / v_t."""
    _, L = alpha_inv(speeds, t_grid)
    return np.where(speeds > 1e-12, L / speeds, 0.0)

def smooth_weighting(w, sigma_pts=2):
    """Gaussian-smooth w(t) on the discrete grid using reflection padding."""
    ksize = max(int(4 * sigma_pts + 1) | 1, 3)
    x = np.arange(ksize) - ksize // 2
    kernel = np.exp(-0.5 * (x / sigma_pts) ** 2)
    kernel /= kernel.sum()
    pad = ksize // 2
    w_pad = np.pad(w, pad, mode='reflect')
    return np.convolve(w_pad, kernel, mode='valid')

def schedule_from_weighting(w, t_grid):
    """alpha(t) = (alpha^{-1})^{-1}(t) where (alpha^{-1})'(t) = 1/w(t)."""
    integrand = 1.0 / np.maximum(w, 1e-12)
    cum = np.zeros(len(t_grid))
    for i in range(1, len(t_grid)):
        cum[i] = np.trapz(integrand[:i+1], t_grid[:i+1])
    cum /= max(cum[-1], 1e-12)
    return np.interp(t_grid, cum, t_grid)

# ─── plot ─────────────────────────────────────────────────────────────────────

def plot_and_save(t_grid, vot, w_raw, w_smooth, alpha):
    fig, axes = plt.subplots(3, 1, figsize=(7, 13), sharex=True)
    fig.suptitle('CIFAR-10 (TargetCFM, $\\sigma=0$)', fontsize=13, fontweight='bold')

    axes[0].plot(t_grid, vot, color='darkorange', lw=2,
                 label=r'$v_t^{OT}=\sqrt{E[\|\partial_t u_t(X_t)\|^2]}$')
    axes[0].set_ylabel(r'speed $v_t^{OT}$')
    axes[0].set_title('Speed')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_grid, w_raw, color='darkorange', lw=1.5, alpha=0.4,
                 label=r'$w(t)=L/v_t^{OT}$ (raw)')
    axes[1].plot(t_grid, w_smooth, color='darkorange', lw=2,
                 label=r'$\tilde{w}(t)$ (Gaussian smoothed)')
    axes[1].set_ylabel(r'weighting $w(t)$')
    axes[1].set_title(r'Weighting  $w(t) = L / v_t^{OT}$')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_grid, alpha, color='darkorange', lw=2,
                 label=r'$\alpha(t)$, $(\alpha^{-1})\'(t)=1/\tilde{w}(t)$')
    axes[2].plot(t_grid, t_grid, 'k--', lw=1, alpha=0.5, label='identity')
    axes[2].set_ylabel(r'$\alpha(t)$')
    axes[2].set_xlabel('$t$')
    axes[2].set_title(r'Schedule  $\alpha(t) = (\alpha^{-1})^{-1}(t)$')
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f'{OUTDIR}/cifar10_speed.png'
    plt.savefig(fname, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'Saved: {fname}')

# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    T = 40
    t_grid = np.linspace(0, 0.98, T + 1)

    print('Loading CIFAR-10 reference images...')
    x1_ref = load_cifar10_ref(n=200)
    print(f'x1_ref shape: {x1_ref.shape}, range: [{x1_ref.min():.2f}, {x1_ref.max():.2f}]')

    print('Computing OT speeds...')
    vot = compute_ot_speeds(x1_ref, t_grid)

    w_raw = weighting(vot, t_grid)
    w_smooth = np.minimum(smooth_weighting(w_raw, sigma_pts=5), 100.0)
    alpha = schedule_from_weighting(w_smooth, t_grid)

    print('\nOT speed range:', vot.min(), '->', vot.max())
    print('Weighting (raw) range:', w_raw.min(), '->', w_raw.max())
    print('Weighting (smooth) range:', w_smooth.min(), '->', w_smooth.max())

    # Save arrays for use in weighted training
    np.save(f'{OUTDIR}/ot_speed.npy', vot)
    np.save(f'{OUTDIR}/ot_weighting.npy', w_smooth)
    np.save(f'{OUTDIR}/t_grid.npy', t_grid)
    print(f'Saved npy files to {OUTDIR}/')

    # Plot
    plot_and_save(t_grid, vot, w_raw, w_smooth, alpha)
    print('Done!')


if __name__ == '__main__':
    main()
