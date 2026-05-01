"""
Post-training speed analysis for all 4 FM weighting variants.

For each trained model (none, ot, score, fr) and a grid of t values:
  v_t^vel  = sqrt(E[||u_θ(t, X_t)||²])          (velocity speed)
  v_t^FR   = sqrt(E[(div u_θ(t, X_t))²])         (Fisher-Rao speed via Hutchinson)
where X_t = t·X1 + (1-t)·X0, X0~N(0,I), X1~CIFAR-10.

div u_θ(t, x) ≈ E_z[ z^T ∂_x u_θ(t,x) z ] via Hutchinson (z ~ Rademacher).

Saves:
  {out_dir}/speed_analysis/{key}/velocity_speed.npy
  {out_dir}/speed_analysis/{key}/fr_speed.npy
  {out_dir}/speed_analysis/{key}/t_grid.npy
  {out_dir}/speed_analysis/comparison.png
"""
import os, sys, copy, argparse
sys.path.insert(0, '/nfs/ghome/live/cmarouani/FREE')

import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from torchcfm.models.unet.unet import UNetModelWrapper


# ─── model loading ────────────────────────────────────────────────────────────

def load_ema(checkpoint_path, device, num_channel=128):
    net = UNetModelWrapper(
        dim=(3, 32, 32), num_res_blocks=2, num_channels=num_channel,
        channel_mult=[1, 2, 2, 2], num_heads=4, num_head_channels=64,
        attention_resolutions='16', dropout=0.0,
    ).to(device)
    ck = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(ck['ema'])
    net.eval()
    return net


def find_last_checkpoint(out_dir):
    ckpt_dir = os.path.join(out_dir, 'checkpoints')
    if not os.path.isdir(ckpt_dir):
        return None
    files = sorted(f for f in os.listdir(ckpt_dir) if f.endswith('.pt'))
    return os.path.join(ckpt_dir, files[-1]) if files else None


# ─── data ─────────────────────────────────────────────────────────────────────

def load_cifar10(n, data_dir, device):
    ds = datasets.CIFAR10(
        root=data_dir, train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=n, shuffle=True)
    x1, _ = next(iter(loader))
    return x1.to(device)


# ─── speed estimators ─────────────────────────────────────────────────────────

@torch.no_grad()
def velocity_speed(model, t_val, x1_ref, B=256):
    """sqrt(E[||u_θ(t, X_t)||²])"""
    N = len(x1_ref)
    idx = torch.randint(0, N, (B,), device=x1_ref.device)
    x1 = x1_ref[idx]
    x0 = torch.randn_like(x1)
    xt = t_val * x1 + (1 - t_val) * x0
    t  = torch.full((B,), t_val, device=x1_ref.device)
    v  = model(t, xt)
    return (v ** 2).mean(dim=(1, 2, 3)).mean().sqrt().item()


def fr_speed_hutchinson(model, t_val, x1_ref, B=128, n_hutch=4):
    """sqrt(E[(div u_θ)²]) via Hutchinson trace estimator."""
    N = len(x1_ref)
    idx = torch.randint(0, N, (B,), device=x1_ref.device)
    x1  = x1_ref[idx]
    x0  = torch.randn_like(x1)
    xt  = (t_val * x1 + (1 - t_val) * x0).detach().requires_grad_(True)
    t   = torch.full((B,), t_val, device=x1_ref.device)

    divs = []
    for _ in range(n_hutch):
        z = torch.randint(0, 2, xt.shape, device=xt.device).float() * 2 - 1  # Rademacher
        v = model(t, xt)                     # (B,3,32,32)
        # vjp: z^T (∂v/∂x) -- gradient of (v * z).sum() w.r.t. xt
        grad = torch.autograd.grad(
            (v * z).sum(), xt, create_graph=False, retain_graph=True
        )[0]
        div = (grad * z).sum(dim=(1, 2, 3))  # (B,) — Hutchinson estimate of div per sample
        divs.append(div.detach())

    div_mean = torch.stack(divs, 0).mean(0)  # (B,) averaged over probes
    return (div_mean ** 2).mean().sqrt().item()


# ─── analysis loop ────────────────────────────────────────────────────────────

def analyse_model(model, x1_ref, t_grid, B=256, n_hutch=4):
    vel_speeds = []
    fr_speeds  = []
    T = len(t_grid)
    for i, t in enumerate(t_grid):
        if i % 5 == 0:
            print(f'  t={t:.3f} ({i+1}/{T})', flush=True)
        t_f = float(t)
        vel_speeds.append(velocity_speed(model, t_f, x1_ref, B=B))
        fr_speeds.append(fr_speed_hutchinson(model, t_f, x1_ref, B=B, n_hutch=n_hutch))
    return np.array(vel_speeds), np.array(fr_speeds)


# ─── comparison plot ──────────────────────────────────────────────────────────

COLORS = {'none': 'C0', 'ot': 'C1', 'score': 'C2', 'fr': 'C3'}
LABELS = {'none': 'Uniform', 'ot': 'OT weight', 'score': 'Score weight', 'fr': 'FR weight'}


def make_comparison_plot(results, out_dir):
    """results: dict key -> {'t': array, 'vel': array, 'fr': array}"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Learned velocity speed (post-training analysis)', fontsize=13)

    for key, d in results.items():
        c = COLORS.get(key, 'gray')
        lbl = LABELS.get(key, key)
        axes[0].plot(d['t'], d['vel'], lw=2, color=c, label=lbl)
        axes[1].plot(d['t'], d['fr'],  lw=2, color=c, label=lbl)

    axes[0].set_title(r'Velocity speed  $\sqrt{E[\|u_\theta(t,X_t)\|^2]}$')
    axes[0].set_xlabel('$t$'); axes[0].set_ylabel('speed')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].set_title(r'Fisher-Rao speed  $\sqrt{E[(\mathrm{div}\,u_\theta)^2]}$')
    axes[1].set_xlabel('$t$'); axes[1].set_ylabel('speed')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(out_dir, 'comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--comparison_dir', type=str,
                        default='/nfs/ghome/live/cmarouani/FREE/outputs/cifar10/comparison')
    parser.add_argument('--keys', type=str, nargs='+',
                        default=['none', 'ot', 'score', 'fr'])
    parser.add_argument('--n_ref', type=int, default=2000,
                        help='CIFAR-10 reference images')
    parser.add_argument('--T', type=int, default=40,
                        help='Number of t values')
    parser.add_argument('--B', type=int, default=128,
                        help='Batch size for speed estimation')
    parser.add_argument('--n_hutch', type=int, default=4,
                        help='Hutchinson probe vectors per point')
    parser.add_argument('--num_channel', type=int, default=128)
    parser.add_argument('--data_dir', type=str, default='/tmp/cifar10_speed_data')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Override checkpoint path (for a single key run)')
    args = parser.parse_args()

    torch.manual_seed(42); np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    save_dir = os.path.join(args.comparison_dir, 'speed_analysis')
    os.makedirs(save_dir, exist_ok=True)

    t_grid = np.linspace(0.02, 0.95, args.T)

    print(f'Loading {args.n_ref} CIFAR-10 reference images...')
    x1_ref = load_cifar10(args.n_ref, args.data_dir, device)

    results = {}
    for key in args.keys:
        ckpt_path = args.checkpoint
        if ckpt_path is None:
            model_dir = os.path.join(args.comparison_dir, key)
            ckpt_path = find_last_checkpoint(model_dir)
        if ckpt_path is None or not os.path.exists(ckpt_path):
            print(f'[{key}] No checkpoint found at {ckpt_path} — skipping.')
            continue
        print(f'\n[{key}] Loading checkpoint: {ckpt_path}')
        model = load_ema(ckpt_path, device, num_channel=args.num_channel)

        key_dir = os.path.join(save_dir, key)
        os.makedirs(key_dir, exist_ok=True)

        print(f'[{key}] Computing speeds over {args.T} t-values...')
        vel, fr = analyse_model(model, x1_ref, t_grid, B=args.B, n_hutch=args.n_hutch)

        np.save(os.path.join(key_dir, 'velocity_speed.npy'), vel)
        np.save(os.path.join(key_dir, 'fr_speed.npy'),       fr)
        np.save(os.path.join(key_dir, 't_grid.npy'),         t_grid)
        print(f'  vel range: {vel.min():.4f} -> {vel.max():.4f}')
        print(f'  FR  range: {fr.min():.4f} -> {fr.max():.4f}')
        results[key] = {'t': t_grid, 'vel': vel, 'fr': fr}

        del model
        torch.cuda.empty_cache()

    if results:
        make_comparison_plot(results, save_dir)
    else:
        print('No models found — nothing to plot.')

    print('\nDone.')


if __name__ == '__main__':
    main()
